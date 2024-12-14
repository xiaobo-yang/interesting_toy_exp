from tqdm import tqdm
import os
import json
import random
import transformers
import torch
import torch.distributed as dist


"""
    torchrun --nproc_per_node=8 test_llama3_gsm8k.py

    For llama3.2 1b, Total Accuracy (greedy decoding): 0.453 and 0.408 (4-bit quantization)
"""

def load_jsonlines(file_name: str):
    with open(file_name, 'r') as f:
        return [json.loads(line) for line in f]

def nshot_chats(nshot_data: list, n: int, question: str) -> dict:

    def question_prompt(s):
        return f'Question: {s}'

    def answer_prompt(s):
        return f'Answer: {s}'

    chats = [
        # {
        #     "role": "system",
        #     "content": "You are a grade school math problem solver. At the end, you MUST write the answer as an integer after '####'. Let's think step by step.",
        # },
    ]

    random.seed(42)
    for qna in random.sample(nshot_data, n):
        chats.append(
            {"role": "user", "content": question_prompt(qna["question"])})
        chats.append(
            {"role": "assistant", "content": answer_prompt(qna["answer"])})

    chats.append({"role": "user", "content": question_prompt(question)+" Let's think step by step. At the end, you MUST write the answer as an integer after '####'."})

    return chats


def extract_ans_from_response(answer: str, eos=None):
    if eos:
        answer = answer.split(eos)[0].strip()

    answer = answer.split('####')[-1].strip()

    for remove_char in [',', '$', '%', 'g']:
        answer = answer.replace(remove_char, '')

    try:
        return int(answer)
    except ValueError:
        return answer



dist.init_process_group(backend='nccl')
rank = int(os.environ['RANK'])
local_rank = int(os.environ['LOCAL_RANK'])
world_size = int(os.environ['WORLD_SIZE'])
device = f'cuda:{local_rank}'
torch.cuda.set_device(device)


model_name = "/data/my_data/models/Llama-3.2-1B-Instruct"
N_SHOT = 8  # gsm8k usually uses 8-shot
eval_batch_size = 192 # test data has 1319 samples, use 8 gpus, each gpu has 165 samples
os.makedirs('logs', exist_ok=True)
log_file_path = 'logs/errors.txt'
if rank == 0:
    with open(log_file_path, 'w') as log_file:
        log_file.write('')
dist.barrier(device_ids=[rank])

# quantization
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# 加载模型到指定GPU
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map={"": rank},  # 指定设备
    torch_dtype="auto",
    # quantization_config=bnb_config,
)


# 创建tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = 'left' # for batch encoding

# 创建generator
generator = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    pad_token_id=tokenizer.eos_token_id,
    device_map={"": rank},
    batch_size=eval_batch_size,
    max_new_tokens=1024,
    do_sample=False,
)
generator.tokenizer.pad_token_id = generator.tokenizer.eos_token_id # padding for batching

@torch.no_grad()
def get_response(chats): 
    gen_texts = generator(chats)  # First return sequence
    return [gen_text[0]['generated_text'][-1]['content'] for gen_text in gen_texts]

# 加载数据
test_data = load_jsonlines('data/gsm8k/test.jsonl')
train_data = load_jsonlines('data/gsm8k/train.jsonl')

# 计算当前GPU负责的数据范围
per_gpu_samples = len(test_data) // world_size
start_idx = rank * per_gpu_samples
end_idx = start_idx + per_gpu_samples if rank != world_size-1 else len(test_data)

# 执行推理
total = correct = 0
for i in tqdm(range(start_idx, end_idx, eval_batch_size)):
    batch_qnas = test_data[i:min(i+eval_batch_size, end_idx)]
    batch_messages = [nshot_chats(train_data, N_SHOT, qna['question']) 
                        for qna in batch_qnas]
    batch_responses = get_response(batch_messages)
    
    # 处理结果...
    for response, qna in zip(batch_responses, batch_qnas):
        pred_ans = extract_ans_from_response(response)
        true_ans = extract_ans_from_response(qna['answer'])

        total += 1
        if pred_ans != true_ans:
            with open(log_file_path, 'a', encoding='utf-8') as log_file:
                log_file.write(f"{qna['question']}\n\n")
                log_file.write(f"Response: {response}\n\n")
                log_file.write(f"Ground Truth: {qna['answer']}\n\n")
                log_file.write(f"Current Accuracy: {correct/total:.3f}\n\n")
                log_file.write('\n\n')
        else:
            correct += 1

# 收集所有GPU的结果
all_total = torch.tensor([total], device=f"cuda:{rank}")
all_correct = torch.tensor([correct], device=f"cuda:{rank}")
print(f"rank: {rank}, total: {total}, correct: {correct}, acc: {correct/total:.3f}")

dist.all_reduce(all_total, op=dist.ReduceOp.SUM)
dist.all_reduce(all_correct, op=dist.ReduceOp.SUM)

dist.barrier(device_ids=[rank])
if rank == 0:
    print(f"Total Accuracy: {all_correct.item()/all_total.item():.3f}")



dist.destroy_process_group()