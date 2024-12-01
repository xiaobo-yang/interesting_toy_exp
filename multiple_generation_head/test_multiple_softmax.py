import torch
import tiktoken

from models.model_gpt2 import GPT
from models.model_gpt2_multi_sofmax import GPT2MultiSoftmax

device = 'cuda' if torch.cuda.is_available() else 'cpu'



# 创建带有多个softmax头的模型，在第6层和最后一层添加预测头
hook_layers = list(range(7,12))  # 假设是12层模型
model = GPT2MultiSoftmax.model_surgery_from_pretrained(
    'gpt2_model_19072.pt',
    hook_layers=hook_layers,
    freeze_base=True  # 设置为True则只训练新增的层
).to(device)
model.load_state_dict(torch.load('log/gpt2_model_19072.pt-multi-softmax_20241129_223536_step_05000.pt')['model'])

# test generation
enc = tiktoken.get_encoding('gpt2')
prompt = "Donald Trump is the president of"
max_new_tokens = 64
input_ids = [enc.encode(prompt)]
input_ids = torch.tensor(input_ids, device=device)
output_list = model.generate(input_ids, max_new_tokens, do_sample=True, temperature=0.3)
for layer, output in output_list.items():
    print(f"Model Layer {layer}:\n{enc.decode(output[0].tolist())}\n")


# compare with the original model
base_model = GPT.from_pretrained('gpt2_model_19072.pt').to(device)
output_list = base_model.generate(input_ids, max_new_tokens, do_sample=True, temperature=0.3)
print(f"Base Model:\n{enc.decode(output_list[0].tolist())}\n")
