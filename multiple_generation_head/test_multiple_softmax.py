import torch
import tiktoken

from models.model_gpt2 import GPT
from models.model_gpt2_multi_sofmax import GPT2MultiSoftmax

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# 创建带有多个softmax头的模型，在第6层和最后一层添加预测头
hook_layers = list(range(12))  # 假设是12层模型
model = GPT2MultiSoftmax.from_pretrained(
    'gpt2',
    hook_layers=hook_layers,
    freeze_base=True  # 设置为True则只训练新增的层
).to(device)
# for comparison
base_model = GPT.from_pretrained('gpt2').to(device)



# test generation
enc = tiktoken.get_encoding('gpt2')
prompt = "Donald Trump is the president of"
max_new_tokens = 32

input_ids = [enc.encode(prompt)]
input_ids = torch.tensor(input_ids, device=device)
output_list = model.generate(input_ids, max_new_tokens, do_sample=False)
for layer, output in output_list.items():
    print(f"Model Layer {layer}:\n{enc.decode(output[0].tolist())}\n")

print("-"*30)
base_output = base_model.generate(input_ids, max_new_tokens, do_sample=False)
print(f"Base model final layer:\n{enc.decode(base_output[0].tolist())}\n")
