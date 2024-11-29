import torch
import tiktoken

from models.model_gpt2 import GPT
from models.model_gpt2_multi_sofmax import GPT2MultiSoftmax

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# load
hook_layers = list(range(12))  # 假设是12层模型
model = GPT2MultiSoftmax.from_pretrained(
    'gpt2',
    hook_layers=hook_layers,
    freeze_base=True  # 设置为True则只训练新增的层
).to(device)
enc = tiktoken.get_encoding('gpt2')


# train
