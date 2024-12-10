import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
from copy import deepcopy

import sys
sys.path.append("/data/my_tools/build-nanogpt")
from modelling_gpt2 import GPT, GPTConfig

"""
    测试对nanogpt2权重中和embedding维数相同的向量都做同一个角度的旋转，模型是否等价

    layer norm的结构无法等价，因为需要用到每个embedding向量x自己的标准差来标准化，标准差相当于 x投影到1向量的补空间的模长 / 维度 = x(I - 11^T/n) / n，
    旋转之后会发生改变, 只能通过layer norm的weight参数乘以一个系数来维持等价性，然而这个系数和传入的x有关！

    改为使用rms norm，则也不可以等价，因为rmsnorm的参数和x/sqrt(\|x\|^2/d) 是逐分量相乘，除非weight向量各分量全部相等，不然RMSNorm(U^T x) != U^T RMSNorm(x)
    除非将rms norm里weight参数压缩为仅1个参数，此时可以旋转等价。

    不过，这种旋转等价性可能拿来缩减模型训练所需的参数量，因为只需固定其中一个向量，就可以消除这种旋转等价性，因为所有参与旋转的向量关于这个向量的相对距离（如夹角）的改变，都可能导致模型发生改变。
    这只能节约n_embd个参数，微乎其微。。
"""


# ------------------- model surgery -------------------
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float= 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

def replace_layernorm_with_rmsnorm(model):
    with torch.no_grad():
        # 替换transformer块中的所有LayerNorm
        for layer in range(len(model.transformer.h)):
            # 创建新的RMSNorm并复制权重
            model.transformer.h[layer].ln_1 = RMSNorm(1).to(device) # 单参数rmsnorm
            model.transformer.h[layer].ln_2 = RMSNorm(1).to(device)
        # 替换最终的ln_f
        model.transformer.ln_f = RMSNorm(1).to(device)
    return model


# ------------------- 生成正交矩阵 -------------------
def generate_orthogonal_matrix(dim):
    random_matrix = torch.randn(dim, dim)
    q, r = torch.linalg.qr(random_matrix)
    return q


# ------------------- 旋转权重 -------------------
def local_qk_rotate(block):
    qk_ortho_matrix_list = [generate_orthogonal_matrix(head_dim).to(device) for _ in range(n_head)]
    for i, qk_ortho_matrix in enumerate(qk_ortho_matrix_list):
        block.attn.c_attn.weight.data[i*head_dim:(i+1)*head_dim, :] = qk_ortho_matrix.mm(block.attn.c_attn.weight.data[i*head_dim:(i+1)*head_dim, :])
        block.attn.c_attn.bias.data[i*head_dim:(i+1)*head_dim] = qk_ortho_matrix @ block.attn.c_attn.bias.data[i*head_dim:(i+1)*head_dim]
        block.attn.c_attn.weight.data[i*head_dim+n_embd:(i+1)*head_dim+n_embd, :] = qk_ortho_matrix.mm(block.attn.c_attn.weight.data[i*head_dim+n_embd:(i+1)*head_dim+n_embd, :])
        block.attn.c_attn.bias.data[i*head_dim+n_embd:(i+1)*head_dim+n_embd] = qk_ortho_matrix @ block.attn.c_attn.bias.data[i*head_dim+n_embd:(i+1)*head_dim+n_embd]

def local_v_rotate(block):
    v_ortho_matrix_list = [generate_orthogonal_matrix(head_dim).to(device) for _ in range(n_head)]
    for i, v_ortho_matrix in enumerate(v_ortho_matrix_list):
        block.attn.c_attn.weight.data[i*head_dim+2*n_embd:(i+1)*head_dim+2*n_embd, :] = v_ortho_matrix.mm(block.attn.c_attn.weight.data[i*head_dim+2*n_embd:(i+1)*head_dim+2*n_embd, :])
        block.attn.c_attn.bias.data[i*head_dim+2*n_embd:(i+1)*head_dim+2*n_embd] = v_ortho_matrix @ block.attn.c_attn.bias.data[i*head_dim+2*n_embd:(i+1)*head_dim+2*n_embd]
        block.attn.c_proj.weight.data[:, i*head_dim:(i+1)*head_dim] = block.attn.c_proj.weight.data[:, i*head_dim:(i+1)*head_dim].mm(v_ortho_matrix.T)

def rotate(model):
    new_model = deepcopy(model)
    # 生成768维的正交矩阵
    ortho_matrix = generate_orthogonal_matrix(n_embd).to(device)
    # 验证正交性（可选）
    identity_check = torch.mm(ortho_matrix, ortho_matrix.T)
    print(f"正交性检查 - 与单位矩阵的最大误差: {(identity_check - torch.eye(768, device=device)).abs().max().item()}")
    with torch.no_grad():
        # 1. 词嵌入矩阵 (50304, 768)
        new_model.transformer.wte.weight.data = model.transformer.wte.weight.mm(ortho_matrix)
        # 2. 位置编码矩阵 (1024, 768)
        new_model.transformer.wpe.weight.data = model.transformer.wpe.weight.mm(ortho_matrix)
        # 3. 对每个transformer块中的权重进行旋转
        for layer in range(len(model.transformer.h)):
            block = model.transformer.h[layer]
            new_block = new_model.transformer.h[layer]
            # attention权重
            # qkv投影 (768, 2304) -> (768, 768*3)
            new_block.attn.c_attn.weight.data = block.attn.c_attn.weight.data.mm(ortho_matrix)
            new_block.attn.c_proj.weight.data = ortho_matrix.T.mm(block.attn.c_proj.weight.data)
            new_block.attn.c_proj.bias.data = ortho_matrix.T @ block.attn.c_proj.bias.data
            # q, k 矩阵单独旋转
            print(f"local qk rotate at layer {layer}")
            local_qk_rotate(new_block)
            # v 矩阵和linear层的权重单独旋转
            print(f"local v rotate at layer {layer}")
            local_v_rotate(new_block)
            # MLP权重
            # fc1 (768 to 3072)
            new_block.mlp.c_fc.weight.data = block.mlp.c_fc.weight.data.mm(ortho_matrix)
            # fc2 (3072 to 768)
            new_block.mlp.c_proj.weight.data = ortho_matrix.T.mm(block.mlp.c_proj.weight.data)
            new_block.mlp.c_proj.bias.data = ortho_matrix.T @ block.mlp.c_proj.bias.data
    return new_model

def main(seed):
    # 以下得到相同输出
    torch.manual_seed(seed)
    new_model = rotate(model)
    x = torch.tensor([enc.encode('Once upon a time, there is')]).to(device)
    out = model.generate(x, max_new_tokens=32, do_sample=False)[0].tolist()
    print(f"original: \n{enc.decode(out)}\n")
    rotated_out = new_model.generate(x, max_new_tokens=32, do_sample=False)[0].tolist()
    print(f"rotated: \n{enc.decode(rotated_out)}\n")
    print(f"Is generation equal: {enc.decode(out) == enc.decode(rotated_out)}")
    print(f"Max Probs difference: {(F.softmax(model(x)[0], dim=-1) - F.softmax(new_model(x)[0], dim=-1)).abs().max().item()}")
    print(f"Max Parmas difference in wte: {(model.transformer.wte.weight - new_model.transformer.wte.weight).abs().max().item()}")

if __name__ == "__main__":
    # 以下例子展示单参数rmsnorm的旋转等价性
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = '/data/my_tools/build-nanogpt/log/rmsnorm_1_param_step_02000_upon_rmsnorm_gpt2_20241123_214855_step_19072.pt'
    model_name = model_path.split('/')[-1]
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model_config = GPTConfig(**checkpoint['config'])
    model = GPT(model_config).to(device)
    enc = tiktoken.get_encoding("gpt2")
    n_embd = model.config.n_embd
    n_head = model.config.n_head
    head_dim = n_embd // n_head

    # 模型从layernorm转换成rmsnorm
    model = replace_layernorm_with_rmsnorm(model)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    main(42)