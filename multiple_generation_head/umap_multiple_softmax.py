import torch
import matplotlib.pyplot as plt
import umap
import numpy as np


from models.model_gpt2_multi_sofmax import GPT2MultiSoftmax

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# 创建带有多个softmax头的模型，在第6层和最后一层添加预测头
hook_layers = list(range(7,12))  # 假设是12层模型
model_type = 'gpt2_model_19072.pt'
model_name = f'{model_type}-multi-softmax'
model = GPT2MultiSoftmax.model_surgery_from_pretrained(
    model_type,
    hook_layers=hook_layers,
    freeze_base=True  # 设置为True则只训练新增的层
).to(device)
model.load_state_dict(torch.load('log/gpt2_model_19072.pt-multi-softmax_20241129_223536_step_05000.pt')['model'])

reducer = umap.UMAP(
    n_components=2,
    n_neighbors=15,       # 增大这个值会使结构更平滑
    min_dist=0.1,        # 减小这个值会使聚类更紧密
    metric='cosine',     # 使用余弦距离可能更适合词嵌入
    random_state=42
)
# wte umap
for layer in hook_layers:
    embeddings = model.transformer.h[layer].inter_lm_head.weight[:50257].detach().cpu().numpy()
    embedding_2d = reducer.fit_transform(embeddings)
    plt.figure(figsize=(10, 10))
    norms = np.linalg.norm(embeddings, axis=1)
    plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], 
            c=norms, alpha=0.1, s=1, cmap='viridis')
    plt.colorbar(label='Vector norm')
    plt.title(f'UMAP visualization of {model_name} layer{layer} lm_head embeddings')
    plt.savefig(f'figs/{model_name}_lm_head_umap_layer{layer}.png', dpi=300, bbox_inches='tight')
    plt.close()
