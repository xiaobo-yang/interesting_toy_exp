import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import umap
import numpy as np
import tiktoken
import pandas as pd

import sys
sys.path.append("/data/my_tools/build-nanogpt")
from modelling_gpt2 import GPT, GPTConfig




def wte_umap_with_longest_tokens(model, model_name):
    # 将词嵌入权重转换为numpy数组
    embeddings = model.transformer.wte.weight.detach().cpu().numpy()

    # 创建UMAP对象并进行降维
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,       # 增大这个值会使结构更平滑
        min_dist=0.1,        # 减小这个值会使聚类更紧密
        metric='cosine',     # 使用余弦距离可能更适合词嵌入
        random_state=42
    )
    embedding_2d = reducer.fit_transform(embeddings)

    # 绘制散点图
    plt.figure(figsize=(15, 15))

    longest_20_indices = df_sorted['token_id'].head(20).values

    norms = np.linalg.norm(embeddings, axis=1)
    plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], 
            c=norms, alpha=0.1, s=1, cmap='viridis')

    # 标注最长的20个token
    for idx in longest_20_indices:
        token_text = df.loc[df['token_id'] == idx, 'decoded'].iloc[0]
        x, y = embedding_2d[idx]
        plt.plot(x, y, 'rx', markersize=10)  # 红色叉号标记
        # 添加文本标签，略微偏移以避免重叠
        plt.annotate(token_text[:10] + '...', 
                    (x, y),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=8,
                    alpha=0.7)

    plt.colorbar(label='Vector norm')
    plt.title(f'UMAP visualization of {model_name} embeddings\nRed crosses mark longest tokens')
    plt.tight_layout()
    plt.savefig(f'{model_name}_wte_umap_with_longest_tokens.png', dpi=300, bbox_inches='tight')
    plt.close()


def wte_wpe_umap(model, model_name):
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,       # 增大这个值会使结构更平滑
        min_dist=0.1,        # 减小这个值会使聚类更紧密
        metric='cosine',     # 使用余弦距离可能更适合词嵌入
        random_state=42
    )
    # wte umap
    embeddings = model.transformer.wte.weight[:enc.n_vocab].detach().cpu().numpy()
    embedding_2d = reducer.fit_transform(embeddings)
    plt.figure(figsize=(10, 10))
    norms = np.linalg.norm(embeddings, axis=1)
    plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], 
            c=norms, alpha=0.1, s=1, cmap='viridis')
    plt.colorbar(label='Vector norm')
    plt.title(f'UMAP visualization of {model_name} wte embeddings')
    plt.savefig(f'{model_name}_wte_umap.png', dpi=300, bbox_inches='tight')
    plt.close()
    # wpe umap
    embeddings = model.transformer.wpe.weight.detach().cpu().numpy()
    embedding_2d = reducer.fit_transform(embeddings)
    plt.figure(figsize=(10, 10))
    norms = np.linalg.norm(embeddings, axis=1)
    plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], 
            c=norms, alpha=0.1, s=1, cmap='viridis')
    plt.colorbar(label='Vector norm')
    plt.title(f'UMAP visualization of {model_name} wpe embeddings')
    plt.savefig(f'{model_name}_wpe_umap.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # model_path = '/home/yangxiaobo/my_tools/build-nanogpt/log/gpt2-medium_model_19072.pt'
    # model_name = model_path.split('/')[-1]
    # checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    # model_config = GPTConfig(**checkpoint['config'])
    # model = GPT(model_config).to(device)
    # model.load_state_dict(checkpoint['model'])
    # wte_wpe_umap(model, model_name)

    # 获取GPT2 tokenizer
    enc = tiktoken.get_encoding("gpt2")

    # 获取所有tokens的解码结果和长度
    token_info = []
    for i in range(enc.n_vocab):
        decoded = enc.decode([i])
        token_info.append({
            'token_id': i,
            'decoded': decoded,
            'length': len(decoded),
            'is_special': i in enc.special_tokens_set
        })

    # 转换为DataFrame并排序
    df = pd.DataFrame(token_info)
    df_sorted = df.sort_values('length', ascending=False)

    # 打印特殊token
    print("\nGPT2特殊token:")
    special_tokens = df[df['is_special']]
    print(special_tokens[['token_id', 'decoded']])

    # 打印最长的20个token
    print("\n最长的20个token:")
    print(df_sorted[['token_id', 'decoded', 'length']].head(20))

    for model_type in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']:
        model_path = f'/home/yangxiaobo/my_tools/build-nanogpt/log/{model_type}_model_19072.pt'
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        model_config = GPTConfig(**checkpoint['config'])
        model = GPT(model_config).to(device)
        model.load_state_dict(checkpoint['model'])
        model_name = model_path.split('/')[-1]
        wte_umap_with_longest_tokens(model, model_name)


