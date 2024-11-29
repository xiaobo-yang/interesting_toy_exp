import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    nanogpt2 + multiple head for softmax(next token distribution)
"""

from models.model_gpt2 import GPT, Block

class BlockHook(Block):
    def __init__(self, config, is_hook=False):
        super().__init__(config)
        self.is_hook = is_hook
        if self.is_hook:
            self.inter_ln_f = nn.LayerNorm(config.n_embd)
            self.inter_lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
    
    def forward(self, x):
        x = super().forward(x)
        inter_logits = self.inter_lm_head(self.inter_ln_f(x)) if self.is_hook else None
        return x, inter_logits

class GPT2MultiSoftmax(GPT):
    def __init__(self, config, hook_layers=[], freeze_base=False):
        super().__init__(config)
        # 确保最后一层总是包含在hook_layers中
        self.hook_layers = hook_layers if (config.n_layer - 1) in hook_layers else hook_layers + [config.n_layer - 1]
        self.transformer.h = nn.ModuleList([
            BlockHook(config, is_hook=True if i in self.hook_layers else False) 
            for i in range(config.n_layer)
        ])
        
        # 删除原始的最后层
        del self.transformer.ln_f
        del self.lm_head
        
        
        # 如果需要冻结基础模型参数
        if freeze_base:
            self._freeze_base_parameters()
    
    def _init_hook_weights(self, base_model):
        """使用原始模型的最后层权重初始化所有hook层"""
        # 保存最后的ln_f和lm_head权重用于初始化
        final_ln_weight = base_model.transformer.ln_f.weight.clone()
        final_ln_bias = base_model.transformer.ln_f.bias.clone()
        final_lm_weight = base_model.lm_head.weight.clone()

        for i, block in enumerate(self.transformer.h):
            if block.is_hook:
                # 初始化layer norm
                block.inter_ln_f.weight.data.copy_(final_ln_weight)
                block.inter_ln_f.bias.data.copy_(final_ln_bias)
                # 初始化linear head
                block.inter_lm_head.weight.data.copy_(final_lm_weight)
    
    def _freeze_base_parameters(self):
        """冻结基础模型的所有参数"""
        for name, param in self.named_parameters():
            if not any(x in name for x in ['inter_ln_f', 'inter_lm_head']):
                param.requires_grad = False
    
    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb
        
        inter_logits = {}
        for i, block in enumerate(self.transformer.h):
            x, block_logits = block(x)
            if block.is_hook:
                inter_logits[i] = block_logits
        
        loss = None
        if targets is not None:
            loss = 0
            for layer, logits in inter_logits.items():
                loss += F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss /= len(self.hook_layers)

        return inter_logits, loss

    @torch.no_grad()
    def generate(self, orig_idx, max_new_tokens, do_sample, temperature=1.0, top_k=None, eot_token=50256):
        idx_list = {layer: orig_idx.clone() for layer in self.hook_layers}
        for layer in self.hook_layers:
            idx = idx_list[layer]
            for _ in range(max_new_tokens):
                idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
                inter_logits, _ = self(idx_cond)
                logits = inter_logits[layer]
                
                # 在这里添加截断操作，将logits限制在50257范围内（训练时使用了50304以加速）
                logits = logits[:, -1, :50257]
                
                if do_sample:
                    logits = logits / temperature
                    if top_k is not None:
                        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                        logits[logits < v[:, [-1]]] = -float('Inf')
                    probs = F.softmax(logits, dim=-1)
                    idx_next = torch.multinomial(probs, num_samples=1)
                else:
                    idx_next = torch.argmax(logits, dim=-1, keepdim=True)
                
                idx = torch.cat((idx, idx_next), dim=1)
            idx_list[layer] = idx
        
        return idx_list


    @classmethod
    def from_pretrained(cls, model_type, hook_layers=[], freeze_base=False):
        """从预训练模型加载并添加多softmax头"""
        # 首先加载原始GPT模型
        base_model = super().from_pretrained(model_type)

        # 创建新的配置
        config = base_model.config
        
        # 创建多softmax模型
        model = cls(config, hook_layers=hook_layers, freeze_base=freeze_base)
        
        # 加载基础模型的权重
        model.load_state_dict(base_model.state_dict(), strict=False)
        
        # 初始化新增层的权重
        model._init_hook_weights(base_model)
        
        return model

