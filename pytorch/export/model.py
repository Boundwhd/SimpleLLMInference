import torch
import torch.nn as nn
import numpy as np
import torch

def safe_softmax(x, dim=-1):
    max_vals = torch.max(x, dim=dim, keepdim=True).values  
    x_stable = x - max_vals  
    
    exp_x = torch.exp(x_stable)
    sum_exp = torch.sum(exp_x, dim=dim, keepdim=True) 
    softmax = exp_x / sum_exp
    
    return softmax

def precompute_sin_cos_cache(
    dim: int, 
    max_seq_len: int, 
    theta: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(max_seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)  
    
    sin_cache = torch.sin(freqs)
    cos_cache = torch.cos(freqs)

    return sin_cache, cos_cache

def apply_rope(query, key, sin_cache, cos_cache, pos):

    head_dim = query.size(-1)
    half_dim = head_dim // 2
    
    sin = sin_cache[pos, :half_dim]
    cos = cos_cache[pos, :half_dim]
    
    q1, q2 = query[..., :half_dim], query[..., half_dim:]
    k1, k2 = key[..., :half_dim], key[..., half_dim:]
    
    new_q1 = q1 * cos - q2 * sin
    new_q2 = q1 * sin + q2 * cos
    
    new_k1 = k1 * cos - k2 * sin
    new_k2 = k1 * sin + k2 * cos
    
    rotated_query = torch.cat([new_q1, new_q2], dim=-1)
    rotated_key = torch.cat([new_k1, new_k2], dim=-1)
    
    return rotated_query, rotated_key

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.randn(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        
        return self.weight * (x / rms)
    
class MultiheadAttention(nn.Module):
    def __init__(self, dim, nhead, sin_cache, cos_cache):
        super().__init__()
        self.dim = dim
        self.nhead = nhead
        self.head_dim = dim // nhead
        
        self.sin_cache = sin_cache
        self.cos_cache = cos_cache
        
        self.rmsnorm = RMSNorm(dim)
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        
        self.register_buffer("key_cache", torch.zeros((0, nhead, self.head_dim)))
        self.register_buffer("value_cache", torch.zeros((0, nhead, self.head_dim)))
        
    def clear_cache(self):
        self.key_cache = torch.zeros((0, self.nhead, self.head_dim), device=self.key_cache.device)
        self.value_cache = torch.zeros((0, self.nhead, self.head_dim), device=self.value_cache.device)
        
    def forward(self, x, pos):
        rms_output = self.rmsnorm(x)
        query = self.wq(rms_output)
        key = self.wk(rms_output)
        value = self.wv(rms_output)
        
        query = query.view(1, self.nhead, self.head_dim)     
        key = key.view(1, self.nhead, self.head_dim)
        value = value.view(1, self.nhead, self.head_dim)
        
        query, key = apply_rope(query, key, self.sin_cache, self.cos_cache, pos)  
        
        self.key_cache = torch.cat([self.key_cache, key], dim=0)
        self.value_cache = torch.cat([self.value_cache, value], dim=0)
        
        scores = torch.einsum("bhd,shd->bhs", query, self.key_cache)
        scores = safe_softmax(scores, dim=-1)
        
        attn_out = torch.einsum("bhs,shd->bhd", scores, self.value_cache)
        attn_out = attn_out.reshape(1, self.dim)
        return attn_out

class Transfomer(nn.Module):
    def __init__(self, num_layers=2, dim=768, nhead=12, max_len=1024):
        super().__init__()
        head_dim = dim // nhead
        sin_cache, cos_cache = precompute_sin_cos_cache(head_dim, max_len)
          
        self.register_buffer('sin_cache', sin_cache)
        self.register_buffer('cos_cache', cos_cache)
        
        self.embedding = nn.Embedding(32, dim)
        self.layers = nn.ModuleList([
            MultiheadAttention(dim, nhead, self.sin_cache, self.cos_cache)
            for _ in range(num_layers)
        ])
    
    def forward(self, x, pos):
        x = self.embedding(x)
        
        for layer in self.layers:
            x = layer(x, pos)
        
        return x

if __name__ == "__main__":
    num_layer = 1
    model = Transfomer(num_layers=num_layer)  # 你的模型
    input = torch.tensor([2], dtype=torch.int)
    
    pos = 0
    for cyc in range(2):
        attn = model(input, pos)
        input = torch.tensor([3], dtype=torch.int)
        pos = pos + 1
        
    with open("output.txt", "w") as f:
        for i in range(attn.shape[1]):  
            value = attn[0, i].item()  
            f.write(f"{value:.6f}\n")
          
    weight_order = ['embedding.weight']   
    for i in range(num_layer):
        weight_order.extend([
        f'layers.{i}.rmsnorm.weight',
    ])
    for i in range(num_layer):
        weight_order.extend([
        f'layers.{i}.wq.weight',
    ])
    for i in range(num_layer):
        weight_order.extend([
        f'layers.{i}.wk.weight',
    ])
    for i in range(num_layer):
        weight_order.extend([
        f'layers.{i}.wv.weight',
    ])

    all_weights = []
    for param_name in weight_order:
        param_tensor = model.state_dict()[param_name].detach().cpu().numpy()
        all_weights.append(param_tensor.flatten())

    merged_weights = np.concatenate(all_weights)
    merged_weights.tofile("model_weights.bin")

    print(f"Saved all weights to model_weights.bin")
    print(f"Total size: {merged_weights.size} floats ({merged_weights.size * 4} bytes)")
