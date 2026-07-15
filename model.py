import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Config:
    vocab_size  = 1024
    block_size  = 256
    n_layer     = 3
    n_head      = 8
    n_kv_head   = 2
    n_embd      = 256
    ffn_hidden  = 512
    dropout     = 0.0
    tie_weights = True

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm_x = torch.mean(x * x, dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return self.weight * x_normed

def apply_rotary_emb(x, freqs_cis):
    # x: (B, T, n_heads, head_dim)
    x_c = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)
    x_out = torch.view_as_real(x_c * freqs_cis).flatten(3)
    return x_out.type_as(x)

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.n_head
        self.n_kv_heads = cfg.n_kv_head
        self.head_dim = cfg.n_embd // cfg.n_head
        
        self.wq = nn.Linear(cfg.n_embd, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.n_embd, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg.n_embd, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, cfg.n_embd, bias=False)
        
    def forward(self, x, freqs_cis):
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)
        
        q, k = apply_rotary_emb(q, freqs_cis), apply_rotary_emb(k, freqs_cis)
        
        # Grouped Query Attention replication
        k = k.repeat_interleave(self.n_heads // self.n_kv_heads, dim=2)
        v = v.repeat_interleave(self.n_heads // self.n_kv_heads, dim=2)
        
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(y)

class SwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.w1 = nn.Linear(cfg.n_embd, cfg.ffn_hidden, bias=False)
        self.w2 = nn.Linear(cfg.n_embd, cfg.ffn_hidden, bias=False)
        self.w3 = nn.Linear(cfg.ffn_hidden, cfg.n_embd, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attention = CausalSelfAttention(cfg)
        self.feed_forward = SwiGLU(cfg)
        self.attention_norm = RMSNorm(cfg.n_embd)
        self.ffn_norm = RMSNorm(cfg.n_embd)

    def forward(self, x, freqs_cis):
        h = x + self.attention(self.attention_norm(x), freqs_cis)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.layers = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm = RMSNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        
        if cfg.tie_weights:
            self.head.weight = self.tok_embeddings.weight
        
        # Precompute RoPE frequencies
        freqs = 1.0 / (10000.0 ** (torch.arange(0, cfg.n_embd // cfg.n_head, 2)[: (cfg.n_embd // cfg.n_head // 2)].float() / (cfg.n_embd // cfg.n_head)))
        t = torch.arange(cfg.block_size * 2, dtype=torch.float32)
        freqs = torch.outer(t, freqs).float()
        self.register_buffer('freqs_cis', torch.polar(torch.ones_like(freqs), freqs), persistent=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_embeddings(idx)
        freqs_cis = self.freqs_cis[:T]
        
        for layer in self.layers:
            x = layer(x, freqs_cis)
            
        x = self.norm(x)
        logits = self.head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
            
        return logits, loss

    def n_params(self):
        seen = set()
        total = 0
        for p in self.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                total += p.numel()
        return total
