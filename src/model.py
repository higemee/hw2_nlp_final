"""Encoder-decoder Transformer: pre-norm RMSNorm, RoPE in self-attention, SwiGLU, tied embeddings."""
import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 8192
    d_model: int = 512
    n_heads: int = 8
    n_enc_layers: int = 6
    n_dec_layers: int = 8
    ffn_mult: float = 8 / 3
    max_len: int = 1024
    rope_base: float = 10000.0
    pad_id: int = 0

    def to_dict(self):
        return asdict(self)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


class Rotary(nn.Module):
    def __init__(self, head_dim, max_len, base):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        freqs = torch.outer(torch.arange(max_len).float(), inv_freq)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, x, offset=0):
        # x: (B, H, T, head_dim)
        T = x.size(2)
        cos = self.cos[offset:offset + T].to(x.dtype)[None, None]
        sin = self.sin[offset:offset + T].to(x.dtype)[None, None]
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


class SelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig, causal: bool):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.causal = causal
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x, rope, attn_mask=None, cache=None, offset=0):
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k = rope(q, offset), rope(k, offset)
        if cache is not None:
            cache["k"][:, :, offset:offset + T] = k
            cache["v"][:, :, offset:offset + T] = v
            k = cache["k"][:, :, :offset + T]
            v = cache["v"][:, :, :offset + T]
        if attn_mask is not None:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal and T > 1)
        return self.out(y.transpose(1, 2).reshape(B, T, C))


class CrossAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.q = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.kv = nn.Linear(cfg.d_model, 2 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def project_kv(self, mem):
        B, S, C = mem.shape
        k, v = self.kv(mem).view(B, S, 2, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        return k, v

    def forward(self, x, kv, mem_mask):
        B, T, C = x.shape
        q = self.q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k, v = kv
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mem_mask)
        return self.out(y.transpose(1, 2).reshape(B, T, C))


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden = (int(cfg.ffn_mult * cfg.d_model) + 63) // 64 * 64
        self.w12 = nn.Linear(cfg.d_model, 2 * hidden, bias=False)
        self.w3 = nn.Linear(hidden, cfg.d_model, bias=False)

    def forward(self, x):
        a, b = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(a) * b)


class EncoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n1 = RMSNorm(cfg.d_model)
        self.attn = SelfAttention(cfg, causal=False)
        self.n2 = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg)

    def forward(self, x, rope, mask):
        x = x + self.attn(self.n1(x), rope, attn_mask=mask)
        return x + self.ffn(self.n2(x))


class DecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n1 = RMSNorm(cfg.d_model)
        self.self_attn = SelfAttention(cfg, causal=True)
        self.n2 = RMSNorm(cfg.d_model)
        self.cross_attn = CrossAttention(cfg)
        self.n3 = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg)

    def forward(self, x, rope, cross_kv, mem_mask, cache=None, offset=0):
        x = x + self.self_attn(self.n1(x), rope, cache=cache, offset=offset)
        x = x + self.cross_attn(self.n2(x), cross_kv, mem_mask)
        return x + self.ffn(self.n3(x))


class Seq2Seq(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.rope = Rotary(cfg.d_model // cfg.n_heads, cfg.max_len, cfg.rope_base)
        self.enc_layers = nn.ModuleList([EncoderLayer(cfg) for _ in range(cfg.n_enc_layers)])
        self.enc_norm = RMSNorm(cfg.d_model)
        self.dec_layers = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg.n_dec_layers)])
        self.dec_norm = RMSNorm(cfg.d_model)
        self.apply(self._init)
        # scaled init of the projections that write into the residual stream (GPT-2)
        n_res = 2 * cfg.n_enc_layers + 3 * cfg.n_dec_layers
        for name, p in self.named_parameters():
            if name.endswith("out.weight") or name.endswith("w3.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(n_res))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def encode(self, src_ids):
        mask = (src_ids != self.cfg.pad_id)[:, None, None, :]
        x = self.embed(src_ids)
        for layer in self.enc_layers:
            x = layer(x, self.rope, mask)
        return self.enc_norm(x), mask

    def logits(self, h):
        return F.linear(h, self.embed.weight)

    def forward(self, src_ids, tgt_in, tgt_out=None):
        """Teacher forcing. Returns logits, or (summed loss, number of target tokens) if tgt_out is given."""
        mem, mem_mask = self.encode(src_ids)
        x = self.embed(tgt_in)
        for layer in self.dec_layers:
            x = layer(x, self.rope, layer.cross_attn.project_kv(mem), mem_mask)
        h = self.dec_norm(x)
        if tgt_out is None:
            return self.logits(h)
        sel = tgt_out != self.cfg.pad_id
        loss = F.cross_entropy(self.logits(h[sel]).float(), tgt_out[sel], reduction="sum")
        return loss, sel.sum()

    @torch.no_grad()
    def generate(self, src_ids, bos_id, eos_id, max_new_tokens=256, temperature=0.0, top_p=1.0, num_samples=1):
        """Batched greedy (temperature=0) or sampled decoding with a KV cache.

        Returns B * num_samples token-id lists without BOS/EOS; samples of one source row are adjacent.
        Finished rows are removed from the batch once a quarter of it has finished.
        """
        mem, mem_mask = self.encode(src_ids)
        if num_samples > 1:
            mem = mem.repeat_interleave(num_samples, 0)
            mem_mask = mem_mask.repeat_interleave(num_samples, 0)
        B = mem.size(0)
        device, dtype = mem.device, mem.dtype
        cross_kv = [layer.cross_attn.project_kv(mem) for layer in self.dec_layers]
        H, hd = self.cfg.n_heads, self.cfg.d_model // self.cfg.n_heads
        caches = [{"k": torch.empty(B, H, max_new_tokens + 1, hd, device=device, dtype=dtype),
                   "v": torch.empty(B, H, max_new_tokens + 1, hd, device=device, dtype=dtype)}
                  for _ in self.dec_layers]
        out = torch.full((B, max_new_tokens), eos_id, dtype=torch.long, device=device)
        active = torch.arange(B, device=device)  # row of the compacted batch -> original row
        alive = torch.ones(B, dtype=torch.bool, device=device)
        cur = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        for t in range(max_new_tokens):
            x = self.embed(cur)
            for li, layer in enumerate(self.dec_layers):
                x = layer(x, self.rope, cross_kv[li], mem_mask, cache=caches[li], offset=t)
            logits = self.logits(self.dec_norm(x[:, -1])).float()
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                if top_p < 1.0:
                    sp, si = probs.sort(dim=-1, descending=True)
                    sp[(sp.cumsum(-1) - sp) > top_p] = 0
                    probs = torch.zeros_like(probs).scatter_(-1, si, sp)
                nxt = torch.multinomial(probs, 1).squeeze(-1)
            else:
                nxt = logits.argmax(-1)
            nxt = torch.where(alive, nxt, torch.full_like(nxt, eos_id))
            out[active, t] = nxt
            alive = alive & (nxt != eos_id)
            n_alive = int(alive.sum())
            if n_alive == 0:
                break
            if n_alive <= 0.75 * alive.numel():
                keep = alive
                active, nxt, alive = active[keep], nxt[keep], alive[keep]
                mem_mask = mem_mask[keep]
                cross_kv = [(k[keep], v[keep]) for k, v in cross_kv]
                for c in caches:
                    c["k"], c["v"] = c["k"][keep], c["v"][keep]
            cur = nxt[:, None]
        return [row[:row.index(eos_id)] if eos_id in row else row for row in out.tolist()]
