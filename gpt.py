"""
The GPT-2 (124M) Transformer, exactly as in "Let's reproduce GPT-2", with two additions
that post-training needs:
1) a KV cache, so that sampling (which we do a LOT of in RL) is fast
2) a few extra rows in the token embedding table, for our new special tokens
"""

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import download_file

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # 50257 GPT-2 tokens, padded up to a nice number (and room for special tokens!)
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class KVCache:
    """ Remembers the keys and values of all the tokens seen so far, for one layer. """

    def __init__(self):
        self.k = self.v = None

    def update(self, k, v):
        if self.k is not None:
            k = torch.cat([self.k, k], dim=2)
            v = torch.cat([self.v, v], dim=2)
        self.k, self.v = k, v
        return k, v

    def expand(self, batch_size):
        # prefill once with batch 1, then copy the cache for each of the samples
        self.k = self.k.expand(batch_size, -1, -1, -1)
        self.v = self.v.expand(batch_size, -1, -1, -1)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head

    def forward(self, x, kv_cache=None):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(C, dim=2)
        q, k, v = [t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v)] # (B, nh, T, hs)
        if kv_cache is not None:
            assert kv_cache.k is None or T == 1, "with a non-empty cache we feed in one token at a time"
            k, v = kv_cache.update(k, v) # k, v now also contain all the previous tokens
        y = F.scaled_dot_product_attention(q, k, v, is_causal=T > 1) # a single new token may look at everything
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x), approximate='tanh'))

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, kv_cache=None):
        x = x + self.attn(self.ln_1(x), kv_cache)
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # weight sharing scheme

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        B, T = idx.size()
        t0 = 0 if kv_cache is None or kv_cache[0].k is None else kv_cache[0].k.size(2) # tokens already in the cache
        assert t0 + T <= self.config.block_size, f"sequence of length {t0 + T} is too long"
        pos = torch.arange(t0, t0 + T, dtype=torch.long, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for i, block in enumerate(self.transformer.h):
            x = block(x, None if kv_cache is None else kv_cache[i])
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x).float()
        if targets is None:
            return logits
        # targets of -1 are ignored: this is how we will choose which tokens to train on
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
        return loss

    @classmethod
    def from_pretrained(cls):
        """ Load the OpenAI GPT-2 (124M) weights from HuggingFace, without needing the transformers library """
        from safetensors.torch import load_file
        path = download_file("https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors", "gpt2/model.safetensors")
        sd_hf = load_file(path)
        model = GPT(GPTConfig())
        sd = model.state_dict()
        # OpenAI used a "Conv1D" module, so these weights are transposed relative to nn.Linear
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        for k, w in sd_hf.items():
            if k.endswith('.attn.bias') or k.endswith('.attn.masked_bias'):
                continue # these are just the causal mask buffers, not parameters
            ours = 'transformer.' + k
            if any(k.endswith(t) for t in transposed):
                w = w.t()
            if k == 'wte.weight':
                # GPT-2 has 50257 tokens. Copy those in, and initialize all the new rows
                # (that our special tokens will use) to the average token embedding.
                n = w.size(0)
                with torch.no_grad():
                    sd[ours][:n].copy_(w)
                    sd[ours][n:].copy_(w.mean(dim=0))
                continue
            assert sd[ours].shape == w.shape, f"shape mismatch at {k}"
            with torch.no_grad():
                sd[ours].copy_(w)
        return model

    def configure_optimizer(self, lr, weight_decay=0.0):
        # same as the GPT-2 video: decay the 2D weights, don't decay biases and layernorms
        decay = [p for p in self.parameters() if p.dim() >= 2]
        no_decay = [p for p in self.parameters() if p.dim() < 2]
        groups = [{'params': decay, 'weight_decay': weight_decay}, {'params': no_decay, 'weight_decay': 0.0}]
        return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), eps=1e-8)

def load_model(source, device):
    """ source is one of: base (OpenAI GPT-2) | sft | rl (our own checkpoints) """
    import os
    from common import get_base_dir
    if source == "base":
        model = GPT.from_pretrained()
    else:
        model = GPT(GPTConfig())
        path = os.path.join(get_base_dir(), f"chat{source}.pt")
        model.load_state_dict(torch.load(path, map_location="cpu"))
    return model.to(device)
