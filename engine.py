"""
The inference Engine: sample from the model efficiently (KV cache, many samples
in parallel), and run the python tool whenever the model asks for it.
Mirrors nanochat/engine.py.
"""

from collections import deque
import torch
import torch.nn.functional as F

from gpt import KVCache

def use_calculator(expr):
    """ A very restricted python "tool": arithmetic, and str.count (for counting letters) """
    expr = expr.replace(",", "")
    is_math = all(c in "0123456789*+-/.() " for c in expr) and "**" not in expr # no 9**9**9 please
    is_count = all(c.isalnum() or c in "'\"()._ " for c in expr) and ".count(" in expr and "__" not in expr
    if not (is_math or is_count):
        return None
    try:
        return eval(expr, {"__builtins__": {}}, {})
    except Exception:
        return None # the model wrote something broken, that's ok

@torch.no_grad()
def sample_next_token(logits, generator, temperature=1.0, top_k=None):
    if temperature == 0.0:
        return logits.argmax(dim=-1, keepdim=True)
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits = logits.masked_fill(logits < v[:, [-1]], -float('inf'))
    probs = F.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator)

class Engine:

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @torch.no_grad()
    def generate(self, tokens, num_samples=1, max_tokens=256, temperature=1.0, top_k=None, seed=42):
        """ Streams back one column of tokens (one per sample) at a time, and whether each was sampled (1) or forced (0) """
        model, s = self.model, self.tokenizer.encode_special
        device = next(model.parameters()).device
        python_start, python_end = s("<|python_start|>"), s("<|python_end|>")
        output_start, output_end = s("<|output_start|>"), s("<|output_end|>")
        stop_tokens = {s("<|assistant_end|>"), self.tokenizer.bos}
        generator = torch.Generator(device=device).manual_seed(seed)

        # 1) prefill: run the whole prompt through the model once, with batch size 1
        kv_cache = [KVCache() for _ in range(model.config.n_layer)]
        logits = model(torch.tensor([tokens], device=device), kv_cache=kv_cache)[:, -1, :]
        # 2) then replicate that work for each of the samples, instead of redoing it
        for c in kv_cache:
            c.expand(num_samples)
        logits = logits.expand(num_samples, -1)

        # 3) decode, one token at a time, keeping track of the tool use state of each row
        forced = [deque() for _ in range(num_samples)] # tokens we must insert (the tool output)
        python_expr = [None] * num_samples # the tokens of the python expression being written
        done = [False] * num_samples
        max_tokens = min(max_tokens, model.config.block_size - len(tokens))
        for _ in range(max_tokens):
            sampled = sample_next_token(logits, generator, temperature, top_k)[:, 0].tolist()
            column, masks = [], []
            for i in range(num_samples):
                is_forced = len(forced[i]) > 0
                token = forced[i].popleft() if is_forced else sampled[i]
                column.append(token)
                masks.append(0 if is_forced else 1)
                if token in stop_tokens:
                    done[i] = True
                elif token == python_start:
                    python_expr[i] = []
                elif token == python_end and python_expr[i] is not None:
                    # the model wants to use the tool! run it and force its output into the stream
                    result = use_calculator(self.tokenizer.decode(python_expr[i]))
                    if result is not None:
                        forced[i].extend([output_start, *self.tokenizer.encode(str(result)), output_end])
                    python_expr[i] = None
                elif python_expr[i] is not None:
                    python_expr[i].append(token)
            yield column, masks
            if all(done):
                break
            logits = model(torch.tensor(column, device=device).unsqueeze(1), kv_cache=kv_cache)[:, -1, :]

    def generate_batch(self, tokens, num_samples=1, **kwargs):
        """ Non-streaming version: returns the token sequences (prompt included) and their masks """
        stop_tokens = {self.tokenizer.encode_special("<|assistant_end|>"), self.tokenizer.bos}
        results = [list(tokens) for _ in range(num_samples)]
        masks = [[0] * len(tokens) for _ in range(num_samples)]
        done = [False] * num_samples
        for column, column_masks in self.generate(tokens, num_samples, **kwargs):
            for i, (token, mask) in enumerate(zip(column, column_masks)):
                if done[i] or token in stop_tokens:
                    done[i] = True
                    continue
                results[i].append(token)
                masks[i].append(mask)
        return results, masks
