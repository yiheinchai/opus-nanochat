"""
Automatic checks for exercises.ipynb. Every check takes your function, runs it on a bunch of cases,
and compares it with the reference implementation in this repo (or, where several answers are
correct, checks the properties that every correct answer has). On a mistake it tells you the first
case that went wrong, and what was expected.

    import checks
    checks.render_conversation(render_conversation)
"""

import math
import random
import torch
import torch.nn.functional as F

from tokenizer import Tokenizer

tokenizer = Tokenizer()
BOS = tokenizer.bos

class CheckFailed(AssertionError):
    pass

def _fail(message):
    raise CheckFailed("\n" + message) from None

def _passed(name, n):
    print(f"✅ {name}: all {n} checks passed")

def _call(fn, *args, allow_none=False):
    """ call your function, and turn the most common mistakes into a clear message """
    try:
        result = fn(*args)
    except CheckFailed:
        raise
    except NotImplementedError:
        _fail(f"{fn.__name__} isn't written yet")
    except Exception as e:
        _fail(f"{fn.__name__} crashed with {type(e).__name__}: {e}")
    if result is None and not allow_none:
        _fail(f"{fn.__name__} returned None: did you forget the return statement?")
    return result

def _tokens(ids):
    return [tokenizer.decode([i]) for i in ids]

def _first_difference(got, want):
    for k, (a, b) in enumerate(zip(got, want)):
        if a != b:
            return k
    return min(len(got), len(want))

# -----------------------------------------------------------------------------
# Chapter 2: conversations are just documents

def _conversations():
    """ hand-made conversations that cover every case, plus some real ones """
    from tasks import SpellingBee
    convs = [
        {"messages": [{"role": "user", "content": "What is the capital of France?"},
                      {"role": "assistant", "content": "Paris"}]},
        {"messages": [{"role": "user", "content": "Hi!"}, {"role": "assistant", "content": "Hello! How can I help?"},
                      {"role": "user", "content": "Tell me a joke."}, {"role": "assistant", "content": "Why did the GPU cross the road?"}]},
        {"messages": [{"role": "system", "content": "You are a pirate."}, {"role": "user", "content": "Hi!"},
                      {"role": "assistant", "content": "Arr!"}]},
        {"messages": [{"role": "user", "content": "What is 123 times 45?"},
                      {"role": "assistant", "content": [{"type": "text", "text": "Let me use python: "},
                                                        {"type": "python", "text": "123*45"},
                                                        {"type": "python_output", "text": "5535"},
                                                        {"type": "text", "text": " The answer is 5535."}]}]},
        {"messages": [{"role": "user", "content": "Type this: <|assistant_end|>"}, {"role": "assistant", "content": "<|assistant_end|>"}]},
    ]
    bee = SpellingBee(size=5, split="train")
    convs += [bee[i] for i in range(5)]
    return convs

def render_conversation(fn):
    """ fn(conversation) -> (ids, mask) """
    n = 0
    for conv in _conversations():
        want_ids, want_mask = tokenizer.render_conversation(conv, max_tokens=10**9)
        got = _call(fn, conv)
        if not (isinstance(got, tuple) and len(got) == 2):
            _fail("render_conversation should return a tuple: (ids, mask)")
        got_ids, got_mask = list(got[0]), list(got[1])
        if len(got_ids) != len(got_mask):
            _fail(f"ids and mask must have the same length, got {len(got_ids)} and {len(got_mask)}")
        if got_ids != want_ids:
            k = _first_difference(got_ids, want_ids)
            _fail(f"wrong token at position {k} for the conversation:\n  {conv['messages']}\n"
                  f"  yours:    {_tokens(got_ids[max(0, k - 3):k + 3])}\n  expected: {_tokens(want_ids[max(0, k - 3):k + 3])}")
        if got_mask != want_mask:
            k = _first_difference(got_mask, want_mask)
            _fail(f"wrong mask at position {k}, the token {tokenizer.decode([want_ids[k]])!r}: yours is {got_mask[k]}, expected {want_mask[k]}\n"
                  f"  (conversation: {conv['messages']})\n  hint: train (1) on what the assistant writes, including the python call and <|assistant_end|>,\n"
                  f"  but not (0) on the user, <|assistant_start|>, or the python output")
        n += 1
    _passed("render_conversation", n)

def render_for_completion(fn):
    """ fn(conversation) -> ids, the prompt that primes the assistant to answer """
    n = 0
    for conv in _conversations():
        want = tokenizer.render_for_completion(conv)
        got = list(_call(fn, conv))
        if got != want:
            _fail(f"for {conv['messages'][:-1]}\n  yours:    {_tokens(got)[-6:]} (last 6 tokens)\n  expected: {_tokens(want)[-6:]}")
        n += 1
    _passed("render_for_completion", n)

# -----------------------------------------------------------------------------
# Chapter 3: brain surgery

def expand_embedding(fn):
    """ fn(weight (V, C), new_vocab_size) -> new weight (new_vocab_size, C) """
    torch.manual_seed(0)
    n = 0
    for V, C, new in [(10, 4, 16), (50257, 8, 50304)]:
        w = torch.randn(V, C)
        got = _call(fn, w.clone(), new)
        if tuple(got.shape) != (new, C):
            _fail(f"expected shape {(new, C)}, got {tuple(got.shape)}")
        if not torch.allclose(got[:V], w):
            _fail("the old rows must be copied over unchanged")
        if not torch.allclose(got[V:], w.mean(dim=0).expand(new - V, C), atol=1e-6):
            _fail("every new row should be the average of the old rows")
        n += 3
    _passed("expand_embedding", n)

# -----------------------------------------------------------------------------
# Chapter 4: the data

def gsm8k_parts(fn):
    """ fn(answer) -> list of parts, the <<expr=result>> calculator annotations turned into tool calls """
    from tasks import GSM8K
    gsm = GSM8K("train", stop=200)
    n = 0
    for i in range(len(gsm)):
        answer = gsm.row(i)["answer"]
        want = gsm[i]["messages"][1]["content"]
        got = _call(fn, answer)
        if got != want:
            k = _first_difference(got, want)
            _fail(f"for the answer:\n{answer}\n\npart {k} is wrong:\n  yours:    {got[k] if k < len(got) else 'missing'}\n"
                  f"  expected: {want[k] if k < len(want) else 'nothing (you have too many parts)'}")
        n += 1
    _passed("gsm8k_parts", n)

def extract_answer(fn):
    """ fn(completion) -> the number after '####', commas removed, or None """
    cases = [("so the answer is #### 42", "42"), ("#### 1,024", "1024"), ("no idea", None), ("#### -3", "-3"),
             ("#### 2.5 apples", "2.5"), ("first #### 7 then #### 8", "7"), ("####42", None), ("", None)]
    for completion, want in cases:
        got = _call(fn, completion, allow_none=True)
        if got != want:
            _fail(f"extract_answer({completion!r}) returned {got!r}, expected {want!r}")
    _passed("extract_answer", len(cases))

def render_mc(fn):
    """ fn(question, letters, choices) -> the user message of a multiple choice question """
    from tasks import render_mc as reference
    cases = [("What is the capital of France?", "ABCD", ["Berlin", "Madrid", "Paris", "Rome"]),
             ("2+2=?", "AB", ["4", "5"]), ("Pick one", "ABCDE", ["a", "b", "c", "d", "e"])]
    for q, letters, choices in cases:
        got, want = _call(fn, q, letters, choices), reference(q, letters, choices)
        if got != want:
            _fail(f"yours:\n{got}\n\nexpected:\n{want}\n\n(hint: the letter goes after the choice, with no space before it)")
    _passed("render_mc", len(cases))

def spelling_answer(fn):
    """ fn(word, letter) -> the list of parts of the ideal SpellingBee answer """
    from tasks import SpellingBee
    bee = SpellingBee(size=50, split="train")
    n = 0
    for i in range(len(bee)):
        want = bee[i]["messages"][1]["content"]
        word, letter = want[1]["text"][1:].split("'.count('")
        letter = letter[:-2]
        got = _call(fn, word, letter)
        if got != want:
            k = _first_difference(got, want)
            _fail(f"for word={word!r}, letter={letter!r}, part {k} is wrong:\n  yours:    {got[k] if k < len(got) else 'missing'}\n"
                  f"  expected: {want[k] if k < len(want) else 'nothing'}")
        n += 1
    _passed("spelling_answer", n)

# -----------------------------------------------------------------------------
# Chapter 5: the SFT dataloader

def _walk_row(row, targets, rendered):
    """ split a packed row into whole conversations + padding. Returns (problem or None, number of padding tokens) """
    by_first = {}
    for ids, mask in rendered:
        by_first.setdefault(tuple(ids[:2]), []).append((ids, mask))
    p, expected_targets, padding = 0, [], 0
    while p < len(row):
        if len(set(row[p:])) == 1 and not any(ids == row[p:p + len(ids)] for ids, _ in rendered): # padding: one token, repeated
            expected_targets += [-1] * (len(row) - p)
            padding = len(row) - p
            break
        if row[p] != BOS:
            return f"at position {p} there's {tokenizer.decode([row[p]])!r}, but a conversation should start there, with BOS", 0
        match = next(((ids, mask) for ids, mask in by_first.get(tuple(row[p:p + 2]), []) if row[p:p + len(ids)] == ids), None)
        if match is None:
            return f"at position {p} starts something that isn't a complete conversation (was it cut off?): {tokenizer.decode(row[p:p + 30])!r}...", 0
        ids, mask = match
        expected_targets += [i if m else -1 for i, m in zip(ids, mask)]
        p += len(ids)
    want = expected_targets[1:] # the targets are shifted by one
    if targets != want[:len(targets)]:
        k = _first_difference(targets, want)
        return f"target {k} is {targets[k]}, expected {want[k]} (-1 wherever the mask is 0, and for the padding)", 0
    return None, padding

def sft_data_generator(fn):
    """ fn(dataset, batch_size, seq_len) -> a generator of (inputs, targets, progress) """
    from tasks import SpellingBee, GSM8K, TaskMixture
    dataset = TaskMixture([SpellingBee(size=300, split="train"), GSM8K("train", stop=300)])
    long_one = {"messages": [{"role": "user", "content": "word " * 600}, {"role": "assistant", "content": "ok"}]}
    dataset = [long_one] + [dataset[i] for i in range(len(dataset))] # one conversation that can never fit in a row
    rendered = [tokenizer.render_conversation(c, max_tokens=10**9) for c in dataset]
    B, T = 4, 512
    loader = _call(fn, dataset, B, T)
    n, padded, total = 0, 0, 0
    for _ in range(4):
        inputs, targets, progress = next(loader)
        if tuple(inputs.shape) != (B, T) or tuple(targets.shape) != (B, T):
            _fail(f"inputs and targets should both have shape {(B, T)}, got {tuple(inputs.shape)} and {tuple(targets.shape)}")
        for b in range(B):
            # the row is the inputs plus one last token, which we can see in the targets, unless it's padding
            last = targets[b, -1].item()
            row = inputs[b].tolist() + [last if last >= 0 else inputs[b, -1].item()]
            problem, padding = ("every row must start with BOS", 0) if row[0] != BOS else _walk_row(row, targets[b].tolist(), rendered)
            padded += padding
            if problem:
                _fail(f"row {b}: {problem}")
            n += 1
        total += targets.numel()
        if not 0.0 <= float(progress) <= 1.5:
            _fail(f"progress should be the fraction of the dataset we've gone through, got {progress}")
    if padded / total > 0.15:
        _fail(f"{padded / total:.0%} of every batch is padding: pack several conversations into a row, the longest one that still fits first")
    _passed("sft_data_generator", n)
    print(f"   (padding: {padded / total:.1%} of the tokens)")

def get_lr(fn):
    """ fn(progress) -> lr: constant 1e-4 for the first half, then linearly down to 0 """
    for p in [0.0, 0.1, 0.25, 0.5, 0.6, 0.75, 0.9, 1.0]:
        want = 1e-4 if p < 0.5 else 1e-4 * (1 - p) / 0.5
        got = _call(fn, p)
        if not math.isclose(got, want, rel_tol=1e-6, abs_tol=1e-12):
            _fail(f"get_lr({p}) = {got}, expected {want}")
    _passed("get_lr", 8)

# -----------------------------------------------------------------------------
# Chapter 7: the engine

def _tiny_gpt(seed=0):
    from gpt import GPT, GPTConfig
    torch.manual_seed(seed)
    return GPT(GPTConfig(block_size=128, vocab_size=50304, n_layer=2, n_head=2, n_embd=32)).eval()

def kv_cache(cache_class):
    """ cache_class(): an object with .k (None at first) and update(k, v) -> all keys and values so far """
    model = _tiny_gpt()
    prompt = [BOS, 464, 3616, 286, 1204, 318]
    with torch.no_grad():
        full = model(torch.tensor([prompt + [11, 290, 13]]))[0]
        cache = [cache_class() for _ in range(model.config.n_layer)]
        try:
            logits = [model(torch.tensor([prompt]), kv_cache=cache)[0]]
            for t in [11, 290, 13]:
                logits.append(model(torch.tensor([[t]]), kv_cache=cache)[0])
        except Exception as e:
            _fail(f"the model crashed when using your cache ({type(e).__name__}: {e}). update(k, v) must return all the keys and values so far")
    cached = torch.cat(logits)
    if cache[0].k is None or cache[0].k.size(2) != len(prompt) + 3:
        _fail(f"after a prompt of {len(prompt)} tokens and 3 more, the cache should hold {len(prompt) + 3} positions (dim 2 of k)")
    if not torch.allclose(full, cached, atol=1e-4):
        _fail("the logits with your cache differ from running the whole sequence at once")
    _passed("kv_cache", 2)

def generate_cached(fn):
    """ fn(model, prompt, max_tokens) -> list of tokens (prompt included), greedy, with a KV cache """
    import time
    model = _tiny_gpt(1)
    prompt = [BOS, 464, 3616, 286, 1204, 318]
    with torch.no_grad():
        ids = torch.tensor([prompt])
        for _ in range(40):
            ids = torch.cat([ids, model(ids)[:, -1].argmax(-1, keepdim=True)], dim=1)
    want = ids[0].tolist()
    got = _call(fn, model, prompt, 40)
    got = got[0] if isinstance(got, tuple) else got
    if list(got)[:len(want)] != want:
        k = _first_difference(list(got), want)
        _fail(f"your tokens differ from naive greedy sampling at position {k}")
    _passed("generate_cached", 1)

class _ScriptedModel:
    """ A fake model that 'wants' to write a fixed answer, by mapping the previous token to the next one """
    def __init__(self, script):
        from gpt import GPTConfig
        self.config = GPTConfig(n_layer=1)
        self.next = script
    def __call__(self, idx, kv_cache=None, **kwargs):
        if kv_cache is not None: # keep a fake cache so that code which inspects it doesn't break
            kv_cache[0].update(torch.zeros(1, 1, idx.size(1), 1), torch.zeros(1, 1, idx.size(1), 1))
        logits = torch.zeros(idx.size(0), idx.size(1), 50304)
        for b in range(idx.size(0)):
            for t in range(idx.size(1)):
                logits[b, t, self.next.get(idx[b, t].item(), 13)] = 10.0 # unknown token? say "."
        return logits
    def parameters(self):
        yield torch.zeros(1)

def generate_with_tools(fn):
    """ fn(model, prompt, max_tokens) -> the generated tokens (not the prompt), running the calculator """
    s = tokenizer.encode_special
    e = lambda text: tokenizer.encode(text)
    prompt = tokenizer.render_for_completion({"messages": [{"role": "user", "content": "What is 12 times 3?"}, {"role": "assistant", "content": ""}]})
    t12, times, t3, the, answer = e("12")[0], e("*")[0], e("3")[0], e(" The")[0], e(" answer")[0]
    script = {s("<|assistant_start|>"): s("<|python_start|>"), s("<|python_start|>"): t12, t12: times, times: t3, t3: s("<|python_end|>"),
              s("<|python_end|>"): e("99")[0], # the model would hallucinate the result: the tool's output must win
              s("<|output_end|>"): the, the: answer, answer: s("<|assistant_end|>")}
    want = [s("<|python_start|>"), t12, times, t3, s("<|python_end|>"), s("<|output_start|>"), *e("36"), s("<|output_end|>"),
            the, answer, s("<|assistant_end|>")]
    got = list(_call(fn, _ScriptedModel(script), prompt, 30))
    if got[:len(prompt)] == prompt and len(got) > len(want):
        got = got[len(prompt):]
    if got != want:
        _fail(f"the model wrote python_start 12*3 python_end, and then:\n  yours:    {_tokens(got)}\n  expected: {_tokens(want)}\n"
              f"  hint: after <|python_end|>, run the expression and force <|output_start|>, the result, <|output_end|> into the stream")
    _passed("generate_with_tools", 1)

def use_calculator(fn):
    """ fn(expression) -> the result, or None for anything that isn't simple arithmetic or str.count """
    ok = [("2*12", 24), ("(1+2)/4", 0.75), ("'strawberry'.count('r')", 3), ("1,000*2", 2000)]
    bad = ["__import__('os').system('ls')", "9**9**9", "open('/etc/passwd').read()", "().__class__", "exec('1')", "1/0"]
    for expr, want in ok:
        got = _call(fn, expr, allow_none=True)
        if got != want:
            _fail(f"use_calculator({expr!r}) = {got!r}, expected {want!r}")
    for expr in bad:
        got = _call(fn, expr, allow_none=True)
        if got is not None:
            _fail(f"use_calculator({expr!r}) should refuse and return None, but returned {got!r}")
    _passed("use_calculator", len(ok) + len(bad))

# -----------------------------------------------------------------------------
# Chapter 8: evaluation

def pass_at_k(fn):
    """ fn(outcomes) -> (pass@1, pass@k), outcomes is a list (questions) of lists (samples) of bools """
    cases = [([[True, False], [False, False]], (0.25, 0.5)), ([[True] * 4, [False] * 4], (0.5, 0.5)),
             ([[False, False, True, True]], (0.5, 1.0))]
    for outcomes, want in cases:
        got = tuple(float(x) for x in _call(fn, outcomes))
        if not all(math.isclose(a, b) for a, b in zip(got, want)):
            _fail(f"pass_at_k({outcomes}) = {got}, expected {want}")
    _passed("pass_at_k", len(cases))

# -----------------------------------------------------------------------------
# Chapter 9 and 10: reinforcement learning and preferences

def sequence_logprobs(fn):
    """ fn(model, ids (B, T), mask (B, T)) -> (B,) the summed log probability of the tokens where mask == 1 """
    model = _tiny_gpt(2)
    torch.manual_seed(0)
    ids = torch.randint(0, 50257, (3, 12))
    mask = torch.zeros(3, 12, dtype=torch.long)
    mask[0, 5:] = 1
    mask[1, 2:8] = 1
    mask[2, 11] = 1
    with torch.no_grad():
        logp = F.log_softmax(model(ids[:, :-1]), dim=-1).gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        want = (logp * mask[:, 1:]).sum(dim=1)
        got = _call(fn, model, ids, mask)
    if tuple(got.shape) != (3,):
        _fail(f"expected one number per sequence, shape (3,), got {tuple(got.shape)}")
    if not torch.allclose(got, want, atol=1e-4):
        _fail(f"yours: {got.tolist()}\nexpected: {want.tolist()}\nhint: the token at position t is predicted by the logits at position t-1")
    _passed("sequence_logprobs", 1)

def pg_loss(fn):
    """ fn(logp (B, T), targets (B, T), rewards (B,)) -> the loss of nanochat's simplified GRPO """
    torch.manual_seed(0)
    n = 0
    for B, T in [(8, 20), (4, 5)]:
        logp = -torch.rand(B, T)
        targets = torch.randint(0, 100, (B, T))
        targets[:, :3] = -1
        targets[0, -2:] = -1
        logp = logp * (targets >= 0)
        rewards = (torch.rand(B) > 0.5).float()
        want = -(logp * (rewards - rewards.mean()).unsqueeze(-1)).sum() / (targets >= 0).sum()
        got = _call(fn, logp, targets, rewards)
        if not torch.allclose(torch.as_tensor(got, dtype=torch.float), want, atol=1e-5):
            _fail(f"got {float(got):.5f}, expected {want.item():.5f}\nhint: advantages = rewards - mean, weight every token's logp by its "
                  f"sample's advantage, sum, divide by the number of trained tokens, and negate (we minimize)")
        n += 1
    _passed("pg_loss", n)

def dpo_loss(fn):
    """ fn(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta) -> scalar loss (all (B,) sequence logprobs) """
    torch.manual_seed(0)
    n = 0
    for beta in [0.1, 0.5]:
        pc, pr, rc, rr = (-10 * torch.rand(16) for _ in range(4))
        want = -F.logsigmoid(beta * ((pc - rc) - (pr - rr))).mean()
        got = _call(fn, pc, pr, rc, rr, beta)
        if not torch.allclose(torch.as_tensor(got, dtype=torch.float), want, atol=1e-5):
            _fail(f"beta={beta}: got {float(got):.5f}, expected {want.item():.5f}\n"
                  f"hint: the margin is beta * ((chosen - ref_chosen) - (rejected - ref_rejected)), and the loss is -logsigmoid(margin)")
        n += 1
    pc = torch.zeros(4, requires_grad=True)
    _call(fn, pc, torch.zeros(4), torch.zeros(4), torch.zeros(4), 0.1).backward()
    if pc.grad is None or not (pc.grad < 0).all():
        _fail("the loss should decrease when the chosen answers become more likely")
    _passed("dpo_loss", n + 1)
