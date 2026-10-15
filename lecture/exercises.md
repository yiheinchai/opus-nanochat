# Exercises: build nanochat's post-training yourself

<!--
Source of exercises.ipynb (python lecture/build.py exercises). Every exercise is a heading, some text,
then three fenced blocks: ```stub (what the student starts from), ```check (one line, runs checks.py)
and ```solution (hidden in the notebook until you click). `build.py exercises --verify` runs every
solution through its check.
-->

These are the pieces of the lecture, as exercises. Each one gives you a function to write, and a check that runs your function on a bunch of cases and compares it to the reference in this repo. When something is wrong, the check tells you the first case that failed and what was expected.

Try each one before you open the solution. If you get stuck, rewatch that part of the lecture (the chapter is named in each exercise), then try again. Getting them all to pass means you can rebuild every piece of post-training from scratch.

```setup
# On Colab, uncomment this to get the code:
# !git clone https://github.com/yiheinchai/opus-nanochat && %cd opus-nanochat && pip install -q tiktoken safetensors pyarrow
import re, math, random
from collections import deque
import torch
import torch.nn.functional as F
import checks
from tokenizer import Tokenizer
tokenizer = Tokenizer()
enc = tokenizer.enc # the GPT-2 tokenizer, plus our special tokens, see chapter 2
s = enc.encode_single_token # s("<|user_start|>") -> 50257
```

## 1. Render a conversation into tokens and a mask

Chapter 2. Turn a conversation into one list of token ids, plus a mask that says which tokens the model is trained on. The format: BOS (`<|endoftext|>`, id 50256), then for each message the start token, its content (with `enc.encode_ordinary`, so a user can't type a special token), and the end token. An assistant message's content is either a string, or a list of parts of type `text`, `python` or `python_output`. Wrap python in `<|python_start|>` `<|python_end|>` and outputs in `<|output_start|>` `<|output_end|>`. And if the first message is a system message, glue it onto the first user message with two newlines in between.

The mask is the part to think hard about. It's 1 exactly for the tokens the model has to produce at test time.

```stub
def render_conversation(conversation):
    ids, mask = [], []
    # YOUR CODE HERE
    return ids, mask
```

```check
checks.render_conversation(render_conversation)
```

```solution
def render_conversation(conversation):
    ids, mask = [], []
    def add_tokens(token_ids, mask_val):
        ids.extend(token_ids)
        mask.extend([mask_val] * len(token_ids))
    messages = conversation["messages"]
    if messages[0]["role"] == "system": # no special tokens for system messages: glue it onto the first user message
        messages = [{"role": "user", "content": messages[0]["content"] + "\n\n" + messages[1]["content"]}] + messages[2:]
    add_tokens([tokenizer.bos], 0)
    for message in messages:
        content = message["content"]
        if message["role"] == "user":
            add_tokens([s("<|user_start|>"), *enc.encode_ordinary(content), s("<|user_end|>")], 0)
            continue
        add_tokens([s("<|assistant_start|>")], 0) # we put this one there ourselves
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        for part in content:
            value = enc.encode_ordinary(part["text"])
            if part["type"] == "text":
                add_tokens(value, 1)
            elif part["type"] == "python": # the model writes the code: train on it
                add_tokens([s("<|python_start|>"), *value, s("<|python_end|>")], 1)
            elif part["type"] == "python_output": # python writes the output: don't
                add_tokens([s("<|output_start|>"), *value, s("<|output_end|>")], 0)
        add_tokens([s("<|assistant_end|>")], 1) # learn to stop!
    return ids, mask
```

## 2. The prompt at test time

Chapter 2. At test time it's the assistant's turn. Render everything except the last message (the assistant's, which we don't have yet), and then prime the assistant to answer. You can use your `render_conversation`.

```stub
def render_for_completion(conversation):
    # YOUR CODE HERE
    pass
```

```check
checks.render_for_completion(render_for_completion)
```

```solution
def render_for_completion(conversation):
    ids, _ = render_conversation({"messages": conversation["messages"][:-1]})
    return ids + [s("<|assistant_start|>")]
```

## 3. Brain surgery: rows for the new tokens

Chapter 3. GPT-2's embedding table has 50,257 rows, and our special tokens need rows 50,257 to 50,264. Write the function that makes a bigger table: copy the old rows, and set every new row to the average of the old ones. (Remember that with weight tying, this one table is also the output layer.)

```stub
def expand_embedding(weight, new_vocab_size):
    # weight: (V, C). Returns a (new_vocab_size, C) tensor
    # YOUR CODE HERE
    pass
```

```check
checks.expand_embedding(expand_embedding)
```

```solution
def expand_embedding(weight, new_vocab_size):
    V, C = weight.shape
    new = torch.empty(new_vocab_size, C)
    new[:V] = weight
    new[V:] = weight.mean(dim=0) # the average lives where real embeddings live
    return new
```

## 4. GSM8K: calculator annotations become tool calls

Chapter 4. A GSM8K answer looks like `Natalia sold 48/2 = <<48/2=24>>24 clips in May.` and ends with `#### 72`. Turn it into a list of parts: every `<<expr=result>>` becomes a `python` part with the expression, followed by a `python_output` part with the result, and everything in between is a `text` part (keep the empty strings that `re.split` gives you, the reference does too). Hint: a regex split with a capture group, and `rpartition`.

```stub
def gsm8k_parts(answer):
    parts = []
    # YOUR CODE HERE
    return parts
```

```check
checks.gsm8k_parts(gsm8k_parts)
```

```solution
def gsm8k_parts(answer):
    parts = []
    for chunk in re.split(r'(<<[^>]+>>)', answer): # the capture group keeps the annotations
        if chunk.startswith('<<') and chunk.endswith('>>'):
            expr, _, result = chunk[2:-2].rpartition('=')
            parts.append({"type": "python", "text": expr})
            parts.append({"type": "python_output", "text": result})
        else:
            parts.append({"type": "text", "text": chunk})
    return parts
```

## 5. The grader

Chapter 4. Pull out the final answer: the number right after `#### ` (with a space). Remove commas, so `1,024` becomes `"1024"`. Return a string, or None if there is no answer. This one function is our GSM8K eval, our SpellingBee eval, and later our RL reward.

```stub
def extract_answer(completion):
    # YOUR CODE HERE
    pass
```

```check
checks.extract_answer(extract_answer)
```

```solution
ANSWER_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
def extract_answer(completion):
    match = ANSWER_RE.search(completion)
    return match.group(1).strip().replace(",", "") if match else None
```

## 6. Multiple choice

Chapter 4. Render a multiple choice question as a user message, exactly like nanochat: a first line `Multiple Choice question: <question>`, then one line per choice like `- Paris=C`, then an empty line and `Respond only with the letter of the correct answer.` Why is the letter after the choice, and why is there no space before it?

```stub
def render_mc(question, letters, choices):
    # YOUR CODE HERE
    pass
```

```check
checks.render_mc(render_mc)
```

```solution
def render_mc(question, letters, choices):
    query = f"Multiple Choice question: {question}\n"
    query += "".join(f"- {choice}={letter}\n" for letter, choice in zip(letters, choices)) # "=A": the same token 32 as the answer
    query += "\nRespond only with the letter of the correct answer."
    return query
```

## 7. SpellingBee: design the ideal answer

Chapter 4. Write the ideal answer to "how many `letter` are in `word`", as a list of parts, spreading the work over many tokens so that no single token has to do much. The check compares against nanochat's exact wording, so here it is. A text part:

```
We are asked to find the number 'r' in the word 'strawberry'. Let me try a manual approach first.

First spell the word out:
strawberry:s,t,r,a,w,b,e,r,r,y

Then count the occurrences of 'r':
1:s
2:t
3:r hit! count=1
...
10:y

This gives us 3.

Let me double check this using Python:

```

then a python part `'strawberry'.count('r')`, its output `3`, and a last text part `\n\nPython gives us 3.\n\nMy final answer is:\n\n#### 3`.

```stub
def spelling_answer(word, letter):
    # YOUR CODE HERE
    pass
```

```check
checks.spelling_answer(spelling_answer)
```

```solution
def spelling_answer(word, letter):
    text = f"We are asked to find the number '{letter}' in the word '{word}'. Let me try a manual approach first.\n\n"
    text += f"First spell the word out:\n{word}:{','.join(word)}\n\nThen count the occurrences of '{letter}':\n"
    count = 0
    for i, char in enumerate(word, 1):
        if char == letter:
            count += 1
            text += f"{i}:{char} hit! count={count}\n"
        else:
            text += f"{i}:{char}\n"
    text += f"\nThis gives us {count}.\n\nLet me double check this using Python:\n\n"
    return [
        {"type": "text", "text": text},
        {"type": "python", "text": f"'{word}'.count('{letter}')"},
        {"type": "python_output", "text": str(word.count(letter))},
        {"type": "text", "text": f"\n\nPython gives us {word.count(letter)}.\n\nMy final answer is:\n\n#### {word.count(letter)}"},
    ]
```

## 8. The best-fit packing dataloader

Chapter 5. This is the big one. Write a generator that yields `(inputs, targets, progress)` forever, where inputs and targets are `(batch_size, seq_len)` long tensors. The rules:

- every row is `seq_len + 1` tokens of whole conversations (use `tokenizer.render_conversation(conversation, max_tokens=...)`), and inputs and targets are that row shifted by one, like in pretraining
- never cut a conversation: skip the ones that can't fit in a row at all
- pack: keep a buffer of about 100 conversations, and fill each row by repeatedly taking the longest one that still fits. Pad the rest of the row with BOS tokens
- targets are -1 wherever the mask is 0, and on the padding (-1 is ignored by the loss)
- progress is how far through the dataset we are (the index of the next conversation, divided by the length)

The check walks through your rows and makes sure each one is whole conversations followed by padding, with the right targets. It also checks that less than 15% of every batch is padding, which naive one-conversation-per-row padding doesn't come close to.

```stub
def sft_data_generator(dataset, batch_size, seq_len, buffer_size=100):
    # YOUR CODE HERE
    pass
```

```check
checks.sft_data_generator(sft_data_generator)
```

```solution
def sft_data_generator(dataset, batch_size, seq_len, buffer_size=100):
    row_capacity = seq_len + 1 # +1 because the targets are the inputs shifted by one
    buffer, cursor = [], 0
    while True:
        rows, mask_rows = [], []
        for _ in range(batch_size):
            row, mask_row = [], []
            while True:
                while len(buffer) < buffer_size:
                    ids, mask = tokenizer.render_conversation(dataset[cursor % len(dataset)], max_tokens=row_capacity + 1)
                    if len(ids) <= row_capacity: # skip the ones that can never fit (cropping them can leave nothing to train on)
                        buffer.append((ids, mask))
                    cursor += 1
                remaining = row_capacity - len(row)
                fits = [i for i, (ids, _) in enumerate(buffer) if len(ids) <= remaining]
                if not fits:
                    break
                ids, mask = buffer.pop(max(fits, key=lambda i: len(buffer[i][0]))) # best fit: the longest that fits
                row += ids
                mask_row += mask
            padding = row_capacity - len(row)
            rows.append(row + [tokenizer.bos] * padding)
            mask_rows.append(mask_row + [0] * padding)
        batch, mask = torch.tensor(rows), torch.tensor(mask_rows)
        inputs, targets = batch[:, :-1], batch[:, 1:].clone()
        targets[mask[:, 1:] == 0] = -1
        yield inputs, targets, cursor / len(dataset)
```

## 9. The learning rate schedule

Chapter 6. nanochat's SFT schedule: constant at the peak learning rate of 1e-4 for the first half of training, then linearly down to zero at the end. `progress` goes from 0 to 1.

```stub
def get_lr(progress, lr=1e-4, warmdown_ratio=0.5):
    # YOUR CODE HERE
    pass
```

```check
checks.get_lr(get_lr)
```

```solution
def get_lr(progress, lr=1e-4, warmdown_ratio=0.5):
    if progress < 1 - warmdown_ratio:
        return lr
    return lr * (1 - progress) / warmdown_ratio
```

## 10. A KV cache

Chapter 7. Write the cache for one layer. It starts out empty (`self.k = self.v = None`), and `update(k, v)` appends the new keys and values (shape `(B, n_head, T, head_dim)`) along the time dimension and returns all the keys and values so far. The check plugs your class into our GPT (which already has the other two changes from the lecture: the attention uses `kv_cache.update`, and the positions start at `kv_cache[0].k.size(2)`) and compares against running the whole sequence at once.

```stub
class KVCache:
    def __init__(self):
        self.k = self.v = None
    def update(self, k, v):
        # YOUR CODE HERE
        pass
```

```check
checks.kv_cache(KVCache)
```

```solution
class KVCache:
    def __init__(self):
        self.k = self.v = None
    def update(self, k, v):
        if self.k is not None:
            k = torch.cat([self.k, k], dim=2) # dim 2 is time
            v = torch.cat([self.v, v], dim=2)
        self.k, self.v = k, v
        return k, v
```

## 11. Sampling with the cache

Chapter 7. Greedy sampling with your cache: one forward pass over the whole prompt (the prefill), then one token at a time (decode). Return the prompt plus the new tokens. The check compares with naive sampling, which recomputes everything at every step: the tokens must be identical.

```stub
@torch.no_grad()
def generate_cached(model, prompt, max_tokens):
    kv_cache = [KVCache() for _ in range(model.config.n_layer)]
    # YOUR CODE HERE
    pass
```

```check
checks.generate_cached(generate_cached)
```

```solution
@torch.no_grad()
def generate_cached(model, prompt, max_tokens):
    kv_cache = [KVCache() for _ in range(model.config.n_layer)]
    logits = model(torch.tensor([prompt]), kv_cache=kv_cache)[:, -1, :] # prefill: the whole prompt, once
    out = list(prompt)
    for _ in range(max_tokens):
        token = logits.argmax(dim=-1).item()
        out.append(token)
        logits = model(torch.tensor([[token]]), kv_cache=kv_cache)[:, -1, :] # decode: one new token
    return out
```

## 12. A safe calculator

Chapter 7. We're going to run code that a neural network wrote, so the tool has to be paranoid. Allow arithmetic (digits, `+-*/.()` and spaces, commas get removed, but no `**`) and `.count(` on a string (letters, digits, quotes, parentheses, dots, underscores and spaces, but no `__`). Evaluate with no builtins, and return None for anything else, including errors.

```stub
def use_calculator(expr):
    # YOUR CODE HERE
    pass
```

```check
checks.use_calculator(use_calculator)
```

```solution
def use_calculator(expr):
    expr = expr.replace(",", "")
    is_math = all(c in "0123456789*+-/.() " for c in expr) and "**" not in expr # no 9**9**9 please
    is_count = all(c.isalnum() or c in "'\"()._ " for c in expr) and ".count(" in expr and "__" not in expr
    if not (is_math or is_count):
        return None
    try:
        return eval(expr, {"__builtins__": {}}, {})
    except Exception:
        return None
```

## 13. Tool use

Chapter 7. Greedy sampling (with a KV cache) that runs the tool. When the model emits `<|python_start|>`, start recording tokens. When it emits `<|python_end|>`, decode the recorded tokens, run them with your calculator, and force `<|output_start|>`, the tokens of the result, and `<|output_end|>` into the stream: on the next steps, use those instead of what the model samples. Stop after `<|assistant_end|>`. Return only the new tokens (`<|assistant_end|>` included). The check uses a fake model that writes `12*3`, and that would hallucinate a wrong result if you let it.

```stub
@torch.no_grad()
def generate_with_tools(model, prompt, max_tokens=256):
    kv_cache = [KVCache() for _ in range(model.config.n_layer)]
    logits = model(torch.tensor([prompt]), kv_cache=kv_cache)[:, -1, :]
    out, forced, expr = [], deque(), None
    # YOUR CODE HERE
    return out
```

```check
checks.generate_with_tools(generate_with_tools)
```

```solution
@torch.no_grad()
def generate_with_tools(model, prompt, max_tokens=256):
    kv_cache = [KVCache() for _ in range(model.config.n_layer)]
    logits = model(torch.tensor([prompt]), kv_cache=kv_cache)[:, -1, :]
    out, forced, expr = [], deque(), None
    for _ in range(max_tokens):
        token = forced.popleft() if forced else logits.argmax(dim=-1).item()
        out.append(token)
        if token == s("<|assistant_end|>"):
            break
        if token == s("<|python_start|>"):
            expr = []
        elif token == s("<|python_end|>") and expr is not None:
            result = use_calculator(tokenizer.decode(expr))
            if result is not None:
                forced.extend([s("<|output_start|>"), *tokenizer.encode(str(result)), s("<|output_end|>")])
            expr = None
        elif expr is not None:
            expr.append(token)
        logits = model(torch.tensor([[token]]), kv_cache=kv_cache)[:, -1, :]
    return out
```

## 14. pass@1 and pass@k

Chapter 9. Given the outcomes of k samples for each of several questions (a list of lists of booleans), return pass@1, the fraction of all samples that are correct (averaged per question), and pass@k, the fraction of questions where at least one sample is correct.

```stub
def pass_at_k(outcomes):
    # YOUR CODE HERE
    pass
```

```check
checks.pass_at_k(pass_at_k)
```

```solution
def pass_at_k(outcomes):
    pass1 = sum(sum(o) / len(o) for o in outcomes) / len(outcomes)
    passk = sum(any(o) for o in outcomes) / len(outcomes)
    return pass1, passk
```

## 15. The log probability of an answer

Chapters 9 and 10. Both RL and DPO need the log probability that the model gives to a sampled answer: the sum of the log probabilities of its tokens. Given `ids` of shape `(B, T)` and a mask of the same shape (1 for the answer's tokens), return a `(B,)` tensor. Careful with the off by one: the token at position t is predicted by the logits at position t - 1. Hint: our model returns per-token losses with `model(inputs, targets, loss_reduction='none')`, and ignores targets of -1.

```stub
def sequence_logprobs(model, ids, mask):
    # YOUR CODE HERE
    pass
```

```check
checks.sequence_logprobs(sequence_logprobs)
```

```solution
def sequence_logprobs(model, ids, mask):
    inputs, targets = ids[:, :-1], ids[:, 1:].clone()
    targets[mask[:, 1:] == 0] = -1
    logp = -model(inputs, targets, loss_reduction='none').view_as(inputs) # 0 where the target is -1
    return logp.sum(dim=1)
```

## 16. The policy gradient loss

Chapter 9. nanochat's simplified GRPO, for one group of samples of the same question. `logp` is `(B, T)`: the log probability of every token (already 0 where the target is -1), `targets` is `(B, T)` with -1 for tokens we don't train on, and `rewards` is `(B,)`. The advantage is the reward minus the mean reward of the group. Weight every token's log probability by its sample's advantage, sum, divide by the number of trained tokens, and return the negative (PyTorch minimizes).

```stub
def pg_loss(logp, targets, rewards):
    # YOUR CODE HERE
    pass
```

```check
checks.pg_loss(pg_loss)
```

```solution
def pg_loss(logp, targets, rewards):
    advantages = rewards - rewards.mean()
    return -(logp * advantages.unsqueeze(-1)).sum() / (targets >= 0).sum()
```

## 17. The DPO loss

Chapter 10. All four inputs are `(B,)` tensors of sequence log probabilities: of the chosen and the rejected answer, under the model we're training (the policy) and under the frozen reference model. The implicit reward of an answer is `beta * (log pi(y) - log pi_ref(y))`. Return the mean, over the pairs, of minus the log sigmoid of (reward of chosen minus reward of rejected).

```stub
def dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1):
    # YOUR CODE HERE
    pass
```

```check
checks.dpo_loss(dpo_loss)
```

```solution
def dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1):
    margin = beta * ((policy_chosen - ref_chosen) - (policy_rejected - ref_rejected))
    return -F.logsigmoid(margin).mean()
```

## 18. Capstone: teach a model a new skill, end to end

No check for this one, it's the real thing. Pick a skill the model doesn't have, and post-train it in, with every stage of the lecture:

1. **Data.** Write a synthetic task in the style of SpellingBee (subclass `Task` from tasks.py). Ideas: reverse a word letter by letter, add two big numbers with the calculator, answer "what's your name?" with a name of your choice. Design the ideal answer so that the work is spread over many tokens, and end it with `#### answer` so the same grader works.
2. **SFT.** Add your task to the mixture in chat_sft.py and train (the CPU command at the top of the file works). Check that the loss on your task goes down, and talk to the model with chat_cli.py.
3. **Eval.** Measure greedy accuracy, and pass@1 and pass@8 at temperature 1, on held out examples.
4. **RL.** Add your task to chat_rl.py (`--task`) and run it. Does pass@1 move towards pass@8?
5. **Preferences.** Find something the model does that you don't like, collect pairs of its own answers (one you like, one you don't), and run chat_dpo.py on them.

Write down every number. If something doesn't work, the debugging checklist at the end of the lecture is the place to start.
