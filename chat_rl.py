"""
Reinforcement learning, "GRPO" style. Mirrors nanochat/scripts/chat_rl.py.

I put "GRPO" in quotes because, just like in nanochat, what we end up with is a lot simpler
and closer to plain old REINFORCE:
1) no KL regularization to a reference model (no trust region)
2) we are fully on-policy (one gradient step per batch of samples), so no PPO ratio + clip
3) token-level normalization of the loss (DAPO style), not sequence level
4) the advantage is just (reward - mean reward of the group), no division by the std

GPU:
python chat_rl.py --task=SpellingBee
CPU (a quick demo):
python chat_rl.py --task=SpellingBee --num-steps=40 --examples-per-step=8 --num-samples=8 --eval-examples=48 --lr=3e-5
"""

import os
import time
import argparse
import itertools
import torch

from common import get_base_dir, autodetect_device
from gpt import load_model
from tokenizer import Tokenizer
from engine import Engine
from tasks import GSM8K, SpellingBee

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="SpellingBee", help="SpellingBee|GSM8K")
parser.add_argument("--num-steps", type=int, default=200, help="optimizer steps")
parser.add_argument("--examples-per-step", type=int, default=16, help="questions per optimizer step")
parser.add_argument("--num-samples", type=int, default=16, help="samples (attempts) per question, i.e. the 'group'")
parser.add_argument("--max-new-tokens", type=int, default=256, help="max tokens per sample")
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--top-k", type=int, default=50)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--eval-every", type=int, default=10, help="evaluate every N steps")
parser.add_argument("--eval-examples", type=int, default=128, help="held-out questions to evaluate on")
args = parser.parse_args()

device = autodetect_device()
autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda")
tokenizer = Tokenizer()
model = load_model("sft", device)
engine = Engine(model, tokenizer)

if args.task == "GSM8K":
    train_task, val_task = GSM8K("train"), GSM8K("test")
else:
    train_task, val_task = SpellingBee(size=100_000, split="train"), SpellingBee(size=args.eval_examples, split="test")

# -----------------------------------------------------------------------------
# Rollouts: for one question, sample a whole group of answers and score each one

@torch.no_grad()
def get_batch():
    assistant_end = tokenizer.encode_special("<|assistant_end|>") # used for padding only, never trained on
    for step, example_idx in enumerate(itertools.cycle(range(len(train_task)))):
        conversation = train_task[example_idx]
        prompt = tokenizer.render_for_completion(conversation)
        with autocast:
            sequences, masks = engine.generate_batch(prompt, num_samples=args.num_samples, max_tokens=args.max_new_tokens,
                                                     temperature=args.temperature, top_k=args.top_k, seed=step)
        rewards = [train_task.reward(conversation, tokenizer.decode(seq[len(prompt):])) for seq in sequences]
        # pad the samples to the same length, then build inputs/targets just like in SFT
        max_length = max(len(seq) for seq in sequences)
        ids = torch.tensor([seq + [assistant_end] * (max_length - len(seq)) for seq in sequences], device=device)
        mask = torch.tensor([m + [0] * (max_length - len(m)) for m in masks], device=device)
        inputs, targets = ids[:, :-1], ids[:, 1:].clone()
        targets[mask[:, 1:] == 0] = -1 # don't train on the prompt, the tool outputs, or the padding
        rewards = torch.tensor(rewards, device=device)
        advantages = rewards - rewards.mean() # better than average => push up, worse => push down
        yield inputs, targets, rewards, advantages

def evaluate(num_samples=8):
    """ pass@1: fraction of samples that are correct. pass@k: fraction of questions with >= 1 correct sample """
    pass1, passk = 0, 0
    for i in range(len(val_task)):
        conversation = val_task[i]
        prompt = tokenizer.render_for_completion(conversation)
        with autocast:
            sequences, _ = engine.generate_batch(prompt, num_samples=num_samples, max_tokens=args.max_new_tokens,
                                                 temperature=1.0, top_k=args.top_k, seed=i)
        outcomes = [val_task.evaluate(conversation, tokenizer.decode(seq[len(prompt):])) for seq in sequences]
        pass1 += sum(outcomes) / num_samples
        passk += any(outcomes)
    return pass1 / len(val_task), passk / len(val_task)

# -----------------------------------------------------------------------------
# The training loop
optimizer = model.configure_optimizer(lr=args.lr, weight_decay=0.0)
batch_iterator = get_batch()
for step in range(args.num_steps + 1):

    if step % args.eval_every == 0 or step == args.num_steps:
        pass1, passk = evaluate()
        print(f"step {step:4d} | eval pass@1 {pass1:.4f} | pass@8 {passk:.4f}")
    if step == args.num_steps:
        break

    t0 = time.time()
    step_rewards, step_lengths = [], []
    for _ in range(args.examples_per_step):
        inputs, targets, rewards, advantages = next(batch_iterator)
        with autocast:
            logp = -model(inputs, targets, loss_reduction='none').view_as(inputs) # (B, T), 0 where target is -1
        # the policy gradient objective: the log probability of each sampled token, weighted by its advantage
        pg_obj = (logp * advantages.unsqueeze(-1)).sum()
        num_valid = (targets >= 0).sum().clamp(min=1)
        loss = -pg_obj / (num_valid * args.examples_per_step) # minimize the negative of what we want to maximize
        loss.backward()
        step_rewards.append(rewards.mean().item())
        step_lengths.append((targets >= 0).sum(dim=1).float().mean().item())
    lr = args.lr * (1 - step / args.num_steps) # linear decay to zero
    for group in optimizer.param_groups:
        group["lr"] = lr
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    mean_reward = sum(step_rewards) / len(step_rewards)
    mean_length = sum(step_lengths) / len(step_lengths)
    print(f"step {step:4d} | reward {mean_reward:.4f} | sampled tokens per answer {mean_length:.1f} | lr {lr:.2e} | dt {time.time() - t0:.1f}s")

path = os.path.join(get_base_dir(), "chatrl.pt")
torch.save(model.state_dict(), path)
print(f"Saved the RL model to {path}")
