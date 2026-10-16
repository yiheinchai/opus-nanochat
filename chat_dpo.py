"""
Direct Preference Optimization (DPO). This one is not in nanochat: it's the "what's next" of the
lecture, built from the same pieces (the Engine, render_for_completion, the per-token loss of our GPT).

RLHF wants the policy that maximizes a reward, without drifting far from a reference model:
    max  E[r(x, y)] - beta * KL(pi || pi_ref)
That has a closed-form solution, pi*(y|x) = pi_ref(y|x) * exp(r(x, y) / beta) / Z(x). Solve it for the
reward, r = beta * log(pi* / pi_ref) + beta * log Z(x), plug that into the Bradley-Terry model of
preferences, P(chosen beats rejected) = sigmoid(r(chosen) - r(rejected)), and Z(x) cancels. So we can fit
the policy straight to pairs of answers: no reward model, and no sampling during training.
    loss = -log sigmoid(beta * ((log pi(c) - log pi_ref(c)) - (log pi(r) - log pi_ref(r))))

Our preferences: the RL model sometimes runs the SpellingBee procedure ("We are asked to find...") on
questions that have nothing to do with spelling. We sample its own answers and prefer
1) on ordinary questions: an answer that doesn't do that, over one that does
2) on spelling questions: a correct answer over a wrong one (so we keep what RL taught it)

Plain DPO (--sft-weight=0 --beta=0.1) gets rid of the habit, but SpellingBee drops from 86% to 47%: pushing
the rejected answers down also pushes down the same procedure where it belongs, and DPO only looks at the
difference between the two answers of a pair. So we add an SFT loss on the chosen answers (like Llama 3),
and a bigger beta (a shorter leash): the defaults below. That keeps 78%.

CPU (a quick demo, about 2 hours on 4 cores, most of it collecting the pairs):
python chat_dpo.py --general-prompts=600 --spelling-prompts=250
"""

import os
import re
import json
import time
import random
import argparse
import torch
import torch.nn.functional as F

from common import get_base_dir, autodetect_device
from gpt import load_model
from tokenizer import Tokenizer
from engine import Engine
from tasks import SmolTalk, SpellingBee

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=str, default="rl", help="the model to start from (and the reference): sft|rl")
parser.add_argument("--general-prompts", type=int, default=2000, help="ordinary questions to sample answers for")
parser.add_argument("--spelling-prompts", type=int, default=800, help="spelling questions to sample answers for")
parser.add_argument("--num-samples", type=int, default=8, help="answers per question, we pick one pair out of them")
parser.add_argument("--max-new-tokens", type=int, default=256, help="max tokens per answer to a spelling question")
parser.add_argument("--max-new-tokens-general", type=int, default=128, help="max tokens per answer to an ordinary question")
parser.add_argument("--beta", type=float, default=0.5, help="how strongly to stay close to the reference model")
parser.add_argument("--sft-weight", type=float, default=1.0, help="weight of an SFT loss on the chosen answers (0 = plain DPO)")
parser.add_argument("--eval-before", type=int, default=1, help="evaluate the model before DPO too")
parser.add_argument("--lr", type=float, default=5e-6)
parser.add_argument("--batch-size", type=int, default=8, help="pairs per optimizer step")
parser.add_argument("--epochs", type=int, default=2)
parser.add_argument("--eval-questions", type=int, default=80, help="held-out ordinary questions (x4 samples) to measure the leak")
parser.add_argument("--eval-spelling", type=int, default=100, help="held-out spelling questions, greedy")
args = parser.parse_args()

device = autodetect_device()
tokenizer = Tokenizer()
assistant_end = tokenizer.encode_special("<|assistant_end|>")
LEAK = re.compile(r"We are asked to find|Let me try a manual approach")
ABOUT_SPELLING = re.compile(r"letter|spell|count|how many", re.IGNORECASE)

def ordinary_questions(split, n, start=0):
    """ short questions from SmolTalk that have nothing to do with spelling """
    task, questions = SmolTalk(split), []
    for i in range(start, len(task)):
        messages = task[i]["messages"]
        question = messages[0]["content"] if messages[0]["role"] == "user" else messages[1]["content"]
        if len(question) < 160 and not ABOUT_SPELLING.search(question):
            questions.append({"messages": [{"role": "user", "content": question}, {"role": "assistant", "content": ""}]})
        if len(questions) == n:
            break
    return questions

def sample(engine, conversation, num_samples, seed, max_tokens, temperature=1.0):
    """ answers as token sequences ready for training: prompt + answer (+ assistant_end if it finished), and a mask """
    prompt = tokenizer.render_for_completion(conversation)
    sequences, masks = engine.generate_batch(prompt, num_samples=num_samples, max_tokens=max_tokens,
                                             temperature=temperature, top_k=50, seed=seed)
    answers = []
    for seq, mask in zip(sequences, masks):
        finished = len(seq) - len(prompt) < max_tokens
        ids = seq + [assistant_end] * finished # we want it to learn when to stop, like in SFT
        mask = mask + [1] * finished
        answers.append({"ids": ids, "mask": mask, "text": tokenizer.decode(seq[len(prompt):])})
    return answers

# -----------------------------------------------------------------------------
# 1) Collect preference pairs from the model's own answers

@torch.no_grad()
def collect_pairs(engine, progress_path):
    """ every question's outcome is appended to progress_path as we go, so that an interrupted run picks up where it left off """
    jobs = [("general", q) for q in ordinary_questions("train", args.general_prompts, start=100_000)]
    bee = SpellingBee(size=100_000, split="train")
    jobs += [("spelling", bee[50_000 + i]) for i in range(args.spelling_prompts)] # questions RL didn't practice on
    random.Random(0).shuffle(jobs)
    done = [json.loads(line) for line in open(progress_path)] if os.path.exists(progress_path) else []
    pairs = [d["pair"] for d in done if d["pair"]]
    t0 = time.time()
    with open(progress_path, "a") as progress:
        for k in range(len(done), len(jobs)):
            kind, conversation = jobs[k]
            max_tokens = args.max_new_tokens_general if kind == "general" else args.max_new_tokens
            answers = sample(engine, conversation, args.num_samples, seed=k, max_tokens=max_tokens)
            if kind == "general": # the judge: don't run the spelling procedure when nobody asked for it
                good = [a for a in answers if not LEAK.search(a["text"])]
                bad = [a for a in answers if LEAK.search(a["text"])]
            else: # the judge: the right answer
                good = [a for a in answers if bee.evaluate(conversation, a["text"])]
                bad = [a for a in answers if not bee.evaluate(conversation, a["text"])]
            rng = random.Random(k)
            pair = {"kind": kind, "chosen": rng.choice(good), "rejected": rng.choice(bad)} if good and bad else None
            if pair:
                pairs.append(pair)
            progress.write(json.dumps({"k": k, "pair": pair}) + "\n")
            progress.flush()
            if k % 20 == 0:
                print(f"question {k:4d}/{len(jobs)} | pairs {len(pairs)} | {time.time() - t0:.0f}s", flush=True)
    return pairs

# -----------------------------------------------------------------------------
# 2) The log probability of whole answers, and the DPO loss

def collate(answers):
    """ pad a list of answers into (B, T) ids and mask, the padding has mask 0 """
    T = max(len(a["ids"]) for a in answers)
    ids = torch.tensor([a["ids"] + [assistant_end] * (T - len(a["ids"])) for a in answers], device=device)
    mask = torch.tensor([a["mask"] + [0] * (T - len(a["mask"])) for a in answers], device=device)
    return ids, mask

def sequence_logprobs(model, ids, mask):
    """ (B,) the summed log probability of the answer tokens (where mask == 1) """
    inputs, targets = ids[:, :-1], ids[:, 1:].clone()
    targets[mask[:, 1:] == 0] = -1 # not the prompt, not the tool outputs, not the padding
    logp = -model(inputs, targets, loss_reduction='none').view_as(inputs) # 0 where the target is -1
    return logp.sum(dim=1)

def dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta):
    chosen_reward = beta * (policy_chosen - ref_chosen) # the "implicit reward" of each answer
    rejected_reward = beta * (policy_rejected - ref_rejected)
    loss = -F.logsigmoid(chosen_reward - rejected_reward).mean()
    return loss, chosen_reward, rejected_reward

def pair_logprobs(model, pairs):
    ids, mask = collate([p["chosen"] for p in pairs] + [p["rejected"] for p in pairs]) # one forward pass for both
    logp = sequence_logprobs(model, ids, mask)
    return logp[:len(pairs)], logp[len(pairs):]

# -----------------------------------------------------------------------------
# 3) How we measure it: the leak on held-out ordinary questions, and spelling accuracy

@torch.no_grad()
def evaluate(engine):
    leaks, total = 0, 0
    for i, conversation in enumerate(ordinary_questions("test", args.eval_questions)):
        for answer in sample(engine, conversation, 4, seed=i, max_tokens=args.max_new_tokens_general):
            leaks += bool(LEAK.search(answer["text"]))
            total += 1
    bee = SpellingBee(size=args.eval_spelling, split="test")
    correct = 0
    for i in range(len(bee)):
        answer = sample(engine, bee[i], 1, seed=i, max_tokens=args.max_new_tokens, temperature=0.0)[0]
        correct += bee.evaluate(bee[i], answer["text"])
    return leaks / total, correct / len(bee)

if __name__ == "__main__":
    model = load_model(args.source, device).eval()
    engine = Engine(model, tokenizer)
    pairs_path = os.path.join(get_base_dir(), f"dpo_pairs_{args.source}.jsonl")
    if os.path.exists(pairs_path):
        pairs = [json.loads(line) for line in open(pairs_path)]
        print(f"Loaded {len(pairs)} pairs from {pairs_path}")
    else:
        pairs = collect_pairs(engine, pairs_path + ".progress")
        with open(pairs_path, "w") as f:
            f.writelines(json.dumps(p) + "\n" for p in pairs)
        print(f"Saved {len(pairs)} pairs to {pairs_path}")
    print(f"{sum(p['kind'] == 'general' for p in pairs)} ordinary pairs, {sum(p['kind'] == 'spelling' for p in pairs)} spelling pairs")
    random.Random(1).shuffle(pairs)
    num_val = max(1, len(pairs) // 10)
    val_pairs, train_pairs = pairs[:num_val], pairs[num_val:]

    if args.eval_before:
        leak, spelling = evaluate(engine)
        print(f"before DPO | leak {leak:.3f} | SpellingBee {spelling:.2f}")

    # the reference model never changes, so we compute its log probabilities once, up front
    with torch.no_grad():
        for k in range(0, len(pairs), args.batch_size):
            batch = pairs[k:k + args.batch_size]
            ref_chosen, ref_rejected = pair_logprobs(model, batch)
            for p, c, r in zip(batch, ref_chosen.tolist(), ref_rejected.tolist()):
                p["ref_chosen"], p["ref_rejected"] = c, r

    def val_metrics():
        with torch.no_grad():
            losses, correct = [], 0
            for k in range(0, len(val_pairs), args.batch_size):
                batch = val_pairs[k:k + args.batch_size]
                policy_chosen, policy_rejected = pair_logprobs(model, batch)
                ref = lambda key: torch.tensor([p[key] for p in batch], device=device)
                loss, cr, rr = dpo_loss(policy_chosen, policy_rejected, ref("ref_chosen"), ref("ref_rejected"), args.beta)
                losses.append(loss.item() * len(batch))
                correct += (cr > rr).sum().item()
            return sum(losses) / len(val_pairs), correct / len(val_pairs)

    model.train()
    optimizer = model.configure_optimizer(lr=args.lr, weight_decay=0.0)
    num_steps = args.epochs * (len(train_pairs) // args.batch_size)
    step = 0
    for epoch in range(args.epochs):
        random.Random(epoch).shuffle(train_pairs)
        for k in range(0, len(train_pairs) - args.batch_size + 1, args.batch_size):
            if step % 20 == 0:
                val_loss, val_acc = val_metrics()
                print(f"step {step:4d} | val loss {val_loss:.4f} | val accuracy {val_acc:.3f}")
            t0 = time.time()
            batch = train_pairs[k:k + args.batch_size]
            policy_chosen, policy_rejected = pair_logprobs(model, batch)
            ref = lambda key: torch.tensor([p[key] for p in batch], device=device)
            loss, chosen_reward, rejected_reward = dpo_loss(policy_chosen, policy_rejected, ref("ref_chosen"), ref("ref_rejected"), args.beta)
            # DPO only cares about the margin, so it can push the chosen answers down too, as long as the rejected ones go down
            # faster, and the model drifts away from everything it knows. An SFT loss on the chosen answers (per token) anchors it
            num_chosen_tokens = torch.tensor([sum(p["chosen"]["mask"][1:]) for p in batch], device=device)
            sft_loss = -(policy_chosen / num_chosen_tokens).mean()
            (loss + args.sft_weight * sft_loss).backward()
            lr = args.lr * (1 - step / num_steps) # linear decay to zero
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            accuracy = (chosen_reward > rejected_reward).float().mean().item()
            print(f"step {step:4d} | loss {loss.item():.4f} | sft {sft_loss.item():.3f} | accuracy {accuracy:.2f} | chosen {chosen_reward.mean().item():+.3f} | "
                  f"rejected {rejected_reward.mean().item():+.3f} | lr {lr:.2e} | dt {time.time() - t0:.1f}s", flush=True)
    val_loss, val_acc = val_metrics()
    print(f"step {step:4d} | val loss {val_loss:.4f} | val accuracy {val_acc:.3f}")

    model.eval()
    leak, spelling = evaluate(engine)
    print(f"after DPO  | leak {leak:.3f} | SpellingBee {spelling:.2f}")
    path = os.path.join(get_base_dir(), "chatdpo.pt")
    torch.save(model.state_dict(), path)
    print(f"Saved the DPO model to {path}")
