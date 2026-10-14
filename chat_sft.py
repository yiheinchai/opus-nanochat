"""
Supervised finetuning (SFT): turn the base model into an assistant.
Mirrors nanochat/scripts/chat_sft.py.

GPU (e.g. Colab A100/H100), about 30-40 minutes:
python chat_sft.py

CPU (a quick demo, about 1.5 hours on 4 cores):
python chat_sft.py --device-batch-size=4 --seq-len=512 --grad-accum=1 --num-iterations=600 --smoltalk=20000 --mmlu=20000
"""

import os
import time
import argparse
import torch

from common import get_base_dir, autodetect_device
from gpt import load_model
from tokenizer import Tokenizer
from tasks import TaskMixture, SmolTalk, MMLU, GSM8K, SpellingBee

parser = argparse.ArgumentParser()
parser.add_argument("--device-batch-size", type=int, default=16, help="rows per forward/backward")
parser.add_argument("--seq-len", type=int, default=1024, help="tokens per row")
parser.add_argument("--grad-accum", type=int, default=4, help="forward/backwards per optimizer step")
parser.add_argument("--num-iterations", type=int, default=-1, help="optimizer steps (-1 = one full epoch)")
parser.add_argument("--lr", type=float, default=1e-4, help="peak learning rate")
parser.add_argument("--warmdown-ratio", type=float, default=0.5, help="fraction of training spent decaying the LR to zero")
parser.add_argument("--smoltalk", type=int, default=None, help="max SmolTalk conversations (default: all 460K)")
parser.add_argument("--mmlu", type=int, default=None, help="max MMLU conversations (default: all 100K)")
parser.add_argument("--eval-every", type=int, default=100, help="evaluate the validation loss every N steps")
parser.add_argument("--compile", type=int, default=1, help="torch.compile the model (cuda only)")
args = parser.parse_args()

torch.manual_seed(1337)
device = autodetect_device()
autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda")
tokenizer = Tokenizer()
model = load_model("base", device)
train_model = torch.compile(model) if device == "cuda" and args.compile else model

# -----------------------------------------------------------------------------
# The data: a mixture of conversations, each teaching the assistant something different
train_dataset = TaskMixture([
    SmolTalk("train", stop=args.smoltalk), # general conversations
    MMLU("auxiliary_train", stop=args.mmlu), # multiple choice questions (answer with a letter)
    GSM8K("train"), GSM8K("train"),        # math, and using the calculator tool (2 epochs)
    SpellingBee(size=20000, split="train"),# spelling and counting letters
])
val_dataset = TaskMixture([
    SmolTalk("test", stop=2000), MMLU("test", stop=500), GSM8K("test", stop=200), SpellingBee(size=200, split="test"),
])
print(f"Training on {len(train_dataset):,} conversations")

def sft_data_generator(dataset, batch_size, seq_len, buffer_size=100):
    """
    Pack whole conversations into rows of seq_len+1 tokens (every row starts at a BOS).
    We use "best fit": keep a buffer of conversations, and always grab the longest one that
    still fits in the row. When nothing fits anymore, pad the rest of the row (we never crop).
    Yields inputs, targets, and the fraction of the dataset consumed so far.
    """
    row_capacity = seq_len + 1 # +1 because the targets are the inputs shifted by one
    buffer, cursor = [], 0
    while True:
        rows, mask_rows = [], []
        for _ in range(batch_size):
            row, mask_row = [], []
            while True:
                while len(buffer) < buffer_size:
                    ids, mask = tokenizer.render_conversation(dataset[cursor % len(dataset)], max_tokens=row_capacity + 1)
                    if len(ids) <= row_capacity: # skip the (rare) conversations too long to ever fit
                        buffer.append((ids, mask))
                    cursor += 1
                remaining = row_capacity - len(row)
                fits = [i for i, (ids, _) in enumerate(buffer) if len(ids) <= remaining]
                if not fits:
                    break
                ids, mask = buffer.pop(max(fits, key=lambda i: len(buffer[i][0])))
                row += ids
                mask_row += mask
            padding = row_capacity - len(row)
            rows.append(row + [tokenizer.bos] * padding)
            mask_rows.append(mask_row + [0] * padding) # never train on the padding
        batch, mask = torch.tensor(rows), torch.tensor(mask_rows)
        inputs, targets = batch[:, :-1], batch[:, 1:].clone()
        targets[mask[:, 1:] == 0] = -1 # only train on the assistant's tokens: -1 is ignored by the loss
        yield inputs.to(device), targets.to(device), cursor / len(dataset)

@torch.no_grad()
def evaluate_loss(num_batches=20):
    model.eval()
    loader = sft_data_generator(val_dataset, args.device_batch_size, args.seq_len)
    losses = []
    for _ in range(num_batches):
        x, y, _ = next(loader)
        with autocast:
            losses.append(train_model(x, y).item())
    model.train()
    return sum(losses) / len(losses)

# -----------------------------------------------------------------------------
# The training loop: exactly like pretraining, except for the -1's in the targets
optimizer = model.configure_optimizer(lr=args.lr, weight_decay=0.0)

def get_lr(progress):
    # constant, and then a linear decay to zero over the last warmdown_ratio of training
    if progress < 1 - args.warmdown_ratio:
        return args.lr
    return args.lr * (1 - progress) / args.warmdown_ratio

train_loader = sft_data_generator(train_dataset, args.device_batch_size, args.seq_len)
step = 0
while True:
    x, y, epoch_progress = next(train_loader)
    progress = step / args.num_iterations if args.num_iterations > 0 else epoch_progress
    last_step = progress >= 1.0
    if step % args.eval_every == 0 or last_step:
        print(f"step {step:5d} | val loss {evaluate_loss():.4f}")
    if last_step:
        break

    t0 = time.time()
    lr = get_lr(progress)
    for group in optimizer.param_groups:
        group["lr"] = lr
    for micro_step in range(args.grad_accum):
        if micro_step > 0:
            x, y, _ = next(train_loader)
        with autocast:
            loss = train_model(x, y)
        (loss / args.grad_accum).backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    step += 1
    num_tokens = (y >= 0).sum().item() * args.grad_accum
    print(f"step {step:5d} | progress {100 * progress:.1f}% | loss {loss.item():.4f} | lr {lr:.2e} | trained tokens {num_tokens} | dt {time.time() - t0:.2f}s")

path = os.path.join(get_base_dir(), "chatsft.pt")
torch.save(model.state_dict(), path)
print(f"Saved the SFT model to {path}")
