"""
Evaluate a chat model. Mirrors nanochat/scripts/chat_eval.py.

python chat_eval.py -i sft
python chat_eval.py -i rl -a SpellingBee -x 100
"""

import argparse
import torch

from common import autodetect_device
from gpt import load_model
from tokenizer import Tokenizer
from engine import Engine
from tasks import MMLU, ARC, GSM8K, SpellingBee

@torch.no_grad()
def run_categorical_eval(task, model, tokenizer, max_problems=None):
    """
    Multiple choice: we don't even need to sample. Run the prompt through the model once,
    look at the logits at the position where the assistant answers, and only compare the
    logits of the allowed letters. The highest one is the model's answer.
    """
    device = next(model.parameters()).device
    num_problems = len(task) if max_problems is None else min(len(task), max_problems)
    num_passed = 0
    for i in range(num_problems):
        conversation = task[i]
        ids = tokenizer.render_for_completion(conversation)
        logits = model(torch.tensor([ids], device=device))[0, -1] # (vocab_size,)
        letter_ids = [tokenizer.encode(letter)[0] for letter in conversation["letters"]]
        predicted = conversation["letters"][logits[letter_ids].argmax().item()]
        num_passed += task.evaluate(conversation, predicted)
    return num_passed / num_problems

def run_generative_eval(task, model, tokenizer, engine, max_problems=None, num_samples=1, temperature=0.0, max_tokens=256):
    """ Open ended: sample a response, parse out the answer, and check it. pass@k if num_samples > 1 """
    num_problems = len(task) if max_problems is None else min(len(task), max_problems)
    num_passed = 0
    for i in range(num_problems):
        conversation = task[i]
        ids = tokenizer.render_for_completion(conversation)
        results, _ = engine.generate_batch(ids, num_samples=num_samples, max_tokens=max_tokens, temperature=temperature, top_k=50)
        completions = [tokenizer.decode(r[len(ids):]) for r in results]
        num_passed += any(task.evaluate(conversation, c) for c in completions)
        print(f"\r{num_passed}/{i + 1} ({100 * num_passed / (i + 1):.2f}%)", end="", flush=True)
    print()
    return num_passed / num_problems

TASKS = {
    'ARC-Easy': lambda: ARC("ARC-Easy", "test"),
    'ARC-Challenge': lambda: ARC("ARC-Challenge", "test"),
    'MMLU': lambda: MMLU("test"),
    'GSM8K': lambda: GSM8K("test"),
    'SpellingBee': lambda: SpellingBee(size=256, split="test"),
}
BASELINES = {'ARC-Easy': 0.25, 'ARC-Challenge': 0.25, 'MMLU': 0.25, 'GSM8K': 0.0, 'SpellingBee': 0.0} # random guessing

def run_chat_eval(task_name, model, tokenizer, engine, max_problems=None):
    task = TASKS[task_name]()
    if task_name in ('ARC-Easy', 'ARC-Challenge', 'MMLU'):
        return run_categorical_eval(task, model, tokenizer, max_problems)
    return run_generative_eval(task, model, tokenizer, engine, max_problems)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--source', type=str, default="sft", help="base|sft|rl")
    parser.add_argument('-a', '--task-name', type=str, default=None, help="task name, or several separated by |. default = all")
    parser.add_argument('-x', '--max-problems', type=int, default=None, help="max problems per task")
    args = parser.parse_args()

    model = load_model(args.source, autodetect_device()).eval()
    tokenizer = Tokenizer()
    engine = Engine(model, tokenizer)
    task_names = list(TASKS) if args.task_name is None else args.task_name.split('|')
    results = {}
    for task_name in task_names:
        results[task_name] = run_chat_eval(task_name, model, tokenizer, engine, args.max_problems)
        print(f"{task_name}: {100 * results[task_name]:.2f}%")
    if len(results) == len(TASKS):
        # ChatCORE: accuracy re-scaled so that 0 = random guessing and 1 = perfect, then averaged
        chatcore = sum((results[t] - BASELINES[t]) / (1 - BASELINES[t]) for t in results) / len(results)
        print(f"ChatCORE: {chatcore:.4f}")
