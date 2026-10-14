"""
Talk to your model. Mirrors nanochat/scripts/chat_cli.py.

python chat_cli.py -i sft
python chat_cli.py -i rl -p "How many r are in strawberry?"
"""

import argparse
from common import autodetect_device
from gpt import load_model
from tokenizer import Tokenizer
from engine import Engine

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--source', type=str, default="sft", help="base|sft|rl")
parser.add_argument('-p', '--prompt', type=str, default='', help="ask a single question and exit")
parser.add_argument('-t', '--temperature', type=float, default=0.6)
parser.add_argument('-k', '--top-k', type=int, default=50)
args = parser.parse_args()

model = load_model(args.source, autodetect_device()).eval()
tokenizer = Tokenizer()
engine = Engine(model, tokenizer)
s = tokenizer.encode_special
conversation = [tokenizer.bos]

print("Type 'clear' to start over, 'quit' to exit")
while True:
    user_input = args.prompt or input("\nUser: ").strip()
    if user_input == "quit":
        break
    if user_input == "clear":
        conversation = [tokenizer.bos]
        continue
    message = [s("<|user_start|>"), *tokenizer.encode(user_input), s("<|user_end|>"), s("<|assistant_start|>")]
    if len(conversation) + len(message) > model.config.block_size - 256:
        print("(the conversation got too long for our 1024 token context, starting over)")
        conversation = [tokenizer.bos]
    conversation += message
    print("\nAssistant: ", end="", flush=True)
    response = []
    for column, _ in engine.generate(conversation, max_tokens=512, temperature=args.temperature, top_k=args.top_k):
        token = column[0]
        response.append(token)
        if token != s("<|assistant_end|>"):
            print(tokenizer.decode([token]), end="", flush=True)
    print()
    if response[-1] != s("<|assistant_end|>"):
        response.append(s("<|assistant_end|>")) # ran out of tokens, close the message anyway
    conversation += response
    if args.prompt:
        break
