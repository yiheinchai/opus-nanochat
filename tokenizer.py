"""
The GPT-2 tokenizer (via tiktoken), plus the special tokens that turn a
conversation into a single sequence of tokens. Mirrors nanochat/tokenizer.py.
"""

import tiktoken

SPECIAL_TOKENS = [
    "<|user_start|>", # user messages
    "<|user_end|>",
    "<|assistant_start|>", # assistant messages
    "<|assistant_end|>",
    "<|python_start|>", # assistant invokes the python tool
    "<|python_end|>",
    "<|output_start|>", # python output, back to the assistant
    "<|output_end|>",
]

class Tokenizer:

    def __init__(self):
        gpt2 = tiktoken.get_encoding("gpt2")
        # GPT-2 already has <|endoftext|> = 50256, we use it as the BOS (beginning of sequence) token.
        # The new special tokens take the next ids, 50257..50264, inside our padded vocab of 50304
        special = {"<|endoftext|>": 50256}
        special.update({name: 50257 + i for i, name in enumerate(SPECIAL_TOKENS)})
        self.enc = tiktoken.Encoding(name="gpt2_chat", pat_str=gpt2._pat_str,
                                     mergeable_ranks=gpt2._mergeable_ranks, special_tokens=special)
        self.bos = 50256

    def encode(self, text):
        return self.enc.encode_ordinary(text) # never parses special tokens out of the text itself

    def decode(self, ids):
        return self.enc.decode(ids)

    def encode_special(self, name):
        return self.enc.encode_single_token(name)

    def render_conversation(self, conversation, max_tokens=1024):
        """
        Tokenize a conversation. Returns ids and a mask of the same length:
        mask = 1 for the tokens that the assistant should be trained to produce.
        """
        ids, mask = [], []
        def add_tokens(token_ids, mask_val):
            if isinstance(token_ids, int):
                token_ids = [token_ids]
            ids.extend(token_ids)
            mask.extend([mask_val] * len(token_ids))

        messages = conversation["messages"]
        if messages[0]["role"] == "system":
            # we have no special tokens for system messages, simply prepend to the first user message
            messages = [{"role": "user", "content": messages[0]["content"] + "\n\n" + messages[1]["content"]}] + messages[2:]

        s = self.encode_special
        add_tokens(self.bos, 0)
        for i, message in enumerate(messages):
            assert message["role"] == ("user" if i % 2 == 0 else "assistant"), "messages must alternate user/assistant"
            content = message["content"]
            if message["role"] == "user":
                add_tokens(s("<|user_start|>"), 0)
                add_tokens(self.encode(content), 0)
                add_tokens(s("<|user_end|>"), 0)
            else:
                add_tokens(s("<|assistant_start|>"), 0)
                if isinstance(content, str):
                    content = [{"type": "text", "text": content}]
                for part in content:
                    value_ids = self.encode(part["text"])
                    if part["type"] == "text":
                        add_tokens(value_ids, 1)
                    elif part["type"] == "python":
                        # the assistant writes the tool call, so we do train on it
                        add_tokens(s("<|python_start|>"), 1)
                        add_tokens(value_ids, 1)
                        add_tokens(s("<|python_end|>"), 1)
                    elif part["type"] == "python_output":
                        # python writes the output at test time, so we don't train on it
                        add_tokens(s("<|output_start|>"), 0)
                        add_tokens(value_ids, 0)
                        add_tokens(s("<|output_end|>"), 0)
                add_tokens(s("<|assistant_end|>"), 1) # the assistant must learn when to stop!

        return ids[:max_tokens], mask[:max_tokens]

    def render_for_completion(self, conversation):
        """ Render all but the last (assistant) message, then prime the assistant to respond. """
        messages = conversation["messages"]
        assert messages[-1]["role"] == "assistant"
        ids, _ = self.render_conversation({"messages": messages[:-1]})
        ids.append(self.encode_special("<|assistant_start|>"))
        return ids

    def visualize(self, ids, mask):
        # green = trained on, red = not trained on
        GREEN, RED, RESET = '\033[92m', '\033[91m', '\033[0m'
        return '|'.join(f"{GREEN if m else RED}{self.decode([i])}{RESET}" for i, m in zip(ids, mask))
