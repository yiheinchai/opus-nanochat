# Let's build the post-training of nanochat: from GPT-2 to a ChatGPT

<!--
This file is the single source of truth for the lecture. lecture/build.py turns it into
1) posttraining.ipynb (the follow-along notebook), 2) the narration audio, 3) the screencast video.

Markup:
- plain paragraphs are narration (spoken, and markdown cells in the notebook)
- [[cue ...]] inside the narration directs the screencast (see lecture/screencast.py), it is not spoken
- ```python                         executed code cell
- ```python from=FILE start="..."   code pulled from a repo file (shown in the editor)
  (add exec=0 to only show it, end="..." to stop before a line containing that text)
- ```bash                           a shell command (a ! cell in the notebook, not executed at build time; exec=0 = markdown only)
- ```output                         the output of the preceding bash cell, copied from a real run
- ```whiteboard                     whiteboard elements, drawn by [[draw name]] cues
- > **Exercise:** ...               a "pause the video and try it yourself" moment
-->

## Intro

Hi everyone. So today we are going to continue where Andrej Karpathy's Zero to Hero series left off. If you've been following along, [[write "micrograd → makemore → GPT → GPT-2 (124M)"]] you've built micrograd, you've built makemore, you've built a GPT from scratch on tiny Shakespeare, and in the last video you reproduced GPT-2, the 124 million parameter version. And at the end of all that, [[write "= a base model: it continues text"]] you had a base model. A thing that can continue text. But if you actually try to talk to it, you'll find that it's not ChatGPT. [[write "... but it is not an assistant"]] It's not an assistant. It just sort of babbles.

So the question for today is: [[write "today: what happens after pretraining?"]] what happens after pretraining? How do you go from a base model, an internet document simulator, to something that you can actually talk to, that answers your questions, that uses tools, and that can even count the number of r's in strawberry? [[write "→ post-training"]] This stage is called post-training, and today we are going to build all of it, from scratch, following Karpathy's nanochat repository.

Now here is my goal for this video, and I want to be very explicit about it. When you're done, you should be able to reproduce every piece of nanochat's post-training yourself, without looking. Not just know the words, but have typed every line, and understand why every line is there. So we're going to do this the same way as in the rest of the series. We'll start from a problem, try the most obvious thing, see it fail, and then fix it. Every piece of code gets written in the notebook first and tested on tiny examples, and only at the end of each chapter will we look at the file in the repository, which will be the exact same code, cleaned up.

Here's what you need. You need the base model from the GPT-2 video, and I've put that model, [[open gpt.py "class GPT(nn.Module):"]] with two small additions that we'll write together today, in this gpt.py file. It's the same code, [[point "wte = nn.Embedding"]] the token embeddings, [[point "wpe = nn.Embedding"]] the position embeddings, [[point "h = nn.ModuleList"]] the twelve transformer blocks, [[point "self.lm_head = nn.Linear"]] the language model head, [[underline "self.transformer.wte.weight = self.lm_head.weight"]] with the same weight sharing. And we'll load the original OpenAI weights into it, exactly like we did in that video. Pretraining is the expensive part of the pipeline, it needs a node of eight H100 GPUs for a couple of hours. But post-training is cheap, and everything today runs on a laptop CPU, or on a free Colab GPU if you want it faster.

So here is the roadmap. [[erase]] [[write "1. what a base model actually does"]] First, we'll look at what a base model actually does. [[write "2. conversations → tokens (special tokens, the mask)"]] Then we'll figure out how to turn a conversation into tokens. [[write "3. data: chat, multiple choice, math with a tool, spelling"]] Then we'll build a mixture of training data, [[write "4. a dataloader, and SFT"]] a dataloader for it, and we'll run supervised finetuning. [[write "5. an inference engine: KV cache + tools"]] Then we'll write an inference engine with a KV cache and a python tool, [[write "6. evaluation"]] evaluations to measure what we did, [[write "7. reinforcement learning"]] reinforcement learning, where the model practices and learns from its own successes and failures, [[write "8. preferences: DPO"]] and finally preference optimization, for all the things where we can't check the answer, but we can tell which of two answers is better.

[[erase]] And two things about how to use this video. [[write "pause and predict"]] Every now and then, a yellow box will pop up with a question, and a little countdown. That's your cue to pause, and actually commit to a guess before I show you the answer. You learn a lot more from a wrong guess than from nodding along. [[write "exercises.ipynb: write it, check it"]] And for every piece we build, there's an exercise in the exercises notebook in the repo. You get the signature of a function, you write it, and a check runs it against the real thing and tells you exactly what's wrong. My recommendation is to watch a chapter, then close the video, and do its exercises from memory. When they all pass, you've built nanochat's post-training yourself. Okay, so let's get started.

## The base model is a document simulator

Okay so first, let's load up GPT-2 and sample from it. The Engine here is something we'll write ourselves later in the lecture, for now just think of it as a function that samples tokens from the model. And I'm writing a little complete function that takes a prompt, puts the beginning of sequence token in front of it, and samples greedily, so always the most likely next token.

```python
import torch
from gpt import GPT
from tokenizer import Tokenizer
from engine import Engine

model = GPT.from_pretrained().eval()
tokenizer = Tokenizer()
engine = Engine(model, tokenizer)

def complete(prompt, max_tokens=60):
    ids = [tokenizer.bos] + tokenizer.encode(prompt)
    out, _ = engine.generate_batch(ids, max_tokens=max_tokens, temperature=0.0)
    return tokenizer.decode(out[0][len(ids):])

print(complete("Hello, I'm a language model,", max_tokens=20))
```

[[circle "and I'm a programmer."]] "Hello, I'm a language model, and I'm a programmer." You'll recognize this from the GPT-2 video, [[note "and I'm a programmer." "same as the GPT-2 video"]] it's exactly the same output, so we know the weights are loaded correctly. Now let's ask it a question.

> **Predict:** This model was never trained to answer questions. What will it write after "What is the capital of France?"

```python
print(complete("What is the capital of France?"))
```

Okay, so actually this is not too bad. [[circle "The capital of France is Paris."]] It does know that the capital of France is Paris. [[highlight "Paris. The capital of France is Paris. The capital of France is Paris."]] But then it says it again, and again, and again, forever. And I want you to really think about why. This model was trained on web pages. So when it sees a question, it's asking itself, statistically, what text comes next in a web page that starts like this. [[note "The capital of France is Paris." "a document simulator"]] It's not answering you. It's simulating a document. And documents on the internet don't end after one sentence, so it has no notion of being done.

Now because it's a document simulator, there's actually a fun trick. We can give it the beginning of a document that looks like a conversation with an assistant, and let it continue that document. This is called few-shot prompting.

```python
prompt = """The following is a conversation between a helpful AI Assistant and a Human.

Human: Hi there! Who are you?
Assistant: Hello! I'm an AI Assistant. How can I help you today?

Human: What is 2+2?
Assistant: 2+2 is 4.

Human: What is the capital of France?
Assistant:"""
print(complete(prompt, max_tokens=40))
```

So look at this. [[point "Assistant: The United States."]] It did pick up on the format, it answers in the voice of the Assistant. [[circle " France."]] The answer is "France", which is wrong, it's a small model. But the much bigger problem is what happens next. [[underline "Human: What is the capital of the United States?"]] It writes the next question from the Human, [[underline "Assistant: The United States."]] and then the answer of the Assistant, and so on. [[note "Human: What is the capital of the United Kingdom?" "it plays both sides!"]] It's simulating the whole document, both sides of the conversation, because that's all it knows how to do.

So this tells us three things. [[write "1. the knowledge is already in there"]] Number one, the knowledge is already in there, from pretraining. [[write "2. it plays an assistant if the document looks like a chat"]] Number two, the model is perfectly happy to play an assistant, if the document looks like a chat. [[write "3. it doesn't know when its turn ends"]] And number three, it has no idea where its turn ends. And that suggests the whole idea of post-training. [[write "→ keep training on documents that ARE conversations"]] Instead of tricking the model with a prompt, we just keep training it, on a lot of documents that are conversations, in some very specific format with clearly marked turns. Then conversations become the kind of document it knows.

> **Exercise:** Before we move on, think about how you would do this. If a conversation has to become a single sequence of tokens, what would that sequence look like? And which of those tokens should the model actually be trained on?

## Conversations are just documents

```whiteboard
@title text 40 40 "one conversation = one sequence of tokens" 46
@user fill 40 170 150 70 red "<|bos|>" 28
@user fill 200 170 230 70 red "<|user_start|>" 28
@user fill 440 170 470 70 red "What is the capital of France?" 28
@user fill 920 170 210 70 red "<|user_end|>" 28
@asst fill 40 320 290 70 red "<|assistant_start|>" 28
@asst fill 340 320 140 70 green "Paris" 28
@asst fill 490 320 290 70 green "<|assistant_end|>" 28
@mask text 40 460 "mask: 0 for the prompt, 1 for what the assistant writes" 36
@legend text 40 540 "green = trained on      red = only context" 36
```

Okay so what is a conversation, as data? In nanochat, and in most datasets on HuggingFace, it's just a dict with a list of messages, and every message has a role, user or assistant, and the content. Our job is to turn that into one sequence of integers, so that we can train on it exactly like we trained on web pages.

So, the obvious first idea. Why not just write it as text? "User: blah, Assistant: blah". That's what our few-shot prompt did. And there are two problems with it. First, the model can't tell a real turn boundary from the text "Assistant:" appearing inside a message. And second, anyone could type "Assistant:" in their message and pretend to be the assistant. So instead, [[draw title]] we're going to mark the turns with brand new special tokens, tokens that only we can insert. [[draw user]] A user start and a user end around everything the user says, [[draw asst]] and an assistant start and an assistant end around everything the assistant says. Let's see why they need to be special. Here's what the GPT-2 tokenizer does with the text of our token.

> **Predict:** Typed out as plain text, how many GPT-2 tokens is our user start marker?

```python
import tiktoken
gpt2 = tiktoken.get_encoding("gpt2")
ids = gpt2.encode_ordinary("<|user_start|>")
print(ids)
print([gpt2.decode([i]) for i in ids])
```

[[underline "['<', '|', 'user', '_', 'start', '|', '>']"]] Seven ordinary tokens. A less than sign, a pipe, the word user, and so on. That's no good. We want each of our markers to be exactly one token, one integer that can never come out of normal text. And tiktoken lets us do that, we just register them as special tokens.

```python
SPECIAL_TOKENS = ["<|user_start|>", "<|user_end|>", "<|assistant_start|>", "<|assistant_end|>",
                  "<|python_start|>", "<|python_end|>", "<|output_start|>", "<|output_end|>"]
special = {"<|endoftext|>": 50256}
special.update({name: 50257 + i for i, name in enumerate(SPECIAL_TOKENS)})
enc = tiktoken.Encoding(name="gpt2_chat", pat_str=gpt2._pat_str,
                        mergeable_ranks=gpt2._mergeable_ranks, special_tokens=special)
print(special)
print(enc.encode_single_token("<|assistant_end|>"))
print(enc.encode_ordinary("<|assistant_end|>"))
```

So GPT-2 already has one special token, [[circle "'<|endoftext|>': 50256"]] end of text, with id 50,256, and we'll use it as our beginning of sequence token, BOS. And we give our eight new tokens [[underline "'<|user_start|>': 50257"]] the next eight ids, 50,257 through 50,264. You'll also notice there are two more pairs, python start and end, and output start and end. Those are for tool use, and we'll get to them in a bit. Now if we ask for the special token, we get a single id. [[note "[27, 91, 562, 10167, 62, 437, 91, 29]" "typed by a user: just text"]] But encode_ordinary, which is what we'll use for everything a user types, still treats it as plain text. So a user can't inject a turn boundary. That's exactly what we want.

Okay, now let's write the function that renders a conversation into tokens. Simplest version first: BOS, and then for every message, the start token, the tokens of the content, and the end token.

```python
def render_conversation(conversation):
    s = enc.encode_single_token
    ids = [50256] # every conversation starts with the BOS token
    for message in conversation["messages"]:
        if message["role"] == "user":
            ids += [s("<|user_start|>"), *enc.encode_ordinary(message["content"]), s("<|user_end|>")]
        else:
            ids += [s("<|assistant_start|>"), *enc.encode_ordinary(message["content"]), s("<|assistant_end|>")]
    return ids

conversation = {"messages": [
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "Paris"},
]}
ids = render_conversation(conversation)
print(ids)
print([enc.decode([i]) for i in ids])
```

[[underline "[50256, 50257, 2061, 318, 262, 3139, 286, 4881, 30, 50258, 50259, 40313, 50260]"]] So the whole conversation is thirteen tokens. [[circle "'Paris'"]] And the decoded pieces look right: BOS, user start, the question, user end, assistant start, Paris, assistant end.

Now here's the important question, the one from the exercise. If we train on this sequence like we train on a web page, the model learns to predict every one of these tokens, including the user's question. Is that what we want? Think about it. We don't want a model that simulates users, we want a model that responds to them. Predicting what people ask is not useless, but it's not our job, and with a small model, every bit of capacity matters. So we only want to train on the tokens that the assistant writes. [[draw mask]] And the way we do that is we return a second list, a mask, with a one for every token the assistant should learn to produce, and a zero for everything else.

```python
def render_conversation(conversation):
    s = enc.encode_single_token
    ids, mask = [], []
    def add_tokens(token_ids, mask_val):
        ids.extend(token_ids)
        mask.extend([mask_val] * len(token_ids))
    add_tokens([50256], 0)
    for message in conversation["messages"]:
        if message["role"] == "user":
            add_tokens([s("<|user_start|>"), *enc.encode_ordinary(message["content"]), s("<|user_end|>")], 0)
        else:
            add_tokens([s("<|assistant_start|>")], 0)
            add_tokens(enc.encode_ordinary(message["content"]), 1)
            add_tokens([s("<|assistant_end|>")], 1)
    return ids, mask

ids, mask = render_conversation(conversation)
for i, m in zip(ids, mask):
    print(f"{enc.decode([i])!r:24} mask={m}")
```

[[highlight "'<|user_start|>'"]] So everything the user wrote, and all the special tokens around it, get a zero. [[point "'<|assistant_start|>'"]] The assistant start token also gets a zero, because we always put it there ourselves, the model never has to produce it. [[circle "'Paris'"]] Then the content gets a one, [[circle "'<|assistant_end|>'"]] and then assistant end also gets a one. And this one is really important. [[note "'<|assistant_end|>'" "learn when to stop!"]] This is how the model learns to stop talking. Remember the base model that went on forever? This is the fix. After Paris, the model must learn to emit assistant end, and then we stop sampling, and it's the user's turn.

Now for tool use. Sometimes we want the assistant to run a bit of python, for example a calculator. So the content of an assistant message can also be a list of parts. A text part is just text. A python part is code that the assistant writes, and we wrap it in python start and python end. And a python output part is the result of running that code, wrapped in output start and output end.

> **Predict:** The model writes the python code, and python writes the output. Which of these tokens should get a mask of one?

```python
def render_conversation(conversation):
    s = enc.encode_single_token
    ids, mask = [], []
    def add_tokens(token_ids, mask_val):
        ids.extend(token_ids)
        mask.extend([mask_val] * len(token_ids))
    add_tokens([50256], 0)
    for message in conversation["messages"]:
        content = message["content"]
        if message["role"] == "user":
            add_tokens([s("<|user_start|>"), *enc.encode_ordinary(content), s("<|user_end|>")], 0)
            continue
        add_tokens([s("<|assistant_start|>")], 0)
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        for part in content:
            value = enc.encode_ordinary(part["text"])
            if part["type"] == "text":
                add_tokens(value, 1)
            elif part["type"] == "python": # the assistant writes the code: train on it
                add_tokens([s("<|python_start|>"), *value, s("<|python_end|>")], 1)
            elif part["type"] == "python_output": # python writes the output: don't train on it
                add_tokens([s("<|output_start|>"), *value, s("<|output_end|>")], 0)
        add_tokens([s("<|assistant_end|>")], 1)
    return ids, mask

def visualize(ids, mask):
    GREEN, RED, RESET = '\033[92m', '\033[91m', '\033[0m'
    return '|'.join(f"{GREEN if m else RED}{enc.decode([i])}{RESET}" for i, m in zip(ids, mask))

tool_conversation = {"messages": [
    {"role": "user", "content": "What is 123 times 45?"},
    {"role": "assistant", "content": [
        {"type": "text", "text": "Let me use python: "},
        {"type": "python", "text": "123*45"},
        {"type": "python_output", "text": "5535"},
        {"type": "text", "text": " The answer is 5535."},
    ]},
]}
print(visualize(*render_conversation(tool_conversation)))
```

[[circle "<|python_start|>|123|*|45|<|python_end|>"]] The code gets a one, because at test time the model has to write it. [[circle "<|output_start|>|55|35|<|output_end|>"]] But the output gets a zero. [[note "<|output_start|>|55|35|<|output_end|>" "python writes this, not the model"]] At test time, python is going to produce those tokens, not the model. If we trained on them, we would be teaching the model to predict what python would say, in other words to hallucinate tool outputs, and that's actively harmful. [[underline "| The| answer| is| 55|35|.|<|assistant_end|>"]] And then after the output, the model continues in green, and it can copy the result into its answer.

And one last function. At test time, we have a conversation where it's the assistant's turn. So we render everything except the last assistant message, and then we append assistant start. And that's our prompt. The model takes it from there, and we sample until it emits assistant end.

```python
def render_for_completion(conversation):
    ids, _ = render_conversation({"messages": conversation["messages"][:-1]})
    return ids + [enc.encode_single_token("<|assistant_start|>")]

print([enc.decode([i]) for i in render_for_completion(conversation)])
```

[[circle "'<|assistant_start|>'"]] So the prompt ends with assistant start, and the next token the model writes is the beginning of its answer. Now, I've put exactly this code into the tokenizer file of the repo, and let's just check that our notebook version and the repo version agree.

```python
print(tokenizer.render_conversation(tool_conversation) == render_conversation(tool_conversation))
print(tokenizer.render_for_completion(conversation) == render_for_completion(conversation))
```

[[open tokenizer.py "def render_conversation"]] Good. The repo version, in tokenizer.py, is the same function, [[point "if messages[0][\"role\"] == \"system\":"]] with two small additions: if a conversation starts with a system message, we just glue it onto the first user message, because we don't have special tokens for system messages. [[circle "return ids[:max_tokens], mask[:max_tokens]"]] And there's a max tokens argument that truncates, which, I'll tell you right now, is going to bite us later. And by the way, this exact format is a design choice. [[write "ChatML:  <|im_start|> ... <|im_end|>"]] OpenAI's ChatML uses tokens like im start and im end, [[write "Llama:  [INST] ... [/INST]"]] Llama has its own, and they all do basically the same thing. [[write "→ render the same way at train time and test time!"]] What matters is that you render conversations the same way at training time and at test time. Even a single space different can quietly hurt your model.

> **Exercise:** Add support for system messages to your render_conversation, the way the repo does it. Then write a conversation with two user turns and two assistant turns and visualize it. Which tokens are green? Then, in the exercises notebook, write render conversation and render for completion from scratch, exercises one and two, and make the checks pass.

## Brain surgery on GPT-2

Okay so we have our token ids, but now we have a problem. [[write "GPT-2 vocab: 50,257 tokens"]] GPT-2's vocabulary has 50,257 tokens, so its embedding table has exactly 50,257 rows, [[write "our special tokens: 50,257 ... 50,264"]] and our special tokens have ids 50,257 through 50,264. There's no row for them. If we feed id 50,257 into the original GPT-2, it's an index out of range error.

Now if you remember the GPT-2 reproduction video, [[write "vocab_size = 50,304 (a nice number)"]] we padded the vocab size from 50,257 up to 50,304, because it's a nice number, divisible by 128, and the GPU kernels like that. And we said those extra 47 rows are never used. Well, today they get used. So when we load the OpenAI weights, we copy their 50,257 rows, and we have to decide what to put in the new rows. [[open gpt.py "if k == 'wte.weight':"]] Here's the code. [[circle "sd[ours][:n].copy_(w)"]] Copy GPT-2's rows, and then, [[circle "sd[ours][n:].copy_(w.mean(dim=0))"]] for all the new rows, I'm using the average of all the token embeddings. Why the average? Because random vectors could be anywhere, but the average sort of lives in the region where real embeddings live, so it won't blow anything up. Let's check that it did what we wanted.

```python
w = model.transformer.wte.weight
print(w.shape)
print(model.lm_head.weight is w)
print((w[50257:] - w[:50257].mean(dim=0)).abs().max().item())
```

[[circle "torch.Size([50304, 768])"]] So the table has 50,304 rows. [[circle "True"]] The language model head is literally the same tensor as the embedding table, that's the weight sharing from the GPT-2 paper. And that matters a lot here, because it means the new rows are not only the input embeddings of our special tokens, they're also the output weights that produce their logits. [[note "0.0" "every new row = the average"]] And all the new rows are exactly the average.

Now, and this is something that Karpathy drills into you in the makemore videos, before we train anything, let's sanity check the loss at initialization. What do we expect? [[write "uniform guess: -log(1/50304) = 10.8"]] If the model had no idea and spread its probability evenly over the vocabulary, the loss would be minus log of one over 50,304, so about 10.8. Let's look at the loss of every single token of our conversation.

> **Predict:** The new rows are all the average embedding. Will the loss on the special tokens be above or below the uniform guess of 10.8?

```python
import math
print(-math.log(1 / 50304))
ids, mask = render_conversation(conversation)
x, y = torch.tensor([ids[:-1]]), torch.tensor([ids[1:]])
with torch.no_grad():
    losses = model(x, y, loss_reduction='none')
for token, m, loss in zip(ids[1:], mask[1:], losses):
    print(f"{enc.decode([token])!r:24} mask={m} loss={loss.item():.2f}")
```

[[circle "10.825839875788878"]] So 10.8 is the uniform guess. [[point "loss=1.81"]] The normal English tokens look fine, the model is quite sure that "is" follows "What", and so on. [[circle "loss=14.85"]] But look at the special tokens. [[circle "loss=15.17"]] All of them are around 15, [[note "loss=15.26" "worse than a uniform guess!"]] which is much worse than a uniform guess. The model is confidently wrong about them. And it's because of the weight sharing. The logit of a special token is the dot product of the hidden state with that average embedding, and that just happens to be a small number compared to the logits of real tokens. So the probability is tiny, and the loss is huge. And what this means is that at the start of finetuning, a lot of the loss is going to come from these tokens, and the first thing the model will do is learn that they exist. It's the hockey stick from makemore, all over again.

> **Exercise:** Can you find a better initialization for the new rows, one that gives a loss of about 10.8 or lower for the special tokens at initialization? Try initializing assistant end to the embedding of a newline, or scaling the average vector. Measure the loss with the cell above. And exercise three in the notebook is the surgery itself: expanding the embedding table.

## The data mixture

Okay, now we need conversations to train on. A lot of them. And here's the key idea of the SFT stage in nanochat: the data is a mixture of several datasets, and every one of them teaches the model one specific thing. [[write "SFT data = a mixture, each part teaches one thing"]] So let's build them one at a time. And the one I find the most instructive is grade school math, GSM8K, so let's start there.

All of these datasets live on HuggingFace, and every dataset on HuggingFace has an automatic export to parquet files. [[open tasks.py "def load_hub_dataset"]] So I wrote a tiny function that downloads those parquet files and reads them with pyarrow. You don't need the datasets library for this. Let's load the GSM8K training set and look at one problem.

```python
from tasks import load_hub_dataset
table = load_hub_dataset("openai/gsm8k", "main", split="train")
print(table.num_rows, table.column_names)
row = {col: table[col][3].as_py() for col in table.column_names}
print(row["question"])
print(row["answer"])
```

[[circle "7473"]] 7,473 problems, [[point "['question', 'answer']"]] each with a question and an answer. And look at the answer. [[circle "<<12*2=24>>"]] What are these things in double angle brackets? These are calculator annotations, from the people who made the dataset. "12 times 2 equals 24" is written out as the expression and its result. [[circle "#### 42"]] And at the very end, after four hash signs, is the final answer, which is what we'll use for grading.

So here's the plan. These calculator annotations are a gift. They're exactly the tool calls that we want our model to learn. So we're going to turn every one of them into a python part, followed by a python output part. Step one, let's split the answer into the text and the annotations. A regex split with a capture group keeps the separators, so we get both.

```python
import re
chunks = re.split(r'(<<[^>]+>>)', row["answer"])
print(chunks)
```

[[circle "'<<12*2=24>>'"]] So we get the plain text, then the annotation, then text again, alternating. Step two, for every annotation, we strip the angle brackets and split at the last equals sign. Left of it is the expression, which becomes a python part, and right of it is the result, which becomes a python output part. Everything else is a text part. And then we can render it with our own render_conversation and look at the mask.

```python
parts = []
for chunk in chunks:
    if chunk.startswith('<<') and chunk.endswith('>>'):
        expr, _, result = chunk[2:-2].rpartition('=')
        parts.append({"type": "python", "text": expr})
        parts.append({"type": "python_output", "text": result})
    else:
        parts.append({"type": "text", "text": chunk})
math_conversation = {"messages": [{"role": "user", "content": row["question"]},
                                  {"role": "assistant", "content": parts}]}
print(visualize(*render_conversation(math_conversation)))
```

[[circle "<|python_start|>|12|*|2|<|python_end|>"]] So the model is trained to write the expression, in green, [[circle "<|output_start|>|24|<|output_end|>"]] the result is red, it will come from python, [[underline "|24| pages| today|"]] and then the model copies the result into its sentence. This is the simplest possible version of tool use. The model learns to hand the arithmetic, which it's bad at, to a calculator, which is perfect at it. [[circle "|####| 42|"]] And at the end, the four hash signs and the answer. Now to grade an answer, we just pull out the number after the hash signs, with a regex, and compare it to the reference.

```python
ANSWER_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
def extract_answer(completion):
    match = ANSWER_RE.search(completion)
    return match.group(1).strip().replace(",", "") if match else None

print(extract_answer(row["answer"]), extract_answer("so the answer is #### 1,024"), extract_answer("no idea"))
```

[[circle "42 1024 None"]] So 42 for our problem, commas get removed so 1,024 becomes 1024, and None when there's no answer at all. That's it, that's our grader. And notice that this grader is going to be our reward in reinforcement learning, at the end of the lecture.

The second dataset is MMLU, which is multiple choice questions. And it teaches the model the format of multiple choice, which is how we'll evaluate it later. Let's write the function that renders a multiple choice question, and I'll show you two little details in it.

```python
def render_mc(question, letters, choices):
    query = f"Multiple Choice question: {question}\n"
    query += "".join(f"- {choice}={letter}\n" for letter, choice in zip(letters, choices))
    query += "\nRespond only with the letter of the correct answer."
    return query

print(render_mc("What is the capital of France?", "ABCD", ["Berlin", "Madrid", "Paris", "Rome"]))
print(enc.encode_ordinary(" A"), enc.encode_ordinary("A"), enc.encode_ordinary("=A"))
```

[[circle "- Paris=C"]] The first detail is that the letter comes after the choice, not before it. Big models don't care, but for small models it helps to have the letter right next to the thing it labels. [[note "- Paris=C" "no space before the letter"]] The second detail is that there's no space between the equals sign and the letter. Why? Look at the tokens. [[circle "[317]"]] "Space A" is token 317, [[circle "[32]"]] and "A" without a space is token 32. These are completely different tokens as far as the model is concerned, just different integers. The assistant answers with just "A", token 32. [[note "[28, 32]" "the same token 32"]] So in the prompt, we want the exact same token 32 to appear right next to the choice, so the model can sort of point at it. If you remember the tokenization video, so much weirdness in LLMs comes from the tokenizer, and this is one of those details.

The third dataset is SmolTalk, from HuggingFace. That's 460,000 general conversations, and it's where most of the personality of the assistant comes from.

```python
from tasks import SmolTalk
smoltalk = SmolTalk("train")
for message in smoltalk[0]["messages"]:
    print(f"--- {message['role']}:\n{message['content'][:400]}")
```

[[point "--- assistant:"]] These were mostly generated by bigger language models, and then filtered. And it's worth pausing on this, because Karpathy makes a really important point about it in his talks. [[erase]] [[write "who are you talking to?"]] When you talk to ChatGPT, who are you actually talking to? [[write "→ a simulation of the SFT data"]] You're talking to a statistical simulation of this kind of data. [[write "labelers + labeling instructions, or other LLMs"]] Originally it was written by human labelers following long instructions from a company about being helpful and harmless, and nowadays a lot of it is written by other LLMs. [[write "knowledge ← pretraining,  personality ← SFT"]] But either way, the knowledge comes from pretraining, and the personality, the tone, the length of the answers, all comes from imitating this data.

And the fourth dataset is my favorite. How many r's are there in strawberry? Famously, LLMs get this wrong, and now we can understand why. Look at how it's tokenized.

> **Predict:** How many tokens is " strawberry", with a space in front? And "strawberry" without the space?

```python
print([enc.decode([t]) for t in enc.encode_ordinary(" strawberry")])
print([enc.decode([t]) for t in enc.encode_ordinary("strawberry")])
```

[[circle "[' strawberry']"]] With a space in front, strawberry is a single token. One integer. The model never sees the letters! To count the r's, it would have to have memorized the spelling of every one of its fifty thousand tokens, and then count in its head, in a single forward pass. [[circle "['st', 'raw', 'berry']"]] And without a space, it's three tokens, which is a completely different set of integers again.

So nanochat has a synthetic task for this, called SpellingBee. Synthetic means we generate the data ourselves, as much as we want. And the interesting part is designing the ideal answer. We know the model can't do this in one step, so we write out a procedure that it can do: spell the word out, then go through the letters one at a time and keep a running count, then double check with python, and only then give the answer. Let's write it.

```python
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

for part in spelling_answer("strawberry", "r"):
    print(f"[{part['type']}] {part['text']}")
```

[[circle "strawberry:s,t,r,a,w,b,e,r,r,y"]] First, the spelling. [[circle "9:r hit! count=3"]] Then the counting, one letter per line, with a running count, so that no single token ever needs much computation. [[circle "[python] 'strawberry'.count('r')"]] Then python checks it, [[circle "#### 3"]] and the answer goes after the hash signs, so the same grader as GSM8K works. And I think this is a really nice example of how you have to think about the cognition of the model. Every token only gets a fixed amount of compute, one forward pass through twelve layers. So you want to spread the computation out over many tokens. If you ask the model to blurt out the answer in the first token, you're asking it to do all of the work in one forward pass, and it just can't. [[open tasks.py "class SpellingBee(Task):"]] In the repo, SpellingBee is this same function, plus a random word out of a list of 370,000 English words, a random letter, and a bunch of different ways of asking the question.

Okay, and the last piece is to mix all of these together. A mixture is just a big list of every example in every task, shuffled with a fixed seed. And if you want more of something, you just put it in twice.

```python
import random
from tasks import MMLU, GSM8K, SpellingBee

class TaskMixture:
    def __init__(self, tasks):
        self.tasks = tasks
        self.index_map = [(t, i) for t, task in enumerate(tasks) for i in range(len(task))]
        random.Random(42).shuffle(self.index_map) # mix the tasks together, the same way every time
    def __len__(self):
        return len(self.index_map)
    def __getitem__(self, index):
        t, i = self.index_map[index]
        return self.tasks[t][i]

mixture = TaskMixture([SmolTalk("train", stop=20000), MMLU("auxiliary_train", stop=20000),
                       GSM8K("train"), GSM8K("train"), SpellingBee(size=20000, split="train")])
print(len(mixture))
print([type(mixture.tasks[t]).__name__ for t, _ in mixture.index_map[:8]])
```

[[circle "74946"]] So about 75,000 conversations, [[underline "['SpellingBee', 'SpellingBee', 'GSM8K', 'SmolTalk', 'GSM8K', 'SmolTalk', 'MMLU', 'MMLU']"]] and the tasks are nicely interleaved. Now, I'm using small slices of SmolTalk and MMLU here, because we'll train on a CPU. In nanochat, on a GPU, it's all 460,000 SmolTalk conversations, and [[open chat_sft.py "train_dataset = TaskMixture(["]] here's the mixture in our chat_sft script. And remember what this mixture looks like, because it's going to matter later. Twenty thousand SpellingBee examples is more than a quarter of everything.

> **Exercise:** Write your own synthetic task. For example, a task that teaches the model its own name and who made it, or one that reverses a word letter by letter. Generate the ideal answer with code, and think about how to spread the work across tokens. And exercises four to seven in the notebook are the four pieces of this chapter: the GSM8K parser, the grader, multiple choice, and the SpellingBee answer.

## The SFT dataloader

```whiteboard
@stream text 40 40 "pretraining: one endless stream of text, chopped into rows" 38
@stream fill 40 100 1380 55 gray "... doc doc doc | doc doc | doc doc doc doc | doc ..." 28
@chop text 40 190 "SFT: a conversation must stay in one piece, and start with BOS" 36 red
@naive text 40 280 "naive: one conversation per row, pad the rest" 36
@naive fill 200 330 300 50 blue "conv A" 28
@naive fill 500 330 820 50 gray "padding" 28
@naive fill 200 390 200 50 green "B" 28
@naive fill 400 390 920 50 gray "padding" 28
@naive fill 200 450 460 50 orange "C" 28
@naive fill 660 450 660 50 gray "padding" 28
@packed text 40 560 "best fit: always grab the longest conversation that still fits" 36
@packed fill 200 610 460 50 orange "C" 28
@packed fill 660 610 300 50 blue "A" 28
@packed fill 960 610 200 50 green "B" 28
@packed fill 1160 610 160 50 gray "(pad)" 28
@packed fill 200 670 600 50 purple "D" 28
@packed fill 800 670 420 50 red "E" 28
@packed fill 1220 670 100 50 gray "" 28
@packed fill 200 730 1000 50 blue "F" 28
@packed fill 1200 730 120 50 gray "" 28
```

Okay, so now we need to turn conversations into batches. And in the pretraining videos this was trivial. [[draw stream]] We had one endless stream of text, and we just chopped it into rows of 1024 tokens. A document cut in half, who cares. [[draw chop]] But a conversation is different. If you cut it in half, you get a question without its answer, and an answer without its question. So we want every conversation in one piece, and we want every row to start with a BOS token, because that's what the model will see at test time.

So the first question is, how long are conversations? Let's measure.

```python
lengths = [len(tokenizer.render_conversation(mixture[i], max_tokens=10**6)[0]) for i in range(2000)]
lengths.sort()
print(f"median {lengths[1000]}, 90th percentile {lengths[1800]}, max {lengths[-1]}")
print(f"fraction longer than 512 tokens: {sum(l > 512 for l in lengths) / len(lengths):.3f}")
```

[[circle "median 217"]] Half of them are shorter than about 200 tokens. [[circle "90th percentile 1072"]] But there's a long tail, ten percent are over a thousand tokens, [[note "max 20317" "!!"]] and the longest is 20,000. [[circle "fraction longer than 512 tokens: 0.202"]] And with the rows of 512 tokens we're going to use on the CPU, one in five conversations won't fit. Keep that number in mind.

[[draw naive]] Okay so the simplest thing we could do is one conversation per row, and fill up the rest of the row with padding. And we don't train on the padding, so it gets a mask of zero. Let's measure how much of the batch is wasted.

> **Predict:** Half the conversations are shorter than about 200 tokens, and the rows are 512. What fraction of a naive batch is padding?

```python
def naive_batch(dataset, start, batch_size, seq_len):
    rows, masks = [], []
    for i in range(start, start + batch_size):
        ids, mask = tokenizer.render_conversation(dataset[i], max_tokens=seq_len + 1)
        pad = seq_len + 1 - len(ids)
        rows.append(ids + [tokenizer.bos] * pad)
        masks.append(mask + [0] * pad)
    return torch.tensor(rows), torch.tensor(masks)

rows, masks = naive_batch(mixture, 0, 64, 512)
print(f"padding: {(rows == tokenizer.bos).float().mean():.3f}, trained on: {masks.float().mean():.3f}")
```

[[circle "padding: 0.473"]] 47 percent of every batch is padding. Half of our compute would be spent on nothing. [[circle "trained on: 0.255"]] And only a quarter of the tokens are actually trained on. That's terrible.

[[draw packed]] So the better idea is packing. We put several whole conversations into every row. And the algorithm nanochat uses is called best fit. We keep a buffer of a hundred conversations, and to fill a row, we keep grabbing the longest conversation in the buffer that still fits in the space that's left. [[circle "(pad)"]] And only when nothing fits anymore, we pad the rest of the row. Long conversations go first, and the short ones fill in the gaps. [[note "(pad)" "never crop!"]] And we never cut a conversation.

But before I show you the code, I want to tell you about a bug I hit writing this, because it's really instructive. In my first version, I rendered every conversation with max tokens set to the row length, so long conversations just got truncated. Seems harmless, right? Let's see what that does to a long multiple choice question.

> **Predict:** A long multiple choice question gets cut off at 513 tokens. How many of its tokens will we train on?

```python
long_mcq = next(c for c in MMLU("auxiliary_train") if len(tokenizer.render_conversation(c, max_tokens=10**6)[0]) > 600)
ids, mask = tokenizer.render_conversation(long_mcq, max_tokens=513)
print(len(ids), sum(mask))
import torch.nn.functional as F
logits = torch.randn(1, 4, 50304)
print(F.cross_entropy(logits.view(-1, 50304), torch.tensor([-1, -1, -1, -1]), ignore_index=-1))
```

[[circle "513 0"]] 513 tokens, and a mask that sums to zero. Not a single trainable token! These MMLU training questions include long reading passages, and the answer, the one green token, is at the very end. So truncation cut off the answer, and the whole conversation became prompt. [[note "513 0" "nothing to train on"]] And here's the evil part. The best fit algorithm loves these, because they're exactly the length of a row. [[circle "tensor(nan)"]] So it filled whole batches with them, and the loss on a batch with zero trainable tokens is zero divided by zero, which is not a number. My loss was NaN, and it took me a minute to figure out why. So the fix is simple: if a conversation doesn't fit in a row, we skip it. Okay, now let's write the real thing.

```python
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
        yield inputs, targets, cursor / len(dataset)

loader = sft_data_generator(mixture, batch_size=64, seq_len=512)
x, y, _ = next(loader)
x, y, _ = next(loader)
print(x.shape, y.shape)
print(f"padding: roughly {(x == tokenizer.bos).float().mean():.3f}, trained on: {(y >= 0).float().mean():.3f}")
```

So let's read it. [[point "row_capacity = seq_len + 1"]] The rows have one extra token, because the targets are the inputs shifted by one. [[circle "if len(ids) <= row_capacity:"]] Here's the fix for my bug, we skip conversations that are too long. [[circle "ids, mask = buffer.pop(max(fits, key=lambda i: len(buffer[i][0])))"]] Here's best fit, the longest one that fits. [[circle "mask_rows.append(mask_row + [0] * padding)"]] The padding gets a mask of zero. [[highlight "inputs, targets = batch[:, :-1], batch[:, 1:].clone()"]] Then inputs and targets, exactly like in pretraining. [[circle "targets[mask[:, 1:] == 0] = -1"]] And wherever the mask is zero, we set the target to minus one. [[open gpt.py "ignore_index=-1"]] And minus one is the ignore index of the cross entropy loss in our model. So those positions simply don't count.

[[notebook]] [[circle "padding: roughly 0.081"]] With packing, the padding goes down to about 8 percent, and that even includes the BOS tokens at the start of every conversation. [[circle "trained on: 0.419"]] And 42 percent of the tokens are trained on, instead of 25. That's almost twice as much signal from the same compute. Let's do one more sanity check, on our tiny France conversation, and look at the inputs and targets side by side.

```python
small = sft_data_generator([conversation], batch_size=1, seq_len=12, buffer_size=1)
x, y, _ = next(small)
print([tokenizer.decode([t]) for t in x[0].tolist()])
print(y[0].tolist())
```

[[underline "['<|endoftext|>', '<|user_start|>', 'What', ' is', ' the', ' capital', ' of', ' France', '?', '<|user_end|>', '<|assistant_start|>', 'Paris']"]] So the inputs are the conversation without its last token. [[circle "[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1,"]] And the targets are minus one everywhere, [[circle "40313, 50260]"]] except at the last two positions. After assistant start, the target is Paris, and after Paris, the target is assistant end. Those are the only two predictions that this conversation trains.

> **Exercise:** Change the dataloader so that a conversation that's too long gets cropped in a smarter way, keeping the last assistant answer, instead of being skipped. Is it worth it? Measure how many trained tokens you gain. But first, do exercise eight in the notebook: write the best fit dataloader from scratch. The check will catch a cropped conversation, a wrong target, or too much padding.

## The SFT training loop

Okay, and now the punchline of this whole stage. SFT is literally just pretraining with a different dataloader. Same model, same cross entropy loss, same AdamW optimizer, same training loop. The only differences are the minus ones in the targets, and a gentler learning rate. [[write "SFT = pretraining + a different dataloader"]] The model already knows a lot, we're just nudging its statistics, so we use a smaller learning rate than in pretraining, keep it constant for the first half, and then decay it to zero.

```python
def get_lr(progress, lr=1e-4, warmdown_ratio=0.5):
    if progress < 1 - warmdown_ratio:
        return lr
    return lr * (1 - progress) / warmdown_ratio

print([f"{get_lr(p):.1e}" for p in [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]])
```

[[circle "'1.0e-04', '1.0e-04', '1.0e-04'"]] Constant for the first half, [[circle "'5.0e-05', '2.0e-05', '0.0e+00'"]] then down to zero at the end. That's the schedule nanochat uses. And now, let's actually train, right here in the notebook. A fresh copy of GPT-2, AdamW, our dataloader with small rows of 256 tokens, and the training loop you've written ten times by now.

```python
torch.manual_seed(1337)
student = GPT.from_pretrained()
optimizer = student.configure_optimizer(lr=1e-4)
loader = sft_data_generator(mixture, batch_size=4, seq_len=256)
for step in range(8):
    x, y, _ = next(loader)
    loss = student(x, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    print(f"step {step} | loss {loss.item():.4f} | trained tokens {(y >= 0).sum().item()}")
del student, optimizer
```

[[circle "step 0 | loss 3.3616"]] And the loss is going down. [[circle "step 7 | loss 1.6392"]] From 3.4 to 1.6 in eight steps. [[point "trained tokens 243"]] And notice how the number of trained tokens jumps around from batch to batch, that's because every batch has a different mix of conversations, so the loss of a single step is noisy.

So that's SFT. Everything else is engineering. [[open chat_sft.py "def sft_data_generator"]] I've put the dataloader, [[open chat_sft.py "def get_lr(progress):"]] the schedule and the loop into chat_sft.py, plus command line arguments, a validation loss every hundred steps, gradient accumulation, bfloat16 and torch compile on a GPU, and saving the checkpoint at the end. In nanochat, you do one full epoch over the whole mixture, which on a single A100 is something like half an hour. [[open chat_sft.py "CPU (a quick demo"]] But to show you that you don't need a GPU, I ran a smaller version on a plain four core CPU. 600 steps of four rows of 512 tokens, about an hour and a half.

```bash
python chat_sft.py --device-batch-size=4 --seq-len=512 --grad-accum=1 --num-iterations=600 --smoltalk=20000 --mmlu=20000
```

```output
Training on 74,946 conversations
step     0 | val loss 2.7305
step     1 | progress 0.0% | loss 11.8539 | lr 1.00e-04 | trained tokens 8 | dt 11.71s
step     2 | progress 0.2% | loss 2.6189 | lr 1.00e-04 | trained tokens 428 | dt 10.86s
step     3 | progress 0.3% | loss 3.1415 | lr 1.00e-04 | trained tokens 378 | dt 11.22s
step     4 | progress 0.5% | loss 8.2122 | lr 1.00e-04 | trained tokens 12 | dt 10.30s
step     5 | progress 0.7% | loss 3.7664 | lr 1.00e-04 | trained tokens 380 | dt 10.38s
...
step   100 | val loss 1.6287
step   200 | val loss 1.4900
step   300 | val loss 1.4400
step   400 | val loss 1.4042
step   500 | val loss 1.3579
...
step   600 | progress 99.8% | loss 0.7444 | lr 3.33e-07 | trained tokens 750 | dt 7.81s
step   600 | val loss 1.3377
Saved the SFT model to /root/.cache/nanochat-posttrain/chatsft.pt
```

Now look at the very first step. [[circle "loss 11.8539"]] A loss of 11.9. Why so much higher than in our notebook run? [[circle "trained tokens 8 |"]] Look at the number of trained tokens. Eight. In the whole first batch!

> **Predict:** Why would the very first batch have only eight tokens to train on? Think about what best fit grabs first.

It's best fit. At the start, the buffer is full, and the longest conversations in it are those long MMLU reading passages. So every row is one of them, and each one has exactly two green tokens, the letter and assistant end. And assistant end is one of our special tokens, the ones that start out with a loss of 15. [[circle "trained tokens 428"]] Then in step two there are 428 trained tokens, and the loss is a normal 2.6. So the loss of a single step can be really misleading, [[circle "val loss 2.7305"]] and that's why we look at the validation loss, [[circle "val loss 1.3377"]] which goes down smoothly, from 2.73 to 1.34.

Okay, now the fun part. Let's talk to it. Here's a little chat function: it puts the question in a conversation, renders it for completion, and samples until assistant end.

```python
from gpt import load_model
device = "cpu"
sft_model = load_model("sft", device).eval()
sft_engine = Engine(sft_model, tokenizer)

def chat(question, engine=sft_engine, temperature=0.0, seed=42, max_tokens=100):
    conversation = {"messages": [{"role": "user", "content": question}, {"role": "assistant", "content": ""}]}
    ids = tokenizer.render_for_completion(conversation)
    out, _ = engine.generate_batch(ids, max_tokens=max_tokens, temperature=temperature, top_k=50, seed=seed)
    return tokenizer.decode(out[0][len(ids):])

print(chat("Hi! Who are you?", temperature=0.7, seed=1))
```

[[circle "Hello! How can I help you today?"]] "Hello! How can I help you today?" And then it stopped. [[note "Hello! How can I help you today?" "it stopped by itself!"]] All by itself, it emitted assistant end. Compare that to the base model, which kept going forever. So the format, the turn taking, that's learned. Let's ask about France.

```python
print(chat("What is the capital of France?"))
```

[[circle "The capital of France is Paris, France."]] It answers directly now, as an assistant. [[underline "the capital of France, the capital of France, the capital of France"]] But then it gets stuck in a loop. This is a classic failure of greedy decoding with small models: once it has said "the capital of France" twice, the most likely thing is to say it again. Sampling with a temperature helps, so let's try that.

```python
print(chat("What is the capital of France?", temperature=0.7, seed=2, max_tokens=150))
```

Ha! Look at this. [[circle "We are asked to find the capital of France."]] "We are asked to find the capital of France. Let me try a manual approach first." [[circle "France:f,a,l,e,r,n,e,s,t"]] And then it spells France, badly, [[point "1:f hit! count=1"]] counts the letter f, [[circle "'f'.count('f')"]] and calls python to double check. It's running the SpellingBee procedure on a geography question! Why? Remember the mixture. [[open chat_sft.py 'SpellingBee(size=20000, split="train"),']] SpellingBee is more than a quarter of everything this model saw. And a lot of SpellingBee questions are short questions, like "what is the count of e in some word". So "what is the capital of France" looks like one of those on the surface, and the model pattern matches on the surface. If I sample answers to thirty real questions from the SmolTalk test set, only two start like this, so it's mostly short questions. But it's a great lesson. Every dataset in the mixture shapes the model, in ways that leak out beyond that dataset. When Karpathy added SpellingBee to nanochat, he noted in the commit that it probably uses too many of these examples. [[notebook]] And on a GPU, with all of SmolTalk in the mixture, this mostly goes away.

```python
print(chat("Why is the sky blue?", temperature=0.7, seed=1))
```

[[circle "the meteoroid and the meteorite"]] And this one is kind of hilarious, meteoroids and [[circle "molten metal"]] molten metal. So let's be honest about what we have. This is a 124 million parameter model from 2019, finetuned for an hour and a half on a CPU. It doesn't know why the sky is blue, and it's happy to make things up with total confidence. The nanochat README says talking to its speedrun model is like talking to a kindergartener, and we're well below that. But the format is there, the turn taking is there, and as we'll see, on the narrow tasks we specifically trained for, it can do quite well.

> **Exercise:** Re-run SFT with five thousand SpellingBee examples instead of twenty thousand, and measure how often the model starts an ordinary answer with "We are asked to find". What happens to its SpellingBee accuracy? And exercise nine is the learning rate schedule.

## The inference engine

```whiteboard
@seq text 40 40 "sampling: one new token at a time" 42
@seq fill 40 120 160 60 blue "The" 30
@seq fill 210 120 160 60 blue "meaning" 30
@seq fill 380 120 160 60 blue "of" 30
@seq fill 550 120 160 60 blue "life" 30
@seq fill 720 120 160 60 blue "is" 30
@seq fill 890 120 160 60 orange "???" 30
@naive text 40 230 "naive: push ALL the tokens through the model again, every step" 34 red
@naive text 40 280 "→ the total work grows like T^2" 34 red
@cache text 40 380 "KV cache: remember every layer's keys and values" 34 green
@cache fill 40 440 840 70 green "cached keys and values (computed once)" 30
@cache fill 890 440 160 70 orange "new: q, k, v" 30
@cache arrow 900 540 500 540 green
@cache text 40 570 "the one new query attends to everything in the cache" 34
@prefill text 40 680 "many samples: prefill the prompt once, then copy the cache" 34 purple
```

Okay, so far we've been using the Engine to sample, as a black box. Now let's write it. Sampling has two problems that we need to solve: it needs to be fast, because in reinforcement learning we're going to sample a lot, and it needs to run the python tool. Let's start with speed. Here's the simplest sampling loop you could write, the one from the GPT-2 video.

```python
import time
@torch.no_grad()
def generate_naive(model, prompt, max_tokens):
    ids = torch.tensor([prompt])
    for _ in range(max_tokens):
        logits = model(ids)[:, -1, :] # run the WHOLE sequence through the model, every time
        ids = torch.cat([ids, logits.argmax(dim=-1, keepdim=True)], dim=1)
    return ids[0].tolist()

prompt = [tokenizer.bos] + tokenizer.encode("The meaning of life is")
t0 = time.time()
naive = generate_naive(model, prompt, 100)
print(f"naive: {time.time() - t0:.2f}s")
```

[[circle "naive:"]] Several seconds for 100 tokens. Why is it slow? [[draw seq]] Think about what this loop does. To get token number 7, it runs tokens 1 through 6 through the model. Then to get token 8, it runs tokens 1 through 7 through the model, [[draw naive]] all of them, again. But nothing about tokens 1 through 6 has changed! Because of the causal mask, a token can only look at tokens before it, so its keys and values in every layer are exactly the same as last time. We're recomputing them over and over, and the total work grows like the square of the length.

[[draw cache]] So the fix is to remember them. That's the KV cache. For every layer, we keep the keys and values of all the tokens so far, and when a new token comes in, we only compute the query, key and value for that one token, append its key and value to the cache, [[point "new: q, k, v"]] and let its query attend to everything in the cache. Now, this needs three small changes inside our GPT, and you should make them in your own gpt.py from the GPT-2 video. [[open gpt.py "class KVCache:"]] Change one is the cache itself. [[circle "def update(self, k, v):"]] It's a tiny class that holds the keys and values, and update appends the new ones and returns everything. [[open gpt.py "k, v = kv_cache.update(k, v)"]] Change two is in the attention. [[circle "k, v = kv_cache.update(k, v)"]] If there's a cache, we append and get back all the keys and values. [[circle "is_causal=T > 1"]] And here's a subtle one.

> **Predict:** When we feed in just one new token, and everything else is in the cache, should the attention use a causal mask?

If we feed in a single new token, we don't want a causal mask at all, because that one token is allowed to look at everything in the cache. [[open gpt.py "t0 = 0 if kv_cache is None"]] And change three, the positions. [[circle "pos = torch.arange(t0, t0 + T"]] The position embedding of the new token must start where the cache left off, otherwise every new token would think it's at position zero. And with that, here's sampling with a cache.

```python
from gpt import KVCache

@torch.no_grad()
def generate_cached(model, prompt, max_tokens):
    kv_cache = [KVCache() for _ in range(model.config.n_layer)]
    logits = model(torch.tensor([prompt]), kv_cache=kv_cache)[:, -1, :] # prefill: the whole prompt, once
    out = list(prompt)
    for _ in range(max_tokens):
        token = logits.argmax(dim=-1).item()
        out.append(token)
        logits = model(torch.tensor([[token]]), kv_cache=kv_cache)[:, -1, :] # decode: one token at a time
    return out, kv_cache

t0 = time.time()
cached, kv_cache = generate_cached(model, prompt, 100)
print(f"cached: {time.time() - t0:.2f}s, same tokens: {cached[:len(naive)] == naive}")
print(kv_cache[0].k.shape)
```

[[point "logits = model(torch.tensor([prompt]), kv_cache=kv_cache)[:, -1, :]"]] First we run the whole prompt through once, which fills the cache. That's called the prefill. [[point "logits = model(torch.tensor([[token]]), kv_cache=kv_cache)[:, -1, :]"]] And then every step, we only feed in the one new token. [[circle "same tokens: True"]] Exactly the same tokens as the naive version, which is the first thing to always check, [[circle "cached:"]] and about four times faster. And that gap grows with length. [[circle "torch.Size([1, 12, 106, 64])"]] And here's the cache of the first layer: batch of one, twelve heads, 106 positions, which is our 6 prompt tokens plus 100 new ones, and 64 dimensions per head.

[[draw prefill]] One more trick, which we'll need for reinforcement learning. Often we want many samples for the same prompt. So we do the prefill once, with a batch size of one, and then we just copy the cache for every sample, and decode all of them in parallel, as one batch.

Okay, now the second job, tools. The model is sampling one token at a time. At some point it emits python start, then some tokens, then python end.

> **Predict:** What should happen when the model emits python end, and what should the model see next?

Here's the idea. When we sample python start, we start recording the tokens. When we sample python end, we decode the recorded tokens into a string, and we run it. And then here's the trick. We put output start, the tokens of the result, and output end into a queue, and on the next steps, instead of using what the model sampled, we take tokens from the queue. We force them in. So the result appears in the model's own context, as if it had written it, and then it just continues from there. Let's write it.

```python
from collections import deque
from engine import use_calculator

@torch.no_grad()
def generate_with_tools(model, prompt, max_tokens=256):
    s = tokenizer.encode_special
    kv_cache = [KVCache() for _ in range(model.config.n_layer)]
    logits = model(torch.tensor([prompt]), kv_cache=kv_cache)[:, -1, :]
    out, forced, expr = [], deque(), None
    for _ in range(max_tokens):
        token = forced.popleft() if forced else logits.argmax(dim=-1).item()
        out.append(token)
        if token == s("<|assistant_end|>"):
            break
        if token == s("<|python_start|>"):
            expr = [] # start recording the expression
        elif token == s("<|python_end|>") and expr is not None:
            result = use_calculator(tokenizer.decode(expr)) # run the tool...
            if result is not None: # ...and force its output into the stream
                forced.extend([s("<|output_start|>"), *tokenizer.encode(str(result)), s("<|output_end|>")])
            expr = None
        elif expr is not None:
            expr.append(token)
        logits = model(torch.tensor([[token]]), kv_cache=kv_cache)[:, -1, :]
    return out

question = {"messages": [{"role": "user", "content": "How many u are in the word eurytherm"}, {"role": "assistant", "content": ""}]}
print(tokenizer.decode(generate_with_tools(sft_model, tokenizer.render_for_completion(question))))
```

[[circle "token = forced.popleft() if forced else logits.argmax(dim=-1).item()"]] So every step, if there are forced tokens waiting, we take one, otherwise we use the model's choice. [[circle "expr = [] # start recording the expression"]] On python start we start recording, [[circle "result = use_calculator(tokenizer.decode(expr)) # run the tool..."]] on python end we run it, [[circle 'forced.extend([s("<|output_start|>"), *tokenizer.encode(str(result)), s("<|output_end|>")])']] and we queue up the result. And let's try it on our SFT model, on a spelling question it has never seen. [[circle "eurytherm:e,y,u,r,e,r,m,e,r"]] So look at what happens. It spells the word, and it gets the spelling wrong! e, y, u, r. It roughly knows which letters are in there, but not in which order, because remember, it never sees the letters. [[circle "3:u hit! count=1"]] Then it counts the u's in its own misspelled version and gets one, which happens to be right. [[circle "<|python_start|>'eurytherm'.count('u')<|python_end|>"]] Then it writes the python call, and in the python call it copies the word perfectly, because copying tokens is easy. [[note "<|output_start|>1<|output_end|>" "forced in by us"]] And our code runs it and forces in the result. [[circle "#### 1<|assistant_end|>"]] And the final answer is one. So the tool is robust to the model's own sloppiness.

Now, a word about safety. [[open engine.py "def use_calculator(expr):"]] We're running code that a neural network wrote, so the calculator is deliberately very restricted. [[highlight 'is_math = all(c in "0123456789*+-/.() " for c in expr) and "**" not in expr']] It allows numbers and arithmetic, without powers, [[highlight '".count(" in expr']] or the count method of a string, [[circle '{"__builtins__": {}}']] and it evaluates without any builtins.

```python
print(use_calculator("2*12"), use_calculator("'strawberry'.count('r')"))
print(use_calculator("__import__('os').system('ls')"), use_calculator("9**9**9"))
```

[[circle "24 3"]] So arithmetic and counting work, [[circle "None None"]] and an attempt to run a shell command, or to compute a gigantic number, just returns None. Never give a language model an unrestricted eval on your machine.

[[open engine.py "class Engine:"]] And the Engine in the repo is exactly what we just wrote, [[point "for i in range(num_samples):"]] but for a whole batch of samples at once, with every row keeping its own little state, [[circle "c.expand(num_samples)"]] the prefill once and copy trick, [[circle "masks.append(0 if is_forced else 1)"]] and one more thing: it returns a mask, one for tokens the model sampled, and zero for tokens that were forced in. Remember that mask, because we'll need it for reinforcement learning. Let's sample eight answers to the same question.

```python
example = SpellingBee(size=10, split="test")[6]
ids = tokenizer.render_for_completion(example)
out, masks = sft_engine.generate_batch(ids, num_samples=8, max_tokens=256, temperature=1.0, top_k=50, seed=0)
print(example["messages"][0]["content"])
for o, m in zip(out, masks):
    completion = tokenizer.decode(o[len(ids):])
    print(f"answer {extract_answer(completion)} | {len(o) - len(ids)} tokens, {m[len(ids):].count(0)} forced in by the tool")
```

[[underline "answer 1 | 154 tokens, 3 forced in by the tool"]] Every sample has three forced tokens, output start, the result, and output end, [[circle "172 tokens"]] and they come out with different lengths, because we sampled with a temperature.

> **Exercise:** Add the three KV cache changes to your own GPT from the GPT-2 video, and check that your cached sampling gives exactly the same tokens as naive sampling. Then make the cache faster: instead of concatenating tensors every step, allocate the whole cache up front and write into it. And exercises ten to thirteen in the notebook take you through the whole engine: the cache, cached sampling, the safe calculator, and tool use, checked against a fake model that tries to hallucinate the tool's output.

## Evaluation

Okay, so we have a model that seems to work. But "seems to work" is not good enough, we need numbers. And there are two kinds of evaluation in nanochat.

The first is multiple choice, like MMLU. And here's a neat thing: we don't even have to sample. We render the question, prime the assistant, and run one forward pass. Then we look at the logits at the last position, which is the model's prediction for the first token of its answer, but only at the logits of the allowed letters, A, B, C and D. Whichever is biggest, that's the answer. Let's write it, and compare the base model and our SFT model on 100 questions.

> **Predict:** Four choices per question. What accuracy do you expect from GPT-2, and from our SFT model?

```python
@torch.no_grad()
def multiple_choice_accuracy(model, task, n):
    correct = 0
    for i in range(n):
        conversation = task[i]
        ids = tokenizer.render_for_completion(conversation)
        logits = model(torch.tensor([ids]))[0, -1] # the logits for the assistant's first token
        letter_ids = [tokenizer.encode(letter)[0] for letter in conversation["letters"]]
        predicted = conversation["letters"][logits[letter_ids].argmax().item()]
        correct += predicted == conversation["messages"][-1]["content"]
    return correct / n

mmlu_test = MMLU("test")
print(f"base: {multiple_choice_accuracy(model, mmlu_test, 100):.2f}, sft: {multiple_choice_accuracy(sft_model, mmlu_test, 100):.2f}")
```

[[circle "logits = model(torch.tensor([ids]))[0, -1] # the logits for the assistant's first token"]] One forward pass, the logits at the last position, [[circle "letter_ids = [tokenizer.encode(letter)[0] for letter in conversation[\"letters\"]]"]] restricted to the letters. [[circle "base: 0.29, sft: 0.26"]] And both are about 25 percent, which is exactly random guessing for four choices. And with only 100 questions, the noise on these numbers is about plus or minus five percent, so 29 versus 26 is not a real difference. Now this is important to understand. GPT-2 small simply doesn't know the answers to these questions. SFT can teach a format, but it can't teach knowledge that isn't there. For MMLU to move, you need a much stronger pretrained model, which is a big part of why nanochat trains its own base models.

The second kind is generative. We sample an answer, pull out the number after the hash signs, and compare. Let's do 20 SpellingBee questions the model has never seen.

```python
bee_test = SpellingBee(size=20, split="test")
correct = 0
for i in range(20):
    conversation = bee_test[i]
    ids = tokenizer.render_for_completion(conversation)
    out, _ = sft_engine.generate_batch(ids, max_tokens=256, temperature=0.0)
    correct += bee_test.evaluate(conversation, tokenizer.decode(out[0][len(ids):]))
print(f"SpellingBee: {correct}/20")
```

[[circle "SpellingBee: 9/20"]] Nine out of twenty. So about half. That's a narrow, procedural task, and SFT taught the model a procedure that it can actually execute, most of the time. [[open chat_eval.py "def run_categorical_eval"]] In the repo, chat_eval.py has both of these, [[open chat_eval.py "# ChatCORE: accuracy re-scaled"]] plus something nanochat calls ChatCORE: [[circle "(results[t] - BASELINES[t]) / (1 - BASELINES[t])"]] every accuracy is rescaled so that zero means random guessing and one means perfect, and then averaged into one number. Here's what I got on 100 problems of every task.

```bash
python chat_eval.py -i base -x 100
python chat_eval.py -i sft -x 100
```

```output
$ python chat_eval.py -i base -x 100
ARC-Easy: 29.00%
ARC-Challenge: 26.00%
MMLU: 29.00%
GSM8K: 0.00%
SpellingBee: 0.00%
ChatCORE: 0.0240
$ python chat_eval.py -i sft -x 100
ARC-Easy: 24.00%
ARC-Challenge: 26.00%
MMLU: 26.00%
GSM8K: 0.00%
SpellingBee: 53.00%
ChatCORE: 0.1087
```

[[circle "MMLU: 26.00%"]] Multiple choice is at chance for both, as we saw. [[circle "GSM8K: 0.00%"]] GSM8K is zero for both: the SFT model learned what a math solution looks like, with tool calls and hash signs, but it can't do the reasoning. [[circle "SpellingBee: 53.00%"]] And SpellingBee goes from zero to 53 percent. [[circle "ChatCORE: 0.1087"]] So the ChatCORE goes from 0.02 to 0.11, and almost all of that is SpellingBee. And 53 percent is not great, but it's not zero either, and that's exactly where we want to be for the next stage.

> **Exercise:** Evaluate the base model on MMLU with a few-shot prompt, using plain text instead of our chat format. How close can you get to the SFT model just by prompting?

## Reinforcement learning

```whiteboard
@group text 40 40 "one question → a group of 8 attempts" 42
@group fill 60 130 150 70 green "R = 1" 30
@group fill 225 130 150 70 red "R = 0" 30
@group fill 390 130 150 70 green "R = 1" 30
@group fill 555 130 150 70 green "R = 1" 30
@group fill 720 130 150 70 red "R = 0" 30
@group fill 885 130 150 70 green "R = 1" 30
@group fill 1050 130 150 70 red "R = 0" 30
@group fill 1215 130 150 70 red "R = 0" 30
@mean line 50 240 1380 240 gray
@mean text 40 255 "mean reward = 0.5" 34 gray
@adv text 95 320 "+0.5" 40 green
@adv text 260 320 "-0.5" 40 red
@adv text 425 320 "+0.5" 40 green
@adv text 590 320 "+0.5" 40 green
@adv text 755 320 "-0.5" 40 red
@adv text 920 320 "+0.5" 40 green
@adv text 1085 320 "-0.5" 40 red
@adv text 1250 320 "-0.5" 40 red
@up text 40 420 "advantage > 0: push those tokens up (like SFT on them)" 36 green
@down text 40 480 "advantage < 0: push those tokens down" 36 red
@grpo text 40 560 "GRPO = Group Relative Policy Optimization" 36 purple
@zero text 40 660 "all right, or all wrong → every advantage is 0 → nothing to learn" 36 orange
```

Okay, and now the last stage, and I think the most exciting one. Why do we need it? Karpathy has a great analogy. Think of a textbook. [[write "exposition = pretraining"]] Most of it is exposition, background knowledge. That's pretraining. [[write "worked examples = SFT (imitate)"]] Then there are worked examples, where an expert solves a problem step by step, and you imitate them. That's SFT. [[write "practice problems = RL (try, check the answer)"]] And at the end of every chapter, there are practice problems, with the final answer in the back of the book but no solution. You try, you fail, you try again, and you figure out what works for you. That's reinforcement learning.

And here's the deep reason why this matters. [[write "a human's solution ≠ the best one for the model"]] With SFT, we make the model imitate a solution that some person, or some other model, wrote. But that author has a very different brain. A step that's trivial for them might be a leap that our little model can't make in one token. And a step they wrote out carefully might be obvious to our model. We don't actually know the best sequence of tokens for this particular model. [[write "→ let the model try, and do more of what works"]] So the idea of RL is: don't tell it. Let it try many times, and whatever works, do more of that.

So let's make that precise. [[erase]] [[write "question → sample answer y → reward R (1 if correct, else 0)"]] We have a question, the model samples an answer y, and we check it with our grader and get a reward R, one if it's correct and zero if not. [[write "goal: maximize E[ R ] over the answers the model samples"]] And we want to maximize the expected reward, the average reward of the answers the model would sample. Now how do we get a gradient of an average over samples? Here's the trick, and it's worth deriving once. [[write "grad E[R] = sum over y of R(y) * grad p(y)"]] The expected reward is a sum over all possible answers of the probability of the answer times its reward, so its gradient is the sum of the reward times the gradient of the probability. [[write "grad p = p * grad log p"]] Now the gradient of p is the same thing as p times the gradient of log p, that's just the chain rule backwards. [[write "→ grad E[R] = E[ R * grad log p(y) ]"]] And plugging that in, we get an expectation again! The average, over sampled answers, of the reward times the gradient of the log probability. And an expectation we can estimate by sampling. [[write "log p(y) = sum of log p(token) = -(our loss)"]] And the log probability of an answer is the sum of the log probabilities of its tokens, which is just minus the loss our model already computes.

So in code, it's almost embarrassingly simple. Sample some answers, and for each one, take the log probability of its tokens, multiply by its reward, and do gradient ascent on that. [[write "≈ SFT on your own samples, weighted by R"]] If you squint, it's just SFT on your own samples, weighted by how good they were. This is called the policy gradient, or REINFORCE.

But there's one problem. If rewards are zero or one, the wrong answers get a weight of zero, so they're simply ignored. And on easy questions, where every answer is right, everything gets pushed up, even though there's nothing to learn. [[erase]] So what we do is subtract a baseline. [[draw group]] For every question, we sample a group of answers, say eight of them, [[draw mean]] and the advantage of each answer is its reward minus the mean reward of the group. [[draw adv]] So an answer that's better than average for this question has a positive advantage, [[draw up]] and its probability gets pushed up. [[draw down]] An answer that's worse than average gets pushed down. And you can show that subtracting a baseline doesn't change the expected gradient at all, it just makes it a lot less noisy. [[draw grpo]] This is the group in GRPO, group relative policy optimization.

Before we do this on a transformer, let's build intuition on the tiniest possible example. Imagine a language model that produces a single token, out of four options, and only option two is correct. The whole model is four logits. We sample a group of sixteen attempts, reward the correct ones, compute advantages, and take a gradient step.

```python
def reinforce_toy(logits, steps=40):
    torch.manual_seed(0)
    logits = torch.tensor(logits, requires_grad=True)
    optimizer = torch.optim.SGD([logits], lr=1.0)
    for step in range(steps + 1):
        probs = logits.softmax(dim=0)
        samples = torch.multinomial(probs, num_samples=16, replacement=True) # a group of 16 attempts
        rewards = (samples == 2).float()                                     # only option 2 is correct
        advantages = rewards - rewards.mean()
        loss = -(probs.log()[samples] * advantages).mean()                   # the policy gradient loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 10 == 0:
            print(f"step {step:2d} | p(correct) {probs[2].item():.4f} | mean reward {rewards.mean().item():.3f}")

reinforce_toy([0.0, 0.0, 0.0, 0.0])
```

[[circle "p(correct) 0.2500"]] The probability of the correct answer starts at a quarter, because all logits are zero, [[circle "p(correct) 0.8508"]] after ten steps it's at 85 percent, [[circle "p(correct) 0.9801"]] and it keeps climbing towards one. Nobody ever told it the answer. It just tried things and did more of what worked. Now what if the model starts out really bad, and the correct answer is very unlikely?

> **Predict:** The correct answer starts with a logit of minus ten, a probability of about one in sixty thousand. What happens?

```python
reinforce_toy([0.0, 0.0, -10.0, 0.0])
```

[[circle "p(correct) 0.0000 | mean reward 0.000"]] Nothing happens. The correct answer is never sampled, so the reward is always zero, [[note "p(correct) 0.0000 | mean reward 0.000" "never sampled = no signal"]] so every advantage is zero, and there's no gradient. And this is the most important lesson about RL: it can only amplify things the model already does sometimes. It can't create a behavior from nothing. That's why we do SFT first, to get into a regime where the model sometimes succeeds. [[board]] [[draw zero]] And the same is true at the other end: if all eight answers are right, every advantage is zero too. RL only learns from questions at the edge of the model's ability.

Okay, now the real thing, on our SFT model. Step one, rollouts. We take a SpellingBee question, render it for completion, and use our engine to sample a group of eight answers, at temperature one, because we want variety. Then we grade each one.

```python
train_bee = SpellingBee(size=1000, split="train")
conversation = train_bee[6]
prompt = tokenizer.render_for_completion(conversation)
sequences, masks = sft_engine.generate_batch(prompt, num_samples=8, max_tokens=256, temperature=1.0, top_k=50, seed=0)
rewards = [train_bee.reward(conversation, tokenizer.decode(seq[len(prompt):])) for seq in sequences]
print(conversation["messages"][0]["content"])
print(rewards)
```

[[point "What is the count of 'o' in spook?"]] How many o's in spook. [[circle "[1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0]"]] Four right, four wrong. That's a question right at the edge, perfect for learning. Now step two, the policy gradient. We pad the samples to the same length, and build inputs and targets exactly like in SFT. And here's where the mask from the engine comes in: we set the targets to minus one for the prompt, for the padding, and for the tokens that the tool forced in. We only train on the tokens that the model actually chose. Then advantages, the log probabilities, and the loss.

```python
max_length = max(len(seq) for seq in sequences)
pad = tokenizer.encode_special("<|assistant_end|>")
ids = torch.tensor([seq + [pad] * (max_length - len(seq)) for seq in sequences])
mask = torch.tensor([m + [0] * (max_length - len(m)) for m in masks])
inputs, targets = ids[:, :-1], ids[:, 1:].clone()
targets[mask[:, 1:] == 0] = -1 # not the prompt, not the tool outputs, not the padding
rewards = torch.tensor(rewards)
advantages = rewards - rewards.mean()
print(advantages)
logp = -sft_model(inputs, targets, loss_reduction='none').view_as(inputs) # (8, T), 0 where target is -1
print((logp.sum(dim=1) / (targets >= 0).sum(dim=1)).tolist())
loss = -(logp * advantages.unsqueeze(-1)).sum() / (targets >= 0).sum()
loss.backward()
print(loss.item())
sft_model.zero_grad(set_to_none=True)
```

[[circle 'targets[mask[:, 1:] == 0] = -1 # not the prompt, not the tool outputs, not the padding']] There's the mask. [[circle "tensor([ 0.5000, -0.5000,  0.5000,  0.5000, -0.5000,  0.5000, -0.5000, -0.5000])"]] The advantages are plus or minus a half, exactly like on the whiteboard. [[circle "logp = -sft_model(inputs, targets, loss_reduction='none').view_as(inputs) # (8, T), 0 where target is -1"]] The model gives us the loss of every token, the negative log probability, so we negate it, and the ignored tokens come out as exactly zero. [[underline "[-0.23993052542209625, -0.05462498217821121"]] And here's the average log probability of each answer. [[circle "-0.600749135017395"]] The fifth answer, a wrong one, has a much lower log probability per token than the others, it's a bit of an unusual sample for this model. [[circle "loss = -(logp * advantages.unsqueeze(-1)).sum() / (targets >= 0).sum()"]] Then the loss: every token's log probability times the advantage of its answer, summed, divided by the number of trained tokens, and with a minus sign, because PyTorch minimizes and we want to maximize. And then backward, and an optimizer step. That's the whole algorithm.

[[open chat_rl.py "I put \"GRPO\" in quotes"]] Now if you've read about GRPO or PPO, you may wonder where everything else is. And nanochat makes four deliberate simplifications, written right at the top of the file. [[circle "1) no KL regularization to a reference model (no trust region)"]] Number one, there's no KL penalty to the SFT model. That's a regularizer that keeps the model from drifting too far, and for a short run like ours it's not needed. [[circle "2) we are fully on-policy"]] Number two, no PPO ratio and clipping. Those exist because PPO takes several gradient steps on the same batch of samples, so the samples go stale and you have to correct for that. We take exactly one step per batch, so we're always on-policy, and the ratio would always be exactly one. [[circle "3) token-level normalization of the loss (DAPO style)"]] Number three, we divide by the total number of tokens, not per answer, as the DAPO paper suggests, so long answers aren't down-weighted. [[circle "4) the advantage is just (reward - mean reward of the group)"]] And number four, we don't divide the advantage by the standard deviation. Put it all together, and what's left is good old REINFORCE, with a group mean baseline.

[[open chat_rl.py "def get_batch():"]] And chat_rl.py is exactly the two cells we just wrote, [[circle "advantages = rewards - rewards.mean()"]] the rollouts, [[open chat_rl.py "pg_obj = (logp * advantages.unsqueeze(-1)).sum()"]] and the loss, [[open chat_rl.py "def evaluate(num_samples=8):"]] plus an evaluation on held out questions every few steps. On a GPU you'd use sixteen questions per step and sixteen samples each. [[open chat_rl.py "CPU (a quick demo):"]] On my CPU, I did 40 steps of eight questions with eight samples each, which took about three hours, most of it sampling.

> **Predict:** We measure pass at one, the fraction of single samples that are right, and pass at eight, the fraction of questions with at least one right answer out of eight. Which one will RL improve more?

```bash
python chat_rl.py --task=SpellingBee --num-steps=40 --examples-per-step=8 --num-samples=8 --eval-examples=48 --lr=3e-5
```

```output
step    0 | eval pass@1 0.4427 | pass@8 0.9167
step    0 | reward 0.6406 | sampled tokens per answer 160.5 | lr 3.00e-05 | dt 193.4s
step    1 | reward 0.8750 | sampled tokens per answer 193.8 | lr 2.92e-05 | dt 349.8s
step    2 | reward 0.7812 | sampled tokens per answer 202.0 | lr 2.85e-05 | dt 248.2s
step    3 | reward 0.7344 | sampled tokens per answer 209.7 | lr 2.78e-05 | dt 252.6s
step    4 | reward 0.7812 | sampled tokens per answer 214.7 | lr 2.70e-05 | dt 252.2s
...
step   10 | eval pass@1 0.8385 | pass@8 1.0000
...
step   20 | eval pass@1 0.8672 | pass@8 0.9792
...
step   26 | reward 0.3438 | sampled tokens per answer 160.5 | lr 1.05e-05 | dt 153.6s
step   27 | reward 0.9844 | sampled tokens per answer 154.9 | lr 9.75e-06 | dt 135.3s
...
step   30 | eval pass@1 0.8776 | pass@8 1.0000
...
step   39 | reward 0.7500 | sampled tokens per answer 158.6 | lr 7.50e-07 | dt 142.3s
step   40 | eval pass@1 0.8750 | pass@8 1.0000
Saved the RL model to /root/.cache/nanochat-posttrain/chatrl.pt
```

So let's read this. [[circle "eval pass@1 0.4427"]] Pass at one is the fraction of single samples that are correct, and before RL, it's 44 percent. [[circle "pass@8 0.9167"]] Pass at eight is the fraction of questions where at least one of eight samples is correct, and it's 92 percent. So the model can already solve almost every question, it's just unreliable. [[circle "eval pass@1 0.8385"]] And after only ten steps of RL, pass at one jumps to 84 percent, [[circle "eval pass@1 0.8750"]] and ends at 87 and a half. [[note "eval pass@1 0.8750" "pass@k → pass@1"]] This is a really common pattern. RL takes what the model can do some of the time, and makes it do it most of the time. It turns pass at k into pass at one.

[[circle "reward 0.3438"]] And the training reward is really noisy, look at step 26. That's because every step only has eight questions, and some steps just get harder words. Always trust a held out evaluation, not the training reward. [[circle "sampled tokens per answer 214.7"]] And one more fun thing. The answers get shorter, from about 210 tokens [[circle "sampled tokens per answer 158.6"]] to about 155. Nobody asked for that. But many long answers were ones where the model got lost, misspelling a word into a long string of garbage letters, and those got zero reward, so RL pushed them down. And with greedy decoding on 100 test questions, SpellingBee goes from 53 percent after SFT to 86 percent after RL.

So now, finally, the famous question. Eight samples from the SFT model and from the RL model, and their final answers.

```python
from tasks import extract_answer
rl_model = load_model("rl", device).eval()
rl_engine = Engine(rl_model, tokenizer)

def answers(question, engine):
    ids = tokenizer.render_for_completion({"messages": [{"role": "user", "content": question}, {"role": "assistant", "content": ""}]})
    out, _ = engine.generate_batch(ids, num_samples=8, max_tokens=256, temperature=1.0, top_k=50)
    return [extract_answer(tokenizer.decode(o[len(ids):])) for o in out]

for question in ["How many r are in the word 'strawberry'?", "How many e are in cheese?", "How many r are in strawberry?"]:
    for name, engine_ in [("sft", sft_engine), ("rl", rl_engine)]:
        print(f"{name:3s} | {question:42s} | {answers(question, engine_)}")
```

[[circle "['1', '3', '2', '2', '2', '2', '2', '1']"]] Before RL, three r's in strawberry only once out of eight. [[circle "['3', '3', '3', '3', '3', '3', '3', '3']"]] After RL, eight out of eight. [[circle "How many e are in cheese?"]] Same for cheese. That's RL. [[circle "How many r are in strawberry?"]] But look at the last question, without the quotes and without "the word". [[circle "['4', '2', '3', '3', '1', '2', '2', '1']"]] Both models fall apart.

> **Predict:** With quotes, the RL model gets strawberry right eight times out of eight. Without them, it falls apart. Why? Hint: think about tokens.

Let's look.

```python
print(chat("How many r are in strawberry?", engine=rl_engine, max_tokens=40))
print([tokenizer.decode([t]) for t in tokenizer.encode("How many r are in strawberry?")])
print([tokenizer.decode([t]) for t in tokenizer.encode("How many r are in the word 'strawberry'?")])
```

[[circle "'sweet'"]] It says: we are asked to find the number of r's in the word "sweet". Sweet! It didn't even copy the right word. And the tokens tell you why. [[circle "' strawberry'"]] Without quotes, the word is "space strawberry", one single token. [[circle "'st', 'raw', 'berry'"]] With quotes, it's "st", "raw", "berry", exactly the tokens the model has to write in its spelling and in the python call. So with quotes, it just has to copy three tokens. Without quotes, it has to know how one single token is spelled, and write it as a different set of tokens, and it doesn't know that, so it reaches for something nearby in meaning: sweet. Our SpellingBee data mostly has rare words, which are several tokens either way, so the model never learned to unpack a common single-token word. The tokenizer, once again. And the fix is what you'd do in practice: now that we understand the failure, add training data for exactly this case. You look at the failures, you understand them, and you go back and fix the data.

> **Exercise:** Write the RL training loop yourself, using the two cells above: a loop over questions, an AdamW optimizer with a small learning rate, and an evaluation of pass@1 every ten steps. Then try changing the reward, for example a small penalty for long answers, and see what the model does. Careful: RL will find any loophole in your reward. Exercises fourteen to sixteen in the notebook are the pieces you need: pass at k, the log probability of an answer, and the policy gradient loss.

## Preferences: DPO

```whiteboard
@pipe text 40 40 "RLHF, the way InstructGPT did it" 42
@pipe fill 40 120 380 90 blue "people compare two answers" 28
@pipe arrow 430 165 520 165 black
@pipe fill 530 120 380 90 orange "train a reward model" 28
@pipe arrow 920 165 1010 165 black
@pipe fill 1020 120 400 90 purple "RL against it, on a KL leash" 28
@dpo text 40 300 "DPO: the same objective, straight from the pairs" 38 green
@dpo fill 40 370 380 90 blue "people compare two answers" 28
@dpo arrow 430 415 1010 415 green
@dpo fill 1020 370 400 90 green "the policy" 28
@dpo text 560 470 "no reward model, no sampling" 30 green
```

Okay, so RL works great when we can check the answer. But remember the problem we found in the SFT chapter? On ordinary questions, our model sometimes starts running the SpellingBee procedure. Let's measure how often, for the SFT model and the RL model. I'm taking twenty ordinary questions from the SmolTalk test set, nothing about letters or counting, sampling four answers to each, and checking whether the answer starts the procedure.

```python
from tasks import SmolTalk
LEAK = re.compile(r"We are asked to find|Let me try a manual approach")
smoltalk_test = SmolTalk("test")
first_messages = [smoltalk_test[i]["messages"][0] for i in range(300)]
questions = [m["content"] for m in first_messages if m["role"] == "user" and len(m["content"]) < 160
             and not re.search(r"letter|spell|count|how many", m["content"], re.I)][:20]

def leak_rate(engine):
    leaks = 0
    for i, question in enumerate(questions):
        ids = tokenizer.render_for_completion({"messages": [{"role": "user", "content": question}, {"role": "assistant", "content": ""}]})
        out, _ = engine.generate_batch(ids, num_samples=4, max_tokens=24, temperature=1.0, top_k=50, seed=i)
        leaks += sum(bool(LEAK.search(tokenizer.decode(o[len(ids):]))) for o in out)
    return leaks / (4 * len(questions))

print(f"sft: {leak_rate(sft_engine):.3f}, rl: {leak_rate(rl_engine):.3f}")
```

LEAKNARRATION

So how do we fix it? Honestly, the best fix is in the data: use fewer SpellingBee examples in the SFT mixture, which is the exercise from the SFT chapter. But I want to use this as an example of a much bigger problem. Most of the things we want from an assistant, we can't write a grader for. Is this answer helpful? Is it too long? Is it rude? Is this joke funny? There's no answer key in the back of the book. But there's one thing people are really good at: looking at two answers, and saying which one is better. [[draw pipe]] And that's how ChatGPT was made. In the InstructGPT paper, people compared answers of the model, then OpenAI trained a reward model, a neural net that predicts which answer people will prefer, and then they ran RL against that reward model, with a leash that keeps the model close to where it started. [[draw dpo]] And in 2023, the DPO paper showed that you can skip the reward model, and the RL, and fit the model directly to the pairs, with the exact same objective. Let's see how that's possible. It's only a few lines of math, and it's really beautiful.

[[erase]] [[write "Bradley-Terry: P(A beats B) = sigmoid(r(A) - r(B))"]] First, we need a model of preferences. Say every answer has a hidden score, a reward r. Then the probability that people prefer A over B is the sigmoid of the difference of the rewards. If A is a lot better, that's close to one, and if they're equally good, it's a half. [[write "RLHF: max E[r(y)] - beta * KL(pi || pi_ref)"]] Second, the goal of RLHF. We want a policy pi, that's our model, with a high expected reward, but we want it to stay close to the reference model, the model we started from, measured with the KL divergence. And beta says how tight the leash is. Without the leash, the model would run off and find whatever weird text maximizes the reward. [[write "best pi(y) = pi_ref(y) * exp(r(y) / beta) / Z"]] Now it turns out this objective has an exact solution. The best policy is the reference model, reweighted by the exponential of the reward over beta, and divided by a normalizer Z, so that the probabilities sum to one. Z is a sum over every possible answer, so we could never compute it. But watch what happens. [[write "→ r(y) = beta * log(pi(y) / pi_ref(y)) + beta * log Z"]] Take the log of both sides, and solve for the reward. The reward of an answer is beta times the log ratio of the policy and the reference, plus that constant. [[write "plug into Bradley-Terry: the beta * log Z cancels!"]] And now plug that into Bradley-Terry. We only ever need the difference of two rewards, for two answers to the same question, so the impossible Z simply cancels out. [[write "DPO loss = -log sigmoid(beta * (ratio(chosen) - ratio(rejected)))"]] What's left is a loss that only needs the log probabilities of two answers, [[write "ratio(y) = log pi(y) - log pi_ref(y)"]] under our model, and under the frozen reference. No reward model, and no sampling during training. It's a classification loss on pairs: make the chosen answer more likely than the rejected one, relative to the reference.

Let's build intuition on our tiny toy again. Four possible answers, the reference likes all of them equally, and the labelers prefer answer two over each of the other three. So that's three pairs, and we train with the DPO loss.

```python
def dpo_toy(beta, steps=100):
    ref_logits = torch.zeros(4)                 # the reference: all four answers equally likely
    logits = torch.zeros(4, requires_grad=True) # the policy starts out as a copy of the reference
    optimizer = torch.optim.SGD([logits], lr=1.0)
    chosen, rejected = torch.tensor([2, 2, 2]), torch.tensor([0, 1, 3]) # answer 2 beats each of the others
    for step in range(steps + 1):
        logp, ref_logp = logits.log_softmax(dim=0), ref_logits.log_softmax(dim=0)
        margin = beta * ((logp[chosen] - ref_logp[chosen]) - (logp[rejected] - ref_logp[rejected]))
        loss = -F.logsigmoid(margin).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 25 == 0:
            print(f"step {step:3d} | loss {loss.item():.4f} | p(answer 2) {logp.exp()[2].item():.3f} | weight {torch.sigmoid(-margin).mean().item():.3f}")

dpo_toy(beta=0.5)
```

TOYNARRATION

```python
dpo_toy(beta=2.0)
```

TOY2NARRATION

Okay, now the real thing. First we need pairs. We sample eight answers from the RL model to an ordinary question, and then someone has to judge them. In real life, this is where a person, or a big LLM acting as a judge, looks at two answers and picks one. For us, the judge is that regular expression: does the answer start spelling things out when nobody asked? That's a stand-in, and to be honest, it's so simple that we could have used it as an RL reward. But it means that you can reproduce every number here, and the DPO code doesn't care at all where the preferences came from.

```python
question = {"messages": [{"role": "user", "content": "What are loop quantum gravity transitions?"}, {"role": "assistant", "content": ""}]}
prompt = tokenizer.render_for_completion(question)
sequences, masks = rl_engine.generate_batch(prompt, num_samples=8, max_tokens=40, temperature=1.0, top_k=50, seed=2)
for seq in sequences:
    text = tokenizer.decode(seq[len(prompt):])
    print("rejected" if LEAK.search(text) else "chosen  ", repr(text[:80]))
```

PAIRNARRATION

```python
import copy
def sequence_logprobs(model, ids, mask):
    inputs, targets = ids[:, :-1], ids[:, 1:].clone()
    targets[mask[:, 1:] == 0] = -1 # only the answer's tokens count
    logp = -model(inputs, targets, loss_reduction='none').view_as(inputs)
    return logp.sum(dim=1) # the log probability of the whole answer

def dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1):
    margin = beta * ((policy_chosen - ref_chosen) - (policy_rejected - ref_rejected))
    return -F.logsigmoid(margin).mean()

texts = [tokenizer.decode(seq[len(prompt):]) for seq in sequences]
good = next(i for i, t in enumerate(texts) if not LEAK.search(t))
bad = next(i for i, t in enumerate(texts) if LEAK.search(t))
T = max(len(sequences[good]), len(sequences[bad]))
pad = lambda xs, value: xs + [value] * (T - len(xs))
ids = torch.tensor([pad(sequences[good], 0), pad(sequences[bad], 0)])
mask = torch.tensor([pad(masks[good], 0), pad(masks[bad], 0)])

policy = copy.deepcopy(rl_model).train() # the model we train; rl_model stays frozen as the reference
with torch.no_grad():
    ref_chosen, ref_rejected = sequence_logprobs(rl_model, ids, mask)
policy_chosen, policy_rejected = sequence_logprobs(policy, ids, mask)
loss = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected)
print(f"log p(chosen) {ref_chosen.item():.2f}, log p(rejected) {ref_rejected.item():.2f}, loss {loss.item():.4f}")
```

> **Predict:** Before any training, the policy and the reference are the same model. What will the DPO loss be?

LOSSNARRATION

```python
optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-5)
loss.backward()
optimizer.step()
with torch.no_grad():
    policy_chosen, policy_rejected = sequence_logprobs(policy, ids, mask)
print(f"ratio(chosen) {(policy_chosen - ref_chosen).item():+.3f}, ratio(rejected) {(policy_rejected - ref_rejected).item():+.3f}")
print(f"loss {dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected).item():.4f}")
del policy, optimizer
```

STEPNARRATION

And there's one more thing we must not forget. If we only show the model ordinary questions, nothing stops it from learning "never spell anything out", and then we'd lose everything RL just taught it. So the preference data also has spelling questions, where the chosen answer is a correct one, and the rejected answer is a wrong one. That's RL's checkable reward, turned into preferences. [[open chat_dpo.py "def collect_pairs(engine):"]] And that's chat_dpo.py. [[circle "good = [a for a in answers if not LEAK.search(a[\"text\"])]"]] The judge for ordinary questions, [[circle "good = [a for a in answers if bee.evaluate(conversation, a[\"text\"])]"]] the judge for spelling questions, [[circle "pairs.append({\"kind\": kind, \"chosen\": rng.choice(good), \"rejected\": rng.choice(bad)})"]] and a pair whenever there's at least one good and one bad answer among the eight. [[open chat_dpo.py "# the reference model never changes"]] Then, because the reference model never changes, we compute its log probabilities for all the pairs once, up front, [[open chat_dpo.py "loss.backward()"]] and the training loop is the same loop as always, with our loss. On my CPU, collecting the pairs took about RUNHOURS, and the training about RUNTRAIN.

```bash
python chat_dpo.py --general-prompts=600 --spelling-prompts=250
```

```output
DPOOUTPUT
```

RUNNARRATION

Let's see it for ourselves, on the same question as before, and on strawberry.

```python
dpo_model = load_model("dpo", device).eval()
dpo_engine = Engine(dpo_model, tokenizer)
print(f"leak | rl: {leak_rate(rl_engine):.3f}, dpo: {leak_rate(dpo_engine):.3f}")
sequences, _ = dpo_engine.generate_batch(prompt, num_samples=8, max_tokens=40, temperature=1.0, top_k=50, seed=2)
for seq in sequences:
    print(repr(tokenizer.decode(seq[len(prompt):])[:80]))
print(answers("How many r are in the word 'strawberry'?", dpo_engine))
```

TALKNARRATION

> **Exercise:** Run chat_dpo.py with only the ordinary questions, no spelling pairs, and measure SpellingBee afterwards. Then try a beta of 0.5 and of 0.01, and see what happens to the leak and to the answers. And exercises fifteen and seventeen in the notebook are the two functions of this chapter: the log probability of an answer, and the DPO loss.

## Putting it all together

Okay, so let's zoom out. [[write "1. a chat format: special tokens + a loss mask"]] We took a base model, a document simulator, and we gave it a format for conversations, with special tokens and a loss mask. [[write "2. SFT on a mixture: chat, MC, math + tool, spelling"]] We built a mixture of data where every part teaches something specific, a packing dataloader, and we ran SFT, which is just pretraining on that data. [[write "3. an engine: KV cache + tools"]] We wrote an inference engine with a KV cache and tool use, [[write "4. evals"]] evaluations, [[write "5. RL: sample, grade, push up what works"]] reinforcement learning with a group baseline, [[write "6. DPO: learn from pairs, no grader needed"]] and preference optimization from pairs. And you've now written every one of these pieces yourself.

[[erase]] Now if you want to post-train your own model, here's the recipe, in the order you'd actually do it. [[write "1. format: render the SAME way at train and test time"]] First, the format. Pick your special tokens, and make sure the training data and the prompt at test time are rendered by the same function. [[write "2. mask: train only on what the model must produce"]] The mask: one for everything the model has to write, including the end of its turn, and zero for everything else, including tool outputs. [[write "3. new tokens: new rows, check the loss at init"]] Give the new tokens rows in the embedding table, and check the loss at initialization. [[write "4. data: each part teaches one thing (and leaks)"]] Build a mixture where every dataset has a job, and remember that every dataset shapes the model beyond its job. [[write "5. SFT: pack, never crop, a gentle LR, one epoch"]] Pack conversations into rows, never cut one, and use a gentle learning rate. [[write "6. eval before and after EVERY stage"]] Evaluate before and after every single stage, on held out data, otherwise you don't know what you did. [[write "7. RL where you can check the answer (pass@k > 0)"]] Use RL where you can check the answer automatically, and where the model already succeeds sometimes. [[write "8. preferences where you can only compare"]] And use preferences for everything where you can only say which of two answers is better.

[[erase]] And here's the debugging checklist, which is really just every way things went wrong in this lecture. [[write "loss is NaN → a batch with nothing to train on"]] If your loss is NaN, look for batches with zero trainable tokens, like our truncated multiple choice questions. [[write "first losses huge → the new tokens (~15)"]] If the first losses are huge, it's probably the new special tokens. [[write "never stops talking → is assistant_end trained?"]] If the model never stops talking, check that assistant end gets a one in the mask. [[write "odd habits on other prompts → look at the mixture"]] If it has strange habits on unrelated questions, look at your data mixture, like our SpellingBee leak. [[write "RL reward flat → pass@k is 0 or 1: no signal"]] If the RL reward doesn't move, the questions are too hard or too easy, and every advantage is zero. [[write "reward up, eval down → it found a loophole"]] And if the training reward goes up but your held out eval goes down, the model found a loophole in your reward. All of this is also in the cheat sheet in the repo.

[[erase]] [[write "ours → nanochat: same files, same structure"]] And this maps one to one onto nanochat. Our tokenizer, tasks, engine, and the chat_sft, chat_rl, chat_eval and chat_cli scripts all have the same names there. DPO isn't in nanochat, but it's built from exactly the same pieces. [[write "+ its own base model (--depth), Muon, bf16, 8 GPUs"]] The real thing pretrains its own base model, whose size you pick with one depth parameter, it uses the Muon optimizer, bfloat16, eight GPUs with data parallel, and in nanochat, RL is done on GSM8K. But the ideas are the ones you just built. If you have a GPU node, it's one script.

```bash exec=0
git clone https://github.com/karpathy/nanochat && cd nanochat
bash runs/speedrun.sh                                        # tokenizer, pretraining, SFT, eval (~2 hours on 8XH100)
torchrun --standalone --nproc_per_node=8 -m scripts.chat_rl  # reinforcement learning on GSM8K
python -m scripts.chat_cli -i rl                             # talk to it
```

[[erase]] [[write "what's next:"]] And there's a lot more out there. [[write "reward models: RLHF with a learned judge"]] Instead of DPO, the classic RLHF pipeline trains a reward model on the preference pairs, and runs RL against it. [[write "careful: RL will game a learned reward"]] But a reward model is itself a neural net, and if you run RL against it for long, the policy finds adversarial examples that fool it. That's why the KL leash matters so much there. [[write "reasoning models: long RL on hard math and code"]] And there's the whole world of reasoning models, where you run RL for a long time on hard math and code, and the model discovers on its own that it should think for a long time, try different approaches and check its work. Our little SpellingBee procedure is a tiny cartoon of that.

> **Exercise:** The capstone. Pick a new skill, like reversing a word letter by letter, and post-train it into the model with every stage of this lecture: a synthetic task, SFT, an eval, RL, and a preference fix for whatever habit you don't like. It's exercise eighteen in the notebook, and it's the real test of whether you can do this from scratch.

Okay, so that's it for today. Take the code, run it, break it, add your own tasks, and see what your model learns. See you next time.
