# Let's build the post-training of nanochat: from GPT-2 to a ChatGPT

<!--
This file is the single source of truth for the lecture. lecture/build.py turns it into
1) posttraining.ipynb (the follow-along notebook), 2) narration audio via ElevenLabs, 3) the video.

Markup:
- plain paragraphs are narration (spoken, and markdown cells in the notebook)
- ```python                         executed code cell
- ```python from=FILE start="..."   code cell whose content is pulled from a repo file (single source of truth)
  (add exec=0 to only show it, end="..." to stop before a line containing that text)
- ```bash                           a shell command (shown, not executed when building the notebook)
- ```output                         the output of the preceding bash cell, copied from a real run
- > **Exercise:** ...               a "pause the video and try it yourself" moment
-->

## Intro

Hi everyone. So today we are going to continue where Andrej Karpathy's Zero to Hero series left off. If you've been following along, you've built micrograd, you've built makemore, you've built a GPT from scratch on tiny Shakespeare, and in the last video you reproduced GPT-2, the 124 million parameter version, and you watched the loss come down. And at the end of all that, you had a base model. A thing that can continue text. And it's really cool, but if you actually try to talk to it, you'll find that it's not ChatGPT. It's not an assistant. It just sort of babbles.

So the question for today is: what happens after pretraining? How do you go from a base model, an internet document simulator, to something that you can actually talk to, that answers your questions, that uses tools, that solves math problems and that can even count the number of r's in strawberry? This stage is called post-training, and today we are going to build all of it, from scratch, in the style of Karpathy's nanochat repository.

Now nanochat is the full-stack ChatGPT clone that does everything, tokenizer training, pretraining, supervised finetuning, reinforcement learning, evaluation and inference, in a single minimal codebase. Pretraining is the expensive part. It needs a node of eight H100 GPUs for a couple of hours. But post-training is much cheaper, and so what we're going to do is we're going to take the GPT-2 base model that you already know and love, and we're going to put it through exactly the post-training pipeline of nanochat. Every file we write today has a counterpart in nanochat, with the same name and the same structure, and at the end I'll show you how to run the real thing at scale.

So here is the roadmap. First, we'll look at what a base model actually does, and we'll see why it's not an assistant. Then we'll figure out how to represent a conversation as a sequence of tokens, with some new special tokens. Then we'll assemble a mixture of training data, write a dataloader for it, and run supervised finetuning, or SFT. Then we'll write an inference engine with a KV cache and a little python tool. Then we'll write some evaluations so we can see what we're doing. And finally, we'll do reinforcement learning, where the model practices on problems and learns from its own successes and failures. Okay, so let's get started.

## The base model is a document simulator

Okay so first, let's load up GPT-2. I've written a gpt.py file here, and this is basically exactly the GPT-2 module from the GPT-2 reproduction video. There's the token embeddings, the position embeddings, the twelve transformer blocks, the final layer norm and the language model head, which shares its weights with the token embedding. I've only added two things to it, a KV cache, and a few extra rows in the embedding table, and we'll come back to both of those in a bit. For now let's just load the OpenAI weights and sample from it.

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

So here I'm doing greedy decoding, so always taking the most likely next token, and you'll recognize this one. "Hello, I'm a language model, and I'm a programmer." That's the same output as in the GPT-2 reproduction video, so we know that we've loaded the weights correctly.

Now let's ask it a question.

```python
print(complete("What is the capital of France?"))
```

Okay, so actually this is not too bad. It does know that the capital of France is Paris. But then it says it again, and again, and again, forever. And that's because this is not an assistant answering a question. This is a document simulator. It's seen a question on the internet, and it's asking itself, statistically, what tokens come next in a web page that starts like this. It has no notion of a question being answered, and no notion of being done.

So here's a fun trick. Because the base model is a document simulator, we can actually trick it into being an assistant, by giving it the beginning of a document that looks like a conversation with an assistant. This is sometimes called few-shot prompting. Let's try it.

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

So look at this, this is really interesting. First of all, it picked up on the format. It answers in the voice of the Assistant. But the answer is "France", which is wrong, it's a very small model. And then, it just keeps going. It writes the next question from the Human, and then the answer of the Assistant, and so on. It's simulating the whole document, both sides of the conversation, because that's all it knows how to do.

So this tells us a few things. Number one, the knowledge is in there, the capabilities are mostly there from pretraining. Number two, the base model is perfectly happy to adopt the persona of an assistant if it thinks that's the kind of document it's in. And number three, it has no idea where its turn ends. So the idea of post-training is basically: instead of tricking the model with a prompt, let's just continue training it on a lot of documents that look exactly like conversations, in some very specific format, with clearly marked turns. Then the model's statistics shift, and conversations become the only kind of document it knows.

> **Exercise:** Before we move on, think about how you would do this. If a conversation has to become a single sequence of tokens, what would that sequence look like? And which of those tokens should the model actually be trained on?

## Conversations are just documents

Okay so let's look at what a conversation looks like as data. In nanochat, and also in most of the datasets on HuggingFace, a conversation is just a dict with a list of messages, and each message has a role, user or assistant, and some content.

```python
conversation = {"messages": [
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "Paris"},
]}
```

And the question is how do we flatten this into one sequence of tokens. What we're going to do is we're going to introduce new special tokens that delimit the turns. There's a user start and a user end, an assistant start and an assistant end, and then there are two more pairs that we'll need later for tool use, python start and end for when the assistant wants to run some python code, and output start and end, for the output of that code. These are exactly the special tokens of nanochat.

```python from=tokenizer.py start="SPECIAL_TOKENS = [" end="class Tokenizer"
```

The most important one by far is assistant end, because that is how the model is going to tell us that it's done talking. Remember the base model that just went on and on? This token is the fix. When we sample from the model and it emits assistant end, we stop, and it's the user's turn.

Now here's the function that renders a conversation into tokens. It's called render_conversation, and it's very simple, but there's one very important subtlety. Besides the token ids, it also returns a mask, and the mask is one for every token that the assistant should be trained to produce, and zero for everything else.

```python from=tokenizer.py start="    def render_conversation" end="    def render_for_completion" exec=0
```

So we start with the beginning of sequence token. Then we go through the messages. For a user message, we add user start, the tokens of the content, and user end, and all of these get a mask of zero. For an assistant message, we add assistant start with a mask of zero, and then the content with a mask of one, and then assistant end, also with a mask of one, because the assistant has to learn when to stop.

The content of an assistant message can also be a list of parts, and that's for tool use. A python part is the assistant writing some code, so it's wrapped in python start and python end, and we do train on it, because at test time the assistant has to write that code. But the python output part is wrapped in output start and output end, and that one is masked out, because at test time this text is going to come from the python interpreter, not from the model. There's no point in training the model to predict the output of python, in fact it would be actively bad, it would be training the model to hallucinate tool outputs.

Let's look at it. I have a little helper to visualize the tokens, where green means trained on, and red means not trained on.

```python
ids, mask = tokenizer.render_conversation(conversation)
print(ids)
print(mask)
print(tokenizer.visualize(ids, mask))
```

So the whole conversation is thirteen tokens. The user part is all red, and the only green tokens are Paris and assistant end. So the model is only going to receive gradient on those two tokens. The user's question is just context.

And why do we mask the user tokens? Well, think about what we want. We are not trying to build a model that simulates users, we're trying to build a model that responds to them. If we train on the user tokens, a part of the model's capacity is spent on learning what kinds of questions people ask, which is not useless, but it's not the thing we care about, and with smaller models every bit of capacity matters. The other thing to note is that the very last token here, the assistant end, means something. The model has to learn that after Paris, it is done.

Now one more function, and this is the one we'll use at test time. When we want the model to respond, we render everything except for the last assistant message, and then we append assistant start. And that's the prompt. The model then takes it from there, and we sample until it emits assistant end.

```python from=tokenizer.py start="    def render_for_completion" end="    def visualize" exec=0
```

And by the way, this format is a design choice. OpenAI's ChatML uses tokens like im start and im end, Llama has its own format, and so on, but they all do basically the same thing. What matters is that you're consistent between training and test time. If you render a conversation one way during training, and a slightly different way at test time, even a single space different, things can go subtly wrong.

## Brain surgery on GPT-2

Okay so now we have a problem. GPT-2's vocabulary has 50,257 tokens and none of them are our special tokens. The only special token in GPT-2 is end of text, which has id 50,256, and we're going to use that one as our beginning of sequence token. But we need eight more.

Now here's where something fun comes in. If you remember the GPT-2 reproduction video, at some point we changed the vocab size from 50,257 to 50,304, because 50,304 is a nice number, it's divisible by 128 and so on, and the GPU kernels run faster. And we said those extra 47 tokens are never used, the model just learns to drive their probabilities to zero. Well, today we actually use them. Our special tokens get the ids 50,257 through 50,264, right inside that padding. So in the tokenizer we just tell tiktoken about these new tokens.

```python from=tokenizer.py start="    def __init__" end="    def encode(self" exec=0
```

And in the model, when we load the OpenAI weights, we copy the 50,257 rows of the embedding table that GPT-2 had, and then we have to decide what to do with all the new rows. Here I've initialized them to the average embedding of all the tokens. That's a reasonable thing to do. It's a neutral vector that sort of lives in the right part of the space, and it doesn't blow anything up.

```python from=gpt.py start="            if k == 'wte.weight':" end="            assert sd[ours].shape" exec=0
```

But now, and this is something that Karpathy really drills into you in the makemore videos, let's sanity check the loss at initialization, before we train anything. What loss do we expect? Well, if the model had no idea and was spreading its probability uniformly over the whole vocabulary, the loss would be the negative log of one over 50,304, so about 10.8. Let's look at the loss of every single token of our little conversation.

```python
x, y = torch.tensor([ids[:-1]]), torch.tensor([ids[1:]])
with torch.no_grad():
    losses = model(x, y, loss_reduction='none')
for token, m, loss in zip(ids[1:], mask[1:], losses):
    print(f"{tokenizer.decode([token])!r:25} mask={m} loss={loss.item():.2f}")
```

Okay so this is very interesting. The normal English tokens look fine. The model is very confident about the word "is" after "what", and so on. But look at the special tokens. They all have a loss of around 15, which is way worse than the 10.8 of a uniform guess. The model is confidently wrong about them. And that's because their embeddings are the average embedding, and because the weights are tied, the logit of a special token is the dot product of the hidden state with that average vector, and in GPT-2 that happens to be a pretty low logit. So in the very beginning of finetuning, the loss is going to be dominated by the special tokens, and a lot of the early optimization is just going to be spent on learning that these tokens exist. We'll see that in the loss curve. It's sort of the hockey stick from the makemore videos all over again.

By the way, in nanochat the special tokens are part of the tokenizer from the very beginning, so this surgery isn't needed. But they are never seen during pretraining, except for the beginning of sequence token, so they're in a similar situation. Their embeddings exist, but the model has never had to predict them.

> **Exercise:** Can you come up with a better initialization for the new embedding rows, one that gives a loss of about 10.8 or lower at initialization for the special tokens? What happens if you initialize them to the embedding of a related token, for example the embedding of a newline for user end?

## The data mixture

Okay, now we need data. A lot of conversations. And in nanochat, the SFT data is a mixture of several datasets, and each one of them teaches the model something different. So let's go through them one by one. In nanochat these live in a tasks directory, and I've put them all in one tasks.py file here. A task is basically just a list of conversations, plus optionally a way of grading the answer.

The first one is SmolTalk, from HuggingFace. This is 460,000 general conversations, and this is where most of the "personality" of the assistant comes from. Let's look at one.

```python
from tasks import SmolTalk, MMLU, GSM8K, SpellingBee
smoltalk = SmolTalk("train")
example = smoltalk[0]
for message in example["messages"]:
    print(f"--- {message['role']}:\n{message['content'][:500]}")
```

So these conversations were mostly generated synthetically, by bigger language models, and then filtered. And I think it's worth pausing on this, because it's a really important point that Karpathy makes in his talks about LLMs. When you talk to ChatGPT, who are you talking to? Well, the answer is that you're talking to a statistical simulation of the data in the SFT stage. Originally that data was written by human labelers following long labeling instructions from a company, about how to be helpful, truthful and harmless. Nowadays a lot of it is written by other LLMs. But either way, the persona, the tone, the length of the answers, the little bullet points, all of that comes from imitating this data. The knowledge comes from pretraining, but the personality is downloaded from SFT.

The second one is MMLU, which is multiple choice questions across 57 subjects. And this one teaches the model the format of multiple choice, which is how we're going to evaluate it later. Let's look.

```python
mmlu = MMLU("auxiliary_train")
example = mmlu[1]
print(example["messages"][0]["content"])
print("--- answer:", example["messages"][1]["content"])
```

So the user gives a question and some choices, and the assistant responds with a single letter. Now, the way this prompt is rendered actually hides two subtle details that matter a lot for small models. And I'll show you the function.

```python from=tasks.py start="def render_mc" end="class MMLU" exec=0
```

The first detail is that the letter comes after the choice, not before it. Big models don't care much, but for small models, having the letter right after the choice text helps them bind the letter to the choice. The second detail is that there is no space between the equals sign and the letter. And that's because of the tokenizer. Let's check.

```python
print(tokenizer.encode(" A"), tokenizer.encode("A"), tokenizer.encode("=A"))
```

The token "space A" is a completely different token from the token "A". They have nothing to do with each other as far as the model is concerned, they're just different integers. And the assistant's answer is going to be the token "A" without a space. So in the prompt we want the exact same token to appear, so that the model can sort of just point back at it. If you remember the tokenization video, a lot of the weirdness of LLMs traces back to the tokenizer, and this is one of those little things.

The third one is GSM8K, grade school math. And this one is really interesting because the solutions use a calculator. Let's look at a raw example first.

```python
gsm8k = GSM8K("train")
print(gsm8k.row(0)["answer"])
```

So you see these little things in double angle brackets? Those are the calculator annotations of the original dataset. So two times twelve equals twenty four is written as double angle bracket, two times twelve equals twenty four, double angle bracket. And what we do in our task is we parse these out, and turn them into a python tool call, followed by the output of the tool.

```python from=tasks.py start="class GSM8K" end="    def evaluate(self, conversation, completion):" exec=0
```

```python
ids, mask = tokenizer.render_conversation(gsm8k[0])
print(tokenizer.visualize(ids, mask))
```

So now look at this. The model is trained to write "2 times 12", and then python end, and then the output tokens, the 24, are red. They're not trained on, because at test time they will be filled in by our python tool. And then the model copies that 24 into its answer. And at the very end, the answer is given after four hash signs, and that's what we'll use to grade it. This is the simplest possible version of tool use. The model learns to offload arithmetic, which it is bad at, to a calculator, which is perfect at it.

And the fourth one is my favorite. It's the SpellingBee task, which was added to nanochat so that it could answer the famous question, how many r's are there in strawberry. Now why is this hard for an LLM? Well, let's look at how strawberry is tokenized.

```python
print([tokenizer.decode([t]) for t in tokenizer.encode(" strawberry")])
print([tokenizer.decode([t]) for t in tokenizer.encode("strawberry")])
```

So "space strawberry" is a single token. It's just one integer. The model never sees the letters. It has to somehow memorize, for every one of its fifty thousand tokens, what letters are inside of it, and then count them in its head, in a single forward pass. That's really hard. And without a space in front, it's three tokens, "st", "raw", "berry", which is a completely different set of integers again.

So SpellingBee is a synthetic task. We generate as many examples as we want, by taking a random word from a list of 370,000 English words and a random letter, and then we write out the ideal solution. Let's look at one.

```python
bee = SpellingBee(size=1000, split="train")
for part in bee[3]["messages"][1]["content"]:
    print(f"[{part['type']}]\n{part['text']}")
```

So the ideal answer first spells the word out letter by letter, and spelling is something the model can learn. Then it goes through the letters one at a time and keeps a running count, so that it never has to do too much computation in a single token. And then it double checks the answer with python, using the count method of strings, and then gives a final answer. This is a really nice example of how you have to think about the cognition of the model. You want to spread the computation across many tokens, because every token only gets a fixed amount of compute, a single forward pass through twelve layers. If you ask the model to blurt out the answer immediately, you're asking it to do the whole computation in one forward pass, and it just can't.

And then we mix all of these together. In nanochat this is a TaskMixture, which just takes a list of tasks, builds the full list of all their examples, and shuffles it with a fixed seed. And if you want to oversample something, you can just pass the same task in twice.

```python from=tasks.py start="class TaskMixture" end="# ---" exec=0
```

```python from=chat_sft.py start="train_dataset = TaskMixture([" end="val_dataset" exec=0
```

> **Exercise:** Write your own synthetic task. For example, a task that teaches the model its own name and who made it, or a task that reverses a word. Generate the ideal response programmatically, and think about how to spread the computation across tokens.

## The SFT dataloader

Okay, now we need to batch these conversations up so that we can train on them. And this is actually more interesting than you'd think.

In pretraining, this was easy. We had one infinite stream of text, and we just chopped it up into rows of 1024 tokens. If a document got cut in half, who cares. But conversations are different. A conversation is a unit. If you chop a conversation in half, the second half starts with no user question, and the first half has no answer, so we really want every conversation to be in one piece. And also, every row should start at the beginning of a conversation, with a beginning of sequence token, because that's what it's going to look like at test time.

So the naive approach is to put one conversation per row and pad the rest. But conversations are short. Most of them are a few hundred tokens, so you'd be wasting most of your compute on padding.

The better approach is packing. We put as many whole conversations into each row as we can. And nanochat uses a "best fit" algorithm for this. We keep a buffer of, say, 100 conversations, and for each row, we keep grabbing the longest conversation in the buffer that still fits in the remaining space. When nothing fits anymore, we pad the rest of the row, and we never, ever crop. Let's look at the code.

```python from=chat_sft.py start="def sft_data_generator" end="@torch.no_grad()"
```

So we build up each row, and a mask row alongside it. The padding gets a mask of zero. And then at the end, here's the crux of it. The inputs are the row without its last token, and the targets are the row shifted by one, exactly like in pretraining. And then wherever the mask is zero, we set the target to minus one. And minus one is the ignore index of the cross entropy loss in our model. So those positions simply don't contribute to the loss.

```python from=gpt.py start="        # targets of -1 are ignored" end="        return loss" exec=0
```

Let's try it out and see how efficient this is.

```python
from tasks import TaskMixture
device = "cpu"
small_mixture = TaskMixture([SmolTalk("train", stop=2000), MMLU("auxiliary_train", stop=2000), GSM8K("train"), bee])
loader = sft_data_generator(small_mixture, batch_size=4, seq_len=512)
num_tokens = num_padding = num_trained = 0
for _ in range(20):
    x, y, _ = next(loader)
    num_tokens += x.numel()
    num_padding += (x == tokenizer.bos).sum().item()
    num_trained += (y >= 0).sum().item()
print(f"fraction trained: {num_trained / num_tokens:.3f}, roughly fraction padding: {num_padding / num_tokens:.3f}")
```

So in a typical batch, only a fraction of the tokens are actually trained on. The rest is user prompts, special tokens and a bit of padding. And that's normal. It just means that in SFT, a batch of a given size carries a lot less "signal" than in pretraining.

Now I want to tell you about a bug that I hit while writing this, because it's really instructive. In my first version, I rendered every conversation with a maximum length, and if it was longer than a row, I just truncated it. Seems harmless, right? But when I ran a small test, the loss came out as nan, and the number of trained tokens was zero. So what happened? Well, MMLU's auxiliary train set has a lot of really long reading comprehension questions, and the answer is a single letter at the very end. If you truncate that, you cut off the answer, and you're left with a conversation that's all prompt and no green tokens at all. And the best fit algorithm loves these, because they are exactly as long as a row! So it was filling entire batches with conversations that have nothing to train on. And the mean over zero tokens is zero divided by zero, which is nan. The fix is the one line you see here: if a conversation doesn't fit in a row, we skip it. That's what "we never crop" means.

> **Exercise:** Measure how much padding you'd have with the naive one-conversation-per-row approach, compared to best fit packing.

## The SFT training loop

Okay so now we can write the training loop, and the punchline is that SFT is literally just pretraining, with a different dataloader. It's the same model, the same cross entropy loss, the same optimizer. The only difference is the minus ones in the targets. Let's look.

```python from=chat_sft.py start="optimizer = model.configure_optimizer" end="path = os.path.join" exec=0
```

The optimizer is AdamW, the same as in the GPT-2 video. The learning rate is constant for the first half of training, and then decays linearly to zero. That's the schedule that nanochat uses. There's no warmup, and there's no weight decay. The learning rate is a lot lower than in pretraining, we're not trying to learn the internet again, we're just gently nudging the statistics of a model that already knows a lot.

In nanochat, the default is to do one full epoch over the whole mixture. Here, on a GPU, that's around 600,000 conversations, and it takes something like half an hour on a single A100. But I want to show you that you don't need a GPU to see this work. So I'm going to run a much smaller version, on a plain four core CPU: 600 steps of four rows of 512 tokens each. That takes about an hour and a half.

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

So look at the very first step. The loss is 11.9, and that's the special tokens we talked about, the model is confidently wrong about them. But also look at the number of trained tokens: eight. In the entire first batch of four rows of 512 tokens, there were only eight tokens to train on. And that's the best fit algorithm at work. At the start, the longest conversations in the buffer are those long MMLU reading comprehension questions, so each row gets one of them, and each one has exactly two green tokens, the letter and the assistant end. Then step two has 428 trained tokens and a loss of 2.6. So the training loss per step is really noisy, because every batch has a completely different composition, and it's much better to look at the validation loss. And the validation loss goes down nicely, from 2.73 to 1.34.

Okay, so now the fun part. Let's talk to it. I'm going to write a little chat function, that renders a conversation, primes the assistant, and samples until the assistant end token.

```python
from gpt import load_model
sft_model = load_model("sft", device).eval()
sft_engine = Engine(sft_model, tokenizer)

def chat(question, engine=sft_engine, temperature=0.0, seed=42, max_tokens=100):
    conversation = {"messages": [{"role": "user", "content": question}, {"role": "assistant", "content": ""}]}
    ids = tokenizer.render_for_completion(conversation)
    out, _ = engine.generate_batch(ids, max_tokens=max_tokens, temperature=temperature, top_k=50, seed=seed)
    return tokenizer.decode(out[0][len(ids):])

print(chat("Hi! Who are you?", temperature=0.7, seed=1))
```

"Hello! How can I help you today?" And then it stopped. It emitted the assistant end token, all by itself. Compare that to the base model, which would have just kept going forever, writing both sides of the conversation. So that's the format, learned. Let's ask it about France again.

```python
print(chat("What is the capital of France?"))
```

So it answers directly now, as an assistant, "The capital of France is Paris". But then it gets stuck in a loop. And this is a very common failure of greedy decoding with small models: once it has said "the capital of France" a couple of times, the most likely next thing is to say it again, and it falls into a rut that it can't get out of. Sampling with a temperature helps. But let me show you something even more interesting.

```python
print(chat("What is the capital of France?", temperature=0.7, seed=2, max_tokens=150))
```

Ha! Look at this. "We are asked to find the capital of France. Let me try a manual approach first." And then it spells France, badly, counts the letter f, and calls python to double check. It's running the SpellingBee procedure on a geography question! So what's going on here? Well, in this little CPU run, SpellingBee is twenty thousand of the 75,000 conversations, so more than a quarter of everything the model saw. And a lot of the SpellingBee questions are short questions like "What is the count of e in some word". So "What is the capital of France" sort of looks like one of those, on the surface, and the model pattern matches on the surface form. If I sample answers to thirty real questions from the SmolTalk test set, only two of them start like this, so it's mostly the short questions. But it's a really good lesson about the data mixture. Every dataset you put in there shapes the behavior of the model, in ways that leak out beyond the dataset itself. In fact, when SpellingBee was added to nanochat, Karpathy noted in the commit that it probably uses way too many of these synthetic examples. On a GPU, with a full epoch, SmolTalk dominates the mixture and this mostly goes away.

```python
print(chat("Why is the sky blue?", temperature=0.7, seed=1))
```

And this one is kind of hilarious, meteoroids and molten metal. So let's be honest about what we have here. This is a 124 million parameter model from 2019, finetuned for an hour and a half on a CPU. It doesn't really know why the sky is blue, and it's very happy to make things up, with total confidence. The nanochat README says that talking to its speedrun model is a bit like talking to a kindergartener, and we're well below that. But the format is there, the turn taking is there, and as we'll see, on the narrow tasks that we specifically trained for, it can actually do quite well.

> **Exercise:** Re-run SFT with fewer SpellingBee examples, say five thousand instead of twenty thousand, and measure how often the model answers ordinary questions with "We are asked to find". What happens to its SpellingBee accuracy?

## The inference engine

Okay, so now let's talk about the Engine, which is what we've been using to sample from the model. In nanochat, the engine has two jobs. The first is to make sampling fast, and the second is to run the tools.

So first, why is naive sampling slow? Well, think about what we do in a naive sampling loop. We take the whole sequence, run it through the transformer, get the logits at the last position, sample a token, append it, and then run the whole sequence through the transformer again. But for all the tokens that we already processed, nothing changed! Their keys and values in every attention layer are exactly the same as last time, because of the causal mask, a token can never be affected by tokens that come after it. So we're recomputing the same things over and over, and for long sequences this gets really wasteful.

The fix is the KV cache. We keep the keys and values of every layer around, and when a new token comes in, we only compute the query, key and value for that one token, append the new key and value to the cache, and let the single new query attend to everything in the cache. Here's the KV cache in our gpt.py. It is tiny.

```python from=gpt.py start="class KVCache" end="class CausalSelfAttention" exec=0
```

```python from=gpt.py start="class CausalSelfAttention" end="class MLP" exec=0
```

So in the attention, if there's a cache, we append the new keys and values and get back the full keys and values. And one subtle thing: if we're feeding in a single token, we don't want a causal mask at all, because that one new token is allowed to look at everything before it. And in the model's forward, the positions have to start where the cache left off, otherwise the new token would get the position embedding of position zero. Let's make sure the cache actually gives exactly the same answer, and see how much faster it is.

```python
import time
prompt = [tokenizer.bos] + tokenizer.encode("The meaning of life is")

t0 = time.time()
ids = torch.tensor([prompt])
with torch.no_grad():
    for _ in range(100):
        next_id = model(ids)[:, -1, :].argmax(dim=-1, keepdim=True) # recompute everything, every time
        ids = torch.cat([ids, next_id], dim=1)
t1 = time.time()
out, _ = engine.generate_batch(prompt, max_tokens=100, temperature=0.0)
t2 = time.time()
print(f"naive: {t1 - t0:.2f}s, with kv cache: {t2 - t1:.2f}s, same result: {ids[0].tolist() == out[0]}")
```

{{KV_COMMENT}}

The other trick in the engine is for when we want many samples for the same prompt, which is going to be very important for reinforcement learning. We process the prompt once, with a batch size of one, and this is called the prefill. And then we just copy the cache for every sample, and then decode all the samples in parallel, as one batch.

Now the second job of the engine is tool use. And this is where it gets kind of cute. Let's look at the main loop.

```python from=engine.py start="        # 3) decode, one token at a time" end="            yield column, masks" exec=0
```

> **Exercise:** Before I explain it, pause and think about it. The model is sampling tokens one at a time. At some point it emits python start, then some tokens, then python end. What exactly should the engine do at that point, and what should the model see next?

So every row keeps a little bit of state. When the model samples python start, we start recording the tokens of the expression. When it samples python end, we decode the expression, and evaluate it with our calculator. And then here's the trick: we put output start, the tokens of the result, and output end into a queue of forced tokens. And on the next steps, instead of using what the model sampled, we pop tokens from that queue. So the model sees the result of the tool appear in its own context, as if it wrote it, and then it continues sampling from there. And we also return a mask, one for sampled tokens and zero for forced tokens, and you'll see why that matters in reinforcement learning.

The calculator itself is deliberately very restricted. We only allow numbers and arithmetic, or the count method of a string, and we evaluate without any builtins. You really don't want to give a language model an unrestricted eval on your machine. In nanochat it's the same story, and there's a separate, more careful sandbox for running real python code.

```python from=engine.py start="def use_calculator" end="@torch.no_grad()" exec=0
```

Let's watch it in action. Here's a SpellingBee question from the test set, and I'm going to visualize what comes out of the engine, where green tokens were sampled by the model, and red tokens were forced in by the engine.

```python
example = SpellingBee(size=10, split="test")[6]
print(example["messages"][0]["content"])
ids = tokenizer.render_for_completion(example)
out, masks = sft_engine.generate_batch(ids, max_tokens=256, temperature=0.0)
print(tokenizer.visualize(out[0][len(ids):], masks[0][len(ids):]))
```

So look at what happened here, because this is a really nice little example. The question is how many u's are in eurytherm. The model tries to spell it out, and it gets the spelling wrong. It writes e, y, u, r, and so on. It roughly knows which letters are in there, but not in the right order, because remember, it never actually sees the letters. Then it counts the u's in its own misspelled version, and it gets one, which happens to be right. And then it writes the python call, and in the python call it copies the word exactly, because copying a token is easy. Then the engine runs the code and forces in output start, one, output end, and those are the only three red tokens in the whole response. And Python says one, and the model reports one. So the tool is robust to the model's own sloppiness. Copying the word into python is a lot easier than spelling it.

## Evaluation

Okay so we have a model that seems to work, but "seems to work" is not good enough. We need numbers. And so in nanochat there's a chat_eval script, and there are basically two kinds of evaluations.

The first kind is categorical, which is for multiple choice. And here's a neat thing. We don't actually have to sample anything. We render the question, prime the assistant, run a single forward pass, and then we look at the logits at the very last position, but only at the logits of the allowed letters, A, B, C and D. And whichever one is highest, that's the model's answer.

```python from=chat_eval.py start="def run_categorical_eval" end="def run_generative_eval" exec=0
```

So this is kind of generous to the model, because even if it would have said something else, we're forcing it to pick one of the letters. But that's how these evaluations are typically done, and it makes them fast and low-noise. And the random baseline for four choices is 25 percent.

The second kind is generative, for things like GSM8K and SpellingBee. Here we actually sample a response, then extract the number after the four hash signs, and compare it to the reference answer.

```python from=chat_eval.py start="def run_generative_eval" end="TASKS = {" exec=0
```

And then to get one number, nanochat has something called ChatCORE. It takes the accuracy on every task, rescales it so that zero means random guessing and one means perfect, and then averages them. So let's evaluate both the base model and our SFT model on 100 problems of each task. For the base model, I'm going to use the same chat format, which it has never seen, so we expect it to be pretty bad.

```bash
python chat_eval.py -i base -x 100
python chat_eval.py -i sft -x 100
```

```output
{{EVAL_TABLE}}
```

{{EVAL_COMMENT}}

> **Exercise:** Try evaluating the base model on MMLU with a few-shot prompt instead of the chat format. How close can you get to the SFT model just by prompting?

## Reinforcement learning

Okay, so now we get to the last stage, and I think the most exciting one. Reinforcement learning.

So first, why do we need it? Karpathy has a really nice analogy for this. Think about how you learn from a textbook. Most of the textbook is exposition, background knowledge, and that's like pretraining. Then there are worked examples, where an expert shows you how to solve a problem step by step, and you imitate them. And that's SFT. But then at the end of every chapter, there are practice problems. You're given the problem, and the final answer is in the back of the book, but not the solution. And you have to try, and fail, and try again, and figure out what works for you. And that's reinforcement learning.

And here's the deep reason why this matters. When we do SFT on a GSM8K solution, we're asking the model to imitate a human's solution. But the human has a very different brain from the model. Some step that's trivial for the human might be a huge leap for the model, a leap that it can't make in a single token, and some step that the human spells out carefully might be totally obvious to the model. We don't actually know what the best token sequence is for this particular model to get to the right answer. And so the idea of reinforcement learning is, let's not tell it. Let the model try many times, and whatever works, do more of that.

So how do we actually do that? Let's say we have a question, and the model samples an answer y, and then we check the answer and give it a reward R, one if it's correct and zero if it's wrong. And what we want is to maximize the expected reward, the average reward over all the answers that the model might sample. Now, it turns out there's a really neat trick for computing the gradient of that. The gradient of the expected reward is equal to the expected value of the reward times the gradient of the log probability of the sample. This is called the policy gradient, or the REINFORCE trick, and it comes from the fact that the gradient of p is equal to p times the gradient of log p. And the log probability of a whole answer is just the sum of the log probabilities of its tokens, which is exactly the negative of the loss that our model already computes.

So in code, it's almost embarrassingly simple. Sample a bunch of answers. For each one, compute the sum of the log probabilities of its tokens, multiply by its reward, and do gradient ascent on that. And if you squint at it, it's basically SFT on your own samples, weighted by how good they were.

But there's one problem. If the rewards are zero or one, then the wrong answers get a weight of zero, so they're just ignored, and every right answer gets pushed up, even on easy questions where all the answers were right. So what we do is we subtract a baseline. For every question, we sample a group of answers, say sixteen of them, and the advantage of each answer is its reward minus the mean reward of the group. So if an answer is better than average for this question, its probability gets pushed up. If it's worse than average, its probability gets pushed down. And you can show that subtracting a baseline like this doesn't change the expected gradient at all, it just makes it a lot less noisy. This is the "group" in GRPO, group relative policy optimization.

Before we do this on a transformer, let's build some intuition on the tiniest possible example. Imagine a "language model" that only ever produces a single token, out of four options, and only option two is correct. The model is just four logits. We sample a group of sixteen attempts, reward the correct ones, compute the advantages, and do the policy gradient update.

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

So the probability of the correct answer starts at a quarter, because the logits are all zero, and after only ten steps it's already at 85 percent, and it keeps climbing towards one. Nobody ever told it the answer. It just tried things, and did more of what worked. Now watch what happens if the model starts out really bad at this, if the correct answer is very unlikely to begin with.

```python
reinforce_toy([0.0, 0.0, -10.0, 0.0])
```

Nothing happens. The correct answer is never sampled, so the reward is always zero, so all the advantages are zero, and there's no gradient at all. And this is a really important lesson about RL: it can only amplify behaviors that the model already does at least sometimes. It can't conjure up something from nothing. That's why we do SFT first, to get the model into a regime where it sometimes succeeds, and then RL takes it from there.

> **Exercise:** Pause here and try to write the loss yourself. You have the inputs and targets of a batch of sixteen sampled answers, the targets are minus one wherever we shouldn't train, and you have a tensor of sixteen advantages. Our model can return the per-token loss if you pass loss_reduction equals none.

Okay, let's look at the code. First, the rollouts.

```python from=chat_rl.py start="def get_batch" end="def evaluate" exec=0
```

So for every question, we render the prompt and use the engine to sample a whole group of answers in parallel, at temperature one, because we want diversity. We decode each one and get its reward from the task, and for SpellingBee and GSM8K that's just whether the answer after the hash signs is correct. Then we build inputs and targets exactly like in SFT, and here's where the engine's mask comes in: we set the targets to minus one for the prompt, for the padding, and for the tool outputs that were forced into the sequence. We only train on the tokens that the model actually chose. And then the advantages are the rewards minus their mean.

And here's the training step.

```python from=chat_rl.py start="    for _ in range(args.examples_per_step):" end="    lr = args.lr" exec=0
```

So the model gives us the per-token loss, which is the negative log probability, so we negate it to get the log probabilities, and the ignored tokens have a log probability of exactly zero. We multiply each token's log probability by the advantage of the answer it belongs to, sum it all up, and normalize by the number of valid tokens. And then the loss is the negative of that objective, because PyTorch minimizes and we want to maximize. And that's it. That's the whole algorithm.

Now, if you've read about GRPO or PPO, you might be wondering where all the other stuff is. And nanochat makes four deliberate simplifications, and they're written right at the top of the file.

```python from=chat_rl.py start="I put" end="GPU:" exec=0
```

Number one, there's no KL penalty to a reference model. That's a regularizer that keeps the model from drifting too far from the SFT model, and in practice, for a short run like this, it's not really needed. Number two, there's no PPO ratio and clipping. Those exist because in PPO you take multiple gradient steps on the same batch of samples, so the samples become stale, off-policy, and you need to correct for that. But we take exactly one gradient step per batch, so we are always on-policy, and the ratio would always be exactly one. Number three, we normalize by the total number of tokens, not per answer, which is what the DAPO paper recommends, so that long answers aren't down-weighted. And number four, the advantage is just the reward minus the mean, we don't divide by the standard deviation. Put all of that together and what you're left with is honestly just good old REINFORCE, with a group mean baseline.

One more thing to notice. If all sixteen answers to a question are correct, or all of them are wrong, then all the advantages are zero, and that question contributes no gradient at all. So RL only learns from questions that are at the edge of the model's ability, where it sometimes succeeds and sometimes fails. If your model can never solve a problem, RL can't help you, and that's why we do SFT first.

{{RL_INTRO}}

```bash
{{RL_CMD}}
```

```output
{{RL_LOG}}
```

{{RL_COMMENT}}

And let's ask it the famous question.

```python
rl_model = load_model("rl", device).eval()
rl_engine = Engine(rl_model, tokenizer)
print(chat("How many r are in strawberry?", engine=rl_engine))
```

{{STRAWBERRY_COMMENT}}

> **Exercise:** Look at the samples during RL. Does the model start trusting python over its manual count, or the other way around? Try changing the reward, for example a small penalty for long answers, and see what the model does. Be careful, RL will find any loophole in your reward.

## Scaling up, and what's next

Okay, so that's everything. Let's zoom out and recap. We took a base model, a document simulator, and we did three things. We defined a format for conversations, with special tokens and a loss mask. We did supervised finetuning on a mixture of conversations, where every dataset teaches something specific: chatting, multiple choice, math with a calculator, and spelling. And then we did reinforcement learning, where the model samples its own answers and we push up the ones that work. Along the way we wrote an inference engine with a KV cache and tool use, and evaluations to measure everything.

Now everything we wrote today maps one to one onto nanochat. Our tokenizer is nanochat's tokenizer.py, our tasks are its tasks directory, our engine is its engine.py, and our chat_sft, chat_rl, chat_eval and chat_cli are its scripts. The real thing has a few more bells and whistles. The base model is pretrained from scratch instead of GPT-2, and you can pick its size with a single depth parameter. It uses the Muon optimizer for the matrices, it runs in bfloat16 across eight GPUs with distributed data parallel, and it warm-starts the optimizer state from pretraining. And in nanochat, RL is done on GSM8K. But the ideas are exactly the ones we built today. If you have a GPU node, the whole thing is one script.

```bash
git clone https://github.com/karpathy/nanochat && cd nanochat
bash runs/speedrun.sh                                        # tokenizer, pretraining, SFT, eval (~2 hours on 8XH100)
torchrun --standalone --nproc_per_node=8 -m scripts.chat_rl  # reinforcement learning on GSM8K
python -m scripts.chat_cli -i rl                             # talk to it
```

And there's a lot that we didn't cover. We only did reinforcement learning on problems where we can check the answer automatically. But what about writing a poem, or a joke? There's no answer key for that. And that's where RLHF comes in, reinforcement learning from human feedback, where you train a separate reward model on human preferences, and then do RL against that reward model. Or methods like DPO, which skip the reward model entirely. But be careful, because a reward model is itself a neural network, and if you run RL against it for too long, the policy will find adversarial examples that fool it. So RLHF is a bit of a different beast from the kind of RL we did today, where the reward is the actual ground truth.

And there's the whole world of reasoning models, where you run RL for a very long time on hard math and code problems, and the model discovers on its own that it should think for a long time, try different approaches, backtrack, and double check its work. That's really what our SpellingBee task is a tiny cartoon of.

Okay, so that's it for today. I hope you enjoyed it. Take the code, run it, break it, add your own tasks, and see what your model learns. See you next time.
