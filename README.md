# Let's build the post-training of nanochat

A from-scratch lecture on **all of LLM post-training**: chat formats and special tokens, supervised finetuning (SFT), an inference engine with a KV cache and tool use, evaluation, and reinforcement learning. It is written in the style of Andrej Karpathy's [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) and follows the post-training pipeline of his [nanochat](https://github.com/karpathy/nanochat) repo file for file.

We start where ["Let's reproduce GPT-2 (124M)"](https://www.youtube.com/watch?v=l8pRSuU81PU) ends: with a GPT-2 base model that can only babble. We end with an assistant that answers questions, uses a python tool for arithmetic, answers multiple choice, and counts the r's in strawberry. Then RL makes it better at that by practicing on its own.

This is an independent tutorial. It isn't made by or affiliated with Andrej Karpathy, and the narration uses a stock ElevenLabs voice, not a clone of his.

## What's in here

| file | what it does | nanochat counterpart |
|---|---|---|
| [gpt.py](gpt.py) | GPT-2 (124M), plus a KV cache and room for new special tokens | `nanochat/gpt.py` |
| [tokenizer.py](tokenizer.py) | GPT-2 BPE + chat special tokens, `render_conversation` with the loss mask | `nanochat/tokenizer.py` |
| [tasks.py](tasks.py) | SmolTalk, MMLU, ARC, GSM8K (with calculator calls), SpellingBee, TaskMixture | `tasks/*.py` |
| [engine.py](engine.py) | Fast sampling (KV cache, prefill once, many samples) and the tool use state machine | `nanochat/engine.py` |
| [chat_sft.py](chat_sft.py) | SFT with a best-fit packing dataloader and masked loss | `scripts/chat_sft.py` |
| [chat_eval.py](chat_eval.py) | Categorical (multiple choice) and generative evals, ChatCORE | `scripts/chat_eval.py` |
| [chat_rl.py](chat_rl.py) | "GRPO", which is really REINFORCE with a group mean baseline | `scripts/chat_rl.py` |
| [chat_cli.py](chat_cli.py) | Talk to your model | `scripts/chat_cli.py` |
| [posttraining.ipynb](posttraining.ipynb) | The lecture as a follow-along notebook, with real outputs | |
| [lecture/lecture.md](lecture/lecture.md) | The lecture script (narration + code): the single source of truth | |
| [lecture/build.py](lecture/build.py) | lecture.md → notebook, ElevenLabs narration, video + subtitles | |

The lecture's chapters:

0. Intro
1. The base model is a document simulator: why GPT-2 isn't an assistant, and the few-shot trick
2. Conversations are just documents: special tokens, `render_conversation`, the loss mask
3. Brain surgery on GPT-2: new token embeddings, and sanity checking the loss at init
4. The data mixture: SmolTalk, MMLU (and its tokenizer subtleties), GSM8K tool calls, SpellingBee
5. The SFT dataloader: padding vs. best-fit packing, and a real bug
6. The SFT training loop: it's just pretraining with −1s in the targets
7. The inference engine: KV cache, prefill, and tool use with forced tokens
8. Evaluation: categorical vs. generative, ChatCORE
9. Reinforcement learning: policy gradients from scratch, advantages, nanochat's four simplifications, pass@k
10. Scaling up with nanochat, and what's next (RLHF, DPO, reasoning)

Every chapter has a "pause the video and try it yourself" exercise.

## Results from the lecture's run

Everything shown in the lecture comes from a real run on a 4-core CPU. That was 600 SFT steps (about 1.5 hours) and 40 RL steps on SpellingBee (about 3 hours). Evals use 100 problems per task, so expect about ±5% noise.

| | ARC-Easy | ARC-Challenge | MMLU | GSM8K | SpellingBee (greedy) |
|---|---|---|---|---|---|
| GPT-2 base (chat format) | 29% | 26% | 29% | 0% | 0% |
| after SFT | 24% | 26% | 26% | 0% | 53% |
| after RL | | | | | 86% |

During RL, held-out SpellingBee pass@1 at temperature 1 went from 44% to 87.5%, and pass@8 from 92% to 100%. Multiple choice and GSM8K stay at chance: GPT-2 small doesn't have the knowledge, and SFT can't add it. The lecture discusses this honestly, along with the two failures worth learning from: SpellingBee "leaking" into unrelated answers, and the single-token " strawberry" that the model can't spell.

## Follow along

Locally (CPU or Apple Silicon works for everything; training is just slow):

```bash
pip install torch tiktoken safetensors pyarrow numpy jupyter
jupyter notebook posttraining.ipynb
```

On Colab, open `posttraining.ipynb` and pick a GPU runtime (A100/H100 if you have Colab Pro). The first cell clones the repo. With a GPU you can run the full-size versions:

```bash
python chat_sft.py                        # one epoch over ~600K conversations, ~30-40 min on an A100
python chat_eval.py -i sft                # all tasks
python chat_rl.py --task=SpellingBee      # or --task=GSM8K like nanochat
python chat_eval.py -i rl
python chat_cli.py -i rl
```

The defaults assume a 40GB+ GPU. On a 16GB GPU (e.g. a T4), use `python chat_sft.py --device-batch-size=4 --grad-accum=16`, which gives the same total batch size. The small CPU-sized commands used in the lecture are in the docstrings at the top of `chat_sft.py` and `chat_rl.py`. Downloads and checkpoints go to `~/.cache/nanochat-posttrain` (override with `NANOCHAT_BASE_DIR`).

## Make the narrated video (ElevenLabs)

`lecture/build.py` turns `lecture/lecture.md` into a video. Each narration paragraph becomes one ElevenLabs clip, and the screen shows the code and real output being discussed. It also writes an `.srt` subtitle file.

1. Create an ElevenLabs account and an API key (Profile → API Keys). The lecture is about 49,000 characters (roughly an hour of audio), so check that your plan's monthly character quota covers it. `python lecture/build.py stats` prints the exact count.
2. Keep the key out of the code and the chat. Put it in an environment variable or a Colab Secret named `ELEVENLABS_API_KEY`.
3. Run:

```bash
export ELEVENLABS_API_KEY=...            # or, on Colab, use the Secrets panel (see lecture/make_video_colab.ipynb)
python lecture/build.py audio            # cached per paragraph, so edits only re-synthesize what changed
python lecture/build.py video            # -> lecture/build/lecture.mp4 and lecture/build/lecture.srt
```

The easiest path is [lecture/make_video_colab.ipynb](lecture/make_video_colab.ipynb), which needs no GPU. The default voice is the stock ElevenLabs voice "George". To use a different one, set `ELEVENLABS_VOICE_ID` to any voice id from your ElevenLabs voice library. To preview timing without an API key, run `python lecture/build.py audio --tts=silent` and then `python lecture/build.py video --tts=silent`.

To change the lecture, edit `lecture/lecture.md` and rebuild. Code shown from the repo files is pulled in at build time with `from=FILE start="..."`, so the video always shows the current code. Re-execute the notebook (`python lecture/build.py notebook --execute`) so that the outputs on screen stay real.

## Credits

All the ideas and the structure come from Andrej Karpathy's [nanochat](https://github.com/karpathy/nanochat) (MIT), including the chat format, the best-fit SFT dataloader, the Engine's tool use, the SpellingBee task and the simplified GRPO. The teaching style comes from [Zero to Hero](https://karpathy.ai/zero-to-hero.html). Datasets: [SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk), [MMLU](https://huggingface.co/datasets/cais/mmlu), [ARC](https://huggingface.co/datasets/allenai/ai2_arc), [GSM8K](https://huggingface.co/datasets/openai/gsm8k), and the [english-words](https://github.com/dwyl/english-words) list.
