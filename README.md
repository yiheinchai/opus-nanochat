# Let's build the post-training of nanochat

A from-scratch lecture on **all of LLM post-training**: chat formats and special tokens, supervised finetuning (SFT), an inference engine with a KV cache and tool use, evaluation, and reinforcement learning. It is written in the style of Andrej Karpathy's [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) and follows the post-training pipeline of his [nanochat](https://github.com/karpathy/nanochat) repo file for file.

We start where ["Let's reproduce GPT-2 (124M)"](https://www.youtube.com/watch?v=l8pRSuU81PU) ends: with a GPT-2 base model that can only babble. We end with an assistant that answers questions, uses a python tool for arithmetic, answers multiple choice, and counts the r's in strawberry. Then RL makes it better at that by practicing on its own.

This is an independent tutorial. It isn't made by or affiliated with Andrej Karpathy, and the narration uses a stock synthetic voice (Kokoro's `am_michael` by default), not a clone of his.

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
| [lecture/build.py](lecture/build.py) | lecture.md → notebook, narration (Kokoro, ElevenLabs or Fish Audio Drama 3), video + subtitles | |
| [lecture/screencast.py](lecture/screencast.py) | Renders the lecture as a screencast: typing in a notebook, an editor with tabs, a terminal, a whiteboard, and a mouse that points and draws, driven by `[[cues]]` in lecture.md | |
| [lecture/align.py](lecture/align.py), [lecture/avatar.py](lecture/avatar.py) | Word timings from Whisper (so actions land on the spoken word), and the animated narrator | |
| [video/](video/) | The rendered lectures: `v2/` (the step-by-step screencast) and `v1/` (the first, slideshow version), each with subtitles | |

The goal of the lecture: if you've finished Zero to Hero, you should be able to rebuild every piece of nanochat's post-training yourself. So nothing appears fully formed. Every component starts from a problem, tries the obvious thing, measures it failing, and then gets written in the notebook and tested before we look at the repo file:

0. Intro: the goal, and the roadmap
1. The base model is a document simulator: why GPT-2 isn't an assistant, and the few-shot trick
2. Conversations are just documents: why special tokens (typed as text they're 7 ordinary tokens), `render_conversation` in three steps (tokens → mask → tool calls), the prompt at test time, checked against the repo
3. Brain surgery on GPT-2: new embedding rows, weight tying, and sanity checking the loss at init (~15 for the special tokens vs 10.8 uniform)
4. The data mixture: parsing GSM8K's calculator annotations into tool calls, the grader, multiple choice and its tokenizer detail, SmolTalk, writing the SpellingBee procedure, a TaskMixture in ten lines
5. The SFT dataloader: measuring conversation lengths, naive padding (47% wasted) vs best-fit packing (8%), and the truncation bug reproduced live (a cut-off answer → zero trainable tokens → NaN)
6. The SFT training loop: the schedule, a few live SFT steps in the notebook, then the full CPU run, and talking to the result
7. The inference engine: naive sampling, the three KV cache changes to GPT-2, prefill and decode, tool use with forced tokens, a safe calculator
8. Evaluation: multiple choice from the logits, generative grading, ChatCORE
9. Reinforcement learning: the policy gradient derived on the whiteboard, a 4-logit toy REINFORCE, one real rollout group and one policy gradient step on the SFT model, nanochat's four simplifications, the RL run, pass@k → pass@1, and why "strawberry" still fails
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

## Make the narrated video

`lecture/build.py` turns `lecture/lecture.md` into a screencast, recorded the way Karpathy records his. Code gets typed into a Jupyter-style notebook and run, with the real outputs. Repo files open in an editor with tabs and scroll to the part being discussed. Training logs stream in a terminal, diagrams get drawn on a whiteboard, and a mouse cursor points at things, circles them, underlines them and writes notes. These actions are directed by `[[cues]]` inside the narration (see the top of [lecture/screencast.py](lecture/screencast.py)), and each one happens on the exact word, using word timestamps from Whisper. A lecture outline sits in a sidebar, and an animated teacher talks from a "webcam" corner. The teacher is an original cartoon presenter, labelled as an AI narrator, not a likeness of Karpathy or anyone else ([lecture/avatar.py](lecture/avatar.py)). Its lip sync is driven by the audio itself, so it works with every voice backend. Pass `--no-teacher` to leave it out. It also writes an `.srt` subtitle file. Clips are cached per paragraph, so edits only re-synthesize what changed.

The default voice is [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) (Apache-2.0). It runs locally, on a plain CPU, with no account or API key, at about 2x real time on 4 cores:

```bash
pip install kokoro soundfile faster-whisper pygments pillow   # Python 3.10-3.12 (faster-whisper times the cues to the words)
python lecture/build.py audio            # ~30 min on a 4-core CPU
python lecture/build.py video            # -> lecture/build/lecture.mp4 and lecture/build/lecture.srt
```

Two other backends are built in. Pass the same `--tts` to both the `audio` and the `video` step:

- `--tts=elevenlabs`: hosted. Create an API key, keep it out of code and chat (environment variable or Colab Secret `ELEVENLABS_API_KEY`), and check that your plan's character quota covers the ~49,000 characters (`python lecture/build.py stats`). [lecture/make_video_colab.ipynb](lecture/make_video_colab.ipynb) does this on Colab.
- `--tts=fish`: [Fish Audio Drama 3](https://huggingface.co/fishaudio/drama-3-int4), open weights, served on your own GPU with [sglang-omni](https://github.com/sgl-project/sglang-omni). Even the int4 version needs an Ampere-or-newer GPU and used about 57GB at startup in Fish Audio's A100-80GB test, so think Colab H100 rather than a laptop. Start the server (`sgl-omni serve --model-path fishaudio/drama-3-int4 --port 8000`), then run the build with `FISH_TTS_URL=http://localhost:8000/v1/audio/speech`. This backend is untested: it was written against the model card, without access to a big enough GPU. Mind the Fish Audio Community License: it is free for research, non-commercial use, and small commercial use.

`--voice` picks a voice for the backend. For example `--voice=af_heart` for Kokoro, or an ElevenLabs voice id. To preview timing without any TTS, run `audio --tts=silent` and then `video --tts=silent`, and add `--chapters=2` to render only the first two chapters.

To change the lecture, edit `lecture/lecture.md` and rebuild. Code shown from the repo files is pulled in at build time with `from=FILE start="..."`, so the video always shows the current code. Re-execute the notebook (`python lecture/build.py notebook --execute`) so that the outputs on screen stay real.

## Credits

All the ideas and the structure come from Andrej Karpathy's [nanochat](https://github.com/karpathy/nanochat) (MIT), including the chat format, the best-fit SFT dataloader, the Engine's tool use, the SpellingBee task and the simplified GRPO. The teaching style comes from [Zero to Hero](https://karpathy.ai/zero-to-hero.html). Datasets: [SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk), [MMLU](https://huggingface.co/datasets/cais/mmlu), [ARC](https://huggingface.co/datasets/allenai/ai2_arc), [GSM8K](https://huggingface.co/datasets/openai/gsm8k), and the [english-words](https://github.com/dwyl/english-words) list.
