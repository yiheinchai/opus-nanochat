# Post-training cheat sheet

The whole lecture on one page: the recipe, the numbers to sanity check against, and the debugging checklist.

## The recipe

| # | stage | what to do | where |
|---|---|---|---|
| 1 | format | Special tokens for the turns (one id each, typed text can never produce them). Render training data and the test-time prompt with the **same** function. | `tokenizer.py` |
| 2 | mask | 1 for everything the model must write: its text, its tool calls, and `<\|assistant_end\|>` (that's how it learns to stop). 0 for the user, `<\|assistant_start\|>`, tool outputs, padding. Targets of -1 are ignored by the loss. | `render_conversation` |
| 3 | new tokens | New rows in the embedding table (with weight tying, that's also the output layer). Initialize them to the mean embedding. Check the loss at init. | `gpt.py` `from_pretrained` |
| 4 | data | A mixture where every dataset teaches one thing: chat (SmolTalk), a format (MMLU), tool use (GSM8K), a procedure (SpellingBee). Every dataset also leaks into unrelated behavior. | `tasks.py`, `chat_sft.py` |
| 5 | SFT | Pretraining with a different dataloader. Best-fit packing. **Never crop** a conversation: skip it. Gentle LR (1e-4, constant, then linear to 0 over the second half). One epoch. | `chat_sft.py` |
| 6 | engine | KV cache: prefill once, decode one token at a time, copy the cache for many samples. Tools: on `<\|python_end\|>`, run a **restricted** calculator and force its output in. Return a mask of sampled (1) vs forced (0) tokens. | `engine.py` |
| 7 | eval | Multiple choice from the logits of the allowed letters (no sampling). Generative: sample, grade `#### answer`. Before and after every stage, on held-out data. | `chat_eval.py` |
| 8 | RL | Where you can check the answer. A group of samples per question, advantage = reward - group mean, loss = -(logp x advantage) summed over sampled tokens / number of tokens. Needs 0 < pass@k < 1. Turns pass@k into pass@1. | `chat_rl.py` |
| 9 | preferences | Where you can only compare. Pairs of the model's own answers, chosen vs rejected. DPO loss = -log sigmoid(beta x ((log pi - log pi_ref)(chosen) - (log pi - log pi_ref)(rejected))). Add an SFT loss on the chosen answers and keep beta big enough (we used 0.5), or the model drifts. Cover the behavior you want to keep, too, and measure it before and after. | `chat_dpo.py` |

## Numbers to sanity check against

- Uniform guess over the vocabulary: -ln(1/50304) = **10.8**. New special tokens at init: about **15** (worse than uniform).
- First SFT batch with long MMLU passages: about 8 trained tokens and loss about 12. That's normal, the val loss is what counts.
- Naive padding wastes about **47%** of a batch, best-fit packing about **8%**.
- Multiple choice with 4 options: chance is **25%**, and with 100 questions the noise is about ±5%.
- DPO at step 0 (policy = reference): loss is exactly **ln 2 = 0.693**, accuracy about 0 (all margins are 0, up to float noise).

## Debugging checklist

| symptom | likely cause |
|---|---|
| loss is NaN | a batch with zero trainable tokens (e.g. a truncated conversation whose answer got cut off) |
| first losses are huge | the new special tokens (about 15 each at init) |
| the model never stops | `<\|assistant_end\|>` isn't trained (mask 0), or the prompt at test time doesn't match training |
| strange habits on unrelated prompts | the data mixture: look at what's oversampled (our SpellingBee leak: "We are asked to find...") |
| tool output is hallucinated | you trained on the tool output (it must be mask 0), or the engine doesn't force it in |
| MMLU at chance after SFT | the base model doesn't know it. SFT teaches format, not knowledge |
| RL reward doesn't move | every group is all right or all wrong: zero advantages. Pick questions at the edge of the model's ability |
| RL reward up, held-out eval down | the model found a loophole in the reward, or you're overfitting a small question set |
| DPO fixes the habit but breaks another skill | DPO pushes the rejected answers down everywhere, including where they belong (our SpellingBee went from 86% to 47%). Add an SFT loss on the chosen answers, use a bigger beta, fewer steps (we got back to 78%) |
| works with quotes, fails without | tokenization: `" strawberry"` is one token, `"strawberry"` is three |
