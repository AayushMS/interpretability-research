# jlens-qwen-experiment

Trying out Anthropic's [Jacobian lens](https://github.com/anthropics/jacobian-lens) (`jlens`)
on the smallest Qwen model that has a published pre-fitted lens -- **Qwen/Qwen3.5-0.8B** --
on a plain CPU (WSL2, no GPU).

Background: the lens comes from Anthropic's July 2026 interpretability work,
[*Verbalizable Representations Form a Global Workspace in Language Models*](https://transformer-circuits.pub/2026/workspace/index.html)
([announcement](https://www.anthropic.com/research/global-workspace)). It transports a
hidden-state vector from any layer into the final-layer basis via a fitted Jacobian and
decodes it through the unembedding -- "what is this activation disposed to make the model say."

## What's here

- `demo.py` -- end-to-end demo: loads Qwen3.5-0.8B in bf16 on CPU, pulls the pre-fitted lens
  from the Hub ([`neuronpedia/jacobian-lens`](https://huggingface.co/neuronpedia/jacobian-lens),
  revision `qwen-n1000`), and reads out three multi-hop factual prompts.
  Compares **J-lens vs plain logit-lens vs the model's actual next-token distribution**.
- `results.md` -- the raw captured output of a full run (regenerated on every run).

## Results (TL;DR)

**Part 1 -- the lens works, and clearly beats the logit-lens baseline.**
At mid layers (L6-L12) the plain logit-lens readout is incoherent multilingual noise while the
J-lens returns on-topic tokens (currency / country / ocean). By the penultimate layer (L22 of 24)
the J-lens top-10 already contains the actual answer tokens (` yen`, ` euro`/` franc`,
` Atlantic`), matching what the model goes on to predict.

**Part 2 -- the paper's headline "hidden workspace" effect does NOT reproduce at 0.8B.**
The paper's flagship demo reads out a hidden intermediate concept at the descriptor token
(e.g. *Italy* at the word "boot" in "the country shaped like a boot", without Italy ever
appearing in the text). On this 0.8B model the intermediate ranks in the hundreds to thousands
(`Italy@boot`: best rank 5792; `France@Tower`: 100; `Brazil@Carnival`: 338) -- the lens mostly
just shows synonyms of the descriptor token itself. Expected: Anthropic's demos use 4B+ models
and the strong J-space claims are only established for Claude models, so the workspace-like
structure looks capability-dependent.

## Reproduce

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # skip if you have CUDA
pip install -r requirements.txt
python demo.py          # ~1 min on CPU after downloads; writes results.md
```

First run downloads ~1.7 GB (model + 48 MB lens) into `~/.cache/huggingface`.
Needs roughly 4 GB free RAM (model is loaded in bf16).

## Notes

- `jlens` is installed straight from Anthropic's repo (Apache-2.0). It's a
  **reference implementation** -- unmaintained, not accepting contributions.
- The pre-fitted lens was fitted by Neuronpedia on Salesforce/wikitext (233 prompts).
  The same Hub repo ships lenses for larger Qwen3 / Qwen3.5 / Qwen3.6 models if you have
  the RAM/GPU to try the 4B+ ones, where the multi-hop effect should be stronger.
