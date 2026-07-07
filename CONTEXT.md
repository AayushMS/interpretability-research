# CONTEXT -- read this first when resuming work

Handoff document for future sessions (human or AI). Everything below was true as of
**2026-07-07** and was verified by execution, not assumed.

## What this repo is

Personal interpretability research (Aayush's pet project, unrelated to MARE work): testing
how much of Anthropic's "global workspace / J-space" paper transfers to small open-weights
models using their open-sourced Jacobian lens. `README.md` has the background and experiment
index; `PLANS.md` has the roadmap. Next planned step: **experiment 02, the scale sweep**.

## Story so far (chronological)

1. **2026-07-06**: Anthropic published the paper
   ([announcement](https://www.anthropic.com/research/global-workspace),
   [paper](https://transformer-circuits.pub/2026/workspace/index.html)) claiming Claude models
   contain a sparse "J-space" workspace: reportable, steerable, causally involved in multi-hop
   reasoning. Results are on Claude Sonnet/Haiku/Opus 4.5 only -- generality to other models is
   explicitly left open. The measurement tool (`jlens`) was open-sourced at
   [anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens) as an
   **unmaintained reference implementation** (Apache-2.0).
2. **2026-07-07**: ran it end-to-end on **Qwen/Qwen3.5-0.8B** on CPU (experiment 01 here).
   Findings below. An independent adversarial re-run then reproduced `results.md`
   **byte-for-byte** (only wall-clock timings differed), verified the lens checkpoint
   provenance on disk, and code-reviewed the readout path for canned outputs (none).
3. Repo was created as `jlens-qwen-experiment`, then reframed/renamed to
   `interpretability-research` with the Qwen run filed as experiment 01.

## Verified findings (experiment 01)

- **The lens works on open models and clearly beats the plain logit-lens**: at mid layers
  (L6-L12) logit-lens output is incoherent multilingual noise while J-lens is on-topic; by
  L22 (of 24) the J-lens top-10 contains the actual answer tokens (` yen`, ` euro`/` franc`,
  ` Atlantic`) on all three multi-hop prompts -- answer formation is visible pre-emission.
- **The paper's headline hidden-intermediate effect does NOT reproduce at 0.8B**: best rank of
  the hidden intermediate at the descriptor token -- Italy@` boot` = 5792 (L15),
  France@` Tower` = 100 (L21), Brazil@` Carnival` = 338 (L12). The lens mostly returns
  synonyms of the descriptor token. This is the expected capability-dependence, not a bug.
  **Do not "fix" the demo to force this result -- the negative result IS the finding.**

## Key technical facts (all verified by execution)

- Model: `Qwen/Qwen3.5-0.8B`, 24 layers, d_model=1024, loaded in **bf16** (RAM constraint).
- Lens: pre-fitted, HF repo `neuronpedia/jacobian-lens`, revision `qwen-n1000`, file
  `qwen3.5-0.8b/jlens/Salesforce-wikitext/Qwen3.5-0.8B_jacobian_lens.pt` (48 MB, fitted on
  Salesforce/wikitext, 233 prompts, converged early from a 1000 budget). The same HF repo has
  lenses for Qwen3 1.7B-32B and Qwen3.5 2B-27B and Qwen3.6-27B -- that's what experiment 02 uses.
- Stack that worked: Python 3.12.3, torch 2.12.1+cpu, transformers 5.13.0, jlens 0.1.0
  (editable install from the GitHub repo, HEAD 581d398 "Initial release"). Upstream test
  suite: 32 passed.
- Full run of `demo.py`: ~1 min on CPU (each forward+readout 2-4 s). First run downloads
  ~1.7 GB to `~/.cache/huggingface`.
- One benign warning: flash-linear-attention fast path unavailable -> torch fallback (harmless
  on CPU).

## Environment constraints (this box)

- **WSL2 Ubuntu, no GPU** (`nvidia-smi` absent; `torch.cuda.is_available()` False).
- **~7.6 GB RAM**: 0.8B bf16 fits; 1.7B is borderline; 4B+ does not fit. Bigger models need a
  rented GPU (Colab/RunPod/vast.ai) on a personal account -- NOT MARE AWS.
- Install CPU torch FIRST (`pip install torch --index-url https://download.pytorch.org/whl/cpu`)
  or pip pulls the ~2.5 GB CUDA wheel.
- Long-running shell commands (model downloads, fits) should run via nohup + log polling;
  interactive shells here time out around 10 min.
- A ready-made venv from the original run may still exist at
  `~/work/mare/mare_extras/mare-agentic-workflow/scratch/jlens/.venv` (with the cloned
  jacobian-lens repo next to it), but don't rely on it -- the quickstart in experiment 01's
  README rebuilds everything.

## Repo conventions

- `experiments/NN-short-name/` per experiment: `README.md` (writeup), code, `results.md`
  (build artifact -- regenerate and commit together with the code change that produced it),
  own `requirements.txt`.
- Plans live in `PLANS.md`; update statuses in place. This file (`CONTEXT.md`) gets a short
  appended note when an experiment lands or a major decision is made.
- Git/GitHub: personal account -- see `CLAUDE.md` for the exact push flow (author
  `AayushMS <mansinghaayush@gmail.com>`, `gh auth switch --user AayushMS`, HTTPS remote).

## 2026-07-07 (evening): experiment 02 CPU half landed; repo made public; GUIDE + playground added

- **Repo is now PUBLIC** on GitHub (user request).
- **Experiment 02 (scale sweep), CPU half done** -- pythia-70m / gpt2 (fp32) and
  Qwen3.5-0.8B / Qwen3-1.7B (bf16), all 93 prompts of `lens-eval-multihop`, with a
  shuffled-intermediate control. Verified findings (see `experiments/02-scale-sweep/`):
  best-anywhere intermediate visibility saturates by 0.8B (84%/91% of prompts <=10 vs ~30%
  control); it does NOT differ between solved and unsolved prompts at any size; hits
  cluster at end-of-prompt function tokens in the back half of the network -- so at <=2B it
  reads as association/late retrieval, NOT the paper's descriptor-token workspace effect
  (exp 01's negative stands: Brazil@`Carnival` r338 at the descriptor vs r3 anywhere).
  Remaining: gemma-3-270m/1b (just needs HF login + license click), Qwen3 4B/8B/14B (rented
  GPU); same `sweep.py --model` command, copy `raw/*.jsonl` back, re-run `report.py`.
- **Methodological trap found: bf16 collapses pythia-70m's final logits into thousands of
  exact ties** (rank becomes meaningless). Small models now run fp32 via a per-model dtype
  in the `sweep.py` registry; every record carries a `model_max_ties` audit field (Qwen
  bf16 is fine: worst case 3 tied tokens on 4/93 prompts).
- **New for humans: `GUIDE.md`** (from-zero setup + nine DIY experiment types) and
  **`playground/peek.py`** (one-command per-layer readout of any prompt with `--watch`
  word rank tracking, `--position`, `--logit-lens` baseline; imports the exp-02 registry).
- Driver quirk learned: `nohup script.sh & echo $!` reports the nohup wrapper's pid, not
  the script's -- write `$$` to a pidfile inside the script instead (see scratchpad
  drivers), and never `pkill -f` a pattern that matches your own shell.

## Open questions

- At what parameter count does the hidden-intermediate effect become lens-visible? (Exp 02)
- How corpus-sensitive is lens fitting? (Exp 03)
- Do reportability/modulation hold on open instruct models at that scale? (Exp 04)
- Does the reference implementation support write/injection at all? (Exp 05)
