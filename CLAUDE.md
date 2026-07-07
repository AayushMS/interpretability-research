# jlens-qwen-experiment

Personal pet project (not MARE work). Runs Anthropic's Jacobian lens (`jlens`) on
Qwen/Qwen3.5-0.8B on CPU and records what the lens can and cannot show at that scale.
Read `README.md` first -- it has the background, quickstart, and findings.

## Commands

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # this box has no GPU
pip install -r requirements.txt
python demo.py    # full run ~1 min after downloads; REWRITES results.md
```

## Hard constraints of this box (WSL2)

- **No CUDA.** Always install CPU torch first (`--index-url .../whl/cpu`) or pip pulls the
  ~2.5 GB CUDA wheel.
- **~7.6 GB RAM.** Load models in bf16. Qwen3.5-0.8B fits; the 4B+ models (where the paper's
  multi-hop effect should appear) do not -- don't try them here.

## Key facts

- Lens checkpoint is PRE-FITTED, from HF `neuronpedia/jacobian-lens`, revision `qwen-n1000`,
  file `qwen3.5-0.8b/jlens/Salesforce-wikitext/Qwen3.5-0.8B_jacobian_lens.pt`. Do not refit
  locally -- fitting is backward-pass heavy and pointless on CPU when checkpoints exist.
- `results.md` is a build artifact of `demo.py`. If you change the prompts or readout code,
  re-run the demo and commit the regenerated `results.md` in the same commit.
- Known negative result (documented in README): the hidden-intermediate multi-hop readout
  does not reproduce at 0.8B. Don't "fix" the demo to force it; that's the finding.
- Upstream `jlens` is an unmaintained reference implementation -- pin problems get solved by
  pinning versions in requirements.txt, not by patching upstream.

## Git / GitHub

- This repo belongs to the PERSONAL account: author `AayushMS <mansinghaayush@gmail.com>`
  (already set in local git config). Never commit here as ams-maitri.
- Remote is HTTPS on the personal account. To push:
  `gh auth switch --user AayushMS && git push && gh auth switch --user ams-maitri`
  (the gh credential helper follows the active account; the box's SSH keys are work keys).
- Never commit `.venv/`, `*.log`, or anything from `~/.cache/huggingface`.
