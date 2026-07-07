# Interpretability Research

Personal research repo exploring LLM interpretability -- specifically, how much of Anthropic's
July 2026 **"global workspace" / J-space** findings transfer to small open-weights models
(currently Qwen), using their open-sourced **Jacobian lens** (`jlens`).

This project was sparked by Anthropic's announcement post,
[**A Global Workspace in Language Models**](https://www.anthropic.com/research/global-workspace)
-- worth reading first for the big picture.

## Background

- [A Global Workspace in Language Models](https://www.anthropic.com/research/global-workspace) -- the announcement article that kicked this off
- Links from that article:
  - [*Verbalizable Representations Form a Global Workspace in Language Models*](https://transformer-circuits.pub/2026/workspace/index.html) -- the full paper (results on Claude Sonnet/Haiku/Opus 4.5)
  - [anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens) -- the open-sourced tool (Apache-2.0 reference implementation)
  - [J-lens interactive demo on Neuronpedia](https://neuronpedia.org/jlens) -- browse readouts on open-weights models, zero setup
  - [Expert commentary (PDF)](https://www-cdn.anthropic.com/files/4zrzovbb/website/cc4be2488d65e54a6ed06492f8968398ddc18ebe.pdf) -- independent perspectives from neuroscientists, philosophers, and interpretability researchers
  - [Agentic misalignment](https://www.anthropic.com/research/agentic-misalignment) -- earlier Anthropic work on models pursuing hidden goals (J-space monitoring is pitched as a detector for this)
- [neuronpedia/jacobian-lens on HF](https://huggingface.co/neuronpedia/jacobian-lens) -- pre-fitted lens checkpoints for open models (what experiment 01 uses)

The lens transports a hidden-state vector from any layer into the final-layer basis via a
fitted Jacobian and decodes it through the unembedding -- *"what is this activation disposed
to make the model say?"* The paper claims Claude-scale models contain a sparse, reportable,
causally load-bearing workspace. Whether small open models have one is an open question --
that's what this repo pokes at.

## Experiments

| # | Experiment | Status | Headline result |
|---|-----------|--------|-----------------|
| 01 | [jlens on Qwen3.5-0.8B, CPU](experiments/01-jlens-qwen-cpu/) | **Done, independently verified** | Lens works and beats the logit-lens baseline (answers visible by L22); the paper's hidden-intermediate multi-hop effect does **not** reproduce at 0.8B |

Planned experiments are in [PLANS.md](PLANS.md).

## For future sessions (human or AI)

**Read [CONTEXT.md](CONTEXT.md) first** -- it's the handoff doc: full story so far, verified
findings, environment constraints, and where to pick up. [CLAUDE.md](CLAUDE.md) has the
working conventions for Claude Code sessions.

## License

MIT for code in this repo. The `jlens` dependency is Apache-2.0 (Anthropic PBC); models and
lens checkpoints carry their own licenses on the Hub.
