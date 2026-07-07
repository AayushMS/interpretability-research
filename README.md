# Interpretability Research

Personal research repo exploring LLM interpretability -- specifically, how much of Anthropic's
July 2026 **"global workspace" / J-space** findings transfer to small open-weights models,
using their open-sourced **Jacobian lens** (`jlens`).

## Background

- [*Verbalizable Representations Form a Global Workspace in Language Models*](https://transformer-circuits.pub/2026/workspace/index.html) -- the paper (results on Claude Sonnet/Haiku/Opus 4.5)
- [Anthropic announcement](https://www.anthropic.com/research/global-workspace)
- [anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens) -- the tool (Apache-2.0 reference implementation)
- [neuronpedia/jacobian-lens on HF](https://huggingface.co/neuronpedia/jacobian-lens) -- pre-fitted lens checkpoints for open models

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
