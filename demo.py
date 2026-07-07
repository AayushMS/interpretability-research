#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC lens code is Apache-2.0; this demo is a thin driver.
"""End-to-end Jacobian-lens demo on the smallest Qwen with a published lens.

Loads Qwen/Qwen3.5-0.8B on CPU, pulls the pre-fitted Jacobian lens from the
Hub (neuronpedia/jacobian-lens, revision qwen-n1000), and reads out a handful
of multi-hop factual prompts at several layers -- Jacobian lens vs. the plain
logit-lens baseline vs. the model's actual next-token distribution.

Re-run: .venv/bin/python demo.py   (writes results to results.md)
"""

from __future__ import annotations

import sys
import time

import torch
import transformers

import jlens

MODEL_NAME = "Qwen/Qwen3.5-0.8B"
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_REVISION = "qwen-n1000"
LENS_FILE = "qwen3.5-0.8b/jlens/Salesforce-wikitext/Qwen3.5-0.8B_jacobian_lens.pt"

# (title, prompt, hidden intermediate concept, expected answer, focus word). The
# intermediate never appears in the prompt; a good lens surfaces it mid-network.
# `focus` is a descriptor token in the prompt (boot->Italy, Tower->France, ...):
# Part 2 reads the lens out AT that token to look for the intermediate concept.
PROMPTS = [
    (
        "Multi-hop currency (README/examples.py 'multihop')",
        "Fact: The capital of Japan is Tokyo.\n"
        "Fact: The currency used in the country shaped like a boot is",
        "Italy",
        "euro / lira",
        "boot",
    ),
    (
        "Multi-hop Eiffel Tower currency",
        "Fact: the currency of the country where the Eiffel Tower stands is",
        "France",
        "euro / franc",
        "Tower",
    ),
    (
        "Multi-hop Carnival ocean (data/evaluations/lens-eval-multihop 'carnival-ocean')",
        "Fact: The ocean on the coast of the country where Carnival is "
        "most famously celebrated is the",
        "Brazil",
        "Atlantic",
        "Carnival",
    ),
]

TOPK = 10


def fmt_tokens(tok, logits: torch.Tensor, k: int = TOPK) -> str:
    """Top-k decoded tokens for a [vocab] logit vector, as repr'd strings."""
    ids = logits.topk(k).indices.tolist()
    return "  ".join(repr(tok.decode([t])) for t in ids)


def rank_of(logits: torch.Tensor, token_id: int) -> int:
    """Rank (1-based) of token_id in a [vocab] logit vector."""
    return int((logits > logits[token_id]).sum().item()) + 1


def first_token_id(tok, word: str) -> int:
    """Leading token id of ' word' (space-prefixed, how it appears mid-text)."""
    return tok(" " + word, add_special_tokens=False).input_ids[0]


def focus_index(toks: list[str], focus: str) -> int:
    """Index of the last prompt token whose text contains `focus`."""
    hits = [i for i, t in enumerate(toks) if focus.lower() in t.lower()]
    return hits[-1] if hits else len(toks) - 1


def main() -> None:
    out = []

    def emit(line: str = "") -> None:
        print(line)
        out.append(line)

    emit(f"# Jacobian-lens demo — {MODEL_NAME}")
    emit()
    emit(f"- torch {torch.__version__}  (cuda available: {torch.cuda.is_available()})")
    emit(f"- transformers {transformers.__version__}")
    emit(f"- python {sys.version.split()[0]}")
    emit(f"- lens: {LENS_REPO} @ {LENS_REVISION} :: {LENS_FILE}")
    emit()

    t0 = time.perf_counter()
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16
    )
    hf_model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    model = jlens.from_hf(hf_model, tokenizer)
    emit(f"model loaded on CPU in {time.perf_counter() - t0:.0f}s: {model!r}")

    lens = jlens.JacobianLens.from_pretrained(
        LENS_REPO, filename=LENS_FILE, revision=LENS_REVISION
    )
    emit(f"lens loaded: {lens!r}")
    emit()

    n = model.n_layers
    layers = [n // 4, n // 2, (3 * n) // 4, n - 2]  # quarter / half / 3-q / penult.
    emit(f"n_layers={n}, reading out at layers {layers} (+ model output at L{n-1})")
    emit(f"readout position = -1 (the token that predicts the answer), top-{TOPK}")
    emit()

    emit("#" * 100)
    emit("# PART 1 — readout at the ANSWER position (-1): lens vs model next-token")
    emit("#" * 100)
    emit()
    for title, prompt, intermediate, answer, _focus in PROMPTS:
        t = time.perf_counter()
        jl, model_logits, input_ids = lens.apply(
            model, prompt, layers=layers, positions=[-1]
        )
        ll, _, _ = lens.apply(
            model, prompt, layers=layers, positions=[-1], use_jacobian=False
        )
        toks = [tokenizer.decode([i]) for i in input_ids[0].tolist()]

        emit("=" * 100)
        emit(f"## {title}")
        emit(f"prompt: {prompt!r}")
        emit(f"hidden intermediate (not in prompt): {intermediate!r} | "
             f"expected answer: {answer!r}")
        emit(f"tokenized ({len(toks)} tok), last 6: {toks[-6:]}")
        emit(f"(forward+readout {time.perf_counter() - t:.1f}s)")
        emit()
        for layer in layers:
            emit(f"  L{layer:>2} J-lens    : {fmt_tokens(tokenizer, jl[layer][0])}")
            emit(f"  L{layer:>2} logit-lens: {fmt_tokens(tokenizer, ll[layer][0])}")
            emit()
        emit(f"  MODEL next-token (L{n-1}): {fmt_tokens(tokenizer, model_logits[0])}")
        emit()

    # ------------------------------------------------------------------ Part 2
    # The paper's claim: the hidden intermediate concept (Italy, France, ...) is
    # read out by the lens AT the descriptor token, mid-network, though it never
    # appears in the prompt. Scan all fitted layers at that token position.
    all_layers = lens.source_layers
    emit("#" * 100)
    emit("# PART 2 — does the hidden intermediate surface at the descriptor token?")
    emit("#" * 100)
    emit()
    for title, prompt, intermediate, answer, focus in PROMPTS:
        jl, _, input_ids = lens.apply(model, prompt, layers=all_layers, positions=None)
        toks = [tokenizer.decode([i]) for i in input_ids[0].tolist()]
        fi = focus_index(toks, focus)
        inter_id = first_token_id(tokenizer, intermediate)

        emit("=" * 100)
        emit(f"## {title}")
        emit(f"prompt: {prompt!r}")
        emit(f"descriptor token = {toks[fi]!r} (index {fi}); "
             f"hidden intermediate = {intermediate!r} "
             f"(token {inter_id} = {tokenizer.decode([inter_id])!r})")
        emit()
        # Rank of the intermediate token at the descriptor position, per layer.
        best_layer, best_rank = None, None
        for layer in all_layers:
            r = rank_of(jl[layer][fi], inter_id)
            if best_rank is None or r < best_rank:
                best_layer, best_rank = layer, r
        emit(f"  best lens rank of {intermediate!r} over all layers at that token: "
             f"rank {best_rank} at L{best_layer}")
        emit()
        # Show the top-10 J-lens tokens at that position for a few layers.
        for layer in [n // 4, n // 2, best_layer, (3 * n) // 4]:
            r = rank_of(jl[layer][fi], inter_id)
            emit(f"  L{layer:>2} @ {toks[fi]!r:>12}  "
                 f"[{intermediate} rank {r}]: {fmt_tokens(tokenizer, jl[layer][fi])}")
        emit()

    with open("results.md", "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print("\n[wrote results.md]")


if __name__ == "__main__":
    main()
