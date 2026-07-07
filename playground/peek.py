#!/usr/bin/env python3
# MIT license; the jlens dependency (anthropics/jacobian-lens) is Apache-2.0.
"""peek.py -- look inside a model while it reads YOUR prompt.

The one-command playground for this repo. Give it any prompt and it prints,
layer by layer, what the model is "disposed to say" at that depth (Jacobian
lens readout). Optionally watch a word and see its rank climb (rank 1 = the
model's top pick).

Examples:
    python peek.py --prompt "The capital of France is"
    python peek.py --prompt "The capital of France is" --watch Paris
    python peek.py --prompt "Fact: The currency of the country shaped like a boot is" \
        --watch euro --watch Italy
    python peek.py --model gpt2 --prompt "The opposite of hot is" --watch cold
    python peek.py --prompt "..." --position -2      # peek at an earlier token
    python peek.py --prompt "..." --logit-lens        # the (worse) baseline

Models (same registry as experiments/02-scale-sweep): pythia-70m, gpt2,
qwen3.5-0.8b (default), qwen3-1.7b, ... Run with --list to see all.
"""

from __future__ import annotations

import argparse
import sys
import time

import torch
import transformers

import jlens

# reuse the model registry from experiment 02 (single source of truth)
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "experiments", "02-scale-sweep"))
from sweep import (  # noqa: E402
    DTYPES,
    LENS_REPO,
    LENS_REVISION,
    MODELS,
    variant_token_ids,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen3.5-0.8b", choices=MODELS)
    ap.add_argument("--prompt", help="the text to peek into")
    ap.add_argument("--watch", action="append", default=[],
                    help="word to track the rank of (repeatable)")
    ap.add_argument("--position", type=int, default=-1,
                    help="which token position to read out (default -1 = last)")
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--logit-lens", action="store_true",
                    help="skip the Jacobian (plain logit-lens baseline)")
    ap.add_argument("--list", action="store_true", help="list models and exit")
    args = ap.parse_args()

    if args.list or not args.prompt:
        for k, (hf_id, _, dtype, note) in MODELS.items():
            print(f"{k:>14}  {hf_id:<32} {dtype}  {note}")
        if not args.list:
            print("\nerror: --prompt is required", file=sys.stderr)
            sys.exit(2)
        return

    hf_id, lens_file, dtype, _ = MODELS[args.model]
    print(f"loading {hf_id} on CPU (first time downloads to ~/.cache/huggingface)...")
    t0 = time.perf_counter()
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        hf_id, dtype=DTYPES[dtype]
    )
    hf_model.eval()
    tok = transformers.AutoTokenizer.from_pretrained(hf_id)
    model = jlens.from_hf(hf_model, tok)
    lens = jlens.JacobianLens.from_pretrained(
        LENS_REPO, filename=lens_file, revision=LENS_REVISION
    )
    print(f"loaded in {time.perf_counter() - t0:.0f}s ({model.n_layers} layers)\n")

    jl, model_logits, input_ids = lens.apply(
        model, args.prompt, positions=[args.position],
        use_jacobian=not args.logit_lens,
    )
    toks = [tok.decode([i]) for i in input_ids[0].tolist()]
    pos_tok = toks[args.position]

    watch_ids = {w: variant_token_ids(tok, w) for w in args.watch}

    def describe(logits: torch.Tensor) -> str:
        top = "  ".join(repr(tok.decode([t]))
                        for t in logits.topk(args.topk).indices.tolist())
        marks = []
        for w, ids in watch_ids.items():
            best = min(int((logits > logits[t]).sum()) + 1 for t in ids)
            marks.append(f"{w}: r{best}")
        return top + ("    [" + ", ".join(marks) + "]" if marks else "")

    kind = "logit-lens (baseline)" if args.logit_lens else "Jacobian lens"
    print(f"prompt: {args.prompt!r}")
    print(f"reading out at token {args.position} = {pos_tok!r} with the {kind}")
    if args.watch:
        print(f"watching rank of: {', '.join(args.watch)}  (r1 = top pick)")
    print()
    for layer in lens.source_layers:
        print(f"  L{layer:>2}  {describe(jl[layer][0])}")
    print()
    print(f"  MODEL actually says next: {describe(model_logits[0])}")


if __name__ == "__main__":
    main()
