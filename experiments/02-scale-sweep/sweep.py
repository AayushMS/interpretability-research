#!/usr/bin/env python3
# MIT license; the jlens dependency (anthropics/jacobian-lens) is Apache-2.0.
"""Experiment 02: scale sweep of Jacobian-lens visibility on multi-hop prompts.

For ONE model (picked with --model), run all 93 prompts from multihop.json
(Anthropic's lens-eval-multihop set, Apache-2.0) and record, per prompt:

  1. task competence  -- rank of the expected ANSWER in the model's actual
     next-token distribution (can the model do the task at all?)
  2. answer preview   -- best lens rank of the answer at the last prompt
     token, over all fitted layers (does the lens see the answer forming?)
  3. hidden intermediate -- best lens rank of the unstated intermediate
     concept (e.g. "Italy" in the boot->currency hop) over ALL layers and ALL
     prompt positions. This is the paper's headline effect, measured as a
     generous upper bound: if even the best (layer, position) pair ranks the
     intermediate poorly, the workspace readout is absent at this scale.

Appends one JSON line per prompt to raw/<model-key>.jsonl (resumable: already
-done prompts are skipped on re-run). Run report.py afterwards to build
results.md from all raw/*.jsonl files.

Usage:
    python sweep.py --model qwen3.5-0.8b
    python sweep.py --list          # show the model registry
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import transformers

import jlens

HERE = os.path.dirname(os.path.abspath(__file__))
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_REVISION = "qwen-n1000"

# key -> (HF model id, lens file inside the neuronpedia/jacobian-lens repo,
#         dtype, note).
# dtype: fp32 whenever the model is small enough -- pythia-70m's final-layer
# logits are so large that bf16 collapses thousands of tokens into exact ties
# (rank becomes meaningless); bf16 only where RAM leaves no choice (Qwen 0.8B+
# on a 7.6 GB box, and anything on a rented GPU).
MODELS = {
    # -- fit in ~7.6 GB RAM on CPU (this box) --
    "pythia-70m": (
        "EleutherAI/pythia-70m-deduped",
        "pythia-70m-deduped/jlens/Salesforce-wikitext/pythia-70m-deduped_jacobian_lens.pt",
        "fp32",
        "CPU-friendly",
    ),
    "gpt2": (
        "openai-community/gpt2",
        "gpt2-small/jlens/Salesforce-wikitext/gpt2_jacobian_lens.pt",
        "fp32",
        "CPU-friendly",
    ),
    "qwen3.5-0.8b": (
        "Qwen/Qwen3.5-0.8B",
        "qwen3.5-0.8b/jlens/Salesforce-wikitext/Qwen3.5-0.8B_jacobian_lens.pt",
        "bf16",
        "CPU-friendly (exp 01 model)",
    ),
    "qwen3-1.7b": (
        "Qwen/Qwen3-1.7B",
        "qwen3-1.7b/jlens/Salesforce-wikitext/Qwen3-1.7B_jacobian_lens.pt",
        "bf16",
        "borderline on 7.6 GB RAM",
    ),
    # -- license-gated on HF: run `hf auth login` and accept the Gemma license --
    "gemma-3-270m": (
        "google/gemma-3-270m",
        "gemma-3-270m/jlens/Salesforce-wikitext/gemma-3-270m_jacobian_lens.pt",
        "fp32",
        "gated (accept Gemma license)",
    ),
    "gemma-3-1b": (
        "google/gemma-3-1b-pt",
        "gemma-3-1b/jlens/Salesforce-wikitext/gemma-3-1b-pt_jacobian_lens.pt",
        "bf16",
        "gated (accept Gemma license)",
    ),
    # -- too big for this box: rent a GPU (Colab/RunPod), same command --
    "qwen3-4b": (
        "Qwen/Qwen3-4B",
        "qwen3-4b/jlens/Salesforce-wikitext/Qwen3-4B_jacobian_lens.pt",
        "bf16",
        "needs GPU",
    ),
    "qwen3-8b": (
        "Qwen/Qwen3-8B",
        "qwen3-8b/jlens/Salesforce-wikitext/Qwen3-8B_jacobian_lens.pt",
        "bf16",
        "needs GPU",
    ),
    "qwen3-14b": (
        "Qwen/Qwen3-14B",
        "qwen3-14b/jlens/Salesforce-wikitext/Qwen3-14B_jacobian_lens.pt",
        "bf16",
        "needs GPU (A100-class)",
    ),
}
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}


def variant_token_ids(tok, word: str) -> dict[int, str]:
    """Leading token ids for the surface forms of `word` (with/without a
    leading space, original/capitalized/lower case).

    The space-prefixed leading token always counts: it is how the word starts
    being emitted mid-text (same convention as experiment 01). The bare form
    (start-of-line / right after a quote) only counts when its leading token
    is a solid prefix of the word -- at least 3 chars, or the whole word for
    short words like '8' -- to stop degenerate matches like 'Italy' -> 'It'.
    """
    ids: dict[int, str] = {}
    for w in {word, word.capitalize(), word.lower()}:
        toks = tok(" " + w, add_special_tokens=False).input_ids
        if toks:
            ids[toks[0]] = w
        toks = tok(w, add_special_tokens=False).input_ids
        if toks:
            lead = tok.decode([toks[0]]).strip()
            if (
                lead
                and w.lower().startswith(lead.lower())
                and len(lead) >= min(3, len(w))
            ):
                ids[toks[0]] = w
    return ids


def ranks_of(logits: torch.Tensor, token_id: int) -> torch.Tensor:
    """1-based rank of `token_id` at every position. logits: [pos, vocab]."""
    return (logits > logits[:, token_id : token_id + 1]).sum(-1) + 1


def chunked(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS, help="model key to sweep")
    ap.add_argument("--list", action="store_true", help="list models and exit")
    ap.add_argument(
        "--mem-budget-mb",
        type=int,
        default=400,
        help="cap on readout memory; layers are processed in chunks under it",
    )
    args = ap.parse_args()

    if args.list or not args.model:
        for k, (hf_id, _, dtype, note) in MODELS.items():
            print(f"{k:>14}  {hf_id:<32} {dtype}  {note}")
        return

    hf_id, lens_file, dtype, _ = MODELS[args.model]
    out_path = os.path.join(HERE, "raw", f"{args.model}.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            done = {json.loads(l)["name"] for l in f if l.strip() and "name" in l}

    items = json.load(open(os.path.join(HERE, "multihop.json")))["items"]
    # shuffled control: each prompt is also scored against an intermediate
    # borrowed from another prompt (deterministic offset; never overlapping
    # the prompt's own intermediates or appearing in its text). If the real
    # intermediate doesn't beat this, the "visibility" is just the noise
    # floor of a best-over-layers-x-positions metric.
    for idx, it in enumerate(items):
        off = 7
        while True:
            cand = items[(idx + off) % len(items)]["intermediates"]
            own = {w.lower() for w in it["intermediates"]}
            if not own & {w.lower() for w in cand} and not any(
                w.lower() in it["prompt"].lower() for w in cand
            ):
                break
            off += 1
        it["control"] = cand
    todo = [it for it in items if it["name"] not in done]
    print(f"[{args.model}] {len(todo)}/{len(items)} prompts to run -> {out_path}")

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
    n_params = sum(p.numel() for p in hf_model.parameters())
    vocab = int(hf_model.get_output_embeddings().weight.shape[0])
    print(
        f"loaded in {time.perf_counter() - t0:.0f}s: {n_params/1e6:.0f}M params, "
        f"{model.n_layers} layers, vocab {vocab}, lens {lens!r}",
        flush=True,
    )

    meta = {
        "meta": True,
        "model_key": args.model,
        "hf_id": hf_id,
        "lens_file": lens_file,
        "lens_revision": LENS_REVISION,
        "n_params": n_params,
        "n_layers": model.n_layers,
        "vocab": vocab,
        "dtype": dtype,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }
    if not done:
        with open(out_path, "a") as f:
            f.write(json.dumps(meta) + "\n")

    layers = lens.source_layers
    for i, it in enumerate(todo):
        t = time.perf_counter()
        prompt, name = it["prompt"], it["name"]
        inter_ids: dict[int, str] = {}
        for w in it["intermediates"]:
            inter_ids.update(variant_token_ids(tok, w))
        ctrl_ids: dict[int, str] = {}
        for w in it["control"]:
            ctrl_ids.update(variant_token_ids(tok, w))
        target_ids = variant_token_ids(tok, it["target"])

        rec = {
            "name": name,
            "intermediates": it["intermediates"],
            "control": it["control"],
            "target": it["target"],
            # filled below:
            "inter_best": None,  # {rank, layer, pos, token} over layers x positions
            "inter_best_by_layer": {},  # layer -> best rank over positions
            "control_best": None,  # same metric for the borrowed intermediate
            "answer_model_rank": None,  # rank of target in model next-token dist
            "answer_lens_best": None,  # {rank, layer} for target at last position
            "answer_lens_rank_by_layer": {},  # layer -> target rank at last pos
            "model_top5": None,
        }

        # size layer chunks so lens logits stay under the memory budget
        n_tok = len(tok(prompt).input_ids)
        per_layer_bytes = max(1, n_tok * vocab * 4)
        chunk = max(1, (args.mem_budget_mb * 1_000_000) // per_layer_bytes)

        toks_text: list[str] | None = None
        for group in chunked(layers, chunk):
            jl, model_logits, input_ids = lens.apply(
                model, prompt, layers=group, positions=None
            )
            if toks_text is None:
                toks_text = [tok.decode([i]) for i in input_ids[0].tolist()]
                # positions where an intermediate token literally appears in
                # the prompt don't count (reading out a copy is trivial)
                banned = torch.tensor(
                    [
                        any(
                            t.strip().lower() == w.lower()
                            for w in rec["intermediates"]
                        )
                        for t in toks_text
                    ]
                )
                if rec["answer_model_rank"] is None:
                    for tid in target_ids:
                        r = int(ranks_of(model_logits, tid)[-1])
                        if (
                            rec["answer_model_rank"] is None
                            or r < rec["answer_model_rank"]
                        ):
                            rec["answer_model_rank"] = r
                    top5 = model_logits[-1].topk(5).indices.tolist()
                    rec["model_top5"] = [tok.decode([t]) for t in top5]
                    # sanity: exact ties at the max logit mean the dtype is
                    # too coarse for this model and ranks can't be trusted
                    rec["model_max_ties"] = int(
                        (model_logits[-1] == model_logits[-1].max()).sum()
                    )

            for layer in group:
                logits = jl[layer]  # [pos, vocab]
                # hidden intermediate: best rank over (non-banned) positions
                layer_best = None
                for tid, w in inter_ids.items():
                    r = ranks_of(logits, tid)
                    r[banned] = vocab + 1
                    pos = int(r.argmin())
                    rank = int(r[pos])
                    if layer_best is None or rank < layer_best:
                        layer_best = rank
                    if rec["inter_best"] is None or rank < rec["inter_best"]["rank"]:
                        rec["inter_best"] = {
                            "rank": rank,
                            "layer": layer,
                            "pos": pos,
                            "pos_token": toks_text[pos],
                            "matched": w,
                        }
                rec["inter_best_by_layer"][str(layer)] = layer_best
                # control: same best-over-positions search for the borrowed
                # word (guaranteed absent from the prompt, so no ban needed)
                for tid, w in ctrl_ids.items():
                    r = ranks_of(logits, tid)
                    pos = int(r.argmin())
                    rank = int(r[pos])
                    if (
                        rec["control_best"] is None
                        or rank < rec["control_best"]["rank"]
                    ):
                        rec["control_best"] = {
                            "rank": rank,
                            "layer": layer,
                            "matched": w,
                        }
                # answer preview: target rank at the last prompt position
                ans_best = None
                for tid in target_ids:
                    r = int(ranks_of(logits, tid)[-1])
                    if ans_best is None or r < ans_best:
                        ans_best = r
                rec["answer_lens_rank_by_layer"][str(layer)] = ans_best
                if ans_best is not None and (
                    rec["answer_lens_best"] is None
                    or ans_best < rec["answer_lens_best"]["rank"]
                ):
                    rec["answer_lens_best"] = {"rank": ans_best, "layer": layer}
            del jl, model_logits

        with open(out_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        ib = rec["inter_best"]
        print(
            f"[{i + 1:>2}/{len(todo)}] {name:<32} "
            f"inter best r{ib['rank']}@L{ib['layer']} "
            f"(ctrl r{rec['control_best']['rank']}) "
            f"| answer: model r{rec['answer_model_rank']}, "
            f"lens best r{rec['answer_lens_best']['rank']}"
            f"@L{rec['answer_lens_best']['layer']} "
            f"({time.perf_counter() - t:.1f}s)",
            flush=True,
        )

    print(f"[{args.model}] done in {(time.perf_counter() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
