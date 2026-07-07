# GUIDE — run these experiments yourself

This is the *"I just want to poke at a model's brain"* guide. No interpretability
background needed. Every command below is copy-paste ready and runs on a plain
laptop CPU (this repo was built on a 7.6 GB RAM WSL2 box with no GPU).

## The idea in 60 seconds

A language model reads your prompt through a stack of **layers** (Qwen3.5-0.8B has
24). Each layer refines an internal "working state". Normally you only see the
final output — the next word. The **Jacobian lens** (open-sourced by Anthropic) is
a tool that stops at any layer and asks:

> *"If the model had to speak right now, from this depth — what would it say?"*

So you can literally watch an answer **form**, layer by layer, before the model
says anything. Anthropic's paper claims big models keep a "global workspace" of
in-progress thoughts readable this way. This repo tests how much of that is true
for small open models. Rank is the main number you'll see everywhere:
**rank 1 = the lens's top pick for the next word.** Rank 50,000 = nowhere.

## One-time setup (~5 minutes + downloads)

```bash
cd experiments/02-scale-sweep
python3 -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU box: ALWAYS first
pip install -r requirements.txt
```

That's it. Models and lenses download automatically to `~/.cache/huggingface`
on first use (0.8B model ≈ 1.7 GB, pythia-70m ≈ 160 MB).

## Your first look inside a model (2 minutes)

```bash
cd ../../playground        # (still inside the venv from the setup step)
python peek.py --model pythia-70m --prompt "The capital of France is" --watch Paris
```

You'll see the lens's top-8 words at every layer, plus `Paris: r<N>` — the rank
of the word you're watching. Then try the real thing:

```bash
python peek.py --prompt "The capital of France is" --watch Paris
```

(default model is `qwen3.5-0.8b`; ~1 min load on CPU). Watch `Paris` climb from
rank thousands at early layers to rank ~1 near the end. **That's a thought
forming.** Everything else in this repo is variations of this.

---

## The experiment menu

Each of these is a real experiment type from the interpretability literature.
All of them work on this box.

### 1. Watch an answer form (the basic readout)

```bash
python peek.py --prompt "The opposite of hot is" --watch cold
python peek.py --prompt "2 + 2 =" --watch 4
python peek.py --prompt "Roses are red, violets are" --watch blue
```

**What to look for:** at which layer does the answer enter the top-10? Early =
the task is "easy" for the model; last-minute = the model computes it late.

### 2. Lens vs baseline (why the Jacobian matters)

Run the same prompt with the plain "logit lens" (an older, simpler tool):

```bash
python peek.py --prompt "The opposite of hot is" --watch cold
python peek.py --prompt "The opposite of hot is" --watch cold --logit-lens
```

**What to look for:** mid-layer readouts. The baseline is usually noise there;
the Jacobian lens is on-topic. This is experiment 01's positive result — you can
reproduce it in 2 minutes.

### 3. Hunt for hidden thoughts (the paper's headline claim)

Multi-hop prompts force the model to *think of something it never says*. "The
currency of the country shaped like a boot" → the model must think **Italy** to
answer **euro**. Is "Italy" readable inside?

```bash
python peek.py \
  --prompt "Fact: The currency used in the country shaped like a boot is" \
  --watch euro --watch Italy
# also peek at the word "boot" itself (second-to-last token, position -2/-3...):
python peek.py \
  --prompt "Fact: The currency used in the country shaped like a boot is" \
  --watch Italy --position -4
```

**What we found (exp 01/02):** at small scale the answer (`euro`) becomes
visible, but the hidden step (`Italy`) mostly does NOT. That's a real negative
result — Anthropic's effect appears to need bigger models.

### 4. Compare model sizes (the scale sweep)

The full 93-prompt version of #3, across every model that fits on the box:

```bash
cd ../experiments/02-scale-sweep
python sweep.py --list                       # see all models + notes
python sweep.py --model pythia-70m           # ~1 min
python sweep.py --model gpt2                 # ~2 min
python sweep.py --model qwen3.5-0.8b         # ~15 min
python report.py                             # rebuilds results.md from raw/
```

Runs are resumable (Ctrl-C is safe; re-run the same command and it continues).
For anything slow, run it in the background and watch the log:

```bash
nohup python sweep.py --model qwen3-1.7b > sweep.log 2>&1 &
tail -f sweep.log        # Ctrl-C stops the tail, not the run
```

### 5. Add a new model family (is this Qwen-specific?)

Google's Gemma models have pre-fitted lenses too, but need a (free) license
click. One-time:

```bash
pip install "huggingface_hub[cli]"
hf auth login            # paste a token from https://huggingface.co/settings/tokens
# then visit https://huggingface.co/google/gemma-3-270m and click "agree"
python sweep.py --model gemma-3-270m
python sweep.py --model gemma-3-1b
python report.py
```

Two new points on the scale curve, from a different lab's models — this tests
whether the findings are a Qwen quirk. (Registry already has them; nothing to
code.)

### 6. Design your own prompt set

Open `experiments/02-scale-sweep/multihop.json` — the schema is 4 fields:

```json
{"name": "boot-currency",
 "prompt": "Fact: The currency used in the country shaped like a boot is",
 "intermediates": ["Italy"],
 "target": "euro"}
```

Copy the file, write 10-20 prompts of your own theme (movies? football?
Nepali geography?), point `sweep.py` at it by replacing `multihop.json`, and
you've built a benchmark nobody has run before. Rules of thumb: the
*intermediate* must never appear in the prompt; the *target* should be a single
word; end the prompt right before the answer.

### 7. Prompt surgery (one-word ablations)

Take a prompt where something works and break it one word at a time:

```bash
python peek.py --prompt "The currency of the country shaped like a boot is" --watch Italy
python peek.py --prompt "The currency of the country shaped like a shoe is" --watch Italy
python peek.py --prompt "The currency of Italy is" --watch euro
```

**What to look for:** which single word carries the effect? Does saying "Italy"
outright change *when* (which layer) "euro" appears vs making the model infer it?

### 8. Where in the sentence does the model think? (position scan)

The lens reads out at any token, not just the last one. Scan a few positions:

```bash
python peek.py --prompt "Fact: The capital of Japan is Tokyo. The capital of France is" \
  --watch Paris --position -1
# now try -2, -5, -8 ... where does 'Paris' first become visible?
```

### 9. Rent-a-GPU day (the big models)

Everything above, unchanged, on Colab/RunPod/vast.ai (personal account) covers
the models this box can't hold:

```bash
pip install torch  # (GPU wheel; no --index-url needed there)
pip install -r requirements.txt
python sweep.py --model qwen3-4b
python sweep.py --model qwen3-8b
python sweep.py --model qwen3-14b     # A100-class
```

Copy the produced `raw/*.jsonl` files back into the repo and run `report.py` —
the results table grows automatically. This fills in the interesting region of
the scale curve: Anthropic's own multi-hop demos start at ~4B.

---

## Tips (learned the hard way)

- **CPU torch first.** On a no-GPU box always `pip install torch --index-url
  https://download.pytorch.org/whl/cpu` *before* `-r requirements.txt`, or pip
  downloads a ~2.5 GB CUDA wheel you can't use.
- **RAM is the wall, not speed.** 7.6 GB fits 0.8B comfortably, 1.7B barely,
  4B never. If a run dies with "Killed", it was the kernel OOM-killer.
- **Tiny models need fp32.** We found pythia-70m's logits collapse into
  thousands of *exact ties* in bf16 (its final-layer values are too large for
  bf16's coarse grid), which silently corrupts every rank. The model registry
  in `sweep.py` pins the right dtype per model, and every record carries a
  `model_max_ties` sanity counter so this can't sneak through again.
- **One model per process.** Run sweeps sequentially, not in parallel — each
  loads a full model into RAM.
- **First runs are slow, later runs are fast.** Downloads are cached in
  `~/.cache/huggingface` (delete old models there if disk fills up).
- **Rank beats eyeballing.** "The word looks kind of visible in the top-10" is
  vibes; "rank 3 at L18 vs rank 5792 baseline" is a result.
- **One prompt proves nothing.** Any single prompt can be a fluke; that's why
  sweep.py runs 93. For quick peeks, try at least 3-5 rephrasings.
- **Negative results count.** "The effect does not appear at 0.8B" is the most
  useful thing this repo has produced so far. Don't tweak prompts until the
  answer you want appears — that's fooling yourself.
- **Write it down.** Repo convention: every experiment directory gets a
  `README.md` (what/why/how), a `results.md` regenerated by the code, and its
  own `requirements.txt`. Your future self is the audience.
