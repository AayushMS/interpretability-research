# Research Plans

Working roadmap. Update statuses in place; add findings links when an experiment lands.
Guiding question: **at what scale, and in which model families, does workspace-like structure
(the paper's "J-space") emerge -- and what is it good for in practice?**

## 01 -- jlens on Qwen3.5-0.8B, CPU baseline ✅ done (2026-07-07)

See [experiments/01-jlens-qwen-cpu/](experiments/01-jlens-qwen-cpu/). Established: the lens
works on open models and beats logit-lens; the hidden-intermediate multi-hop effect is absent
at 0.8B (best ranks 5792 / 100 / 338 for Italy/France/Brazil).

## 02 -- Scale sweep: where does the multi-hop effect emerge? ✅ CPU half done (2026-07-07)

**Goal:** find the model size where the hidden intermediate (Italy@boot etc.) becomes lens-visible.
- **Method:** run the full 93-prompt `lens-eval-multihop` set (copied into the experiment as
  `multihop.json`) across every model with a published lens on `neuronpedia/jacobian-lens`.
  Metric: best rank of the intermediate token over all layers x positions (a generous upper
  bound), plus task-competence and answer-preview controls. See
  [experiments/02-scale-sweep/](experiments/02-scale-sweep/).
- **Done on this box (CPU):** pythia-70m, gpt2 (both fp32 -- bf16 collapses pythia's logits
  into exact ties), Qwen3.5-0.8B, Qwen3-1.7B (bf16). Headline: best-anywhere intermediate
  visibility saturates by 0.8B (84->91% of prompts <=10 vs ~30% shuffled control), but it
  does NOT track task success and surfaces at end-of-prompt tokens in late layers --
  association/late retrieval, not the paper's descriptor-token workspace readout. See the
  experiment README.
- **Remaining:** gemma-3-270m / gemma-3-1b (just needs an HF login + Gemma license click --
  registry entries exist); Qwen3 4B / 8B / 14B on a rented GPU (Colab Pro / RunPod / vast.ai --
  personal account, NOT MARE AWS). A single A10/T4 session covers 4B-8B; 14B+ wants an A100.
  Same `sweep.py --model <key>` command everywhere; copy `raw/*.jsonl` back and re-run
  `report.py`.
- **Success criteria:** a rank-vs-parameters curve for the same prompts; identify the size
  where median best-rank drops under ~20.

## 03 -- Custom lens fitting 📋 planned

**Goal:** understand how much the lens depends on its fitting corpus.
- **Method:** fit our own lens on Qwen3.5-0.8B (the only size fit-able on CPU, and even that
  is slow -- consider the same rented GPU as 02) using (a) wikitext to replicate Neuronpedia's,
  and (b) a domain corpus (e.g. multi-hop factual prompts). Compare readout quality and the
  Part 2 ranks against the published checkpoint. The published one converged at 233 prompts
  from a 1000 budget -- check how quality degrades at 50 / 100 prompts.
- **Success criteria:** a table of lens-fit corpus/size vs readout coherence on held-out prompts.

## 04 -- Workspace property probes on open models 📋 planned

**Goal:** test the paper's five J-space properties (reportability, modulation, reasoning,
flexibility, selective involvement) at whatever scale experiment 02 says the effect exists.
- **Method:** start with the two cheapest: *reportability* (ask the model what it's thinking;
  compare to lens readout at the same positions) and *modulation* ("think about X while
  answering Y"; check whether X's lens rank moves). Instruction-following needs the instruct
  variants, not base models.
- **Success criteria:** quantified agreement between self-report and lens readout vs a
  shuffled-baseline control.

## 05 -- Injection / steering ❓ exploratory

**Goal:** the paper doesn't just read the workspace, it writes to it (concept swap/injection).
- **Method:** check what the open-source `jlens` API exposes beyond `apply()` -- if transport
  matrices are invertible enough to construct write-vectors, try swapping an intermediate
  concept (Italy -> Spain) and observe whether the answer flips (euro -> euro is a bad test;
  use currency pairs that differ, e.g. boot->Japan swap should flip euro -> yen).
- **Risk:** may simply not be supported by the reference implementation; timebox it.

## 06 -- Practical monitoring prototype 💡 someday

**Goal:** the original question that started this repo: can lens readouts serve as a runtime
monitoring surface (the thing you *cannot* do on a hosted API like Bedrock)?
- **Method:** wrap a self-hosted model in a thin inference server that streams, alongside each
  generated token, the J-lens top-k at 2-3 mid layers; flag divergences between "disposed to
  say" and "actually said". This only makes sense after 02 identifies a scale where readouts
  carry real signal.

## Longer-term ideas (unordered)

- Other model families (Llama, Gemma) once lenses exist or 03 makes fitting cheap -- tests
  whether findings are Qwen-specific.
- Compare against Neuronpedia's hosted readouts for the same model/prompts (sanity-check our
  local pipeline).
- Write up the scale-sweep result as a short blog post if the curve is clean.
