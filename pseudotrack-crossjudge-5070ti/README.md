# pseudotrack-crossjudge-5070ti

McMiner pseudocode track on two local Ollama models, run in **both directions** so that
neither model ever grades its own output.

| Direction | Miner | Judge |
|---|---|---|
| **A** | `gpt-oss:20b` | `qwen3.6:27b` |
| **B** | `qwen3.6:27b` | `gpt-oss:20b` |

Built for an **RTX 5070 Ti (16 GB VRAM) with 32 GB system RAM**. Local Ollama only — no API key,
no vendor SDK, no network. The only Python dependencies are `requests` and `tqdm`.

---

## ⚠️ Read this before interpreting a correct-only run

**In a correct-bags-only run the judge makes ZERO calls, in both directions.**

Ground truth for every correct-only bag is `NONE`, so neither scorer needs a judge — they decide
by rule:

| Scorer | Method | Rule |
|---|---|---|
| `src/compute_eval_metrics_multi.py` | `correct_bag_rule` | bag is correct ⇔ model predicted no misconception |
| `src/evaluate_single_multi_predictions.py` | `empty_check` | same, at the single-code level |

A judge is only consulted when there is a ground-truth misconception *description* to compare a
prediction against, and in a correct-only run there is none.

**Consequence: directions A and B differ only in the miner.** Both judges are still built,
preflighted and probed — so the same folder can run the full track later, where the judge does
matter — but neither is asked anything.

This is a feature, not a limitation. It makes the correct-only run the cleanest measurement in the
whole project: the result cannot be contaminated by judge choice, judge bias, judge token budget,
or judge timeouts, all of which were shown to move scores by 7–8 points elsewhere in this work.

If you want the judge to actually do something, use `run_full.sh`.

---

## Quick start

```bash
# 0. one-time: pull the bases and build all four models
ollama pull gpt-oss:20b
ollama pull qwen3.6:27b
bash scripts/build_models.sh        # reads back the GPU/CPU split -- check it

# 1. preflight only, no mining
DRY_RUN=1 bash run_correct_only.sh

# 2. ALWAYS smoke first: 6 codes/bags per arm, a few minutes
SMOKE=1 bash run_correct_only.sh

# 3. the real run: both directions, four arms
bash run_correct_only.sh
```

On Windows use Git Bash and point `PYTHON` at your interpreter:

```bash
PYTHON=../.venv/Scripts/python.exe bash run_correct_only.sh
```

Dependencies:

```bash
pip install -r requirements.txt     # requests, tqdm -- that is the whole list
```

Do **not** `pip install -r` the parent repo's `requirements.txt`: it lists `pathlib` and `argparse`,
stdlib backports that fail to build on Python 3.13.

### Useful variations

```bash
DIRECTIONS=A bash run_correct_only.sh          # only gpt-oss mining
ARMS=baseline bash run_correct_only.sh         # only the baseline arm
CORRECT_PASSES=5 bash run_correct_only.sh      # 5 shufflings -> ~20 bags/arm
FORCE=1 bash run_correct_only.sh               # re-mine bags that already exist
RUN_EVAL=0 bash run_correct_only.sh            # mine only, score later
OLLAMA_THINK=medium bash run_correct_only.sh   # override the thinking setting
```

---

## What this run measures

Every **correct** program in the dataset, partitioned into bags exactly the way McMiner's bag
former does, mined, and scored. A bag counts as correct only if the model predicts *no*
misconception, so the run measures one thing: **how often each miner invents a misconception in
code that has nothing wrong with it** — the false-positive rate, or specificity.

Per arm per direction at the default 1 pass:

| Step | Calls |
|---|---|
| McMiner-M, correct-only bags | 4 |
| McMiner-S, every correct code | 96 |
| **Total mining** | **100** |
| Judge | **0** |

With four arms and two directions that is **800 mining calls**, no judge calls.

McMiner-S on the 209 corrupted codes is deliberately skipped in this mode: with no misconception
bags to align them to, those calls would be pure waste.

---

## Why 96 correct files are 19 programs

`dataset/pseudocode_track/pseudocode_codes_none/` holds **96** files, but they span only **19
distinct `problem_id`s**, and every one carries the literal string `NONE` as its code. At
prompt-build time the pipeline substitutes that problem's correct answer-key pseudocode, so the
model is shown **19 unique programs**, each appearing between **1 and 12 times** under different
inapplicable-misconception labels.

A file named `problem_130_misc_38.json` in that directory does **not** mean "a correct code for
misconception 38". It means "misconception 38 was judged *inapplicable* to problem 130". All such
files for problem 130 resolve to the same program.

Three consequences:

1. **The effective sample is 19, not 96** — no number of passes changes that.
2. **Per-row rates are weighted by an artefact.** Problem 60 contributes 12 rows; problems 73, 121
   and 242 contribute one each. `summarize.py` and `compare_miners.py` therefore also print a
   per-program majority-vote view, which removes the weighting.
3. **The repetition accidentally measures self-consistency.** The same program, the same prompt,
   the same temperature, asked up to 12 times. Both reporting scripts count programs that got
   *both* answers; that is pure model noise and it bounds how much of any A-vs-B gap is real.

Bag formation deduplicates by problem, so `CORRECT_PASSES=1` gives 4 bags covering all 19 programs
(sizes 5/5/5/4). `CORRECT_PASSES=5` re-shuffles five times for 20 bags — still 19 programs.

---

## What REF does in a correct-only run

Near enough to nothing, **by construction**. The REF arm injects the APR-retrieved reference
solution, which is keyed by `(problem_id, misconception_id)`. Correct-only codes carry
`misconception_id = None`, so they never join that table and always render the
`No reference solution available for this submission.` placeholder.

So in a correct-only run, **`ref` ≈ `baseline` and `rag_ref` ≈ `rag`** up to sampling noise. Keep
the arms anyway — they cost little and they confirm the plumbing — but do not report a REF effect
on correct bags as a finding. The RAG arms *do* differ: correct codes get a real retrieved
shortlist from `dataset/retrival_correct_codes.csv`, keyed by `problem_id`.

---

## Hardware notes (RTX 5070 Ti, 16 GB)

| Model | Weights | Expected `ollama ps` |
|---|---|---|
| `gpt-oss:20b` @16K | ~13.8 GB | **100% GPU** |
| `qwen3.6:27b` @16K | ~17 GB | **~75%/25% GPU/CPU**, ~2–5 GB spilled to RAM |

The 32 GB of system RAM is what makes qwen3.6 usable here. On an 8 GB laptop with 15.7 GB of RAM
the same model paged to **disk** and generated at 0.6–1.3 tokens/s, which made a full track take
weeks.

`scripts/build_models.sh` loads each model once and prints the split. Read it:

- `100% GPU` — ideal
- partial GPU/CPU — expected for qwen3.6; fine **if** system RAM is free
- a **0 GB working set** or mostly-CPU — it is paging to disk. Stop, free RAM, and retry. Lower
  `num_ctx` to 8192 in `Modelfile.qwen36-judge` / `Modelfile.qwen36-miner` to buy back ~1.3 GB.

The two models are never resident at once: `unload_model` stops the previous one before the next
loads, because 17 GB and 13.8 GB cannot co-reside on a 16 GB card.

---

## Thinking: the setting that silently destroys a run

Both models reason before answering, and **those tokens come out of the same budget as the
answer**. When the budget runs out the reply is truncated — and the parsers' defaults are not
neutral, so a clipped reply silently becomes a *score*, or a "no misconception predicted", rather
than an error.

`think: false` is **not** a universal off switch. Measured on Ollama 0.32.14:

| Model | Setting | Result |
|---|---|---|
| gpt-oss | `think=false` | `content=''`, `done_reason='length'` — reasoned until the budget ran out, returned **nothing** |
| gpt-oss | `think='low'` | answered in 17 tokens, 5.5 s |
| qwen3.6 | `think=true` | **>10 min** per call |
| qwen3.6 | `think=false` | ~160 s per call |

A reasoning-only model needs a **level**; qwen3.6 needs the **boolean**. That mapping lives in
`THINK_BY_MODEL` in `utils/ollama_client.py` and is applied **by model name**, so nothing can leak
from one step into the next. The client raises rather than proceeding if gpt-oss is handed `false`.

Override for a whole run to measure the cost of thinking:
`OLLAMA_THINK=low|medium|high|false|default`.

---

## What went wrong before, and why this bundle is safe from it

A previous attempt to use **qwen3.6:27b as a judge** produced numbers that had to be thrown away:
**31.2% of its judge calls (145 of 465) timed out**, and the pipeline scored every timeout as
`match: False`. There is no "unknown" state, so a call that never returned was indistinguishable,
in the metrics, from a judge that looked and said no. The run burned ~91 hours of calendar time
and its results are unusable.

Root cause: the judge client was built on the OpenAI SDK with **no `timeout` and no `max_retries`
argument**, inheriting defaults of 600 s × 3 attempts — so one slow call consumed ~30 minutes
before being recorded as a wrong answer.

This bundle avoids that in four ways:

1. **No OpenAI SDK.** `utils/ollama_client.py` talks to Ollama's native `/api/chat` with
   `requests`, and its timeout is **configurable and explicit**: `OLLAMA_TIMEOUT`, default **900 s**.
2. **Thinking is turned off for qwen3.6 by name**, which is what made judge calls slow enough to
   time out in the first place (>10 min → ~160 s).
3. **In correct-only mode the judge is never called at all**, so the failure mode cannot occur.
4. **`JUDGE_ABORT_AFTER=5`** stops a run after five *consecutive* judge failures rather than
   writing a fabricated low score.

**Known remaining gap.** `JUDGE_ABORT_AFTER` is consecutive-only and cannot catch a *scattered*
failure pattern — which is exactly the pattern the failed run had. It never tripped. So for
`run_full.sh`, **check the judge-failure counts in the summary before quoting any number.** A
rate-based abort is the right fix and is not implemented here.

---

## Output layout

```
results/dir<A|B>_<model>_<arm>_correctbags/
    single/predictions.json                   per-code predictions (+ raw response)
    single/summary.json                       parse rates, NONE statistics
    multi/multi_predictions.json              bag-level predictions
    multi/multi_summary.json                  bag-formation parameters
    single_multi/grouped_predictions.json     per-code predictions aligned into bags

results/evaluations/dir<A|B>_<model>_<arm>_correctbags/
    single_multi/evaluation_metrics.json      McMiner-S metrics
    single_multi/bag_evaluation_results.json  per-bag detail
    multi/evaluation_metrics.json             McMiner-M metrics
    multi/claude_evaluation_results.json      per-bag verdicts
```

Two naming traps, both inherited from upstream McMiner and both harmless:

- **`claude_evaluation_results.json`** and the field `evaluation_method: "claude_existing"` are
  legacy strings. **No Anthropic model is used anywhere in this bundle.** The authoritative field
  is `judge_model`.
- The directory prefix is `dirA_` / `dirB_`, which is what `summarize.py --direction` and
  `compare_miners.py` expect. It must stay in sync with `MODEL_TAG` in `scripts/_common.sh`.

**Reading a result directory:** check the parse-failure and judge-failure counts *before* reading
any accuracy.

---

## Reporting

Both entry points print a summary at the end. Additionally:

```bash
python scripts/summarize.py --mode correct_only --model gpt-oss-mcminer:latest --direction A
python scripts/compare_miners.py                  # A vs B, head to head
```

`compare_miners.py` prints three views in increasing order of trustworthiness:

1. **bag level** — the headline, but n is tiny (4 bags at 1 pass, so one bag is 25 pp)
2. **code level** — all 96 mined correct codes
3. **program level** — majority vote over the 19 distinct programs, plus an inconsistency count

It warns automatically when the bag count is too small to separate the two miners.

---

## Files

```
run_correct_only.sh          BOTH directions, correct bags only   <- the main entry point
run_full.sh                  BOTH directions, full track (judge active; days, not hours)
Modelfile.gpt-oss-miner      gpt-oss:20b  as miner   (16K)
Modelfile.gpt-oss-judge      gpt-oss:20b  as judge   (16K, temp 0)
Modelfile.qwen36-miner       qwen3.6:27b  as miner   (16K)
Modelfile.qwen36-judge       qwen3.6:27b  as judge   (16K, temp 0)
scripts/
    _common.sh               direction mapping, guardrails, the pipeline body
    build_models.sh          builds all four, reports the GPU/CPU split
    preflight.py             checks server, models, dataset, and probes both models
    run_<arm>.sh             baseline | rag | ref | rag_ref
    summarize.py             per-direction result tables
    compare_miners.py        A vs B head to head
src/                         mining + scoring (from McMiner, Ollama-native)
utils/ollama_client.py       the only network code; THINK_BY_MODEL lives here
dataset/pseudocode_track/    209 corrupted + 96 correct pseudocode, 22 misconceptions
```

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `no Ollama server reachable` | `ollama serve` |
| `'<model>' is not built/pulled` | `bash scripts/build_models.sh`; pull the base first |
| A model loads at 0 GB / mostly CPU | paging to disk — free RAM, or lower `num_ctx` to 8192 |
| Judge calls taking >10 min | thinking is on — check `THINK_BY_MODEL`, and `ollama ps` for the split |
| Everything scores 0.00% | check parse/judge failure counts; a failed call is scored as a non-match |
| `UnicodeEncodeError` on Windows | handled (`PYTHONIOENCODING=utf-8`); if it reappears, export it yourself |
| Run died partway | re-run; bag mining is skipped when bags exist (`FORCE=1` to rebuild) |

---

## Attribution

Pipeline code, prompt templates and dataset derive from
[McMiner](https://github.com/taisazero/mcminer) (MIT, © 2025 Erfan Al-Hossami). This bundle is a
local-Ollama, cross-judge repackaging for the pseudocode track.
