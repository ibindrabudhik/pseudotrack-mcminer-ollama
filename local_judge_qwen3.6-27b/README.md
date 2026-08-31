# local_judge_qwen3.6-27b — mined by qwen3.6:27b, judged by gpt-oss:20b

Self-contained bundle that runs the McMiner **pseudocode track** end-to-end on
two *different* local Ollama models:

| Role | Model | Built as | Context |
|---|---|---|---|
| **Miner** | `qwen3.6:27b` | `qwen3.6-mcminer` | 16K |
| **Judge** | `gpt-oss:20b` | `gpt-oss-judge` | 16K |

Neither model grades its own output. That is the difference from the earlier
runs in the parent repo, where the mining model was also the judge.

**Local Ollama only.** No paid API, no API key, and no OpenAI / Anthropic /
Gemini SDK anywhere in the folder — the only dependencies are `requests` and
`tqdm`.

Two runs live in this folder:

| | Entry point | What it measures |
|---|---|---|
| **Run 1** | `run_all.sh` | the full track — misconception bags *and* correct bags, four prompt arms |
| **Run 2** | `run_correct_only.sh` | correct-only bags on their own: the false-positive rate on code that has nothing wrong with it |

---

## Quick start

```bash
ollama pull qwen3.6:27b
ollama pull gpt-oss:20b

bash scripts/build_models.sh          # create both models, report GPU/CPU split
DRY_RUN=1 bash run_all.sh             # preflight + live probe of both, no mining

SMOKE=1 bash run_all.sh               # 6 codes per arm — always do this first
bash run_all.sh                       # the real run (Run 1)
bash run_correct_only.sh              # Run 2
```

On Windows use Git Bash and point `PYTHON` at your interpreter:

```bash
PYTHON=../.venv/Scripts/python.exe bash run_all.sh
```

Dependencies are **`requests` and `tqdm`**, and that is the whole list:

```bash
pip install -r requirements.txt
```

No AI-vendor SDK is involved — see "How it reaches Ollama". Do **not**
`pip install -r` the parent repo's `requirements.txt`: it lists `pathlib` and
`argparse`, stdlib backports that fail to build on Python 3.13.

Results land in `results/<tag>/`, metrics in
`results/evaluations/<tag>/{single_multi,multi}/evaluation_metrics.json`, and
both entry points print a summary table at the end (`scripts/summarize.py`).

---

## Run 1 — the full track

Four prompt arms, each running **both** McMiner-S (one code at a time) and
McMiner-M (a bag of five at once):

| Arm | Prompt contains | Script |
|---|---|---|
| `baseline` | problem + student pseudocode | `scripts/run_baseline.sh` |
| `rag` | + retrieved similar submissions & correct codes | `scripts/run_rag.sh` |
| `ref` | + APR-repaired reference code | `scripts/run_ref.sh` |
| `rag_ref` | + both | `scripts/run_rag_ref.sh` |

Per arm the preflight counts **~342 mining calls** (33 misconception bags +
4 correct bags + 209 corrupted codes + 96 correct codes), so ~1,370 across four
arms, plus up to 33 bag judgements and 209 single judgements per arm.

Run one arm on its own with `ARMS="baseline" bash run_all.sh`, or
`bash scripts/run_baseline.sh`.

### Reading the output

Read the **misconception** column, not `overall`. Correct-only bags sit at or
near ceiling and contribute the same constant to every arm, so the pooled figure
compresses whatever difference the arms actually have. The summary table splits
them for exactly this reason.

`judge_parse_failures > 0` flags a row with `<-- CHECK`. Those scores are not
trustworthy — a judge reply that did not parse is scored, not errored.

---

## Run 2 — correct-only bags

`run_correct_only.sh` takes every correct program in the dataset, partitions
them into bags the way McMiner's own bag former does, mines each bag, and scores
the result through the same two evaluators Run 1 uses.

Bagging is `create_correct_only_bags(cover_all=True)` in
[src/run_infer_misc_multi.py](src/run_infer_misc_multi.py) — the unaltered
McMiner path, driven by `--correct-bags-only --correct-bags-cover-all`. It
shuffles the pool of correct programs and partitions it into chunks of five, so
every correct program lands in exactly one bag per pass. `CORRECT_PASSES=5`
re-partitions five times with different shufflings if you want more bags.

Per arm: **4 bags + 96 single correct codes = 100 mining calls**, and
**0 judge calls**.

### Why zero judge calls — read this before reporting the numbers

Ground truth for a correct-only bag is `NONE`. There is no misconception
description to compare a prediction against, so neither scorer asks a model:

| Scorer | Method it records | Rule |
|---|---|---|
| `compute_eval_metrics_multi.py` | `correct_bag_rule` | bag matches ⟺ miner predicted nothing |
| `evaluate_single_multi_predictions.py` | `empty_check` | code matches ⟺ prediction list is empty |

The judge is still built, preflighted and live-probed — a broken judge would
otherwise only surface hours into the next full run — but it is not consulted.
The upside is that **Run 2's numbers do not depend on the judge at all**; they
are deterministic given the mined predictions. The corollary is that Run 2 says
nothing about judge quality, and swapping the judge cannot change its results.

### Why 96 correct files are 19 programs

`dataset/pseudocode_track/pseudocode_codes_none/` holds 96 files, but every one
of them carries the literal string `NONE` as its code — the real program is
substituted at prompt-build time from the problem's first correct solution. The
96 files span only **19 distinct `problem_id`s**, so the model sees 19 unique
programs, each between 1 and 12 times.

A file named `problem_130_misc_38.json` in that directory does not mean "a
correct code for misconception 38". It means "misconception 38 was judged
**inapplicable** to problem 130". All such files for problem 130 resolve to the
same program.

Two consequences the summary table handles explicitly:

- **The bag former deduplicates by problem**, so cover-all mode produces 4 bags
  (5+5+5+4) covering all 19 programs — complete coverage, not a subsample.
- **The per-code rate is weighted by an artefact.** Problem 60 contributes 12
  rows, problems 73/121/242 one each. `summarize.py` therefore also prints a
  per-program majority vote, and counts programs that returned *both* answers
  across their repeated rows — same program, same prompt, same temperature. That
  count bounds how much of any arm-to-arm difference is model noise.

### The comparison Run 2 exists to make

`summarize.py --mode correct_only` puts bag-level abstention next to per-code
abstention. On the earlier `gpt-oss` run the gap was stark: individually mined
correct code drew a spurious misconception 15–40% of the time, while bagged
correct code never did. Bag size and prompt wording change together in this
design, so a gap does not by itself say which one causes it.

---

## Fixed here: correct-only bags shared one `prediction_id`

Upstream, correct-only bags were emitted without a `bag_index`, so every one of
them got the id `group_correct_only_None_0`. In a mixed run that only broke
id-keyed joins downstream. In a **correct-bags-only** run every bag in the file
collides on that single id, which makes any per-bag analysis meaningless.

[src/run_infer_misc_multi.py](src/run_infer_misc_multi.py) in this bundle sets
`bag_index` on correct-only bags the way misconception bags already did. Metrics
were never affected either way — both scorers iterate the predictions *list* —
but the ids are now usable.

---

## Reasoning control: one setting, decided per model

Both models reason before answering, and **those tokens come out of the same
budget as the answer**. When the budget runs out the reply is truncated — and
the parsers' defaults are not neutral, so a clipped reply would silently become
a *score*, or a "no misconception predicted", rather than an error.

**`think: false` is not the universal off switch.** Measured here against Ollama
0.32.14:

| Model | `think` | Result |
|---|---|---|
| `gpt-oss` | `false` | `content=''`, `thinking='The user says…'`, `done_reason=length` — reasoned until the budget ran out and returned **nothing** |
| `gpt-oss` | `"low"` | answered in **17 tokens, 5.5 s** |
| `qwen3.6` | `false` | thinking off (with it on, a call was measured at **>10 min** vs ~160 s) |

A reasoning-only model needs a *level*; a model that reasons optionally needs the
*boolean*. That mapping lives in one place — `THINK_BY_MODEL` in
[utils/ollama_client.py](utils/ollama_client.py) — and is applied by model name,
so there is nothing to export per step and nothing that can leak from the miner
into the judge.

Override for a whole run to measure the cost of thinking:

```bash
OLLAMA_THINK=medium  bash run_all.sh    # force a level
OLLAMA_THINK=default bash run_all.sh    # leave each model's own default
```

Related: the client **raises** on an empty or truncated reply instead of
returning `""`. The old client returned `""` for both a genuine "no
misconception" and a blown token budget, and the parser recorded both as a
confident negative.

---

## Hardware: read this before starting a full run

**This machine is the constraint, not the pipeline.** Measured here:

| | |
|---|---|
| GPU | NVIDIA RTX 5070 **Laptop** — **8 GB** VRAM (7.9 GB free) |
| RAM | **15.7 GB** total |

That is *not* the 16 GB-VRAM / 32 GB-RAM desktop the parent `localjudge/`
bundle was written for, and it changes what is realistic:

| Model | Weights | On this machine |
|---|---|---|
| `gpt-oss-judge` @16K | ~13 GB | ~7 GB on GPU, ~6-8 GB in RAM. Proven: the parent repo mined all four arms with this model in **7h 56m**. |
| `qwen3.6-mcminer` @16K | ~17 GB | ~7 GB on GPU, **~10-11 GB in RAM**. Needs most of the machine's RAM free. |

A live probe of both models failed outright while ~14 GB of RAM was held by
Chrome, VS Code and WSL:

```
failed to allocate buffer of size 8227307648
alloc_tensor_range: failed to allocate CUDA_Host buffer
```

With 1.8 GB free, even the 13 GB judge could not load. **Close the memory hogs
before running**, and confirm with `DRY_RUN=1 bash run_all.sh` — the live probe
prints seconds-per-call and extrapolates the full run, so you find out in one
call rather than four hours in.

If qwen3.6:27b ends up mostly on CPU, or `ollama ps` shows a 0 GB working set,
it is paging to disk — measured at ~1 token/s on a machine this size, which
makes the 1,368-call full run impractical. Two ways out:

- **Run 2 only** (`run_correct_only.sh`) — 400 mining calls across four arms
  instead of 1,368, and no judge calls at all.
- **Swap the roles** — `MODEL=gpt-oss-mcminer:latest JUDGE_MODEL=qwen36-judge:latest`.
  Still two different models, still no self-evaluation, and the 13 GB miner is
  the one already proven on this hardware.

### Why the miner is 16K, not 32K

The parent repo built qwen at 32768 on the assumption that a `rag_ref` bag runs
~10K tokens. Measured on this dataset (`scripts/preflight.py` prints this table
every run):

| arm | max prompt chars | ~tokens | + 4000-token response cap |
|---|---|---|---|
| baseline | 6,220 | 1,681 | 5,681 |
| rag | 10,848 | 2,931 | 6,931 |
| ref | 10,585 | 2,860 | 6,860 |
| **rag_ref** | **15,011** | **4,057** | **8,057** |

The true ceiling is ~8.1K tokens. 16384 is 2x headroom; 32768 bought nothing and
cost ~2.7 GB of KV cache — which, on 8 GB of VRAM, is the difference between a
partial offload and a model that will not load. Preflight now compares `num_ctx`
against these measured sizes and fails if it is genuinely too small, rather than
trusting a rule of thumb.

After the first load, check `ollama ps`:

- `100% GPU` — ideal (expected for the judge)
- partial GPU/CPU — expected for the miner, fine as long as RAM is free
- mostly CPU, or a 0 GB working set — something else holds VRAM, or it is paging

**The two models never run at the same time.** `unload_model` in
`scripts/_common.sh` stops the miner before judging starts and vice versa; on
8 GB of VRAM they cannot come close to co-residing. `UNLOAD_BETWEEN=0` disables
it if you move to a machine with room for both.

---

## What preflight checks

`scripts/preflight.py` runs automatically from both entry points
(`SKIP_PREFLIGHT=1` to bypass, `DRY_RUN=1` to stop after it):

1. Ollama reachable; **both** models built; each one's actual `num_ctx` read
   back from `/api/show` and compared against what the prompts need.
2. Dataset files and every prompt template the requested arms will ask for.
3. Exact call counts, by running the **real bag former** rather than estimating.
4. **Live probe** — one real mining call and one real judge call, each through
   the real parser, checking the thinking switch took effect and warning if a
   call is slow enough to make the full run impractical.

Skip the probe with `--no-probe` if you only want the counts.

---

## Guardrails

- **`JUDGE_ABORT_AFTER=5`** — stops after 5 consecutive judge failures instead
  of writing a plausible but fabricated low score. A real run once wrote a
  **0.00% match rate** from 106 consecutive API failures, each recorded as
  `match: False`, with nothing crashing. Set `0` to disable.
- **`JUDGE_MAX_TOKENS=3000`** — raise if the probe reports `parse_ok=False`.
- **Context window is the top silent failure.** Overflow is truncated with no
  error, so you get confident-looking misconceptions mined from a clipped
  prompt. Build the bundled Modelfiles, or start the server with
  `OLLAMA_CONTEXT_LENGTH=16384 ollama serve`.
- **A `SMOKE=1` run always reports broken alignment — that is expected.** The
  two smoke limits truncate different things: `--max-requests` cuts McMiner-M to
  the first few *bags* while `--max-files` cuts McMiner-S to the first few
  *codes*, so step 3 prints `⚠️ WARNING: N missing single predictions`. Judge a
  smoke test on "did every step exit cleanly and is the parse rate high".

---

## Knobs

All are environment variables; every script honours them.

| Var | Default | Meaning |
|---|---|---|
| `MODEL` | `qwen3.6-mcminer:latest` | mining model |
| `JUDGE_MODEL` | `gpt-oss-judge:latest` | judging model |
| `OLLAMA_HOST_URL` | `http://localhost:11434` | server |
| `ARMS` | all four | e.g. `ARMS="baseline rag"` |
| `SMOKE` / `SMOKE_LIMIT` | `0` / `6` | limit codes for a trial run |
| `DRY_RUN` | `0` | `1` = preflight only |
| `SKIP_PREFLIGHT` | `0` | `1` = go straight to mining |
| `RUN_EVAL` | `1` | `0` = mine only, skip judging |
| `FORCE` | `0` | `1` = rebuild McMiner-M bags instead of reusing |
| `CORRECT_PASSES` | `1` | Run 2: passes over the correct-code pool (4 bags each) |
| `COVER_ALL` | `1` | `0` = ratio-based sampling instead of full coverage |
| `UNLOAD_BETWEEN` | `1` | `0` = keep both models resident |
| `JUDGE_MAX_TOKENS` | `3000` | judge response budget |
| `JUDGE_ABORT_AFTER` | `5` | consecutive judge failures before aborting |
| `RAG_TOP_K` | `3` | retrieved examples per code |
| `REF_COLUMN` | `Reference_Code` | or `Best_Reference` / `Repaired_Code` |
| `MODEL_TAG` | `ollama_<model>_<arm>[_correctbags]` | output folder name |
| `PYTHON` | `python` | interpreter with the deps installed |

Re-running an arm **reuses existing McMiner-M bags** (`FORCE=1` to rebuild), so
an interrupted run resumes without re-mining or re-forming bags.

---

## How it reaches Ollama

Directly. [utils/ollama_client.py](utils/ollama_client.py) POSTs to Ollama's
**native** `/api/chat`, with `requests` as its only dependency:

```json
{ "model": "...", "messages": [...], "stream": false,
  "think": false, "options": { "temperature": 0.1, "num_predict": 4000 } }
```

There is no provider switch, no API key, and no OpenAI-compatible `/v1` shim.
Nothing in this folder imports `openai`, `anthropic` or `google-genai` — the
four-provider client the bundle inherited was deleted, along with the ~1,900
lines of Anthropic / Gemini / vLLM code that never ran here.

`python-dotenv` is gone too, and that one was not just tidiness: every entry
script used to call `load_dotenv(override=True)`, so a `.env` file **beat** the
environment the scripts set. A stray `.env` with a real key and base URL could
silently redirect the judge to a paid endpoint. There is now no `.env` path at
all.

Two consequences worth stating plainly:

- **Nothing can leave the machine.** The only host in the code is
  `OLLAMA_HOST_URL`, default `http://localhost:11434`.
- **The reply shape is richer.** `/api/chat` returns `thinking` separately from
  `content`, which is how the client can tell "the model reasoned itself out of
  a budget" from "the model answered nothing" and report the difference.

---

## Layout

```
local_judge_qwen3.6-27b/
  run_all.sh                  Run 1 — full track, four arms
  run_correct_only.sh         Run 2 — correct-only bags
  Modelfile.qwen36-miner      16K context miner
  Modelfile.gpt-oss-judge     16K context judge, temp 0
  scripts/
    _common.sh                config + the two thinking switches + pipeline body
    build_models.sh           ollama create both, report GPU/CPU split
    preflight.py              checks 1-4 above
    summarize.py              result tables for both runs
    run_{baseline,rag,ref,rag_ref}.sh
  src/
    run_infer_misc.py                    McMiner-S mining
    run_infer_misc_multi.py              McMiner-M mining + bag forming
    create_single_multi_predictions.py   aligns S predictions into M bags
    evaluate_single_multi_predictions.py McMiner-S scorer
    compute_eval_metrics_multi.py        McMiner-M scorer / LLM judge
    rag_retrieval.py, ref_retrieval.py   prompt aids
    prompt_templates/mining-pseudocode/       8 mining templates (4 arms x S/M)
    prompt_templates/evaluation-pseudocode/   judge_prediction_match.md
    evaluation-from-mcmining/                 upstream evaluation prompts
  utils/ollama_client.py      the only backend: Ollama /api/chat over requests
  dataset/pseudocode_track/   209 corrupted + 96 correct pseudocode files,
                              22-misconception bank, problem descriptions
  dataset/*.csv               RAG retrieval + APR reference sources
```

---

## Caveats

1. **n = 37 bags per arm in Run 1, 4 in Run 2** (at `CORRECT_PASSES=1`). One bag
   is 2.70 pp in Run 1 and 25 pp in Run 2. Between-arm rankings are not
   supported at this size; raise `CORRECT_PASSES` for Run 2 if you need weight.
2. **19 correct programs is the real ceiling** on false-positive measurement in
   this dataset. Adding rows cannot fix it; adding problems would.
3. **`judge_prediction_match.md` is a reconstruction.** The original was missing
   from the upstream bundle; the criteria come from the paper's evaluation
   prompts, the wording does not. Disclose this if the numbers are published.
   (Run 2 is unaffected — it never uses this template.)
4. **Single-run numbers include model noise.** The earlier run found 5 of 19
   correct programs flipping their answer on identical input at temperature 0.1.
   `summarize.py --mode correct_only` reports that count for this run.

---

## Attribution

The pipeline code (`src/`, `utils/`), prompt templates and pseudocode-track
dataset come from the McMiner project:

> **McMiner** — https://github.com/taisazero/mcminer
> MIT License, Copyright (c) 2025 Erfan Al-Hossami

This folder repackages that work to mine with a local Ollama model and judge
with a *different* local model, and adds the Ollama wiring, the shared arm
runner, preflight, the correct-only run, and the correct-only `bag_index` fix.
Redistributed under the same MIT license — see `LICENSE`.

The dataset files are the synthetic misconception-injected pseudocode used by
the pseudocode track; they contain no student identifiers.
