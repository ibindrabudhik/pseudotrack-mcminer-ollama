# McMiner pseudotrack on a local Ollama model

Self-contained bundle: runs the McMiner **pseudocode track** end-to-end against a
model served by [Ollama](https://ollama.com) — default `qwen3.6:27b` — across
four prompt arms, each with **both McMiner-S and McMiner-M**.

| Arm | Prompt contains | Script |
|---|---|---|
| `baseline` | problem + student pseudocode | `scripts/run_baseline.sh` |
| `rag` | + retrieved similar submissions & correct codes | `scripts/run_rag.sh` |
| `ref` | + APR-repaired reference code | `scripts/run_ref.sh` |
| `rag_ref` | + both | `scripts/run_rag_ref.sh` |

Nothing here calls a paid API. No API key is required.

---

## Quick start

```bash
# 1. deps (needs the anthropic / google-genai packages too — llm_clients.py
#    imports them at module load even when you only use the OpenAI-compatible path)
pip install -r requirements.txt

# 2. pull the model
ollama pull qwen3.6:27b

# 3. give it a context window big enough for these prompts (see below)
ollama create qwen3.6-mcminer -f Modelfile

# 4. smoke test — 6 codes per arm, ~minutes
SMOKE=1 MODEL=qwen3.6-mcminer bash run_all.sh

# 5. full run — 209 codes per arm
MODEL=qwen3.6-mcminer bash run_all.sh
```

Run a single arm instead: `MODEL=qwen3.6-mcminer bash scripts/run_rag.sh`

Results land in `results/<tag>/` and metrics in
`results/evaluations/<tag>/{single_multi,multi}/evaluation_metrics.json`.

---

## Read this before trusting any number

**The judge is the same local model that did the mining.** `qwen3.6:27b` grades
its own output. That is self-evaluation bias, and these scores are **not
comparable** to the `o3-mini` / `gemini-2.5-flash` / `qwen3-14b` results in the
main repo, which were judged by GPT-5 (the paper's main-table judge). Use this
bundle to verify the pipeline runs and to compare the four arms *against each
other under one judge*; do not put these numbers next to the paper's table.
See "Swapping the judge" to fix that.

**Context window is the top failure mode.** Ollama's default context is far
smaller than these prompts need. A McMiner-M bag is 5 pseudocode solutions
(~1.3KB each) plus problem text; the `rag_ref` arm adds 3 retrieved examples and
an APR reference — roughly 10K tokens before the model responds. Overflow is
truncated **silently**, so you get confident-looking misconceptions mined from a
clipped prompt. Either build the bundled `Modelfile` (32K, persistent, per-model)
or start the server with `OLLAMA_CONTEXT_LENGTH=32768 ollama serve`.

**A `SMOKE=1` run always reports broken alignment — that is expected.** The two
smoke limits truncate different things: `--max-requests` cuts McMiner-M to the
first few *bags* (which are the correct-only bags), while `--max-files` cuts
McMiner-S to the first few *codes*. So step 3 prints
`⚠️ WARNING: N missing single predictions` and `coverage 0.0%`, and
`Misconception groups: 0`. That is an artifact of the limits, not a failure —
verified on a real run. Judge the smoke test on "did every step exit cleanly and
is the parse success rate high", and only expect real coverage on a full run.

**`qwen3.6:27b` is a 27B model.** Expect roughly 400 mining calls per arm
(209 single + 96 NONE + ~60 bags), ×4 arms, plus judge calls. On consumer
hardware plan for hours per arm. Always `SMOKE=1` first.

---

## How it reaches Ollama

There is no `ollama` provider in this codebase and none was added. The pipeline
uses the existing **`openrouter`** provider, which is just `OpenAIClient` with a
settable `base_url` (`utils/llm_clients.py`), pointed at Ollama's
OpenAI-compatible `/v1` endpoint:

```
--llm openrouter --openrouter-model qwen3.6:27b \
--openrouter-base-url http://localhost:11434/v1
```

`OPENROUTER_API_KEY` is set to the dummy value `ollama` because
`create_llm_client()` raises when it is unset; the Ollama server ignores it.

Two deliberate choices worth knowing:

- **Not `--llm vllm`.** The scripts build the model flag as `--${PROVIDER}-model`,
  and passing `--vllm-model` switches `VLLMClient` into *offline* mode, which
  imports the real `vllm` package and loads weights into local GPU memory. vLLM
  *server* mode would work over HTTP, but it cannot select a model by name — it
  calls `/v1/models` and takes `models[0]`, so it would grab whichever model
  Ollama happens to list first.
- **No `--reasoning`.** That flag sends OpenRouter's unified
  `reasoning: {effort}` field via `extra_body`, which is a gateway concept, not
  an Ollama one. Set `REASONING_FLAG="--reasoning"` only if you have verified
  your Ollama build accepts it.

---

## Knobs

All are environment variables; every script honours them.

| Var | Default | Meaning |
|---|---|---|
| `MODEL` | `qwen3.6:27b` | Ollama model tag |
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | server endpoint |
| `SMOKE` / `SMOKE_LIMIT` | `0` / `6` | limit codes for a trial run |
| `RUN_EVAL` | `1` | `0` = mine only, skip judging |
| `FORCE` | `0` | `1` = rebuild McMiner-M bags instead of reusing |
| `ARMS` | all four | e.g. `ARMS="baseline rag"` |
| `MODEL_TAG` | `ollama_<model>_<arm>` | output folder name |
| `RAG_TOP_K` | `3` | retrieved examples per code |
| `REF_COLUMN` | `Reference_Code` | or `Best_Reference` / `Repaired_Code` |
| `PYTHON` | `python` | interpreter (must be the one with the deps — on conda that is usually `python`, not the system `python3`) |

Re-running an arm **reuses existing McMiner-M bags** (`FORCE=1` to rebuild), so
an interrupted run resumes without re-mining or re-forming bags.

---

## Swapping the judge

To get numbers comparable to the paper's main table, judge with GPT-5 instead.
The judge is configured separately from the mining model, so mining stays local:

```bash
export OPENROUTER_API_KEY=sk-or-...        # a real key
JUDGE_PROVIDER=openrouter \
JUDGE_MODEL=openai/gpt-5 \
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1 \
MODEL=qwen3.6-mcminer bash run_all.sh
```

`OPENROUTER_BASE_URL` must be overridden too — the bundle otherwise points it at
Ollama for the judge. This costs money per run, ×4 arms.

Already mined and only want to re-judge? Set `RUN_EVAL=0` for the mining pass,
then run the two eval scripts directly against the existing
`results/<tag>/` files (see steps 4 and 5 in `scripts/_common.sh`).

---

## Differences from the main repo's scripts

- **McMiner-M is evaluated in all four arms.** In the main repo,
  `run_pseudocode_mcminer.sh` (baseline) and `run_pseudocode_mcminer_ref.sh` stop
  after the McMiner-S evaluation and never run `compute_eval_metrics_multi.py`,
  so their McMiner-M predictions were mined but never scored. Here every arm runs
  both.
- **One RAG CSV across arms.** The main repo's `_rag.sh` defaults to
  `retrival_openai_embedding_large.csv` while `_rag_ref.sh` defaults to
  `retrieval_gemini_11jul.csv`, which makes those two arms non-comparable. This
  bundle uses `retrival_openai_embedding_large.csv` everywhere.
- **One shared pipeline.** The four arms differ only in template names and
  prompt-aid flags, so the ~300 duplicated lines per script collapse into
  `scripts/_common.sh` plus a ~12-line arm script.
- **Preflight checks.** Missing dataset files, an unreachable Ollama server, or
  an un-pulled model fail immediately with a specific message instead of
  hundreds of API errors.

## Layout

```
dataset/pseudocode_track/   209 corrupted + 96 correct pseudocode files,
                            22-misconception bank, problem descriptions
dataset/*.csv               RAG retrieval + APR reference sources
src/                        mining, alignment and evaluation scripts
src/prompt_templates/mining-pseudocode/   the 8 templates (4 arms × S/M)
utils/llm_clients.py        provider clients
scripts/                    the four arm scripts + shared pipeline
```

---

## Attribution

The pipeline code (`src/`, `utils/`), prompt templates and pseudocode-track
dataset in this bundle come from the McMiner project:

> **McMiner** — https://github.com/taisazero/mcminer
> MIT License, Copyright (c) 2025 Erfan Al-Hossami

This repository repackages that work to run against a local Ollama model and
adds the Ollama wiring, the shared arm runner in `scripts/`, preflight checks,
and the McMiner-M evaluation in all four arms. Redistributed under the same MIT
license — see `LICENSE`.

The dataset files here are the synthetic misconception-injected pseudocode used
by the pseudocode track (generated by injecting misconceptions into reference
solutions); they contain no student identifiers.
