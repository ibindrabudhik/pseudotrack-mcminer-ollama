# localjudge — dual local-judge evaluation for the McMiner pseudocode track

Self-contained bundle that re-judges already-mined McMiner predictions with
**two local Ollama models** and reports how much they disagree:

| Judge | Model | Weights |
|---|---|---|
| `gpt-oss-judge` | `gpt-oss:20b` | ~13.8 GB |
| `qwen36-judge` | `qwen3.6:27b` | ~17 GB |

Nothing is re-mined. Everything needed is in this folder — dataset, mined
predictions, judging code, prompt template.

---

## Quick start

```bash
ollama pull gpt-oss:20b
ollama pull qwen3.6:27b

bash scripts/build_models.sh          # create both judges, report GPU/CPU split
DRY_RUN=1 bash run_dual_judge.sh      # preflight + live probe, no judging
bash run_dual_judge.sh                # the real run
```

On Windows use Git Bash, and point `PYTHON` at your interpreter:

```bash
PYTHON=/c/Python313/python.exe bash run_dual_judge.sh
```

Dependencies: `pandas openai anthropic requests python-dotenv tqdm google-genai`.
Do **not** `pip install -r` the parent repo's `requirements.txt` — it lists
`pathlib` and `argparse`, stdlib backports that fail to build on Python 3.13.

---

## Hardware notes (RTX 5070 Ti, 16 GB VRAM + 32 GB RAM)

| Model | Fits 16 GB VRAM? | Expectation |
|---|---|---|
| `gpt-oss:20b` @16K | Yes, ~15 GB with KV cache | fully GPU-resident, fast |
| `qwen3.6:27b` @16K | **No** — ~17 GB weights + ~2.7 GB KV | ~2–5 GB spills to RAM |

The qwen spill is fine **because you have 32 GB of RAM**. The thing to avoid is
spilling to *disk*: on a 16 GB-RAM machine the same model paged to the page file
and ran at ~1 token/s with the working set evicted to 0 GB. With 32 GB there is
room for the overflow, so expect a normal partial offload instead.

After the first load, check:

```bash
ollama ps
```

- `100% GPU` — ideal (expected for gpt-oss)
- `~75%/25% GPU/CPU` — expected for qwen3.6:27b, fine
- mostly CPU, or a 0 GB working set — something else holds VRAM, or lower `num_ctx`

`Modelfile.qwen36-judge` documents dropping `num_ctx` to 8192 to buy back ~1.3 GB
of VRAM; the judge prompts peak around **2,020 tokens**, so 8K is ample.

**The judges run one at a time and the previous model is unloaded first.** 14 GB
+ 17 GB cannot co-reside on a 16 GB card, and leaving the first loaded pushes the
second further into RAM than necessary.

---

## Why each judge needs a *different* "stop thinking" switch

Both models reason before answering, and **those tokens come out of the same
budget as the answer**. When the budget runs out the reply is truncated — and
the parser's defaults are not neutral, so a clipped reply silently becomes a
*score* rather than an error. Both switches below are set automatically by
`judge_env()` in `scripts/_common.sh`.

| Model | Switch | Measured consequence of getting it wrong |
|---|---|---|
| `gpt-oss` | top-level `reasoning_effort: low` | At default `medium` it spent **4000/4000** tokens reasoning and returned **empty content**, recorded as "no misconception predicted" |
| `qwen3.6` | `chat_template_kwargs.enable_thinking: false` | With thinking on, one judge call took **>10 min** vs ~160 s with it off |

Override per run if you want to measure the cost of thinking:

```bash
GPTOSS_REASONING_EFFORT=medium bash run_dual_judge.sh
QWEN_EXTRA_BODY='{}'           bash run_dual_judge.sh   # leave qwen thinking ON
```

---

## What preflight checks

`scripts/preflight.py` runs automatically. Step 4 is the one worth waiting for:

1. Ollama reachable, both judge models built
2. Judge inputs present for every arm
3. Exact judge-call counts (same skip rules as the judge itself)
4. **Live probe** — one real judge call per model, verifying it returns a
   parseable `<evaluation>` block within the token budget, and warning if a call
   takes >90 s (a sign thinking is still on)

Skip the probe with `--no-probe` if you only want the counts.

---

## Guardrails

- **`JUDGE_ABORT_AFTER=5`** — stops after 5 consecutive judge failures instead of
  writing a plausible but fabricated low score. This exists because a real run
  once wrote a **0.00% match rate** from 106 consecutive API failures, each
  recorded as `match: False`, with nothing crashing. Set `0` to disable.
- **`JUDGE_MAX_TOKENS=3000`** — raise if the probe reports `parse_ok=False`.
- **`judge_parse_failures`** is reported per arm; anything above 0 is flagged
  `<-- CHECK` in the comparison table and those scores should not be trusted.

---

## Reading the output

`scripts/compare_judges.py` prints accuracy per judge, then per-case agreement
and Cohen's kappa.

**Read the kappa column, not just the accuracy table.** Two judges can produce
identical totals while disagreeing case by case — in an earlier run one arm's
McMiner-S totals matched to the decimal while the judges disagreed on 14
individual predictions (9 one way, 5 the other). The `-only` columns show which
judge is the more lenient one.

```
kappa: <0.40 poor | 0.40-0.60 moderate | 0.60-0.80 substantial | >0.80 strong
```

---

## Layout

```
localjudge/
  run_dual_judge.sh            main entry point
  Modelfile.gpt-oss-judge      16K context, temp 0
  Modelfile.qwen36-judge       16K context, temp 0
  scripts/
    _common.sh                 config + per-judge thinking switches + unload
    build_models.sh            ollama create both, report GPU/CPU split
    preflight.py               checks 1-4 above
    compare_judges.py          accuracy, agreement, Cohen's kappa
  src/
    compute_eval_metrics_multi.py        McMiner-M judge
    evaluate_single_multi_predictions.py McMiner-S judge
    prompt_templates/evaluation-pseudocode/judge_prediction_match.md
  utils/llm_clients.py         + LLM_EXTRA_BODY / LLM_REASONING_EFFORT passthrough
  dataset/pseudocode_track/    codes, correct codes, 22 misconceptions
  predictions/<arm>/           mined predictions (judge INPUTS, not re-run)
  results/<judge>/<arm>/       judge OUTPUTS (gitignored)
```

Arms: `baseline`, `rag`, `ref`, `rag_ref`. Restrict with `ARMS="baseline"` or
`JUDGES="gpt-oss-judge"`.

---

## Scale and caveats

**582 judge calls per judge, 1,164 for both.** Per arm: 159 / 126 / 152 / 145.

Two things this bundle cannot fix:

1. **n = 37 bags per arm.** One bag is 2.70 percentage points and 95% intervals
   are ~29 pp wide. Judge *agreement* is measurable at this size; a between-arm
   ranking is not.
2. **`judge_prediction_match.md` is a reconstruction.** The original was missing
   from the upstream bundle; the criteria come from the paper's evaluation
   prompts, the wording does not. Disclose this if the numbers are published.

**Not yet verified on 16 GB/32 GB hardware.** This bundle was built and tested on
an 8 GB-VRAM / 16 GB-RAM laptop, where `gpt-oss-judge` was probed end-to-end
successfully but `qwen3.6:27b` could not be loaded without disk paging. The
qwen thinking switch is verified to reach the request payload, but its *effect
on that model* has not been observed. **Run `DRY_RUN=1` first** and read the
probe line for `qwen36-judge` before committing to a full run.
