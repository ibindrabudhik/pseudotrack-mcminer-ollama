# Dual Local-Judge Re-evaluation — Run Report

**Judging dates:** 2026-08-20 (`gpt-oss-judge`) · 2026-08-21 → 2026-08-25 (`qwen36-judge`)
**Predictions judged:** the 1,220 mined on 2026-08-17 — **byte-identical**, not re-mined
**Judges:** `gpt-oss:20b` (self-evaluation) **and** `qwen3.6:27b` (independent, local)
**Hardware:** NVIDIA RTX 5070 Ti (16 GB VRAM), 32 GB system RAM, Windows 11
**Bundle:** [`localjudge/`](localjudge/) — self-contained, offline

---

## 1. Summary

[`REPORT.md`](REPORT.md) judged the pseudocode track twice: once by the mining
model itself, once by **GPT-5 via OpenRouter** at a cost of $4.66. This run asks
whether the independent judge can be **local and free** — the hardware moved
from 8 GB VRAM / 15.7 GB RAM to 16 GB / 32 GB, which is what first made
`qwen3.6:27b` runnable at all.

**The answer is no, not as configured.** The headline is a defect, not a result.

Three findings, in order of confidence:

1. **The `qwen36-judge` results are invalid and must not be quoted.** 31.2% of
   its judge calls (145 of 465) **timed out**, and the pipeline scores a timeout
   as a non-match (§5.2). Its apparently "stricter" scores are a plumbing
   artifact, not a judgement.
2. **On the calls it did complete, `qwen3.6:27b` is *more lenient* than
   `gpt-oss:20b`, not stricter** — in **all seven** comparable cells, and the two
   judges agree on 94.9–100% of those cases (§6.2). This is the opposite of what
   the raw table shows.
3. **Changing only the judge token budget (1500 → 3000) moved individual arm
   scores by up to 8.11 pp** with the judge model, predictions and prompt all
   held constant (§7). That is **larger than any between-arm difference the
   study is trying to detect**, and it independently confirms the n=37 fragility
   warning in `REPORT.md` §6.2.

> **Correction to `REPORT.md`'s framing.** That report's "quote the independent
> judge" rule is sound, but it applies to the **GPT-5** numbers. Nothing in this
> local run corroborates the +7.4 pp self-judging bias measured there — this run
> is too broken to speak to it either way.

---

## 2. What this run is

Mining is the expensive step and was already done. This re-runs **only the two
judging steps** against the existing predictions, with two local models.

### 2.1 The inputs are provably identical

Every prediction file in `localjudge/predictions/` was compared byte-for-byte
against the original mining output in `results/ollama_gpt-oss-mcminer-latest_<arm>/`:

| File | baseline | rag | ref | rag_ref |
|---|---|---|---|---|
| `multi/multi_predictions.json` | identical | identical | identical | identical |
| `single/predictions.json` | identical | identical | identical | identical |
| `single_multi/grouped_predictions.json` | identical | identical | identical | identical |

The judge prompt template and both evaluation scripts are also byte-identical to
the parent repo's. **Any difference in the numbers therefore comes from the
judge, its settings, or nondeterminism — never from the inputs.** That is what
makes §7 a clean natural experiment.

The mining provenance is recorded explicitly in
[`localjudge/predictions/PROVENANCE.json`](localjudge/predictions/PROVENANCE.json),
because the mining model used to be encoded only in a directory name that this
bundle does not preserve:

| | |
|---|---|
| Mining model | `gpt-oss-mcminer:latest` (base `gpt-oss:20b`), via Ollama |
| Settings | `num_ctx 16384`, `reasoning_effort low` |
| Run | 2026-08-17, 7h 56m 35s |
| Predictions | 1,220 (305 × 4 arms), **0 parse failures** |

### 2.2 The two judges

| Judge tag | Base model | Weights | "Stop thinking" switch |
|---|---|---|---|
| `gpt-oss-judge` | `gpt-oss:20b` | ~13.8 GB | top-level `reasoning_effort: low` |
| `qwen36-judge` | `qwen3.6:27b` | ~17 GB | `chat_template_kwargs.enable_thinking: false` |

Both at `temperature 0`, `num_ctx 16384`, served locally by Ollama.

Two naming traps worth stating plainly, because both invite a wrong conclusion:

- **The `claude_evaluation_results.json` filenames and the
  `evaluation_method: "claude_existing"` field are legacy strings from upstream
  McMiner**, hardcoded at [`compute_eval_metrics_multi.py:433`](localjudge/src/compute_eval_metrics_multi.py#L433)
  and [`evaluate_single_multi_predictions.py:188`](localjudge/src/evaluate_single_multi_predictions.py#L188).
  **No Anthropic model was used anywhere in this run.**
- **`JUDGE_PROVIDER=openrouter` does not mean OpenRouter.** `_common.sh` points
  `OPENROUTER_BASE_URL` at `http://localhost:11434/v1` with a dummy key, because
  Ollama speaks the OpenAI-compatible protocol and the SDK will not construct a
  client without one. **Nothing left the machine.**

### 2.3 Why each judge needs a different switch

Both models reason before answering, and those tokens come out of the **same
budget as the answer**. The parser's defaults are not neutral, so a clipped
reply becomes a *score* rather than an error.

| Model | Measured consequence of getting it wrong |
|---|---|
| `gpt-oss` | At default `medium` effort: **4000/4000** tokens spent reasoning, **empty content** returned |
| `qwen3.6` | With thinking on: one judge call **>10 min** vs ~160 s with it off |

§5.2 shows the qwen switch was necessary but **not sufficient**.

---

## 3. Hardware

This is the run the original report could not do. `REPORT.md` §3 recorded
`qwen3.6:27b` as unusable on the old machine — 17 GB of weights against 15.7 GB
of RAM meant Windows paged the model to *disk* at 0.6–1.3 tok/s.

| | old (RTX 5070 Laptop) | **new (RTX 5070 Ti)** |
|---|---|---|
| VRAM | 7.7 GB | **16 GB** |
| System RAM | 15.7 GB | **32 GB** |
| `gpt-oss:20b` @16K | 59/41 CPU/GPU split | ~100% GPU |
| `qwen3.6:27b` @16K | **paged to disk** | ~2–5 GB spill to RAM |

The 32 GB is the part that matters: the spill goes to RAM instead of the page
file. That made qwen3.6 *runnable*. §5 shows it did not make it *fast enough*.

Judges run strictly one at a time with the previous model unloaded first — 14 GB
and 17 GB cannot co-reside on a 16 GB card.

---

## 4. Configuration

```bash
bash localjudge/run_dual_judge.sh
```

| Setting | This run | Original self-judge | GPT-5 run |
|---|---|---|---|
| `JUDGE_MAX_TOKENS` | **3000** | **1500** (code default) | **4000** |
| `JUDGE_TEMPERATURE` | 0.0 | 0.0 | 0.0 |
| `JUDGE_ABORT_AFTER` | 5 | — | 5 |
| Judge `num_ctx` | 16384 | 16384 | n/a |

**The three runs used three different token budgets.** The parent repo's
`scripts/_common.sh` never sets `JUDGE_MAX_TOKENS`, so the original self-judge
run silently took the code default of 1500 from
[`compute_eval_metrics_multi.py:225`](localjudge/src/compute_eval_metrics_multi.py#L225),
while `judge_with_gpt5.sh` sets 4000. This is a real confound in the original
report's bias measurement, and §7 quantifies what it is worth.

---

## 5. Execution

### 5.1 Wall time

| Judge | Span | Note |
|---|---|---|
| `gpt-oss-judge` | 2026-08-20 14:56:20 → 15:10:07 = **13m 47s** | all 4 arms, 8 cells |
| `qwen36-judge` | 2026-08-21 23:33 → 2026-08-25 18:35 | **~91 h calendar**, resumed across days |

The recorded timestamp is when each step *finished*, so the gpt-oss span omits
the duration of its own first step (baseline McMiner-S). The other three S steps
took 186 s, 232 s and 207 s, so **the true gpt-oss total is roughly 17 minutes**,
not 13m 47s. The qwen span is calendar time across a manually resumed run, not
continuous compute.

Per-step times for the qwen McMiner-M stage, from file write timestamps:

| Arm | McMiner-M duration | Judge calls |
|---|---|---|
| baseline | **7h 32m** | 28 |
| rag | **3h 24m** | 20 |
| ref | **7h 21m** | 26 |

**Seven and a half hours for 28 judge calls** is the tell. The same 28 calls took
`gpt-oss-judge` 54 seconds.

### 5.2 Pipeline health — the finding

| Judge | Calls attempted | **Timeouts** | Mistagged | Clean |
|---|---|---|---|---|
| `gpt-oss-judge` | 582 | **0** (0.0%) | 5 | **577 (99.1%)** |
| `qwen36-judge` | 465 | **145 (31.2%)** | 19 | **301 (64.7%)** |

qwen's 465 counts only calls that reached a results file. The aborted `rag_ref`
McMiner-S step (§5.4) wrote nothing, so its calls — including the five
consecutive timeouts that triggered the abort — are **not** in this total. The
true attempt count and timeout rate are both higher.

Every one of the 145 failures carries the rationale `judge error: Request timed
out.` Per cell:

| Arm | McMiner-S calls | S timeouts | McMiner-M calls | M timeouts |
|---|---|---|---|---|
| baseline | 131 | **44** | 28 | **11** |
| rag | 106 | **29** | 20 | 2 |
| ref | 126 | **40** | 26 | **11** |
| rag_ref | *aborted* | — | 28 | **8** |

**A timeout is scored as a non-match.** The exception handler at
[`compute_eval_metrics_multi.py:393`](localjudge/src/compute_eval_metrics_multi.py#L393)
writes `{"match": False, "match_with_novel": False}` and continues. There is no
"unknown" state — a call that never returned is indistinguishable, in the
metrics, from a judge that looked and said no.

### 5.3 Root cause

The judge client is constructed at
[`llm_clients.py:55`](localjudge/utils/llm_clients.py#L55) as
`OpenAI(api_key=..., base_url=...)` — **with no `timeout` argument and no
`max_retries` argument.** The OpenAI SDK defaults are therefore 600 s and 2
retries. So a qwen call that exceeds ten minutes is retried twice and consumes
**~30 minutes** before being recorded as a non-match.

That reconstructs the wall time almost exactly. For baseline McMiner-M:
11 timeouts × 30 min = 330 min, plus 17 completed calls at ~7 min = 119 min,
totalling **~7h 29m** against the 7h 32m measured. (An inference from the
recorded counts and timestamps, not a direct measurement — the per-call timings
were not logged.)

There is no environment variable for the timeout. It cannot be raised without a
code change.

### 5.4 The abort guard fired — correctly

`qwen36-judge/rag_ref/single_multi` has **no `judge_details_single.json` and no
`judge_single/` directory**, and 165 of its 184 single evaluations carry
`"evaluation_method": "no_evaluation"`. Its reported 8.11% is an empty cell, not
a score.

This is `JUDGE_ABORT_AFTER=5` working as designed: the timeouts are scattered
rather than clustered (see the run trace below, `X` = timeout), and this arm was
the one that happened to hit five in a row.

```
baseline S  .X..XXX...XX......XXX...X...XX..X..X.X.X.XXX.XXX.......X....
            X.X.X..X........X...X..XXXX..XXX..........X..X........X.....
            X..XX.X...X
ref      S  X...XX............XX...X..X.XX.....XX.X.XX..X.XXXX.....XX...
            X.....X..X.....X..XXX..XXX.XXX..........X..X..X....X.....X..
            ..X...
```

The guard prevented one fabricated cell. It did **not** prevent the other seven,
because scattered failures never trip a *consecutive* counter. **A rate-based
guard is what this run needed.**

---

## 6. Results

### 6.1 As scored — and why the qwen column is unusable

| Arm | `gpt-oss-judge` S std | S novel | M std | M novel |
|---|---|---|---|---|
| baseline | 59.46% | 67.57% | 59.46% | 75.68% |
| rag | 56.76% | 70.27% | 62.16% | 64.86% |
| ref | 59.46% | 75.68% | 64.86% | 72.97% |
| **rag_ref** | **64.86%** | **78.38%** | **78.38%** | **86.49%** |

| Arm | `qwen36-judge` S std | S novel | M std | M novel |
|---|---|---|---|---|
| baseline | 48.65% | 54.05% | 54.05% | 54.05% |
| rag | 54.05% | 54.05% | 59.46% | 59.46% |
| ref | 48.65% | 51.35% | 48.65% | 48.65% |
| rag_ref | ~~8.11%~~ *aborted* | — | 64.86% | 64.86% |

Two signatures mark the qwen column as broken before any accuracy argument:

- **Its novelty-aware score equals its standard score in six of seven cells.**
  The novelty metric can only ever go up. A judge that awards *zero* novelty
  credit across 33 misconception bags is not judging novelty — and indeed, a
  timeout sets both fields to `False` simultaneously.
- **`rag_ref` S at 8.11%** — an empty cell (§5.4).

### 6.2 What the completed calls actually show

Restricting to cases each judge genuinely returned a parseable verdict for:

| Arm | Mode | qwen **as scored** | qwen **completed calls only** |
|---|---|---|---|
| baseline | S | 61.8% | **92.7%** |
| baseline | M | 57.1% | **100.0%** |
| rag | S | 71.7% | **98.6%** |
| rag | M | 90.0% | **100.0%** |
| ref | S | 65.1% | **95.0%** |
| ref | M | 57.7% | **100.0%** |
| rag_ref | M | 71.4% | **100.0%** |

And head-to-head, on only the cases **both** judges completed cleanly:

| Arm | Mode | n | Agreement | `gpt-oss` says match | `qwen3.6` says match |
|---|---|---|---|---|---|
| baseline | S | 82 | 97.6% | 74 | **76** |
| baseline | M | 16 | 87.5% | 14 | **16** |
| rag | S | 72 | 97.2% | 69 | **71** |
| rag | M | 17 | 94.1% | 16 | **17** |
| ref | S | 78 | 94.9% | 70 | **74** |
| ref | M | 15 | 100.0% | 15 | 15 |
| rag_ref | M | 18 | 100.0% | 18 | 18 |

**`qwen3.6` awards at least as many matches as `gpt-oss` in all seven cells, and
strictly more in five.** The two judges agree on 94.9–100% of shared cases.

The conclusion inverts: **`qwen3.6:27b` is the more lenient judge**, and it
largely agrees with `gpt-oss:20b`. The entire apparent gap in §6.1 is the
timeout artifact.

⚠️ **Caveat.** Timeouts are not guaranteed random. If slower calls are also
harder ones, the completed subset is optimistic, and the true qwen leniency is
lower than 92–100%. The run trace in §5.4 shows failures scattered rather than
clustered, which argues against a simple warm-up or thermal explanation, but a
prompt-length correlation was not ruled out. **These numbers establish the
*direction* of the error, not a corrected score.**

### 6.3 Cohen's κ on the as-scored verdicts

For completeness, `scripts/compare_judges.py` output — this **includes** the
timeouts, so it measures the broken run, not the judges:

| Arm | M agreement | κ | S agreement | κ |
|---|---|---|---|---|
| baseline | 83.8% | 0.67 | 97.7% | 0.84 |
| rag | 91.9% | 0.83 | 96.1% | **0.39** |
| ref | 83.8% | 0.68 | 95.3% | 0.64 |
| rag_ref | 86.5% | 0.67 | *aborted* | — |

The `rag` McMiner-S row is the cautionary one: **96.1% agreement at κ=0.39**.
High raw agreement with near-chance κ is exactly what a sparse, artifact-driven
disagreement pattern looks like. It is the same lesson the script's own docstring
records — read κ, not the totals.

---

## 7. The token budget moves scores more than the arms do

`gpt-oss-judge` re-judged predictions it had already judged in the original run.
Same model, same prompt template, same predictions, same temperature. **The only
change was `JUDGE_MAX_TOKENS`: 1500 → 3000.**

Delta = 3000-token run minus 1500-token run, in percentage points:

| Arm | S std | S novel | M std | M novel |
|---|---|---|---|---|
| baseline | 0.0 | **−5.41** | **−8.11** | −2.70 |
| rag | 0.0 | 0.0 | +2.70 | 0.0 |
| ref | +2.70 | **+5.41** | 0.0 | 0.0 |
| rag_ref | 0.0 | −2.70 | **+8.11** | +5.41 |

**Mean shift is under 1 pp — but individual arms move by up to 8.11 pp, which is
three bags.** The direction is not consistent, so this is variance, not bias.

Two consequences:

1. **Larger than the effect under study.** `REPORT.md` §6.2 established that one
   bag = 2.70 pp and that every arm's Wilson interval overlaps every other. This
   adds that a **judge-side knob unrelated to prompt design** can move a single
   arm by three bags. `rag_ref`'s McMiner-M lead grew from 70.27% to 78.38%
   purely from a token budget change.
2. **It confounds the original bias measurement.** `REPORT.md` §7.2 compared
   GPT-5 at 4000 tokens against gpt-oss at 1500 and attributed the whole +7.4 pp
   gap to self-evaluation. Part of that gap is budget. The finding's *direction*
   still stands — it was negative in all four arms, which budget variance is not —
   but **the magnitude is not clean** and should be restated with this caveat.

---

## 8. Three judges, same predictions

| Arm | gpt-oss @1500 | **GPT-5 @4000** | gpt-oss @3000 | qwen3.6 @3000 |
|---|---|---|---|---|
| | *self, cloud-free* | *independent, $4.66* | *self, this run* | *invalid* |
| **McMiner-S standard** |
| baseline | 59.46% | 48.65% | 59.46% | ~~48.65%~~ |
| rag | 56.76% | 54.05% | 56.76% | ~~54.05%~~ |
| ref | 56.76% | 48.65% | 59.46% | ~~48.65%~~ |
| rag_ref | **64.86%** | **56.76%** | **64.86%** | ~~aborted~~ |
| **McMiner-M standard** |
| baseline | 67.57% | 64.86% | 59.46% | ~~54.05%~~ |
| rag | 59.46% | 59.46% | 62.16% | ~~59.46%~~ |
| ref | 64.86% | 56.76% | 64.86% | ~~48.65%~~ |
| rag_ref | **70.27%** | **72.97%** | **78.38%** | ~~64.86%~~ |

`rag_ref` leads under every judge and every configuration — including the broken
one. That consistency is worth noting, but it does **not** rescue the ranking:
`REPORT.md` §6.2's overlapping confidence intervals still apply, and §7 above
shows a single arm moving three bags on an unrelated setting.

**The quotable column remains GPT-5.** This run does not replace it.

---

## 9. The mistagged-closing-tag defect, revisited

`REPORT.md` §8.6 recorded 5 occurrences of `gpt-oss` writing
`</match_withnovel>` while opening the tag correctly. This run reproduces it and
shows it is **not model-specific**:

| Judge | Mistagged replies |
|---|---|
| `gpt-oss-judge` | 5 of 582 (0.9%) |
| `qwen36-judge` | **19 of 465 (4.1%)** |

qwen's variant is a *mismatched* close rather than a typo:

```xml
<match>N</match>
<match_with_novel>Y</match>          <-- closes the wrong tag
<confidence>high</confidence>
<rationale>...the code genuinely uses "!=" in Code 5, which correctly
exhibits the predicted syntax misconception... satisfying the novelty
condition for match_with_novel.</rationale>
```

The regex at [`compute_eval_metrics_multi.py:249`](localjudge/src/compute_eval_metrics_multi.py#L249)
requires the closing tag, so `match_with_novel` falls back to `match` — here
discarding an explicit `Y`.

Applying a tolerant regex (capture from `<tag>` to the next `<`) across all 24
mistagged replies in this run recovers **exactly 1** silently discarded novelty
credit. **No reported number in this report changes.** As in the original run,
the defect stayed nearly harmless by luck: the fallback is `match`, which is
already correct whenever `match=Y`. It is still worth fixing, because 4.1% on a
new judge model is five times the rate on the old one — the next model may not
be so lucky.

---

## 10. Threats to validity

1. **`qwen36-judge` results are invalid.** 31.2% timeout rate, scored as
   non-matches (§5.2). Do not quote §6.1's qwen column anywhere.
2. **The §6.2 correction shows direction, not magnitude.** The completed-call
   subset may be biased toward easier cases (§6.2 caveat).
3. **Judge token budget is a live variable**, worth up to 8.11 pp per arm (§7),
   and it differs across all three judging runs.
4. **No timeout configuration exists.** The 600 s / 2-retry SDK defaults are
   unreachable without editing `llm_clients.py` (§5.3).
5. **`JUDGE_ABORT_AFTER` is consecutive-only** and cannot catch a scattered
   failure mode (§5.4).
6. **Everything inherited from the mining run still applies** — n=37 bags,
   `reasoning_effort=low` unquantified, only 19 unique correct programs, and the
   judge prompt template being a reconstruction. See `REPORT.md` §8.
7. **`enable_thinking: false` may not have been honoured.** It was passed via
   `LLM_EXTRA_BODY`, and no reasoning trace appears in the visible replies — but
   a quarter of calls still exceeded 600 s against a documented ~160 s
   expectation. Ollama may be silently dropping `chat_template_kwargs`, or
   stripping the trace into a separate field. **Unresolved**, and it is the first
   thing to check before any re-run.

---

## 11. Recommended next steps

In priority order. The first three are what make a local independent judge
viable at all.

1. **Make the timeout configurable and set it high.** Pass
   `timeout=float(os.getenv("LLM_TIMEOUT", "1800"))` and `max_retries=0` when
   constructing the client in `llm_clients.py`. At 600 s × 3 attempts the current
   defaults burn 30 minutes to produce a wrong score.
2. **Never score an unreturned call.** A `judge_error` should mark the case
   *excluded*, not `match=False`. Report `n_judged` alongside every accuracy so a
   degraded run is visible in the metrics file itself.
3. **Add a rate-based abort** — e.g. halt if the failure rate exceeds 10% over a
   rolling window — alongside the existing consecutive counter. This run would
   have stopped in the first arm instead of burning four days.
4. **Verify `enable_thinking: false` actually reaches the model** (§10.7), then
   re-run qwen. Check `ollama ps` for the GPU/CPU split at the same time.
5. **Pin `JUDGE_MAX_TOKENS` explicitly in every judging script**, including the
   parent repo's `_common.sh`, so no run silently takes the 1500 default.
6. **Apply the tolerant-regex parser fix** (§9) — 4.1% mistagging on a new judge
   model is the warning the original 0.9% did not give.
7. **Restate `REPORT.md` §7.2's +7.4 pp** with the budget confound noted (§7).
8. **Then, and only then, re-attempt the local independent judge.** The premise
   is sound: `gpt-oss-judge` completed 582 calls in ~17 minutes with a 99.1%
   clean rate, and the judges agree on 94.9–100% of cases they both finished. The
   hardware is not the obstacle — the client configuration is.

---

## 12. Output locations

```
localjudge/predictions/PROVENANCE.json          what mined these, and with what settings
localjudge/predictions/<arm>/                   judge INPUTS (byte-identical to the 08-17 run)
    single/predictions.json                     305 per-code predictions
    single_multi/grouped_predictions.json       the same, aligned into 37 bags
    multi/multi_predictions.json                37 bag-level predictions

localjudge/results/<judge>/<arm>/
    single_multi/evaluation_metrics.json        McMiner-S metrics
    single_multi/bag_evaluation_results.json    per-bag detail
    single_multi/judge_details_single.json      per-prediction verdicts  <- absent = the run failed
    multi/claude_evaluation_results.json        per-bag verdicts (legacy filename; NOT Claude)
    multi/evaluation_metrics.json               McMiner-M metrics
```

`<judge>` is `gpt-oss-judge` or `qwen36-judge`.

**Reading a result directory:** check `summary.judge_parse_failures` and the
presence of `judge_details_single.json` *before* reading any accuracy. A missing
details file means the abort guard fired and the metrics file beside it is empty
scaffolding.

---

## Attribution

Pipeline code, prompt templates and dataset derive from
[McMiner](https://github.com/taisazero/mcminer) (MIT, © 2025 Erfan Al-Hossami).
This report covers a local dual-judge re-evaluation of that work; see
[`REPORT.md`](REPORT.md) for the mining run it scores.
