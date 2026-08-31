---
marp: true
theme: default
paginate: true
size: 16:9
header: 'McMiner Pseudocode Track — Dual Local Judge'
footer: '2026-08-20 → 08-25 · gpt-oss:20b + qwen3.6:27b · RTX 5070 Ti'
---

<!-- _paginate: false -->
<!-- _header: '' -->
<!-- _footer: '' -->

# Can the Independent Judge Be Local and Free?

## Re-judging the McMiner pseudocode track with two local models

**Same predictions · two local judges · zero cloud calls**

`gpt-oss:20b` (self) vs `qwen3.6:27b` (independent)

2026-08-20 → 2026-08-25

---

## The question

`REPORT.md` judged the track twice:

- **`gpt-oss:20b`** judging its own mining output → self-evaluation
- **GPT-5 via OpenRouter** → independent, **$4.66**, 2 hours

Self-judging was measured to inflate McMiner-S accuracy by **+7.4 pp**.
So the independent judge is the one worth quoting.

**But it costs money and leaves the machine.**

New hardware — **16 GB VRAM / 32 GB RAM**, up from 8 GB / 15.7 GB — finally makes
`qwen3.6:27b` runnable locally.

> **Can a second *local* model do the independent judge's job for free?**

---

## Answer, up front

# ❌ Not as configured

**31.2% of `qwen3.6:27b`'s judge calls timed out** — and the pipeline scores a
timeout as a **non-match**.

Its "stricter" numbers are a **plumbing artifact**, not a judgement.

**Three findings:**

1. The `qwen36-judge` results are **invalid** — do not quote them
2. On calls it *completed*, qwen3.6 is **more lenient** than gpt-oss, not stricter
3. Changing only the **judge token budget** moved arm scores by up to **8.11 pp**

---

## Setup — the inputs are provably identical

Nothing was re-mined. Only the **two judging steps** were re-run.

Every prediction file compared **byte-for-byte** against the 2026-08-17 run:

| File | baseline | rag | ref | rag_ref |
|---|---|---|---|---|
| `multi_predictions.json` | ✅ | ✅ | ✅ | ✅ |
| `single/predictions.json` | ✅ | ✅ | ✅ | ✅ |
| `grouped_predictions.json` | ✅ | ✅ | ✅ | ✅ |

Judge prompt template and both eval scripts: **also byte-identical.**

→ Any difference comes from the **judge**, never from the inputs.

---

## Two naming traps

**1. `claude_evaluation_results.json` has nothing to do with Claude.**

Legacy hardcoded strings from upstream McMiner — also
`evaluation_method: "claude_existing"`. **No Anthropic model ran.**

**2. `JUDGE_PROVIDER=openrouter` does not mean OpenRouter.**

`OPENROUTER_BASE_URL` → `http://localhost:11434/v1` with a dummy key, because
Ollama speaks the OpenAI protocol and the SDK needs *some* key.

**Nothing left the machine.**

> Both invite exactly the wrong conclusion about what ran.

---

## The two judges

| Judge | Base | Weights | "Stop thinking" switch |
|---|---|---|---|
| `gpt-oss-judge` | `gpt-oss:20b` | ~13.8 GB | `reasoning_effort: low` |
| `qwen36-judge` | `qwen3.6:27b` | ~17 GB | `enable_thinking: false` |

Both `temperature 0`, `num_ctx 16384`, local Ollama, one at a time.

**Why two different switches?** Reasoning tokens come out of the *same budget as
the answer* — and the parser's defaults are **not neutral**.

| Model | Consequence of getting it wrong |
|---|---|
| `gpt-oss` | 4000/4000 tokens reasoning → **empty content** |
| `qwen3.6` | one call **>10 min** vs ~160 s |

---

## The hardware that made this possible

`REPORT.md` recorded `qwen3.6:27b` as **unusable** — 17 GB of weights against
15.7 GB of RAM meant Windows paged it to **disk**.

| | old (5070 Laptop) | **new (5070 Ti)** |
|---|---|---|
| VRAM | 7.7 GB | **16 GB** |
| System RAM | 15.7 GB | **32 GB** |
| `gpt-oss:20b` | 59/41 CPU/GPU | **~100% GPU** |
| `qwen3.6:27b` | **paged to disk**, 0.6–1.3 tok/s | ~2–5 GB spill to **RAM** |

The 32 GB is what matters — spill goes to RAM, not the page file.

**That made qwen3.6 runnable. It did not make it fast enough.**

---

## Execution — 17 minutes vs 91 hours

| Judge | Span | |
|---|---|---|
| `gpt-oss-judge` | 14:56:20 → 15:10:07 | **~17 min**, all 4 arms |
| `qwen36-judge` | 08-21 23:33 → 08-25 18:35 | **~91 h calendar** |

<!-- Timestamps mark step *completion*, so the 13m 47s span omits the first
     step's own duration; the other three S steps ran 186–232 s. -->


Per-step, McMiner-M stage only:

| Arm | Duration | Judge calls |
|---|---|---|
| baseline | **7h 32m** | 28 |
| ref | **7h 21m** | 26 |
| rag | 3h 24m | 20 |

**7½ hours for 28 judge calls.**
The same 28 calls took `gpt-oss-judge` **54 seconds**.

---

## Pipeline health — the finding

| Judge | Attempted | **Timeouts** | Mistagged | Clean |
|---|---|---|---|---|
| `gpt-oss-judge` | 582 | **0** (0.0%) | 5 | **577 (99.1%)** |
| `qwen36-judge` | 465 | **145 (31.2%)** | 19 | **301 (64.7%)** |

All 145 carry the same rationale: `judge error: Request timed out.`

| Arm | S calls | S timeouts | M calls | M timeouts |
|---|---|---|---|---|
| baseline | 131 | **44** | 28 | **11** |
| rag | 106 | 29 | 20 | 2 |
| ref | 126 | **40** | 26 | **11** |
| rag_ref | *aborted* | — | 28 | 8 |

---

## ⚠️ A timeout is scored as a non-match

```python
except Exception as e:
    parsed = {"match": False, "match_with_novel": False, ...
              "method": "judge_error", "parse_ok": False}
```

**There is no "unknown" state.**

A call that never returned is indistinguishable, in the metrics file, from a
judge that looked at the prediction and said **no**.

> 145 calls that never happened are sitting in the results table as
> **145 negative verdicts**.

---

## Root cause: two SDK defaults nobody set

```python
self.client = OpenAI(api_key=api_key, base_url=base_url)
#                    ^ no timeout      ^ no max_retries
```

SDK defaults: **timeout = 600 s**, **max_retries = 2**

→ one slow call burns **600 × 3 = 30 minutes**, then scores as a non-match.

**Reconstructing baseline McMiner-M:**

| | |
|---|---|
| 11 timeouts × 30 min | 330 min |
| 17 completed × ~7 min | 119 min |
| **total** | **~7h 29m** |
| **measured** | **7h 32m** |

**There is no env var for the timeout.** It needs a code change.

---

## The abort guard fired — correctly, once

`qwen36-judge/rag_ref/single_multi` → **no `judge_details_single.json`**,
165 of 184 evaluations marked `no_evaluation`.

Its reported **8.11%** is an empty cell, not a score.

```
baseline S  .X..XXX...XX......XXX...X...XX..X..X.X.X.XXX.XXX.......X....
            X.X.X..X........X...X..XXXX..XXX..........X..X........X.....
ref      S  X...XX............XX...X..X.XX.....XX.X.XX..X.XXXX.....XX...
```

`JUDGE_ABORT_AFTER=5` counts **consecutive** failures. Timeouts are
**scattered** — `rag_ref` just happened to hit five in a row.

**It saved one cell and missed seven.** A *rate-based* guard was what this run
needed.

---

## Results as scored

| Arm | `gpt-oss` S std | S nov | M std | M nov |
|---|---|---|---|---|
| baseline | 59.46% | 67.57% | 59.46% | 75.68% |
| rag | 56.76% | 70.27% | 62.16% | 64.86% |
| ref | 59.46% | 75.68% | 64.86% | 72.97% |
| **rag_ref** | **64.86%** | **78.38%** | **78.38%** | **86.49%** |

| Arm | `qwen3.6` S std | S nov | M std | M nov |
|---|---|---|---|---|
| baseline | 48.65% | 54.05% | 54.05% | 54.05% |
| rag | 54.05% | 54.05% | 59.46% | 59.46% |
| ref | 48.65% | 51.35% | 48.65% | 48.65% |
| rag_ref | ~~8.11%~~ | — | 64.86% | 64.86% |

---

## Two signatures that mark the qwen column broken

**Before any accuracy argument:**

**1. Novelty score = standard score in 6 of 7 cells.**

The novelty-aware metric can only ever go **up**. A judge awarding *zero*
novelty credit across 33 bags is not judging novelty.

→ Because a timeout sets **both** fields to `False` at once.

**2. `rag_ref` McMiner-S at 8.11%** — that is the empty cell.

> When a metric that is mathematically ≥ another one **equals** it everywhere,
> stop reading the table and go look at the pipeline.

---

## What the completed calls actually show

| Arm | Mode | **as scored** | **completed calls only** |
|---|---|---|---|
| baseline | S | 61.8% | **92.7%** |
| baseline | M | 57.1% | **100.0%** |
| rag | S | 71.7% | **98.6%** |
| rag | M | 90.0% | **100.0%** |
| ref | S | 65.1% | **95.0%** |
| ref | M | 57.7% | **100.0%** |
| rag_ref | M | 71.4% | **100.0%** |

The entire apparent gap was the timeouts.

---

## The conclusion inverts

Head-to-head, on cases **both** judges completed cleanly:

| Arm | Mode | n | Agreement | gpt-oss Y | qwen3.6 Y |
|---|---|---|---|---|---|
| baseline | S | 82 | 97.6% | 74 | **76** |
| baseline | M | 16 | 87.5% | 14 | **16** |
| rag | S | 72 | 97.2% | 69 | **71** |
| rag | M | 17 | 94.1% | 16 | **17** |
| ref | S | 78 | 94.9% | 70 | **74** |
| ref | M | 15 | 100% | 15 | 15 |
| rag_ref | M | 18 | 100% | 18 | 18 |

**qwen3.6 ≥ gpt-oss in all 7 cells, strictly more in 5.**
Judges agree on **94.9–100%** of shared cases.

→ **qwen3.6 is the *more lenient* judge** — the opposite of the raw table.

⚠️ Establishes the **direction** of the error, not a corrected score.

---

## Read κ, not the totals

`compare_judges.py`, on the as-scored verdicts:

| Arm | M agree | κ | S agree | κ |
|---|---|---|---|---|
| baseline | 83.8% | 0.67 | 97.7% | 0.84 |
| rag | 91.9% | 0.83 | **96.1%** | **0.39** |
| ref | 83.8% | 0.68 | 95.3% | 0.64 |
| rag_ref | 86.5% | 0.67 | *aborted* | — |

**`rag` McMiner-S: 96.1% agreement at κ = 0.39.**

High raw agreement with **near-chance κ** is the signature of a sparse,
artifact-driven disagreement pattern.

---

## The token budget moves scores more than the arms do

Same model. Same prompt. Same predictions. Same temperature.
**Only `JUDGE_MAX_TOKENS` changed: 1500 → 3000.**

| Arm | S std | S novel | M std | M novel |
|---|---|---|---|---|
| baseline | 0.0 | **−5.41** | **−8.11** | −2.70 |
| rag | 0.0 | 0.0 | +2.70 | 0.0 |
| ref | +2.70 | **+5.41** | 0.0 | 0.0 |
| rag_ref | 0.0 | −2.70 | **+8.11** | +5.41 |

Mean shift **< 1 pp** — but individual arms move **up to 8.11 pp = three bags**.
Direction inconsistent → **variance, not bias**.

**`rag_ref`'s McMiner-M lead grew 70.27% → 78.38% from a token budget change.**

---

## Why that matters twice

**1. It is larger than the effect under study.**

`REPORT.md` §6.2: one bag = 2.70 pp, every Wilson interval overlaps every other.

Now add: a **judge-side knob unrelated to prompt design** is worth three bags.

**2. It confounds the original bias measurement.**

Three runs, three budgets — nobody set them deliberately:

| Run | `JUDGE_MAX_TOKENS` |
|---|---|
| original self-judge | **1500** ← silent code default |
| GPT-5 | **4000** |
| this run | **3000** |

The **+7.4 pp** self-judging figure compared **GPT-5@4000 vs gpt-oss@1500**.
Direction still stands (negative in all 4 arms). **Magnitude is not clean.**

---

## Three judges, same predictions

| Arm | gpt-oss @1500 | **GPT-5 @4000** | gpt-oss @3000 | qwen3.6 |
|---|---|---|---|---|
| **McMiner-S standard** |
| baseline | 59.46% | 48.65% | 59.46% | ~~48.65%~~ |
| rag | 56.76% | 54.05% | 56.76% | ~~54.05%~~ |
| ref | 56.76% | 48.65% | 59.46% | ~~48.65%~~ |
| **rag_ref** | **64.86%** | **56.76%** | **64.86%** | ~~aborted~~ |
| **McMiner-M standard** |
| baseline | 67.57% | 64.86% | 59.46% | ~~54.05%~~ |
| rag | 59.46% | 59.46% | 62.16% | ~~59.46%~~ |
| ref | 64.86% | 56.76% | 64.86% | ~~48.65%~~ |
| **rag_ref** | **70.27%** | **72.97%** | **78.38%** | ~~64.86%~~ |

`rag_ref` leads under **every** judge — but overlapping CIs still apply.

**The quotable column remains GPT-5.**

---

## Bonus: the tag defect is not model-specific

`REPORT.md` recorded 5 cases of `gpt-oss` writing `</match_withnovel>`.
It reproduces here — **and qwen is 5× worse**:

| Judge | Mistagged |
|---|---|
| `gpt-oss-judge` | 5 / 582 (0.9%) |
| `qwen36-judge` | **19 / 465 (4.1%)** |

```xml
<match>N</match>
<match_with_novel>Y</match>     <!-- closes the WRONG tag -->
```

The regex requires the closing tag → `match_with_novel` falls back to `match`,
**discarding an explicit `Y`**.

Tolerant regex across all 24 recovers **exactly 1** lost credit —
**no reported number changes.** Still worth fixing: it stayed harmless by luck.

---

## Threats to validity

1. **`qwen36-judge` results are invalid** — 31.2% timeouts scored as non-matches
2. **The correction shows direction, not magnitude** — completed-call subset may
   favour easier cases
3. **Judge token budget is a live variable** — up to 8.11 pp/arm, differs across
   all three runs
4. **No timeout configuration exists** — 600 s / 2 retries unreachable without a
   code edit
5. **`JUDGE_ABORT_AFTER` is consecutive-only** — cannot catch scattered failures
6. **All mining-run caveats still apply** — n=37, `reasoning_effort=low`,
   19 unique correct programs, reconstructed judge prompt
7. **`enable_thinking: false` may not have been honoured** — no visible reasoning
   trace, yet a quarter of calls exceeded 600 s vs a documented ~160 s.
   **Unresolved — check this first.**

---

## Recommended next steps

1. **Make the timeout configurable and set it high** — `timeout=1800`,
   `max_retries=0`
2. **Never score an unreturned call** — `judge_error` should **exclude**, not
   mark `match=False`; report `n_judged` beside every accuracy
3. **Add a rate-based abort** — halt above ~10% rolling failure rate
4. **Verify `enable_thinking: false` reaches the model**, then re-run qwen
5. **Pin `JUDGE_MAX_TOKENS` explicitly** in every script — no silent defaults
6. **Apply the tolerant-regex fix** — 4.1% on a new model is the warning 0.9%
   did not give
7. **Restate `REPORT.md` §7.2's +7.4 pp** with the budget confound noted

---

<!-- _paginate: false -->

# Summary

**The premise is sound. The plumbing is not.**

`gpt-oss-judge`: **582 calls, ~17 min, 99.1% clean** — a local judge is entirely
practical on this hardware.

`qwen36-judge`: **31.2% timeouts**, four days, results unusable.

**Three findings:**
- A timeout scored as a **non-match** produced a whole fabricated column
- On completed calls the judges **agree 94.9–100%** — and qwen is the **lenient** one
- A **token budget change** moved one arm by **8.11 pp**, more than any arm gap

**The hardware was never the obstacle — the client configuration was.**

Full detail: `REPORT_localjudge.md` · `REPORT.md`
