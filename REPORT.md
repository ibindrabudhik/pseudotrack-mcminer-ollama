# McMiner Pseudocode Track on a Local Model — Run Report

**Run date:** 2026-08-17
**Mining model:** `gpt-oss:20b` via Ollama, 16K context
**Judges:** `gpt-oss:20b` (self-evaluation) **and** `gpt-5` via OpenRouter (independent)
**Total wall time:** 7h 56m mining + 2h 01m independent judging
**Hardware:** NVIDIA RTX 5070 Laptop (8 GB VRAM), 15.7 GB system RAM, Windows 11

---

## 1. Summary

The McMiner pseudocode track was run end-to-end against a locally served
`gpt-oss:20b`, across all four prompt arms, with both McMiner-S (per-code
mining) and McMiner-M (whole-bag mining) evaluated in every arm.

All four arms completed with **zero mining parse failures** across 1,220
predictions. Every arm was then judged twice: once by the mining model itself,
and once by **GPT-5 as an independent judge** (§7), which is what makes these
numbers comparable to the McMiner paper's main table.

Three findings, in order of confidence:

1. **Self-judging inflated McMiner-S standard accuracy by a mean of 7.4 pp**,
   in the same direction in all four arms (§7.2). The independent numbers are
   the ones to quote.
2. **Retrieval context systematically increases abstention** — more true
   negatives *and* more false negatives, effects that partly cancel in the
   accuracy table (§6.3).
3. **At n=37 bags per arm the between-arm ranking is not statistically
   distinguishable** (§6.2). `rag_ref` leads under both judges, but every
   confidence interval overlaps every other. The ranking should not be quoted
   as a finding.

---

## 2. Data

### 2.1 Pseudocode corpus

| Item | Count |
|---|---|
| Corrupted pseudocode files (one injected misconception each) | **209** |
| Correct pseudocode files (ground truth = NONE) | **96** |
| **Total codes mined per arm** | **305** |
| Misconception bank entries | 22 |
| Problem descriptions | 19 |

The pseudocode is *Notasi Algoritmik* (Telkom University), not Python — `<-` for
assignment, `kamus`/`algoritma` blocks, `endfor`/`endwhile`. The mining prompt
tells the model to judge against this notation rather than Python.

The corrupted files are synthetic: a known misconception was injected into a
reference solution. That injected misconception is the ground truth. The 96
correct files carry the literal string `NONE` as their code; the pipeline
substitutes the problem's first correct solution at prompt-build time, so the
model sees real working code and should predict "no misconception".

### 2.2 Prompt-aid sources

| File | Rows | Used by |
|---|---|---|
| `retrival_openai_embedding_large.csv` | 418 | RAG arms (similar submissions) |
| `retrival_correct_codes.csv` | 19 | RAG arms (correct references) |
| `Submission_Code_with_reference_from_APR.csv` | 29,378 | REF arms (APR-repaired reference) |

A single RAG retrieval CSV is used across both RAG arms. Upstream defaulted the
`rag` and `rag_ref` scripts to *different* CSVs, which made those two arms
non-comparable; this bundle fixes that.

### 2.3 Bag construction for McMiner-M

Bags group codes sharing a misconception, so the model mines a misconception
from several examples at once (Multiple Instance Learning).

| Item | Count |
|---|---|
| Total bags | **37** |
| — misconception bags | 33 |
| — correct-only bags | 4 |
| Bag size | 5 (one bag of 4) |
| Codes covered by bags | 184 |

**On the 4 correct-only bags.** These cover **19 of 19 unique correct
programs — complete coverage.** The 96 files in `pseudocode_codes_none` span
only 19 distinct `problem_id`s and all contain the literal string `NONE`; the
pipeline substitutes the problem's correct solution at prompt-build time, so the
model sees 19 unique programs, each appearing 1–12 times under different
misconception labels. The bag former deduplicates by problem, hence 4 bags.

The real constraint is therefore that the dataset contains only 19 correct
programs, not that bagging undersamples them. Full analysis, including the
model's 100% correct-only bag accuracy and its inconsistency on repeated
identical input, is in [`REPORT_correct_only_bags.md`](REPORT_correct_only_bags.md).

---

## 3. Configuration

```bash
MODEL=gpt-oss-mcminer:latest \
PYTHON=./.venv/Scripts/python.exe \
REASONING_EFFORT=low \
bash run_all.sh
```

| Setting | Value | Why |
|---|---|---|
| Model | `gpt-oss:20b` (MoE, ~3.6B active) | Only capable model that fits this hardware |
| Context | 16,384 | Worst-case prompt ~10K tokens + 4K response cap |
| Reasoning effort | **low** | Mandatory — see §4.2 |
| Judge | Same model, same endpoint | Self-evaluation; see §7 |
| CPU/GPU split | 59% / 41% | 13.8 GB weights vs 7.7 GB VRAM |
| Throughput | ~27–34 tok/s gen, ~600 tok/s prompt | Measured warm |

### Why not a larger model

`qwen3.6:27b` was tried first and is not viable on this machine. Its 17 GB of
weights exceed both the 7.7 GB of VRAM *and* the 15.7 GB of system RAM, so
Windows paged the model to disk (`llama-server` working set fell to 0 GB while
private bytes held 22.8 GB, with system commit at 99%). Measured throughput was
**0.6–1.3 tok/s generation**, and a single judge call exceeded 10 minutes. The
full track would have taken weeks.

`gpt-oss:20b` fits because its mixture-of-experts design activates only ~3.6B
parameters per token, so the CPU-resident share costs far less. Its working set
stayed **7.39 GB resident** — nothing paged.

---

## 4. Pipeline

Each arm runs five steps:

| Step | Action | Output |
|---|---|---|
| 1 | McMiner-M: form bags, mine each whole bag | `multi/multi_predictions.json`, `multi_summary.json` |
| 2 | McMiner-S: mine each corrupted code | `single/predictions.json`, `summary.json` |
| 2b | McMiner-S: mine each correct code → NONE | appended to the same file |
| 3 | Align single predictions into bags | `single_multi/grouped_predictions.json` |
| 4 | Evaluate McMiner-S (LLM judge) | `evaluation_metrics.json`, `bag_evaluation_results.json`, `judge_details_single.json` |
| 5 | Evaluate McMiner-M (LLM judge) | `claude_evaluation_results.json`, `evaluation_metrics.json` |

### 4.1 Defects found and fixed before the run

Four problems, each of which would have invalidated or blocked the run:

1. **The judge prompt template was missing.** `compute_eval_metrics_multi.py`
   reads `src/prompt_templates/evaluation-pseudocode/judge_prediction_match.md`,
   which shipped in neither this bundle nor upstream. Both eval steps would have
   crashed with `FileNotFoundError`. The template was written against the
   parser's contract, using the criteria from the paper's real evaluation
   prompts in `src/evaluation-from-mcmining/` — the "misconception is a false
   belief, not a bug" framing, the three analysis guidelines, and the Multiple
   Instance Learning rule for bags.

2. **`requirements.txt` is not installable on Python 3.13.** It lists `pathlib`
   and `argparse` (stdlib backports that fail to build) plus `streamlit`,
   `psycopg2-binary` and `sqlalchemy`, none of which this pipeline imports. A
   venv was built with the seven packages actually imported.

3. **`UnicodeEncodeError` on Windows.** The pipeline prints status banners
   containing emoji; Python defaults stdout to cp1252, which cannot encode them,
   so the first banner raised before any work started. Because `run_all.sh`
   catches per-arm failures and continues, this would have produced four empty
   result sets while still exiting cleanly. Fixed with `PYTHONIOENCODING=utf-8`.

4. **Step 4 deleted the judge's per-prediction verdicts.** They were written to
   a scratch directory that was removed on the way out, leaving McMiner-S with
   aggregate numbers and no way to inspect *why* anything scored as it did. Now
   preserved as `judge_details_single.json`.

### 4.2 Why `reasoning_effort=low` is mandatory

At its default `medium` effort, `gpt-oss` spends its entire 4,000-token budget
reasoning and returns `finish_reason='length'` with **empty content**. The miner
records that as `parse_success=false` with `no_predicted_misconceptions=true` —
a *silent false negative*, not an error.

Measured on a correct-code mining prompt:

| Effort | Time | finish_reason | Completion tokens | Usable output |
|---|---|---|---|---|
| medium | 139.7s | `length` | 4000 / 4000 | **no** |
| low | 14.3s | `stop` | 312 | yes |

Mining parse success across the smoke test went from **66.67% to 100%**. This
was not a speed optimisation — at medium the model produces nothing at all on
hard prompts. Ollama's OpenAI-compatible endpoint accepts a top-level
`reasoning_effort` field, which is distinct from OpenRouter's nested
`extra_body.reasoning`; the plumbing was added behind an env var so
hosted-provider paths are unaffected.

**Caveat:** the quality cost of `low` effort is unmeasured. It was forced by
necessity, not chosen on merit.

---

## 5. Execution

| Arm | Bags (M) | Single corrupted | Single correct | Judge S | Judge M | Total |
|---|---|---|---|---|---|---|
| baseline | 37 @ 11:51 | 209 @ 1:07:32 | 96 @ 23:46 | 165 @ 21:46 | 37 @ 05:11 | **2h 10m** |
| rag | 37 @ 10:18 | 209 @ 1:00:21 | 96 @ 18:06 | 165 @ 17:54 | 37 @ 03:49 | **1h 50m** |
| ref | 37 @ 09:59 | 209 @ 58:07 | 96 @ 23:21 | 165 @ 21:39 | 37 @ 04:45 | **1h 57m** |
| rag_ref | 37 @ 11:33 | 209 @ 1:03:03 | 96 @ 16:57 | 165 @ 20:08 | 37 @ 05:14 | **1h 56m** |

Total 7h 56m 35s. Notably the RAG/REF arms were *not* slower despite larger
prompts, because prompt evaluation runs at ~600 tok/s while generation runs at
~30 tok/s — generated length dominates, and the aided arms generate less.

### Pipeline health

| Arm | Mining predictions | Parse failures | Judge calls (S + M) | Judge parse failures |
|---|---|---|---|---|
| baseline | 305 | 0 | 131 + 28 | 0 |
| rag | 305 | 0 | 106 + 20 | 3 |
| ref | 305 | 0 | 126 + 26 | 2 |
| rag_ref | 305 | 0 | 117 + 28 | 0 |
| **Total** | **1,220** | **0** | **~480** | **5** |

---

## 6. Results

### 6.1 Accuracy

| Arm | McMiner-S standard | S novelty-aware | McMiner-M standard | M novelty-aware |
|---|---|---|---|---|
| baseline | 59.46% | 72.97% | 67.57% (25/37) | 78.38% (29/37) |
| rag | 56.76% | 70.27% | 59.46% (22/37) | 64.86% (24/37) |
| ref | 56.76% | 70.27% | 64.86% (24/37) | 72.97% (27/37) |
| **rag_ref** | **64.86%** | **81.08%** | **70.27% (26/37)** | **81.08% (30/37)** |

*Standard* = the predicted misconception matches the injected ground truth.
*Novelty-aware* = also credits a prediction that misses the ground truth but
identifies a different misconception the code genuinely exhibits.

### 6.2 These differences are not statistically distinguishable

With n=37 bags, **one bag is worth 2.70 percentage points.** Wilson 95%
confidence intervals:

| Arm | McMiner-M standard | 95% CI | Width |
|---|---|---|---|
| baseline | 67.57% | [51.5%, 80.4%] | 29 pp |
| rag | 59.46% | [43.5%, 73.7%] | 30 pp |
| ref | 64.86% | [48.8%, 78.2%] | 29 pp |
| rag_ref | 70.27% | [54.2%, 82.5%] | 28 pp |

**Every interval overlaps every other interval.** The `rag` and `ref` deficits
against baseline are −2.70 pp — literally one bag. `rag_ref`'s lead is two bags.

Conclusion: **rag_ref ranks first on all four measures, but this run cannot
establish that it is genuinely better.** Sample size, not prompt design, is the
binding constraint on every comparative claim here.

### 6.3 What the run does support

**The novelty-aware gain is consistent.** Because it is a *paired* comparison on
identical bags, it is far more trustworthy than any between-arm difference:

| Arm | McMiner-S gain | McMiner-M gain |
|---|---|---|
| baseline | +13.51 pp | +10.81 pp |
| rag | +13.51 pp | +5.41 pp |
| ref | +13.51 pp | +8.11 pp |
| rag_ref | +16.22 pp | +10.81 pp |

Consistent in direction and magnitude across all four arms. Roughly one
prediction in seven that misses the injected misconception nevertheless
identifies a real one.

**Retrieval context systematically increases abstention.** This is the clearest
monotonic pattern in the run:

| Arm | True negatives (correct code → NONE) | False negatives (missed injected misconception) |
|---|---|---|
| baseline | 57/96 = **59.4%** | 41/209 = **19.6%** |
| ref | 64/96 = 66.7% | 46/209 = 22.0% |
| rag | 75/96 = 78.1% | 71/209 = 34.0% |
| rag_ref | 82/96 = **85.4%** | 60/209 = **28.7%** |

Adding context makes the model *more conservative in both directions* — better
at recognising clean code, worse at spotting injected misconceptions. RAG has
the strongest effect. This is a genuine behavioural finding and it is not
visible in the accuracy table, because the two effects partly cancel.

**Abstention calibration is the model's real weakness.** Even at best, it
invents a misconception in ~15% of clean submissions, and at baseline it does so
in 4 of every 10. A prediction the judge never sees cannot be scored.

---

## 7. Independent judge: GPT-5

Mining is the expensive step and was already done, so only the two evaluation
steps were re-run against the existing predictions, with GPT-5 in place of the
local model. Run via `scripts/judge_with_gpt5.sh`.

| | |
|---|---|
| Judge | `openai/gpt-5` via OpenRouter (Chat Completions path) |
| Judge calls | 582 (480 McMiner-S + 102 McMiner-M) |
| Wall time | 2h 00m 47s |
| Errors / parse failures | **0 / 0** |
| Actual cost | **$4.66** (estimated $2.83–$24.36) |

### 7.1 Results under the independent judge

| Arm | S standard | S novelty | M standard | M novelty |
|---|---|---|---|---|
| baseline | 48.65% | 75.68% | 64.86% | **86.49%** |
| rag | 54.05% | 70.27% | 59.46% | 64.86% |
| ref | 48.65% | 81.08% | 56.76% | 78.38% |
| **rag_ref** | **56.76%** | **81.08%** | **72.97%** | 83.78% |

`rag_ref` again leads on three of four measures — the same ordering the local
judge produced, and subject to the same n=37 caveat from §6.2.

### 7.2 Self-evaluation bias, measured

Delta = GPT-5 minus self-judged, in percentage points:

| Arm | S standard | S novelty | M standard | M novelty |
|---|---|---|---|---|
| baseline | **−10.8** | +2.7 | −2.7 | +8.1 |
| rag | −2.7 | 0.0 | 0.0 | 0.0 |
| ref | **−8.1** | **+10.8** | **−8.1** | +5.4 |
| rag_ref | **−8.1** | 0.0 | +2.7 | +2.7 |

**Mean inflation on McMiner-S standard accuracy: +7.4 pp**, negative in all four
arms. The consistency of direction is what distinguishes this from noise.

The correction is **not** uniformly downward. Novelty scores *rose* under GPT-5
(baseline M +8.1, ref S +10.8). GPT-5 is stricter about "did the system identify
the injected misconception" and more generous about "the system identified a
different misconception the code genuinely exhibits". The two effects partly
cancel, which is why the pooled figures look more similar than the underlying
judgements are.

### 7.3 Per-case agreement: aggregates hide disagreement

Equal totals do not mean equal verdicts. Agreement and Cohen's κ on the same
predictions:

| Arm | M agreement | κ | S agreement | κ | local-only Y | GPT-5-only Y |
|---|---|---|---|---|---|---|
| baseline | 81.1% | 0.58 | 87.0% | 0.71 | 13 | 4 |
| rag | **100%** | **1.00** | 86.8% | **0.63** | 9 | 5 |
| ref | 86.5% | 0.72 | 88.9% | 0.75 | 12 | 2 |
| rag_ref | 86.5% | 0.67 | 88.0% | 0.72 | 11 | 3 |

Two things worth noting:

- **The `rag` arm's identical aggregate rates were misleading.** Three of its
  four metrics matched the local judge to the decimal. Its McMiner-M agreement
  is genuinely perfect (κ=1.00, all 37 bags), but its McMiner-S agreement is
  86.8% with the *lowest* κ of any arm — 9 predictions the local judge scored
  as matches that GPT-5 rejected, and 5 the other way. Same totals, different
  cases.
- **`local-only Y` exceeds `GPT-5-only Y` in every arm** (13v4, 9v5, 12v2,
  11v3). That asymmetry is the mechanical signature of self-favouring.

κ of 0.58–0.75 is moderate-to-substantial, not high: the two judges genuinely
disagree on roughly 12–19% of cases.

### 7.4 A path caveat

These calls went through OpenRouter's **Chat Completions** path, because
`is_gpt5_model` in `utils/llm_clients.py` tests `startswith("gpt-5")` and the
namespaced id `openai/gpt-5` fails that test. Calling OpenAI directly with the
bare id `gpt-5` instead routes to the **Responses API** with an explicit
`reasoning: {effort}` setting.

A spot check on one prediction returned `match=Y` via the direct route and
`match=N, novel=Y` via OpenRouter. All four arms here used the identical
OpenRouter path, so internal comparisons are consistent — but these numbers are
not interchangeable with the OpenAI-direct route, and a future run should fix
the prefix test rather than rely on which id string was passed.

---

## 8. Threats to validity

1. ~~**Self-evaluation.**~~ **Addressed.** Every arm was re-judged by GPT-5
   (§7). The self-judged figures in §6 are retained only as the contrast that
   makes the bias measurable (+7.4 pp mean inflation); **quote §7.1, not §6.1.**

2. **The judge prompt is a reconstruction.** The bundle's judge template was
   missing entirely. It was rebuilt from the criteria in the paper's real
   evaluation prompts, but the exact wording is not the paper's. Any formal use
   of these numbers should disclose this.

3. **Sample size.** n=37 bags. See §6.2.

4. **`reasoning_effort=low`** was forced by a token-budget failure, not chosen.
   Its accuracy cost is unquantified.

5. **Only 19 unique correct programs exist** in the dataset (§2.3), so
   false-positive measurement is capped at n=19 regardless of the 96 rows. The
   4 correct-only bags cover all 19 and score 100% in every arm, adding a
   constant +10.8 pp to McMiner-M overall accuracy — between-arm comparison is
   better read on the 33 misconception bags alone. Separately, the model gave
   *different* answers on identical input for 5 of those 19 programs at
   baseline, so a single run measures model noise alongside prompt effect. See
   [`REPORT_correct_only_bags.md`](REPORT_correct_only_bags.md).

6. **Known cosmetic parser defect.** `gpt-oss` sometimes writes the closing tag
   as `</match_withnovel>` while getting the opening `<match_with_novel>` right,
   which fails the judge regex. This occurred 5 times in ~480 judge calls. All
   five were verified against the judge's intended verdict — extracted from
   inside the malformed tag — and **all five scored exactly as intended**, so no
   reported number is affected. The defect is nonetheless worth fixing, because
   it only stayed harmless by coincidence: the fallback default for
   `match_with_novel` is `match`, which happens to be correct whenever
   `match=Y`. A tolerant regex (capture from `<tag>` up to the next `<`) removes
   the dependence on luck.

---

## 9. Recommended next steps

1. **Apply the tolerant-regex parser fix** so future runs do not depend on the
   coincidence described above.
2. **Increase bag count.** n=37 is what blocks every comparative conclusion.
   More bags per misconception would give the arm comparison real power.
3. **Fix correct-only bag coverage** so all 96 correct codes enter bags.
4. ~~Swap in an independent judge~~ — **done** (§7), via
   `scripts/judge_with_gpt5.sh`. $4.66 and 2 hours.
5. **Re-test `medium` reasoning effort** with a raised `max_tokens`, to measure
   what `low` costs in accuracy.
6. **Fix the GPT-5 routing test** in `utils/llm_clients.py` so a namespaced id
   (`openai/gpt-5`) reaches the Responses API rather than falling through to
   Chat Completions (§7.4).
7. **Re-judge with a third judge** if the arm ranking matters. Two judges
   disagree on 12–19% of cases (§7.3); a third would show whether `rag_ref`'s
   lead survives judge choice, independent of the sample-size problem.

---

## 10. Output locations

```
results/<tag>/single/predictions.json              McMiner-S predictions + raw_response
results/<tag>/single/summary.json                  parse rates, NONE stats
results/<tag>/multi/multi_predictions.json         McMiner-M bag predictions
results/<tag>/multi/multi_summary.json             bag formation parameters
results/<tag>/single_multi/grouped_predictions.json aligned S-into-bags
results/evaluations/<tag>/single_multi/
    evaluation_metrics.json                        McMiner-S metrics
    bag_evaluation_results.json                    per-bag detail
    judge_details_single.json                      per-prediction judge verdicts
results/evaluations/<tag>/multi/
    claude_evaluation_results.json                 per-bag judge verdicts
    evaluation_metrics.json                        McMiner-M metrics

results/evaluations_gpt5/<arm>/                    SAME layout, GPT-5 judged
    single_multi/evaluation_metrics.json           <- quote these
    multi/claude_evaluation_results.json
```

where `<tag>` is `ollama_gpt-oss-mcminer-latest_<arm>`. The self-judged tree
(`results/evaluations/`) and the independent tree (`results/evaluations_gpt5/`)
are kept side by side so the bias in §7.2 stays reproducible.

---

## Attribution

Pipeline code, prompt templates and dataset derive from
[McMiner](https://github.com/taisazero/mcminer) (MIT, © 2025 Erfan Al-Hossami).
This report covers a local-Ollama repackaging of that work.
