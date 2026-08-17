# McMiner Pseudocode Track on a Local Model — Run Report

**Run date:** 2026-08-17
**Model (mining and judging):** `gpt-oss:20b` via Ollama, 16K context
**Total wall time:** 7h 56m 35s, all four arms completed
**Hardware:** NVIDIA RTX 5070 Laptop (8 GB VRAM), 15.7 GB system RAM, Windows 11

---

## 1. Summary

The McMiner pseudocode track was run end-to-end against a locally served
`gpt-oss:20b`, across all four prompt arms, with both McMiner-S (per-code
mining) and McMiner-M (whole-bag mining) evaluated in every arm.

All four arms completed with **zero mining parse failures** across 1,220
predictions. The headline accuracy numbers are reported below, but the single
most important result of this run is a negative one: **at n=37 bags per arm, the
differences between arms are not statistically distinguishable.** The ranking
should not be quoted as a finding. What the run does establish is that the
pipeline is sound, that the novelty-aware metric behaves consistently, and that
retrieval context systematically increases the model's tendency to abstain.

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

**Limitation worth flagging.** Only 4 correct-only bags were formed, covering
roughly 20 of the 96 correct codes. The recorded grouping parameter is
`correct_bags_ratio = 0.15` rather than the cover-all policy the runner intends
to pass. Consequently McMiner-M's correct-only accuracy rests on a 4-bag
subsample and should be treated as indicative only.

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

## 7. Threats to validity

1. **Self-evaluation.** `gpt-oss:20b` judged its own mining output. These
   numbers are **not comparable** to the paper's o3-mini / gemini-2.5-flash /
   qwen3-14b rows, which were judged by GPT-5. Use them to compare the four arms
   against each other under one judge, nothing more.

2. **The judge prompt is a reconstruction.** The bundle's judge template was
   missing entirely. It was rebuilt from the criteria in the paper's real
   evaluation prompts, but the exact wording is not the paper's. Any formal use
   of these numbers should disclose this.

3. **Sample size.** n=37 bags. See §6.2.

4. **`reasoning_effort=low`** was forced by a token-budget failure, not chosen.
   Its accuracy cost is unquantified.

5. **Correct-only bags undersampled.** 4 bags covering ~20 of 96 correct codes
   (§2.3), so McMiner-M's correct-bag accuracy is weakly supported.

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

## 8. Recommended next steps

1. **Apply the tolerant-regex parser fix** so future runs do not depend on the
   coincidence described above.
2. **Increase bag count.** n=37 is what blocks every comparative conclusion.
   More bags per misconception would give the arm comparison real power.
3. **Fix correct-only bag coverage** so all 96 correct codes enter bags.
4. **Swap in an independent judge** (e.g. GPT-5 via OpenRouter — the bundle
   supports this via `JUDGE_PROVIDER`/`JUDGE_MODEL`) to remove self-evaluation
   bias and produce paper-comparable numbers. Mining stays local.
5. **Re-test `medium` reasoning effort** with a raised `max_tokens`, to measure
   what `low` costs in accuracy.

---

## 9. Output locations

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
```

where `<tag>` is `ollama_gpt-oss-mcminer-latest_<arm>`.

---

## Attribution

Pipeline code, prompt templates and dataset derive from
[McMiner](https://github.com/taisazero/mcminer) (MIT, © 2025 Erfan Al-Hossami).
This report covers a local-Ollama repackaging of that work.
