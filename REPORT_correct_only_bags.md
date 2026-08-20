# Correct-Only Bags — Focused Report

**Companion to [`REPORT.md`](REPORT.md).** Run of 2026-08-17, `gpt-oss:20b`,
four arms.

> ⚠️ **This report corrects two claims made in `REPORT.md` §2.3 and §7.5.** The
> correct-only bag coverage was described there as deficient. It is not. See §2.

---

## 1. What a correct-only bag is

McMiner-M mines a *bag* of related codes at once rather than one code at a time.
Most bags group codes that share an injected misconception. A **correct-only
bag** contains nothing but correct programs, so the ground truth is **NONE**:
the bag scores as a match only if the model predicts *no misconception*.

They are the false-positive control. Without them, a model that flags a
misconception everywhere would look excellent.

---

## 2. Correction: the 96 "correct codes" are 19 unique programs

The dataset has **96** files under `pseudocode_track/pseudocode_codes_none`.
Those 96 files span only **19 distinct `problem_id`s**, and every one of them
carries the literal string `NONE` as its code — there is exactly **1** distinct
code string across all 96 files before substitution.

At prompt-build time `run_infer_misc.py` substitutes the problem's first correct
solution. Since there are 19 problems, the model is shown **19 unique
programs**, each appearing between **1 and 12 times** under different
misconception labels:

| | Value |
|---|---|
| Correct NONE files | 96 |
| Distinct `problem_id`s | **19** |
| Distinct code strings in the files | 1 (all literal `NONE`) |
| Distinct substituted programs the model saw | **19** |
| Files per problem | 1 to 12 |

A file named `problem_130_misc_38.json` in this directory does not mean "a
correct code for misconception 38". It means "misconception 38 was judged
**inapplicable** to problem 130". All 3 such files for problem 130 resolve to
the *same* correct program.

### What this corrects

`REPORT.md` stated that only ~20 of 96 correct codes entered bags and that
McMiner-M's correct-bag accuracy therefore "rests on a 4-bag subsample" that is
"indicative only". That framing was wrong. The bag former **deduplicates by
problem**, so the 4 correct-only bags cover **19 of 19 unique programs —
complete coverage**:

```
bag 1: correct_problem_313, 242, 93, 176, 94
bag 2: correct_problem_335, 385, 121, 213, 348
bag 3: correct_problem_178, 54, 200, 60, 501
bag 4: correct_problem_130, 152, 46, 73
        -> 19 distinct programs, all 19 covered
```

The correct-only side is *not* undersampled relative to the data available. The
real constraint is that the dataset contains only 19 correct programs at all.

---

## 3. Results: the model never false-positives on a bag

| Arm | Correct-only bags | Misconception bags | Overall |
|---|---|---|---|
| baseline | **4/4 = 100%** | 21/33 = 63.6% | 25/37 = 67.57% |
| rag | **4/4 = 100%** | 18/33 = 54.5% | 22/37 = 59.46% |
| ref | **4/4 = 100%** | 21/33 = 63.6% | 25/37 = 67.57% |
| rag_ref | **4/4 = 100%** | 22/33 = 66.7% | 26/37 = 70.27% |

**Perfect on correct-only bags in every arm.** All four bags in all four arms
returned `no_predicted_misconceptions = true`.

This means the entire McMiner-M accuracy spread between arms comes from the 33
misconception bags. Correct-only bags contribute a constant +4 to every arm, and
inflate every arm's overall figure by the same ~10.8 pp.

---

## 4. The striking part: bags beat individual codes at abstaining

The same model, shown the same 19 programs, behaves completely differently
depending on whether it sees them **individually** or **in a bag**:

| | Correct programs, mined individually (McMiner-S) | Correct programs, mined as bags (McMiner-M) |
|---|---|---|
| baseline | 57/96 rows = **59.4%** correct abstention | **100%** |
| ref | 64/96 = 66.7% | **100%** |
| rag | 75/96 = 78.1% | **100%** |
| rag_ref | 82/96 = 85.4% | **100%** |

Shown one correct program on its own, the baseline model invents a misconception
**4 times in 10**. Shown five correct programs together, it invents one **zero
times out of 16 bag-arm combinations**.

A plausible mechanism: the bag prompt asks *"what misconception do these codes
share?"*, and finding no common thread across five unrelated correct programs is
easy. The single-code prompt asks *"what misconception does this code have?"*,
which pushes toward producing an answer. If that reading is right, the
false-positive rate is partly an artefact of prompt framing rather than a pure
capability limit — but this run cannot separate the two, since bag size and
prompt wording change together.

---

## 5. The model is inconsistent on identical input

Because 19 programs are mined 1–12 times each, the run accidentally measures
**self-consistency**: the same program, the same prompt, the same temperature
(0.1), asked repeatedly.

Baseline, per program:

| Problem | Times mined | Said NONE | Rate | |
|---|---|---|---|---|
| 46 | 2 | 2 | 100% | |
| 54 | 6 | 0 | 0% | |
| **60** | **12** | **7** | **58%** | ⚠️ inconsistent |
| 73 | 1 | 1 | 100% | |
| **93** | **5** | **1** | **20%** | ⚠️ inconsistent |
| 94 | 9 | 9 | 100% | |
| 121 | 1 | 0 | 0% | |
| 130 | 3 | 3 | 100% | |
| **152** | **5** | **4** | **80%** | ⚠️ inconsistent |
| 176 | 4 | 4 | 100% | |
| 178 | 3 | 3 | 100% | |
| 200 | 5 | 0 | 0% | |
| 213 | 7 | 7 | 100% | |
| 242 | 1 | 0 | 0% | |
| **313** | **6** | **3** | **50%** | ⚠️ inconsistent |
| 335 | 7 | 7 | 100% | |
| **348** | **5** | **2** | **40%** | ⚠️ inconsistent |
| 385 | 4 | 4 | 100% | |
| 501 | 10 | 0 | 0% | |

**5 of 19 programs got different answers on identical input.** Problem 60 is a
coin flip across 12 attempts.

Consistency improves as prompt context is added:

| Arm | Rows | Programs (majority vote) | Programs answered inconsistently |
|---|---|---|---|
| baseline | 59.4% | 11/19 = 57.9% | **5/19** |
| ref | 66.7% | 14/19 = 73.7% | 6/19 |
| rag | 78.1% | 15/19 = 78.9% | 2/19 |
| rag_ref | 85.4% | 17/19 = 89.5% | **1/19** |

`rag_ref` is both the most accurate *and* the most stable on correct code — one
inconsistent program out of 19, versus five at baseline. Note `ref` alone is the
least stable (6/19), so the effect is not simply "more context is better".

### Why the per-row rate is still usable

Per-row (59.4%) and per-program majority (57.9%) agree closely at baseline, so
the redundancy does not badly distort the headline number. But the **effective
sample size for correct-code abstention is 19, not 96**, and rows are weighted
by how many misconceptions happened to be declared inapplicable for that problem
— problem 60 carries 12 rows, problems 73/121/242 carry 1 each. That weighting
is an artefact of dataset construction, not a design choice.

---

## 6. Defect found: correct-only bags share one `prediction_id`

All four correct-only bags are emitted with the **same** id:

```
group_correct_only_None_0   (x4)
```

So `multi_predictions.json` holds 37 predictions under only **34 distinct ids**,
and the same collision propagates into `grouped_predictions.json` and
`claude_evaluation_results.json`.

**The reported metrics are unaffected.** Both the judge loop and the metric
computation iterate the predictions *list* and append one detail per element, so
all 37 bags are scored individually — `evaluation_details` correctly holds 37
entries and `summary.total` is 37.

**What it does break** is any id-keyed lookup or join downstream — building a
`{prediction_id: detail}` map silently collapses the four bags into one. That is
a live hazard for analysis scripts (it caught me while writing this report) and
for the existing `eval_map` pattern in
`evaluate_single_multi_predictions.py`.

Fix: include the group index in the id when `group_type == "correct_only"`, the
way misconception bags already do.

---

## 7. Implications

1. **Correct-only bags are not the weak spot** — they are at ceiling (100%,
   every arm) and cover all 19 available programs. My earlier characterisation
   was wrong.
2. **They flatter every arm equally.** They add a constant +10.8 pp to McMiner-M
   overall accuracy, so between-arm comparison should really be read on the 33
   misconception bags alone.
3. **The false-positive problem is a McMiner-S problem**, not a model-wide one.
   Individually mined correct code draws a spurious misconception 15–40% of the
   time; bagged correct code never does.
4. **Non-determinism is material.** 5 of 19 programs flip their answer on
   identical input at baseline. Any single-run comparison of arms is measuring model noise as well
   as prompt effect — which reinforces the `REPORT.md` §6.2 conclusion that the
   arm ranking is not supported at this sample size.
5. **19 correct programs is the real ceiling** on false-positive measurement in
   this dataset. Adding rows cannot fix it; adding problems would.

---

## 8. Recommended follow-ups

1. **Make correct-only bag ids unique** (§6).
2. **Report McMiner-M accuracy split** by `misconception` vs `correct_only`
   rather than pooled, so the constant +4 does not mask arm differences.
3. **De-duplicate correct-code mining**, or weight per program rather than per
   row, so the 12-row problem 60 does not outweigh the 1-row problem 73.
4. **Measure non-determinism deliberately** — repeat one arm at fixed settings
   and report variance, to establish how much of any arm gap is noise.
5. **Probe the bag-vs-single framing effect** by mining single correct codes with
   the bag-style prompt ("what misconception, if any, do these share?") to test
   whether the false-positive gap is prompt framing or bag size.
