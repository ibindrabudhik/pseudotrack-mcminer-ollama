---
marp: true
theme: default
paginate: true
size: 16:9
header: 'McMiner Pseudocode Track — Local Run'
footer: '2026-08-17 · gpt-oss:20b · RTX 5070 Laptop'
---

<!-- _paginate: false -->
<!-- _header: '' -->
<!-- _footer: '' -->

# Mining Programming Misconceptions from Student Pseudocode

## McMiner Pseudocode Track, run locally on `gpt-oss:20b`

**Four prompt arms · mined locally · judged twice**

7h 56m mining · 2h 01m independent judging · zero parse failures

2026-08-17

---

## What the task is

Given a student's pseudocode solution, **identify the programming misconception**
the student holds.

A *misconception* is a **false belief about a language construct** — not a bug.

> "The student believes the first character of a string is at index 1" ✅ specific
> "The student has an unclear understanding of loops" ❌ too vague

**Key point:** misconceptions do **not** always produce broken code. Code can be
syntactically and logically correct and still reveal a false belief.

---

## The data — corpus

| Item | Count |
|---|---|
| Corrupted pseudocode files (1 injected misconception each) | **209** |
| Correct pseudocode files (ground truth = NONE) | **96** |
| **Total codes mined, per arm** | **305** |
| Misconception bank | 22 |
| Problem descriptions | 19 |

- Language is **Notasi Algoritmik** (Telkom University), not Python
  — `<-` assignment, `kamus` / `algoritma` blocks, `endfor`
- Corrupted files are **synthetic**: a known misconception was injected into a
  reference solution → that injection is the ground truth
- The 96 correct files test whether the model can say **"nothing wrong here"**

---

## The data — prompt aids

| Source file | Rows | Used by |
|---|---|---|
| `retrival_openai_embedding_large.csv` | 418 | RAG (similar submissions) |
| `retrival_correct_codes.csv` | 19 | RAG (correct references) |
| `Submission_Code_with_reference_from_APR.csv` | 29,378 | REF (APR-repaired reference) |

One RAG retrieval CSV is used across **both** RAG arms.

> Upstream defaulted `rag` and `rag_ref` to *different* CSVs, which made those
> two arms non-comparable. Fixed here.

---

## Four prompt arms

| Arm | Prompt contains |
|---|---|
| `baseline` | problem + student pseudocode |
| `rag` | + retrieved similar submissions & correct codes |
| `ref` | + APR-repaired reference code |
| `rag_ref` | + both |

**Every arm runs two miners:**

- **McMiner-S** — mines each code individually (305 calls/arm)
- **McMiner-M** — mines a whole *bag* of related codes at once (37 calls/arm)

> Upstream scored McMiner-M in only 2 of 4 arms. All four are scored here.

---

## Bags — Multiple Instance Learning

McMiner-M groups codes that share a misconception and mines them **together**:
if at least one code in the bag exhibits it, the bag does.

| Item | Count |
|---|---|
| Total bags | **37** |
| — misconception bags | 33 |
| — correct-only bags | 4 |
| Bag size | 5 (one of 4) |
| Codes covered | 184 |

The 4 correct-only bags cover **19 of 19 unique correct programs** — the 96
"correct" files span only 19 problems and all contain the literal string `NONE`,
substituted at prompt time. **Coverage is complete**; 19 is the dataset's ceiling.

---

## First problem: the hardware

Target was `qwen3.6:27b`. **It does not fit this machine.**

| | qwen3.6:27b |
|---|---|
| Weights | 17 GB |
| VRAM available | 7.7 GB |
| System RAM | 15.7 GB |

Windows paged the whole model to disk:

- `llama-server` working set → **0 GB** (private bytes 22.8 GB)
- System commit → **99%** of limit
- Generation → **0.6–1.3 tok/s**; one judge call **>10 min**

**Weights alone exceed total RAM.** Not a tuning problem — a hardware ceiling.
The full track would have taken **weeks**.

---

## Model selection

No Ollama model that fits 7.7 GB VRAM / 15.7 GB RAM outperforms
gemini-2.5-flash or o3-mini. The qualifying Qwen tier starts at `qwen3:32b`
(20.2 GB) — *larger* than the model that already failed.

**Chosen: `gpt-oss:20b`** — 13.8 GB, mixture-of-experts, **~3.6B active
parameters per token**, so CPU spill costs far less than a dense model.

| | qwen3.6:27b | **gpt-oss:20b** | change |
|---|---|---|---|
| CPU/GPU split | 72 / 28 | **59 / 41** | more on GPU |
| Working set | 0 GB (paged) | **7.39 GB resident** | no paging |
| Prompt eval | 13–15 tok/s | **~600 tok/s** | **~45×** |
| Generation | 0.6–1.3 tok/s | **27–34 tok/s** | **~28×** |

---

## Four defects fixed before the run

1. **Judge prompt template was missing** — present in neither this bundle nor
   upstream. Both eval steps would have crashed. Rebuilt from the paper's real
   evaluation prompts.

2. **`requirements.txt` uninstallable** on Python 3.13 (`pathlib`, `argparse`
   backports; unused `streamlit`/`psycopg2`).

3. **`UnicodeEncodeError` on Windows** — emoji banners vs cp1252 killed every
   arm at the first print. `run_all.sh` would have reported four "completed"
   empty arms.

4. **Step 4 deleted the judge's verdicts** — McMiner-S produced aggregate
   numbers with no way to see *why*. Now kept as `judge_details_single.json`.

---

## The one that would have corrupted the results

`gpt-oss` defaults to **medium** reasoning effort and spends its *entire*
4,000-token budget thinking — returning **empty content**.

The miner records that as `parse_success=false` → **"no misconception predicted"**.
A **silent false negative**, not an error.

| Effort | Time | finish_reason | Tokens | Usable output |
|---|---|---|---|---|
| medium | 139.7s | `length` | 4000/4000 | ❌ **none** |
| **low** | **14.3s** | `stop` | **312** | ✅ yes |

Mining parse success: **66.67% → 100%**

> Not a speed optimisation — at medium the model produces *nothing* on hard
> prompts. Caveat: the accuracy cost of `low` is unmeasured.

---

## Pipeline — five steps per arm

| # | Step | Output |
|---|---|---|
| 1 | McMiner-M: form bags + mine each bag | `multi_predictions.json` |
| 2 | McMiner-S: mine 209 corrupted codes | `single/predictions.json` |
| 2b | McMiner-S: mine 96 correct codes → NONE | appended |
| 3 | Align single predictions into bags | `grouped_predictions.json` |
| 4 | Judge McMiner-S | `evaluation_metrics.json` + `judge_details_single.json` |
| 5 | Judge McMiner-M | `claude_evaluation_results.json` |

**Judged twice:** once by the same `gpt-oss:20b` (self-evaluation), then again
by **GPT-5** as an independent judge — steps 4–5 only, mining reused.

---

## Execution — 7h 56m total

| Arm | Bags | Single corrupted | Single correct | Judge S | Judge M | Total |
|---|---|---|---|---|---|---|
| baseline | 37 @ 11:51 | 209 @ 1:07:32 | 96 @ 23:46 | 165 @ 21:46 | 37 @ 5:11 | **2h10m** |
| rag | 37 @ 10:18 | 209 @ 1:00:21 | 96 @ 18:06 | 165 @ 17:54 | 37 @ 3:49 | **1h50m** |
| ref | 37 @ 9:59 | 209 @ 58:07 | 96 @ 23:21 | 165 @ 21:39 | 37 @ 4:45 | **1h57m** |
| rag_ref | 37 @ 11:33 | 209 @ 1:03:03 | 96 @ 16:57 | 165 @ 20:08 | 37 @ 5:14 | **1h56m** |

The aided arms were **not** slower despite bigger prompts — prompt eval runs at
~600 tok/s vs ~30 tok/s generation, so **generated length dominates**.

---

## Pipeline health

| Arm | Mining predictions | Parse failures | Judge calls | Judge parse failures |
|---|---|---|---|---|
| baseline | 305 | 0 | 159 | 0 |
| rag | 305 | 0 | 126 | 3 |
| ref | 305 | 0 | 152 | 2 |
| rag_ref | 305 | 0 | 145 | 0 |
| **Total** | **1,220** | **0** | **~480** | **5** |

All 5 judge-parse failures were a **tag typo** (`</match_withnovel>`), not
truncation. Each was checked against the judge's intended verdict:
**all 5 scored exactly as intended — no reported number is affected.**

---

## Results — self-judged (gpt-oss judging itself)

| Arm | McMiner-S std | S novelty | McMiner-M std | M novelty |
|---|---|---|---|---|
| baseline | 59.46% | 72.97% | 67.57% (25/37) | 78.38% (29/37) |
| rag | 56.76% | 70.27% | 59.46% (22/37) | 64.86% (24/37) |
| ref | 56.76% | 70.27% | 64.86% (24/37) | 72.97% (27/37) |
| **rag_ref** | **64.86%** | **81.08%** | **70.27% (26/37)** | **81.08% (30/37)** |

- **standard** — prediction matches the injected ground truth
- **novelty-aware** — also credits a *different* misconception the code really exhibits

⚠️ The model graded its own homework. **Two slides on why that matters.**

---

## ⚠️ The ranking is not statistically supported

**n = 37 bags → one bag = 2.70 percentage points**

| Arm | McMiner-M std | Wilson 95% CI | Width |
|---|---|---|---|
| baseline | 67.57% | [51.5%, 80.4%] | 29 pp |
| rag | 59.46% | [43.5%, 73.7%] | 30 pp |
| ref | 64.86% | [48.8%, 78.2%] | 29 pp |
| rag_ref | 70.27% | [54.2%, 82.5%] | 28 pp |

**Every interval overlaps every other.**

- `rag` and `ref` deficits vs baseline = **−2.70 pp = one bag**
- `rag_ref` lead = **two bags**

→ **Sample size, not prompt design, is the binding constraint.**

---

## Fixing the bigger problem: an independent judge

Mining is the expensive step and was already done — so only the **two judging
steps** were re-run, with GPT-5 in place of the local model.

| | |
|---|---|
| Judge | `openai/gpt-5` via OpenRouter |
| Judge calls | **582** (480 McMiner-S + 102 McMiner-M) |
| Wall time | 2h 00m 47s |
| Errors / parse failures | **0 / 0** |
| Cost | **$4.66** (estimated $2.83–$24.36) |

These are the numbers comparable to the McMiner paper's main table.

---

## Results — GPT-5 judged (quote these)

| Arm | McMiner-S std | S novelty | McMiner-M std | M novelty |
|---|---|---|---|---|
| baseline | 48.65% | 75.68% | 64.86% | **86.49%** |
| rag | 54.05% | 70.27% | 59.46% | 64.86% |
| ref | 48.65% | **81.08%** | 56.76% | 78.38% |
| **rag_ref** | **56.76%** | **81.08%** | **72.97%** | 83.78% |

`rag_ref` leads again on three of four measures — **same ordering as the local
judge**, and subject to the same n=37 caveat.

---

## Self-judging was inflating the scores

**delta = GPT-5 − self-judged (percentage points)**

| Arm | S std | S novelty | M std | M novelty |
|---|---|---|---|---|
| baseline | **−10.8** | +2.7 | −2.7 | +8.1 |
| rag | −2.7 | 0.0 | 0.0 | 0.0 |
| ref | **−8.1** | **+10.8** | **−8.1** | +5.4 |
| rag_ref | **−8.1** | 0.0 | +2.7 | +2.7 |

**Mean inflation on McMiner-S standard accuracy: +7.4 pp** — negative in *all
four* arms. Consistent direction is what makes it bias, not noise.

But novelty scores went **up**. GPT-5 is **stricter** on "did you find the
injected one" and **more generous** on "you found a different real one".
The two effects partly cancel in the pooled table.

---

## Equal totals ≠ equal verdicts

The `rag` arm matched the local judge to the decimal on 3 of 4 metrics.
Was that agreement — or coincidence?

| Arm | M agree | κ | S agree | κ | local-only Y | GPT-5-only Y |
|---|---|---|---|---|---|---|
| baseline | 81.1% | 0.58 | 87.0% | 0.71 | 13 | 4 |
| rag | **100%** | **1.00** | 86.8% | **0.63** | 9 | 5 |
| ref | 86.5% | 0.72 | 88.9% | 0.75 | 12 | 2 |
| rag_ref | 86.5% | 0.67 | 88.0% | 0.72 | 11 | 3 |

- `rag` McMiner-M **is** genuine unanimity (κ=1.00, all 37 bags)
- `rag` McMiner-S is **86.8% with the lowest κ of any arm** — 9 local-only vs
  5 GPT-5-only matches. **Same totals, different cases.**
- **local-only Y > GPT-5-only Y in every arm** — the signature of self-favouring

κ 0.58–0.75: the judges genuinely disagree on **12–19%** of cases.

---

## What the run *does* support (1): novelty gain

A **paired** comparison on identical bags — far more trustworthy than any
between-arm difference.

| Arm | McMiner-S gain | McMiner-M gain |
|---|---|---|
| baseline | +13.51 pp | +10.81 pp |
| rag | +13.51 pp | +5.41 pp |
| ref | +13.51 pp | +8.11 pp |
| rag_ref | +16.22 pp | +10.81 pp |

Consistent in **direction and magnitude** across all four arms.

**≈ 1 in 7** predictions that miss the injected misconception still identify a
real one — the novelty-aware metric is measuring something genuine.

---

## What the run *does* support (2): retrieval → abstention

The clearest **monotonic** pattern in the whole run.

| Arm | True negatives<br>(clean code → NONE) | False negatives<br>(missed injection) |
|---|---|---|
| baseline | 57/96 = **59.4%** | 41/209 = **19.6%** |
| ref | 64/96 = 66.7% | 46/209 = 22.0% |
| rag | 75/96 = 78.1% | 71/209 = 34.0% |
| rag_ref | 82/96 = **85.4%** | 60/209 = **28.7%** |

**More context → more conservative in both directions.** Better at recognising
clean code, worse at spotting injected misconceptions.

⚠️ Invisible in the accuracy table — the two effects partly **cancel**.

---

## The model's real weakness

**Abstention calibration**, not prompt design.

- At baseline the model invents a misconception in **4 of every 10** clean submissions
- Even at best (`rag_ref`) it still does so in **~15%**
- It misses **20–34%** of injected misconceptions outright

**A prediction the judge never sees cannot be scored.**

This is where accuracy is actually lost — and it is a property of
`gpt-oss:20b` at low reasoning effort, not of the pipeline.

---

## Threats to validity

1. ~~**Self-evaluation**~~ — **addressed.** Re-judged by GPT-5; bias measured
   at **+7.4 pp** mean inflation.
2. **The judge prompt is a reconstruction** — the original was missing from the
   bundle; criteria come from the paper, wording does not.
3. **n = 37 bags** — still blocks every comparative claim.
4. **`reasoning_effort=low`** was forced by a token-budget failure, not chosen;
   its accuracy cost is unquantified.
5. **Only 19 unique correct programs** exist (96 rows = 19 programs). Bags cover
   all 19 and score **100%** in every arm.
6. **GPT-5 routing** — `openai/gpt-5` took the Chat Completions path, not the
   Responses API. Internally consistent; not interchangeable with OpenAI-direct.

---

## Recommended next steps

1. **Increase bag count** — the single change that would make the arm comparison
   meaningful (n=37 blocks everything)
2. ~~Swap in an independent judge~~ — **done.** $4.66, 2 hours
3. **Apply the tolerant-regex parser fix** — remove dependence on a coincidence
4. **Make correct-only bag ids unique** — 4 bags currently share one id
5. **Fix the GPT-5 routing test** so `openai/gpt-5` reaches the Responses API
6. **Re-test `medium` effort with raised `max_tokens`** — quantify what `low` costs
7. **Add a third judge** — two disagree on 12–19% of cases

---

<!-- _paginate: false -->

# Summary

**The pipeline works.** 1,220 mining predictions and 582 judge calls,
**zero parse failures**, on a consumer laptop.

**The ranking does not.** All four arms sit inside overlapping ~29 pp intervals
at n=37 — under *both* judges.

**Three real findings:**
- Self-judging inflated McMiner-S accuracy by **+7.4 pp**, in every arm
- The novelty-aware metric captures something genuine (+13.5 pp, consistent)
- Retrieval context makes the model measurably more **conservative**

Full detail: `REPORT.md` · `REPORT_correct_only_bags.md`
