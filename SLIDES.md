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

**All four prompt arms · 7h 56m · zero mining parse failures**

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

⚠️ **Limitation:** only ~20 of the 96 correct codes entered bags, so McMiner-M's
correct-bag accuracy rests on a 4-bag subsample.

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

**Judge = the same `gpt-oss:20b`** → self-evaluation (see caveats)

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

## Results

| Arm | McMiner-S std | S novelty | McMiner-M std | M novelty |
|---|---|---|---|---|
| baseline | 59.46% | 72.97% | 67.57% (25/37) | 78.38% (29/37) |
| rag | 56.76% | 70.27% | 59.46% (22/37) | 64.86% (24/37) |
| ref | 56.76% | 70.27% | 64.86% (24/37) | 72.97% (27/37) |
| **rag_ref** | **64.86%** | **81.08%** | **70.27% (26/37)** | **81.08% (30/37)** |

- **standard** — prediction matches the injected ground truth
- **novelty-aware** — also credits a *different* misconception the code really exhibits

`rag_ref` ranks first on all four measures. **But read the next slide.**

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

1. **Self-evaluation** — the model judged its own output. **Not comparable** to
   the paper's GPT-5-judged table.
2. **The judge prompt is a reconstruction** — the original was missing from the
   bundle; criteria come from the paper, wording does not.
3. **n = 37 bags** — blocks every comparative claim.
4. **`reasoning_effort=low`** was forced by a token-budget failure, not chosen;
   its accuracy cost is unquantified.
5. **Correct-only bags undersampled** — 4 bags, ~20 of 96 correct codes.
6. **Known parser defect** (tag typo) — harmless here, but only by coincidence.

---

## Recommended next steps

1. **Apply the tolerant-regex parser fix** — remove dependence on the coincidence
2. **Increase bag count** — the single change that would make the arm comparison
   meaningful
3. **Fix correct-only bag coverage** — get all 96 correct codes into bags
4. **Swap in an independent judge** (GPT-5 via OpenRouter; mining stays local)
   → removes self-evaluation bias, gives paper-comparable numbers
5. **Re-test `medium` effort with raised `max_tokens`** — quantify what `low` costs

---

<!-- _paginate: false -->

# Summary

**The pipeline works.** 1,220 mining predictions, **zero parse failures**,
7h 56m on a consumer laptop.

**The ranking does not.** All four arms sit inside overlapping ~29 pp confidence
intervals at n=37.

**Two real findings:**
- The novelty-aware metric captures something genuine (+13.5 pp, consistent)
- Retrieval context makes the model measurably more **conservative**

Full detail: `REPORT.md`
