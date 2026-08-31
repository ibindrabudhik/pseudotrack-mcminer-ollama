"""
APR-reference loader + prompt-context formatter for McMiner (see REFERENCE_MCMINER_PLAN.md).

Parallel to rag_retrieval.py, but instead of injecting the top-k most-similar *catalog
misconceptions*, it injects the **correct reference solution** that the APR (automated program
repair) stage retrieved to repair each buggy student submission. One reference per submission,
no ranking.

Source:
  * Submission_Code_with_reference_from_APR.csv — keyed by (problem_id, misconception_id), one row
    per corrupted student submission (the same 418-row key as the RAG submission CSV). The injected
    column defaults to `Reference_Code` (the retrieved correct reference); `Best_Reference` /
    `Repaired_Code` are selectable via ref_column for ablations.

Design notes (mirrors rag_retrieval.py contracts):
  - Correct-only bag codes carry misconception_id=None, so they never join and always render the
    NO_REFERENCE line — reference is injected only for actual buggy submissions.
  - If no row / empty reference is found, format_context() returns the NO_REFERENCE placeholder, so
    the {reference_code} slot is never left dangling.
  - The module is inert unless a caller constructs a RefIndex; drivers only do that when --ref-csv is
    passed, so non-reference runs are byte-for-byte the baseline.
"""
import csv

NO_REFERENCE = "No reference solution available for this submission."
DEFAULT_REF_COLUMN = "Reference_Code"


class RefIndex:
    def __init__(self, reference_map, ref_column=DEFAULT_REF_COLUMN):
        self.reference_map = reference_map      # (problem_id:int, misc_id:int) -> code:str
        self.ref_column = ref_column

    # ------------------------------------------------------------------ lookup
    def reference_for(self, problem_id, misconception_id=None):
        """Return the reference code string for a submission, or None if not found."""
        try:
            pid = int(problem_id)
            mid = int(misconception_id)
        except (TypeError, ValueError):
            return None
        return self.reference_map.get((pid, mid))

    def format_context(self, problem_id, misconception_id=None):
        """Render the reference block for injection into a prompt."""
        code = self.reference_for(problem_id, misconception_id)
        return format_reference(code)


def _clean(v):
    v = (v or "").strip()
    return v if v and v.upper() != "NONE" else ""


def load_reference_index(ref_csv, ref_column=DEFAULT_REF_COLUMN):
    """Build a RefIndex from the APR reference CSV."""
    reference_map = {}
    n_bad = 0
    n_empty = 0
    with open(ref_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if ref_column not in (reader.fieldnames or []):
            raise ValueError(f"Reference column '{ref_column}' not in {ref_csv}. "
                             f"Available: {reader.fieldnames}")
        for row in reader:
            try:
                key = (int(row["problem_id"]), int(row["misconception_id"]))
            except (KeyError, ValueError, TypeError):
                n_bad += 1
                continue
            code = _clean(row.get(ref_column))
            if not code:
                n_empty += 1
                continue
            reference_map[key] = code

    print(f"📎 REF: loaded {len(reference_map)} reference rows from column '{ref_column}'"
          f" ({n_empty} rows had empty {ref_column}"
          + (f", {n_bad} rows skipped for bad problem/misc id" if n_bad else "")
          + ")")
    return RefIndex(reference_map, ref_column)


def format_reference(code):
    """Render the {reference_code} block. Empty/None code -> NO_REFERENCE line."""
    code = _clean(code)
    if not code:
        return NO_REFERENCE
    return (
        "A correct reference solution for this problem, retrieved by the repair system, is shown\n"
        "below. Use it to contrast against the student's code and localize the false belief. It is a\n"
        "**reference for contrast, not an answer key** — the student's misconception is what you must\n"
        "name.\n\n"
        "```\n" + code + "\n```"
    )
