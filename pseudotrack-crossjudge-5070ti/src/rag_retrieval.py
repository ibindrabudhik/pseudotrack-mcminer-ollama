"""
RAG retrieval loader + prompt-context formatter for McMiner (see RAG_MCMINER_PLAN.md).

Loads the precomputed embedding retrieval and renders a "top-k most similar misconceptions"
block that gets injected into the mining prompt. Two sources:

  * submission CSV  (dataset/retrival_openai_embedding_large.csv) — keyed by
    (problem_id, misconception_id); one row per corrupted student submission.
  * correct CSV     (dataset/retrival_correct_codes.csv, optional) — keyed by problem_id;
    one row per correct reference pseudocode (for correct-only bags / NONE substitutions).

Design notes:
  - Similarity scores are parsed but NOT shown in the prompt (locked decision): only the ranked
    list of misconception descriptions is surfaced, so retrieval is advisory, not an answer key.
  - If no row is found for a code, format_context() returns the NO_CONTEXT placeholder line, so
    the {retrieved_context} slot is never left dangling.
  - The whole module is inert unless a caller constructs a RagIndex; drivers only do that when
    --rag-csv is passed, so non-RAG runs are byte-for-byte the baseline.
"""
import csv

TOP_K_MAX = 5
NO_CONTEXT = "No retrieved context available for this submission."


class RagIndex:
    def __init__(self, submission_map, correct_map, default_top_k=3):
        self.submission_map = submission_map   # (problem_id:int, misc_id:int) -> [Candidate]
        self.correct_map = correct_map         # problem_id:int -> [Candidate]
        self.default_top_k = default_top_k

    # ------------------------------------------------------------------ lookup
    def candidates_for(self, problem_id, misconception_id=None, is_correct=False):
        """Return the ordered candidate list for a code, or [] if not found."""
        try:
            pid = int(problem_id)
        except (TypeError, ValueError):
            return []
        if is_correct or misconception_id in (None, "", "None"):
            return self.correct_map.get(pid, [])
        try:
            mid = int(misconception_id)
        except (TypeError, ValueError):
            return []
        return self.submission_map.get((pid, mid), [])

    def format_context(self, problem_id, misconception_id=None, is_correct=False, top_k=None):
        """Render the retrieved-context block for injection into a prompt."""
        cands = self.candidates_for(problem_id, misconception_id, is_correct)
        return format_candidates(cands, top_k or self.default_top_k)


class Candidate(dict):
    """{'rank','code','description','kc_tags','similarity'} with attribute access."""
    __getattr__ = dict.get


def _parse_row(row):
    cands = []
    for i in range(1, TOP_K_MAX + 1):
        code = (row.get(f"top{i}_code") or "").strip()
        desc = (row.get(f"top{i}_description") or "").strip()
        if not code or not desc:
            continue
        sim = row.get(f"top{i}_similarity")
        try:
            sim = float(sim)
        except (TypeError, ValueError):
            sim = None
        cands.append(Candidate(rank=i, code=code, description=desc,
                               kc_tags=(row.get(f"top{i}_kc_tags") or "").strip(), similarity=sim))
    return cands


def load_index(submission_csv, correct_csv=None, default_top_k=3):
    """Build a RagIndex from the submission CSV (+ optional correct CSV)."""
    submission_map, correct_map = {}, {}
    n_sub_bad = 0
    with open(submission_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                key = (int(row["problem_id"]), int(row["misconception_id"]))
            except (KeyError, ValueError, TypeError):
                n_sub_bad += 1
                continue
            submission_map[key] = _parse_row(row)

    if correct_csv:
        with open(correct_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    correct_map[int(row["problem_id"])] = _parse_row(row)
                except (KeyError, ValueError, TypeError):
                    continue

    print(f"🔎 RAG: loaded {len(submission_map)} submission retrieval rows"
          + (f", {len(correct_map)} correct-code rows" if correct_csv else " (no correct CSV)")
          + f", top_k={default_top_k}")
    if n_sub_bad:
        print(f"   ⚠️  {n_sub_bad} submission rows skipped (bad problem/misc id)")
    _validate(submission_map)
    return RagIndex(submission_map, correct_map, default_top_k)


def _validate(submission_map):
    """Light sanity check on the new [0,1] cosine scale; prints one summary line."""
    off_scale = blank = 0
    for cands in submission_map.values():
        if not cands:
            blank += 1
        for c in cands:
            if c.similarity is not None and not (0.0 <= c.similarity <= 1.0):
                off_scale += 1
    if off_scale or blank:
        print(f"   validate: {blank} empty rows, {off_scale} similarities outside [0,1]")


def format_candidates(cands, top_k=3):
    """Render the {retrieved_context} block. Similarity is intentionally NOT shown."""
    cands = [c for c in cands if c.description][:top_k]
    if not cands:
        return NO_CONTEXT
    lines = [
        "The following known misconceptions were retrieved by semantic similarity to the student's",
        "code, ordered from most to least similar. They are a **shortlist to consider, not an answer",
        "key** — treat them as candidate hypotheses. Pick one if it truly fits, adapt one, or identify",
        "a different misconception entirely if the code warrants it.",
        "",
    ]
    for rank, c in enumerate(cands, start=1):
        tags = f"  (related constructs: {c.kc_tags})" if c.kc_tags else ""
        lines.append(f"{rank}. [{c.code}] {c.description}{tags}")
    return "\n".join(lines)
