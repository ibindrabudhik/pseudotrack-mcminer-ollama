# Judge: does the mined misconception match the ground truth?

You are evaluating whether one or more code samples exhibit a specific
programming misconception, and whether a mining system correctly identified it.

The code below was produced by injecting a known misconception into a correct
solution. You are given that injected (ground truth) misconception and the
misconception(s) the system predicted.

## The Ground Truth Misconception
<misconception>
**Description:** {gt_description}
</misconception>

## The Code to Analyze
<code>
{code}
</code>

If several code samples appear above (marked `# Code 1`, `# Code 2`, ...), treat
them as one bag and apply the Multiple Instance Learning principle: a property
holds for the bag if **at least one** sample exhibits it.

## The System's Prediction
<predicted>
{predicted}
</predicted>

## Your Task

Make two separate judgements.

### Key Understanding
**A misconception does NOT necessarily induce bugs or errors!** Code can be:
- **Syntactically correct** (no syntax errors)
- **Logically correct** (produces expected output)
- **Yet still exhibit a misconception** (shows the student holds a false belief)

Misconceptions can be:
- **Benign**: Leading to style issues, inefficiencies, or non-idiomatic patterns while still working correctly
- **Harmful**: Causing incorrect behavior, logical errors, or runtime errors

### Analysis Guidelines

1. **Understand the misconception deeply**
   - What incorrect belief does the student have?
   - What coding patterns would reveal this belief?
   - Would this belief lead to different code structure/approach?

2. **Analyze the code systematically**
   - Look for patterns that match the misconception
   - Check if the code structure reflects the incorrect belief
   - Consider if the code shows unnecessary complexity or unusual patterns due to the misconception

3. **Focus on the belief, not the outcome**
   - Does the code structure suggest the student holds this false belief?
   - Even if the code works, does it show the misconception pattern?
   - Would someone with correct understanding write it differently?

### Judgement 1 — `match`

Does any predicted misconception describe the **same underlying false belief**
as the ground truth?

Judge on meaning, not wording: the system writes free text and will not reuse
the ground truth phrasing. Answer **Y** when a prediction identifies the same
mistaken mental model, showing up in the same place in the code, even if worded
differently, more generally, or more specifically.

Answer **N** when every prediction describes a *different* false belief, points
at an unrelated part of the code, or is so vague it could describe almost any
error ("the logic is wrong", "an off-by-one somewhere"). Two descriptions of the
same failing line do **not** match if they attribute it to different beliefs.

If several misconceptions were predicted, `match` is **Y** if **any one of them**
matches the ground truth.

### Judgement 2 — `match_with_novel`

This is the novelty-aware judgement and is **never N when `match` is Y**.

Answer **Y** if either:
- `match` is Y; **or**
- the prediction misses the ground truth, but the code **really does exhibit**
  at least one predicted misconception. The injected misconception is not
  necessarily the only one present, and identifying a different real one is
  still a correct observation.

Apply the Analysis Guidelines above to the predicted misconception exactly as
you would to the ground truth one: answer Y if the code shows patterns
consistent with it, **even if the code works correctly**.

Answer **N** if the prediction misses the ground truth *and* the predicted
misconceptions are not actually exhibited by this code — the system invented a
problem that is not there. Verify the claim against the code before giving
novelty credit; plausible-sounding but absent misconceptions get **N**.

## Output Format

Respond with the evaluation block and nothing else. Keep the rationale to one or
two sentences.

<evaluation>
<match>Y or N</match>
<match_with_novel>Y or N</match_with_novel>
<confidence>high|medium|low</confidence>
<rationale>[Explain whether and how the code exhibits the misconception(s) and why you reached your conclusion.]</rationale>
</evaluation>
