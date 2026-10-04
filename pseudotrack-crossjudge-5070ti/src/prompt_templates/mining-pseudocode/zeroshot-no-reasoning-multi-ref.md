You are an expert in analyzing student pseudocode to identify programming misconceptions.

## Context: the pseudocode notation

The code samples below are written in the *Notasi Algoritmik* pseudocode taught in "pseudocode dan
golang dasar" (Fakultas Informatika, Telkom University): `program/kamus/algoritma/endprogram`,
assignment `<-`, I/O `input(...)`/`output(...)`, `if ... then ... else ... endif`,
`for i <- a to b do ... endfor`, `while ... do ... endwhile`, `repeat ... until`, integer ops `mod`
and `div`, and scalar types only. Judge the students against **this notation**, not Python (here the
correct assignment operator is `<-` and the correct equality operator is `==`).

## Key Terminology

A **programming misconception** is a false belief a student holds about some programming-language
construct (its syntax or semantics). It must be **concrete and specific** to a construct, not a
vague misunderstanding, and not a misreading of the problem. For example, "The student believes the
first character of a string is at index 1" is a valid, specific misconception. Misconceptions often
cause bugs but sometimes do not (e.g. "The student believes every identifier must be one letter").

## Important Note

**Misconceptions do not always result in buggy code!** Some lead only to stylistic differences or
redundant patterns rather than errors.

## Your Task

You will be given a set of (problem description, code) pairs, where the code in each pair attempts to
solve the corresponding problem. Output the description of the single programming misconception that
is exhibited by the code samples in the input set. The input set will contain either:

* Code samples that all exhibit the same single misconception (though not every sample may show it), or
* Code samples that contain no misconceptions at all

If at least one code sample exhibits a misconception, identify and describe that one shared
misconception. If none do, output NONE.

## Reference Solutions

Each student code below may be followed by a **Reference Solution**: a correct solution for that
code's problem, retrieved by the repair system. It is a **contrast aid, not an answer key** — use
the differences between a student's code and its reference to search for the single *shared*
misconception, but base your final answer on the student codes themselves. Not every deviation from
a reference is a misconception (problems have many correct solutions), and where no reference is
available, analyze the code on its own.

## Input

**Problem Descriptions:** {problem_description}

**Student Code(s):**
{corrupted_codes}

## Output Format

Structure your response using these XML tags:

```xml
<misconception>
{misconception_block}
</misconception>
```

If you find a misconception, output it as follows:
```xml
<misconception>
<description>[Clear description of the ONE shared misconception, starting with "The student believes"]</description>
<explanation>[Explain how the given code exhibits the misconception]</explanation>
</misconception>
```

If you cannot find any misconception, output NONE as follows:
```xml
<misconception>
NONE
</misconception>
```
