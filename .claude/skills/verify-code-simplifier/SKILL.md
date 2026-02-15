---
name: verify-code-simplifier
description: Checks if code could be simplified — redundant booleans, manual loops, verbose patterns, if-else chains replaceable by dict lookups. Use after adding or modifying Python code.
disable-model-invocation: false
argument-hint: "[optional: specific file or directory to check]"
---

# Code Simplification Verification

## Purpose

Detects code patterns that could be simplified for readability and maintainability:

1. **Redundant boolean comparisons** — `if x is True:`, `if x == False:` instead of `if x:` / `if not x:`
2. **Redundant length checks** — `if len(x) == 0:` instead of `if not x:`
3. **If-else chains replaceable by dict lookups** — Long `if/elif/else` chains that map values to functions or objects
4. **Manual loop accumulations** — Empty list + for-loop + append instead of list/dict comprehensions
5. **Verbose dict patterns** — `if d.get(k) is None: d[k] = []; d[k].append(v)` instead of `d.setdefault(k, []).append(v)`
6. **Unnecessary ternary verbosity** — `True if condition else False` instead of just `condition`
7. **Verbose single-use variables** — Variable assigned and used only once on the very next line

## When to Run

- After adding new Python files
- After modifying existing functions or methods
- Before creating a Pull Request
- When reviewing code for readability improvements
- After refactoring sessions

## Related Files

| File | Purpose |
|------|---------|
| `recommender/train.py` | Torch training entry point (thin wrapper) |
| `recommender/train_csr.py` | CSR training entry point (thin wrapper) |
| `recommender/pipeline/base.py` | Base pipeline class — shared training logic |
| `recommender/pipeline/torch.py` | Torch pipeline — training loop, negative sampling, model setup |
| `recommender/pipeline/csr.py` | CSR pipeline — ALS/user_based training loop, model setup |
| `recommender/model/recommender_base.py` | Base model class — contains if-else chains and manual loop accumulations |
| `recommender/model/neighborhood/user_based.py` | User-based CF — contains verbose dict patterns, manual loops, redundant length checks |
| `recommender/libs/sampling/negative_sampling.py` | Negative sampling — contains if-else chains and manual loop accumulations |
| `recommender/libs/utils/csr.py` | CSR utilities — contains verbose single-use variable patterns |
| `recommender/load_data/movielens_1m.py` | Data loader — contains redundant boolean comparisons |
| `recommender/load_data/movielens_10m.py` | Data loader — contains redundant boolean comparisons |
| `recommender/load_data/pinterest.py` | Data loader — contains redundant boolean comparisons and manual loop accumulations |
| `scripts/download/movielens.py` | Download script — contains redundant boolean comparisons |

## Workflow

### Step 1: Identify Changed Files

**Tool:** Bash

Determine which Python files to check. If an argument is provided, use that path. Otherwise, check files changed in the current session:

```bash
git diff HEAD --name-only -- '*.py'
git diff main...HEAD --name-only -- '*.py' 2>/dev/null
```

Merge into a deduplicated list. If no changed files, check all files under `recommender/` and `tests/`.

### Step 2: Check Redundant Boolean Comparisons

**Tool:** Grep

Search for explicit `True`/`False` comparisons:

```
pattern: "is True[:\s]|is False[:\s]|== True[:\s]|== False[:\s]"
path: recommender/
glob: "*.py"
```

Also check in `tests/` and `scripts/`:

```
pattern: "is True[:\s]|is False[:\s]|== True[:\s]|== False[:\s]"
path: tests/
glob: "*.py"
```

```
pattern: "is True[:\s]|is False[:\s]|== True[:\s]|== False[:\s]"
path: scripts/
glob: "*.py"
```

**PASS:** No explicit boolean comparisons found.

**FAIL:** Lines found with `is True`, `is False`, `== True`, or `== False`.

**Fix:** Replace `if x is True:` with `if x:` and `if x is False:` with `if not x:`.

### Step 3: Check Redundant Length Checks

**Tool:** Grep

Search for `len()` comparisons that check emptiness:

```
pattern: "if len\(.+\) == 0|if len\(.+\) > 0|if len\(.+\) != 0|if len\(.+\) >= 1"
path: recommender/
glob: "*.py"
```

**PASS:** No redundant length checks found.

**FAIL:** Lines using `len()` to check for empty/non-empty collections.

**Fix:** Replace `if len(x) == 0:` with `if not x:` and `if len(x) > 0:` with `if x:`.

### Step 4: Check Unnecessary Ternary Verbosity

**Tool:** Grep

Search for `True if ... else False` or `False if ... else True` patterns:

```
pattern: "= True if .+ else False|= False if .+ else True"
path: recommender/
glob: "*.py"
```

**PASS:** No verbose ternary expressions found.

**FAIL:** Lines with `True if condition else False`.

**Fix:** Replace `x = True if condition else False` with `x = condition`. Replace `x = False if condition else True` with `x = not condition`.

### Step 5: Check If-Else Chains Replaceable by Dict Lookups

**Tool:** Read

Read each changed file and look for if-elif chains with 3+ branches that:
- Compare the same variable against different literal values
- Assign to the same target or call similar functions

Count the number of `elif` statements per if-block. Flag blocks with 3 or more `elif` branches that follow a value-to-action mapping pattern.

**PASS:** No long if-elif chains that are simple value mappings.

**FAIL:** If-elif chains with 3+ branches mapping values to actions.

**Fix:** Extract the mapping into a dictionary and use `dict.get()` or `dict[key]()` for dispatch.

### Step 6: Check Manual Loop Accumulations

**Tool:** Grep

Search for the pattern of empty list initialization followed by loop append:

```
pattern: "= \[\]\s*$"
path: recommender/
glob: "*.py"
output_mode: content
```

For each match, read surrounding lines to check if it is followed by a for-loop that only appends to that list. Flag cases where a list comprehension would be a direct replacement (single append in the loop body, no complex side effects).

**PASS:** No manual loop accumulations that could be list comprehensions.

**FAIL:** Empty list + for-loop + append patterns found.

**Fix:** Replace with list comprehension: `result = [transform(x) for x in iterable if condition]`.

### Step 7: Check Verbose Dict Patterns

**Tool:** Grep

Search for `dict.get(key) is None` followed by initialization:

```
pattern: "\.get\(.+\) is None"
path: recommender/
glob: "*.py"
output_mode: content
```

Read surrounding lines to check if it follows the pattern:
```python
if d.get(k) is None:
    d[k] = []
d[k].append(v)
```

**PASS:** No verbose dict initialization patterns found.

**FAIL:** Dict get-check-initialize patterns found.

**Fix:** Use `d.setdefault(k, []).append(v)` or `collections.defaultdict(list)`.

## Output Format

```markdown
| # | Check | Status | Details |
|---|-------|--------|---------|
| 1 | Redundant boolean comparisons | PASS/FAIL | N instances found |
| 2 | Redundant length checks | PASS/FAIL | N instances found |
| 3 | Unnecessary ternary verbosity | PASS/FAIL | N instances found |
| 4 | If-else chains (dict lookup) | PASS/FAIL | N chains with 3+ branches |
| 5 | Manual loop accumulations | PASS/FAIL | N replaceable patterns |
| 6 | Verbose dict patterns | PASS/FAIL | N instances found |
```

## Exceptions

1. **Boolean comparisons in type-sensitive contexts** — `if x is True:` is valid when `x` could be a truthy non-boolean value (e.g., `1`, `"yes"`) and you strictly need `True`. This is rare but legitimate.
2. **Length checks with specific counts** — `if len(x) == 1:` or `if len(x) > 3:` are not redundant; only `== 0`, `> 0`, `!= 0`, `>= 1` are flagged.
3. **Loop accumulations with side effects** — Loops that do more than just append (e.g., logging, mutation, conditional logic with multiple branches) should not be converted to comprehensions.
4. **If-elif chains with complex logic** — Chains where branches contain different logic beyond simple assignment/call are not suitable for dict lookup replacement.
5. **Existing code not changed in session** — Only check files changed in the current session; do not report violations in untouched files.
6. **Test files** — Test files may use verbose patterns for clarity; treat these as lower-priority suggestions rather than strict violations.
