---
name: verify-code-convention
description: Verifies PEP 8 naming, type hints, ruff compliance, import ordering, and project-specific conventions. Use after adding or modifying Python code.
disable-model-invocation: true
argument-hint: "[optional: specific file or directory to check]"
---

# Code Convention Verification

## Purpose

Ensures all code follows the project's coding standards:

1. **Ruff compliance** — Code must pass `ruff check` and `ruff format --check` with the project's configuration (line-length 88, Python 3.11, rules E4/E7/E9/F)
2. **PEP 8 naming** — `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants, `_leading_underscore` for private methods
3. **Type hints** — All function signatures must have type hints (parameters and return types)
4. **Import ordering** — Imports must follow isort/ruff `I` rules (stdlib, third-party, local)
5. **String formatting** — Double quotes for strings (Black-compatible)

## When to Run

- After adding new Python files
- After modifying existing functions or classes
- After adding new imports
- Before creating a Pull Request
- When pre-commit hooks report issues

## Related Files

| File | Purpose |
|------|---------|
| `ruff.toml` | Ruff configuration: line-length 88, target-version py311, lint rules E4/E7/E9/F |
| `.pre-commit-config.yaml` | Pre-commit hooks: ruff-format, ruff --fix, trailing-whitespace, end-of-file-fixer, codespell |
| `Makefile` | `make lint` command for running ruff format + check |
| `recommender/model/recommender_base.py` | Reference for type hint patterns and naming conventions |
| `recommender/libs/constant/model/name.py` | Reference for class naming (`ModelName`, `IMPLEMENTED_MODELS`) |

## Workflow

### Step 1: Run Ruff Format Check

**Tool:** Bash

```bash
uv run ruff format --check recommender/ tests/
```

**PASS:** No files need reformatting.

**FAIL:** Files listed as needing reformatting.

**Fix:** Run `uv run ruff format recommender/ tests/` or `make lint`.

### Step 2: Run Ruff Lint Check

**Tool:** Bash

```bash
uv run ruff check recommender/ tests/
```

**PASS:** "All checks passed!" output.

**FAIL:** Lint violations reported.

**Fix:** Run `uv run ruff check --fix recommender/ tests/` for auto-fixable issues. Manually fix remaining issues.

### Step 3: Check PEP 8 Naming Conventions

**Tool:** Grep

Check for non-PEP 8 class names (should be PascalCase):

```bash
grep -rn "^class [a-z]" recommender/ tests/ --include="*.py"
```

Check for camelCase function/method names (should be snake_case):

```bash
grep -rn "def [a-z]*[A-Z]" recommender/ tests/ --include="*.py"
```

Check for non-UPPER_SNAKE_CASE module-level constants (variables at module level that are all assigned and look like constants):

Read changed files and visually inspect module-level variable assignments to ensure constants use `UPPER_SNAKE_CASE`.

**PASS:** All names follow PEP 8 conventions.

**FAIL:** Names found that violate conventions.

**Fix:** Rename to follow PEP 8: `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants.

### Step 4: Check Type Hints on Function Signatures

**Tool:** Grep

Find functions missing return type hints:

```bash
grep -rn "def " recommender/ --include="*.py" | grep -v "->.*:" | grep -v "__init__" | grep -v "def _repr"
```

Find functions missing parameter type hints (check non-self, non-cls parameters):

```bash
grep -rn "def .*([^)]*[a-zA-Z_][a-zA-Z0-9_]*[^:)]*)" recommender/ --include="*.py"
```

**PASS:** All public functions and methods have type hints for parameters and return types.

**FAIL:** Functions found without type hints.

**Fix:** Add type hints. Use `typing` module types (`Dict`, `List`, `Optional`, `Tuple`, `Union`) or built-in generics for Python 3.11+.

### Step 5: Check Import Ordering

**Tool:** Bash

```bash
uv run ruff check --select I recommender/ tests/
```

**PASS:** No import ordering violations.

**FAIL:** Import ordering violations found.

**Fix:** Run `uv run ruff check --select I --fix recommender/ tests/`.

### Step 6: Check Pre-commit Compliance

**Tool:** Bash

Check for trailing whitespace and missing end-of-file newlines in changed files:

```bash
git diff --name-only HEAD | xargs -I {} sh -c 'grep -Pn "\s+$" {} 2>/dev/null && echo "TRAILING WHITESPACE: {}"'
```

**PASS:** No trailing whitespace or EOF issues.

**FAIL:** Files with trailing whitespace or missing EOF newline.

**Fix:** Run `pre-commit run --all-files` or configure editor to auto-trim whitespace.

## Output Format

```markdown
| # | Check | Status | Details |
|---|-------|--------|---------|
| 1 | Ruff format | PASS/FAIL | N files need reformatting |
| 2 | Ruff lint | PASS/FAIL | N violations found |
| 3 | PEP 8 naming | PASS/FAIL | List of violations |
| 4 | Type hints | PASS/FAIL | N functions missing hints |
| 5 | Import ordering | PASS/FAIL | N violations |
| 6 | Pre-commit compliance | PASS/FAIL | Issues found |
```

## Exceptions

1. **`__init__` methods** — `__init__` does not require a return type hint (it is always `None`)
2. **Test files** — Test functions may omit parameter type hints for fixtures (pytest infers them), but should still have return type hints
3. **Dunder methods** — `__repr__`, `__str__`, `__len__` etc. do not require type hints
4. **Existing code** — Only check files that were changed in the current session; do not report violations in untouched files
5. **Ruff-ignored rules** — Rules not in the `select` list (`E4`, `E7`, `E9`, `F`) are intentionally not enforced per `ruff.toml` config
