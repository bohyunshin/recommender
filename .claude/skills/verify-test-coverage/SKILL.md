---
name: verify-test-coverage
description: Verifies that new models and pipeline components have corresponding unit tests and integration tests. Use after adding new models or modifying training pipeline.
disable-model-invocation: true
argument-hint: "[optional: specific model or module name]"
---

# Test Coverage Verification

## Purpose

Ensures all production code in the ML pipeline has adequate test coverage:

1. **Model test coverage** — Every model in `IMPLEMENTED_MODELS` must have an integration test in `tests/pipeline/test_train.py` or `tests/pipeline/test_train_csr.py`
2. **Module test coverage** — New modules under `recommender/` must have corresponding test files under `tests/`
3. **Test naming convention** — Test files must follow `test_{module_name}.py`, test classes `Test{ComponentName}`, test methods `test_{behavior}`
4. **Fixture completeness** — New models requiring custom config must have fixtures in `tests/conftest.py`

## When to Run

- After adding a new model to `ModelName` enum or `IMPLEMENTED_MODELS`
- After adding new modules under `recommender/` (model, preprocess, load_data, libs, etc.)
- After modifying `RecommenderBase` abstract methods
- Before creating a Pull Request

## Related Files

| File | Purpose |
|------|---------|
| `recommender/libs/constant/model/name.py` | `ModelName` enum and `IMPLEMENTED_MODELS` — source of truth for registered models |
| `recommender/model/recommender_base.py` | Abstract base class defining required methods |
| `tests/pipeline/test_train.py` | Integration tests for torch-based models |
| `tests/pipeline/test_train_csr.py` | Integration tests for CSR-based models (ALS, user_based) |
| `tests/conftest.py` | Shared fixtures (`setup_config`) |
| `tests/module/libs/test_csr.py` | Unit tests for CSR utilities |
| `tests/module/libs/test_evaluation_metric.py` | Unit tests for evaluation metrics |
| `tests/module/libs/test_plot.py` | Unit tests for plotting |
| `tests/module/load_data/test_load_data.py` | Unit tests for data loading |
| `tests/module/preprocess/test_preprocess.py` | Unit tests for preprocessing |

## Workflow

### Step 1: Check All Registered Models Have Integration Tests

**Tool:** Grep, Read

Extract all model names from `IMPLEMENTED_MODELS`:

```bash
grep -n "ModelName\." recommender/libs/constant/model/name.py
```

Then verify each model name appears in integration test files as a parametrized test case or dedicated test:

```bash
grep -n '"svd"\|"svd_bias"\|"gmf"\|"mlp"\|"two_tower"' tests/pipeline/test_train.py
grep -n '"als"\|"user_based"' tests/pipeline/test_train_csr.py
```

**PASS:** Every model in `IMPLEMENTED_MODELS` has a corresponding integration test (either parametrized in `test_train.py`/`test_train_csr.py` or in a dedicated test file).

**FAIL:** A model is registered but has no integration test.

**Fix:** Add a parametrized test case to `tests/pipeline/test_train.py` (torch models) or `tests/pipeline/test_train_csr.py` (CSR models).

### Step 2: Check New Source Modules Have Test Files

**Tool:** Glob, Grep

List all Python modules under `recommender/` (excluding `__init__.py`, `__pycache__`):

```bash
find recommender -name "*.py" -not -name "__init__.py" -not -path "*__pycache__*" | sort
```

List all test files:

```bash
find tests -name "test_*.py" | sort
```

For each source module directory (`model/`, `preprocess/`, `load_data/`, `libs/`, etc.), verify at least one corresponding test file exists under `tests/`.

**PASS:** Every source module directory with production logic has at least one test file.

**FAIL:** A source module directory has no corresponding test file under `tests/`.

**Fix:** Create `tests/module/{module}/test_{component}.py` with appropriate tests.

### Step 3: Check Test File Naming Convention

**Tool:** Glob

```bash
find tests -name "*.py" -not -name "test_*" -not -name "conftest.py" -not -name "__init__.py" -not -path "*__pycache__*"
```

**PASS:** All test files follow `test_{name}.py` naming (except `conftest.py`, `__init__.py`).

**FAIL:** Test files found that don't follow the naming convention.

**Fix:** Rename to `test_{descriptive_name}.py`.

### Step 4: Check Test Class and Method Naming

**Tool:** Grep

```bash
grep -rn "^class " tests/ --include="*.py" | grep -v "^.*:class Test"
```

```bash
grep -rn "def test" tests/ --include="*.py" | grep -v "def test_"
```

**PASS:** All test classes use `Test{Name}` and all test methods use `test_{description}`.

**FAIL:** Classes or methods not following naming convention.

**Fix:** Rename to `Test{ComponentName}` / `test_{behavior_description}`.

### Step 5: Check New Models Have Config Fixtures

**Tool:** Read

Read `tests/conftest.py` and check that the `setup_config` fixture can provide config for all models. Verify the fixture parameters cover all model/loss/dataset combinations used in integration tests.

**PASS:** Every registered model has a config fixture available.

**FAIL:** A model is registered but has no config fixture coverage.

**Fix:** Add a fixture parameter set to `tests/conftest.py` following the existing `setup_config` pattern.

## Output Format

```markdown
| # | Check | Status | Details |
|---|-------|--------|---------|
| 1 | Registered model integration tests | PASS/FAIL | List of untested models |
| 2 | Module test coverage | PASS/FAIL | List of uncovered modules |
| 3 | Test file naming | PASS/FAIL | List of misnamed files |
| 4 | Test class/method naming | PASS/FAIL | List of violations |
| 5 | Config fixtures | PASS/FAIL | List of missing fixtures |
```

## Exceptions

1. **Utility modules** — `recommender/libs/utils/` helper files (e.g., `logger.py`, `parse_args.py`, `utils.py`) do not require dedicated test files unless they contain complex logic
2. **`__init__.py` files** — These are re-export files and do not need tests
3. **Script files** — Files in `scripts/` are one-off download scripts and are exempt
4. **Abstract base classes** — `recommender_base.py`, `torch_model_base.py`, `fit_model_base.py` are tested indirectly through concrete model implementations
5. **Constant files** — Files under `recommender/libs/constant/` are simple value definitions and do not need dedicated tests
