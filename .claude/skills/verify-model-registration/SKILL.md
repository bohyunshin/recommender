---
name: verify-model-registration
description: Verifies new models are registered in ModelName enum, IMPLEMENTED_MODELS, MODEL_PATH, and have proper train.py routing. Use after adding a new model.
disable-model-invocation: true
argument-hint: "[optional: specific model name to verify]"
---

# Model Registration Verification

## Purpose

Ensures new models are fully integrated into the training pipeline:

1. **ModelName enum** — Model must be added to `ModelName` enum in `recommender/libs/constant/model/name.py`
2. **IMPLEMENTED_MODELS list** — Model must be included in `IMPLEMENTED_MODELS` list
3. **MODEL_PATH mapping** — Model must have an entry in `MODEL_PATH` dict in `recommender/libs/constant/model/module_path.py`
4. **Train script routing** — Torch-based models must work via `recommender/train.py`, CSR-based models via `recommender/train_csr.py`
5. **Abstract method implementation** — Model class must implement the `predict` abstract method from `RecommenderBase`

## When to Run

- After adding a new model class in `recommender/model/`
- After modifying `ModelName` enum or `IMPLEMENTED_MODELS`
- After modifying `MODEL_PATH` dictionary
- After modifying `recommender/train.py` or `recommender/train_csr.py`
- Before creating a Pull Request that introduces a new model

## Related Files

| File | Purpose |
|------|---------|
| `recommender/libs/constant/model/name.py` | `ModelName` enum and `IMPLEMENTED_MODELS` list — source of truth for registered models |
| `recommender/libs/constant/model/module_path.py` | `MODEL_PATH` dict — maps model name to importable module path |
| `recommender/model/recommender_base.py` | `RecommenderBase` ABC with abstract method `predict` |
| `recommender/model/torch_model_base.py` | Base class for torch-based models |
| `recommender/model/fit_model_base.py` | Base class for CSR/fit-based models |
| `recommender/train.py` | Entry point for torch-based model training |
| `recommender/train_csr.py` | Entry point for CSR-based model training (ALS, user_based) |
| `recommender/model/mf/svd.py` | SVD model implementation |
| `recommender/model/mf/svd_bias.py` | SVD with bias model implementation |
| `recommender/model/mf/als.py` | ALS model implementation |
| `recommender/model/deep_learning/gmf.py` | GMF model implementation |
| `recommender/model/deep_learning/mlp.py` | MLP model implementation |
| `recommender/model/deep_learning/two_tower.py` | Two-tower model implementation |
| `recommender/model/neighborhood/user_based.py` | User-based CF model implementation |

## Workflow

### Step 1: Extract Registered Models from ModelName Enum

**Tool:** Read

Read `recommender/libs/constant/model/name.py` and extract all values from the `ModelName` enum.

Current registered models:
- `ALS` -> `"als"`
- `USER_BASED` -> `"user_based"`
- `SVD` -> `"svd"`
- `SVD_BIAS` -> `"svd_bias"`
- `GMF` -> `"gmf"`
- `MLP` -> `"mlp"`
- `TWO_TOWER` -> `"two_tower"`

Check for any new models added to the enum.

**PASS:** All enum values are also present in `IMPLEMENTED_MODELS`.

**FAIL:** A model is in `ModelName` but not in `IMPLEMENTED_MODELS`, or vice versa.

**Fix:** Add the missing model to `IMPLEMENTED_MODELS` list or `ModelName` enum.

### Step 2: Verify MODEL_PATH Exists for Each Model

**Tool:** Read

Read `recommender/libs/constant/model/module_path.py` and verify every model in `IMPLEMENTED_MODELS` has an entry in `MODEL_PATH`.

```bash
grep -c '"als"\|"user_based"\|"svd"\|"svd_bias"\|"gmf"\|"mlp"\|"two_tower"' recommender/libs/constant/model/module_path.py
```

**PASS:** Every registered model has a corresponding entry in `MODEL_PATH`.

**FAIL:** A registered model has no `MODEL_PATH` entry.

**Fix:** Add `"<model_name>": "recommender.model.<category>.<model_name>"` to the `MODEL_PATH` dict.

### Step 3: Verify Model Module Exists at Declared Path

**Tool:** Glob

For each model in `MODEL_PATH`, verify the Python module exists at the declared path:

```bash
# Convert dot path to file path: recommender.model.mf.svd -> recommender/model/mf/svd.py
ls recommender/model/mf/svd.py
ls recommender/model/mf/svd_bias.py
ls recommender/model/mf/als.py
ls recommender/model/deep_learning/gmf.py
ls recommender/model/deep_learning/mlp.py
ls recommender/model/deep_learning/two_tower.py
ls recommender/model/neighborhood/user_based.py
```

**PASS:** All module files exist at declared paths.

**FAIL:** A module path in `MODEL_PATH` points to a non-existent file.

**Fix:** Either create the missing module file or correct the path in `MODEL_PATH`.

### Step 4: Verify Model Implements Abstract Methods

**Tool:** Read, Grep

Read `recommender/model/recommender_base.py` to confirm the abstract method is `predict`.

For each model class, verify it implements the `predict` method:

```bash
grep -n "def predict" recommender/model/mf/svd.py
grep -n "def predict" recommender/model/mf/svd_bias.py
grep -n "def predict" recommender/model/mf/als.py
grep -n "def predict" recommender/model/deep_learning/gmf.py
grep -n "def predict" recommender/model/deep_learning/mlp.py
grep -n "def predict" recommender/model/deep_learning/two_tower.py
grep -n "def predict" recommender/model/neighborhood/user_based.py
```

**PASS:** All model classes implement `predict`.

**FAIL:** A model class is missing the `predict` method.

**Fix:** Implement the `predict` method in the model class following the signature from `RecommenderBase`.

### Step 5: Verify Model Has Exported `Model` Class

**Tool:** Grep

Each model module must export a class named `Model` (used by `train.py` and `train_csr.py` via `importlib`):

```bash
grep -n "^class Model" recommender/model/mf/svd.py
grep -n "^class Model" recommender/model/mf/svd_bias.py
grep -n "^class Model" recommender/model/mf/als.py
grep -n "^class Model" recommender/model/deep_learning/gmf.py
grep -n "^class Model" recommender/model/deep_learning/mlp.py
grep -n "^class Model" recommender/model/deep_learning/two_tower.py
grep -n "^class Model" recommender/model/neighborhood/user_based.py
```

**PASS:** Every model module has a `class Model` that can be imported.

**FAIL:** A model module does not export `class Model`.

**Fix:** Ensure the model class is named `Model` or aliased as `Model` at module level.

## Output Format

```markdown
| # | Check | Status | Details |
|---|-------|--------|---------|
| 1 | ModelName enum & IMPLEMENTED_MODELS | PASS/FAIL | Missing models: [list] |
| 2 | MODEL_PATH entries | PASS/FAIL | Missing entries: [list] |
| 3 | Module file existence | PASS/FAIL | Missing files: [list] |
| 4 | Abstract method (predict) | PASS/FAIL | Missing implementations: [list] |
| 5 | Model class export | PASS/FAIL | Missing Model class: [list] |
```

## Exceptions

1. **Intermediate base classes** — `recommender_base.py`, `torch_model_base.py`, `fit_model_base.py` are abstract and do not need to be in `ModelName` or `MODEL_PATH`
2. **CSR-based models** — ALS and user_based use `train_csr.py` instead of `train.py`; they still need `MODEL_PATH` entries but follow a different training flow
3. **Shared model patterns** — Multiple model variants may share a base class but each variant must have its own `ModelName` entry if it can be selected independently
