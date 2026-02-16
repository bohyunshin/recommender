# CLAUDE.md

## Project Overview

Recommender system implementations with a unified training pipeline. Supports various ML algorithms from matrix factorization to deep learning-based recommendation.

## Common Commands

```bash
# Install dependencies
uv sync

# Run linting
make lint

# Run tests
make test

# Run specific test
uv run pytest tests/pipeline/test_train.py

# Download dataset
uv run python scripts/download/movielens.py --package ml-1m

# Train torch-based model
uv run python recommender/train.py --dataset movielens --model svd --loss mse --epochs 30 --num_factors 16 --train_ratio 0.8 --random_state 42

# Train CSR-based model
uv run python recommender/train_csr.py --dataset movielens --model als --loss als --epochs 30 --num_factors 16 --train_ratio 0.8 --random_state 42
```

## Architecture

- **Training pipeline**: Two entry points based on model type
  - `recommender/train.py` — Torch-based models (SVD, SVD_BIAS, GMF, MLP, TWO_TOWER)
  - `recommender/train_csr.py` — CSR-based models (ALS, USER_BASED)
- **Model discovery**: Dynamic import via `MODEL_PATH` dict in `recommender/libs/constant/model/module_path.py`
- **Base class**: `RecommenderBase` (ABC) in `recommender/model/recommender_base.py` with abstract `predict` method
- **Key modules**:
  - `recommender/model/` — Model implementations (mf/, deep_learning/, neighborhood/)
  - `recommender/load_data/` — Dataset loaders
  - `recommender/preprocess/` — Data preprocessing
  - `recommender/prepare_model_data/` — Model data preparation (torch, csr)
  - `recommender/libs/` — Utilities, constants, validation, plotting, sampling
  - `recommender/loss/` — Custom loss functions
- **Constants**: Enum-based in `recommender/libs/constant/` (ModelName, LossName, Field, etc.)
- **Tests**: `tests/pipeline/` (integration), `tests/module/` (unit)

## Code Style

- Python 3.11, Ruff (line-length 88, rules E4/E7/E9/F), pre-commit hooks
- Config: `ruff.toml`, `.pre-commit-config.yaml`

## Skills

| Skill | Description |
|-------|-------------|
| `manage-skills` | Analyzes session changes, detects missing verification skills, creates/updates skills |
| `verify-implementation` | Runs all verify skills sequentially and generates integrated report |
| `verify-test-coverage` | Verifies models/pipelines have corresponding unit and integration tests |
| `verify-code-convention` | Validates PEP 8 naming, type hints, ruff compliance, import ordering |
| `verify-model-registration` | Ensures new models are registered in constants, have module paths, and implement abstract methods |
| `verify-code-simplifier` | Checks if code could be simplified — redundant booleans, manual loops, verbose patterns, dict lookup replacements |
