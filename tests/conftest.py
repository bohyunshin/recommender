import argparse
import os

import numpy as np
import pandas as pd
import pytest

from recommender.libs.constant.data.name import Field


def generate_mock_interaction_data(implicit: bool) -> dict:
    """Generate mock interaction data matching LoadData.load() format."""
    rng = np.random.RandomState(42)
    rows = []
    for user_id in range(1, 11):
        n_interactions = rng.randint(8, 16)
        item_ids = rng.choice(range(1, 21), size=n_interactions, replace=False)
        for item_id in item_ids:
            rating = 1.0 if implicit else float(rng.randint(1, 6))
            rows.append((user_id, item_id, rating, 0))

    df = pd.DataFrame(
        rows, columns=[Field.USER_ID, Field.ITEM_ID, Field.INTERACTION, "timestamp"]
    )
    return {Field.INTERACTION: df}


@pytest.fixture(scope="function")
def mock_data_factory():
    return generate_mock_interaction_data


@pytest.fixture(scope="function")
def setup_config(request):
    dataset, model, loss, implicit, num_neg, neg_sample_strategy = request.param
    args = argparse.ArgumentParser()
    args.dataset = dataset
    args.model = model
    args.loss = loss
    args.device = "cpu"
    args.implicit = implicit
    args.num_neg = num_neg
    args.neg_sample_strategy = neg_sample_strategy
    args.batch_size = 32
    args.regularization = 1e-4
    args.lr = 1e-2
    args.epochs = 10
    args.num_factors = 16
    args.train_ratio = 0.8
    args.random_state = 42
    args.patience = 5
    args.result_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), f"../result/{model}"
    )
    args.num_sim_user_top_N = 45
    args.is_test = False
    return args


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "requires_data" in item.keywords:
            item.add_marker(pytest.mark.skip(reason="requires real dataset files"))
