import argparse
import os

import pytest


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
    args.is_test = True
    return args
