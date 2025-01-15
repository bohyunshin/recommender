import argparse
import os

from recommender.train import main

def run_model(
        model: str,
        dataset: str,
        implicit: bool
):
    args = argparse.ArgumentParser()
    args.dataset = dataset
    args.model = model
    args.implicit = implicit
    args.num_neg = 1
    args.batch_size = 32
    args.regularization = 1e-4
    args.lr = 1e-2
    args.epochs = 1
    args.num_factors = 16
    args.train_ratio = 0.8
    args.random_state = 42
    args.patience = 5
    args.result_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"../result/{model}")
    args.num_sim_user_top_N = 45
    args.test = True

    main(args)


def test_svd():
    run_model("svd", "movielens", False)

def test_svd_bias():
    run_model("svd_bias", "movielens", False)

def test_bpr():
    run_model("bpr", "movielens", True)

def test_gmf():
    run_model("gmf", "movielens", True)

def test_mlp():
    run_model("mlp", "movielens", True)

def test_two_tower():
    run_model("two_tower", "movielens", True)