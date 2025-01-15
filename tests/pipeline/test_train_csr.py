import argparse
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),"../"))

from recommender.train_csr import main

def run_csr_model(
        model: str,
        dataset: str
):
    args = argparse.ArgumentParser()
    args.dataset = dataset
    args.model = model
    args.implicit = True
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
    args.num_sim_user_top_N = 5

    main(args)


def test_als():
    run_csr_model("als", "movielens")

def test_user_based():
    run_csr_model("user_based", "movielens")