import argparse
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),"../recommender"))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),"../"))

from train import main

def run_model(model, dataset, implicit):
    args = argparse.ArgumentParser()
    args.dataset = dataset
    args.model = model
    args.implicit = implicit
    args.batch_size = 32
    args.regularization = 1e-4
    args.lr = 1e-2
    args.epochs = 1
    args.num_factors = 16
    args.train_ratio = 0.8
    args.random_state = 42
    args.patience = 5
    args.log_path = f"{model}.log"
    args.model_path = f"{model}.pkl"
    args.num_sim_user_top_N = 45
    args.movielens_data_type = "ml-1m"

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