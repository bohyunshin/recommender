import argparse
import sys
import os
# sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),"../recommender"))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),"../"))

from train_csr import main

def run_csr_model(model, dataset):
    args = argparse.ArgumentParser()
    args.dataset = dataset
    args.model = model
    args.implicit = True
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


def test_als():
    run_csr_model("als", "movielens")
