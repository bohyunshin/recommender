import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),"../.."))

import argparse
import importlib

from model.mf.implicit_mf import AlternatingLeastSquares
from tools.evaluation import ranking_metrics_at_k

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--num_factors", type=int, default=128)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--movielens_data_type", type=str, default="ml-latest-small")
    return parser.parse_args()


def main(args):
    preprocessor_module = importlib.import_module(f"recommender.data.{args.dataset}.preprocess_csr").Preprocessor
    preprocessor = preprocessor_module(movielens_data_type="ml-1m", test_size=0.2)
    csr_train, csr_val = preprocessor.preprocess()

    params = {
        'factors': args.num_factors,
        'iterations': args.epochs,
        'regularization': 0.01,
        'random_state': 42
    }
    als = AlternatingLeastSquares(**params)
    als.fit(user_items=csr_train, val_user_items=csr_val)

    metric = ranking_metrics_at_k(als, csr_train, csr_val, K=10)
    print(f"NDCG@10: {metric['ndcg']}")
    print(f"mAP@10: {metric['map']}")


if __name__ == "__main__":
    args = parse_args()
    main(args)