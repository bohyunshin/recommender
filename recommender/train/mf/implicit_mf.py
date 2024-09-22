import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),"../.."))

import argparse
import importlib
import pickle
import time

from model.mf.implicit_mf import AlternatingLeastSquares
from tools.evaluation import ranking_metrics_at_k
from tools.logger import setup_logger

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--regularization", type=float, default=0.01)
    parser.add_argument("--num_factors", type=int, default=128)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--movielens_data_type", type=str, default="ml-latest-small")
    return parser.parse_args()


def main(args):
    logger = setup_logger(args.save_path.split(".")[0] + ".log")
    logger.info(f"selected dataset: {args.dataset}")
    logger.info(f"selected movielens data type: {args.movielens_data_type}")
    preprocessor_module = importlib.import_module(f"recommender.data.{args.dataset}.preprocess_csr").Preprocessor
    preprocessor = preprocessor_module(movielens_data_type=args.movielens_data_type,
                                       test_ratio=args.test_ratio,
                                       random_state=args.random_state)
    csr_train, csr_val = preprocessor.preprocess()

    params = {
        'factors': args.num_factors,
        'iterations': args.epochs,
        'regularization': args.regularization,
        'random_state': args.random_state,
    }

    start = time.time()
    als = AlternatingLeastSquares(**params)
    als.fit(user_items=csr_train, val_user_items=csr_val)
    logger.info(f"executed time: {(time.time() - start)/60}")

    K = [10, 20, 50]
    for k in K:
        metric = ranking_metrics_at_k(als, csr_train, csr_val, K=k)
        logger.info(f"Metric for K={k}")
        logger.info(f"NDCG@{k}: {metric['ndcg']}")
        logger.info(f"mAP@{k}: {metric['map']}")

    pickle.dump(als, open(args.save_path, "wb"))


if __name__ == "__main__":
    args = parse_args()
    main(args)