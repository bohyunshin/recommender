import os
import sys
import traceback
from argparse import ArgumentParser
import logging
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),"../.."))

import importlib
import pickle
import time

from tools.evaluation import ranking_metrics_at_k
from tools.logger import setup_logger
from tools.parse_args import parse_args


def main(args: ArgumentParser.parse_args):
    setup_logger(args.log_path)
    try:
        logging.info(f"selected dataset: {args.dataset}")
        logging.info(f"selected model: {args.model}")
        if args.model in ["als"]:
            logging.info(f"batch size: {args.batch_size}")
            logging.info(f"learning rate: {args.lr}")
            logging.info(f"regularization: {args.regularization}")
            logging.info(f"epochs: {args.epochs}")
            logging.info(f"number of factors for user / item embedding: {args.num_factors}")
            logging.info(f"patience for watching validation loss: {args.patience}")
        logging.info(f"train ratio: {args.train_ratio}")
        if args.movielens_data_type != None:
            logging.info(f"selected movielens data type: {args.movielens_data_type}")

        # set preprocessor for csr input models
        preprocessor_module = importlib.import_module(f"preprocess.{args.dataset}.preprocess_csr").Preprocessor
        preprocessor = preprocessor_module(movielens_data_type=args.movielens_data_type,
                                           test=args.test)
        csr_train, csr_val = preprocessor.preprocess(test_ratio=1-args.train_ratio,
                                                     random_state=args.random_state)

        params = {
            "factors": args.num_factors,
            "iterations": args.epochs,
            "regularization": args.regularization,
            "random_state": args.random_state,
            "num_users": csr_train.shape[0],
            "num_items": csr_train.shape[1],
            "num_sim_user_top_N": args.num_sim_user_top_N,
        }

        # set models
        start = time.time()
        if args.model in ["als"]:
            model_module = importlib.import_module(f"model.mf.{args.model}").Model
        elif args.model in ["user_based"]:
            model_module = importlib.import_module(f"model.neighborhood.{args.model}").Model
        model = model_module(**params)
        model.fit(user_items=csr_train, val_user_items=csr_val)
        logging.info(f"total executed time: {(time.time() - start)/60}")

        K = [10, 20, 50]
        ndcg = []
        map = []
        for k in K:
            metric = ranking_metrics_at_k(model, csr_train, csr_val, K=k)
            logging.info(f"Metric for K={k}")
            logging.info(f"NDCG@{k}: {metric['ndcg']}")
            logging.info(f"mAP@{k}: {metric['map']}")

            ndcg.append(round(metric['ndcg'], 4))
            map.append(round(metric['map'], 4))

        logging.info(f"NDCG result: {'|'.join([str(i) for i in ndcg])}")
        logging.info(f"mAP result: {'|'.join([str(i) for i in map])}")

        pickle.dump(model, open(args.model_path, "wb"))
        logging.info("Save final model")
    except Exception:
        logging.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    args = parse_args()
    main(args)