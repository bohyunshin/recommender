import os
import sys
import traceback
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),"../.."))

import importlib
import pickle
import time

from tools.evaluation import ranking_metrics_at_k
from tools.logger import setup_logger
from tools.parse_args import parse_args


def main(args):
    logger = setup_logger(args.log_path)
    try:
        logger.info(f"selected dataset: {args.dataset}")
        logger.info(f"selected model: {args.model}")
        if args.model in ["als"]:
            logger.info(f"batch size: {args.batch_size}")
            logger.info(f"learning rate: {args.lr}")
            logger.info(f"regularization: {args.regularization}")
            logger.info(f"epochs: {args.epochs}")
            logger.info(f"number of factors for user / item embedding: {args.num_factors}")
            logger.info(f"patience for watching validation loss: {args.patience}")
        logger.info(f"train ratio: {args.train_ratio}")
        if args.movielens_data_type != None:
            logger.info(f"selected movielens data type: {args.movielens_data_type}")

        # set preprocessor for csr input models
        preprocessor_module = importlib.import_module(f"preprocess.{args.dataset}.preprocess_csr").Preprocessor
        preprocessor = preprocessor_module(movielens_data_type=args.movielens_data_type,
                                           test_ratio=args.train_ratio,
                                           random_state=args.random_state)
        csr_train, csr_val = preprocessor.preprocess()

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
        logger.info(f"total executed time: {(time.time() - start)/60}")

        K = [10, 20, 50]
        for k in K:
            metric = ranking_metrics_at_k(model, csr_train, csr_val, K=k)
            logger.info(f"Metric for K={k}")
            logger.info(f"NDCG@{k}: {metric['ndcg']}")
            logger.info(f"mAP@{k}: {metric['map']}")

        pickle.dump(model, open(args.model_path, "wb"))
        logger.info("Save final model")
    except Exception:
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    args = parse_args()
    main(args)