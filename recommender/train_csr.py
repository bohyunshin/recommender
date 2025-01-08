import os
import traceback
from argparse import ArgumentParser
import logging
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import importlib
import pickle
import time

from recommender.prepare_model_data.prepare_model_data_csr import PrepareModelDataCsr
from recommender.libs.constant.model.module_path import MODEL_PATH
from recommender.libs.evaluation import ranking_metrics_at_k
from recommender.libs.logger import setup_logger
from recommender.libs.parse_args import parse_args


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

        # load raw data
        load_data_module = importlib.import_module(f"load_data.load_data_{args.dataset}").LoadData
        data = load_data_module().load(test=args.test)

        # preprocess data
        preprocess_module = importlib.import_module(f"preprocess.preprocess_{args.dataset}").Preprocessor
        data = preprocess_module().preprocess(data)
        NUM_USERS = data.get("num_users")
        NUM_ITEMS = data.get("num_items")

        # prepare dataset for model
        prepare_model_data = PrepareModelDataCsr(
            model=args.model,
            num_users=NUM_USERS,
            num_items=NUM_ITEMS,
            train_ratio=args.train_ratio,
            num_negative_samples=args.num_neg,
            implicit=args.implicit,
            random_state=args.random_state,
            batch_size=args.batch_size,
            user_meta=data.get("users"),
            item_meta=data.get("items"),
        )
        csr_train, csr_val = prepare_model_data.get_train_validation_data(data=data)

        # setup models
        start = time.time()
        model_path = MODEL_PATH.get(args.model)
        if model_path is None:
            raise
        model_module = importlib.import_module(model_path).Model
        model = model_module(
            factors=args.num_factors,
            regularization=args.regularization,
            iterations=args.epochs,
            random_state=args.random_state,
            num_users=NUM_USERS,
            num_items=NUM_ITEMS,
            num_sim_user_top_N=args.num_sim_user_top_N,
        )
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