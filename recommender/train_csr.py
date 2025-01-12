import os
import traceback
from argparse import ArgumentParser
import logging
import pickle
import importlib
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch

from recommender.prepare_model_data.prepare_model_data_csr import PrepareModelDataCsr
from recommender.libs.logger import setup_logger
from recommender.libs.parse_args import parse_args
from recommender.libs.plot.plot import plot_metric_at_k
from recommender.libs.constant.model.module_path import MODEL_PATH
from recommender.libs.constant.inference.recommend import TOP_K_VALUES
from recommender.libs.constant.save.save import FileName
from recommender.libs.constant.model.name import ModelName


def main(args: ArgumentParser.parse_args):
    os.makedirs(args.result_path, exist_ok=True)
    setup_logger(os.path.join(args.result_path, FileName.LOG.value))
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
        preprocessed_data = preprocess_module().preprocess(data)
        NUM_USERS = preprocessed_data.get("num_users")
        NUM_ITEMS = preprocessed_data.get("num_items")

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
        csr_train, csr_val = prepare_model_data.get_train_validation_data(data=preprocessed_data)

        # setup models
        model_path = MODEL_PATH.get(args.model)
        if model_path is None:
            raise
        model_module = importlib.import_module(model_path).Model
        model = model_module(
            user_ids=torch.tensor(list(preprocessed_data.get("user_id2idx").values())),  # common model parameter
            item_ids=torch.tensor(list(preprocessed_data.get("item_id2idx").values())),  # common model parameter
            num_users=NUM_USERS,  # common model parameter
            num_items=NUM_ITEMS,  # common model parameter
            num_factors=args.num_factors,  # common model parameter
            regularization=args.regularization,  # als parameter
            iterations=args.epochs,  # als parameter
            random_state=args.random_state,  # als parameter
            num_sim_user_top_N=args.num_sim_user_top_N,  # user_based parameter
        )

        # train model
        best_loss = float("inf")
        for epoch in range(args.epochs):
            logging.info(f"####### Epoch {epoch} #######")
            model.fit(user_items=csr_train, val_user_items=csr_val)

            if args.model != ModelName.USER_BASED.value:  # no loss defined in user_based model
                if best_loss > model.current_val_loss:
                    prev_best_loss = best_loss
                    best_loss = model.current_val_loss
                    patience = args.patience
                    pickle.dump(model, open(os.path.join(args.result_path, FileName.MODEL_PKL.value), "wb"))
                    logging.info(f"Best validation: {best_loss}, Previous validation loss: {prev_best_loss}")
                else:
                    patience -= 1
                    logging.info(f"Validation loss did not decrease. Patience {patience} left.")
                    if patience == 0:
                        logging.info(f"Patience over. Early stopping at epoch {epoch} with {best_loss} validation loss")
                        break
            else:
                pickle.dump(model, open(os.path.join(args.result_path, FileName.MODEL_PKL.value), "wb"))

            # calculate metrics for all users
            model.recommend_all(
                X_train=prepare_model_data.X_y.get("X_train"),
                X_val=prepare_model_data.X_y.get("X_val"),
                top_k_values=TOP_K_VALUES,
                filter_already_liked=True,
                user_items=csr_train,
            )

        # logging calculated metrics for current epoch
        model.collect_metrics()

        if args.model != ModelName.USER_BASED.value:
            # save metrics at every epoch
            pickle.dump(
                model.metric_at_k_total_epochs,
                open(os.path.join(args.result_path, FileName.METRIC.value), "wb")
            )

            # save loss
            pickle.dump(
                model.tr_loss,
                open(os.path.join(args.result_path, FileName.TRAINING_LOSS.value), "wb")
            )
            pickle.dump(
                model.val_loss,
                open(os.path.join(args.result_path, FileName.VALIDATION_LOSS.value), "wb")
            )

            # plot metrics
            plot_metric_at_k(
                metric=model.metric_at_k_total_epochs,
                tr_loss=model.tr_loss,
                val_loss=model.val_loss,
                parent_save_path=args.result_path,
            )

    except Exception:
        logging.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    args = parse_args()
    main(args)