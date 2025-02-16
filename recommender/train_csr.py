import importlib
import logging
import os
import pickle
import traceback
from argparse import ArgumentParser

import torch

from recommender.libs.constant.inference.recommend import TOP_K_VALUES
from recommender.libs.constant.model.module_path import MODEL_PATH
from recommender.libs.constant.model.name import ModelName
from recommender.libs.constant.save.save import FileName
from recommender.libs.constant.data.name import Field
from recommender.libs.plot.plot import plot_metric_at_k
from recommender.libs.utils.logger import setup_logger
from recommender.libs.utils.parse_args import parse_args
from recommender.prepare_model_data.csr import PrepareModelDataCsr

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def main(args: ArgumentParser.parse_args):
    os.makedirs(args.result_path, exist_ok=True)
    setup_logger(os.path.join(args.result_path, FileName.LOG.value))
    try:
        logging.info(f"selected dataset: {args.dataset}")
        logging.info(f"selected model: {args.model}")
        logging.info(f"selected loss: {args.loss}")
        if args.model == ModelName.ALS.value:
            logging.info(f"batch size: {args.batch_size}")
            logging.info(f"learning rate: {args.lr}")
            logging.info(f"regularization: {args.regularization}")
            logging.info(f"epochs: {args.epochs}")
            logging.info(
                f"number of factors for user / item embedding: {args.num_factors}"
            )
            logging.info(f"patience for watching validation loss: {args.patience}")
            logging.info(f"random state: {args.random_state}")
            logging.info(f"patience for watching validation loss: {args.patience}")
        logging.info(f"train ratio: {args.train_ratio}")
        if args.model == ModelName.USER_BASED.value:
            args.epochs = (
                1  # for user_based model, iterations no more than 2 is not needed
            )
            logging.info(f"num_sim_user_top_N: {args.num_sim_user_top_N}")
        logging.info(f"result path: {args.result_path}")
        logging.info(f"test mode: {args.is_test}")

        # load raw data
        load_data_module = importlib.import_module(
            f"recommender.load_data.{args.dataset}"
        ).LoadData
        data = load_data_module().load(is_test=args.is_test)

        # preprocess data
        preprocess_module = importlib.import_module(
            f"recommender.preprocess.{args.dataset}"
        ).Preprocessor
        preprocessed_data = preprocess_module().preprocess(data)
        NUM_USERS = preprocessed_data.get(Field.NUM_USERS.value)
        NUM_ITEMS = preprocessed_data.get(Field.NUM_ITEMS.value)

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
        )
        csr_train, csr_val = prepare_model_data.get_train_validation_data(
            data=preprocessed_data
        )

        # setup models
        model_path = MODEL_PATH.get(args.model)
        if model_path is None:
            raise
        model_module = importlib.import_module(model_path).Model
        model = model_module(
            user_ids=torch.tensor(
                list(preprocessed_data.get(Field.USER_ID2IDX.value).values())
            ),  # common model parameter
            item_ids=torch.tensor(
                list(preprocessed_data.get(Field.ITEM_ID2IDX.value).values())
            ),  # common model parameter
            num_users=NUM_USERS,  # common model parameter
            num_items=NUM_ITEMS,  # common model parameter
            num_factors=args.num_factors,  # common model parameter
            loss_name=args.loss,  # als parameter
            regularization=args.regularization,  # als parameter
            iterations=args.epochs,  # als parameter
            random_state=args.random_state,  # als parameter
            num_sim_user_top_N=args.num_sim_user_top_N,  # user_based parameter
        )

        # train model
        best_loss = float("inf")
        early_stopping = False
        for epoch in range(args.epochs):
            logging.info(f"####### Epoch {epoch} #######")
            model.fit(user_items=csr_train, val_user_items=csr_val)

            if (
                args.model != ModelName.USER_BASED.value
            ):  # no loss defined in user_based model
                # calculate training / validation loss
                tr_loss = model.calculate_loss(
                    user_items=csr_train,
                    user_factors=model.user_factors,
                    item_factors=model.item_factors,
                    regularization=args.regularization,
                )
                model.tr_loss.append(tr_loss)
                logging.info(f"training loss: {tr_loss}")

                val_loss = model.calculate_loss(
                    user_items=csr_val,
                    user_factors=model.user_factors,
                    item_factors=model.item_factors,
                    regularization=args.regularization,
                )
                model.val_loss.append(val_loss)
                logging.info(f"validation loss: {val_loss}")

                if best_loss > val_loss:
                    prev_best_loss = best_loss
                    best_loss = val_loss
                    patience = args.patience
                    pickle.dump(
                        model,
                        open(
                            os.path.join(args.result_path, FileName.MODEL_PKL.value),
                            "wb",
                        ),
                    )
                    logging.info(
                        f"Best validation: {best_loss}, Previous validation loss: {prev_best_loss}"
                    )
                else:
                    patience -= 1
                    logging.info(
                        f"Validation loss did not decrease. Patience {patience} left."
                    )
                    if patience == 0:
                        logging.info(
                            f"Patience over. Early stopping at epoch {epoch} with {best_loss} validation loss"
                        )
                        early_stopping = True
            else:
                # when user_based model, do not have to iterate training
                pickle.dump(
                    model,
                    open(
                        os.path.join(args.result_path, FileName.MODEL_PKL.value), "wb"
                    ),
                )
                break

            # calculate metrics for all users
            model.recommend_all(
                X_train=prepare_model_data.X_y.get(Field.X_TRAIN.value),
                X_val=prepare_model_data.X_y.get(Field.X_VAL.value),
                top_k_values=TOP_K_VALUES,
                filter_already_liked=True,
                user_items=csr_train,
            )

            # logging calculated metrics for current epoch
            model.collect_metrics()

            if early_stopping is True:
                break

        if args.model != ModelName.USER_BASED.value:
            # save metrics at every epoch
            pickle.dump(
                model.metric_at_k_total_epochs,
                open(os.path.join(args.result_path, FileName.METRIC.value), "wb"),
            )

            # save loss
            pickle.dump(
                model.tr_loss,
                open(
                    os.path.join(args.result_path, FileName.TRAINING_LOSS.value), "wb"
                ),
            )
            pickle.dump(
                model.val_loss,
                open(
                    os.path.join(args.result_path, FileName.VALIDATION_LOSS.value), "wb"
                ),
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
