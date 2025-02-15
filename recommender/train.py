import copy
import logging
import os
import pickle
import traceback
from argparse import ArgumentParser

import importlib

import torch
from torch import optim

from recommender.libs.constant.inference.recommend import TOP_K_VALUES
from recommender.libs.constant.loss.name import LossName
from recommender.libs.constant.model.module_path import MODEL_PATH
from recommender.libs.constant.model.name import ModelForwardArgument
from recommender.libs.constant.save.save import FileName
from recommender.libs.constant.data.name import Field
from recommender.libs.plot.plot import plot_metric_at_k
from recommender.libs.sampling.negative_sampling import NegativeSampling
from recommender.libs.utils.logger import setup_logger
from recommender.libs.utils.parse_args import parse_args
from recommender.libs.validate.config import validate_config
from recommender.prepare_model_data.torch import (
    PrepareModelDataTorch,
)

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def main(args: ArgumentParser.parse_args):
    validate_config(args)
    os.makedirs(args.result_path, exist_ok=True)
    setup_logger(os.path.join(args.result_path, FileName.LOG.value))
    try:
        logging.info(f"selected dataset: {args.dataset}")
        logging.info(f"selected model: {args.model}")
        logging.info(f"selected loss: {args.loss}")
        if args.num_neg is not None:
            logging.info(f"implicit dataset: {args.implicit}")
            logging.info(f"negative sampling strategy: {args.neg_sample_strategy}")
            logging.info(
                f"number of negative samples: {args.num_neg * len(args.neg_sample_strategy)}"
            )
        logging.info(f"batch size: {args.batch_size}")
        logging.info(f"learning rate: {args.lr}")
        logging.info(f"regularization: {args.regularization}")
        logging.info(f"epochs: {args.epochs}")
        logging.info(f"number of factors for user / item embedding: {args.num_factors}")
        logging.info(f"train ratio: {args.train_ratio}")
        logging.info(f"random state: {args.random_state}")
        logging.info(f"patience for watching validation loss: {args.patience}")
        logging.info(f"result path: {args.result_path}")
        logging.info(f"test mode: {args.is_test}")
        if args.device == "cuda":
            if not torch.cuda.is_available():
                logging.warning(
                    f"device {args.device} is not available, setting device as cpu"
                )
                args.device = "cpu"

        is_triplet = True if args.loss == LossName.BPR.value else False

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
        prepare_model_data = PrepareModelDataTorch(
            model=args.model,
            num_users=NUM_USERS,
            num_items=NUM_ITEMS,
            train_ratio=args.train_ratio,
            num_negative_samples=args.num_neg,
            implicit=args.implicit,
            random_state=args.random_state,
            batch_size=args.batch_size,
            device=args.device,
        )
        train_dataloader, validation_dataloader = (
            prepare_model_data.get_train_validation_data(data=preprocessed_data)
        )

        # set up model
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
            mu=prepare_model_data.mu,  # for svd_bias model
            loss_name=args.loss,
        ).to(args.device)

        optimizer = optim.SGD(model.parameters(), lr=args.lr)

        # train model
        best_loss = float("inf")
        early_stopping = False
        for epoch in range(args.epochs):
            logging.info(f"####### Epoch {epoch} #######")

            # training
            model.train()
            tr_loss = 0.0
            for user_id, pos_item_id, y_train in train_dataloader:
                inputs = {
                    ModelForwardArgument.USER_IDX.value: user_id,
                    ModelForwardArgument.ITEM_IDX.value: pos_item_id,
                }
                if args.num_neg is not None:
                    ng_sample = NegativeSampling(
                        batch_user_id=user_id,
                        batch_item_id=pos_item_id,
                        user_item_summ=prepare_model_data.user_item_summ_tr,
                        num_ng=args.num_neg,
                        is_triplet=is_triplet,
                        num_item=NUM_ITEMS,
                        strategy=args.neg_sample_strategy,
                        device=args.device,
                    )
                    ng_sample.ng()
                    ng_res = ng_sample.format_dataset()
                    inputs = {
                        **inputs,
                        **ng_res,
                    }
                    y_train = ng_res.get(ModelForwardArgument.Y.value)
                optimizer.zero_grad()
                if is_triplet:
                    y_pred = model.triplet(**inputs)
                else:
                    y_pred = model(**inputs)
                loss = model.calculate_loss(
                    y_pred=y_pred,
                    y=y_train.to(args.device),
                    params=[param for param in model.parameters()],
                    regularization=args.regularization,
                    user_idx=user_id,  # used in svd, svd_bias
                    item_idx=pos_item_id,  # used in svd, svd_bias
                    num_users=NUM_USERS,  # used in svd, svd_bias
                    num_items=NUM_ITEMS,  # used in svd, svd_bias
                )
                loss.backward()
                optimizer.step()

                tr_loss += loss.item()

            tr_loss = round(tr_loss / len(train_dataloader), 6)
            model.tr_loss.append(tr_loss)

            # validation
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                for user_id, pos_item_id, y_val in validation_dataloader:
                    inputs = {
                        ModelForwardArgument.USER_IDX.value: user_id,
                        ModelForwardArgument.ITEM_IDX.value: pos_item_id,
                    }
                    if args.num_neg is not None:
                        ng_sample = NegativeSampling(
                            batch_user_id=user_id,
                            batch_item_id=pos_item_id,
                            user_item_summ=prepare_model_data.user_item_summ_tr_val,
                            num_ng=args.num_neg,
                            is_triplet=is_triplet,
                            num_item=NUM_ITEMS,
                            strategy=args.neg_sample_strategy,
                            device=args.device,
                        )
                        ng_sample.ng()
                        ng_res = ng_sample.format_dataset()
                        inputs = {
                            **inputs,
                            **ng_res,
                        }
                        y_val = ng_res.get(ModelForwardArgument.Y.value)
                    optimizer.zero_grad()
                    if is_triplet:
                        y_pred = model.triplet(**inputs)
                    else:
                        y_pred = model(**inputs)
                    loss = model.calculate_loss(
                        y_pred=y_pred,
                        y=y_val.to(args.device),
                        params=[param for param in model.parameters()],
                        regularization=args.regularization,
                        user_idx=user_id,  # used in svd, svd_bias
                        item_idx=pos_item_id,  # used in svd, svd_bias
                        num_users=NUM_USERS,  # used in svd, svd_bias
                        num_items=NUM_ITEMS,  # used in svd, svd_bias
                    )

                    val_loss += loss.item()
                val_loss = round(val_loss / len(validation_dataloader), 6)
                model.val_loss.append(val_loss)

            logging.info(f"Train Loss: {tr_loss}")
            logging.info(f"Validation Loss: {val_loss}")

            if best_loss > val_loss:
                prev_best_loss = best_loss
                best_loss = val_loss
                best_model_weights = copy.deepcopy(model.state_dict())
                patience = args.patience
                torch.save(
                    model.state_dict(),
                    os.path.join(args.result_path, FileName.WEIGHT_PT.value),
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

            # calculate metrics for all users
            model.recommend_all(
                X_train=prepare_model_data.X_y.get(Field.X_TRAIN.value),
                X_val=prepare_model_data.X_y.get(Field.X_VAL.value),
                top_k_values=TOP_K_VALUES,
                filter_already_liked=True,
            )

            # logging calculated metrics for current epoch
            model.collect_metrics()

            if early_stopping is True:
                break

        # save metrics at every epoch
        pickle.dump(
            model.metric_at_k_total_epochs,
            open(os.path.join(args.result_path, FileName.METRIC.value), "wb"),
        )

        # save loss
        pickle.dump(
            model.tr_loss,
            open(os.path.join(args.result_path, FileName.TRAINING_LOSS.value), "wb"),
        )
        pickle.dump(
            model.val_loss,
            open(os.path.join(args.result_path, FileName.VALIDATION_LOSS.value), "wb"),
        )

        # plot metrics
        plot_metric_at_k(
            metric=model.metric_at_k_total_epochs,
            tr_loss=model.tr_loss,
            val_loss=model.val_loss,
            parent_save_path=args.result_path,
        )

        # Load the best model weights
        model.load_state_dict(best_model_weights)
        logging.info("Load weight with best validation loss")

        torch.save(
            model.state_dict(), os.path.join(args.result_path, FileName.WEIGHT_PT.value)
        )
        logging.info("Save final model")
    except Exception:
        logging.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    args = parse_args()
    main(args)
