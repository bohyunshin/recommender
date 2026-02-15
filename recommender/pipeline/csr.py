import logging
import os
import pickle
from argparse import ArgumentParser

import importlib

import torch

from recommender.libs.constant.inference.recommend import TOP_K_VALUES
from recommender.libs.constant.model.module_path import MODEL_PATH
from recommender.libs.constant.model.name import ModelName
from recommender.libs.constant.save.save import FileName
from recommender.libs.constant.data.name import Field
from recommender.pipeline.base import BaseTrainPipeline
from recommender.prepare_model_data.csr import PrepareModelDataCsr


class CsrTrainPipeline(BaseTrainPipeline):
    def _log_params(self, args: ArgumentParser.parse_args):
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

    def _prepare_model_data(
        self, args: ArgumentParser.parse_args, preprocessed_data: dict
    ):
        self.prepare_model_data = PrepareModelDataCsr(
            model=args.model,
            num_users=self.num_users,
            num_items=self.num_items,
            train_ratio=args.train_ratio,
            num_negative_samples=args.num_neg,
            implicit=args.implicit,
            random_state=args.random_state,
            batch_size=args.batch_size,
        )
        self.csr_train, self.csr_val = (
            self.prepare_model_data.get_train_validation_data(data=preprocessed_data)
        )

    def _setup_model(self, args: ArgumentParser.parse_args, preprocessed_data: dict):
        model_path = MODEL_PATH.get(args.model)
        if model_path is None:
            raise
        model_module = importlib.import_module(model_path).Model
        self.model = model_module(
            user_ids=torch.tensor(
                list(preprocessed_data.get(Field.USER_ID2IDX.value).values())
            ),
            item_ids=torch.tensor(
                list(preprocessed_data.get(Field.ITEM_ID2IDX.value).values())
            ),
            num_users=self.num_users,
            num_items=self.num_items,
            num_factors=args.num_factors,
            loss_name=args.loss,
            regularization=args.regularization,
            iterations=args.epochs,
            random_state=args.random_state,
            num_sim_user_top_N=args.num_sim_user_top_N,
        )

    def _train(self, args: ArgumentParser.parse_args):
        best_loss = float("inf")
        early_stopping = False
        for epoch in range(args.epochs):
            logging.info(f"####### Epoch {epoch} #######")
            self.model.fit(user_items=self.csr_train, val_user_items=self.csr_val)

            if args.model != ModelName.USER_BASED.value:
                # calculate training / validation loss
                tr_loss = self.model.calculate_loss(
                    user_items=self.csr_train,
                    user_factors=self.model.user_factors,
                    item_factors=self.model.item_factors,
                    regularization=args.regularization,
                )
                self.model.tr_loss.append(tr_loss)
                logging.info(f"training loss: {tr_loss}")

                val_loss = self.model.calculate_loss(
                    user_items=self.csr_val,
                    user_factors=self.model.user_factors,
                    item_factors=self.model.item_factors,
                    regularization=args.regularization,
                )
                self.model.val_loss.append(val_loss)
                logging.info(f"validation loss: {val_loss}")

                if best_loss > val_loss:
                    prev_best_loss = best_loss
                    best_loss = val_loss
                    patience = args.patience
                    pickle.dump(
                        self.model,
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
                    self.model,
                    open(
                        os.path.join(args.result_path, FileName.MODEL_PKL.value), "wb"
                    ),
                )
                break

            # calculate metrics for all users
            self.model.recommend_all(
                X_train=self.prepare_model_data.X_y.get(Field.X_TRAIN.value),
                X_val=self.prepare_model_data.X_y.get(Field.X_VAL.value),
                top_k_values=TOP_K_VALUES,
                filter_already_liked=True,
                user_items=self.csr_train,
            )

            # logging calculated metrics for current epoch
            self.model.collect_metrics()

            if early_stopping:
                break

    def _save_artifacts(self, args: ArgumentParser.parse_args):
        if args.model != ModelName.USER_BASED.value:
            self._save_metrics_and_losses(self.model, args.result_path)
