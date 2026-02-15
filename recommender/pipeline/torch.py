import copy
import importlib
import logging
import os
import traceback
from argparse import ArgumentParser

import torch
from torch import optim

from recommender.libs.constant.data.name import Field
from recommender.libs.constant.inference.recommend import TOP_K_VALUES
from recommender.libs.constant.loss.name import LossName
from recommender.libs.constant.model.module_path import MODEL_PATH
from recommender.libs.constant.model.name import ModelForwardArgument
from recommender.libs.constant.save.save import FileName
from recommender.libs.sampling.negative_sampling import NegativeSampling
from recommender.libs.validate.config import validate_config
from recommender.pipeline.base import BaseTrainPipeline
from recommender.prepare_model_data.torch import PrepareModelDataTorch


class TorchTrainPipeline(BaseTrainPipeline):
    def run(self, args: ArgumentParser.parse_args):
        validate_config(args)
        self._setup(args)
        try:
            self._log_params(args)
            self._validate_device(args)
            self.is_triplet = args.loss == LossName.BPR.value
            data = self._load_data(args)
            preprocessed_data = self._preprocess(args, data)
            self.num_users = preprocessed_data.get(Field.NUM_USERS.value)
            self.num_items = preprocessed_data.get(Field.NUM_ITEMS.value)
            self._prepare_model_data(args, preprocessed_data)
            self._setup_model(args, preprocessed_data)
            self._train(args)
            self._save_artifacts(args)
        except Exception:
            logging.error(traceback.format_exc())
            raise

    def _log_params(self, args: ArgumentParser.parse_args):
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

    def _validate_device(self, args: ArgumentParser.parse_args):
        if args.device == "cuda":
            if not torch.cuda.is_available():
                logging.warning(
                    f"device {args.device} is not available, setting device as cpu"
                )
                args.device = "cpu"

    def _prepare_model_data(
        self, args: ArgumentParser.parse_args, preprocessed_data: dict
    ):
        self.prepare_model_data = PrepareModelDataTorch(
            model=args.model,
            num_users=self.num_users,
            num_items=self.num_items,
            train_ratio=args.train_ratio,
            num_negative_samples=args.num_neg,
            implicit=args.implicit,
            random_state=args.random_state,
            batch_size=args.batch_size,
            device=args.device,
        )
        self.train_dataloader, self.validation_dataloader = (
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
            mu=self.prepare_model_data.mu,
            loss_name=args.loss,
        ).to(args.device)

        self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr)

    def _run_train_epoch(self, args: ArgumentParser.parse_args) -> float:
        self.model.train()
        tr_loss = 0.0
        for user_id, pos_item_id, y_train in self.train_dataloader:
            inputs = {
                ModelForwardArgument.USER_IDX.value: user_id,
                ModelForwardArgument.ITEM_IDX.value: pos_item_id,
            }
            if args.num_neg is not None:
                ng_sample = NegativeSampling(
                    batch_user_id=user_id,
                    batch_item_id=pos_item_id,
                    user_item_summ=self.prepare_model_data.user_item_summ_tr,
                    num_ng=args.num_neg,
                    is_triplet=self.is_triplet,
                    num_item=self.num_items,
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
            self.optimizer.zero_grad()
            if self.is_triplet:
                y_pred = self.model.triplet(**inputs)
            else:
                y_pred = self.model(**inputs)
            loss = self.model.calculate_loss(
                y_pred=y_pred,
                y=y_train.to(args.device),
                params=[param for param in self.model.parameters()],
                regularization=args.regularization,
                user_idx=user_id,
                item_idx=pos_item_id,
                num_users=self.num_users,
                num_items=self.num_items,
            )
            loss.backward()
            self.optimizer.step()

            tr_loss += loss.item()

        return round(tr_loss / len(self.train_dataloader), 6)

    def _run_validation_epoch(self, args: ArgumentParser.parse_args) -> float:
        self.model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for user_id, pos_item_id, y_val in self.validation_dataloader:
                inputs = {
                    ModelForwardArgument.USER_IDX.value: user_id,
                    ModelForwardArgument.ITEM_IDX.value: pos_item_id,
                }
                if args.num_neg is not None:
                    ng_sample = NegativeSampling(
                        batch_user_id=user_id,
                        batch_item_id=pos_item_id,
                        user_item_summ=self.prepare_model_data.user_item_summ_tr_val,
                        num_ng=args.num_neg,
                        is_triplet=self.is_triplet,
                        num_item=self.num_items,
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
                self.optimizer.zero_grad()
                if self.is_triplet:
                    y_pred = self.model.triplet(**inputs)
                else:
                    y_pred = self.model(**inputs)
                loss = self.model.calculate_loss(
                    y_pred=y_pred,
                    y=y_val.to(args.device),
                    params=[param for param in self.model.parameters()],
                    regularization=args.regularization,
                    user_idx=user_id,
                    item_idx=pos_item_id,
                    num_users=self.num_users,
                    num_items=self.num_items,
                )

                val_loss += loss.item()
            return round(val_loss / len(self.validation_dataloader), 6)

    def _train(self, args: ArgumentParser.parse_args):
        best_loss = float("inf")
        early_stopping = False
        for epoch in range(args.epochs):
            logging.info(f"####### Epoch {epoch} #######")

            tr_loss = self._run_train_epoch(args)
            self.model.tr_loss.append(tr_loss)

            val_loss = self._run_validation_epoch(args)
            self.model.val_loss.append(val_loss)

            logging.info(f"Train Loss: {tr_loss}")
            logging.info(f"Validation Loss: {val_loss}")

            if best_loss > val_loss:
                prev_best_loss = best_loss
                best_loss = val_loss
                self.best_model_weights = copy.deepcopy(self.model.state_dict())
                patience = args.patience
                torch.save(
                    self.model.state_dict(),
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
            self.model.recommend_all(
                X_train=self.prepare_model_data.X_y.get(Field.X_TRAIN.value),
                X_val=self.prepare_model_data.X_y.get(Field.X_VAL.value),
                top_k_values=TOP_K_VALUES,
                filter_already_liked=True,
            )

            # logging calculated metrics for current epoch
            self.model.collect_metrics()

            if early_stopping:
                break

    def _save_artifacts(self, args: ArgumentParser.parse_args):
        self._save_metrics_and_losses(self.model, args.result_path)

        # Load the best model weights
        self.model.load_state_dict(self.best_model_weights)
        logging.info("Load weight with best validation loss")

        torch.save(
            self.model.state_dict(),
            os.path.join(args.result_path, FileName.WEIGHT_PT.value),
        )
        logging.info("Save final model")
