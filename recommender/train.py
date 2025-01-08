import os
import traceback
from argparse import ArgumentParser
import logging
import copy
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
from torch import optim
import numpy as np
import importlib

from recommender.prepare_model_data.prepare_model_data_torch import PrepareModelDataTorch
from recommender.loss.criterion import Criterion
from recommender.libs.logger import setup_logger
from recommender.libs.parse_args import parse_args
from recommender.libs.evaluation import ranking_metrics_at_k
from recommender.libs.csr import implicit_to_csr
from recommender.libs.constant.model.module_path import MODEL_PATH
from recommender.libs.constant.torch.device import DEVICE


def main(args: ArgumentParser.parse_args):
    setup_logger(args.log_path)
    try:
        logging.info(f"selected dataset: {args.dataset}")
        logging.info(f"selected model: {args.model}")
        logging.info(f"batch size: {args.batch_size}")
        logging.info(f"learning rate: {args.lr}")
        logging.info(f"regularization: {args.regularization}")
        logging.info(f"epochs: {args.epochs}")
        logging.info(f"number of factors for user / item embedding: {args.num_factors}")
        logging.info(f"train ratio: {args.train_ratio}")
        logging.info(f"number of negative samples: {args.num_neg}")
        logging.info(f"patience for watching validation loss: {args.patience}")
        if args.movielens_data_type != None:
            logging.info(f"selected movielens data type: {args.movielens_data_type}")
        logging.info(f"device info: {DEVICE}")

        # load raw data
        load_data_module = importlib.import_module(f"load_data.load_data_{args.dataset}").LoadData
        data = load_data_module().load(test=args.test)

        # preprocess data
        preprocess_module = importlib.import_module(f"preprocess.preprocess_{args.dataset}").Preprocessor
        data = preprocess_module().preprocess(data)
        NUM_USERS = data.get("num_users")
        NUM_ITEMS = data.get("num_items")

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
            user_meta=data.get("users"),
            item_meta=data.get("items"),
        )
        train_dataloader, validation_dataloader = prepare_model_data.get_train_validation_data(data=data)

        # set up model
        model_path = MODEL_PATH.get(args.model)
        if model_path is None:
            raise
        model_module = importlib.import_module(model_path).Model
        model = model_module(
            num_users=NUM_USERS, # common model parameter
            num_items=NUM_ITEMS, # common model parameter
            num_factors=args.num_factors, # common model parameter
            mu=prepare_model_data.mu, # for svd_bias model
            user_meta=prepare_model_data.user_meta, # for two_tower model
            item_meta=prepare_model_data.item_meta, # for two_tower model
        )

        criterion = Criterion(args.model)
        optimizer = optim.SGD(model.parameters(), lr=args.lr)

        # train model
        best_loss = float('inf')
        for epoch in range(args.epochs):
            logging.info(f"####### Epoch {epoch} #######")

            # training
            model.train()
            tr_loss = 0.0
            for data in train_dataloader:
                loss_kwargs = {
                    "regularization": args.regularization,
                    "params": [param for param in model.parameters()]
                }
                if args.implicit == True:
                    inputs = data[:-1]
                    y_train = data[-1]
                else:
                    X_train, y_train = data
                    users, items = X_train[:, 0], X_train[:, 1]
                    inputs = (users, items)
                    loss_kwargs["user_idx"] = users
                    loss_kwargs["item_idx"] = items
                    loss_kwargs["num_users"] = NUM_USERS
                    loss_kwargs["num_items"] = NUM_ITEMS
                optimizer.zero_grad()
                y_pred = model(*inputs)
                loss_kwargs["y_pred"] = y_pred
                loss_kwargs["y"] = y_train
                loss = criterion.calculate_loss(**loss_kwargs)
                loss.backward()
                optimizer.step()

                tr_loss += loss.item()

            tr_loss = round(tr_loss / len(train_dataloader), 6)

            # validation
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                for data in validation_dataloader:
                    loss_kwargs = {
                        "regularization": args.regularization,
                        "params": [param for param in model.parameters()]
                    }
                    if args.implicit == True:
                        inputs = data[:-1]
                        y_val = data[-1]
                    else:
                        X_val, y_val = data
                        users, items = X_val[:, 0], X_val[:, 1]
                        inputs = (users, items)
                        loss_kwargs["user_idx"] = users
                        loss_kwargs["item_idx"] = items
                        loss_kwargs["num_users"] = NUM_USERS
                        loss_kwargs["num_items"] = NUM_ITEMS

                    y_pred = model(*inputs)
                    loss_kwargs["y_pred"] = y_pred
                    loss_kwargs["y"] = y_val
                    loss = criterion.calculate_loss(**loss_kwargs)

                    val_loss += loss.item()
                val_loss = round(val_loss / len(validation_dataloader), 6)

            logging.info(f"Train Loss: {tr_loss}")
            logging.info(f"Validation Loss: {val_loss}")

            if best_loss > val_loss:
                prev_best_loss = best_loss
                best_loss = val_loss
                best_model_weights = copy.deepcopy(model.state_dict())
                patience = args.patience
                torch.save(model.state_dict(), args.model_path)
                logging.info(f"Best validation: {best_loss}, Previous validation loss: {prev_best_loss}")
            else:
                patience -= 1
                logging.info(f"Validation loss did not decrease. Patience {patience} left.")
                if patience == 0:
                    logging.info(f"Patience over. Early stopping at epoch {epoch} with {best_loss} validation loss")
                    break


        K = [10, 20, 50]
        if args.implicit: # torch & implicit > bpr, ncf, gmf
            tr_pos_idx = np.intersect1d(
                (prepare_model_data.train_dataset.dataset.label == 1).nonzero().squeeze().detach().cpu().numpy(),
                prepare_model_data.train_dataset.indices
            )
            val_pos_idx = np.intersect1d(
                (prepare_model_data.validation_dataset.dataset.label == 1).nonzero().squeeze().detach().cpu().numpy(),
                prepare_model_data.validation_dataset.indices
            )
        else: # torch & explicit > svd, svd_bias
            tr_pos_idx = prepare_model_data.train_dataset.indices
            val_pos_idx = prepare_model_data.validation_dataset.indices
        csr_train = implicit_to_csr(prepare_model_data.train_dataset.dataset.X[tr_pos_idx], (NUM_USERS, NUM_ITEMS))
        csr_val = implicit_to_csr(prepare_model_data.validation_dataset.dataset.X[val_pos_idx], (NUM_USERS, NUM_ITEMS))
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

        # Load the best model weights
        model.load_state_dict(best_model_weights)
        logging.info("Load weight with best validation loss")

        torch.save(model.state_dict(), args.model_path)
        logging.info("Save final model")
    except Exception:
        logging.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    args = parse_args()
    main(args)