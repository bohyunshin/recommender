import os
import traceback
from argparse import ArgumentParser
import logging
import copy
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
from torch import optim
import importlib

from recommender.prepare_model_data.prepare_model_data_torch import PrepareModelDataTorch
from recommender.loss.criterion import Criterion
from recommender.libs.logger import setup_logger
from recommender.libs.parse_args import parse_args
from recommender.libs.constant.model.module_path import MODEL_PATH
from recommender.libs.constant.torch.device import DEVICE
from recommender.libs.constant.inference.recommend import TOP_K_VALUES


def main(args: ArgumentParser.parse_args):
    os.makedirs(args.result_path, exist_ok=True)
    setup_logger(os.path.join(args.result_path, "log.log"))
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
        preprocessed_data = preprocess_module().preprocess(data)
        NUM_USERS = preprocessed_data.get("num_users")
        NUM_ITEMS = preprocessed_data.get("num_items")

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
            user_meta=preprocessed_data.get("users"),
            item_meta=preprocessed_data.get("items"),
        )
        train_dataloader, validation_dataloader = prepare_model_data.get_train_validation_data(data=preprocessed_data)

        # set up model
        model_path = MODEL_PATH.get(args.model)
        if model_path is None:
            raise
        model_module = importlib.import_module(model_path).Model
        model = model_module(
            user_ids=torch.tensor(list(preprocessed_data.get("user_id2idx").values())), # common model parameter
            item_ids=torch.tensor(list(preprocessed_data.get("item_id2idx").values())), # common model parameter
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
                torch.save(model.state_dict(), os.path.join(args.result_path, "model.pt"))
                logging.info(f"Best validation: {best_loss}, Previous validation loss: {prev_best_loss}")
            else:
                patience -= 1
                logging.info(f"Validation loss did not decrease. Patience {patience} left.")
                if patience == 0:
                    logging.info(f"Patience over. Early stopping at epoch {epoch} with {best_loss} validation loss")
                    break

            # calculate metrics for all users
            model.recommend_all(
                X_train=prepare_model_data.X_y.get("X_train"),
                X_val=prepare_model_data.X_y.get("X_val"),
                top_k_values=TOP_K_VALUES,
                filter_already_liked=True
            )

            # logging calculated metrics for current epoch
            model.collect_metrics()

        # Load the best model weights
        model.load_state_dict(best_model_weights)
        logging.info("Load weight with best validation loss")

        torch.save(model.state_dict(), os.path.join(args.result_path, "model.pt"))
        logging.info("Save final model")
    except Exception:
        logging.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    args = parse_args()
    main(args)