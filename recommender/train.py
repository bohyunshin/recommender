import os
import sys
import traceback
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),"../.."))

import torch
from torch.utils.data import DataLoader, random_split
from torch import optim
import numpy as np
import importlib
import copy
import time

from tools.logger import setup_logger
from tools.parse_args import parse_args
from tools.evaluation import ranking_metrics_at_k
from tools.csr import implicit_to_csr
from loss.criterion import Criterion


def main(args):
    logger = setup_logger(args.log_path)
    try:
        logger.info(f"selected dataset: {args.dataset}")
        logger.info(f"selected model: {args.model}")
        logger.info(f"batch size: {args.batch_size}")
        logger.info(f"learning rate: {args.lr}")
        logger.info(f"regularization: {args.regularization}")
        logger.info(f"epochs: {args.epochs}")
        logger.info(f"number of factors for user / item embedding: {args.num_factors}")
        logger.info(f"train ratio: {args.train_ratio}")
        logger.info(f"patience for watching validation loss: {args.patience}")
        if args.movielens_data_type != None:
            logger.info(f"selected movielens data type: {args.movielens_data_type}")

        # set device type: cpu or gpu
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_default_device(device.type)
        logger.info(f"device info: {torch.get_default_device()}")

        # prepare train / validation dataset
        # we use preprocessor in preprocess_csr.py when running pytorch based models
        preprocessor_module = importlib.import_module(f"preprocess.{args.dataset}.preprocess_torch").Preprocessor
        preprocessor = preprocessor_module(movielens_data_type=args.movielens_data_type)
        X,y = preprocessor.preprocess()

        X = X.to(device)
        y = y.to(device)

        # when implicit feedback, i.e., args.implicit equals True,
        # user-item interaction information is required when negative sampling
        shape = (preprocessor.num_users, preprocessor.num_items)
        user_items_dct = implicit_to_csr(X, shape, True)

        seed = torch.Generator(device=device.type).manual_seed(args.random_state)
        dataset_args = {
            "X":X,
            "y":y,
            "user_items_dct":user_items_dct,
            "num_items":preprocessor.num_items
        }

        if args.implicit == True:
            if args.model == "bpr":
                dataset_path = "data_loader.triplet_uniform_negative_sampling_dataset"
            elif args.model in ["gmf", "mlp"]:
                dataset_path = "data_loader.bce_uniform_negative_sampling_dataset"
        else:
            dataset_path = f"data_loader.data"
        dataset_module = importlib.import_module(dataset_path).Data
        dataset = dataset_module(**dataset_args)

        if args.implicit == True:
            start = time.time()
            dataset.negative_sampling()
            logger.info(f"token time for negative sampling: {time.time() - start}")

        # split train / validation dataset
        train_dataset, validation_dataset = random_split(dataset, [args.train_ratio, 1-args.train_ratio], generator=seed)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, generator=seed)
        validation_dataloader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=True, generator=seed)
        mu = train_dataset.dataset.y[train_dataset.indices].mean() if args.model in ["svd", "svd_bias"] else None

        # set up model
        if args.model in ["svd", "svd_bias"]:
            model_path = f"model.mf.{args.model}"
        elif args.model in ["gmf", "mlp"]:
            model_path = f"model.deep_learning.{args.model}"
        else: # bpr
            model_path = f"model.{args.model}"
        model_module = importlib.import_module(model_path).Model
        args.num_users = preprocessor.num_users
        args.num_items = preprocessor.num_items
        args.mu = mu
        model = model_module(**vars(args))
        criterion = Criterion(args.model)
        optimizer = optim.SGD(model.parameters(), lr=args.lr)

        # train model
        best_loss = float('inf')
        for epoch in range(args.epochs):
            logger.info(f"####### Epoch {epoch} #######")

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
                    loss_kwargs["num_users"] = preprocessor.num_users
                    loss_kwargs["num_items"] = preprocessor.num_items
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
                        loss_kwargs["num_users"] = preprocessor.num_users
                        loss_kwargs["num_items"] = preprocessor.num_items

                    y_pred = model(*inputs)
                    loss_kwargs["y_pred"] = y_pred
                    loss_kwargs["y"] = y_val
                    loss = criterion.calculate_loss(**loss_kwargs)

                    val_loss += loss.item()
                val_loss = round(val_loss / len(validation_dataloader), 6)

            logger.info(f"Train Loss: {tr_loss}")
            logger.info(f"Validation Loss: {val_loss}")

            if best_loss > val_loss:
                prev_best_loss = best_loss
                best_loss = val_loss
                best_model_weights = copy.deepcopy(model.state_dict())
                patience = args.patience
                torch.save(model.state_dict(), args.model_path)
                logger.info(f"Best validation: {best_loss}, Previous validation loss: {prev_best_loss}")
            else:
                patience -= 1
                logger.info(f"Validation loss did not decrease. Patience {patience} left.")
                if patience == 0:
                    logger.info(f"Patience over. Early stopping at epoch {epoch} with {best_loss} validation loss")
                    break
        model.set_trained_embedding()


        K = [10, 20, 50]
        if args.implicit: # torch & implicit > bpr, ncf, gmf
            tr_pos_idx = np.intersect1d(
                (train_dataset.dataset.label == 1).nonzero().squeeze().detach().cpu().numpy(),
                train_dataset.indices
            )
            val_pos_idx = np.intersect1d(
                (validation_dataset.dataset.label == 1).nonzero().squeeze().detach().cpu().numpy(),
                validation_dataset.indices
            )
        else: # torch & explicit > svd, svd_bias
            tr_pos_idx = train_dataset.indices
            val_pos_idx = validation_dataset.indices
        csr_train = implicit_to_csr(train_dataset.dataset.X[tr_pos_idx], shape)
        csr_val = implicit_to_csr(validation_dataset.dataset.X[val_pos_idx], shape)
        for k in K:
            metric = ranking_metrics_at_k(model, csr_train, csr_val, K=k)
            logger.info(f"Metric for K={k}")
            logger.info(f"NDCG@{k}: {metric['ndcg']}")
            logger.info(f"mAP@{k}: {metric['map']}")

        # Load the best model weights
        model.load_state_dict(best_model_weights)
        logger.info("Load weight with best validation loss")

        torch.save(model.state_dict(), args.model_path)
        logger.info("Save final model")
    except Exception:
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    args = parse_args()
    main(args)