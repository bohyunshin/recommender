import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),"../.."))

import argparse
import torch
from torch.utils.data import DataLoader, random_split
from torch import nn, optim
import importlib
import copy

from data_loader.triplet_uniform_negative_sampling_dataset import UniformNegativeSamplingDataset
from model.bpr import BayesianPersonalizedRanking
from tools.csr import implicit_to_csr
from loss.custom import bpr_loss
from tools.logger import setup_logger

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--regularization", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--num_factors", type=int, default=128)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--log_path", type=str, required=True)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--movielens_data_type", type=str, default="ml-latest-small")
    return parser.parse_args()


def main(args):
    logger = setup_logger(args.log_path)
    logger.info(f"selected dataset: {args.dataset}")
    logger.info(f"selected movielens data type: {args.movielens_data_type}")
    preprocessor_module = importlib.import_module(f"recommender.preprocess.{args.dataset}.preprocess_torch").Preprocessor
    preprocessor = preprocessor_module(movielens_data_type=args.movielens_data_type)
    X,y = preprocessor.preprocess()
    shape = (preprocessor.num_users, preprocessor.num_items)
    user_items = implicit_to_csr(X, shape)
    seed = torch.Generator().manual_seed(42)

    dataset = UniformNegativeSamplingDataset(X, user_items)
    train_dataset, validation_dataset = random_split(dataset, [args.train_ratio, 1-args.train_ratio], generator=seed)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=True)

    # set up pytorch model
    model = BayesianPersonalizedRanking(
        num_users=preprocessor.num_users,
        num_items=preprocessor.num_items,
        num_factors=args.num_factors,
    )
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    best_loss = float('inf')
    for epoch in range(args.epochs):
        logger.info(f"####### Epoch {epoch} #######")

        model.train()
        # training
        tr_loss = 0.0
        for data in train_dataloader:
            user_id, pos_item_id, neg_item_id = data

            optimizer.zero_grad()
            y_pred = model(user_id, pos_item_id, neg_item_id)
            loss = bpr_loss(y_pred, model.parameters(), args.regularization)
            loss.backward()
            optimizer.step()

            tr_loss += loss.item()

        tr_loss = round(tr_loss / len(train_dataloader), 6)

        model.eval()
        # validation
        with torch.no_grad():
            val_loss = 0.0
            for data in validation_dataloader:
                user_id, pos_item_id, neg_item_id = data

                optimizer.zero_grad()
                y_pred = model(user_id, pos_item_id, neg_item_id)
                loss = bpr_loss(y_pred, model.parameters(), args.regularization)

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

    # Load the best model weights
    model.load_state_dict(best_model_weights)
    logger.info("Load weight with best validation loss")

    torch.save(model.state_dict(), args.model_path)
    logger.info("Save final model")


if __name__ == "__main__":
    args = parse_args()
    main(args)