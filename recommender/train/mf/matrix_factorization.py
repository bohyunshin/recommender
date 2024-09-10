import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),"../.."))

import argparse
import torch
from torch.utils.data import DataLoader, random_split
from torch import nn, optim
import importlib

from data.data import Data
from model.mf.matrix_factorization import MatrixFactorization

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--num_factors", type=int, default=128)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    # parser.add_argument("--loss", type=int, required=True)
    return parser.parse_args()


def main(args):
    preprocessor_module = importlib.import_module(f"recommender.data.{args.dataset}.preprocess").Preprocessor
    preprocessor = preprocessor_module("ml-latest-small")
    X,y = preprocessor.preprocess()
    n = X.shape[0]
    n_batch = n // args.batch_size + (n % args.batch_size >= 1)
    seed = torch.Generator().manual_seed(42)

    dataset = Data(X, y)
    train_dataset, validation_dataset = random_split(dataset, [args.train_ratio, 1-args.train_ratio], generator=seed)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=True)

    # set up pytorch model
    model = MatrixFactorization(
        num_users=preprocessor.num_users,
        num_items=preprocessor.num_items,
        num_factors=args.num_factors,
    )
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        print(f"####### Epoch {epoch} #######")

        # training
        tr_loss = 0.0
        for data in train_dataloader:
            X_train, y_train = data
            users, items = X_train[:, 0], X_train[:, 1]

            optimizer.zero_grad()
            y_pred = model(users, items)
            loss = criterion(y_pred, y_train)
            loss.backward()
            optimizer.step()

            tr_loss += loss.item()

        tr_loss /= len(train_dataloader)

        # validation
        with torch.no_grad():
            val_loss = 0.0
            for data in validation_dataloader:
                X_val, y_val = data
                users, items = X_val[:, 0], X_val[:, 1]

                y_pred = model(users, items)
                loss = criterion(y_pred, y_val)

                val_loss += loss.item()
            val_loss /= len(validation_dataloader)

        print(f"Train Loss: {tr_loss}")
        print(f"Validation Loss: {val_loss}")


if __name__ == "__main__":
    args = parse_args()
    main(args)