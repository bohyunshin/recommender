import pandas as pd
import os
import torch

from preprocess.preprocess_base import PreoprocessorBase

class Preprocessor(PreoprocessorBase):
    def __init__(self, **kwargs):
        super().__init__()
        # self.ratings = pd.read_csv(os.path.join( os.path.dirname(os.path.abspath(__file__)), f".movielens/{kwargs['movielens_data_type']}/ratings.csv" ))
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            f".movielens/{kwargs['movielens_data_type']}/ratings.dat")
        ratings_cols = ["userId", "movieId", "rating", "timestamp"]
        self.ratings = pd.read_csv(path, sep='::', names=ratings_cols, engine='python', encoding="ISO-8859-1")

        self.num_users = len(self.ratings["userId"].unique())
        self.num_items = len(self.ratings["movieId"].unique())

    def preprocess(self):
        self.user_id2idx = {id_: idx for (idx, id_) in enumerate(sorted(self.ratings["userId"].unique()))}
        self.movie_id2idx = {id_: idx for (idx, id_) in enumerate(sorted(self.ratings["movieId"].unique()))}
        self.ratings["userId"] = self.ratings["userId"].map(self.user_id2idx)
        self.ratings["movieId"] = self.ratings["movieId"].map(self.movie_id2idx)
        X = torch.tensor(self.ratings[["userId", "movieId"]].values)
        y = torch.tensor(self.ratings[["rating"]].values, dtype=torch.float32)
        return X, y
