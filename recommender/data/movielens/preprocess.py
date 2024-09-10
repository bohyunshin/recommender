import pandas as pd
import os
import torch

class Preprocessor:
    def __init__(self, **kwargs):
        self.ratings = pd.read_csv(os.path.join( os.path.dirname(os.path.abspath(__file__)), f".movielens/{kwargs['movielens_data_type']}/ratings.csv" ))
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
