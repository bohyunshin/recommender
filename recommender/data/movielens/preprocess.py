import pandas as pd
import torch

class Preprocessor:
    def __init__(self, data_type):
        self.ratings = pd.read_csv(f".movielens/{data_type}/ratings.csv")

    def preprocess(self):
        self.user_id2idx = {id_: idx for (idx, id_) in enumerate(sorted(self.ratings["userId"].unique()))}
        self.movie_id2idx = {id_: idx for (idx, id_) in enumerate(sorted(self.ratings["movieId"].unique()))}
        self.ratings["userId"] = self.ratings["userId"].map(self.user_id2idx)
        self.ratings["movieId"] = self.ratings["movieId"].map(self.movie_id2idx)
        X = torch.from_numpy(self.ratings[["userId", "movieId"]].values)
        y = torch.from_numpy(self.ratings[["rating"]].values)
        return X, y
