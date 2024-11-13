import pandas as pd
import numpy as np
import os

from preprocess.preprocess_base import PreoprocessorBase

class PreoprocessorMovielensBase(PreoprocessorBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # load rating and meta data
        path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            ".movielens/{ml_type}/{data}.dat"
        )
        rating_cols = ["user_id", "movie_id", "rating", "timestamp"]
        self.ratings = pd.read_csv(
            path.format(ml_type=kwargs['movielens_data_type'], data="ratings"),
            sep='::', names=rating_cols,
            engine='python',
            encoding="ISO-8859-1"
        )

        # for quick pytest
        if kwargs["test"] == True:
            idxs = np.random.choice(range(self.ratings.shape[0]), size=1000, replace=False)
            self.ratings = self.ratings.iloc[idxs, :]

        movie_cols = ["movie_id", "movie_name", "genres"]
        self.movies = pd.read_csv(
            path.format(ml_type=kwargs['movielens_data_type'], data="movies"),
            sep='::',
            names=movie_cols,
            engine='python',
            encoding="ISO-8859-1"
        ).sort_values(by="movie_id")

        user_cols = ["user_id", "gender", "age", "occupation", "zip_code"]
        self.users = pd.read_csv(
            path.format(ml_type=kwargs['movielens_data_type'], data="users"),
            sep='::',
            names=user_cols,
            engine='python',
            encoding="ISO-8859-1"
        ).sort_values(by="user_id")

        self.num_users = self.users.shape[0]
        self.num_items = self.movies.shape[0]

        # mapping dictionary user_id, movie_id to ascending id
        self.user_id2idx = {id_: idx for (idx, id_) in enumerate(sorted(self.users["user_id"].unique()))}
        self.movie_id2idx = {id_: idx for (idx, id_) in enumerate(sorted(self.movies["movie_id"].unique()))}

    def preprocess(self, **kwargs):
        raise NotImplementedError