from typing import Dict

import pandas as pd
import numpy as np

from load_data.load_data_base import LoadDataBase
from recommender.libs.constant.data.movielens import (
    MovieLens1mPath,
    RATINGS_COLUMNS,
    USERS_COLUMNS,
    ITEMS_COLUMNS
)

class LoadData(LoadDataBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load(self, **kwargs) -> Dict[str, pd.DataFrame]:
        """
        Loads movielens data.
        After downloading movielens data using script in scripts/download/movielens.py,
        load data into pandas dataframe.
        Depending on type of movielens dataset such as ml-1m or ml-10m, loading logic will be
        different because format or column names are different.
        First, focus on consisting pipeline with movielens 1m, then will integrate 10m later.

        Returns (Dict[str, pd.DataFrame]):
            Basically, abstractmethod `load` is designed to return three types of dataframes.
            Ratings, user-meta, item-meta related are target dataset to be loaded.
            However, depending on situation of different dataset (movielens, yelp, pinterest),
            this return type will be modified.
        """
        # load rating and meta data
        ratings = pd.read_csv(
            MovieLens1mPath.ratings.value,
            sep="::",
            names=RATINGS_COLUMNS,
            engine="python",
            encoding="ISO-8859-1"
        )

        # for quick pytest
        if kwargs["test"] == True:
            idxs = np.random.choice(range(ratings.shape[0]), size=5000, replace=False)
            ratings = ratings.iloc[idxs, :]

        items = pd.read_csv(
            MovieLens1mPath.items.value,
            sep="::",
            names=ITEMS_COLUMNS,
            engine="python",
            encoding="ISO-8859-1"
        ).sort_values(by="movie_id")

        users = pd.read_csv(
            MovieLens1mPath.users.value,
            sep="::",
            names=USERS_COLUMNS,
            engine="python",
            encoding="ISO-8859-1"
        ).sort_values(by="user_id")

        return {
            "ratings": ratings,
            "users": users,
            "items": items
        }