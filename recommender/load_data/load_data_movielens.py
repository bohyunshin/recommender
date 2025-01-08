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
        """
        Base class for preprocessing movielens data.
        In init function, loads rating, movie, user data.

        rating.csv: data with likes pair (user_id, item_id)
        movie.csv: metadata associated with movies.
        user.csv: metadata associated with users.

        After loading required dataset, it maps original user_id and item_id into
        ascending integer by 1-1 relationship.
        """
        super().__init__(**kwargs)

    def load(self, **kwargs) -> Dict[str, pd.DataFrame]:
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
            idxs = np.random.choice(range(ratings.shape[0]), size=1000, replace=False)
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