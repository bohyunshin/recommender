from typing import Dict

import numpy as np
import pandas as pd

from recommender.libs.constant.data.movielens_1m import (
    RATINGS_COLUMNS,
    MovieLens1mPath,
)
from recommender.libs.constant.data.name import Field
from recommender.load_data.base import LoadDataBase


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
            Basically, abstractmethod `load` is designed to return one type of dataframes.
            Interaction dataset is target dataset to be loaded.
            When rating values exist in float type, interaction will be explicit dataset.
            When rating values does not exist, interaction will be implicit dataset.
        """
        # load rating and meta data
        ratings = pd.read_csv(
            MovieLens1mPath.ratings.value,
            sep="::",
            names=RATINGS_COLUMNS,
            engine="python",
            encoding="ISO-8859-1",
        )

        # for quick pytest
        if kwargs.get("is_test") is True:
            user_pools = ratings[Field.USER_ID.value].unique()
            sampled_user_ids = np.random.choice(user_pools, size=30, replace=False)
            ratings = ratings[lambda x: x[Field.USER_ID.value].isin(sampled_user_ids)]

        return {Field.INTERACTION.value: ratings}
