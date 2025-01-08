from abc import ABC, abstractmethod
from typing import Dict

import pandas as pd


class LoadDataBase(ABC):
    def __init__(self, **kwargs):
        """
        Abstract base class for all data loaders.
        """
        pass

    @abstractmethod
    def load(self, **kwargs) -> Dict[str, pd.DataFrame]:
        """
        Abstract method for loading data.

        Returns (Dict[str, pd.DataFrame]):
            Basically, abstractmethod `load` is designed to return three types of dataframes.
            Ratings, user-meta, item-meta related are target dataset to be loaded.
            However, depending on situation of different dataset (movielens, yelp, pinterest),
            this return type will be modified.
        """
        raise NotImplementedError