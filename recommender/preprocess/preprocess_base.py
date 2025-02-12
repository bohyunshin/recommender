from abc import ABC, abstractmethod
from typing import Dict, Union

import pandas as pd


class PreoprocessorBase(ABC):
    def __init__(self, **kwargs):
        """
        Abstract base class for all preprocessors.
        """
        pass

    @abstractmethod
    def preprocess(
        self, data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Union[pd.DataFrame, Dict[int, int]]]:
        """
        Abstract method for preprocssing data.
        In preprocessing step, various preprocess logic could be included depending on dataset.
        However, preprocess step which maps user ids and item ids is included as basis.

        Args:
            data (Dict[str, pd.DataFrame]): Return value from
                recommender/load_data/load_data_base.LoadDataBase.base will be used.

        Returns (Dict[str, Union[pd.DataFrame, Dict[int, int]]]):
            Preprocessed pandas dataframe and its mapping information.
        """
        raise NotImplementedError

    def mapping(
        self, data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Union[pd.DataFrame, Dict[int, int]]]:
        """
        Map original user_id and item_id into ascending integer.

        Args:
            data (Dict[str, pd.DataFrame]): Return value from
                recommender/load_data/load_data_base.LoadDataBase.base will be used.

        Returns (Dict[str, Union[pd.DataFrame, Dict[int, int]]]):
            Preprocessed pandas dataframe and its mapping information.
        """
        ratings = data.get("ratings")

        user_ids = sorted(ratings["user_id"].unique())
        item_ids = sorted(ratings["movie_id"].unique())

        # mapping dictionary user_id, movie_id to ascending id
        user_id2idx = {
            id_: idx for (idx, id_) in enumerate(user_ids)
        }
        item_id2idx = {
            id_: idx for (idx, id_) in enumerate(item_ids)
        }

        # mapping ids
        ratings["user_id"] = ratings["user_id"].map(user_id2idx)
        ratings["movie_id"] = ratings["movie_id"].map(item_id2idx)

        return {
            "ratings": ratings,
            "num_users": len(user_id2idx),
            "num_items": len(item_id2idx),
            "user_id2idx": user_id2idx,
            "item_id2idx": item_id2idx,
        }
