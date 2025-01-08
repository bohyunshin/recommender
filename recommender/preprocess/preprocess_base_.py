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
    def preprocess(self, data: Dict[str, pd.DataFrame]):
        """
        Abstract method for preprocssing data.
        """
        raise NotImplementedError

    def mapping(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Union[pd.DataFrame, Dict[int, int]]]:
        ratings = data.get("ratings")
        users = data.get("users")
        items = data.get("items")
        # mapping dictionary user_id, movie_id to ascending id
        user_id2idx = {id_: idx for (idx, id_) in enumerate(sorted(users["user_id"].unique()))}
        item_id2idx = {id_: idx for (idx, id_) in enumerate(sorted(items["movie_id"].unique()))}

        # id in users and movies should be same ascending order with mapping dictionary
        assert users["user_id"].tolist() == sorted(list(user_id2idx.keys()))
        assert items["movie_id"].tolist() == sorted(list(item_id2idx.keys()))

        # mapping ids
        ratings["user_id"] = ratings["user_id"].map(user_id2idx)
        ratings["movie_id"] = ratings["movie_id"].map(item_id2idx)
        users["user_id"] = users["user_id"].map(user_id2idx)
        items["movie_id"] = items["movie_id"].map(item_id2idx)

        return {
            "ratings": ratings,
            "users": users,
            "items": items,
            "num_users": len(user_id2idx),
            "num_items": len(item_id2idx),
            "user_id2idx": user_id2idx,
            "item_id2idx": item_id2idx,
        }