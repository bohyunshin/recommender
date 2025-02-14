from abc import ABC, abstractmethod
from typing import Dict, Union

import pandas as pd

from recommender.libs.constant.data.name import Field


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
        ratings = data.get(Field.INTERACTION.value)

        user_ids = sorted(ratings[Field.USER_ID.value].unique())
        item_ids = sorted(ratings[Field.ITEM_ID.value].unique())

        # mapping dictionary user_id, movie_id to ascending id
        user_id2idx = {id_: idx for (idx, id_) in enumerate(user_ids)}
        item_id2idx = {id_: idx for (idx, id_) in enumerate(item_ids)}

        # mapping ids
        ratings[Field.USER_ID.value] = ratings[Field.USER_ID.value].map(user_id2idx)
        ratings[Field.ITEM_ID.value] = ratings[Field.ITEM_ID.value].map(item_id2idx)

        return {
            Field.INTERACTION.value: ratings,
            Field.NUM_USERS.value: len(user_id2idx),
            Field.NUM_ITEMS.value: len(item_id2idx),
            Field.USER_ID2IDX.value: user_id2idx,
            Field.ITEM_ID2IDX.value: item_id2idx,
        }
