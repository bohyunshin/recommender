from abc import ABC, abstractmethod
from typing import Tuple, Any, Dict, Union

import pandas as pd
from sklearn.model_selection import train_test_split

from recommender.libs.constant.prepare_model_data.prepare_model_data import STRATIFY_COLUMN
from recommender.libs.constant.data import DatasetName


class PrepareModelDataBase(ABC):
    def __init__(
            self,
            model: str,
            num_users: int,
            num_items: int,
            train_ratio: float,
            implicit: bool,
            random_state: int,
            **kwargs):
        """
        Abstract class for making dataset for specific model.
        """
        self.model = model
        self.num_users = num_users
        self.num_items = num_items
        self.train_ratio = train_ratio
        self.implicit = implicit
        self.random_state = random_state

    def split_train_validation(self, ratings: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split rating dataframe using scikit-learn function.
        Stratified column name will be given depending on dataset.

        Args:
            ratings (pd.DataFrame): Rating dataframe consisting of user_id and item_id.

        Returns (Tuple[pd.DataFrame, pd.DataFrame]):
            Tuple of train and validation dataframes.
        """
        train, validation = train_test_split(
            ratings,
            test_size=1 - self.train_ratio,
            random_state=self.random_state,
            stratify=ratings[STRATIFY_COLUMN[DatasetName.MOVIELENS.value]],
        )
        return train, validation

    @abstractmethod
    def get_train_validation_data(
            self,
            data: Dict[str, Union[pd.DataFrame, Dict[int, int]]],
        ) -> Tuple[Any, Any]:
        """
        Split data into train and validation in corresponding format for model.

        Different data formats are required depending on which models are run.
        For example, numpy type will be appropriate for scikit-learn based models.
        Or data_loader from torch library will be appropriate for torch based models.

        This abstractmethod aims for splitting data into train and validation in the
        preferred data format depending on models.

        Args:
            data (Dict[str, Union[pd.DataFrame, Dict[int, int]]]):
                Return value from `recommender/preprocess/preprocess_base/PreoprocessorBase.preprocess`
                will be used.

        Returns (Tuple[Any, Any]):
            Train, validation dataset in order. Format could be numpy, data_loader or csr.
        """
        raise NotImplementedError
