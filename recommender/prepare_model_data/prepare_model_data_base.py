from abc import ABC, abstractmethod
from typing import Tuple, Any, Dict, Union

import pandas as pd


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
