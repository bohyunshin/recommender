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
        pass
