from typing import Tuple, Dict, Union, Any
import pandas as pd

from recommender.prepare_model_data.prepare_model_data_base import PrepareModelDataBase
from recommender.libs.csr import dataframe_to_csr
from recommender.preprocess.train_test_split import train_test_split


class PrepareModelDataCsr(PrepareModelDataBase):
    def __init__(
            self,
            model: str,
            num_users: int,
            num_items: int,
            train_ratio: float,
            num_negative_samples: int,
            implicit: bool,
            random_state: int,
            **kwargs,
        ):
        super().__init__(
            model=model,
            num_users=num_users,
            num_items=num_items,
            train_ratio=train_ratio,
            num_negative_samples=num_negative_samples,
            implicit=implicit,
            random_state=random_state,
            **kwargs
        )

    def get_train_validation_data(
            self,
            data: Dict[str, Union[pd.DataFrame, Dict[int, int]]]
        ) -> Tuple[Any, Any]:
        """
                Split rating data into train / validation dataset, in csr_matrix format.

                Returns (Tuple[csr_matrix, csr_matrix]):
                    Tuple of train / validation dataset in csr_matrix.
                """
        ratings = data.get("ratings")
        shape = (self.num_users, self.num_items)
        csr_train, csr_val = train_test_split(
            ratings=dataframe_to_csr(
                df=ratings,
                shape=shape,
                implicit=True
            ),
            train_percentage=self.train_ratio,
            random_state=self.random_state
        )
        return csr_train, csr_val