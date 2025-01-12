from typing import Tuple, Dict, Union, Any

import pandas as pd
import torch

from recommender.prepare_model_data.prepare_model_data_base import PrepareModelDataBase
from libs.utils.csr import dataframe_to_csr
from recommender.libs.constant.prepare_model_data.prepare_model_data import MIN_REVIEWS


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
        shape = (self.num_users, self.num_items)
        ratings = data.get("ratings")

        # filter user_id whose number of reviews is lower than MIN_REVIEWS
        user2item_count = ratings["user_id"].value_counts().to_dict()
        user_id_min_reviews = [user_id for user_id, item_count in user2item_count.items() if item_count >= MIN_REVIEWS]
        ratings = ratings[lambda x: x["user_id"].isin(user_id_min_reviews)]

        # split train / validation
        train, val = self.split_train_validation(ratings=ratings)

        # for inference
        X_train = torch.tensor(train[["user_id", "movie_id"]].values)
        y_train = torch.tensor(train["rating"].values, dtype=torch.float32)

        X_val = torch.tensor(val[["user_id", "movie_id"]].values)
        y_val = torch.tensor(val["rating"].values, dtype=torch.float32)

        self.X_y = {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
        }

        csr_train = dataframe_to_csr(
            df=train,
            shape=shape,
            implicit=True,
        )
        csr_val = dataframe_to_csr(
            df=val,
            shape=shape,
            implicit=True,
        )
        return csr_train, csr_val