import logging
from typing import Any, Dict, Tuple, Union

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from recommender.libs.constant.prepare_model_data.prepare_model_data import MIN_REVIEWS
from recommender.libs.constant.data.name import Field
from recommender.libs.torch_dataset.dataset import Data
from recommender.libs.utils.user_item_count import convert_tensor_to_user_item_summary
from recommender.prepare_model_data.prepare_model_data_base import PrepareModelDataBase


class PrepareModelDataTorch(PrepareModelDataBase):
    def __init__(
        self,
        model: str,
        num_users: int,
        num_items: int,
        train_ratio: float,
        num_negative_samples: int,
        implicit: bool,
        random_state: int,
        batch_size: int,
        device: str,
        **kwargs,
    ):
        """
        Prepare train / validation torch data_loader.
        """
        super().__init__(
            model=model,
            num_users=num_users,
            num_items=num_items,
            train_ratio=train_ratio,
            implicit=implicit,
            random_state=random_state,
            **kwargs,
        )
        self.num_negative_samples = num_negative_samples
        self.batch_size = batch_size
        self.device = device

    def get_train_validation_data(
        self, data: Dict[str, Union[pd.DataFrame, Dict[int, int]]]
    ) -> Tuple[Any, Any]:
        """
        Getting pandas dataframe, make train / validation torch data_loader.

        First, split dataset into train / validation and separate feature / target by X, y.
        Second, make torch dataset. Depending on chosen model, modified dataset will be used.
        For example, for some models, negative sampling is required.
        See recommender/libs/torch_dataset for more details.
        Finally, split torch dataset into train and validation and make them as torch data_loader.

        Args:
            data (Dict[str, Union[pd.DataFrame, Dict[int, int]]]):
                Return value from `recommender/preprocess/preprocess_base/PreoprocessorBase.preprocess`
                will be used.

        Returns (Tuple[Any, Any]):
            Train, validation dataset in order. Format could be numpy, data_loader or csr.
        """

        # get feature / target tensor
        train_val_tensors = self.get_X_y_train_validation(data=data)

        # make torch dataset
        torch_dataset = self.get_torch_dataset(
            X_train=train_val_tensors.get(Field.X_TRAIN.value),
            y_train=train_val_tensors.get(Field.Y_TRAIN.value),
            X_val=train_val_tensors.get(Field.X_VAL.value),
            y_val=train_val_tensors.get(Field.Y_VAL.value),
        )

        # get train / validation data loader
        train_dataloader, validation_dataloader = self.get_torch_data_loader(
            train_dataset=torch_dataset.get(Field.TRAIN.value),
            val_dataset=torch_dataset.get(Field.VAL.value),
        )

        return train_dataloader, validation_dataloader

    def get_X_y_train_validation(
        self,
        data: Dict[str, Union[pd.DataFrame, Dict[int, int]]],
    ) -> Dict[str, torch.Tensor]:
        """
        Generates one-hot encoded metadata and returns (X, y) tensor.
        X includes interaction information with user_id and movie_id.
        y includes true rating value.

        Args:
            data (Dict[str, Union[pd.DataFrame, Dict[int, int]]]):
                Return value from `recommender/preprocess/preprocess_base/PreoprocessorBase.preprocess`
                will be used.

        Returns (Tuple[torch.Tensor, torch.Tensor]):
            Input and target tensors.
        """
        ratings = data.get(Field.INTERACTION.value)

        # filter user_id whose number of reviews is lower than MIN_REVIEWS
        user2item_count = ratings[Field.USER_ID.value].value_counts().to_dict()
        user_id_min_reviews = [
            user_id
            for user_id, item_count in user2item_count.items()
            if item_count >= MIN_REVIEWS
        ]
        ratings = ratings[lambda x: x[Field.USER_ID.value].isin(user_id_min_reviews)]

        # split train / validation
        train, val = self.split_train_validation(ratings=ratings)

        X_train = torch.tensor(train[[Field.USER_ID.value, Field.ITEM_ID.value]].values)
        y_train = torch.tensor(train[Field.INTERACTION.value].values, dtype=torch.float32)

        X_val = torch.tensor(val[[Field.USER_ID.value, Field.ITEM_ID.value]].values)
        y_val = torch.tensor(val[Field.INTERACTION.value].values, dtype=torch.float32)

        self.user_item_summ_tr = convert_tensor_to_user_item_summary(
            ts=X_train,
            structure=dict,
        )
        self.user_item_summ_tr_val = convert_tensor_to_user_item_summary(
            ts=torch.concat([X_train, X_val], dim=0),
            structure=dict,
        )

        self.X_y = {
            Field.X_TRAIN.value: X_train,
            Field.Y_TRAIN.value: y_train,
            Field.X_VAL.value: X_val,
            Field.Y_VAL.value: y_val,
        }

        return self.X_y

    def get_torch_dataset(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
    ) -> Dict[str, Dataset]:
        """
        Make torch dataset to be used for torch data_loader.

        Different torch dataset will be made depending on chosen model.
        See recommender/libs/torch_dataset for more details.

        Args:
            X_train (torch.Tensor): Input tensors for train step. Usually, user_id and item_id.
            y_train (torch.Tensor): Target tensors for train step. Usually, rating value.
            X_val (torch.Tensor): Input tensors for validation step. Usually, user_id and item_id.
            y_val (torch.Tensor): Target tensors for validation step. Usually, rating value.

        Returns (Dataset):
            Torch dataset.
        """
        tensors = {
            Field.TRAIN.value: (X_train, y_train),
            Field.VAL.value: (X_val, y_val),
        }
        torch_dataset = {}

        for name, (X, y) in tensors.items():
            dataset = Data(
                X=X,
                y=y,
            )
            torch_dataset[name] = dataset

        return torch_dataset

    def get_torch_data_loader(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Make train / validation torch data_loader.

        Args:
            train_dataset (Dataset): Torch dataset from `get_torch_dataset` method
                will be used as `dataset` argument.

        Returns (Tuple[DataLoader, DataLoader]):
            Train / validation torch data_loader in order.
        """
        seed = torch.Generator(device=self.device)
        logging.info(f"Torch dataloader device: {self.device}")
        self.train_dataset = train_dataset
        self.validation_dataset = val_dataset
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=seed,
        )
        validation_dataloader = DataLoader(
            dataset=val_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=seed,
        )
        self.mu = train_dataset.y.mean() if self.model in ["svd", "svd_bias"] else None

        return train_dataloader, validation_dataloader
