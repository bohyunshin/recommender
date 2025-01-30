import importlib
import logging
import time
from typing import Any, Dict, Tuple, Union

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from recommender.libs.constant.prepare_model_data.prepare_model_data import MIN_REVIEWS
from recommender.libs.constant.torch.dataset import DATASET_PATH
from recommender.libs.constant.torch.device import DEVICE
from recommender.libs.torch_dataset.dataset import Data
from recommender.libs.utils.user_item_count import convert_tensor_to_user_item_summary
from recommender.libs.utils.utils import mapping_dict
from recommender.prepare_model_data.prepare_model_data_base import PrepareModelDataBase

# in case cuda, following error occurs.
# RuntimeError: Expected a 'cpu' device type for generator but found 'cuda'
# it seems that when setting `_base_seed`, device setting in `torch.empty()` does not work.
torch.set_default_device(DEVICE)


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
        user_meta: pd.DataFrame,
        item_meta: pd.DataFrame,
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
        self.user_meta = self.get_user_meta(users=user_meta)
        self.item_meta = self.get_item_meta(items=item_meta)

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
            X_train=train_val_tensors.get("X_train"),
            y_train=train_val_tensors.get("y_train"),
            X_val=train_val_tensors.get("X_val"),
            y_val=train_val_tensors.get("y_val"),
        )

        # get train / validation data loader
        train_dataloader, validation_dataloader = self.get_torch_data_loader(
            train_dataset=torch_dataset.get("train"),
            val_dataset=torch_dataset.get("val"),
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
        ratings = data.get("ratings")
        users = data.get("users")
        items = data.get("items")

        # filter user_id whose number of reviews is lower than MIN_REVIEWS
        user2item_count = ratings["user_id"].value_counts().to_dict()
        user_id_min_reviews = [
            user_id
            for user_id, item_count in user2item_count.items()
            if item_count >= MIN_REVIEWS
        ]
        ratings = ratings[lambda x: x["user_id"].isin(user_id_min_reviews)]

        # get one-hot encoded metadata
        self.user_meta = self.get_user_meta(users)
        self.item_meta = self.get_item_meta(items)

        # split train / validation
        train, val = self.split_train_validation(ratings=ratings)

        X_train = torch.tensor(train[["user_id", "movie_id"]].values)
        y_train = torch.tensor(train["rating"].values, dtype=torch.float32)

        X_val = torch.tensor(val[["user_id", "movie_id"]].values)
        y_val = torch.tensor(val["rating"].values, dtype=torch.float32)

        self.user_item_summ_tr = convert_tensor_to_user_item_summary(
            ts=X_train,
            structure=dict,
        )
        self.user_item_summ_tr_val = convert_tensor_to_user_item_summary(
            ts=torch.concat([X_train, X_val], dim=0),
            structure=dict,
        )

        self.X_y = {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
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
            "train": (X_train, y_train),
            "val": (X_val, y_val),
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
        seed = torch.Generator(device=DEVICE)
        logging.info(f"Torch dataloader device: {DEVICE}")
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

    def get_user_meta(
        self,
        users: pd.DataFrame,
    ) -> torch.Tensor:
        """
        Convert user meta dataframe into one-hot encoded tensor.

        Args:
            users (pd.DataFrame): Input metadata for users

        Returns (torch.Tensor):
            Converted one-hot encoded tensor.
        """
        user_meta_cols = ["gender", "age", "occupation"]
        user_meta = torch.tensor([])
        for col in user_meta_cols:
            vals = users[col].tolist()
            mapping = mapping_dict(vals)
            num_classes = len(mapping)
            vals = [mapping[val] for val in vals]
            one_hot_vector = F.one_hot(torch.tensor(vals), num_classes=num_classes)
            user_meta = torch.concat((user_meta, one_hot_vector), dim=1)
        return user_meta

    def get_item_meta(
        self,
        items: pd.DataFrame,
    ) -> torch.Tensor:
        """
        Convert item meta dataframe into one-hot encoded tensor.

        Args:
            items (pd.DataFrame): Input metadata for items

        Returns (torch.Tensor):
            Converted one-hot encoded tensor.
        """
        genres = items["genres"].map(lambda x: x.split("|")).tolist()
        unique_genres = set()
        for genre in genres:  # genre: ["Animation", "Action"]
            for g in genre:
                unique_genres.add(g)
        unique_genres = sorted(list(unique_genres))
        mapping_genres = mapping_dict(unique_genres)
        num_classes = len(mapping_genres)

        movie_meta = torch.tensor([])
        for genre in genres:
            genre_vector_for_one_movie = torch.tensor([0] * num_classes)
            for g in genre:
                g_encoded = mapping_genres[g]
                genre_vector_for_one_movie = genre_vector_for_one_movie + F.one_hot(
                    torch.tensor([g_encoded]), num_classes=num_classes
                )
            movie_meta = torch.concat((movie_meta, genre_vector_for_one_movie), dim=0)
        return movie_meta
