from typing import Tuple, Dict, Union, Any
import time
import importlib
import logging

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from recommender.prepare_model_data.prepare_model_data_base import PrepareModelDataBase
from recommender.libs.utils import mapping_dict
from recommender.libs.csr import implicit_to_csr
from recommender.libs.constant.torch.device import DEVICE
from recommender.libs.constant.torch.dataset import DATASET_PATH


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
        super().__init__(
            model=model,
            num_users=num_users,
            num_items=num_items,
            train_ratio=train_ratio,
            implicit=implicit,
            random_state=random_state,
            **kwargs
        )
        self.num_negative_samples = num_negative_samples
        self.batch_size = batch_size
        self.user_meta = self.get_user_meta(users=user_meta)
        self.item_meta = self.get_item_meta(items=item_meta)

    def get_train_validation_data(
            self,
            data: Dict[str, Union[pd.DataFrame, Dict[int, int]]]
        ) -> Tuple[Any, Any]:

        # get feature / target tensor
        X,y = self.get_X_y(data=data)

        # make torch dataset
        dataset = self.get_torch_dataset(
            X=X,
            y=y,
        )

        # get train / validation data loader
        train_dataloader, validation_dataloader = self.get_torch_data_loader(dataset=dataset)

        return train_dataloader, validation_dataloader

    def get_torch_dataset(
            self,
            X: torch.Tensor,
            y: torch.Tensor,
        ) -> Dataset:
        shape = self.num_users, self.num_items
        user_items_dct = implicit_to_csr(X, shape, True)

        dataset_args = {
            "X": X,
            "y": y,
            "user_items_dct": user_items_dct,
            "num_items": self.num_items,
            "num_neg": self.num_negative_samples,
        }

        dataset_path = DATASET_PATH.get(self.model)
        if dataset_path is None:
            raise
        dataset_module = importlib.import_module(dataset_path).Data
        dataset = dataset_module(**dataset_args)

        if self.implicit == True:
            start = time.time()
            dataset.negative_sampling()
            logging.info(f"token time for negative sampling: {time.time() - start}")

        return dataset

    def get_X_y(
            self,
            data: Dict[str, Union[pd.DataFrame, Dict[int, int]]],
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates one-hot encoded metadata and returns (X, y) tensor.
        X includes interaction information with user_id and movie_id.
        y includes true rating value.

        Returns (Tuple[torch.Tensor, torch.Tensor]):
            Input and target tensors.
        """
        ratings = data.get("ratings")
        users = data.get("users")
        items = data.get("items")
        self.user_meta = self.get_user_meta(users)
        self.item_meta = self.get_item_meta(items)

        X = torch.tensor(ratings[["user_id", "movie_id"]].values)
        y = torch.tensor(ratings[["rating"]].values, dtype=torch.float32)
        return X, y

    def get_torch_data_loader(self, dataset: Dataset) -> Tuple[DataLoader, DataLoader]:
        seed = torch.Generator(device=DEVICE.type).manual_seed(self.random_state)
        # split train / validation dataset
        train_dataset, validation_dataset = random_split(
            dataset=dataset,
            lengths=[self.train_ratio, 1 - self.train_ratio],
            generator=seed
        )
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=seed
        )
        validation_dataloader = DataLoader(
            dataset=validation_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=seed
        )
        self.mu = train_dataset.dataset.y[train_dataset.indices].mean() if self.model in ["svd", "svd_bias"] else None

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
            users (pd.DataFrame): Input metadata for items

        Returns (torch.Tensor):
            Converted one-hot encoded tensor.
        """
        genres = items["genres"].map(lambda x: x.split("|")).tolist()
        unique_genres = set()
        for genre in genres: # genre: ["Animation", "Action"]
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
                genre_vector_for_one_movie = genre_vector_for_one_movie + F.one_hot(torch.tensor([g_encoded]), num_classes=num_classes)
            movie_meta = torch.concat((movie_meta, genre_vector_for_one_movie), dim=0)
        return movie_meta