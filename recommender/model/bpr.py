from typing import Union

import numpy as np
from numpy.typing import NDArray

import torch
from torch import nn

from recommender.model.torch_model_base import TorchModelBase


class Model(TorchModelBase):
    def __init__(
            self,
            user_ids: torch.Tensor,
            item_ids: torch.Tensor,
            num_users: int,
            num_items: int,
            num_factors: int,
            **kwargs
        ):
        """
        Bayesian personalized ranking model using triplet loss.
        In the original paper, bpr loss is used in any forms of model, including
        matrix factorization or deep learning based models.
        This class applies bpr loss to simple matrix factorization model.
        Using bpr loss dynamically with various models is todo.

        Args:
            num_users (int): Number of users.
            num_items (int): Number of items.
            num_factors (int): Number of factors.
        """
        super().__init__(
            user_ids=user_ids,
            item_ids=item_ids,
            num_users=num_users,
            num_items=num_items,
            num_factors=num_factors,
            **kwargs
        )

        self.embed_user = nn.Embedding(num_users, num_factors)
        self.embed_item = nn.Embedding(num_items, num_factors)

        nn.init.xavier_normal_(self.embed_user.weight)
        nn.init.xavier_normal_(self.embed_item.weight)

    def forward(
            self,
            user_idx: torch.Tensor,
            pos_item_idx: torch.Tensor,
            neg_item_idx: torch.Tensor,
        ) -> torch.Tensor:
        """
        Forward pass for bpr loss model.
        This functions get 3 arguments, user id, positive item id and negative item id,
        because bpr uses triplet loss with negative samples.
        Note that `neg_item_idx` is already sampled before using negative sampling logic.

        Args:
             user_idx (torch.Tensor): User index.
             pos_item_idx (torch.Tensor): Positive item index.
             neg_item_idx (torch.Tensor): Negative item index.

        Returns (torch.Tensor):
            Prediction scores indicating strength of likeness of positive item over negative item.
            If this score is passed to sigmoid layer, we can interpret it as the probability
            that user likes positive item more than negative item.
        """
        embed_user = self.embed_user(user_idx)  # batch_size * num_factors
        embed_pos_item = self.embed_item(pos_item_idx)  # batch_size * num_factors
        embed_neg_item = self.embed_item(neg_item_idx)  # batch_size * num_factors
        output = (embed_user * (embed_pos_item - embed_neg_item)).sum(axis=1)  # batch_size * 1
        return output

    def predict(
            self,
            user_id: Union[NDArray, torch.Tensor],
            item_id: Union[NDArray, torch.Tensor],
            **kwargs,
    ) -> Union[NDArray, torch.Tensor]:
        """
        Predicts users' ratings (or preference, scores) based on factorized user_factors and item_factors.
        For matrix factorization models, this could be dot product between user_factors and item_factors.
        For deep learning models, this could be inference step fed with user/item information to neural network

        Args:
            user_id (Union[NDArray, torch.Tensor]): Set of user_ids who are recommendation target.
                Typically, batch user_ids will be given.
            item_id (Union[NDArray, torch.Tensor]): Set of item_ids to calculate scores.
                Typically, all item_ids will be given because all scores should be cauclated with one user.

        Returns (Union[NDArray, torch.Tensor]):
            User x item prediction score matrix.
        """
        embed_users = self.embed_user(user_id)
        embed_items = self.embed_item(item_id)
        scores = torch.mm(embed_users, embed_items.t())
        return scores