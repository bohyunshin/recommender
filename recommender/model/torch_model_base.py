from typing import Union

from numpy.typing import NDArray

import torch
from torch import nn

from recommender.model.recommender_base import RecommenderBase


class TorchModelBase(nn.Module, RecommenderBase):
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
        Abstract base class for torch based models.

        Args:
            user_ids (torch.Tensor): List of user_id.
            item_ids (torch.Tensor): List of item_id.
            num_users (int): Number of users. Should match with dimension of user_ids.
            num_items (int): Number of items. Should match with dimension of item_ids.
            num_factors (int): Embedding dimension for user, item embeddings.
        """
        self.call_super_init = True
        super().__init__(
            user_ids=user_ids,
            item_ids=item_ids,
            num_users=num_users,
            num_items=num_items,
            num_factors=num_factors,
        )

    def predict(
            self,
            user_id: Union[NDArray, torch.Tensor],
            item_id: Union[NDArray, torch.Tensor],
            **kwargs,
        ) -> torch.Tensor:
        """
        For batch users, calculates prediction score for all of item ids.
        In inference pipeline, `kwargs["item_idx"]` will be all of item ids.
        Using `forward` method in torch model, batch_sie x num_items score matrix will be created.

        Args:
            user_id (Union[NDArray, torch.Tensor]): Set of user_ids who are recommendation target.
                Typically, batch user_ids will be given.
            item_id (Union[NDArray, torch.Tensor]): Set of item_ids to calculate scores.
                Typically, all item_ids will be given because all scores should be cauclated with one user.

        Returns (Union[NDArray, torch.Tensor]):
            Batch_size x num_items score matrix.
        """
        # user_id and item_id are torch.Tensor in torch based model
        num_batch_users = kwargs.get("num_batch_users")
        user_id = user_id.repeat_interleave(self.num_items)
        item_id = item_id.tile(num_batch_users)
        with torch.no_grad():
            user_item_score = self.forward(user_id, item_id)
        return user_item_score.reshape(-1, self.num_items)
