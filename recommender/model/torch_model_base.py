import numpy as np
from numpy.typing import NDArray
import torch
from torch import nn

from model.recommender_base import RecommenderBase


class TorchModelBase(nn.Module, RecommenderBase):
    def __init__(self):
        """
        Abstract base class for torch based models.
        """
        super().__init__()

    def predict(
            self,
            user_idx: NDArray,
            **kwargs,
        ) -> NDArray:
        """
        For batch users, calculates prediction score for all of item ids.
        In inference pipeline, `kwargs["item_idx"]` will be all of item ids.
        Using `forward` method in torch model, batch_sie x num_items score matrix will be created.

        Args:
            user_idx (NDArray): User ids.

        Returns (NDArray):
            Batch_size x num_items score matrix.
        """
        item_idx = kwargs["item_idx"]
        num_users = len(user_idx)
        num_items = len(item_idx)
        user_idx = torch.tensor(np.repeat(user_idx, num_items))
        item_idx = torch.tensor(np.tile(item_idx, num_users))
        with torch.no_grad():
            user_item_score = self.forward(user_idx, item_idx).detach().cpu().numpy()
        return user_item_score.reshape(-1, num_items)
