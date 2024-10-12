import torch
import numpy as np
from torch import nn

from model.torch_model_base import TorchModelBase


class Model(TorchModelBase):
    def __init__(self, num_users, num_items, num_factors, **kwargs):
        super().__init__()

        self.embed_user = nn.Embedding(num_users, num_factors)
        self.embed_item = nn.Embedding(num_items, num_factors)

        nn.init.xavier_normal_(self.embed_user.weight)
        nn.init.xavier_normal_(self.embed_item.weight)

    def forward(self, user_idx, pos_item_idx, neg_item_idx):
        embed_user = self.embed_user(user_idx)  # batch_size * num_factors
        embed_pos_item = self.embed_item(pos_item_idx)  # batch_size * num_factors
        embed_neg_item = self.embed_item(neg_item_idx)  # batch_size * num_factors
        output = (embed_user * (embed_pos_item - embed_neg_item)).sum(axis=1)  # batch_size * 1
        return output

    def predict(self, user_factors, item_factors, userid, **kwargs):
        """
        Calculate user-item scores based on learned user / item embeddings

        Parameters
        ----------
        user_factors : Tensor (M1 x K)

        item_factors : Tensor (N1 x K)

        Returns
        -------
        user_item_scores : Tensor (M1 x N1)
        """
        return np.dot(user_factors[userid], item_factors.T)