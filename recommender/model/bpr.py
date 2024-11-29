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

    def predict(self, user_idx, **kwargs):
        embed_users = self.embed_user.weight[user_idx].detach().cpu().numpy()
        embed_items = self.embed_item.weight.detach().cpu().numpy()
        return np.dot(embed_users, embed_items.T)