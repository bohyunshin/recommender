from abc import abstractmethod
import numpy as np
import torch
from torch import nn

from model.recommender_base import RecommenderBase


class TorchModelBase(nn.Module, RecommenderBase):
    def __init__(self):
        super().__init__()

    def predict(self, user_idx, **kwargs):
        item_idx = kwargs["item_idx"]
        num_users = len(user_idx)
        num_items = len(item_idx)
        user_idx = torch.tensor(np.repeat(user_idx, num_items))
        item_idx = torch.tensor(np.tile(item_idx, num_users))
        with torch.no_grad():
            user_item_score = self.forward(user_idx, item_idx).detach().numpy()
        return user_item_score.reshape(-1, num_items)

    def set_trained_embedding(self):
        self.user_factors = self.embed_user.weight.data.clone().detach().cpu().numpy()
        self.item_factors = self.embed_item.weight.data.clone().detach().cpu().numpy()
