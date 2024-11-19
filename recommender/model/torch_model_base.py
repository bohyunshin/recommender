from abc import abstractmethod
import numpy as np
from torch import nn

from model.recommender_base import RecommenderBase


class TorchModelBase(nn.Module, RecommenderBase):
    def __init__(self):
        super().__init__()

    def set_trained_embedding(self):
        self.user_factors = self.embed_user.weight.data.clone().detach().cpu().numpy()
        self.item_factors = self.embed_item.weight.data.clone().detach().cpu().numpy()
