import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from model.torch_model_base import TorchModelBase

class Model(TorchModelBase):
    def __init__(self, num_users, num_items, num_factors, **kwargs):
        super(Model, self).__init__()

        self.embed_user = nn.Embedding(num_users, num_factors)
        self.embed_item = nn.Embedding(num_items, num_factors)

        # nn.init.xavier_normal_(self.embed_user.weight)
        # nn.init.xavier_normal_(self.embed_item.weight)

    def forward(self, user_idx, item_idx):
        return F.sigmoid((self.embed_user(user_idx) * self.embed_item(item_idx)).sum(axis=1))