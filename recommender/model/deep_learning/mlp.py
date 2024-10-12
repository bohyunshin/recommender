import torch
import torch.nn as nn
import torch.nn.functional as F

from model.torch_model_base import TorchModelBase


class Model(TorchModelBase):
    def __init__(self, num_users, num_items, num_factors, **kwargs):
        super().__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.num_factors = num_factors

        self.embed_user = nn.Embedding(num_users, num_factors)
        self.embed_item = nn.Embedding(num_items, num_factors)

        self.layer1 = nn.Sequential(
            nn.Linear(num_factors, num_factors // 2),
            nn.ReLU()
        )
        num_factors = num_factors // 2
        self.layer2 = nn.Sequential(
            nn.Linear(num_factors, num_factors // 2),
            nn.ReLU()
        )
        num_factors = num_factors // 2
        self.layer3 = nn.Sequential(
            nn.Linear(num_factors, 1)
        )

    def forward(self, user_idx, item_idx):
        user_item_concat = torch.concat((self.embed_user(user_idx), self.embed_item(item_idx)), dim=1)
        x = self.layer1(user_item_concat)
        x = self.layer2(x)
        x = self.layer3(x)
        return F.sigmoid(x)
