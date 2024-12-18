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

        num_factors = num_factors * 2 # concat user & item embedding
        layers = []
        num_layers = 3
        for i in range(num_layers):
            output_dim = num_factors // 2
            layers.append(nn.Linear(num_factors, output_dim))
            layers.append(nn.ReLU())
            num_factors = output_dim
        self.layers = nn.Sequential(*layers)
        self.h = nn.Linear(num_factors, 1, bias=False)

        nn.init.xavier_normal_(self.embed_user.weight)
        nn.init.xavier_normal_(self.embed_item.weight)
        for layer in self.layers:
            if getattr(layer, "weight", None) != None:
                nn.init.xavier_normal_(layer.weight)
        nn.init.xavier_normal_(self.h.weight)

    def forward(self, user_idx, item_idx):
        x = torch.concat((self.embed_user(user_idx), self.embed_item(item_idx)), dim=1)
        for layer in self.layers:
            x = layer(x)
        x = self.h(x)
        return F.sigmoid(x)
