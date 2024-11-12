import torch
import torch.nn as nn
import torch.nn.functional as F

from model.torch_model_base import TorchModelBase


class Model(TorchModelBase):
    def __init__(self, num_users, num_items, num_factors, **kwargs):
        super().__init__()

        self.embed_user = nn.Embedding(num_users, num_factors)
        self.embed_item = nn.Embedding(num_items, num_factors)
        self.user_meta = kwargs["user_meta"]
        self.item_meta = kwargs["item_meta"]

        nn.init.xavier_normal_(self.embed_user.weight)
        nn.init.xavier_normal_(self.embed_item.weight)

        user_input_dim = num_factors + self.user_meta.shape[1]
        item_input_dim = num_factors + self.item_meta.shape[1]

        self.user_layers = self.create_sequential_layer(user_input_dim)
        self.item_layers = self.create_sequential_layer(item_input_dim)

        user_output_dim = self.user_layers[-2].out_features
        item_output_dim = self.item_layers[-2].out_features
        self.h = nn.Linear(user_output_dim+item_output_dim, 1, bias=False)

    def forward(self, user_idx, item_idx):
        user = torch.concat((self.embed_user(user_idx), self.user_meta[user_idx]), axis=1)
        user = self.user_layers(user)

        item = torch.concat((self.embed_item(item_idx), self.item_meta[item_idx]), axis=1)
        item = self.item_layers(item)

        x = (user * item).sum(dim=1)
        x = F.sigmoid(x)
        return x

    def create_sequential_layer(self, input_dim, num_layers=3, last_dim=16):
        layers = []
        for i in range(num_layers):
            if i == num_layers-1:
                output_dim = last_dim
            else:
                output_dim = input_dim // 2
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.ReLU())
            input_dim = output_dim
        return nn.Sequential(*layers)