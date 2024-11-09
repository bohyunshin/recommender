import torch
import torch.nn as nn
import torch.nn.functional as F

from model.torch_model_base import TorchModelBase


class Model(TorchModelBase):
    def __init__(self, num_users, num_items, num_factors, **kwargs):
        super().__init__()

        self.user_embedding = nn.Embedding(num_users, num_factors)
        self.item_embedding = nn.Embedding(num_items, num_factors)
        self.user_meta = kwargs["user_meta"]
        self.item_meta = kwargs["item_meta"]

        nn.init.xavier_normal_(self.user_embedding.weight)
        nn.init.xavier_normal_(self.item_embedding.weight)

        user_input_dim = num_factors + self.user_meta.shape[1]
        item_input_dim = num_factors + self.item_meta.shape[1]

        self.user_layers = self.create_sequential_layer(user_input_dim)
        self.item_layers = self.create_sequential_layer(item_input_dim)

        user_output_dim = self.user_layers[-2].out_features
        item_output_dim = self.item_layers[-2].out_features
        self.h = nn.Linear(user_output_dim+item_output_dim, 1, bias=False)

    def forward(self, user_idx, item_idx):
        user = torch.concat((self.user_embedding(user_idx), self.user_meta[user_idx]), axis=1)
        user = self.user_layers(user)

        item = torch.concat((self.item_embedding(item_idx), self.item_meta[item_idx]), axis=1)
        item = self.item_layers(item)

        concat = self.h(torch.concat((user, item), axis=1))

        return F.sigmoid(concat)

    def create_sequential_layer(self, input_dim, num_layers=3):
        layers = []
        for _ in range(num_layers):
            output_dim = input_dim // 2
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.ReLU())
            input_dim = output_dim
        return nn.Sequential(*layers)