import torch
import torch.nn as nn
import torch.nn.functional as F

from recommender.model.torch_model_base import TorchModelBase


class Model(TorchModelBase):
    def __init__(
            self,
            user_ids: torch.Tensor,
            item_ids: torch.Tensor,
            num_users: int,
            num_items: int,
            num_factors: int,
            **kwargs,
        ):
        """
        Two-tower neural network passing user/item embedding on separate layers, finally concatenates them.

        Args:
            user_ids (torch.Tensor): List of user_id.
            item_ids (torch.Tensor): List of item_id.
            num_users (int): Number of users. Should match with dimension of user_ids.
            num_items (int): Number of items. Should match with dimension of item_ids.
            num_factors (int): Embedding dimension for user, item embeddings.
        """
        super().__init__(
            user_ids=user_ids,
            item_ids=item_ids,
            num_users=num_users,
            num_items=num_items,
            num_factors=num_factors,
            **kwargs
        )

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

    def forward(
            self,
            user_idx: torch.Tensor,
            item_idx: torch.Tensor,
            **kwargs,
        ) -> torch.Tensor:
        """
        Calculates associated probability between user_idx and item_idx using two-tower architecture.

        [Forward step]
        1. Concatenates user/item and associated meta embeddings.
          - For meta embeddings, these could be one-hot encoded vector.
        2. Pass concatenated embeddings to pre-defined user/item tower.
          - Note that user/item towers are separately defined, therefore,
          concatenated vectors are trained on separate layers.
        3. Dot products between user and item embeddings from separate two-tower layers.
        4. Finally pass to sigmoid layer.


        Args:
            user_idx (torch.Tensor): User index.
            item_idx (torch.Tensor): Item index.

        Returns (torch.Tensor):
            Probability between user_idx and item_idx.
        """
        user = torch.concat((self.embed_user(user_idx), self.user_meta[user_idx]), axis=1)
        user = self.user_layers(user)

        item = torch.concat((self.embed_item(item_idx), self.item_meta[item_idx]), axis=1)
        item = self.item_layers(item)

        x = (user * item).sum(dim=1)
        x = F.sigmoid(x)
        return x

    def create_sequential_layer(
            self,
            input_dim,
            num_layers=3,
            last_dim=16
        ) -> nn.Sequential:
        """
        Creates sequential layers for each of user and item layers.

        Args:
            input_dim (int): Input dimension.
            num_layers (int): Number of layers.
            last_dim (int): Dimension of last layer just before sigmoid layer.

        Returns (nn.Sequential):
            Sequential neural network layers used in two-tower architecture.
        """
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