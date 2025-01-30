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
        MLP neural network getting input as concatenation of user and item embeddings.

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

    def forward(
            self,
            user_idx: torch.Tensor,
            item_idx: torch.Tensor,
            **kwargs,
        ) -> torch.Tensor:
        """
        Calculates associated probability between user_idx and item_idx using mlp architecture.

        [Forward step]
        1. Concatenates user and item embeddings
        2. Pass linear layers.
        3. Finally, pass sigmoid function which is done in BCEWithLogitsLoss.

        Args:
            user_idx (torch.Tensor): User index.
            item_idx (torch.Tensor): Item index.

        Returns (torch.Tensor):
            Probability between user_idx and item_idx.
        """
        x = torch.concat((self.embed_user(user_idx), self.embed_item(item_idx)), dim=1)
        for layer in self.layers:
            x = layer(x)
        x = self.h(x)
        return x
