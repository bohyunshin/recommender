import torch
import torch.nn as nn

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
            **kwargs,
        )

        self.embed_user = nn.Embedding(num_users, num_factors)
        self.embed_item = nn.Embedding(num_items, num_factors)

        nn.init.xavier_normal_(self.embed_user.weight)
        nn.init.xavier_normal_(self.embed_item.weight)

        self.user_layers = self.create_sequential_layer(num_factors)
        self.item_layers = self.create_sequential_layer(num_factors)

    def forward(
        self,
        user_idx: torch.Tensor,
        item_idx: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Calculates associated probability between user_idx and item_idx using two-tower architecture.

        [Forward step]
        1. Pass user and item embeddings to pre-defined user/item tower.
          - Note that user/item towers are separately defined, therefore,
          concatenated vectors are trained on separate layers.
        2. Dot products between user and item embeddings from separate two-tower layers.

        Note that there are not any sigmoid layer in final layer to open flexible loss function.


        Args:
            user_idx (torch.Tensor): User index.
            item_idx (torch.Tensor): Item index.

        Returns (torch.Tensor):
            Probability between user_idx and item_idx.
        """
        user = self.user_layers(self.embed_user(user_idx))
        item = self.item_layers(self.embed_item(item_idx))
        x = (user * item).sum(dim=1)
        return x

    def create_sequential_layer(
        self, input_dim, last_dim=4
    ) -> nn.Sequential:
        """
        Creates sequential layers for each of user and item layers.

        Args:
            input_dim (int): Input dimension.
            last_dim (int): Dimension of last layer just before sigmoid layer.

        Returns (nn.Sequential):
            Sequential neural network layers used in two-tower architecture.
        """
        if input_dim <= last_dim:
            raise ValueError(f"input dimension {input_dim} should be larger than last layer dimension {last_dim}")
        layers = []
        while True:
            if input_dim // 2 < last_dim:
                output_dim = last_dim
            else:
                output_dim = input_dim // 2
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.ReLU())
            input_dim = output_dim

            if output_dim == last_dim:
                break
        return nn.Sequential(*layers)
