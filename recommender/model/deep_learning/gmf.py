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
        Generalized matrix factorization model.
        In basic mf, no linear layer exists before sigmoid layer.
        In generalized mf, linear layer exists before sigmoid layer.
        This permits more flexible modeling between input and output.

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
        self.h = nn.Linear(num_factors, 1, bias=False)

        nn.init.xavier_normal_(self.embed_user.weight)
        nn.init.xavier_normal_(self.embed_item.weight)
        nn.init.xavier_normal_(self.h.weight)

    def forward(
        self,
        user_idx: torch.Tensor,
        item_idx: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Calculates associated probability between user_idx and item_idx using gmf architecture.

        [Forward step]
        1. Dot products between user and item embeddings
        2. Pass dot products values to linear layer
        3. Finally, pass sigmoid function which is done in BCEWithLogitsLoss.

        Args:
            user_idx (torch.Tensor): User index.
            item_idx (torch.Tensor): Item index.

        Returns (torch.Tensor):
            Probability between user_idx and item_idx.
        """
        x = self.embed_user(user_idx) * self.embed_item(item_idx)
        x = self.h(x)
        return x
