import torch
import torch.nn as nn

from recommender.model.torch_model_base import TorchModelBase


class Model(TorchModelBase):

    def __init__(
            self,
            num_users: int,
            num_items: int,
            num_factors: int,
            **kwargs
        ):
        """
        SVD model decomposing user x item matrix.
        This model does not use user, item bias term and only trains their embeddings.

        Args:
            num_users (int): Number of users.
            num_items (int): Number of items.
            num_factors (int): Number of factors.
        """
        super(Model, self).__init__()

        self.embed_user = nn.Embedding(num_users, num_factors)
        self.embed_item = nn.Embedding(num_items, num_factors)

        nn.init.xavier_normal_(self.embed_user.weight)
        nn.init.xavier_normal_(self.embed_item.weight)

    def forward(
            self,
            user_idx: torch.Tensor,
            item_idx:torch.Tensor,
        ) -> torch.Tensor:
        """
        Forward pass of SVD model.
        After getting user and item embedding, dot product them to get prediction score.

        Args:
            user_idx (torch.Tensor): Batch user index.
            item_idx (torch.Tensor): Batch item index.

        Returns (torch.Tensor):
            Prediction score with batch_size dimension.
        """
        embed_user = self.embed_user(user_idx) # batch_size * num_factors
        embed_item = self.embed_item(item_idx) # batch_size * num_factors
        output = (embed_user * embed_item).sum(axis=1) # batch_size * 1
        return output