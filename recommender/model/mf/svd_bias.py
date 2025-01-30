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
        SVD model decomposing user x item matrix which uses user, item bias term.
        This model is famous for the history that it was used in netflix competition.

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
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.mu = kwargs["mu"]

        nn.init.xavier_normal_(self.embed_user.weight)
        nn.init.xavier_normal_(self.embed_item.weight)
        nn.init.xavier_normal_(self.user_bias.weight)
        nn.init.xavier_normal_(self.item_bias.weight)

    def forward(
        self,
        user_idx: torch.Tensor,
        item_idx: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass of SVD model.
        After getting user and item embedding, dot product them to get prediction score.
        Finally, add associated bias term of batch user and item.
        Note that grand mean of training dataset is also included as `self.mu`.

        Args:
            user_idx (torch.Tensor): Batch user index.
            item_idx (torch.Tensor): Batch item index.

        Returns (torch.Tensor):
            Prediction score with batch_size dimension.
        """
        embed_user = self.embed_user(user_idx)  # batch_size * num_factors
        embed_item = self.embed_item(item_idx)  # batch_size * num_factors
        user_bias = self.user_bias(user_idx)  # batch_size * 1
        item_bias = self.item_bias(item_idx)  # batch_size * 1
        output = (
            (embed_user * embed_item).sum(axis=1)
            + user_bias.squeeze()
            + item_bias.squeeze()
            + self.mu
        )  # batch_size * 1
        return output
