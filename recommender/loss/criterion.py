from typing import Optional

import torch
import torch.nn as nn

from loss.custom import bpr_loss, svd_loss


class Criterion:
    def __init__(self, model: str):
        """
        Selects appropriate loss function for given model, and calculates loss.

        Args:
            model (str): Model name to train.
        """
        self.model = model
        # TODO: fix `if-else` statement to better program.
        if model in ["svd", "svd_bias"]:
           self.criterion = svd_loss
        elif model == "bpr":
            self.criterion = bpr_loss
        elif model in ["gmf", "mlp", "two_tower"]:
            self.criterion = nn.BCELoss()

    def calculate_loss(
            self,
            y_pred: Optional[torch.Tensor] = None,
            y: Optional[torch.Tensor] = None,
            params: Optional[torch.nn.parameter] = None,
            regularization: Optional[int] = None,
            user_idx: Optional[torch.Tensor] = None,
            item_idx: Optional[torch.Tensor] = None,
            num_users: Optional[int] = None,
            num_items: Optional[int] = None,
            **kwargs
        ) -> torch.Tensor:
        """
        Calculates loss for given model using defined loss function.
        Because arguments for each loss function are different, `**kwargs` is used as function argument.
        """
        y_pred = kwargs.get("y_pred").squeeze()
        y = kwargs.get("y")
        params = kwargs.get("params")
        regularization = kwargs.get("regularization")
        user_idx = kwargs.get("user_idx")
        item_idx = kwargs.get("item_idx")
        num_users = kwargs.get("num_users")
        num_items = kwargs.get("num_items")
        # TODO: fix `if-else` statement to better program.
        if self.model in ["svd", "svd_bias"]:
            return self.criterion(y_pred, y.squeeze(), params, regularization, user_idx, item_idx, num_users, num_items)
        elif self.model in ["gmf", "mlp", "two_tower"]:
            return self.criterion(y_pred, y.squeeze())
        elif self.model == "bpr":
            return self.criterion(y_pred, params, regularization)
