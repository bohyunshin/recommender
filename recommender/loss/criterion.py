from typing import List, Optional

import torch
import torch.nn as nn

from recommender.libs.constant.model.name import ModelName
from recommender.loss.custom import bpr_loss, svd_loss


class Criterion:
    def __init__(self, model: str):
        """
        Selects appropriate loss function for given model, and calculates loss.

        Args:
            model (str): Model name to train.
        """
        self.model = model
        # TODO: fix `if-else` statement to better program.
        if model in [ModelName.SVD.value, ModelName.SVD_BIAS.value]:
            self.criterion = svd_loss
        elif model == ModelName.BPR.value:
            self.criterion = bpr_loss
        elif model in [
            ModelName.GMF.value,
            ModelName.MLP.value,
            ModelName.TWO_TOWER.value,
        ]:
            self.criterion = nn.BCELoss()

    def calculate_loss(
        self,
        y_pred: Optional[torch.Tensor],
        y: Optional[torch.Tensor],
        params: Optional[List[nn.parameter.Parameter]],
        regularization: Optional[int],
        user_idx: Optional[torch.Tensor],
        item_idx: Optional[torch.Tensor],
        num_users: Optional[int],
        num_items: Optional[int],
    ) -> torch.Tensor:
        """
        Calculates loss for given model using defined loss function.
        Because arguments for each loss function are different, Optional type is used.

        Args:
            y_pred (Optional[torch.Tensor]): Prediction value. Could be logit or probability.
            y (Optional[torch.Tensor]): True y value. If implicit data, dummy value is used here.
            params (Optional[torch.nn.parameter]): Model parameters from torch model.
            regularization (Optional[int]): Regularization parameter balancing between main loss and penalty.
            user_idx (Optional[torch.Tensor]): User index in current batch.
            item_idx (Optional[torch.Tensor]): Item index in current batch.
            num_users (Optional[int]): Number of total users.
            num_items (Optional[int]): Number of total items.

        Returns (torch.Tensor):
            Calculated loss.
        """
        # TODO: fix `if-else` statement to better program.
        if self.model in [ModelName.SVD.value, ModelName.SVD_BIAS.value]:
            return self.criterion(
                pred=y_pred.squeeze(),
                true=y.squeeze(),
                params=params,
                regularization=regularization,
                user_idx=user_idx,
                item_idx=item_idx,
                num_users=num_users,
                num_items=num_items,
            )
        elif self.model in [
            ModelName.GMF.value,
            ModelName.MLP.value,
            ModelName.TWO_TOWER.value,
        ]:
            return self.criterion(
                input=y_pred.squeeze(),
                target=y.squeeze(),
            )
        elif self.model == ModelName.BPR.value:
            return self.criterion(
                pred=y_pred,
                params=params,
                regularization=regularization,
            )
