import torch.nn as nn
from loss.custom import bpr_loss, svd_loss


class Criterion:
    def __init__(self, model):
        self.model = model
        if model in ["svd", "svd_bias"]:
           self.criterion = svd_loss
        elif model == "bpr":
            self.criterion = bpr_loss
        elif model in ["gmf", "mlp"]:
            self.criterion = nn.BCELoss()

    def calculate_loss(self, **kwargs):
        y_pred = kwargs.get("y_pred").squeeze()
        y = kwargs.get("y")
        params = kwargs.get("params")
        regularization = kwargs.get("regularization")
        user_idx = kwargs.get("user_idx")
        item_idx = kwargs.get("item_idx")
        num_users = kwargs.get("num_users")
        num_items = kwargs.get("num_items")
        if self.model in ["svd", "svd_bias"]:
            return self.criterion(y_pred, y.squeeze(), params, regularization, user_idx, item_idx, num_users, num_items)
        elif self.model in ["gmf", "mlp"]:
            return self.criterion(y_pred, y.squeeze())
        elif self.model == "bpr":
            return self.criterion(y_pred, params, regularization)
