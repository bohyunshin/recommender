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
        y_pred = kwargs.get("y_pred")
        y = kwargs.get("y")
        params = kwargs.get("params")
        regularization = kwargs.get("regularization")
        if self.model in ["svd", "svd_bias"]:
            return self.criterion(y_pred, y, params, regularization)
        elif self.model in ["gmf", "mlp"]:
            return self.criterion(y_pred.squeeze(), y)
        elif self.model == "bpr":
            return self.criterion(y_pred, params, regularization)
