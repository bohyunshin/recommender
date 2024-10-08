import torch.nn as nn
from loss.custom import bpr_loss


class Criterion:
    def __init__(self, model):
        self.model = model
        if model == "svd":
           self.criterion = nn.MSELoss()
        elif model == "bpr":
            self.criterion = bpr_loss

    def calculate_loss(self, **kwargs):
        y_pred = kwargs.get("y_pred")
        y = kwargs.get("y")
        params = kwargs.get("params")
        regularization = kwargs.get("regularization")
        if self.model == "svd":
            return self.criterion(y_pred, y)
        elif self.model == "bpr":
            return self.criterion(y_pred, params, regularization)
