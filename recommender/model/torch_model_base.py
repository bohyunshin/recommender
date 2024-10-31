from abc import abstractmethod
import numpy as np
from torch import nn

from model.recommender_base import RecommenderBase


class TorchModelBase(nn.Module, RecommenderBase):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def predict(self, user_factors, item_factors, userid, **kwargs):
        """
        Calculate user-item scores based on learned user / item embeddings

        Parameters
        ----------
        user_factors : Tensor (M1 x K)

        item_factors : Tensor (N1 x K)

        Returns
        -------
        user_item_scores : Tensor (M1 x N1)
        """
        return np.dot(user_factors[userid], item_factors.T)

    def set_trained_embedding(self):
        self.user_factors = self.embed_user.weight.data.clone().detach().cpu().numpy()
        self.item_factors = self.embed_item.weight.data.clone().detach().cpu().numpy()
