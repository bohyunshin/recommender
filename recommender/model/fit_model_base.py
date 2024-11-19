from abc import abstractmethod
from model.recommender_base import RecommenderBase


class FitModelBase(RecommenderBase):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def fit(self, user_items, val_user_items):
        """
        Trains recommendation algorithm on a sparse matrix

        Parameters
        ----------
        user_items : csr_matrix
        """
        raise NotImplementedError

    @abstractmethod
    def calculate_loss(self, user_items):
        """
        Calculates training / validation loss for early stopping to prevent overfitting.
        This function is run after fitting user_factors and item_factors.

        Parameters
        ----------
        user_items : M x N user item matrix (csr)
            Dataset could be train set or validation set

        Returns
        -------
        loss
        """
        raise NotImplementedError