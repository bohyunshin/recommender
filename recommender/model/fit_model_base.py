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
    def predict(self, user_factors, item_factors):
        """
        Predicts users' ratings (or preference) based on factorized user_factors and item_factors.
        For matrix factorization models, this could be dot product between user_factors and item_factors.
        For deep learning models, this could be inference step fed with user/item information to neural network

        Parameters
        ----------
        user_factors : M1 x K (np.ndarray)
            Genearally, M1 does not need to be M, which is total number of users in dataset. We consider general
            situation where we want to recommend for selected M1 users
        item_factors : N x K (np.ndarray)
            Although we may not want to recommend items for all users, still we need all item embedding vectors.
            That's why we use all of item vectors.

        Returns
        -------
        user_item_score: M1 x N (np.ndarray)
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