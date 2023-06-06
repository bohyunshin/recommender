from abc import abstractmethod
import numpy as np

class RecommenderBase:

    @abstractmethod
    def fit(self, user_items, val_user_items):
        """
        Trains recommendation algorithm on a sparse matrix

        Parameters
        ----------
        user_items : csr_matrix
        """
        pass

    @abstractmethod
    def recommend(self, userid, user_items, N=10):
        """
        Recommends items for users

        Parameters
        ----------
        userid : Union[int, array_like]
        user_items : csr_matrix
        N : int, optional

        Returns
        -------
        indices : M1 x k numpy array
            2D array of selected k item indices for each user
        distances : M1 x k numpy array
            2D array of selected k item similarities for each user
        """

    @abstractmethod
    def predict(self, user_factors, item_factors):
        """
        Predicts users' ratings (or preference) based on factorized user_factors and item_factors.
        For matrix factorization models, this could be dot product between user_factors and item_factors.
        For deep learning models, this could be inference step fed with user/item information to neural network

        user_factors : M1 x K
            Genearally, M1 does not need to be M, which is total number of users in dataset. We consider general
            situation where we want to recommend for selected M1 users
        item_factors : N x K
            Although we may not want to recommend items for all users, still we need all item embedding vectors.
            That's why we use all of item vectors.
        """

    @abstractmethod
    def calculate_loss(self, user_items):
        """
        Calculates training / validation loss for early stopping to prevent overfitting

        Parameters
        ----------
        user_items : M x N user item matrix
        user_factors : M x K user latent vectors
        item_factors : N x K item latent vectors
        """
