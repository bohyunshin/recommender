from abc import abstractmethod
from typing import Optional
from scipy.sparse import csr_matrix

from model.recommender_base import RecommenderBase


class FitModelBase(RecommenderBase):
    def __init__(self):
        """
        Abstract base class for fit based model such as als, user-based model.
        """
        super().__init__()

    @abstractmethod
    def fit(
            self,
            user_items: csr_matrix,
            val_user_items: Optional[csr_matrix]
        ) -> None:
        """
        Factorizes the user_items matrix.

        Args:
            user_items (csr_matrix): This is user x item matrix whose dimension
            is M x N. It is used in training step.
            val_user_items (csr_matrix, optional): This is user x item matrix
            whose dimension is M x N. It is used in validation step.
            Note that in training step, we do not use this matrix.
        """
        raise NotImplementedError

    @abstractmethod
    def calculate_loss(
            self,
            user_items: csr_matrix
        ) -> float:
        """
        Calculates training/validation loss in each iteration.

        We calculate loss in each iteration to check if parameters are converged or not.
        Depending on the user_items argument, it calculates training or validation loss.

        It is strongly recommended that it should be checked whether validation loss drops
        and becomes stable

        Args:
            user_items (csr_matrix): Training or validation user x item matrix.

        Returns (float):
            Calculated loss value.
        """
        raise NotImplementedError