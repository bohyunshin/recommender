from abc import abstractmethod
from typing import Tuple, Optional

from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from recommender.libs.topk import topk


class RecommenderBase:
    def __init__(self):
        self.user_factors = None
        self.item_factors = None

    def recommend(
            self,
            user_idx: NDArray,
            item_idx: NDArray,
            user_items: csr_matrix,
            N: Optional[int] = 10,
        ) -> Tuple[NDArray, NDArray]:
        """
        Recommends N items to given users.

        Args:
            user_idx (NDArray): User ids to recommend.
            item_idx (NDArray): List of all items.
            user_items (csr_matrix): User x items matrix

        Returns (Tuple[NDArray, NDArray]):
            indices (NDArray): 2D array of selected k item indices for each user
            distances (NDArray): 2D array of selected k item similarities for each user
        """

        indices, distances = topk(self.predict(user_idx, user_items=user_items, item_idx=item_idx), user_items[user_idx], N)

        return indices, distances

    @abstractmethod
    def predict(
            self,
            user_idx: NDArray,
            **kwargs,
        ) -> NDArray:
        """
        Predicts users' ratings (or preference) based on factorized user_factors and item_factors.
        For matrix factorization models, this could be dot product between user_factors and item_factors.
        For deep learning models, this could be inference step fed with user/item information to neural network

        Args:
            user_idx (NDArray): Set of user idxs who are recommendation target.

        Returns (NDArray):
            User x item prediction score matrix.
        """
        raise NotImplementedError