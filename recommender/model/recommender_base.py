from abc import abstractmethod
from tools.topk import topk


class RecommenderBase:
    def __init__(self):
        self.user_factors = None
        self.item_factors = None

    def recommend(self, userid, user_items, filter_already_liked_items=True, N=10):
        """
        Recommends items for users

        Parameters
        ----------
        userid : Union[int, array_like]
        N : int, optional

        Returns
        -------
        indices : M1 x k numpy array
            2D array of selected k item indices for each user
        distances : M1 x k numpy array
            2D array of selected k item similarities for each user
        """

        indices, distances = topk(self.predict(userid, user_items=user_items), user_items[userid], N)

        return indices, distances

    @abstractmethod
    def predict(self, user_idx, **kwargs):
        """
        Predicts users' ratings (or preference) based on factorized user_factors and item_factors.
        For matrix factorization models, this could be dot product between user_factors and item_factors.
        For deep learning models, this could be inference step fed with user/item information to neural network

        Parameters
        ----------
        user_idx : M1, (np.ndarray)
            Set of user idxs who are recommendation target

        Returns
        -------
        user_item_score: M1 x N (np.ndarray)
        """
        raise NotImplementedError