import numpy as np
from scipy.sparse import csr_matrix

from topk import topk

from recommender_base import RecommenderBase

class MatrixFactorizationBase(RecommenderBase):
    def __init__(self):
        self.user_factors = None
        self.item_factors = None

    def recommend(self, userid, user_items, N=10):
        user_factors = self.user_factors[userid]
        item_factors = self.item_factors
        indices, distances = topk(self.predict(user_factors, item_factors), N)

        return indices, distances

    def fit(self, user_items, val_user_items):
        pass

    def predict(self, user_factors, item_factors):
        pass