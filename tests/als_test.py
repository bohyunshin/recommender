import unittest
import numpy as np
import implicit
from implicit.datasets.movielens import get_movielens
from implicit.nearest_neighbours import bm25_weight

from scipy.sparse import csr_matrix
from numpy.testing import assert_array_equal, assert_almost_equal

from model.mf.implicit_mf import AlternatingLeastSquares

from tools.utils import nonzeros

class TestALS(unittest.TestCase):
    def test_binarize(self):
        als = AlternatingLeastSquares()
        tc = np.array([1,0,5,10,0,0,100])
        expected = np.array([1,0,1,1,0,0,1])
        assert_array_equal(expected, als.binarize(tc))

    def test_als_result_same_as_implicit_library(self):
        params = {
            'factors':100,
            'iterations':2,
            'regularization':0.01,
            'random_state':42
        }
        als = AlternatingLeastSquares(**params)
        imp_als = implicit.als.AlternatingLeastSquares(**params)
        imp_als.calculate_training_loss = True
        tc_user_items = csr_matrix(np.array(
            [[2, 0, 0, 0, 3, 4],
             [0, 0, 2, 1, 0, 0],
             [5, 1, 0, 0, 2, 1]]
        ))

        M, N = tc_user_items.shape
        user_factors = np.random.rand(M, params["factors"]).astype(np.float32) * 0.01
        item_factors = np.random.rand(N, params["factors"]).astype(np.float32) * 0.01

        als.user_factors = user_factors.copy()
        als.item_factors = item_factors.copy()
        imp_als.user_factors = user_factors.copy()
        imp_als.item_factors = item_factors.copy()

        # titles, ratings = get_movielens("1m")
        # min_rating = 4
        #
        # # remove things < min_rating, and convert to implicit dataset
        # # by considering ratings as a binary preference only
        # ratings.data[ratings.data < min_rating] = 0
        # ratings.eliminate_zeros()
        # ratings.data = np.ones(len(ratings.data))
        # ratings = (bm25_weight(ratings, B=0.9) * 5).tocsr()
        # user_ratings = ratings.T.tocsr()
        #
        # M,N = user_ratings.shape
        # user_factors = np.random.rand(M, params["factors"]).astype(np.float32) * 0.01
        # item_factors = np.random.rand(N, params["factors"]).astype(np.float32) * 0.01
        #
        # als.user_factors = user_factors.copy()
        # als.item_factors = item_factors.copy()
        # imp_als.user_factors = user_factors.copy()
        # imp_als.item_factors = item_factors.copy()
        #
        # als.fit(user_ratings)
        # imp_als.fit(user_ratings)

        assert_almost_equal(als.user_factors, imp_als.user_factors, decimal=2)
        assert_almost_equal(als.item_factors, imp_als.item_factors, decimal=2)

if '__name__' == '__main__':
    unittest.main()