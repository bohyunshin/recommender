import unittest
import numpy as np
import implicit
from scipy.sparse import csr_matrix
from numpy.testing import assert_array_equal, assert_almost_equal

from model.mf.als import AlternatingLeastSquares

class TestALS(unittest.TestCase):
    def test_binarize(self):
        als = AlternatingLeastSquares()
        tc = np.array([1,0,5,10,0,0,100])
        expected = np.array([1,0,1,1,0,0,1])
        assert_array_equal(expected, als.binarize(tc))

    def test_als_result_same_as_implicit_library(self):
        params = {
            'factors':10,
            'iterations':100,
            'regularization':0.01,
            'random_state':42
        }
        als = AlternatingLeastSquares(**params)
        imp_als = implicit.als.AlternatingLeastSquares(**params)
        tc_user_items = csr_matrix(np.array(
            [[2, 0, 0, 0, 3, 4],
             [0, 0, 2, 1, 0, 0],
             [5, 1, 0, 0, 2, 1]]
        ))

        als.fit(tc_user_items)
        imp_als.fit(tc_user_items)

        assert_almost_equal(als.user_factors, imp_als.user_factors, decimal=2)
        assert_almost_equal(als.item_factors, imp_als.item_factors, decimal=2)

if '__name__' == '__main__':
    unittest.main()