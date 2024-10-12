from scipy.sparse import csr_matrix
import numpy as np
from numpy.testing import assert_array_equal, assert_almost_equal
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),"../recommender"))

from tools.topk import topk, heapq_select_k


def test_heapq_select_k():
    pred_item_pair = [
        (10,20), (1, 15), (1000, 99), (100,43), (5,10)
    ]
    k = 5
    expected = [99, 43, 20, 10, 15], [1000, 100, 10, 5, 1]
    assert_array_equal(expected, heapq_select_k(pred_item_pair, k))

def test_topk():
    user_items_score = np.array(
        [[0.43, 1.35, 0.72, 0.8],
         [0.6, 0.48, 0.41, 0.62]]
    )
    k = 3
    ind_exp = np.array([[1, 3, 2],[3, 0, 1]])
    dist_exp = np.array([[1.35, 0.8, 0.72],[0.62, 0.6, 0.48]])
    ind_real, dist_real = topk(user_items_score, csr_matrix(user_items_score.shape), k)
    assert_array_equal(ind_exp, ind_real)
    assert_almost_equal(dist_exp, dist_real)
