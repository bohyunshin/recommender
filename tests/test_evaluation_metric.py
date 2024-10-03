import numpy as np
from scipy.sparse import csr_matrix

from tools.evaluation import ranking_metrics_at_k
from model.mf.als import AlternatingLeastSquares


def test_evaluation_metric():
    implicit_mf = AlternatingLeastSquares()
    user_factors = np.array([
        [1,0,0],
        [0,1,0],
        [0,0,1],
        [0,0.5,0],
        [0,0,0.5]
    ])
    item_factors = np.array([
        [1,0,0],
        [0,1,0],
        [0,0,1],
        [0,0,0.5]
    ])
    implicit_mf.user_factors = user_factors
    implicit_mf.item_factors = item_factors

    data = [1,1,1,1,1]
    indices = [0,1,2,2,1]
    indptr = [0,1,2,3,4,5]
    test_user_items = csr_matrix((data, indices, indptr), shape=(5,4)) # [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,1,0], [0,1,0,0]]
    users, items = test_user_items.shape

    metric = ranking_metrics_at_k(implicit_mf, csr_matrix((5,4)), test_user_items, K=3)
    expected_map = (1 + 1 + 1 + 1/3 + 0) / users

    np.testing.assert_almost_equal(metric["map"], expected_map)