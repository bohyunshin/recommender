from typing import Optional, Dict

import numpy as np
from scipy.sparse import csr_matrix
from collections import defaultdict

from model.recommender_base import RecommenderBase


def ranking_metrics_at_k(
        model: RecommenderBase,
        train_user_items: csr_matrix,
        test_user_items: csr_matrix,
        K: Optional[int] = 10
) -> Dict[str, float]:
    """
    Calculates ranking metrics for a given trained model

    Args:
        model (RecommenderBase): The fitted recommendation model to test.
        test_user_items (csr_matrix): Sparse matrix of user by item that contains
        withheld elements to test on.
        K (int): Number of items to test on.

    Returns (float):
        Calculated metrics.
    """

    if not isinstance(test_user_items, csr_matrix):
        test_user_items = test_user_items.tocsr()

    users, items = test_user_items.shape

    if items < K:
        raise ValueError(f"K cannot be larger than number of items, got K as {K} but number of items is {items}")

    # precision
    total = 0
    # map
    mean_ap = 0
    # ndcg
    cg = (1.0 / np.log2(np.arange(2, K + 2)))
    cg_sum = np.cumsum(cg)
    ndcg = 0

    test_indptr = test_user_items.indptr
    test_indices = test_user_items.indices

    batch_size = 1000
    start_idx = 0

    # get an array of userids that have at least one item in the test set
    to_generate = np.arange(users, dtype="int32")
    to_generate = to_generate[np.ediff1d(test_user_items.indptr) > 0]

    while start_idx < len(to_generate):
        batch = to_generate[start_idx: start_idx + batch_size]
        ids, _ = model.recommend(batch, np.arange(items), train_user_items, N=K)
        start_idx += batch_size

        for batch_idx in range(len(batch)):
            u = batch[batch_idx]
            likes = defaultdict(int)
            m = 0
            for i in range(test_indptr[u], test_indptr[u+1]):
                likes[test_indices[i]] = 1
                m += 1

            ap = 0
            hit = 0
            idcg = cg_sum[min(K, m) - 1]

            for i in range(K):
                if likes[ids[batch_idx, i]] == 1:
                    hit += 1
                    ap += hit / (i + 1)
                    ndcg += cg[i] / idcg
            mean_ap += ap / min(K, m)
            total += 1

    return {
        "map": mean_ap / total,
        "ndcg": ndcg / total
    }