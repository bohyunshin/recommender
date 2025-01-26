import numpy as np
from sklearn.metrics import ndcg_score

from recommender.libs.utils.evaluation import ranking_metrics_at_k


def test_ndcg_same_as_sklearn():
    """
    total number of items: 5
    item_ids that user liked: [0, 3, 4]
    numer of recommended items (top k value): starting from 1 to 5
    definition of relevance: 1 if user liked else 0
    scores of each item: [10, 4, 8, 1, 3]
    recommended item_ids in descending score order: [0, 2, 1, 4, 3]
    """
    true_relevance = np.array([[1, 0, 0, 1, 1]])
    num_items = true_relevance.shape[1]
    # dummy score values which will be predictions from recommender system
    scores = np.array([[10, 4, 8, 1, 3]])
    liked_items = np.array([0, 3, 4])
    reco_items = np.array([0, 2, 1, 4, 3])

    for k in range(1, num_items+1):
        ndcg_sklearn = ndcg_score(true_relevance, scores, ignore_ties=True, k=k)
        ndcg_implemented = ranking_metrics_at_k(liked_items, reco_items[:k])["ndcg"]
        np.testing.assert_almost_equal(ndcg_sklearn, ndcg_implemented)


def test_map_ndcg():
    liked_items = np.array([100, 10, 50, 0, 11, 22, 33, 44, 55, 66])
    # case 1: hit at item_id 100, 10
    reco_items = np.array([2, 3, 4, 100, 10])
    K = len(reco_items)
    metric = ranking_metrics_at_k(liked_items, reco_items)
    dcg = 1.0 / np.log2(np.arange(2, K + 2))
    idcg = np.sum(dcg[:5])

    expected_ndcg = (dcg[3] + dcg[4]) / idcg
    expected_map = (1 / 4 + 2 / 5) / len(liked_items)

    np.testing.assert_almost_equal(metric["ndcg"], expected_ndcg)
    np.testing.assert_almost_equal(metric["ap"], expected_map)

    # case 2: hit at item_id 100, 10, 50
    reco_items = np.array([100, 10, 50, 2, 3])
    metric = ranking_metrics_at_k(liked_items, reco_items)
    expected_ndcg = (dcg[0] + dcg[1] + dcg[2]) / idcg
    expected_map = (1 + 1 + 1) / len(liked_items)
    np.testing.assert_almost_equal(metric["ndcg"], expected_ndcg)
    np.testing.assert_almost_equal(metric["ap"], expected_map)


def test_map_increasing_as_k_larger():
    # reco_items = np.array([0, 1, 2, 5, 6, 7, 3, 100, 4])
    reco_items = np.array([0]*10 + [2] + [0]*5 + [3] + [0]*5)
    liked_items = np.array([2, 3, 4, 100, 10])
    map = -1e10
    for k in range(len(reco_items)):
        metric = ranking_metrics_at_k(liked_items, reco_items[:k+1])
        assert metric["ap"] >= map
        map = metric["ap"]
        print(metric["ndcg"])