import numpy as np

from libs.utils.evaluation import ranking_metrics_at_k


def test_map_ndcg():
    liked_items = np.array([100, 10, 50, 0, 11, 22, 33, 44, 55, 66])
    # case 1: hit at item_id 100, 10
    reco_items = np.array([2, 3, 4, 100, 10])
    K = len(reco_items)
    metric = ranking_metrics_at_k(liked_items, reco_items)
    dcg = 1.0 / np.log2(np.arange(2, K + 2))
    idcg = np.sum(dcg)

    expected_ndcg = (dcg[3] + dcg[4]) / idcg
    expected_map = (1 / 4 + 2 / 5) / K

    np.testing.assert_almost_equal(metric["ndcg"], expected_ndcg)
    np.testing.assert_almost_equal(metric["ap"], expected_map)

    # case 2: hit at item_id 100, 10, 50
    reco_items = np.array([100, 10, 50, 2, 3])
    metric = ranking_metrics_at_k(liked_items, reco_items)
    expected_ndcg = (dcg[0] + dcg[1] + dcg[2]) / idcg
    expected_map = (1 + 1 + 1) / K
    np.testing.assert_almost_equal(metric["ndcg"], expected_ndcg)
    np.testing.assert_almost_equal(metric["ap"], expected_map)