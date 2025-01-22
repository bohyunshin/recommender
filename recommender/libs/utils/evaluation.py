from typing import Dict

import numpy as np
from numpy.typing import NDArray

from recommender.libs.constant.inference.evaluation import Metric
from recommender.libs.utils.utils import safe_divide


def ranking_metrics_at_k(
    liked_items: NDArray,
    reco_items: NDArray,
) -> Dict[str, float]:
    """
    Calculates ndcg, average precision (aP), hit, and recall for `one user`.
    If you want to derive ndcg, map for n users, you should average them over n.

    liked_items:
        item ids selected by one user in test dataset.
    reco_items:
        item ids recommended for one user.
    """
    # number of recommended items
    K = len(reco_items)
    # # in case user liked items less than K
    # K = min(len(liked_items), K)
    # reco_items = reco_items[:K]

    ap = 0
    cg = (1.0 / np.log2(np.arange(2, K + 2)))
    idcg = cg.sum()
    ndcg = 0
    hit = 0

    for i in range(K):
        if reco_items[i] in liked_items:
            hit += 1
            ap += hit / (i + 1)
            ndcg += cg[i] / idcg
    ap /= K

    # Calculate recall
    recall = safe_divide(hit, len(liked_items))

    return {Metric.AP.value: ap, Metric.NDCG.value: ndcg, Metric.RECALL.value: recall}