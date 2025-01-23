from typing import Dict

import numpy as np
from numpy.typing import NDArray

from recommender.libs.constant.inference.evaluation import Metric


def ranking_metrics_at_k(
    liked_items: NDArray,
    reco_items: NDArray,
) -> Dict[str, float]:
    """
    References
    - https://sdsawtelle.github.io/blog/output/mean-average-precision-MAP-for-recommender-systems.html
    - https://gist.github.com/bwhite/3726239

    Calculates ndcg, average precision (aP), hit, and recall for `one user`.
    If you want to derive ndcg, map for n users, you should average them over n.

    For average precision, we use following definition.

    AP = \dfrac{1}{m} \sum_{i=1}^{K} P(i) r(i)
    where m is number of items user liked and K is number of recommended items,
    P(i) is precision at i and r(i) is indicator variable 1 if ith item is hitted else 0.
    Here, precision@i = # of hitted items until ith ranked item / K

    Note that for some variations of AP, \dfrac{1}{ min(m, K) } is used instead of \dfrac{1}{m} to prevent deviding
    large value when K < m for some special case. If defined denominator as min(m, K), map may not increase as K is
    getting larger.

    For normalized discounted cumulative gain, we use following definition.

    NDCG = \dfrac{DCG}{IDCG}
    where DCG = \sum_{i=1}^{K} \dfrac{1}{\log2{i+1}} * r(i) and
    IDCG = \sum_{i=1}^{K} \dfrac{1}{\log2{i+1}}.
    Here, r(i) is indicator variable 1 if ith item is hitted else 0.

    Maximum value of NDCG is IDCG by their definitions, therefore, 0 <= NDCG <= 1.

    Note that NDCG does NOT gurantee increasing value as K is getting larger because denominator, which is IDCG
    is getting larger as K is getting larger.

    liked_items:
        item ids selected by one user in test dataset.
    reco_items:
        item ids recommended for one user.
    """
    # number of recommended items
    K = len(reco_items)

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
    ap /= len(liked_items)

    # Calculate recall
    recall = hit / len(liked_items)

    return {Metric.AP.value: ap, Metric.NDCG.value: ndcg, Metric.RECALL.value: recall}