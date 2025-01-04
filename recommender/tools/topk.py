from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from heapq import heappush, heappop

def topk(
        user_item_score: NDArray,
        filter_query_items: csr_matrix,
        k: int
) -> Tuple[NDArray, NDArray]:
    """
    Calculates top k items for selected users

    Args:
        user_item_score (NDArray): 2D array of item score for selected M1 users

    Returns (Tuple[NDArray, NDArray]):
        indices: 2D array of selected k item indices for each user
        distances: 2D array of selected k item similarities for each user
    """
    assert user_item_score.shape == filter_query_items.shape

    neginf = -1e100
    indptr = filter_query_items.indptr
    indices = filter_query_items.indices
    for u in range(len(indptr)-1):
        for i in range(indptr[u], indptr[u+1]):
            user_item_score[u,indices[i]] = neginf


    M1, N = user_item_score.shape
    indices = np.zeros((M1, k))
    distances = np.zeros((M1, k))
    for u in range(M1):
        scores = user_item_score[u]
        score_item_pair = []
        for item, score in enumerate(scores):
            score_item_pair.append((score, item))
        reco, dist = heapq_select_k(score_item_pair, k)
        indices[u,:] = reco
        distances[u,:] = dist
    return indices, distances

def heapq_select_k(score_item_pair, k):
    """
    Select top k items from scores.
    To efficiently select top k items according to scores, we use heapq
    datastructure which results in O(n + klogn) time complexity.
    Note that still, we need to improve performance of this function

    Parameters
    ----------
    score_item_pair: (score, item_idx) array
        Stores pairs of score and corresponding item index
    k: int
        Integer value
    """
    heap = []
    distances = []
    reco_items = []
    for score, item in score_item_pair:
        heappush(heap, (-score, item))
    count = 0
    while count != k:
        score, item = heappop(heap)
        reco_items.append(item)
        distances.append(-score)
        count += 1
    return reco_items, distances