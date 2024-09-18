import numpy as np
from heapq import heappush, heappop

def topk(user_item_score, k):
    """
    Calculates top k items for selected users

    Parameters
    ----------
    user_item_score : M1 x N (np.ndarray)
        2D array of item score for selected M1 users

    Returns
    -------
    indices : M1 x k numpy array
        2D array of selected k item indices for each user
    distances : M1 x k numpy array
        2D array of selected k item similarities for each user
    """
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