import numpy as np
from heapq import heappush, heappop

def topk(user_items_score, k):
    """
    Calculates top k items for selected users

    Parameters
    ----------
    user_items_scores : M1 x N csr_matrix
        2D array of item score for selected M1 users

    Returns
    -------
    indices : M1 x k numpy array
        2D array of selected k item indices for each user
    distances : M1 x k numpy array
        2D array of selected k item similarities for each user
    """
    M1, N = user_items_score.shape
    indptr = user_items_score.indptr
    indices = np.zeros((M1, k))
    distances = np.zeros((M1, k))
    for u in range(M1):
        pred_item_pair = []
        for i in range(indptr[u], indptr[u+1]):
            pred, item = user_items_score.data[i], user_items_score.indices[i]
            pred_item_pair.append((pred, item))
        reco, dist = heapq_select_k(pred_item_pair, k)
        indices[u,:] = reco
        distances[u,:] = dist
    return indices, distances

def heapq_select_k(pred_item_pair, k):
    """
    Select top k items from scores.
    To efficiently select top k items according to scores, we use heapq
    datastructure which results in O(n + klogn) time complexity.
    Note that still, we need to improve performance of this function

    Parameters
    ----------
    pred_item_pair: (score, item_idx) array
        Stores pairs of score and corresponding item index
    k: int
        Integer value
    """
    heap = []
    distances = []
    reco_items = []
    for pred, item in pred_item_pair:
        heappush(heap, (-pred, item))
    count = 0
    while count != k:
        pred, item = heappop(heap)
        reco_items.append(item)
        distances.append(-pred)
        count += 1
    return reco_items, distances