import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from tqdm.auto import tqdm
from collections import defaultdict

from .utils import check_random_state

def ranking_metrics_at_k(model, train_user_items, test_user_items, K=10,
                         show_progress=True, num_threads=1):
    """ Calculates ranking metrics for a given trained model

    Parameters
    ----------
    model : RecommenderBase
        The fitted recommendation model to test
    test_user_items : csr_matrix
        Sparse matrix of user by item that contains withheld elements to
        test on
    K : int
        Number of items to test on
    show_progress : bool, optional
        Whether to show a progress bar
    num_threads : int, optional
        The number of threads to use for testing. Specifying 0 means to default
        to the number of cores on the machine. Note: aside from the ALS and BPR
        models, setting this to more than 1 will likely hurt performance rather than
        help.

    Returns
    -------
    float
        the calculated p@k
    """

    if not isinstance(test_user_items, csr_matrix):
        test_user_items = test_user_items.tocsr()

    users, items = test_user_items.shape

    if items < K:
        raise ValueError(f"K cannot be larger than number of items, got K as {K} but number of items is {items}")

    # precision
    relevant = 0
    pr_div = 0
    total = 0
    # map
    mean_ap = 0
    ap = 0
    # ndcg
    cg = (1.0 / np.log2(np.arange(2, K + 2)))
    cg_sum = np.cumsum(cg)
    ndcg = 0
    # auc
    mean_auc = 0

    test_indptr = test_user_items.indptr
    test_indices = test_user_items.indices

    batch_size = 1000
    start_idx = 0

    # get an array of userids that have at least one item in the test set
    to_generate = np.arange(users, dtype="int32")
    to_generate = to_generate[np.ediff1d(test_user_items.indptr) > 0]

    while start_idx < len(to_generate):
        batch = to_generate[start_idx: start_idx + batch_size]
        ids, _ = model.recommend(batch, train_user_items, N=K)
        start_idx += batch_size

        for batch_idx in range(len(batch)):
            u = batch[batch_idx]
            likes = defaultdict(int)
            m = 0
            for i in range(test_indptr[u], test_indptr[u+1]):
                likes[test_indices[i]] = 1
                m += 1

            pr_div += min(K, m)
            ap = 0
            hit = 0
            miss = 0
            auc = 0
            idcg = cg_sum[min(K, m) - 1]
            num_pos_items = m
            num_neg_items = items - num_pos_items

            for i in range(K):
                if likes[ids[batch_idx, i]] == 1:
                    relevant += 1
                    hit += 1
                    ap += hit / (i + 1)
                    ndcg += cg[i] / idcg
                else:
                    miss += 1
                    auc += hit
            auc += ((hit + num_pos_items) / 2.0) * (num_neg_items - miss)
            mean_ap += ap / min(K, m)
            mean_auc += auc / (num_pos_items * num_neg_items)
            total += 1

    return {
        "precision": relevant / pr_div,
        "map": mean_ap / total,
        "ndcg": ndcg / total,
        "auc": mean_auc / total
    }