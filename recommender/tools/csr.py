import numpy as np
from scipy.sparse import csr_matrix
from collections import defaultdict


def dataframe_to_csr(df, shape, implicit):
    """
    Converts a pandas dataframe to a csr matrix.

    Parameters
    ----------
    df : pd.DataFrame (user_id | item_id | interactions.)
        Dataframe should be sort by user_id and timestamp (if timestamp column is given)
        When implicit is true, interactions column denotes total number of interaction
        between user and item.
        When implicit is false, interactions column denotes explicit rating
        between user and item.

    shape : tuple
        (total number of user_ids, total number of item_ids)

    implicit : bool
        True when feedback data is implicit, False if explicit.

    Returns
    -------
    user_item : csr matrix
    """
    assert "user_id" in df.columns
    assert "item_id" in df.columns
    assert "interactions" in df.columns

    user2item2value = defaultdict(dict)

    user_ids = sorted(df["user_id"].unique())

    for user, item, interaction in zip(df["user_id"], df["item_id"], df["interactions"]):
        if user2item2value[user].get(item, 0) == 0:
            user2item2value[user][item] = interaction
        else:
            if implicit:
                user2item2value[user][item] += interaction
            else:
                # for explicit feedback, assume only one interaction between user and item
                user2item2value[user][item] = interaction

    indices = []
    indptr = []
    data = []

    row_index = 0
    indptr.append(row_index)
    for user_id in user_ids:
        item2value = user2item2value[user_id]
        count = 0
        for item, value in item2value.items():
            indices.append(item)
            data.append(value)
            count += 1
        row_index += count
        indptr.append(row_index)

    csr = csr_matrix((data, indices, indptr), shape=shape)
    return csr


def implicit_to_csr(arr, shape):
    """
    Converts user-item interaction data to user-item interaction csr matrix.

    Parameters
    ----------
    arr : np.ndarray (n_samples, 2)
        Each row represents (user_id, item_id) interaction.

    shape : tuple
        (total number of user_ids, total number of item_ids)

    Returns
    -------
    user_item : csr matrix
    """

    assert arr.shape[1] == 2

    user2item2value = defaultdict(dict)

    user_ids = np.arange(shape[0])

    for interaction in arr:
        user, item = interaction
        user2item2value[user.item()][item.item()] = 1

    indices = []
    indptr = []
    data = []

    row_index = 0
    indptr.append(row_index)
    for user_id in user_ids:
        item2value = user2item2value[user_id]
        count = 0
        for item, value in item2value.items():
            indices.append(item)
            data.append(value)
            count += 1
        row_index += count
        indptr.append(row_index)

    csr = csr_matrix((data, indices, indptr), shape=shape)
    return csr


def mapping_index(ids):
    ids = list(set(ids))
    id2idx = {}
    for idx, val in enumerate(sorted(ids)):
        id2idx[val] = idx
    return id2idx


def slice_csr_matrix(csr, row, col):
    """
    Returns csr[row, col] value not using slicing operation in csr matrix
    """
    indices = csr.indices[csr.indptr[row]:csr.indptr[row + 1]]
    data = csr.data[csr.indptr[row]:csr.indptr[row + 1]]
    for i,d in zip(indices, data):
        if i == col:
            return d
    return 0