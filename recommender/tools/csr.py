import numpy as np
from scipy.sparse import csr_matrix
from collections import defaultdict


def dataframe_to_csr(df, implicit):
    """
    Converts a pandas dataframe to a csr matrix.

    Parameters
    ----------
    df : pd.DataFrame
        Columns should be user_id | item_id | interactions.
        When implicit is true, interactions column denotes total number of interaction
        between user and item.
        When implicit is false, interactions column denotes explicit rating
        between user and item.

    implicit : bool
        True when feedback data is implicit, False if explicit.

    Returns
    -------
    user_item : csr matrix
    """
    assert set(df.columns) == set(["user_id", "item_id", "interactions"])

    user_mapping = mapping_index(df["user_id"])
    item_mapping = mapping_index(df["item_id"])

    df["user_id"] = df["user_id"].map(user_mapping)
    df["item_id"] = df["item_id"].map(item_mapping)

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

    csr = csr_matrix((data, indices, indptr))
    return csr, user_mapping, item_mapping


def mapping_index(ids):
    ids = list(set(ids))
    id2idx = {}
    for idx, val in enumerate(sorted(ids)):
        id2idx[val] = idx
    return id2idx