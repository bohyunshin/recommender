from collections import defaultdict
from typing import Dict, Tuple

import pandas as pd
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from recommender.libs.constant.data.name import Field


def dataframe_to_csr(
    df: pd.DataFrame, shape: Tuple[int, int], implicit: bool
) -> csr_matrix:
    """
    Converts a pandas dataframe to a csr matrix.

    Args:
        df (pd.DataFrame[user_id | item_id | interactions]):
            Dataframe should be sort by user_id and timestamp (if timestamp column is given)
            When implicit is true, interactions column denotes total number of interaction
            between user and item.
            When implicit is false, interactions column denotes explicit rating
            between user and item.
        shape (Tuple[int, int]): Total number of user_ids, total number of item_ids
        implicit (bool): True when feedback data is implicit, False if explicit.

    Returns (csr_matrix):
        Converted csr matrix.
    """
    assert Field.USER_ID.value in df.columns
    assert Field.ITEM_ID.value in df.columns
    assert Field.INTERACTION.value in df.columns

    user2item2value = defaultdict(dict)

    user_ids = range(shape[0])

    for user, item, interaction in zip(
        df[Field.USER_ID.value], df[Field.ITEM_ID.value], df[Field.INTERACTION.value]
    ):
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
        if user2item2value[user_id] == {}:
            indptr.append(row_index)
            continue
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


def mapping_index(ids: NDArray) -> Dict[int, int]:
    """
    Maps original ids to ascending integer.

    Args:
        ids (NDArray): Some unique or non-unique integer ids.

    Returns (Dict[int, int]):
        Mapping dictionary.
    """
    ids = list(set(ids))
    id2idx = {}
    for idx, val in enumerate(sorted(ids)):
        id2idx[val] = idx
    return id2idx


def slice_csr_matrix(csr: csr_matrix, row: int, col: int) -> int:
    """
    Returns csr[row, col] value not using slicing operation in csr matrix.
    When dimension of csr_matrix is too large, slicing such as csr_matrix[row, col]
    could be inefficient.

    Args:
        csr (csr_matrix): csr_matrix to be sliced.
        row (int): The index of the row to be sliced.
        col (int): The index of the column to be sliced.

    Returns (int):
        Sliced value.
    """
    indices = csr.indices[csr.indptr[row] : csr.indptr[row + 1]]
    data = csr.data[csr.indptr[row] : csr.indptr[row + 1]]
    for i, d in zip(indices, data):
        if i == col:
            return d
    return 0
