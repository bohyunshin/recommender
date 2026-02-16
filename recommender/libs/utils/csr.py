from collections import defaultdict
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from tqdm import tqdm

from recommender.libs.constant.data.name import Field

try:
    from recommender.libs.utils._csr_ops import (
        calculate_user_sim_row_cy,
        slice_csr_matrix_cy,
    )

    _HAS_CYTHON = True
except ImportError:
    _HAS_CYTHON = False


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
    assert Field.USER_ID in df.columns
    assert Field.ITEM_ID in df.columns
    assert Field.INTERACTION in df.columns

    user2item2value = defaultdict(dict)

    user_ids = range(shape[0])

    for user, item, interaction in zip(
        df[Field.USER_ID], df[Field.ITEM_ID], df[Field.INTERACTION]
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
    if _HAS_CYTHON:
        return slice_csr_matrix_cy(
            np.asarray(csr.data, dtype=np.float64),
            np.asarray(csr.indices, dtype=np.int32),
            np.asarray(csr.indptr, dtype=np.int32),
            row,
            col,
        )
    indices = csr.indices[csr.indptr[row] : csr.indptr[row + 1]]
    data = csr.data[csr.indptr[row] : csr.indptr[row + 1]]
    for i, d in zip(indices, data):
        if i == col:
            return d
    return 0


def calculate_user_sim(
    user_ids: NDArray, csr: csr_matrix
) -> Dict[Tuple[int, int], float]:
    n = len(user_ids)
    res = {}

    if _HAS_CYTHON:
        data_arr = np.asarray(csr.data, dtype=np.float64)
        indices_arr = np.asarray(csr.indices, dtype=np.int32)
        indptr_arr = np.asarray(csr.indptr, dtype=np.int32)
        user_ids_arr = np.asarray(user_ids, dtype=np.int64)
        for i in tqdm(range(n), desc="User similarity"):
            row_res = calculate_user_sim_row_cy(
                data_arr, indices_arr, indptr_arr, user_ids_arr, i
            )
            res.update(row_res)
        return res

    for i in tqdm(range(n), desc="User similarity"):
        x = int(user_ids[i])
        items_x = csr.indices[csr.indptr[x] : csr.indptr[x + 1]]
        data_x = csr.data[csr.indptr[x] : csr.indptr[x + 1]]

        r_x = sum(v**2 for v in data_x) ** 0.5
        if r_x == 0.0:
            for j in range(i + 1, n):
                res[(x, int(user_ids[j]))] = 0.0
            continue

        for j in range(i + 1, n):
            y = int(user_ids[j])
            r_xy = 0.0
            for k, col_x in enumerate(items_x):
                r_xy += data_x[k] * slice_csr_matrix(csr, y, col_x)

            if r_xy == 0.0:
                res[(x, y)] = 0.0
                continue

            data_y = csr.data[csr.indptr[y] : csr.indptr[y + 1]]
            r_y = sum(v**2 for v in data_y) ** 0.5
            res[(x, y)] = r_xy / (r_x * r_y)

    return res
