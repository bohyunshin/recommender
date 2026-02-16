# cython: boundscheck=False, wraparound=False

import numpy as np
cimport numpy as np


def slice_csr_matrix_cy(
    np.ndarray[np.float64_t, ndim=1] data,
    np.ndarray[np.int32_t, ndim=1] indices,
    np.ndarray[np.int32_t, ndim=1] indptr,
    int row,
    int col,
) -> float:
    cdef int start = indptr[row]
    cdef int end = indptr[row + 1]
    cdef int i

    for i in range(start, end):
        if indices[i] == col:
            return data[i]
    return 0.0


def calculate_user_sim_row_cy(
    np.ndarray[np.float64_t, ndim=1] data,
    np.ndarray[np.int32_t, ndim=1] indices,
    np.ndarray[np.int32_t, ndim=1] indptr,
    np.ndarray[np.int64_t, ndim=1] user_ids,
    int row_idx,
):
    cdef int n = user_ids.shape[0]
    cdef int j, k, m
    cdef long long x, y
    cdef int x_start, x_end, y_start, y_end
    cdef double r_x, r_y, r_xy, val
    cdef int col_x

    res = {}
    x = user_ids[row_idx]
    x_start = indptr[x]
    x_end = indptr[x + 1]

    # Precompute norm for user x
    r_x = 0.0
    for k in range(x_start, x_end):
        val = data[k]
        r_x += val * val
    if r_x == 0.0:
        for j in range(row_idx + 1, n):
            y = user_ids[j]
            res[(x, y)] = 0.0
        return res
    r_x = r_x ** 0.5

    for j in range(row_idx + 1, n):
        y = user_ids[j]
        y_start = indptr[y]
        y_end = indptr[y + 1]

        # Dot product: for each item in x's row, linear-scan y's row
        r_xy = 0.0
        for k in range(x_start, x_end):
            col_x = indices[k]
            for m in range(y_start, y_end):
                if indices[m] == col_x:
                    r_xy += data[k] * data[m]
                    break

        if r_xy == 0.0:
            res[(x, y)] = 0.0
            continue

        # Compute norm for user y only when needed
        r_y = 0.0
        for m in range(y_start, y_end):
            val = data[m]
            r_y += val * val
        r_y = r_y ** 0.5

        res[(x, y)] = r_xy / (r_x * r_y)

    return res
