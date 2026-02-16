import os
import sys

import numpy as np
import pandas as pd

sys.path.append(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../recommender")
)

import pytest
from scipy.sparse import csr_matrix

from recommender.libs.utils.csr import (
    calculate_user_sim,
    dataframe_to_csr,
    mapping_index,
    slice_csr_matrix,
)


def test_dataframe_to_csr():
    df = pd.DataFrame(
        {
            "user_id": [1, 2, 3, 1, 1, 1, 2, 3, 4],
            "item_id": [100, 203, 404, 100, 100, 100, 100, 203, 100],
            "interaction": [1, 1, 1, 1, 1, 1, 1, 1, 1],
        }
    )
    user_mapping = mapping_index(df["user_id"])
    item_mapping = mapping_index(df["item_id"])
    df["user_id"] = df["user_id"].map(user_mapping)
    df["item_id"] = df["item_id"].map(item_mapping)
    expected = np.array([[4, 0, 0], [1, 1, 0], [0, 1, 1], [1, 0, 0]])
    csr = dataframe_to_csr(df, (4, 3), True)
    assert np.array_equal(csr.toarray(), expected)

    df = pd.DataFrame(
        {"user_id": [1, 2, 3, 4], "item_id": [3, 3, 2, 1], "interaction": [1, 1, 1, 1]}
    )
    expected = np.array(
        [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]]
    )
    csr = dataframe_to_csr(df, (5, 4), True)
    assert np.array_equal(csr.toarray(), expected)


class TestSliceCsrMatrix:
    def setup_method(self):
        # 3x4 matrix:
        # [[1, 0, 2, 0],
        #  [0, 0, 0, 0],
        #  [0, 3, 0, 4]]
        self.csr = csr_matrix(
            np.array([[1, 0, 2, 0], [0, 0, 0, 0], [0, 3, 0, 4]], dtype=np.float64)
        )

    def test_existing_values(self):
        assert slice_csr_matrix(self.csr, 0, 0) == 1
        assert slice_csr_matrix(self.csr, 0, 2) == 2
        assert slice_csr_matrix(self.csr, 2, 1) == 3
        assert slice_csr_matrix(self.csr, 2, 3) == 4

    def test_missing_values_return_zero(self):
        assert slice_csr_matrix(self.csr, 0, 1) == 0
        assert slice_csr_matrix(self.csr, 0, 3) == 0
        assert slice_csr_matrix(self.csr, 2, 0) == 0
        assert slice_csr_matrix(self.csr, 2, 2) == 0

    def test_empty_row(self):
        assert slice_csr_matrix(self.csr, 1, 0) == 0
        assert slice_csr_matrix(self.csr, 1, 1) == 0
        assert slice_csr_matrix(self.csr, 1, 2) == 0
        assert slice_csr_matrix(self.csr, 1, 3) == 0

    def test_cython_extension_loaded(self):
        from recommender.libs.utils.csr import _HAS_CYTHON

        if not _HAS_CYTHON:
            pytest.skip("Cython extension not compiled")
        from recommender.libs.utils._csr_ops import slice_csr_matrix_cy

        assert callable(slice_csr_matrix_cy)


class TestCalculateUserSim:
    def setup_method(self):
        # 4 users x 5 items matrix:
        #   User 0: items 0(1.0), 2(3.0)
        #   User 1: items 0(2.0), 1(1.0)
        #   User 2: items 2(4.0), 3(2.0), 4(1.0)
        #   User 3: (empty row)
        self.csr = csr_matrix(
            np.array(
                [
                    [1.0, 0.0, 3.0, 0.0, 0.0],
                    [2.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 4.0, 2.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ],
                dtype=np.float64,
            )
        )

    def test_basic_cosine_similarity(self):
        user_ids = np.array([0, 1, 2])
        result = calculate_user_sim(user_ids, self.csr)

        # Users 0,1 share item 0: dot=1*2=2, |u0|=sqrt(1+9)=sqrt(10), |u1|=sqrt(4+1)=sqrt(5)
        expected_01 = 2.0 / (10**0.5 * 5**0.5)
        assert result[(0, 1)] == pytest.approx(expected_01)

        # Users 0,2 share item 2: dot=3*4=12, |u0|=sqrt(10), |u2|=sqrt(16+4+1)=sqrt(21)
        expected_02 = 12.0 / (10**0.5 * 21**0.5)
        assert result[(0, 2)] == pytest.approx(expected_02)

    def test_no_overlap_returns_zero(self):
        user_ids = np.array([1, 2])
        result = calculate_user_sim(user_ids, self.csr)
        assert result[(1, 2)] == 0.0

    def test_empty_user_returns_zero(self):
        user_ids = np.array([0, 3])
        result = calculate_user_sim(user_ids, self.csr)
        assert result[(0, 3)] == 0.0

    def test_single_user(self):
        user_ids = np.array([0])
        result = calculate_user_sim(user_ids, self.csr)
        assert result == {}

    def test_result_keys_are_ordered_pairs(self):
        user_ids = np.array([0, 1, 2, 3])
        result = calculate_user_sim(user_ids, self.csr)
        for x, y in result.keys():
            assert x < y
