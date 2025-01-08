import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),"../recommender"))

from recommender.libs.csr import dataframe_to_csr, mapping_index


def test_dataframe_to_csr():
    df = pd.DataFrame({
        "user_id": [1, 2, 3, 1, 1, 1, 2, 3, 4],
        "movie_id": [100, 203, 404, 100, 100, 100, 100, 203, 100],
        "rating": [1, 1, 1, 1, 1, 1, 1, 1, 1]
    })
    user_mapping = mapping_index(df["user_id"])
    item_mapping = mapping_index(df["movie_id"])
    df["user_id"] = df["user_id"].map(user_mapping)
    df["movie_id"] = df["movie_id"].map(item_mapping)
    expected = np.array([
        [4,0,0],
        [1,1,0],
        [0,1,1],
        [1,0,0]
    ])
    csr = dataframe_to_csr(df, (4,3), True)
    assert np.array_equal(csr.toarray(), expected)

    df = pd.DataFrame({
        "user_id": [1, 2, 3, 4],
        "movie_id": [3, 3, 2, 1],
        "rating": [1, 1, 1, 1]
    })
    expected = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0]
    ])
    csr = dataframe_to_csr(df, (5, 4), True)
    assert np.array_equal(csr.toarray(), expected)