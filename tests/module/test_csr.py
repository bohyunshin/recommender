import os
import sys

import numpy as np
import pandas as pd

sys.path.append(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../recommender")
)

from recommender.libs.utils.csr import dataframe_to_csr, mapping_index


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
