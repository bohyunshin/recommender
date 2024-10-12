import numpy as np

from tools.csr import dataframe_to_csr, mapping_index
import pandas as pd


def test_dataframe_to_csr():
    df = pd.DataFrame({
        "user_id": [1, 2, 3, 1, 1, 1, 2, 3, 4],
        "item_id": [100, 203, 404, 100, 100, 100, 100, 203, 100],
        "interactions": [1, 1, 1, 1, 1, 1, 1, 1, 1]
    })
    user_mapping = mapping_index(df["user_id"])
    item_mapping = mapping_index(df["item_id"])
    df["user_id"] = df["user_id"].map(user_mapping)
    df["item_id"] = df["item_id"].map(item_mapping)
    expected = np.array([
        [4,0,0],
        [1,1,0],
        [0,1,1],
        [1,0,0]
    ])
    csr = dataframe_to_csr(df, (4,3), True)
    assert np.array_equal(csr.toarray(), expected)