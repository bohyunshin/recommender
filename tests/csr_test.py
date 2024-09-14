import numpy as np

from tools.csr import dataframe_to_csr
import pandas as pd


def test_dataframe_to_csr():
    df = pd.DataFrame({
        "user_id": [1, 2, 3, 1, 1, 1, 2, 3, 4],
        "item_id": [100, 203, 404, 100, 100, 100, 100, 203, 100],
        "interactions": [1, 1, 1, 1, 1, 1, 1, 1, 1]
    })
    expected = np.array([
        [4,0,0],
        [1,1,0],
        [0,1,1],
        [1,0,0]
    ])
    csr, _, _ = dataframe_to_csr(df, True)
    assert np.array_equal(csr.toarray(), expected)