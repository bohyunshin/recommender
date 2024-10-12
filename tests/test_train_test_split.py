import pandas as pd
import numpy as np

from preprocess.train_test_split import TrainTestSplit


def test_train_test_split():
    split = TrainTestSplit(0.2, True)

    # test case 1
    df = pd.DataFrame({
        "user_id": [1, 1, 1, 1, 1, 2, 2, 2, 2, 3],
        "item_id": [5, 10, 2, 3, 3, 4, 6, 2, 1, 9],
        "interaction": [1] * 10,
        "timestamp": range(10)
    })
    train, val = split.split(df)
    df = df.drop("timestamp", axis=1)
    train_expected = df.iloc[[0, 1, 2, 5, 6, 7, 9]]
    val_expected = df.iloc[[3, 4, 8]]

    np.testing.assert_array_equal(train.values, train_expected.values)
    np.testing.assert_array_equal(val.values, val_expected.values)

    # test case 2
    df = pd.DataFrame({
        "user_id": [1, 2, 3, 4],
        "item_id": [100, 101, 102, 103],
        "interaction": [1] * 4,
        "timestamp": range(4)
    })
    train, val = split.split(df)
    df = df.drop("timestamp", axis=1)
    train_expected = df.iloc[[0, 1, 2, 3]]
    val_expected = df.iloc[[]]

    np.testing.assert_array_equal(train.values, train_expected.values)
    np.testing.assert_array_equal(val.values, val_expected.values)