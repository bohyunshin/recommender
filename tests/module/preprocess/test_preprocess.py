import importlib
import pandas as pd
import pytest

from recommender.libs.constant.data.name import Field


@pytest.mark.parametrize(
    "dataset",
    [
        "movielens_1m",
        "movielens_10m",
        "pinterest",
    ],
)
def test_preprocess(dataset):
    df = pd.DataFrame(
        {
            Field.USER_ID: [1, 1, 2, 2, 3],
            Field.ITEM_ID: [100, 101, 102, 105, 106],
            Field.INTERACTION: [1, 1, 1, 1, 1],
        }
    )
    data = {Field.INTERACTION: df}
    preprocess_module = importlib.import_module(
        f"recommender.preprocess.{dataset}"
    ).Preprocessor
    data = preprocess_module().preprocess(data)
    assert isinstance(data.get(Field.INTERACTION), pd.DataFrame)
    assert isinstance(data.get(Field.NUM_USERS), int)
    assert isinstance(data.get(Field.NUM_ITEMS), int)
    assert isinstance(data.get(Field.USER_ID2IDX), dict)
    assert isinstance(data.get(Field.ITEM_ID2IDX), dict)
