import os

import importlib
import pandas as pd
import pytest

from recommender.libs.constant.data.name import Field

os.chdir(
    os.path.join(
        os.path.dirname(__file__),
        "../../..",  # set to repository root path
    )
)


@pytest.mark.parametrize(
    "dataset",
    [
        "movielens_1m",
        "movielens_10m",
        "pinterest",
    ],
    indirect=False,
)
def test_load(dataset):
    load_data_module = importlib.import_module(
        f"recommender.load_data.{dataset}"
    ).LoadData
    data = load_data_module().load()
    assert isinstance(data.get(Field.INTERACTION.value), pd.DataFrame)
