import pytest

from recommender.pipeline.base import BaseTrainPipeline
from recommender.train_csr import main


@pytest.mark.parametrize(
    "setup_config",
    [
        # als
        ("movielens_1m", "als", "als", False, None, []),
        ("movielens_10m", "als", "als", False, None, []),
        # user_based
        ("movielens_1m", "user_based", "not_defined", False, None, []),
        ("movielens_10m", "user_based", "not_defined", False, None, []),
    ],
    indirect=["setup_config"],
)
def test_train_csr(setup_config, mock_data_factory, monkeypatch):
    mock_data = mock_data_factory(implicit=setup_config.implicit)
    monkeypatch.setattr(BaseTrainPipeline, "_load_data", lambda self, args: mock_data)
    main(setup_config)
