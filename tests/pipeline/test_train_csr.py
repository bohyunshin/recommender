import pytest

from recommender.train_csr import main


@pytest.mark.parametrize(
    "setup_config",
    [
        # als
        ("movielens", "als", "als", False, None, []),
        ("movielens_10m", "als", "als", False, None, []),
        # user_based
        ("movielens", "user_based", "not_defined", False, None, []),
        ("movielens_10m", "user_based", "not_defined", False, None, []),
    ],
    indirect=["setup_config"],
)
def test_train_csr(setup_config):
    main(setup_config)
