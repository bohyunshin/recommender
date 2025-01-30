import pytest

from recommender.train_csr import main


@pytest.mark.parametrize(
    "setup_config",
    [
        ("movielens", "als", "als", False, None, []),
        ("movielens", "user_based", "not_defined", False, None, []),
    ],
    indirect=["setup_config"],
)
def test_train_csr(setup_config):
    main(setup_config)
