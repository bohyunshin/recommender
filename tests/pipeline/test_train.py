import pytest

from recommender.train import main


@pytest.mark.parametrize(
    "setup_config",
    [
        ("movielens", "svd", "mse", False, None, []),
        ("movielens", "svd", "bce", True, 2, ["in_batch", "random_from_total_pool"]),
        ("movielens", "svd", "bpr", True, 2, ["in_batch", "random_from_total_pool"]),
        ("movielens", "svd_bias", "mse", False, None, []),
        ("movielens", "svd_bias", "bce", True, 2, ["in_batch", "random_from_total_pool"]),
        ("movielens", "svd_bias", "bpr", True, 2, ["in_batch", "random_from_total_pool"]),
        ("movielens", "gmf", "bce", True, 2, ["in_batch", "random_from_total_pool"]),
        ("movielens", "gmf", "bpr", True, 2, ["in_batch", "random_from_total_pool"]),
        ("movielens", "mlp", "bce", True, 2, ["in_batch", "random_from_total_pool"]),
        ("movielens", "mlp", "bpr", True, 2, ["in_batch", "random_from_total_pool"]),
        ("movielens", "two_tower", "bce", True, 2, ["in_batch", "random_from_total_pool"]),
        ("movielens", "two_tower", "bpr", True, 2, ["in_batch", "random_from_total_pool"]),
    ],
    indirect=["setup_config"]
)
def test_train(setup_config):
    main(setup_config)
