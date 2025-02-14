import pytest

from recommender.train import main


@pytest.mark.parametrize(
    "setup_config",
    [
        # svd
        ("movielens_1m", "svd", "mse", False, None, []),
        ("movielens_1m", "svd", "bce", True, 2, ["in_batch", "random_from_total_pool"]),
        ("movielens_1m", "svd", "bpr", True, 2, ["in_batch", "random_from_total_pool"]),
        ("movielens_10m", "svd", "mse", False, None, []),
        (
            "movielens_10m",
            "svd",
            "bce",
            True,
            2,
            ["in_batch", "random_from_total_pool"],
        ),
        (
            "movielens_10m",
            "svd",
            "bpr",
            True,
            2,
            ["in_batch", "random_from_total_pool"],
        ),
        # svd_bias
        ("movielens_1m", "svd_bias", "mse", False, None, []),
        (
            "movielens_1m",
            "svd_bias",
            "bce",
            True,
            2,
            ["in_batch", "random_from_total_pool"],
        ),
        (
            "movielens_1m",
            "svd_bias",
            "bpr",
            True,
            2,
            ["in_batch", "random_from_total_pool"],
        ),
        ("movielens_10m", "svd_bias", "mse", False, None, []),
        (
            "movielens_10m",
            "svd_bias",
            "bce",
            True,
            2,
            ["in_batch", "random_from_total_pool"],
        ),
        (
            "movielens_10m",
            "svd_bias",
            "bpr",
            True,
            2,
            ["in_batch", "random_from_total_pool"],
        ),
        # gmf
        ("movielens_1m", "gmf", "bce", True, 2, ["in_batch", "random_from_total_pool"]),
        ("movielens_1m", "gmf", "bpr", True, 2, ["in_batch", "random_from_total_pool"]),
        (
            "movielens_10m",
            "gmf",
            "bce",
            True,
            2,
            ["in_batch", "random_from_total_pool"],
        ),
        (
            "movielens_10m",
            "gmf",
            "bpr",
            True,
            2,
            ["in_batch", "random_from_total_pool"],
        ),
        # mlp
        ("movielens_1m", "mlp", "bce", True, 2, ["in_batch", "random_from_total_pool"]),
        ("movielens_1m", "mlp", "bpr", True, 2, ["in_batch", "random_from_total_pool"]),
        (
            "movielens_10m",
            "mlp",
            "bce",
            True,
            2,
            ["in_batch", "random_from_total_pool"],
        ),
        (
            "movielens_10m",
            "mlp",
            "bpr",
            True,
            2,
            ["in_batch", "random_from_total_pool"],
        ),
        # two_tower
        (
            "movielens_1m",
            "two_tower",
            "bce",
            True,
            2,
            ["in_batch", "random_from_total_pool"],
        ),
        (
            "movielens_1m",
            "two_tower",
            "bpr",
            True,
            2,
            ["in_batch", "random_from_total_pool"],
        ),
        (
            "movielens_10m",
            "two_tower",
            "bce",
            True,
            2,
            ["in_batch", "random_from_total_pool"],
        ),
        (
            "movielens_10m",
            "two_tower",
            "bpr",
            True,
            2,
            ["in_batch", "random_from_total_pool"],
        ),
    ],
    indirect=["setup_config"],
)
def test_train(setup_config):
    main(setup_config)
