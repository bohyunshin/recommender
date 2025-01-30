MODEL_PATH = {
    # torch based models
    "svd": "recommender.model.mf.svd",
    "svd_bias": "recommender.model.mf.svd_bias",
    "gmf": "recommender.model.deep_learning.gmf",
    "mlp": "recommender.model.deep_learning.mlp",
    "two_tower": "recommender.model.deep_learning.two_tower",
    # csr based models
    "als": "recommender.model.mf.als",
    "user_based": "recommender.model.neighborhood.user_based",
}
