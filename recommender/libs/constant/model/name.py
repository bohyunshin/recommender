from enum import Enum


class ModelName(Enum):
    ALS = "als"
    USER_BASED = "user_based"
    SVD = "svd"
    SVD_BIAS = "svd_bias"
    BPR = "bpr"
    GMF = "gmf"
    MLP = "mlp"
    TWO_TOWER = "two_tower"