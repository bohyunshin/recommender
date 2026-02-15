from enum import StrEnum


class ModelName(StrEnum):
    ALS = "als"
    USER_BASED = "user_based"
    SVD = "svd"
    SVD_BIAS = "svd_bias"
    GMF = "gmf"
    MLP = "mlp"
    TWO_TOWER = "two_tower"


class ModelForwardArgument(StrEnum):
    USER_IDX = "user_idx"
    ITEM_IDX = "item_idx"
    POS_ITEM_IDX = "pos_item_idx"
    NEG_ITEM_IDX = "neg_item_idx"
    Y = "y"


IMPLEMENTED_MODELS = [
    ModelName.ALS,
    ModelName.USER_BASED,
    ModelName.SVD,
    ModelName.SVD_BIAS,
    ModelName.GMF,
    ModelName.MLP,
    ModelName.TWO_TOWER,
]
