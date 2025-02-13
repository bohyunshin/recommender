from enum import Enum


class ModelName(Enum):
    ALS = "als"
    USER_BASED = "user_based"
    SVD = "svd"
    SVD_BIAS = "svd_bias"
    GMF = "gmf"
    MLP = "mlp"
    TWO_TOWER = "two_tower"


class ModelForwardArgument(Enum):
    USER_IDX = "user_idx"
    ITEM_IDX = "item_idx"
    POS_ITEM_IDX = "pos_item_idx"
    NEG_ITEM_IDX = "neg_item_idx"
    Y = "y"


IMPLEMENTED_MODELS = [
    ModelName.ALS.value,
    ModelName.USER_BASED.value,
    ModelName.SVD.value,
    ModelName.SVD_BIAS.value,
    ModelName.GMF.value,
    ModelName.MLP.value,
    ModelName.TWO_TOWER.value,
]
