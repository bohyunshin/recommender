from enum import Enum


class DatasetName(Enum):
    MOVIELENS_1M = "movielens_1m"
    MOVIELENS_10M = "movielens_10m"
    PINTEREST = "pinterest"


class Field(Enum):
    INTERACTION = "interaction"
    USER_ID = "user_id"
    ITEM_ID = "item_id"
    NUM_USERS = "num_users"
    NUM_ITEMS = "num_items"
    USER_ID2IDX = "user_id2idx"
    ITEM_ID2IDX = "item_id2idx"
    X_TRAIN = "X_train"
    Y_TRAIN = "y_train"
    X_VAL = "X_val"
    Y_VAL = "y_val"
    TRAIN = "train"
    VAL = "val"


INTEGRATED_DATASET = [
    DatasetName.MOVIELENS_1M.value,
    DatasetName.MOVIELENS_10M.value,
    DatasetName.PINTEREST.value,
]
