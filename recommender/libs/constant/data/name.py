from enum import StrEnum


class DatasetName(StrEnum):
    MOVIELENS_1M = "movielens_1m"
    MOVIELENS_10M = "movielens_10m"
    PINTEREST = "pinterest"


class Field(StrEnum):
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
    DatasetName.MOVIELENS_1M,
    DatasetName.MOVIELENS_10M,
    DatasetName.PINTEREST,
]
