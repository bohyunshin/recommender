from enum import Enum


class DatasetName(Enum):
    MOVIELENS = "movielens"


INTEGRATED_DATASET = [
    DatasetName.MOVIELENS.value,
]
