from enum import Enum


class Metric(Enum):
    """
    Enum for metric when there are no candidates
    """

    AP = "ap"
    MAP = "map"
    NDCG = "ndcg"
    RECALL = "recall"
    COUNT = "count"
