from enum import StrEnum


class Metric(StrEnum):
    """
    Enum for metric when there are no candidates
    """

    AP = "ap"
    MAP = "map"
    NDCG = "ndcg"
    RECALL = "recall"
    COUNT = "count"
