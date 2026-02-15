from enum import StrEnum


class NegativeSamplingStrategy(StrEnum):
    IN_BATCH = "in_batch"
    RANDOM_FROM_TOTAL_POOL = "random_from_total_pool"
