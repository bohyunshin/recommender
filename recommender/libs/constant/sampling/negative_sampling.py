from enum import Enum


class NegativeSamplingStrategy(Enum):
    IN_BATCH = "in_batch"
    RANDOM_FROM_TOTAL_POOL = "random_from_total_pool"
