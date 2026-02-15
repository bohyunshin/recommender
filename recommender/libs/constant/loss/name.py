from enum import StrEnum


class LossName(StrEnum):
    BPR = "bpr"
    BCE = "bce"
    MSE = "mse"
    ALS = "als"
    NOT_DEFINED = "not_defined"


IMPLEMENTED_LOSS = list(LossName)
