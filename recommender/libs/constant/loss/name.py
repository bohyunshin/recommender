from enum import Enum


class LossName(Enum):
    BPR = "bpr"
    BCE = "bce"
    MSE = "mse"
    ALS = "als"
    NOT_DEFINED = "not_defined"


IMPLEMENTED_LOSS = [e.value for e in LossName]