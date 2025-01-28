from enum import Enum


class LossName(Enum):
    BPR = "bpr"
    BCE = "bce"
    MSE = "mse"


IMPLEMENTED_LOSS = [
    LossName.BPR.value,
    LossName.BCE.value,
    LossName.MSE.value,
]