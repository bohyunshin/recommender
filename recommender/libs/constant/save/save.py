from enum import Enum


class FileName(Enum):
    LOG = "log.log"
    WEIGHT_PT = "weight.pt"
    MODEL_PKL = "model.pkl"
    TRAINING_LOSS = "training_loss.pkl"
    VALIDATION_LOSS = "validation_loss.pkl"
    METRIC = "metric.pkl"