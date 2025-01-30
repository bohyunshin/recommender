from typing import Tuple

import torch
from torch.utils.data import Dataset


class Data(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor, **kwargs):
        """
        Basic pytorch dataset class used in dataloader.
        This class does nothing special, just samples positive samples and y labels.
        """
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X[idx, 0], self.X[idx, 1], self.y[idx]  # user, item, rating
