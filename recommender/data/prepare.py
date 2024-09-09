import pandas as pd
import torch
from torch.utils.data import Dataset


class MovieLensDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass