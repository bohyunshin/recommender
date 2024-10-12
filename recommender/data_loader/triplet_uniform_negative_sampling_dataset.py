import numpy as np
import torch
from torch.utils.data import Dataset


class Data(Dataset):
    def __init__(self, X, user_items_dct, **kwargs):
        self.X = X
        self.user_items_dct = user_items_dct
        self.num_items = kwargs["num_items"]

    def negative_sampling(self):
        self.triplet = []
        for pos in self.X:
            u, i = pos
            j = np.random.randint(self.num_items)  # sample only ONE negative sample
            while self.user_items_dct[u.item()].get(j) != None:
                j = np.random.randint(self.num_items)
            self.triplet.append((u,i,j))
        self.label = [1]*len(self.triplet)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        u, i, j = self.triplet[idx]
        return u, i, j, -1 # last index is reserved for y, which is unnecessary for bpr
