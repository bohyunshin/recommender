import numpy as np
import torch
from torch.utils.data import Dataset


class Data(Dataset):
    def __init__(self, X, user_items_dct, **kwargs):
        self.X = X
        self.triplet_len = X.shape[0]
        self.user_items_dct = user_items_dct
        self.num_items = kwargs["num_items"]

    def negative_sampling(self):
        self.neg_samples = []
        for pos in self.X:
            u, i = pos
            j = np.random.randint(self.num_items)  # sample only ONE negative sample
            while self.user_items_dct[u.item()].get(j) != None:
                j = np.random.randint(self.num_items)
            self.neg_samples.append((u,j))
        self.label = torch.tensor([1.]*len(self.X) + [0.]*len(self.neg_samples))
        self.X = torch.concat((self.X, torch.tensor(self.neg_samples)))

        # shuffle negative sampling result
        idx = torch.randperm(self.label.shape[0])
        self.label = self.label[idx]
        self.X = self.X[idx,:]

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        u, i = self.X[idx]
        y = self.label[idx]
        return u, i, y
