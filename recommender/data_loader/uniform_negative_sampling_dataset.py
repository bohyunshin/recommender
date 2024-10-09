import numpy as np
import torch
from torch.utils.data import Dataset


class Data(Dataset):
    def __init__(self, X, user_items_dct, **kwargs):
        self.X = X
        self.triplet_len = X.shape[0]
        self.user_items_dct = user_items_dct
        self.num_items = kwargs["num_items"]
        self.data_type = kwargs["data_type"] # triplet or bce

    def negative_sampling(self):
        self.triplet = []
        self.neg_samples = []
        for pos in self.X:
            u, i = pos
            j = np.random.randint(self.num_items)  # sample only ONE negative sample
            while self.user_items_dct[u].get(j) != None:
                j = np.random.randint(self.num_items)
            self.triplet.append((u,i,j))
            self.neg_samples.append((u,j))
        self.label = torch.tensor([1.]*len(self.X) + [0.]*len(self.neg_samples))
        self.X = torch.concat((self.X, torch.tensor(self.neg_samples)))

        # shuffle negative sampling result
        idx = torch.randperm(self.label.shape[0])
        self.label = self.label[idx]
        self.X = self.X[idx,:]

    def __len__(self):
        if self.data_type == "triplet":
            return self.triplet_len
        elif self.data_type == "bce":
            return len(self.label)

    def __getitem__(self, idx):
        if self.data_type == "triplet":
            u, i, j = self.triplet[idx]
            return u, i, j
        elif self.data_type == "bce":
            u, i = self.X[idx]
            y = self.label[idx]
            return u, i, y
