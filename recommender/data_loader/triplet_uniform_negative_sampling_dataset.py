import numpy as np
import logging
import torch
from torch.utils.data import Dataset


class Data(Dataset):
    def __init__(self, X, user_items_dct, **kwargs):
        self.X = X
        self.user_items_dct = user_items_dct
        self.num_items = kwargs["num_items"]
        self.num_neg = kwargs["num_neg"]

    def negative_sampling(self):
        self.triplet = []
        for pos in self.X:
            u, i = pos
            neg_samples_per_pos_sample = []
            for _ in range(self.num_neg):
                j = np.random.randint(self.num_items)
                while self.user_items_dct[u.item()].get(j) != None or j in neg_samples_per_pos_sample:
                    j = np.random.randint(self.num_items)
                self.triplet.append((u,i,j))
                neg_samples_per_pos_sample.append(j)
        logging.info(f"number of negative samples: {len(self.triplet)}")
        self.label = torch.tensor([1.0]).expand(len(self.triplet))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        u, i, j = self.triplet[idx]
        return u, i, j, -1 # last index is reserved for y, which is unnecessary for bpr
