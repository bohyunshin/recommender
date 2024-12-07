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
        self.neg_samples = []
        is_sampled = {}
        for pos in self.X:
            u, i = pos
            # if already sampled, pass
            if is_sampled.get(u) == True:
                continue
            is_sampled[u] = True
            neg_samples_per_user = []
            for _ in range(self.num_neg):
                j = np.random.randint(self.num_items)  # sample only ONE negative sample
                while self.user_items_dct[u.item()].get(j) != None or j in neg_samples_per_user:
                    j = np.random.randint(self.num_items)
                self.neg_samples.append((u,j))
                neg_samples_per_user.append(j)
        logging.info(f"number of negative samples: {len(self.neg_samples)}")
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
