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
        is_sampled = {}
        for pos in self.X:
            u, i = pos
            neg_samples_per_pos_user = []
            for _ in range(self.num_neg):
                j = np.random.randint(self.num_items)  # sample only ONE negative sample
                while self.user_items_dct[u.item()].get(j) != None or j in neg_samples_per_pos_user:
                    j = np.random.randint(self.num_items)
                self.triplet.append((u,i,j))
                neg_samples_per_pos_user.append(j)

                if is_sampled.get(u.item()) == True:
                    break

            is_sampled[u.item()] = True
        logging.info(f"number of total samples: {len(self.triplet)}")
        self.label = torch.tensor([1.0]).expand(len(self.triplet))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        u, i, j = self.triplet[idx]
        return u, i, j, -1 # last index is reserved for y, which is unnecessary for bpr
