from typing import Dict, Tuple
import logging
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset


class Data(Dataset):
    def __init__(
            self,
            X: torch.Tensor,
            user_items_dct: Dict[int, Dict[int, bool]],
            **kwargs
        ):
        """
        Pytorch dataset class performing negative sampling when binary cross entropy loss.

        Args:
            X (torch.Tensor): Input tensor consisting of user_id and item_id in order.
            user_items_dct (Dict[int, Dict[int, bool]]): Dictionary to store which item user liked.
        """
        self.X = X
        self.user_items_dct = user_items_dct
        self.num_items = kwargs["num_items"]
        self.num_neg = kwargs["num_neg"]
        self.user_items_neg_dct = defaultdict(dict)

    def negative_sampling(self) -> None:
        """
        Perform negative random sampling while specified `self.num_neg`.
        This is done in advance before training loop.
        When random sampling, exclude item_id that is already liked by user.
        Negative samples are stored in `self.neg_samples` and concatenated with
        positive samples `self.X`, shuffling them.
        While training loop, using dataloader in pytorch, just take out positive or
        negative samples using `__getitem__` function.
        """
        self.neg_samples = []
        for pos in self.X:
            u, i = pos
            for _ in range(self.num_neg):
                j = np.random.randint(self.num_items)
                # if sampled item id is positive pair or
                # already sampled as a negative sample, resample item id
                while self.user_items_dct[u.item()].get(j) != None or self.user_items_neg_dct[u.item()].get(j) != None:
                    j = np.random.randint(self.num_items)
                self.neg_samples.append((u,j))
                self.user_items_neg_dct[u.item()][j] = True
        logging.info(f"number of negative samples: {len(self.neg_samples)}")
        self.label = torch.tensor([1.]*len(self.X) + [0.]*len(self.neg_samples))
        self.X = torch.concat((self.X, torch.tensor(self.neg_samples)))

        # shuffle negative sampling result
        idx = torch.randperm(self.label.shape[0])
        self.label = self.label[idx]
        self.X = self.X[idx,:]

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Used when pytorch dataloader samples data while training loop.
        """
        u, i = self.X[idx]
        y = self.label[idx]
        return u, i, y
