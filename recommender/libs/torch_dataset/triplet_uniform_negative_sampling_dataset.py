from typing import Dict, Tuple

import numpy as np
import logging

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
        Pytorch dataset class performing negative sampling when triplet loss.

        Args:
            X (torch.Tensor): Input tensor consisting of user_id and item_id in order.
            user_items_dct (Dict[int, Dict[int, bool]]): Dictionary to store which item user liked.
        """
        self.X = X
        self.user_items_dct = user_items_dct
        self.num_items = kwargs["num_items"]
        self.num_neg = kwargs["num_neg"]

    def negative_sampling(self) -> None:
        """
        Perform negative random sampling while specified `self.num_neg`.
        This is done in advance before training loop.
        When random sampling, exclude item_id that is already liked by user using `self.user_items_dct`.
        Negative samples are stored in `self.triplet` and concatenated with positive sample in triplet form.
        While training loop, using dataloader in pytorch, just take out one of triplet samples using `__getitem__`.
        Note that when triplet loss, `y` does not play a role, therefore `self.label` is dummy tensor.
        """
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

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx) -> Tuple[int, int, int, int]:
        """
        Triplet indexing with user_id, pos_item_id, neg_item_id in order.
        """
        u, i, j = self.triplet[idx]
        return u, i, j, -1 # last index is reserved for y, which is unnecessary for bpr
