import numpy as np
from torch.utils.data import Dataset


class Data(Dataset):
    def __init__(self, X, user_items, **kwargs):
        self.X = X
        self.user_items = user_items

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        u,i = self.X[idx]
        j = np.random.randint(self.user_items.shape[1]) # sample only ONE negative sample
        while self.user_items[u].toarray().reshape(-1)[j] == 1:
            j = np.random.randint(self.user_items.shape[1])
        return u, i, j