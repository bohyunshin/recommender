import os
import pandas as pd

from tools.csr import dataframe_to_csr, mapping_index
from data.preprocess_base import PreoprocessorBase
from data.train_test_split import TrainTestSplit


class Preprocessor(PreoprocessorBase):
    def __init__(self, **kwargs):
        super().__init__()
        ratings = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),f".movielens/{kwargs['movielens_data_type']}/ratings.csv"))
        train_test_split = TrainTestSplit(test_size=kwargs["test_size"], ts=True)

        col_mapping = {"userId":"user_id", "movieId":"item_id", "rating":"interactions", "timestamp":"timestamp"}
        ratings = ratings.rename(columns=col_mapping)
        ratings = ratings.sort_values(by=["user_id", "timestamp"])

        self.user_mapping = mapping_index(ratings["user_id"])
        self.item_mapping = mapping_index(ratings["item_id"])
        self.shape = len(self.user_mapping), len(self.item_mapping)

        ratings["user_id"] = ratings["user_id"].map(self.user_mapping)
        ratings["item_id"] = ratings["item_id"].map(self.item_mapping)

        self.train, self.val = train_test_split.split(ratings)

    def preprocess(self):
        csr_train = dataframe_to_csr(self.train, self.shape, True)
        csr_val = dataframe_to_csr(self.val, self.shape, True)
        return csr_train, csr_val

if __name__ == "__main__":
    preproc = Preprocessor(movielens_data_type="ml-latest-small", test_size=0.2)
    csr_train, csr_val = preproc.preprocess()
    print("hello")