import os
import pandas as pd

from tools.csr import dataframe_to_csr, mapping_index
from preprocess.preprocess_base import PreoprocessorBase
from preprocess.train_test_split import TrainTestSplit, train_test_split


class Preprocessor(PreoprocessorBase):
    def __init__(self, **kwargs):
        super().__init__()
        self.test_ratio = kwargs["test_ratio"]
        self.random_state = kwargs["random_state"]
        # ratings = pd.read_csv()
        # train_test_split = TrainTestSplit(test_size=kwargs["test_size"], ts=True)
        #
        # col_mapping = {"userId":"user_id", "movieId":"item_id", "rating":"interactions", "timestamp":"timestamp"}
        # ratings = ratings.rename(columns=col_mapping)
        # ratings = ratings.sort_values(by=["user_id", "timestamp"])

        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),f".movielens/{kwargs['movielens_data_type']}/ratings.dat")
        ratings_cols = ["user_id", "item_id", "interactions", "timestamp"]
        ratings = pd.read_csv(path, sep='::', names=ratings_cols, engine='python', encoding="ISO-8859-1")

        self.user_mapping = mapping_index(ratings["user_id"])
        self.item_mapping = mapping_index(ratings["item_id"])
        self.num_users = len(self.user_mapping)
        self.num_items = len(self.item_mapping)
        self.shape = len(self.user_mapping), len(self.item_mapping)

        ratings["user_id"] = ratings["user_id"].map(self.user_mapping)
        ratings["item_id"] = ratings["item_id"].map(self.item_mapping)

        self.ratings = dataframe_to_csr(ratings, self.shape, True)

        # self.train, self.val = train_test_split.split(ratings)

    def preprocess(self):
        # csr_train = dataframe_to_csr(self.train, self.shape, True)
        # csr_val = dataframe_to_csr(self.val, self.shape, True)
        csr_train, csr_val = train_test_split(self.ratings, self.test_ratio, self.random_state)
        return csr_train, csr_val

if __name__ == "__main__":
    preproc = Preprocessor(movielens_data_type="ml-latest-small", test_size=0.2)
    csr_train, csr_val = preproc.preprocess()
    print("hello")