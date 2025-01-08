import os
import sys
import pandas as pd

from libs.csr import dataframe_to_csr, mapping_index
from preprocess.preprocess_base import PreoprocessorBase
from preprocess.train_test_split import TrainTestSplit, train_test_split


class Preprocessor(PreoprocessorBase):
    def __init__(self, **kwargs):
        super().__init__()
        self.test_ratio = kwargs["test_ratio"]
        self.random_state = kwargs["random_state"]

        self.num_sim_user_top_N = int(sys.stdin.readline())
        self.num_item_rec_top_M = int(sys.stdin.readline())
        self.num_users = int(sys.stdin.readline())
        self.num_items = int(sys.stdin.readline())
        num_rows = int(sys.stdin.readline())
        ratings = []
        for _ in range(num_rows):
            rating = sys.stdin.readline().split(" ")
            rating[0] = int(rating[0])
            rating[1] = int(rating[1])
            rating[2] = float(rating[2])
            ratings.append(rating)
        num_reco_users = int(sys.stdin.readline())
        self.user_id_for_reco = []
        for _ in range(num_reco_users):
            self.user_id_for_reco.append(int(sys.stdin.readline()))

        ratings = pd.DataFrame(ratings, columns=["user_id", "item_id", "interactions"])

        self.user_mapping = mapping_index(ratings["user_id"])
        self.item_mapping = mapping_index(ratings["item_id"])
        self.num_users = len(self.user_mapping)
        self.num_items = len(self.item_mapping)
        self.shape = len(self.user_mapping), len(self.item_mapping)

        ratings["user_id"] = ratings["user_id"].map(self.user_mapping)
        ratings["item_id"] = ratings["item_id"].map(self.item_mapping)

        self.ratings = dataframe_to_csr(ratings, self.shape, True)

    def preprocess(self):
        csr_train, csr_val = train_test_split(self.ratings, self.test_ratio, self.random_state)
        return csr_train, csr_val
