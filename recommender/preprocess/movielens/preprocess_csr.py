import os
import pandas as pd

from tools.csr import dataframe_to_csr, mapping_index
from preprocess.movielens.preprocess_movielens_base import PreoprocessorMovielensBase
from preprocess.train_test_split import TrainTestSplit, train_test_split


class Preprocessor(PreoprocessorMovielensBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def preprocess(self, **kwargs):
        test_ratio = kwargs["test_ratio"]
        random_state = kwargs["random_state"]
        shape = self.num_users, self.num_items
        csr_train, csr_val = train_test_split(dataframe_to_csr(self.ratings, shape, True),
                                              test_ratio,
                                              random_state)
        return csr_train, csr_val

if __name__ == "__main__":
    preproc = Preprocessor(movielens_data_type="ml-1m", test=False)
    csr_train, csr_val = preproc.preprocess(test_ratio=0.2, random_state=42)
    print("hello")