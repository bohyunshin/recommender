import os
import pandas as pd
from tools.csr import dataframe_to_csr
from data.preprocess_base import PreoprocessorBase


class Preprocessor(PreoprocessorBase):
    def __init__(self, user_id_col, item_id_col, interactions_col, **kwargs):
        super().__init__()
        columns = [user_id_col, item_id_col, interactions_col]
        self.ratings = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),f".movielens/{kwargs['movielens_data_type']}/ratings.csv"))[columns]
        col_mapping = {user_id_col:"user_id", item_id_col:"item_id", interactions_col:"interactions"}
        self.ratings = self.ratings.rename(columns=col_mapping)

    def preprocess(self):
        csr, user_mapping, item_mapping = dataframe_to_csr(self.ratings, True)
        self.user_mapping = user_mapping
        self.item_mapping = item_mapping
        return csr

if __name__ == "__main__":
    preproc = Preprocessor("userId", "movieId", "rating", movielens_data_type="ml-latest-small")
    csr = preproc.preprocess()
    print("hello")