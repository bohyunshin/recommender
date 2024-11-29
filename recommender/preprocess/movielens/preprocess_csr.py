from tools.csr import dataframe_to_csr
from preprocess.movielens.preprocess_movielens_base import PreoprocessorMovielensBase
from preprocess.train_test_split import train_test_split


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
