from typing import Tuple
from scipy.sparse import csr_matrix

from recommender.libs.csr import dataframe_to_csr
from recommender.preprocess.movielens.preprocess_movielens_base import PreoprocessorMovielensBase
from recommender.preprocess.train_test_split import train_test_split


class Preprocessor(PreoprocessorMovielensBase):
    def __init__(self, **kwargs):
        """
        Preprocessor to preprocess csr_matrix data.
        This class inherits from PreprocessorMovielensBase.
        """
        super().__init__(**kwargs)

    def preprocess(self, **kwargs) -> Tuple[csr_matrix, csr_matrix]:
        """
        Split rating data into train / validation dataset, in csr_matrix format.

        Returns (Tuple[csr_matrix, csr_matrix]):
            Tuple of train / validation dataset in csr_matrix.
        """
        test_ratio = kwargs["test_ratio"]
        random_state = kwargs["random_state"]
        shape = self.num_users, self.num_items
        csr_train, csr_val = train_test_split(
            ratings=dataframe_to_csr(
                df=self.ratings,
                shape=shape,
                implicit=True
            ),
            train_percentage=1-test_ratio,
            random_state=random_state
        )
        return csr_train, csr_val
