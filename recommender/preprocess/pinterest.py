from typing import Dict, Union

import pandas as pd

from recommender.preprocess.base import PreprocessorBase


class Preprocessor(PreprocessorBase):
    def __init__(self):
        super().__init__()

    def preprocess(
        self, data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Union[pd.DataFrame, Dict[int, int]]]:
        """
        When preprocessing pinterest dataset, only mapping logic is included.

        Args:
            data (Dict[str, pd.DataFrame]): Return value from
                recommender/load_data/load_data_base.LoadDataBase.base will be used.

        Returns (Dict[str, Union[pd.DataFrame, Dict[int, int]]]):
            Preprocessed pandas dataframe and its mapping information.
        """
        # some other preprocessing logic
        return self.mapping(data=data)
