from typing import Dict, Union

import pandas as pd

from recommender.preprocess.preprocess_base_ import PreoprocessorBase


class Preprocessor(PreoprocessorBase):
    def __init__(self):
        super().__init__()

    def preprocess(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Union[pd.DataFrame, Dict[int, int]]]:
        # some other preprocessing logic
        return self.mapping(data=data)