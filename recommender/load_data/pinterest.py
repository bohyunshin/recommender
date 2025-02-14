import pandas as pd
import numpy as np
from bson import decode_all

from recommender.load_data.base import LoadDataBase
from recommender.libs.constant.data.name import Field
from recommender.libs.constant.data.pinterest import PinterestPath, PinterestField, INTERACTIONS_COLUMNS


def read_bson_file(file_path: str):
    with open(file_path, "rb") as f:
        data = decode_all(f.read())
    return data


class LoadData(LoadDataBase):
    def __init__(self):
        super().__init__()

    def load(self, **kwargs):
        """
        Loads pinterest data.
        After downloading pinterest data using script in scripts/download/pinterest.py,
        convert data into pandas dataframe.
        Original data format is `bson`, which is binary json format. We convert this dataset into
        pandas dataframe for better compatability with current pipeline.

        Returns (Dict[str, pd.DataFrame]):
            Basically, abstractmethod `load` is designed to return one type of dataframes.
            Interaction dataset is target dataset to be loaded.
            When rating values exist in float type, interaction will be explicit dataset.
            When rating values does not exist, interaction will be implicit dataset.
        """
        board_pin_info = read_bson_file(PinterestPath.INTERACTIONS.value)
        interactions = []
        for board in board_pin_info:
            board_id = board.get(PinterestField.BOARD_ID.value)
            for pin_id in board.get(PinterestField.PINS.value):
                interactions.append((board_id, pin_id, 1.0)) # pinterest is implicit data
        interactions = pd.DataFrame(interactions)
        interactions.columns = INTERACTIONS_COLUMNS

        # for quick pytest
        if kwargs.get("is_test") is True:
            user_pools = interactions[Field.USER_ID.value].unique()
            sampled_user_ids = np.random.choice(user_pools, size=30, replace=False)
            interactions = interactions[lambda x: x[Field.USER_ID.value].isin(sampled_user_ids)]

        return {Field.INTERACTION.value: interactions}
