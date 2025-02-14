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
        board_pin_info = read_bson_file(PinterestPath.INTERACTIONS.value)
        interactions = []
        for board in board_pin_info:
            board_id = board.get(PinterestField.BOARD_ID.value)
            for pin_id in board.get(PinterestField.PINS.value):
                interactions.append((board_id, pin_id, 1.0))
        interactions = pd.DataFrame(interactions)
        interactions.columns = INTERACTIONS_COLUMNS

        # for quick pytest
        if kwargs["is_test"]:
            user_pools = interactions[Field.USER_ID.value].unique()
            sampled_user_ids = np.random.choice(user_pools, size=30, replace=False)
            interactions = interactions[lambda x: x[Field.USER_ID.value].isin(sampled_user_ids)]

        return {Field.INTERACTION.value: interactions}
