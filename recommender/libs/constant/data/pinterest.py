from enum import Enum

from recommender.libs.constant.data.name import Field


class PinterestPath(Enum):
    INTERACTIONS = "recommender/.data/pinterest/pinterest_iccv/subset_iccv_board_pins.bson"
    BOARDS = "recommender/.data/pinterest/pinterest_iccv/subset_iccv_board_cate.bson"
    PINS = "recommender/.data/pinterest/pinterest_iccv/subset_iccv_pin_im.bson"


class PinterestField(Enum):
    BOARD_ID = "board_id"
    PINS = "pins"


INTERACTIONS_COLUMNS = [
    Field.USER_ID.value,
    Field.ITEM_ID.value,
    Field.INTERACTION.value,
]