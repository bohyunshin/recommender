from enum import Enum

from recommender.libs.constant.data.name import Field


class MovieLens10mPath(Enum):
    """
    Enum for movielens 1m dataset path
    """

    ratings = "recommender/.data/movielens/ml-10M100K/ratings.dat"
    tags = "recommender/.data/movielens/ml-10M100K/tags.dat"
    items = "recommender/.data/movielens/ml-10M100K/movies.dat"


RATINGS_COLUMNS = [
    Field.USER_ID.value,
    Field.ITEM_ID.value,
    Field.INTERACTION.value,
    "timestamp",
]
TAGS_COLUMNS = [Field.USER_ID.value, Field.ITEM_ID.value, "tag", "timestamp"]
ITEMS_COLUMNS = [Field.ITEM_ID.value, "movie_name", "genres"]
