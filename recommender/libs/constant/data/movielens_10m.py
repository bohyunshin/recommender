from enum import StrEnum

from recommender.libs.constant.data.name import Field


class MovieLens10mPath(StrEnum):
    """
    Enum for movielens 1m dataset path
    """

    ratings = "recommender/.data/movielens/ml-10M100K/ratings.dat"
    tags = "recommender/.data/movielens/ml-10M100K/tags.dat"
    items = "recommender/.data/movielens/ml-10M100K/movies.dat"


RATINGS_COLUMNS = [
    Field.USER_ID,
    Field.ITEM_ID,
    Field.INTERACTION,
    "timestamp",
]
TAGS_COLUMNS = [Field.USER_ID, Field.ITEM_ID, "tag", "timestamp"]
ITEMS_COLUMNS = [Field.ITEM_ID, "movie_name", "genres"]
