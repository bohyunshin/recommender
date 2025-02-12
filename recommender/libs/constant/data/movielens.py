from enum import Enum

from recommender.libs.constant.data.name import Field


class MovieLens1mPath(Enum):
    """
    Enum for movielens 1m dataset path
    """

    ratings = "recommender/.data/movielens/ml-1m/ratings.dat"
    users = "recommender/.data/movielens/ml-1m/users.dat"
    items = "recommender/.data/movielens/ml-1m/movies.dat"


# field commonly used across dataset is renamed with enum
RATINGS_COLUMNS = [Field.USER_ID.value, Field.ITEM_ID.value, Field.INTERACTION.value, "timestamp"]
USERS_COLUMNS = [Field.USER_ID.value, "gender", "age", "occupation", "zip_code"]
ITEMS_COLUMNS = [Field.ITEM_ID.value, "movie_name", "genres"]
