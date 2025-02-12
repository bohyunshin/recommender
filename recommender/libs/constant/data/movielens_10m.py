from enum import Enum


class MovieLens10mPath(Enum):
    """
    Enum for movielens 1m dataset path
    """

    ratings = "recommender/.data/movielens/ml-10M100K/ratings.dat"
    tags = "recommender/.data/movielens/ml-10M100K/tags.dat"
    items = "recommender/.data/movielens/ml-10M100K/movies.dat"


RATINGS_COLUMNS = ["user_id", "movie_id", "rating", "timestamp"]
TAGS_COLUMNS = ["user_id", "movie_id", "tag", "timestamp"]
ITEMS_COLUMNS = ["movie_id", "movie_name", "genres"]
