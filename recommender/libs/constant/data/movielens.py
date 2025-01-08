from enum import Enum


class MovieLens1mPath(Enum):
    """
    Enum for movielens 1m dataset path
    """
    ratings = "recommender/.data/movielens/ml-1m/ratings.dat"
    users = "recommender/.data/movielens/ml-1m/users.dat"
    items = "recommender/.data/movielens/ml-1m/movies.dat"


RATINGS_COLUMNS = ["user_id", "movie_id", "rating", "timestamp"]
USERS_COLUMNS = ["user_id", "gender", "age", "occupation", "zip_code"]
ITEMS_COLUMNS = ["movie_id", "movie_name", "genres"]