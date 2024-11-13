import torch
import torch.nn.functional as F

from preprocess.movielens.preprocess_movielens_base import PreoprocessorMovielensBase
from tools.utils import mapping_dict

class Preprocessor(PreoprocessorMovielensBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def preprocess(self):
        # mapping ids
        self.ratings["user_id"] = self.ratings["user_id"].map(self.user_id2idx)
        self.ratings["movie_id"] = self.ratings["movie_id"].map(self.movie_id2idx)
        self.users["user_id"] = self.users["user_id"].map(self.user_id2idx)
        self.movies["movie_id"] = self.movies["movie_id"].map(self.movie_id2idx)

        # generate one-hot encoded metadata
        # id in users and movies should be same ascending order with mapping dictionary
        assert self.users["user_id"].tolist() == sorted(list(self.user_id2idx.values()))
        assert self.movies["movie_id"].tolist() == sorted(list(self.movie_id2idx.values()))
        self.user_meta = self.get_user_meta(self.users)
        self.item_meta = self.get_item_meta(self.movies)

        X = torch.tensor(self.ratings[["user_id", "movie_id"]].values)
        y = torch.tensor(self.ratings[["rating"]].values, dtype=torch.float32)
        return X, y

    def get_user_meta(self, users):
        user_meta_cols = ["gender", "age", "occupation"]
        user_meta = torch.tensor([])
        for col in user_meta_cols:
            vals = users[col].tolist()
            mapping = mapping_dict(vals)
            num_classes = len(mapping)
            vals = [mapping[val] for val in vals]
            one_hot_vector = F.one_hot(torch.tensor(vals), num_classes=num_classes)
            user_meta = torch.concat((user_meta, one_hot_vector), dim=1)
        return user_meta

    def get_item_meta(self, movies):
        genres = movies["genres"].map(lambda x: x.split("|")).tolist()
        unique_genres = set()
        for genre in genres: # genre: ["Animation", "Action"]
            for g in genre:
                unique_genres.add(g)
        unique_genres = sorted(list(unique_genres))
        mapping_genres = mapping_dict(unique_genres)
        num_classes = len(mapping_genres)

        movie_meta = torch.tensor([])
        for genre in genres:
            genre_vector_for_one_movie = torch.tensor([0] * num_classes)
            for g in genre:
                g_encoded = mapping_genres[g]
                genre_vector_for_one_movie = genre_vector_for_one_movie + F.one_hot(torch.tensor([g_encoded]), num_classes=num_classes)
            movie_meta = torch.concat((movie_meta, genre_vector_for_one_movie), dim=0)
        return movie_meta