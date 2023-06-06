from mf.als import AlternatingLeastSquares
from datasets.movielens import get_movielens

if __name__ == '__main__':
    titles, ratings = get_movielens(variant="1m")
    als = AlternatingLeastSquares()
    als.fit(ratings)