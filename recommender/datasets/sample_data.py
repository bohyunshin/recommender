from scipy.sparse import csr_matrix
import numpy as np

from mf.als import AlternatingLeastSquares

if __name__ == '__main__':
    user_items = np.array(
        [[2,0,0,0,3,4],
         [0,0,2,1,0,0],
         [5,1,0,0,2,1]]
    )
    als = AlternatingLeastSquares()
    als.fit(csr_matrix(user_items))
    print('hi')