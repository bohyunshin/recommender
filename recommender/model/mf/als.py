from typing import Optional
import time
import logging

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from recommender.libs.utils.utils import check_csr, check_random_state, nonzeros
from recommender.model.fit_model_base import FitModelBase

class Model(FitModelBase):
    def __init__(
            self,
            factors: Optional[int] = 10,
            regularization: Optional[float] = 0.01,
            alpha: Optional[float] = 0.1,
            dtype: Optional[np.dtypes] = np.float32,
            iterations: Optional[int] = 100,
            calculate_training_loss: Optional[bool] = False,
            random_state: Optional[int] = 42,
            **kwargs
    ):
        """
        Alternating Least Squares algorithm to factorize user x item matrix into each of embedding matrix.
        Reference: Collaborative Filtering for Implicit Feedback Datasets, Yifan Hu et al.

        Args:
            factors (int, optional): Dimension of user/item embeddings.
            regularization (float, optional): Regularization parameter balancing main loss and penalty term.
            alpha (float, optional): Alpha value in c_ui = 1 + \alpha r_ui. This controls the strength of positive samples.
            dtype (numpy.dtype, optional): Data type of the embedding matrix.
            iterations (int, optional): Maximum number of iterations.
            calculate_training_loss (bool, optional): Whether to calculate training loss.
            random_state (int, optional): Random seed value.
        """
        super().__init__()
        self.factors = factors
        self.regularization = regularization
        self.alpha = alpha
        self.dtype = dtype
        self.iterations = iterations
        self.calculate_training_loss = calculate_training_loss
        self.random_state = random_state

    def fit(
            self,
            user_items: csr_matrix,
            val_user_items: Optional[csr_matrix]
        ) -> None:
        """
        Factorizes the user_items matrix.
        While selected iterations, this method updates user and item factors using closed form
        derived in koren et al. who proposed calculating YtY in advance to reduce computation time

        As noted in Xiangnan et al., time complexity of original als is O((M+N)K^3 + |R|K^2)

        P : M x K user factors matrix
        Q : N x K item factors matrix
        Cui : M x N confidence matrix
        Rui : M x N binarized matrix
        Wu : N x N diagonal matrix whose (i,i) element is c_ui

        Args:
            user_items (csr_matrix): This is user x item matrix whose dimension
            is M x N. It is used in training step.
            val_user_items (csr_matrix, optional): This is user x item matrix
            whose dimension is M x N. It is used in validation step.
            Note that in training step, we do not use this matrix.
        """
        # initialize the random state
        random_state = check_random_state(self.random_state)

        Cui = check_csr(user_items)
        # Cui = self.transform_Cui(Cui)
        M,N = Cui.shape # number of users and items

        # initialize parameters randomly
        if self.user_factors is None:
            self.user_factors = random_state.rand(M, self.factors).astype(self.dtype) * 0.01 # M x K
        if self.item_factors is None:
            self.item_factors = random_state.rand(N, self.factors).astype(self.dtype) * 0.01 # N x K

        self._PtP = None
        self._QtQ = None
        self.tr_loss = []
        self.val_loss = []

        for iteration in range(self.iterations):

            logging.info(f"iteration: {iteration} out of {self.iterations}")

            start = time.time()

            # alternate updating user and item factors
            # update user factors
            self._QtQ = self.QtQ()
            for u in range(M):
                self.update_user_factors(u, self.item_factors, Cui)

            # update item factors
            self._PtP = self.PtP()
            for i in range(N):
                self.update_item_factors(i, self.user_factors, Cui.T.tocsr())

            # calculate training / validation loss
            tr_loss = self.calculate_loss(user_items)
            self.tr_loss.append(tr_loss)
            logging.info(f"training loss: {tr_loss}")

            if val_user_items is not None:
                val_loss = self.calculate_loss(val_user_items)
                self.val_loss.append(val_loss)
                logging.info(f"validation loss: {val_loss}")

            logging.info(f"executed time for {iteration} iteration: {time.time() - start}")

    def update_user_factors(
            self,
            u: int,
            Q: NDArray,
            Cui: csr_matrix,
        ) -> None:
        """
        Update user embedding factors for user u using closed form optimization

        Args:
            u (int): User index to be updated.
            Q (NDArray): N x K item factors matrix.
            Cui (csr_matrix): M x N confidence sparse matrix
        """
        N,K = Q.shape

        # A = QtW_uQ + regularization * I = QtQ + Qt(W_u-1)Q + regularization * I
        # b = QtW_ur_u
        # accumulate Qt(W_u-1)Q using outer product form
        A = self._QtQ + np.eye(K)*self.regularization
        b = np.zeros(K)
        for i, confidence in nonzeros(Cui, u):
            item_factor_i = Q[i]

            if confidence > 0:
                b += confidence * item_factor_i
            else:
                confidence *= -1

            A += np.outer(item_factor_i,item_factor_i) * (confidence - 1)

        # solve linear equation
        p_u = self.linear_equation(A, b)

        self.user_factors[u] = p_u

    def update_item_factors(
            self,
            i: int,
            P: NDArray,
            Ciu: csr_matrix
        ) -> None:
        """
        Update item embedding factors for item i using closed form optimization

        Args:
            i (int): Item index to be updated.
            P (NDArray): M x K user factors matrix.
            Ciu (csr_matrix): N x M confidence sparse matrix.
        """
        M, K = P.shape

        # A = QtW_uQ + regularization * I = QtQ + Qt(W_u-1)Q + regularization * I
        # b = QtW_ur_u
        # accumulate Qt(W_u-1)Q using outer product form
        A = self._PtP + np.eye(K)*self.regularization
        b = np.zeros(K)
        for j, confidence in nonzeros(Ciu, i):
            user_factor_j = P[j]

            if confidence > 0:
                b += confidence * user_factor_j
            else:
                confidence *= -1

            A += np.outer(user_factor_j,user_factor_j) * (confidence - 1)

        # solve linear equation
        q_i = self.linear_equation(A, b)

        self.item_factors[i] = q_i

    def predict(
            self,
            user_idx: NDArray,
            **kwargs
        ) -> NDArray:
        """
        Calculate prediction scores for target user_idx.
        Batch users are user_idx, and prediction scores associated with all items
        are calculated.

        Args:
            user_idx (NDArray): Batch user index.

        Returns (NDArray):
            Prediction scores NDArray whose dimension is (# of batch users, # of total items)
        """
        return np.dot(self.user_factors[user_idx], self.item_factors.T)

    def calculate_loss(
            self,
            user_items: csr_matrix,
        ) -> float:
        """
        Calculates training/validation loss in each iteration.

        We calculate loss in each iteration to check if parameters are converged or not.
        Depending on the user_items argument, it calculates training or validation loss.

        It is strongly recommended that it should be checked whether validation loss drops
        and becomes stable

        Args:
            user_items (csr_matrix): Training or validation user x item matrix.

        Returns (float):
            Calculated loss value.
        """
        loss = 0
        M,N = user_items.shape
        Q = self.item_factors
        total_confidence = 0
        for u in range(M):
            c_u = user_items[u].todense()
            total_confidence += c_u.sum()
            r_u = self.binarize(c_u)
            p_u = self.user_factors[u]
            r_u_hat = Q.dot(p_u)

            temp = np.multiply(c_u, np.power(r_u - r_u_hat, 2))
            loss += temp.sum()

            loss += self.regularization * np.power(p_u, 2).sum()

        for i in range(N):
            q_i = self.item_factors[i]
            loss += self.regularization * np.power(q_i, 2).sum()

        # todo: why divide loss? (from implicit github repo)
        return loss / (total_confidence + user_items.shape[0] * user_items.shape[1] - user_items.nnz)

    def QtQ(self) -> NDArray:
        """
        Helper function used when iterating als.
        When calculating closed forms of als, it is efficient to pre-calculate QtQ.

        Returns (NDArray):
            Calculated QtQ.
        """
        Q = self.item_factors
        return Q.T.dot(Q)

    def PtP(self) -> NDArray:
        """
        Helper function used when iterating als.
        When calculating closed forms of als, it is efficient to pre-calculate PtP.

        Returns (NDArray):
            Calculated PtP.
        """
        P = self.user_factors
        return P.T.dot(P)

    def binarize(
            self,
            c: NDArray,
        ):
        """
        Binarizes input NDArray making dataset as implicit.
        In implicit dataset, even if an user interacted with an item greater than one time,
        the user is regarded as interacting with an item one time.
        This model assumption inherently has disadvantage, therefore fixes it with `alpha` parameter.

        Args:
             c (NDArray): NDArray to be binarized.

        Returns (NDArray):
            Binarized NDArray.
        """
        return np.where(c >= 1, 1, 0)

    def linear_equation(
            self,
            A: NDArray,
            b: NDArray,
        ) -> NDArray:
        """
        Solves linear equation Ax = b.
        When deriving closed forms of als, need to solve this linear equation.

        Args:
            A (NDArray): Weight matrix in linear equation.
            b (NDArray): Bias vector in linear equation.

        Returns (NDArray):
            Solution of linear equation.
        """
        return np.linalg.solve(A, b)

    def transform_Cui(
            self,
            Cui: csr_matrix,
        ) -> csr_matrix:
        """
        Transforms Cui matrix using confidence parameter, `alpha`.
        If `alpha` equals 1, do not transform anything.
        Experiment with validation dataset should be conducted which `alpha` is best value.

        Args:
            Cui (csr_matrix): User x item csr matrix.

        Returns (csr_matrix):
            Transformed Cui matrix.
        """
        indptr = Cui.indptr
        indices = Cui.indices
        data = [1 + self.alpha * c for c in Cui.data]
        return csr_matrix((data, indices, indptr))