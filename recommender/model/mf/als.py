import numpy as np
import time
from scipy.sparse import csr_matrix
import logging

from tools.utils import check_csr, check_random_state, nonzeros
from model.fit_model_base import FitModelBase

class Model(FitModelBase):
    def __init__(
            self,
            factors=10,
            regularization=0.01,
            alpha=0.1,
            dtype=np.float32,
            iterations=100,
            calculate_training_loss=False,
            random_state=42,
            **kwargs
    ):
        super().__init__()
        self.factors = factors
        self.regularization = regularization
        self.alpha = alpha
        self.dtype = dtype
        self.iterations = iterations
        self.calculate_training_loss = calculate_training_loss
        self.random_state = random_state

    def fit(self, user_items, val_user_items=None):
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

        Parameters
        ----------
        user_items : csr_matrix
            This is user x item matrix whose dimension is M x N. It is used in training step
        val_user_items : csr_matrix, optional
            This is user x item matrix whose dimension is M x N. It is used in validation step.
            Note that in training step, we do not use this matrix.

        Returns
        -------
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

    def update_user_factors(self, u, Q, Cui):
        """
        Update user embedding factors for user u using closed form optimization

        Parameters
        ----------
        u : integer
            Index for user u
        Q : np.array
            N x K item factors matrix
        Cui : csr_matrix
            M x N confidence sparse matrix

        Returns
        -------
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

    def update_item_factors(self, i, P, Ciu):
        """
        Update item embedding factors for item i using closed form optimization

        Parameters
        ----------
        i : integer
            Index for item i
        P : np.ndarray
            M x K user factors matrix
        Ciu : csr_matrix
            N x M confidence sparse matrix

        Returns
        -------
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

    def predict(self, user_idx, **kwargs):
        return np.dot(self.user_factors[user_idx], self.item_factors.T)

    def calculate_loss(self, user_items):
        """
        Calculates training/validation loss in each iteration.

        We calculate loss in each iteration to check if parameters are converged or not.
        Depending on the user_items argument, it calculates training or validation loss.

        It is strongly recommended that it should be checked whether validation loss drops
        and becomes stable

        Parameters
        ----------
        user_items : csr_matrix
            Training or validation user x item matrix

        Returns
        -------
        loss : float
            Calculated loss value
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

    def QtQ(self):
        Q = self.item_factors
        return Q.T.dot(Q)

    def PtP(self):
        P = self.user_factors
        return P.T.dot(P)

    def binarize(self, c):
        # binarize c
        return np.where(c >= 1, 1, 0)

    def linear_equation(self, A, b):
        return np.linalg.solve(A, b)

    def transform_Cui(self, Cui):
        indptr = Cui.indptr
        indices = Cui.indices
        data = [1 + self.alpha * c for c in Cui.data]
        return csr_matrix((data, indices, indptr))