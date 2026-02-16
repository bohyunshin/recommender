import logging
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from tqdm import tqdm

from recommender.libs.utils.csr import calculate_user_sim as compute_user_sim
from recommender.model.fit_model_base import FitModelBase


class Model(FitModelBase):
    def __init__(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        num_users: int,
        num_items: int,
        num_sim_user_top_N: int,
        loss_name: str,
        num_factors: int = 10,
        **kwargs,
    ):
        """
        User based collaborative filtering model.
        In this model, user gets recommendation based on likes of user's closest users.
        Definition of closeness depends on metric. This model defines it as cosine similarity.

        Args:
            num_users (int): Number of users.
            num_items (int): Number of items.
            num_sim_user_top_N (int): Number of similar users.
        """
        super().__init__(
            user_ids=user_ids,
            item_ids=item_ids,
            num_users=num_users,
            num_items=num_items,
            num_factors=num_factors,
            loss_name=loss_name,
        )
        self.num_sim_user_top_N = num_sim_user_top_N

    def fit(
        self,
        user_items: csr_matrix,
        val_user_items: csr_matrix,
    ) -> None:
        """
        Fit user based model.

        1. Calculate cosine similarity of between every user.
            This step requires time complexity as O(N^2).
        2. After calculating cosine similarity, for each of user, sort other users
            in descending order of cosine similarity.
        3. Predict user's recommendation based on likes of closest users.
        """
        logging.info("Calculating cosine similarity between every user")
        user_sim_pair = self.calculate_user_sim(self.user_ids, user_items)

        logging.info("Getting similar users ordering by cosine similarity")
        self.top_N_sim_user = self.get_top_N_sim_user(user_sim_pair)

        logging.info("Predicting users' unseen item rating")

    def calculate_user_sim(
        self,
        user_ids: NDArray,
        csr: csr_matrix,
    ) -> Dict[Tuple[Any, Any], float]:
        """
        Calculate cosine similarity between every user based on items liked by both of users.

        Args:
            user_ids (NDArray): List of total user ids.
            csr (csr_matrix): Sparse matrix storing likes of each user.

        Returns (Dict[Tuple[Any, Any], float]):
            Keys are tuple of user ids and its values are corresponding cosine similarity.
        """
        return compute_user_sim(user_ids.numpy(), csr)

    def get_top_N_sim_user(
        self,
        user_sim: Dict[Tuple[Any, Any], float],
    ) -> Dict[int, List[Tuple[int, int]]]:
        """
        Get closest N users for each user.

        Args:
            user_sim (Dict[Tuple[Any, Any], float]): Cosine similarity between every user.

        Returns (Dict[int, List[Tuple[int, int]]]):
            N closest users for each user.
        """
        res = {}
        for pair, sim in user_sim.items():
            x, y = pair
            if res.get(x) is None:
                res[x] = [(y, sim)]
            else:
                res[x].append((y, sim))
            if res.get(y) is None:
                res[y] = [(x, sim)]
            else:
                res[y].append((x, sim))
        final_res = {}
        for u, rank in res.items():
            final_res[u] = sorted(rank, key=lambda x: x[1], reverse=True)[
                : self.num_sim_user_top_N
            ]
        return final_res

    def predict(
        self,
        user_id: Union[NDArray, torch.Tensor],
        item_id: Union[NDArray, torch.Tensor],
        **kwargs,
    ) -> torch.Tensor:
        """
        For batch users, calculates prediction score for all of item ids.
        In inference pipeline, `kwargs["item_idx"]` will be all of item ids.
        Using `forward` method in torch model, batch_sie x num_items score matrix will be created.

        Args:
            user_id (Union[NDArray, torch.Tensor]): Set of user_ids who are recommendation target.
                Typically, batch user_ids will be given.
            item_id (Union[NDArray, torch.Tensor]): Set of item_ids to calculate scores.
                Typically, all item_ids will be given because all scores should be cauclated with one user.

        Returns (torch.Tensor):
            Batch_size x num_items score matrix.
        """
        assert isinstance(user_id, torch.Tensor)
        assert isinstance(item_id, torch.Tensor)
        user_id = user_id.detach().cpu().numpy()
        item_id = item_id.detach().cpu().numpy()
        mean_r = {}
        csr = kwargs.get("user_items")
        user_item_rating = np.zeros((len(user_id), len(item_id)))

        # Cache CSR rows as {item: rating} dicts for O(1) lookup
        ratings_cache = {}

        def get_ratings(u):
            if u not in ratings_cache:
                start, end = csr.indptr[u], csr.indptr[u + 1]
                ratings_cache[u] = {
                    int(csr.indices[j]): csr.data[j] for j in range(start, end)
                }
            return ratings_cache[u]

        def get_mean_r(u):
            if u not in mean_r:
                ratings = get_ratings(u)
                mean_r[u] = 0 if not ratings else sum(ratings.values()) / len(ratings)
            return mean_r[u]

        for u in user_id:
            get_mean_r(u)

        for idx, u in tqdm(enumerate(user_id), total=len(user_id), desc="Predict"):
            ratings_u = get_ratings(u)
            r_u = mean_r[u]

            # filter items not rated by u
            reco_item_ids = [i for i in item_id if i not in ratings_u]

            for i in reco_item_ids:
                summation = 0
                k = 0
                items_liked_by_neighbor = False
                for u_, sim in self.top_N_sim_user[u]:
                    ratings_u_ = get_ratings(u_)
                    r_u__i = ratings_u_.get(i, 0)
                    if r_u__i == 0:
                        continue
                    items_liked_by_neighbor = True
                    k += abs(sim)
                    summation += (r_u__i - get_mean_r(u_)) * sim
                if items_liked_by_neighbor:
                    user_item_rating[idx][i] = r_u + summation / k
        return torch.tensor(user_item_rating)
