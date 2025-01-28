from typing import Dict, Tuple, List, Any, Union
import logging

import torch
import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from recommender.model.fit_model_base import FitModelBase
from recommender.libs.utils.csr import slice_csr_matrix


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
            **kwargs
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
        res = {}
        for i in range(len(user_ids)):
            x = user_ids[i].item()
            for y in user_ids[i + 1:]:
                y = y.item()
                items_liked_by_x = csr.indices[csr.indptr[x]:csr.indptr[x+1]]
                items_liked_by_y = csr.indices[csr.indptr[y]:csr.indptr[y+1]]
                items_liked_by_x_y = list(set(items_liked_by_x) & set(items_liked_by_y))

                if len(items_liked_by_x_y) == 0:
                    res[(x, y)] = 0
                else:
                    res[(x, y)] = self._calculate_user_sim(x, y, items_liked_by_x_y, csr)
        return res

    def _calculate_user_sim(
            self,
            x: int,
            y: int,
            items_liked_by_x_y: List[int],
            csr: csr_matrix,
        ) -> float:
        """
        Inner function calculating cosine similarity.

        Args:
            x (int): User id.
            y (int): User id.
            items_liked_by_x_y (List[int]): List of items liked by both users.
            csr (csr_matrix): Sparse matrix storing likes of each user.

        Returns (float):
            Calculated cosine similarity between user x and y.
        """
        r_x = 0
        r_y = 0
        r_xy = 0

        for r in csr.data[csr.indptr[x]:csr.indptr[x+1]]:
            r_x += r ** 2
        r_x = r_x ** (0.5)

        for r in csr.data[csr.indptr[y]:csr.indptr[y+1]]:
            r_y += r ** 2
        r_y = r_y ** (0.5)

        for i in items_liked_by_x_y:
            r_xi = slice_csr_matrix(csr,x,i)
            r_yi = slice_csr_matrix(csr,y,i)
            r_xy += r_xi * r_yi

        return r_xy / (r_x * r_y)

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
            final_res[u] = sorted(rank, key=lambda x: x[1], reverse=True)[:self.num_sim_user_top_N]
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
        res = {}
        mean_r = {}
        csr = kwargs.get("user_items")
        user_item_rating = np.zeros((len(user_id), len(item_id)))
        for u in user_id:
            rating_by_u = csr.data[csr.indptr[u]:csr.indptr[u+1]]
            mean_r[u] = 0 if len(rating_by_u) == 0 else sum(rating_by_u) / len(rating_by_u)

        for idx, u in enumerate(user_id):
            if res.get(u) is None:
                res[u] = []
            r_u = mean_r[u]

            # filter items not rated by u
            reco_item_ids = []
            for i in item_id:
                if slice_csr_matrix(csr, u, i) == 0:
                    reco_item_ids.append(i)

            for i in reco_item_ids:
                summation = 0
                k = 0
                items_liked_by_neighbor = False
                for u_, sim in self.top_N_sim_user[u]:
                    if slice_csr_matrix(csr, u_, i) == 0:
                        continue
                    items_liked_by_neighbor = True
                    k += abs(sim)
                    mean_r_u_ = mean_r[u_]
                    r_u__i = slice_csr_matrix(csr, u_, i)
                    summation += (r_u__i - mean_r_u_) * sim
                if items_liked_by_neighbor == True:
                    user_item_rating[u][i] = r_u + summation / k
                    res[u].append((i, r_u + summation / k))
        return torch.tensor(user_item_rating)