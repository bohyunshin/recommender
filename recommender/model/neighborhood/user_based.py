import numpy as np
import logging
logger = logging.getLogger("recommender")

from model.fit_model_base import FitModelBase
from tools.csr import slice_csr_matrix


class Model(FitModelBase):
    def __init__(self, num_users, num_items, num_sim_user_top_N, **kwargs):
        super().__init__()
        self.user_ids = np.arange(num_users)
        self.item_ids = np.arange(num_items)
        self.num_sim_user_top_N = num_sim_user_top_N
        self.num_users = num_users
        self.num_items = num_items

    def fit(self, user_items, val_user_items):
        logger.info("Calculating cosine similarity between every user")
        user_sim_pair = self.calculate_user_sim(self.user_ids, user_items)

        logger.info("Getting similar users ordering by cosine similarity")
        self.top_N_sim_user = self.get_top_N_sim_user(user_sim_pair)

        logger.info("Predicting users' unseen item rating")
        # return self.predict(self.user_factors, self.item_factors, user_items=user_items)

    def calculate_user_sim(self, user_ids, csr):
        res = {}
        for i in range(len(user_ids)):
            x = user_ids[i]
            for y in user_ids[i + 1:]:
                items_liked_by_x = csr.indices[csr.indptr[x]:csr.indptr[x+1]]
                items_liked_by_y = csr.indices[csr.indptr[y]:csr.indptr[y+1]]
                items_liked_by_x_y = list(set(items_liked_by_x) & set(items_liked_by_y))

                if len(items_liked_by_x_y) == 0:
                    res[(x, y)] = 0
                else:
                    res[(x, y)] = self._calculate_user_sim(x, y, items_liked_by_x_y, csr)
        return res

    def _calculate_user_sim(self, x, y, items_liked_by_x_y, csr):
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

    def get_top_N_sim_user(self, user_sim):
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

    def predict(self, user_factors, item_factors, userid, **kwargs):
        """
        In user-based CF model, there are not any embeddings w.r.t users / items.
        However, we pass user_factors, item_factors as None value to follow RecommenderBase abstract class.
        By doing this, we can efficiently use asis training pipeline, i.e., train_csr.py
        """
        res = {}
        mean_r = {}
        csr = kwargs["user_items"]
        user_item_rating = np.zeros((self.num_users, self.num_items))
        for u in self.user_ids:
            rating_by_u = csr.data[csr.indptr[u]:csr.indptr[u+1]]
            mean_r[u] = sum(rating_by_u) / len(rating_by_u)

        for idx,u in enumerate(self.user_ids):
            if idx % 5000 == 0:
                logger.info(f"Predicting {idx}th user unseen item rating out of total {len(self.user_ids)} users.")

            if res.get(u) is None:
                res[u] = []
            r_u = mean_r[u]

            # filter items not rated by u
            reco_item_ids = []
            for i in self.item_ids:
                if slice_csr_matrix(csr,u,i) == 0:
                    reco_item_ids.append(i)

            for i in reco_item_ids:
                summation = 0
                k = 0
                items_liked_by_neighbor = False
                for u_, sim in self.top_N_sim_user[u]:
                    if slice_csr_matrix(csr,u_,i) == 0:
                        continue
                    items_liked_by_neighbor = True
                    k += abs(sim)
                    mean_r_u_ = mean_r[u_]
                    r_u__i = slice_csr_matrix(csr,u_,i)
                    summation += (r_u__i - mean_r_u_) * sim
                if items_liked_by_neighbor == True:
                    user_item_rating[u][i] = r_u + summation / k
                    res[u].append((i, r_u + summation / k))
        return user_item_rating[userid]