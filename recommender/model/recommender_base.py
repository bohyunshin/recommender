import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from recommender.libs.constant.inference.evaluation import Metric
from recommender.libs.constant.inference.recommend import (
    RECOMMEND_BATCH_SIZE,
    TOP_K_VALUES,
)
from recommender.libs.constant.loss.name import LossName
from recommender.libs.utils.evaluation import ranking_metrics_at_k
from recommender.libs.utils.user_item_count import convert_tensor_to_user_item_summary
from recommender.libs.utils.utils import safe_divide
from recommender.loss.custom import als_loss, bpr_loss, svd_loss


class RecommenderBase(ABC):
    def __init__(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        num_users: int,
        num_items: int,
        num_factors: int,
        loss_name: str,
        **kwargs,
    ):
        """
        Abstract base class for all recommender models including torch based, fit based models.

        Args:
            user_ids (torch.Tensor): List of user_id.
            item_ids (torch.Tensor): List of item_id.
            num_users (int): Number of users. Should match with dimension of user_ids.
            num_items (int): Number of items. Should match with dimension of item_ids.
            num_factors (int): Embedding dimension for user, item embeddings.
        """
        self.user_ids = user_ids
        self.item_ids = item_ids

        # those will be overridden in children class
        self.embed_user = nn.Embedding(num_users, num_factors)
        self.embed_item = nn.Embedding(num_items, num_factors)

        self.num_users = num_users
        self.num_items = num_items

        self.loss_name = loss_name

        # store metric value at each epoch
        self.metric_at_k_total_epochs = {
            k: {
                Metric.MAP.value: [],
                Metric.NDCG.value: [],
                Metric.RECALL.value: [],
                Metric.COUNT.value: 0,
            }
            for k in TOP_K_VALUES
        }

        self.tr_loss = []
        self.val_loss = []

        self.set_loss_func()

    def set_loss_func(self):
        # TODO: fix `if-else` statement to better program.
        if self.loss_name == LossName.MSE.value:
            self.loss = svd_loss
        elif self.loss_name == LossName.BPR.value:
            self.loss = bpr_loss
        elif self.loss_name == LossName.BCE.value:
            self.loss = nn.BCEWithLogitsLoss()
        elif self.loss_name == LossName.ALS.value:
            self.loss = als_loss
        elif self.loss_name == LossName.NOT_DEFINED.value:
            pass
        else:
            raise

    def calculate_loss(
        self,
        y_pred: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        params: Optional[List[nn.parameter.Parameter]] = None,
        regularization: Optional[int] = None,
        user_idx: Optional[torch.Tensor] = None,
        item_idx: Optional[torch.Tensor] = None,
        num_users: Optional[int] = None,
        num_items: Optional[int] = None,
        user_items: Optional[csr_matrix] = None,
        user_factors: Optional[NDArray] = None,
        item_factors: Optional[NDArray] = None,
    ) -> torch.Tensor:
        """
        Calculates loss for given model using defined loss function.
        Because arguments for each loss function are different, Optional type is used.

        Args:
            y_pred (Optional[torch.Tensor]): Prediction value. Could be logit or probability.
            y (Optional[torch.Tensor]): True y value. If implicit data, dummy value is used here.
            params (Optional[torch.nn.parameter]): Model parameters from torch model.
            regularization (Optional[int]): Regularization parameter balancing between main loss and penalty.
            user_idx (Optional[torch.Tensor]): User index in current batch.
            item_idx (Optional[torch.Tensor]): Item index in current batch.
            num_users (Optional[int]): Number of total users.
            num_items (Optional[int]): Number of total items.

        Returns (torch.Tensor):
            Calculated loss.
        """
        # TODO: fix `if-else` statement to better program.
        if self.loss_name == LossName.MSE.value:
            return self.loss(
                pred=y_pred.squeeze(),
                true=y.squeeze(),
                params=params,
                regularization=regularization,
                user_idx=user_idx,
                item_idx=item_idx,
                num_users=num_users,
                num_items=num_items,
            )
        elif self.loss_name == LossName.BCE.value:
            return self.loss(
                input=y_pred.squeeze(),
                target=y.squeeze(),
            )
        elif self.loss_name == LossName.BPR.value:
            return self.loss(
                pred=y_pred,
                params=params,
                regularization=regularization,
            )
        elif self.loss_name == LossName.ALS.value:
            return self.loss(
                user_items=user_items,
                user_factors=user_factors,
                item_factors=item_factors,
                regularization=regularization,
            )
        elif self.loss_name == LossName.NOT_DEFINED.value:
            return torch.tensor(0.0)
        else:
            raise

    def recommend_all(
        self,
        X_train: torch.Tensor,
        X_val: torch.Tensor,
        top_k_values: List[int],
        filter_already_liked: bool = True,
        **kwargs,
    ):
        """
        Generate recommendations for all users.
        Suppose number of users is U and number of items is D.
        The dimension of associated matrix between users and diners is U x D.
        However, to avoid out of memory error, batch recommendation is run with RECOMMEND_BATCH_SIZE.
        For every batch users, we calculate mAP, NDCG metric.

        Args:
             X_train (Tensor): Users x items interaction tensors in train dataset.
             X_val (Tensor): Users x items interaction tensors in validation dataset.
             top_k_values (List[int]): a list of k values.
             filter_already_liked (bool): whether filtering pre-liked diner in train dataset or not.
        """
        # prepare for metric calculation
        # refresh at every epoch
        self.metric_at_k = {
            k: {
                Metric.MAP.value: 0,
                Metric.NDCG.value: 0,
                Metric.RECALL.value: 0,
                Metric.COUNT.value: 0,
            }
            for k in top_k_values
        }
        max_k = max(top_k_values)
        start = 0

        # store true diner id visited by user in validation dataset
        self.train_liked = convert_tensor_to_user_item_summary(X_train, torch.Tensor)
        self.val_liked = convert_tensor_to_user_item_summary(X_val, list)

        while start < self.num_users:
            batch_users = self.user_ids[start : start + RECOMMEND_BATCH_SIZE]
            num_batch_users = len(batch_users)
            scores = self.predict(
                user_id=batch_users,
                item_id=self.item_ids,
                user_items=kwargs.get("user_items"),
                num_batch_users=num_batch_users,
            )

            # TODO: change for loop to more efficient program
            # filter diner id already liked by user in train dataset
            if filter_already_liked:
                for i, user_id in enumerate(batch_users):
                    already_liked_ids = self.train_liked[user_id.item()]
                    scores[i][already_liked_ids] = -float("inf")

            max_k = min(scores.shape[1], max_k)  # to prevent index error in pytest
            top_k = torch.topk(scores, k=max_k)
            top_k_id = top_k.indices

            self.calculate_metric(
                user_ids=batch_users, top_k_id=top_k_id, top_k_values=top_k_values
            )

            start += RECOMMEND_BATCH_SIZE

        for k in top_k_values:
            # save map
            self.metric_at_k[k][Metric.MAP.value] = safe_divide(
                numerator=self.metric_at_k[k][Metric.MAP.value],
                denominator=self.metric_at_k[k][Metric.COUNT.value],
            )
            self.metric_at_k_total_epochs[k][Metric.MAP.value].append(
                self.metric_at_k[k][Metric.MAP.value]
            )

            # save ndcg
            self.metric_at_k[k][Metric.NDCG.value] = safe_divide(
                numerator=self.metric_at_k[k][Metric.NDCG.value],
                denominator=self.metric_at_k[k][Metric.COUNT.value],
            )
            self.metric_at_k_total_epochs[k][Metric.NDCG.value].append(
                self.metric_at_k[k][Metric.NDCG.value]
            )

            # save recall
            self.metric_at_k[k][Metric.RECALL.value] = safe_divide(
                numerator=self.metric_at_k[k][Metric.RECALL.value],
                denominator=self.metric_at_k[k][Metric.COUNT.value],
            )
            self.metric_at_k_total_epochs[k][Metric.RECALL.value].append(
                self.metric_at_k[k][Metric.RECALL.value]
            )

    def calculate_metric(
        self,
        user_ids: torch.Tensor,
        top_k_id: torch.Tensor,
        top_k_values: List[int],
    ) -> None:
        """
        After calculating scores in `recommend_all` function, calculate metric without any candidates.
        Metrics calculated in this function are NDCG, mAP and recall.
        Note that this function does not consider locality, which means recommendations
        could be given regardless of user's location and diner's location

        Args:
             user_ids (Tensor): batch of user ids.
             top_k_id (Tensor): diner_id whose score is under max_k ranked score.
             top_k_values (List[int]): a list of k values.
        """

        # TODO: change for loop to more efficient program
        # calculate metric
        for i, user_id in enumerate(user_ids):
            user_id = user_id.item()
            val_liked_item_id = np.array(self.val_liked[user_id])

            for k in top_k_values:
                pred_liked_item_id = top_k_id[i][:k].detach().cpu().numpy()
                metric = ranking_metrics_at_k(val_liked_item_id, pred_liked_item_id)
                self.metric_at_k[k][Metric.MAP.value] += metric[Metric.AP.value]
                self.metric_at_k[k][Metric.NDCG.value] += metric[Metric.NDCG.value]
                self.metric_at_k[k][Metric.RECALL.value] += metric[Metric.RECALL.value]
                self.metric_at_k[k][Metric.COUNT.value] += 1

    def collect_metrics(self):
        maps = []
        ndcgs = []
        recalls = []

        for k in TOP_K_VALUES:
            # no candidate metric
            map = round(self.metric_at_k[k][Metric.MAP.value], 5)
            ndcg = round(self.metric_at_k[k][Metric.NDCG.value], 5)
            recall = round(self.metric_at_k[k][Metric.RECALL.value], 5)
            count = self.metric_at_k[k][Metric.COUNT.value]

            logging.info(
                f"maP@{k}: {map} with {count} users out of all {self.num_users} users"
            )
            logging.info(
                f"ndcg@{k}: {ndcg} with {count} users out of all {self.num_users} users"
            )
            logging.info(
                f"recall@{k}: {recall} with {count} users out of all {self.num_users} users"
            )

            maps.append(str(map))
            ndcgs.append(str(ndcg))
            recalls.append(str(recall))

        logging.info(f"top k results for direct prediction @1, @3, @7, @10 in order")
        logging.info(f"map result: {'|'.join(maps)}")
        logging.info(f"ndcg result: {'|'.join(ndcgs)}")
        logging.info(f"recall: {'|'.join(recalls)}")

    @abstractmethod
    def predict(
        self,
        user_id: Union[NDArray, torch.Tensor],
        item_id: Union[NDArray, torch.Tensor],
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts users' ratings (or preference, scores) based on factorized user_factors and item_factors.
        For matrix factorization models, this could be dot product between user_factors and item_factors.
        For deep learning models, this could be inference step fed with user/item information to neural network

        Args:
            user_id (Union[NDArray, torch.Tensor]): Set of user_ids who are recommendation target.
                Typically, batch user_ids will be given.
            item_id (Union[NDArray, torch.Tensor]): Set of item_ids to calculate scores.
                Typically, all item_ids will be given because all scores should be cauclated with one user.

        Returns (Union[NDArray, torch.Tensor]):
            User x item prediction score matrix.
        """
        raise NotImplementedError
