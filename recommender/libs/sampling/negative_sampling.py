from typing import Dict, List

import torch


class NegativeSampling(object):
    def __init__(
            self,
            batch_user_id: torch.Tensor,
            batch_item_id: torch.Tensor,
            user_item_summ: Dict[int, Dict[int, bool]],
            num_ng: int,
            is_triplet: bool,
            num_item: int,
            strategy: List[str],
        ):
        """
        Run negative sampling in every batch.
        Currently, in-batch sampling, random sampling from total pool are supported.

            - in-batch sampling: For each pair of (user_id, pos_item_id) in current batch, sample neg_item_id from
                rest of other pos_item_ids. This negative sampling belongs to easy negative samples.
            - random sampling form total pool: Sample neg_item_id from total item pool not from current batch.
                This negative sampling belongs to harder negative samples.

        Args:
            batch_user_id (torch.Tensor): user_id in current batch.
            batch_item_id (torch.Tensor): item_id in current batch, which is pos_item_id that batch_user_id interacted with.
            user_item_summ (Dict[int, Dict[int, bool]]): History that stores which item user interacted with.
                If user_item_summ[user_id][item_id] is True, user_id interacted with item_id.
                If user_item_summ[user_id][item_id] is None, user_id did not interact with item_id.
            num_ng (int): Number of negative samples.
            is_triplet (bool): Whether triplet loss or not.
            num_item (int): Number of items in total pool.
            strategy (List[str]): Controls which negative sampling strategy to use.
        """
        self.batch_user_id = batch_user_id
        self.batch_item_id = batch_item_id
        self.user_item_summ = user_item_summ
        self.num_ng = num_ng
        self.is_triplet = is_triplet
        self.num_item = num_item
        self.strategy = strategy
        self.ng_samples = []

    def in_batch_ng(self) -> None:
        """
        In batch negative sampling in current batch.

        Suppose following batch.
        user_id | item_id |
        1       | 100     |
        2       | 101     |
        3       | 102     |
        4       | 103     |
        5       | 104     |

        For user_id = 1, negative samples are sampled from item_id = (101, 102, 103, 104)
        For user_id = 2, negative samples are sampled from item_id = (100, 102, 103, 104)
        When sampling, exclude item_id if user_id already liked using `user_item_summ` variable.
        """
        for user_id, pos_item_id in zip(self.batch_user_id, self.batch_item_id):
            neg_item_id_candidate = torch.tensor(
                [
                    t for t in self.batch_item_id
                    # exclude item_id that user_id already liked and pos_item_id
                    if t != pos_item_id and self.user_item_summ[user_id.item()].get(t.item()) is None
                ]
            )
            self._sample(
                user_id=user_id,
                pos_item_id=pos_item_id,
                neg_item_id_candidate=neg_item_id_candidate,
            )

    def random_from_total_pool_ng(self) -> None:
        """
        Sample neg_item_id from total item_id pool.

        Suppose following batch.
        user_id | item_id |
        1       | 100     |
        2       | 101     |
        3       | 102     |
        4       | 103     |
        5       | 104     |

        This sampling strategy does not consider item_id in current batch.
        Therefore, for user_id = 1, neg_item_id could be sampled from item_id = (101, 102, ..., 999, 1000),
        which is total item_id pool.
        When sampling, exclude item_id if user_id already liked using `user_item_summ` variable.
        """
        for user_id, pos_item_id in zip(self.batch_user_id, self.batch_item_id):
            neg_item_id_candidate = torch.tensor(
                [
                    t for t in range(self.num_item)
                    if self.user_item_summ[user_id.item()].get(t) is None # exclude item_id that user_id already liked
                ]
            )
            self._sample(
                user_id=user_id,
                pos_item_id=pos_item_id,
                neg_item_id_candidate=neg_item_id_candidate,
            )

    def _sample(
            self,
            user_id: torch.Tensor,
            pos_item_id: torch.Tensor,
            neg_item_id_candidate: torch.Tensor
        ) -> None:
        """
        Sample item_id from `neg_item_id_candidate` for current (user_id, pos_item_id) pair.

        Args:
            user_id: Current user_id.
            pos_item_id: Current item_id that user_id interacted with.
            neg_item_id_candidate: Neg_item_id candidate pools, depending on negative sampling strategy.
        """
        if len(neg_item_id_candidate) <= self.num_ng:
            for neg_item_id in neg_item_id_candidate:
                self.ng_samples.append((user_id, pos_item_id, neg_item_id))
            return
        # set uniform prob
        unif = torch.ones(neg_item_id_candidate.shape[0])
        # sample num_ng neg_item_ids
        idx = torch.multinomial(unif, num_samples=self.num_ng)
        # get neg_item_id item index
        neg_item_ids = neg_item_id_candidate[idx]
        for neg_item_id in neg_item_ids:
            self.ng_samples.append((user_id, pos_item_id, neg_item_id))

    def ng(self) -> None:
        """
        Sample neg_item_id with negative sampling strategy.
        """
        for s in self.strategy:
            if s == "in_batch":
                self.in_batch_ng()
            elif s == "random_from_total_pool":
                self.random_from_total_pool_ng()
            else:
                raise

    def format_dataset(self) -> Dict[str, torch.Tensor]:
        """
        After sampling neg_item_id, format dataset depending on loss function.

        When loss function is based on triplet dataset, format dataset as triplet.
        When loss function is based on binary cross entropy, format dataset with (user_id, item_id, label),
        where label corresponds with 0 or 1.

        Returns (Dict[str, torch.Tensor]):
            Key is argument name for model forward method, value is corresponding tensor value.
        """
        if self.is_triplet:
            user_ids = []
            pos_item_ids = []
            neg_item_ids = []
            for user_id, pos_item_id, neg_item_id in self.ng_samples:
                user_ids.append(user_id)
                pos_item_ids.append(pos_item_id)
                neg_item_ids.append(neg_item_id)
            return {
                "user_idx": torch.tensor(user_ids),
                "pos_item_idx": torch.tensor(pos_item_ids),
                "neg_item_idx": torch.tensor(neg_item_ids),
                "y": torch.zeros(size=(len(user_ids),)), # dummy y value
            }
        else:
            user_ids = []
            item_ids = []
            y = []
            for user_id, pos_item_id, neg_item_id in self.ng_samples:
                # positive sample
                user_ids.append(user_id)
                item_ids.append(pos_item_id)
                y.append(1.)
                # negative sample
                user_ids.append(user_id)
                item_ids.append(neg_item_id)
                y.append(0.)
            return {
                "user_idx": torch.tensor(user_ids),
                "item_idx": torch.tensor(item_ids),
                "y": torch.tensor(y),
            }