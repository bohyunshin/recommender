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
        self.batch_user_id = batch_user_id
        self.batch_item_id = batch_item_id
        self.user_item_summ = user_item_summ
        self.num_ng = num_ng
        self.is_triplet = is_triplet
        self.num_item = num_item
        self.strategy = strategy
        self.ng_samples = []

    def in_batch_ng(self):
        for user_id, pos_item_id in zip(self.batch_user_id, self.batch_item_id):
            neg_item_id_candidate = torch.tensor(
                [
                    t for t in self.batch_item_id
                    if t != pos_item_id
                ]
            )
            self._sample(
                user_id=user_id,
                pos_item_id=pos_item_id,
                neg_item_id_candidate=neg_item_id_candidate,
            )

    def random_from_total_pool_ng(self):
        for user_id, pos_item_id in zip(self.batch_user_id, self.batch_item_id):
            neg_item_id_candidate = torch.arange(self.num_item)
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
        ):
        num_ng = 0
        sampled = []
        while num_ng != self.num_ng:
            unif = torch.ones(neg_item_id_candidate.shape[0])
            idx = torch.multinomial(unif, num_samples=1)
            neg_item_id = neg_item_id_candidate[idx]
            if self.user_item_summ[user_id.item()].get(neg_item_id.item()) is None and neg_item_id not in sampled:
                num_ng += 1
                sampled.append(neg_item_id)
                self.ng_samples.append((user_id, pos_item_id, neg_item_id))

    def ng(self):
        for s in self.strategy:
            if s == "in_batch":
                self.in_batch_ng()
            elif s == "random_from_total_pool":
                self.random_from_total_pool_ng()
            else:
                raise

    def format_dataset(self):
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