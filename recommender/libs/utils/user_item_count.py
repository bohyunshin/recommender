from collections import defaultdict
from typing import Dict, List, Union

import torch


def convert_tensor_to_user_item_summary(
    ts: torch.Tensor, structure: Union[dict, list]
) -> Dict[int, Union[List[int], Dict[int, int]]]:
    """
    Convert 2 dimensional tensor to dict or list.
    Original tensor includes interaction between user and item.

    Args:
        ts (Tensor): n x 2 dimension tensors whose columns are matched with (user_id, item_id).
            Should be careful of column ordering.
        structure (Union[dict, list]): Data type of value corresponding key in return object.

    Returns (Dict[int, Union[List[int], Dict[int, int]]]):
        Key is user_id and values are diner_id interacted by item_id.
        Data types of values are dictionary or list.
        In case dictionary, res[user_id][item_id] is 1 if interacted else 0.
        In case list, res[user_id] is a list of item_id interacted by user_id.
    """
    assert ts.shape[1] == 2
    assert structure in [dict, list, torch.Tensor]
    if structure in [list, torch.Tensor]:
        res = defaultdict(list)
    else:
        res = defaultdict(structure)
    for user_id, item_id in ts:
        user_id = user_id.item()
        item_id = item_id.item()
        if structure == dict:
            res[user_id][item_id] = 1
        elif structure in [list, torch.Tensor]:
            res[user_id].append(item_id)
    if structure == torch.Tensor:
        for user_id in res.keys():
            res[user_id] = torch.tensor(res[user_id], dtype=torch.long)
    return res
