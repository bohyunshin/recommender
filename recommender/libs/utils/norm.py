from typing import List

import torch
import torch.nn as nn


def parameter_l_k_norm(params: List[nn.parameter.Parameter], k: int) -> torch.Tensor:
    penalty = torch.tensor(0., requires_grad=True)
    for param in params:
        penalty = penalty + param.data.pow(k).sum()
    return penalty.pow(1/k)