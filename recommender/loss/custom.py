import torch
import torch.nn.functional as F


def bpr_loss(pred, params, regularization):
    logprob = F.logsigmoid(pred).sum()
    penalty = torch.tensor(0., requires_grad=True)
    for param in params:
        penalty = penalty + param.data.norm(dim=1).pow(2).sum() * regularization
    return -logprob + penalty