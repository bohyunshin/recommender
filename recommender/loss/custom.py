import torch
import torch.nn.functional as F


def bpr_loss(pred, params, regularization, user_idx, item_idx, num_users, num_items):
    logprob = F.logsigmoid(pred).sum()
    penalty = calculate_penalty(params, regularization, user_idx, item_idx, num_users, num_items)
    return -logprob + penalty


def svd_loss(pred, true, params, regularization, user_idx, item_idx, num_users, num_items):
    true = true.squeeze()
    mse = F.mse_loss(pred, true, reduction="mean")
    penalty = calculate_penalty(params, regularization, user_idx, item_idx, num_users, num_items)
    return mse + penalty


def calculate_penalty(params, regularization, user_idx, item_idx, num_users, num_items):
    penalty = torch.tensor(0., requires_grad=True)
    for param in params:
        if param.shape[0] == num_users:
            param = param[user_idx]
        elif param.shape[0] == num_items:
            param = param[item_idx]
        else:
            continue
        penalty = penalty + param.data.norm(dim=1).pow(2).sum() * regularization
    return penalty