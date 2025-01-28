import torch
import torch.nn.functional as F

from recommender.libs.utils.norm import parameter_l_k_norm


def bpr_loss(
        pred: torch.Tensor,
        params: torch.nn.parameter,
        regularization: float,
    ) -> torch.Tensor:
    """
    Calculates the Bayesian Personalized Ranking Loss with penalty term together.

    Args:
        pred (torch.Tensor): Prediction values from model.
        params (torch.nn.parameter): Parameters of the model which will be used to compuate pently term.
        regularization (float): Regularization term.

    Returns (torch.Tensor):
        BPR loss with penalty term.
    """
    logprob = F.logsigmoid(pred).sum()
    penalty = parameter_l_k_norm(params=params, k=2)
    return -logprob + penalty * regularization


def svd_loss(
        pred: torch.Tensor,
        true: torch.Tensor,
        params: torch.nn.parameter,
        regularization: float,
        user_idx: torch.Tensor,
        item_idx: torch.Tensor,
        num_users: int,
        num_items: int,
    ) -> torch.Tensor:
    """
    Calculates the SVD loss with penalty term together.
    There are lots of SVD models. This loss focuses on SVD with user and item's bias term.

    Args:
        pred (torch.Tensor): Prediction values from svd model.
        true (torch.Tensor): True values from y.
        params (torch.nn.parameter): Parameters of the svd model which  used to compute penalty term.
        regularization (float): Regularization term balancing between main loss and penalty.
        user_idx (torch.Tensor): User id used when calculating bias term.
        item_idx (torch.Tensor): item id used when calculating bias term.
        num_users (int): Number of users used to match dimension of embedding matrix.
        num_items (int): Number of items used to match dimension of embedding matrix.

    Returns (torch.Tensor):
        SVD loss with penalty term.
    """
    true = true.squeeze()
    mse = F.mse_loss(pred, true, reduction="mean")
    penalty = calculate_penalty(
        params=params,
        regularization=regularization,
        user_idx=user_idx,
        item_idx=item_idx,
        num_users=num_users,
        num_items=num_items
    )
    return mse + penalty


def calculate_penalty(
        params: torch.nn.parameter,
        regularization: float,
        user_idx: torch.Tensor,
        item_idx: torch.Tensor,
        num_users: int,
        num_items: int,
    ) -> torch.Tensor:
    """
    Calculates penalty in l2 norm.

    Args:
        params (torch.nn.parameter): Parameters of the model to calculate loss.
        regularization (float): Regularization term.
        user_idx (torch.Tensor): User id used when calculating bias term.
        item_idx (torch.Tensor): item id used when calculating bias term.
        num_users (int): Number of users used to match dimension of embedding matrix.
        num_items (int): Number of items used to match dimension of embedding matrix.

    Returns (torch.Tensor):
        Penalty in l2 norm.
    """
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