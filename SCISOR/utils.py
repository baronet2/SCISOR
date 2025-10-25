import torch
import torch.nn
import torch.nn.functional as F


def kls_probs(dist1, dist2):
    out = F.kl_div(
        torch.log_softmax(dist2, dim=-1), dist1, log_target=False, reduction="none"
    ).sum(-1)
    return out


def kls_probs_both(dist1, dist2):
    probs_clamped = dist2.clamp(min=1e-6, max=1 - 1e-6)
    out = F.kl_div(probs_clamped.log(), dist1, log_target=False, reduction="none").sum(
        -1
    )
    return out


def log1p(x):
    result = torch.log1p(x)
    mask = x < -0.7
    if mask.any():
        x_neg = x[mask]
        neg_x = -x_neg
        inv_xp1 = 1.0 / (neg_x - 1.0)
        result[mask] = torch.log1p(inv_xp1) + torch.log(neg_x)
    return result
