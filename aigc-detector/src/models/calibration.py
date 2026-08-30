"""Post-hoc probability calibration.

The brief asks for a "confidence score ... indicating the likelihood that it
is AIGC-generated". A raw sigmoid output is not a likelihood -- networks
trained with BCE are systematically overconfident, and that matters here for
a concrete reason: the deployment story in the PRD is tiered triage
(auto-action / human review / ignore), and choosing those bands requires the
number to mean what it says.

Temperature scaling is a single learned scalar that divides the logits. It
cannot change the ranking, so AUC is exactly unchanged -- it only fixes the
spread. That property is why it is the right choice under time pressure:
it can improve ECE without any risk of hurting the headline metric.

Fit on a held-out split, never on training data.
"""

from __future__ import annotations

import torch


def fit_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    max_iter: int = 200,
    lr: float = 0.01,
) -> float:
    """Optimise a single temperature to minimise NLL on held-out logits.

    Handles both heads:
      * 1-D logits (or shape (N, 1)) with float labels -> binary, BCE NLL.
      * (N, C) logits with integer labels              -> multiclass, CE NLL.
    Guo et al.'s temperature scaling is defined for the multiclass case; the
    binary case is the same idea with a sigmoid.
    """
    logits = logits.detach().float()
    labels = labels.detach()
    multiclass = logits.dim() == 2 and logits.shape[1] > 1

    log_t = torch.zeros(1, requires_grad=True)  # temperature = exp(log_t) > 0
    opt = torch.optim.LBFGS([log_t], lr=lr, max_iter=max_iter)

    if multiclass:
        y = labels.long().ravel()
        loss_fn = torch.nn.CrossEntropyLoss()

        def closure():
            opt.zero_grad()
            loss = loss_fn(logits / log_t.exp(), y)
            loss.backward()
            return loss
    else:
        flat = logits.ravel()
        y = labels.float().ravel()
        loss_fn = torch.nn.BCEWithLogitsLoss()

        def closure():
            opt.zero_grad()
            loss = loss_fn(flat / log_t.exp(), y)
            loss.backward()
            return loss

    opt.step(closure)
    return float(log_t.exp().item())


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    return torch.sigmoid(logits / max(temperature, 1e-3))
