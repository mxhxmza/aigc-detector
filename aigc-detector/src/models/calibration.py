"""Post-hoc probability calibration.

The brief asks for a "confidence score ... indicating the likelihood that it
is AIGC-generated". A raw sigmoid output is not a likelihood -- networks
trained with BCE are systematically overconfident, and the intended
deployment is tiered triage (auto-action / human review / ignore), where
choosing the bands requires the number to mean what it says.

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
    """Optimise a single temperature to minimise BCE NLL on held-out logits."""
    logits = logits.detach().float().ravel()
    labels = labels.detach().float().ravel()

    log_t = torch.zeros(1, requires_grad=True)  # temperature = exp(log_t) > 0
    opt = torch.optim.LBFGS([log_t], lr=lr, max_iter=max_iter)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad()
        loss = loss_fn(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    return torch.sigmoid(logits / max(temperature, 1e-3))
