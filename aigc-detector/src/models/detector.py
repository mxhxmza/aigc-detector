"""Degradation-aware gated detector.

    semantic embedding (frozen CLIP)  ->  semantic head  --.
                                                            >-- gated fusion -> logit
    forensic features (fixed, hand-designed) -> forensic head --'
                                    |
                                    '--> degradation estimator -> gate weights

The idea in one sentence
------------------------
The forensic branch is precise on clean images and collapses under
compression; the semantic branch is coarser but survives. So rather than
averaging them with fixed weights, estimate how damaged the image is and let
that estimate decide how much to trust each branch.

Why this is not just an ensemble
--------------------------------
A fixed-weight ensemble learns one compromise for all conditions. Here the
fusion weights are a function of the input's estimated degradation, so the
model can behave differently on a pristine PNG and a q30 re-encode without
being told which it is at test time. The degradation estimator is trained
with free supervision -- we applied the degradations, so we know the answer.

Everything here is tiny (~1-3M parameters) and trains on cached features.
The frozen CLIP backbone is NOT part of this module; it lives in the
extraction step and is never backpropagated through. Total parameter count
including the frozen backbone stays far under the 2B cap (C-1) -- run
`count_parameters` to print it for the README.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..transforms.degradations import DEGRADATION_DIM


def mlp(sizes: list[int], dropout: float = 0.1, out_activation: bool = True) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        is_last = i == len(sizes) - 2
        if not is_last or out_activation:
            layers.append(nn.LayerNorm(sizes[i + 1]))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class DegradationEstimator(nn.Module):
    """Predicts the degradation descriptor from the forensic features.

    Deliberately reads only the forensic features, not the semantic ones:
    compression and blur are low-level phenomena, and forcing the estimate
    to come from the low-level branch keeps the gate from quietly becoming
    a second semantic classifier.
    """

    def __init__(self, in_dim: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = mlp([in_dim, hidden, hidden], dropout=dropout)
        self.head = nn.Linear(hidden, DEGRADATION_DIM)

    def forward(self, forensic: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(forensic)
        # Severities live in [0,1]; sigmoid rather than softmax because a
        # chained degradation is damaged in several channels at once.
        return torch.sigmoid(self.head(h)), h


class GatedFusion(nn.Module):
    """Produces per-sample mixing weights over the two branches."""

    def __init__(self, deg_hidden: int, n_branches: int = 2, temperature: float = 1.0):
        super().__init__()
        self.gate = nn.Linear(deg_hidden, n_branches)
        self.temperature = temperature
        # Start near an even split so early training is a plain ensemble and
        # the gate only specialises once the branches are worth weighting.
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, deg_hidden: torch.Tensor, branches: list[torch.Tensor]):
        w = F.softmax(self.gate(deg_hidden) / self.temperature, dim=-1)
        stacked = torch.stack(branches, dim=1)              # (B, n_branches, D)
        fused = (stacked * w.unsqueeze(-1)).sum(dim=1)       # (B, D)
        return fused, w


class Detector(nn.Module):
    """Binary real-vs-AI detector. One sigmoid logit; `prob` is p(AI-generated)."""

    def __init__(
        self,
        semantic_dim: int,
        forensic_dim: int,
        hidden: int = 256,
        dropout: float = 0.1,
        use_forensic: bool = True,
        use_gate: bool = True,
    ):
        super().__init__()
        self.use_forensic = use_forensic
        self.use_gate = use_gate and use_forensic

        self.semantic_head = mlp([semantic_dim, 512, hidden], dropout=dropout)

        if use_forensic:
            # Forensic features are raw physical quantities on wildly
            # different scales (log-power vs kurtosis), so normalise first.
            self.forensic_norm = nn.BatchNorm1d(forensic_dim)
            self.forensic_head = mlp([forensic_dim, 256, hidden], dropout=dropout)
            self.degradation = DegradationEstimator(forensic_dim, dropout=dropout)
            if self.use_gate:
                self.fusion = GatedFusion(128, n_branches=2)

        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

        # Temperature for post-hoc calibration. Not trained with the model --
        # fitted afterwards on held-out data by calibration.fit_temperature.
        self.register_buffer("temperature", torch.ones(1))

    def forward(self, semantic: torch.Tensor, forensic: torch.Tensor | None = None) -> dict:
        sem = self.semantic_head(semantic)

        if not self.use_forensic:
            fused = sem
            deg_pred = None
            gate_w = None
        else:
            fnorm = self.forensic_norm(forensic)
            for_feat = self.forensic_head(fnorm)
            deg_pred, deg_hidden = self.degradation(fnorm)
            if self.use_gate:
                fused, gate_w = self.fusion(deg_hidden, [sem, for_feat])
            else:
                fused = 0.5 * (sem + for_feat)
                gate_w = None

        logit = self.classifier(fused).squeeze(-1)
        return {
            "logit": logit,
            "prob": torch.sigmoid(logit),                     # p(AI-generated)
            "degradation": deg_pred,
            "gate": gate_w,
        }

    @torch.no_grad()
    def predict_proba(self, semantic: torch.Tensor, forensic: torch.Tensor | None = None):
        """Calibrated p(AI-generated), a scalar per image. Use this at inference."""
        out = self.forward(semantic, forensic)
        return torch.sigmoid(out["logit"] / self.temperature.clamp(min=1e-3))


def count_parameters(model: nn.Module, frozen_backbone_params: int = 0) -> dict:
    """Parameter accounting for C-1 compliance. Print this in the README."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = trainable + frozen_backbone_params
    return {
        "trainable": trainable,
        "frozen_backbone": frozen_backbone_params,
        "total": total,
        "limit": 2_000_000_000,
        "compliant": total < 2_000_000_000,
        "headroom_x": round(2_000_000_000 / max(total, 1), 1),
    }


# --------------------------------------------------------------------------
# Loss
# --------------------------------------------------------------------------

def detector_loss(
    out_clean: dict,
    out_aug: dict,
    labels: torch.Tensor,
    deg_target_clean: torch.Tensor,
    deg_target_aug: torch.Tensor,
    lambda_consistency: float = 1.0,
    lambda_degradation: float = 0.1,
) -> tuple[torch.Tensor, dict]:
    """Classification on both views + consistency between them + degradation.

    The consistency term is what separates this from plain augmentation.
    Augmentation says "also classify the damaged version correctly".
    Consistency says "give the damaged version the SAME answer as the clean
    one", which is a strictly stronger requirement and is the property the
    robustness table actually measures.

    Symmetric KL is used rather than MSE on logits because it penalises
    disagreement in probability space, where our metric lives, and because
    it is scale-free -- a pair of confident-but-opposite predictions is
    punished much harder than a pair of uncertain ones.
    """
    bce = nn.functional.binary_cross_entropy_with_logits
    l_cls = 0.5 * (bce(out_clean["logit"], labels) + bce(out_aug["logit"], labels))

    p_clean = torch.sigmoid(out_clean["logit"]).clamp(1e-6, 1 - 1e-6)
    p_aug = torch.sigmoid(out_aug["logit"]).clamp(1e-6, 1 - 1e-6)

    def _kl(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        return (p * (p / q).log() + (1 - p) * ((1 - p) / (1 - q)).log()).mean()

    # Detach the clean view: the clean prediction is the target the augmented
    # view moves toward, not the other way round. Without this the model can
    # trivially satisfy the loss by making both views equally uncertain.
    l_cons = _kl(p_clean.detach(), p_aug)

    l_deg = torch.zeros((), device=labels.device)
    if out_clean["degradation"] is not None:
        l_deg = 0.5 * (
            F.mse_loss(out_clean["degradation"], deg_target_clean)
            + F.mse_loss(out_aug["degradation"], deg_target_aug)
        )

    total = l_cls + lambda_consistency * l_cons + lambda_degradation * l_deg
    return total, {
        "loss": float(total.detach()),
        "cls": float(l_cls.detach()),
        "consistency": float(l_cons.detach()),
        "degradation": float(l_deg.detach()),
    }
