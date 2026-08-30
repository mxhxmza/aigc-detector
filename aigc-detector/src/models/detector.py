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
    def __init__(
        self,
        semantic_dim: int,
        forensic_dim: int,
        hidden: int = 256,
        dropout: float = 0.1,
        use_forensic: bool = True,
        use_gate: bool = True,
        n_classes: int = 3,
    ):
        super().__init__()
        self.use_forensic = use_forensic
        self.use_gate = use_gate and use_forensic
        # n_classes == 1 -> binary sigmoid head (legacy CIFAKE/SID checkpoints).
        # n_classes == 3 -> softmax over {0: real, 1: AI-generated, 2: tampered}.
        # The tampered class is trained but folded into "authentic" for the
        # user-facing verdict; see predict_proba / predict_proba_full.
        self.n_classes = n_classes

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
            nn.Linear(hidden // 2, n_classes),
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

        logits = self.classifier(fused)                       # (B, n_classes)
        out = {"logits": logits, "degradation": deg_pred, "gate": gate_w}

        if self.n_classes == 1:
            logit = logits.squeeze(-1)
            out["logit"] = logit
            out["prob"] = torch.sigmoid(logit)               # p(AI-generated)
        else:
            probs = logits.softmax(dim=-1)
            out["logit"] = None
            out["probs"] = probs                              # (B, n_classes)
            out["prob"] = probs[:, 1]                         # p(AI-generated)
        return out

    def _temp(self) -> torch.Tensor:
        return self.temperature.clamp(min=1e-3)

    @torch.no_grad()
    def predict_proba(self, semantic: torch.Tensor, forensic: torch.Tensor | None = None):
        """Calibrated p(AI-generated) -- a scalar per image.

        For the 3-class head this is softmax(logits / T)[:, 1], i.e. the
        probability of the *fully synthetic* class. Real and tampered images
        both push this toward 0; that is deliberate (see class docstring).
        Every existing caller -- predict.py, evaluate.py, error_analysis.py --
        wants exactly this scalar and keeps working unchanged.
        """
        out = self.forward(semantic, forensic)
        if self.n_classes == 1:
            return torch.sigmoid(out["logit"] / self._temp())
        return (out["logits"] / self._temp()).softmax(dim=-1)[:, 1]

    @torch.no_grad()
    def predict_proba_full(self, semantic: torch.Tensor,
                           forensic: torch.Tensor | None = None):
        """Calibrated per-class probabilities, shape (B, n_classes).

        For the binary head this returns (B, 2) as [p(real), p(AI)] so the
        two heads present the same interface to app.py.
        """
        out = self.forward(semantic, forensic)
        if self.n_classes == 1:
            p_ai = torch.sigmoid(out["logit"] / self._temp())
            return torch.stack([1.0 - p_ai, p_ai], dim=-1)
        return (out["logits"] / self._temp()).softmax(dim=-1)


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

    Works for both heads: a binary sigmoid (labels float in {0,1}) and the
    3-class softmax (labels long in {0,1,2}), branching on the logit shape.
    """
    logits_c, logits_a = out_clean["logits"], out_aug["logits"]

    if logits_c.shape[-1] == 1:
        bce = nn.functional.binary_cross_entropy_with_logits
        lc, la = logits_c.squeeze(-1), logits_a.squeeze(-1)
        l_cls = 0.5 * (bce(lc, labels.float()) + bce(la, labels.float()))
        p_clean = torch.sigmoid(lc).clamp(1e-6, 1 - 1e-6)
        p_aug = torch.sigmoid(la).clamp(1e-6, 1 - 1e-6)

        def _kl(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
            return (p * (p / q).log() + (1 - p) * ((1 - p) / (1 - q)).log()).mean()

        l_cons = _kl(p_clean.detach(), p_aug)
    else:
        y = labels.long()
        l_cls = 0.5 * (F.cross_entropy(logits_c, y) + F.cross_entropy(logits_a, y))
        p_clean = logits_c.softmax(-1).clamp_min(1e-6)
        p_aug = logits_a.softmax(-1).clamp_min(1e-6)
        # KL(clean || aug), detached clean view as the target -- same rationale
        # as the binary case, generalised to a categorical distribution.
        l_cons = (p_clean.detach()
                  * (p_clean.detach().log() - p_aug.log())).sum(-1).mean()

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
