"""Metrics for the robustness evaluation.

The headline number is the organiser's own formula:

    Final Score = 0.50 * AUC_clean + 0.50 * AUC_robust

AUC_robust is NOT defined precisely in the brief (see open question O-5), so
this module implements three readings and reports all of them. That costs
almost nothing and means we cannot be caught out by whichever one they meant:

    mean        mean AUC over every transform cell        (optimistic)
    worst       minimum AUC over every transform cell     (pessimistic)
    per_family  mean over families, after taking the      (balanced)
                worst cell within each family

`per_family` is our primary internal figure. Plain `mean` over cells is
biased by how many parameter settings each family happens to have -- JPEG
contributes four cells and crop only one, so a detector that is great at
JPEG and terrible at cropping scores better than it deserves. Averaging
worst-per-family first removes that accident of the grid's shape.

ECE is included because the brief asks for a confidence score, and a
confidence that is not calibrated is just a logit wearing a disguise.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score

ROBUST_MODES = ("mean", "worst", "per_family")


def auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """ROC AUC, guarding the single-class case that breaks sklearn."""
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def accuracy(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> float:
    y_true = np.asarray(y_true).ravel()
    pred = (np.asarray(y_score).ravel() >= threshold).astype(int)
    return float((pred == y_true).mean())


def tpr_at_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float = 0.01) -> float:
    """Detection rate at a fixed low false-positive rate.

    This matters more than accuracy for the deployment story: a moderation
    system cares how much synthetic content it catches while wrongly
    flagging at most 1% of genuine photographs.
    """
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()
    if len(np.unique(y_true)) < 2:
        return float("nan")
    neg = np.sort(y_score[y_true == 0])
    if len(neg) == 0:
        return float("nan")
    idx = int(np.ceil((1 - target_fpr) * len(neg))) - 1
    thr = neg[np.clip(idx, 0, len(neg) - 1)]
    return float((y_score[y_true == 1] > thr).mean())


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15
) -> float:
    """Standard equal-width-bin ECE."""
    y_true = np.asarray(y_true).ravel().astype(float)
    y_prob = np.asarray(y_prob).ravel()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, edges) - 1, 0, n_bins - 1)

    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        conf = y_prob[mask].mean()
        acc = y_true[mask].mean()
        ece += (mask.mean()) * abs(acc - conf)
    return float(ece)


def brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((np.asarray(y_prob).ravel() - np.asarray(y_true).ravel()) ** 2))


def final_score(
    auc_clean: float,
    cell_aucs: dict[str, float],
    mode: str = "per_family",
) -> float:
    """0.50 * AUC_clean + 0.50 * AUC_robust, for one reading of AUC_robust.

    `cell_aucs` maps a cell name like "jpeg_q30" to its AUC. The family is
    taken as the part before the first underscore.
    """
    if mode not in ROBUST_MODES:
        raise ValueError(f"mode must be one of {ROBUST_MODES}")

    vals = {k: v for k, v in cell_aucs.items()
            if k != "clean" and not np.isnan(v)}
    if not vals:
        return float("nan")

    if mode == "mean":
        robust = float(np.mean(list(vals.values())))
    elif mode == "worst":
        robust = float(np.min(list(vals.values())))
    else:
        by_family: dict[str, list[float]] = defaultdict(list)
        for name, value in vals.items():
            by_family[name.split("_")[0]].append(value)
        robust = float(np.mean([min(v) for v in by_family.values()]))

    return 0.5 * auc_clean + 0.5 * robust


def robustness_gap(auc_clean: float, cell_aucs: dict[str, float]) -> float:
    """M4: mean absolute AUC drop from clean to transformed.

    This is the metric the whole project is really about, so it gets a name
    rather than being buried in a table.
    """
    vals = [v for k, v in cell_aucs.items() if k != "clean" and not np.isnan(v)]
    if not vals:
        return float("nan")
    return float(auc_clean - np.mean(vals))


def summarise_condition(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """All per-condition metrics for one row of the robustness table."""
    return {
        "n": int(len(y_true)),
        "acc": accuracy(y_true, y_prob),
        "auc": auc(y_true, y_prob),
        "tpr@1%fpr": tpr_at_fpr(y_true, y_prob, 0.01),
        "ece": expected_calibration_error(y_true, y_prob),
    }
