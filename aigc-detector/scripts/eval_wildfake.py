"""Evaluate the detector on the WildFake reference benchmark (EVAL ONLY).

`techjam-aigc/wildfake-eval-subset` is the track's demonstration subset. It is
**never trained on** — this script only reads it, and nothing it touches enters
`data/manifest.csv`. The final test set is drawn from the same corpus, so
training on it would leak.

Four configs, and which one you quote matters enormously:

    default          13,841  spec-faithful (4,998 COCO val2017 + 8,843 DALL-E 3)
    normalized       13,841  same images, all 200x200 -- size cue removed
    laion_matched     7,652  LAION vs DALL-E 3, both natively >=1024px -> 512x512
    cross_generator   5,494  LAION vs DALL-E 3 / Midjourney v5 / SDXL / GigaGAN

`default` is separable by image size alone: every COCO real is exactly 200x200
and no DALL-E fake is, so a two-line rule scores AUC 1.000 without a model.
This script measures that shortcut alongside the model so the two numbers are
never confused. `laion_matched` is the honest comparison; `cross_generator`
answers whether anything holds past DALL-E 3.

Classes are unbalanced in `default` (36/64), so balanced accuracy and AUC are
reported rather than raw accuracy.

Usage:
    python scripts/eval_wildfake.py --checkpoint checkpoints/full.pt --out results/
    python scripts/eval_wildfake.py --configs laion_matched --limit 500   # quick look
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

CONFIGS = ("default", "normalized", "laion_matched", "cross_generator")
REPO = "techjam-aigc/wildfake-eval-subset"


def size_shortcut_auc(sizes: list[tuple[int, int]], y: np.ndarray) -> float:
    """AUC of the no-model rule `real iff the image is exactly 200x200`.

    Quantifies how much of a config is winnable without looking at content.
    0.5 means the cue is gone; 1.0 means the config is fully gameable.
    """
    from sklearn.metrics import roc_auc_score
    guess = np.array([0.0 if s == (200, 200) else 1.0 for s in sizes])
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, guess))


def evaluate(scorer, config: str, limit: int, batch_size: int) -> dict:
    from datasets import load_dataset
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 confusion_matrix, precision_recall_fscore_support,
                                 roc_auc_score)

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.metrics import expected_calibration_error, tpr_at_fpr

    ds = load_dataset(REPO, config, split="validation")
    if limit and len(ds) > limit:
        ds = ds.select(range(limit))

    y, p_ai, sources, sizes = [], [], [], []
    t0 = time.time()
    for start in range(0, len(ds), batch_size):
        rows = ds[start : start + batch_size]
        imgs = [im.convert("RGB") for im in rows["image"]]
        sizes += [im.size for im in imgs]
        p_ai += scorer.score_many(imgs, batch_size=batch_size)
        y += list(rows["label"])
        sources += list(rows["source"])
        done = start + len(imgs)
        print(f"  {config}: {done}/{len(ds)}  ({done / max(time.time() - t0, 1e-9):.0f} img/s)",
              end="\r", flush=True)
    print()

    y = np.array(y)
    p_ai = np.array(p_ai)
    sources = np.array(sources)
    pred = (p_ai >= 0.5).astype(int)

    P, R, F1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()

    # The model is calibrated on SID_Set; this benchmark is a different
    # distribution, so 0.5 need not be the right operating point here. Sweeping
    # it separates "cannot separate the classes" (AUC would be low too) from
    # "separates them but the threshold is off" (AUC high, balanced acc low).
    grid = np.unique(np.quantile(p_ai, np.linspace(0, 1, 501)))
    bal = [balanced_accuracy_score(y, (p_ai >= t).astype(int)) for t in grid]
    best_i = int(np.argmax(bal))

    out = {
        "config": config,
        "n": int(len(y)),
        "n_real": int((y == 0).sum()),
        "n_fake": int((y == 1).sum()),
        "auc": float(roc_auc_score(y, p_ai)),
        "balanced_acc": float(balanced_accuracy_score(y, pred)),
        "balanced_acc_best": float(bal[best_i]),
        "best_threshold": float(grid[best_i]),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(P), "recall": float(R), "f1": float(F1),
        "tpr@1%fpr": float(tpr_at_fpr(y, p_ai, 0.01)),
        "ece": float(expected_calibration_error(y, p_ai)),
        "fp": int(fp), "fn": int(fn), "tn": int(tn), "tp": int(tp),
        "size_shortcut_auc": size_shortcut_auc(sizes, y),
        "seconds": round(time.time() - t0, 1),
        "_p_ai": p_ai, "_y": y, "_source": sources,   # popped out to the .npz
    }

    # Per-generator: reals stay fixed, each generator scored against them.
    real_mask = y == 0
    per_gen = {}
    for gen in sorted(set(sources[y == 1])):
        m = real_mask | (sources == gen)
        if len(np.unique(y[m])) < 2:
            continue
        per_gen[gen] = {
            "n_fake": int((sources == gen).sum()),
            "auc": float(roc_auc_score(y[m], p_ai[m])),
            "recall": float((pred[sources == gen] == 1).mean()),
            "mean_p_ai": float(p_ai[sources == gen].mean()),
        }
    if per_gen:
        out["per_generator"] = per_gen
    for src in sorted(set(sources[real_mask])):
        out.setdefault("per_real_source", {})[src] = {
            "n": int((sources == src).sum()),
            "fp_rate": float((pred[sources == src] == 1).mean()),
            "mean_p_ai": float(p_ai[sources == src].mean()),
        }
    return out


def render(rows: list[dict], checkpoint: str) -> str:
    L = [
        "# WildFake Reference Benchmark",
        "",
        f"`{REPO}` | checkpoint: `{checkpoint}` | **evaluation only — never trained on**",
        "",
        "The track's demonstration subset. Which config a number comes from matters:",
        "`default` is separable by image size alone (every COCO real is exactly",
        "200x200, no DALL-E fake is), so its AUC is close to meaningless on its own.",
        "`laion_matched` is the honest comparison — both classes natively >=1024px,",
        "put through one identical downscale.",
        "",
        "| Config | n | real/fake | AUC | Balanced acc @0.5 | Balanced acc @best | F1 | TPR@1%FPR | ECE | size-cue AUC |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        L.append(
            f"| `{r['config']}` | {r['n']} | {r['n_real']}/{r['n_fake']} | "
            f"**{r['auc']:.4f}** | {r['balanced_acc']:.4f} | "
            f"{r['balanced_acc_best']:.4f} (t={r['best_threshold']:.3f}) | {r['f1']:.4f} | "
            f"{r['tpr@1%fpr']:.3f} | {r['ece']:.3f} | {r['size_shortcut_auc']:.3f} |"
        )
    L += [
        "",
        "`size-cue AUC` is the no-model rule *real iff exactly 200x200*, scored on the",
        "same rows: 1.000 means the config is fully winnable without looking at the",
        "image, 0.500 means that cue carries no information.",
        "",
        "`Balanced acc @best` sweeps the decision threshold. The gap from `@0.5` is",
        "*calibration* drift, not a failure to separate the classes: the model's",
        "temperature was fitted on SID_Set, and this is a different distribution, so",
        "the scores are ranked well but shifted. A high AUC with a much lower",
        "balanced accuracy at 0.5 means the operating point is wrong, not the model.",
        "",
    ]

    for r in rows:
        if "per_generator" not in r:
            continue
        L += [f"## Per generator — `{r['config']}`", "",
              "Each generator scored against the same real images.", "",
              "| Generator | n | AUC | Recall @0.5 | mean p(AI) |", "|---|---|---|---|---|"]
        for gen, g in sorted(r["per_generator"].items(), key=lambda kv: -kv[1]["auc"]):
            L.append(f"| {gen} | {g['n_fake']} | {g['auc']:.4f} | "
                     f"{g['recall']:.3f} | {g['mean_p_ai']:.3f} |")
        L.append("")

    L += ["## False positives by real source", "",
          "| Config | Source | n | flagged as AI | mean p(AI) |", "|---|---|---|---|---|"]
    for r in rows:
        for src, s in sorted(r.get("per_real_source", {}).items()):
            L.append(f"| `{r['config']}` | {src} | {s['n']} | "
                     f"{s['fp_rate']:.2%} | {s['mean_p_ai']:.3f} |")
    L.append("")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, default=Path("checkpoints/full.pt"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS), choices=CONFIGS)
    ap.add_argument("--limit", type=int, default=0, help="rows per config (0 = all)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.inference import Scorer

    scorer = Scorer(args.checkpoint, device=args.device)
    print(f"{scorer.config['backbone']} on {scorer.device} | "
          f"temperature {scorer.temperature:.3f}\n")

    rows = [evaluate(scorer, c, args.limit, args.batch_size) for c in args.configs]

    args.out.mkdir(parents=True, exist_ok=True)
    # Raw scores, so a threshold or per-source question can be answered later
    # without re-running the whole benchmark.
    np.savez_compressed(args.out / "wildfake_scores.npz",
                        **{f"{r['config']}_{k}": np.asarray(r.pop(f"_{k}"))
                           for r in rows for k in ("p_ai", "y", "source")})
    (args.out / "wildfake_benchmark.md").write_text(
        render(rows, args.checkpoint.name), encoding="utf-8")
    (args.out / "wildfake_benchmark.json").write_text(
        json.dumps({"dataset": REPO, "checkpoint": str(args.checkpoint),
                    "trained_on": False, "results": rows}, indent=2), encoding="utf-8")
    print("\n" + render(rows, args.checkpoint.name))
    print(f"wrote {args.out / 'wildfake_benchmark.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
