"""Robustness evaluation -- runs the trained model over every cell of the
canonical transform grid and writes a compact markdown table (results/D4).

Unlike training, this re-reads IMAGES (not cached features), because each
evaluation cell is a specific deterministic transform that must be applied
exactly as specified -- not sampled. The cached features used K random
views, which is right for training and wrong for measurement.

Usage:
    python -m src.evaluate --manifest data/manifest.csv --split test \
        --checkpoint checkpoints/full.pt --out results/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def _force_utf8_stdout() -> None:
    """Windows consoles default to cp1252, which cannot encode the table's
    'Δ' and '—'. The results file is written as UTF-8 regardless, but the
    final print would raise UnicodeEncodeError and exit non-zero on a run
    that had actually succeeded."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

from .data import manifest as M
from .metrics import (ROBUST_MODES, final_score, robustness_gap,
                      summarise_condition)
from .transforms.degradations import evaluation_grid


def score_images(model, backbone, preprocess, images, device, batch_size=32,
                 forensic_mu=None, forensic_sd=None):
    """Score PIL images with the trained detector.

    `forensic_mu` / `forensic_sd` MUST be the tensors saved in the checkpoint:
    training standardises the forensic features globally with train-split
    statistics, so the model only ever saw standardised inputs. Feeding raw
    features here is a silent train/test skew.
    """
    import torch

    from .features import frequency as FQ

    probs: list[float] = []
    buf_t, buf_f = [], []

    def flush():
        if not buf_t:
            return
        with torch.no_grad():
            x = torch.stack(buf_t).to(device)
            if device == "cuda":
                x = x.half()
            emb = backbone(x)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            f = torch.from_numpy(np.stack(buf_f).astype(np.float32)).to(device)
            if forensic_mu is not None and forensic_sd is not None:
                f = (f - forensic_mu.to(device)) / forensic_sd.to(device)
            p = model.predict_proba(emb.float(), f)
        probs.extend(p.cpu().numpy().tolist())
        buf_t.clear()
        buf_f.clear()

    for img in images:
        buf_t.append(preprocess(img))
        buf_f.append(FQ.extract(img))
        if len(buf_t) >= batch_size:
            flush()
    flush()
    return np.asarray(probs)


def render_table(rows: list[dict], auc_clean: float) -> str:
    """The compact robustness table. Delta column is the point of the study."""
    lines = [
        "| Condition | n | Acc | AUC | TPR@1%FPR | ECE | ΔAUC vs clean |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        delta = r["auc"] - auc_clean
        delta_str = "—" if r["condition"] == "clean" else f"{delta:+.3f}"
        lines.append(
            f"| {r['condition']} | {r['n']} | {r['acc']:.3f} | {r['auc']:.3f} "
            f"| {r['tpr@1%fpr']:.3f} | {r['ece']:.3f} | {delta_str} |"
        )
    return "\n".join(lines)


def main() -> int:
    _force_utf8_stdout()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--limit", type=int, default=0,
                    help="evaluate on a balanced random subsample of this size "
                         "(the full grid over a large test set is slow)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    import torch
    import open_clip

    from .features.extract import BACKBONES
    from .models.detector import Detector

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model_name, pretrained, _ = BACKBONES[cfg["backbone"]]
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained)
    backbone = clip_model.visual.to(device).eval()
    if device == "cuda":
        backbone = backbone.half()

    model = Detector(
        semantic_dim=cfg["semantic_dim"], forensic_dim=cfg["forensic_dim"],
        hidden=cfg.get("hidden", 256),
        use_forensic=cfg.get("use_forensic", True),
        use_gate=cfg.get("use_gate", True),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    eval_recs = [r for r in M.read(args.manifest) if r.split == args.split]
    if args.limit and len(eval_recs) > args.limit:
        rng = np.random.default_rng(0)
        pos = [r for r in eval_recs if r.label == 1]
        neg = [r for r in eval_recs if r.label == 0]
        k = args.limit // 2
        pick = lambda xs: [xs[i] for i in rng.permutation(len(xs))[:min(k, len(xs))]]
        eval_recs = pick(pos) + pick(neg)
        print(f"balanced subsample: {len(eval_recs)} of the {len(pos) + len(neg)} test images")
    if not eval_recs:
        raise SystemExit(f"no records in split '{args.split}'")

    print(f"evaluating {len(eval_recs)} images across the transform grid")
    y_true = np.array([r.label for r in eval_recs])
    base_images = []
    keep = []
    for i, r in enumerate(eval_recs):
        try:
            base_images.append(Image.open(r.image_path).convert("RGB"))
            keep.append(i)
        except Exception:
            continue
    y_true = y_true[keep]

    rows, cell_aucs = [], {}
    auc_clean = float("nan")

    forensic_mu = ckpt.get("forensic_mu")
    forensic_sd = ckpt.get("forensic_sd")
    if forensic_mu is None or forensic_sd is None:
        print("WARNING: checkpoint has no forensic_mu/sd -- scores will be "
              "wrong. Retrain with the current src/train.py.")

    for deg in evaluation_grid():
        rng = np.random.default_rng(0)  # deterministic noise per cell
        images = [deg.apply(im, rng) for im in base_images]
        probs = score_images(model, backbone, preprocess, images, device,
                             args.batch_size, forensic_mu, forensic_sd)
        stats = summarise_condition(y_true, probs)
        rows.append({"condition": deg.name, **stats})
        cell_aucs[deg.name] = stats["auc"]
        if deg.name == "clean":
            auc_clean = stats["auc"]
        print(f"  {deg.name:32s} acc {stats['acc']:.3f}  auc {stats['auc']:.3f}")

    scores = {m: final_score(auc_clean, cell_aucs, m) for m in ROBUST_MODES}
    gap = robustness_gap(auc_clean, cell_aucs)

    args.out.mkdir(parents=True, exist_ok=True)
    table = render_table(rows, auc_clean)
    summary = (
        f"# Robustness Evaluation\n\n"
        f"Split: `{args.split}` | checkpoint: `{args.checkpoint.name}` | "
        f"{len(base_images)} images x {len(evaluation_grid())} conditions\n\n"
        f"{table}\n\n"
        f"## Headline\n\n"
        f"- **AUC (clean): {auc_clean:.4f}**\n"
        f"- **Mean AUC drop under transformation: {gap:+.4f}**\n"
        f"- Final Score = 0.5*AUC_clean + 0.5*AUC_robust:\n"
        + "".join(f"  - `{m}`: **{scores[m]:.4f}**\n" for m in ROBUST_MODES)
        + "\nAUC_robust has three readings; `per_family` is the primary figure "
          "because it is not biased by how many parameter settings each "
          "transform family happens to contribute.\n"
    )
    (args.out / "robustness_table.md").write_text(summary, encoding="utf-8")
    (args.out / "robustness_raw.json").write_text(
        json.dumps({"rows": rows, "final_scores": scores,
                    "auc_clean": auc_clean, "gap": gap}, indent=2),
        encoding="utf-8")

    print("\n" + summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
