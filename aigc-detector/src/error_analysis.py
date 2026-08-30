"""Error analysis -- produces deliverable D5.

Surfaces the model's most confident mistakes, because those are the ones
that reveal what it actually learned. A low-confidence error is noise; a
false positive at p=0.98 means the model has found a shortcut and is
applying it wrongly.

Outputs:
    results/error_analysis.md      the written note
    results/fp_grid.png            highest-confidence false positives
    results/fn_grid.png            highest-confidence false negatives

The written note is where the "Innovation & Problem Insight" points live.
Do not just list the errors -- state a hypothesis for each cluster. The
question to answer is the one the workshop posed: is the model learning a
real generative artifact, or a dataset shortcut?

Usage:
    python -m src.error_analysis --manifest data/manifest.csv \
        --checkpoint checkpoints/full.pt --split test --out results/
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


def contact_sheet(paths, scores, out_path: Path, cols: int = 4, cell: int = 192,
                  label: str = "") -> None:
    """Grid of thumbnails with scores burned in, for the video and the note."""
    from PIL import ImageDraw

    n = len(paths)
    if n == 0:
        return
    rows = (n + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell, rows * (cell + 18)), "white")
    draw = ImageDraw.Draw(sheet)

    for i, (p, s) in enumerate(zip(paths, scores)):
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            continue
        img.thumbnail((cell, cell))
        x, y = (i % cols) * cell, (i // cols) * (cell + 18)
        sheet.paste(img, (x + (cell - img.width) // 2, y))
        draw.text((x + 4, y + cell + 2), f"p={s:.3f}  {Path(p).name[:22]}",
                  fill="black")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    print(f"wrote {out_path} ({n} {label})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--top-k", type=int, default=12)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    import torch
    import open_clip

    from .data import manifest as M
    from .evaluate import score_images
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

    records = [r for r in M.read(args.manifest) if r.split == args.split]
    if args.limit:
        records = records[: args.limit]

    images, kept = [], []
    for r in records:
        try:
            images.append(Image.open(r.image_path).convert("RGB"))
            kept.append(r)
        except Exception:
            continue

    probs = score_images(model, backbone, preprocess, images, device,
                         forensic_mu=ckpt.get("forensic_mu"),
                         forensic_sd=ckpt.get("forensic_sd"))
    y = np.array([r.label for r in kept])                 # 0 real, 1 AI
    pred = (probs >= args.threshold).astype(int)

    fp_idx = np.where((pred == 1) & (y == 0))[0]
    fn_idx = np.where((pred == 0) & (y == 1))[0]
    fp_idx = fp_idx[np.argsort(-probs[fp_idx])][: args.top_k]
    fn_idx = fn_idx[np.argsort(probs[fn_idx])][: args.top_k]

    contact_sheet([kept[i].image_path for i in fp_idx], probs[fp_idx],
                  args.out / "fp_grid.png", label="false positives")
    contact_sheet([kept[i].image_path for i in fn_idx], probs[fn_idx],
                  args.out / "fn_grid.png", label="false negatives")

    # Error rate by image kind -- a concentration is the signature of a
    # dataset shortcut rather than a real detection signal.
    fp_kind = Counter(kept[i].kind for i in np.where((pred == 1) & (y == 0))[0])
    fn_kind = Counter(kept[i].kind for i in np.where((pred == 0) & (y == 1))[0])
    total_kind = Counter(r.kind for r in kept)

    lines = [
        "# Error Analysis Note",
        "",
        f"Split `{args.split}` | {len(kept)} images | threshold {args.threshold}",
        "",
        f"- False positives (real called AI): **{int(((pred==1)&(y==0)).sum())}** "
        f"({((pred==1)&(y==0)).sum()/max((y==0).sum(),1):.1%} of real images)",
        f"- False negatives (AI called real): **{int(((pred==0)&(y==1)).sum())}** "
        f"({((pred==0)&(y==1)).sum()/max((y==1).sum(),1):.1%} of AI images)",
        "",
        "## Error rate by image kind",
        "",
        "| Kind | total | false pos | false neg | error rate |",
        "|---|---|---|---|---|",
    ]
    for kind, total in total_kind.most_common():
        fp, fn = fp_kind.get(kind, 0), fn_kind.get(kind, 0)
        lines.append(f"| {kind} | {total} | {fp} | {fn} | {(fp+fn)/max(total,1):.1%} |")

    lines += [
        "",
        "![false positives](fp_grid.png)",
        "",
        "![false negatives](fn_grid.png)",
        "",
        "## Interpretation",
        "",
        "> Write this by hand after looking at the grids. Questions to answer:",
        ">",
        "> 1. Do the false positives share a visual property (heavy texture,",
        ">    shallow depth of field, smooth skin, low resolution)? Real photos",
        ">    with smooth regions getting flagged means the model reads 'lack of",
        ">    high-frequency detail' as 'synthetic' -- a real artifact signal",
        ">    misfiring, not a bug.",
        "> 2. Do the tampered images fail more often than genuine photos? That",
        ">    is the AI-edited region leaking a detectable signal.",
        "> 3. What is the cost asymmetry? Calling a real photograph synthetic is",
        ">    an accusation against a person; missing a synthetic image is a gap",
        ">    in coverage. State which error the threshold favours.",
    ]

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "error_analysis.md").write_text("\n".join(lines), encoding="utf-8")
    (args.out / "error_analysis_raw.json").write_text(json.dumps({
        "fp_paths": [kept[i].image_path for i in fp_idx],
        "fn_paths": [kept[i].image_path for i in fn_idx],
        "fp_by_kind": dict(fp_kind),
        "fn_by_kind": dict(fn_kind),
    }, indent=2), encoding="utf-8")
    print(f"wrote {args.out/'error_analysis.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
