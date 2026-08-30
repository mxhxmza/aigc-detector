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
        --checkpoint checkpoints/run.pt --split val --out results/
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
    ap.add_argument("--split", default="val")
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
        n_classes=cfg.get("n_classes", 1),
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
    # AI-detection view: class 1 vs the rest. probs is p(fully AI-generated),
    # so tampered (label 2) is a negative -- flagging it AI is a false positive.
    y = np.array([1 if r.label == 1 else 0 for r in kept])
    pred = (probs >= args.threshold).astype(int)

    fp_idx = np.where((pred == 1) & (y == 0))[0]
    fn_idx = np.where((pred == 0) & (y == 1))[0]
    fp_idx = fp_idx[np.argsort(-probs[fp_idx])][: args.top_k]
    fn_idx = fn_idx[np.argsort(probs[fn_idx])][: args.top_k]

    contact_sheet([kept[i].image_path for i in fp_idx], probs[fp_idx],
                  args.out / "fp_grid.png", label="false positives")
    contact_sheet([kept[i].image_path for i in fn_idx], probs[fn_idx],
                  args.out / "fn_grid.png", label="false negatives")

    # Which generators and sources do the errors concentrate in? A strong
    # concentration is the fingerprint of a shortcut.
    fn_gens = Counter(kept[i].generator for i in np.where((pred == 0) & (y == 1))[0])
    fp_srcs = Counter(kept[i].source for i in np.where((pred == 1) & (y == 0))[0])
    all_gens = Counter(r.generator for r in kept if r.label == 1)

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
        "## Miss rate by generator",
        "",
        "| Generator | total | missed | miss rate |",
        "|---|---|---|---|",
    ]
    for gen, total in all_gens.most_common():
        missed = fn_gens.get(gen, 0)
        lines.append(f"| {gen} | {total} | {missed} | {missed/max(total,1):.1%} |")

    lines += [
        "",
        "## False positives by source dataset",
        "",
        "| Source | false positives |",
        "|---|---|",
    ] + [f"| {s} | {c} |" for s, c in fp_srcs.most_common()]

    lines += [
        "",
        "![false positives](fp_grid.png)",
        "",
        "![false negatives](fn_grid.png)",
        "",
        "## Interpretation",
        "",
        "> TODO -- write this by hand after looking at the grids. This section",
        "> is what earns Innovation & Problem Insight; an auto-generated table",
        "> alone does not. Questions to answer:",
        ">",
        "> 1. Do the false positives share a visual property (heavy texture,",
        ">    shallow depth of field, smooth skin, low resolution)? If real",
        ">    photos with smooth regions get flagged, the model may be reading",
        ">    'lack of high-frequency detail' as 'synthetic' -- which is a",
        ">    genuine artifact signal misfiring, not a bug, and should be said",
        ">    plainly.",
        "> 2. Is the miss rate concentrated in one generator? Strong",
        ">    concentration means the detector generalises worse than the",
        ">    aggregate AUC suggests.",
        "> 3. Do false positives concentrate in one SOURCE dataset? If so,",
        ">    suspect a dataset shortcut (resolution, compression history,",
        ">    camera pipeline) rather than a real detection signal.",
        "> 4. What is the cost asymmetry? Calling a real photograph synthetic",
        ">    is an accusation against a person; missing a synthetic image is",
        ">    a gap in coverage. State which error the threshold favours.",
    ]

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "error_analysis.md").write_text("\n".join(lines), encoding="utf-8")
    (args.out / "error_analysis_raw.json").write_text(json.dumps({
        "fp_paths": [kept[i].image_path for i in fp_idx],
        "fn_paths": [kept[i].image_path for i in fn_idx],
        "miss_by_generator": dict(fn_gens),
        "fp_by_source": dict(fp_srcs),
    }, indent=2), encoding="utf-8")
    print(f"wrote {args.out/'error_analysis.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
