"""Hard-negative mining for the real class.

The detector's residual errors are almost all one shape: a *polished* real
photograph -- travel/landscape framing, shallow depth of field, saturated
colour, foreground bokeh -- read as synthetic, because the training reals
skew more casual than the synthetic images. This adds targeted "real"
examples to fix that failure without touching anything else.

Two sources of hard negatives:

  1. --extra <dir>   real photographs you supply (e.g. ones the model got
                     wrong). Each is augmented into several spatial crops so
                     the model learns the content, not one exact framing.
  2. mined           the training reals the current checkpoint scores highest
                     as AI (the pattern, generalised). The top --mine of them
                     are duplicated with fresh augmentation.

Everything lands in ``data/train/real/`` with ``kind = hard_real`` and is
appended to ``data/manifest.csv``. Purely additive -- no existing image or
row is changed, so the model keeps everything it already learned. A slice is
held out to ``data/test/`` so the fix can be checked without memorisation.

Run after training, then re-extract features and retrain:

    python scripts/add_hard_negatives.py --checkpoint checkpoints/full.pt --extra data/check
    python -m src.features.extract --manifest data/manifest.csv --out features/ ...
    python -m src.train --features features/ --out checkpoints/ --tag full
"""

from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def augment(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    """One randomised view: spatial crop + flip + mild photometric / codec noise."""
    w, h = img.size
    f = rng.uniform(0.62, 0.96)          # keep enough scene context in every crop
    cw, ch = max(32, int(w * f)), max(32, int(h * f))
    x = int(rng.integers(0, w - cw + 1))
    y = int(rng.integers(0, h - ch + 1))
    im = img.crop((x, y, x + cw, y + ch))
    if rng.random() < 0.5:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    for enh in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
        if rng.random() < 0.6:
            im = enh(im).enhance(float(rng.uniform(0.82, 1.18)))
    if rng.random() < 0.55:
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=int(rng.integers(58, 96)))
        buf.seek(0)
        im = Image.open(buf).convert("RGB")
    if rng.random() < 0.4:
        s = rng.uniform(0.4, 0.85)
        im = im.resize((max(32, int(im.width * s)), max(32, int(im.height * s))))
    return im


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, default=Path("checkpoints/full.pt"))
    ap.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    ap.add_argument("--extra", type=Path,
                    help="directory of hard real photos to fold in")
    ap.add_argument("--extra-views", type=int, default=60,
                    help="augmented crops per --extra image")
    ap.add_argument("--mine", type=int, default=40,
                    help="mine this many of the hardest training reals")
    ap.add_argument("--mine-views", type=int, default=3)
    ap.add_argument("--test-fraction", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.data import manifest as M
    from src.inference import Scorer

    rows = list(csv.DictReader(args.manifest.open(encoding="utf-8")))
    if any(r["kind"] == "hard_real" for r in rows):
        raise SystemExit("hard_real rows already in the manifest -- nothing to do.")

    rng = np.random.default_rng(args.seed)
    scorer = Scorer(args.checkpoint)
    train_dir = Path("data/train/real")
    test_dir = Path("data/test/real")
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    sources: list[tuple[str, Image.Image, int]] = []   # (tag, image, n_views)

    if args.extra and args.extra.is_dir():
        for p in sorted(args.extra.rglob("*")):
            if p.suffix.lower() in IMAGE_EXTS:
                sources.append((f"extra_{p.stem}", Image.open(p).convert("RGB"),
                                args.extra_views))
        print(f"--extra: {sum(1 for s in sources if s[0].startswith('extra_'))} images")

    if args.mine:
        reals = [r for r in M.read(args.manifest)
                 if r.split == "train" and r.label == 0]
        imgs = [Image.open(r.image_path).convert("RGB") for r in reals]
        p_ai = np.array(scorer.score_many(imgs, batch_size=96))
        hardest = np.argsort(-p_ai)[: args.mine]
        print(f"mined {len(hardest)} hardest train reals "
              f"(p(AI) {p_ai[hardest].min():.2f}..{p_ai[hardest].max():.2f})")
        for i in hardest:
            sources.append((f"mine_{reals[i].kind}_{Path(reals[i].image_path).stem}",
                            imgs[i], args.mine_views))

    n_train = n_test = 0
    for tag, img, nv in sources:
        for v in range(nv):
            out = augment(img, rng)
            split_test = rng.random() < args.test_fraction
            dst = (test_dir if split_test else train_dir) / f"hard_real_{tag}_{v}.png"
            out.save(dst, "PNG")
            rows.append({"image_path": str(dst), "label": 0, "kind": "hard_real",
                         "split": "test" if split_test else "train"})
            n_test += split_test
            n_train += not split_test

    rows.sort(key=lambda r: r["image_path"])
    with args.manifest.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["image_path", "label", "kind", "split"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nadded {n_train} hard_real train + {n_test} test rows -> {args.manifest}")
    print("now: re-extract features, then retrain.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
