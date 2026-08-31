"""Add the fetched external DALL-E 3 / GAN images to training, additively.

Consumes scripts/fetch_external_ai.py's output:

    data/external_ai/fake_dalle3/   -> label 1, kind ext_dalle3
    data/external_ai/fake_progan/   -> label 1, kind ext_progan
    data/external_ai/real_lsun/     -> label 0, kind lsun
    data/external_ai/real_square/   -> label 0, kind real_square

Same additive contract as build_dataset.py and add_hard_negatives.py: the
existing manifest is kept, every image that already had a train/test split
keeps it, and only the new files are routed. Nothing the current checkpoint
was evaluated on moves into training.

The real buckets are the confound partners, not filler: `real_lsun` balances
the 256px ProGAN images so resolution carries no signal, and `real_square`
balances the square DALL-E images so aspect ratio carries no signal. Dropping
them would reintroduce exactly the shortcuts a BigGAN attempt tripped on.

Usage:
    python scripts/fetch_external_ai.py --per-bucket 3000
    python scripts/add_external_ai.py
    python -m src.features.extract --manifest data/manifest.csv --out features/ \
        --views 4 --backbone ViT-B-16 --batch-size 128 --workers 2
    python -m src.train --features features/ --out checkpoints/ --tag full
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from collections import Counter
from pathlib import Path

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
FIELDS = ["image_path", "label", "kind", "split"]

BUCKETS = {
    "fake_dalle3": (1, "ext_dalle3", "ai"),
    "fake_progan": (1, "ext_progan", "ai"),
    "real_lsun": (0, "lsun", "real"),
    "real_square": (0, "real_square", "real"),
}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=Path("data/external_ai"))
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    ap.add_argument("--test-fraction", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    train, test = args.out / "train", args.out / "test"

    rows: list[dict] = []
    placed: set[str] = set()
    if args.manifest.exists():
        with args.manifest.open(newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if Path(r["image_path"]).exists():
                    rows.append(r)
                    placed.add(Path(r["image_path"]).name)
        print(f"existing manifest: {len(rows)} rows kept")
    if any(r["kind"].startswith(("ext_", "lsun", "real_square")) for r in rows):
        raise SystemExit("external-AI rows already in the manifest -- nothing to do.")

    added: list[dict] = []
    for folder, (label, kind, cls) in BUCKETS.items():
        src_dir = args.raw / folder
        files = sorted(p for p in src_dir.glob("*")
                       if p.suffix.lower() in IMG_EXTS) if src_dir.is_dir() else []
        # recover anything a previous interrupted run already moved
        files += [p for p in (train / cls).glob(f"{kind}_*")
                  if p.name not in placed]
        files += [p for p in (test / cls).glob(f"{kind}_*")
                  if p.name not in placed]
        files = sorted(set(files))
        if not files:
            print(f"  {folder}: nothing to add")
            continue
        files = [files[i] for i in rng.sample(range(len(files)), len(files))]
        n_test = round(len(files) * args.test_fraction)
        for i, src in enumerate(files):
            split = "test" if i < n_test else "train"
            dst_dir = (test if split == "test" else train) / cls
            dst_dir.mkdir(parents=True, exist_ok=True)
            name = src.name if src.name.startswith(f"{kind}_") else f"{kind}_{src.name}"
            dst = dst_dir / name
            if src.resolve() != dst.resolve():
                shutil.move(str(src), dst)
            added.append({"image_path": str(dst), "label": label,
                          "kind": kind, "split": split})
        print(f"  {folder:12s} +{len(files):5d}  ({len(files) - n_test} train / {n_test} test)")

    if not added:
        raise SystemExit(f"no images under {args.raw} -- run fetch_external_ai.py first")

    rows += added
    rows.sort(key=lambda r: r["image_path"])
    with args.manifest.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    shutil.rmtree(args.raw, ignore_errors=True)

    n_tr = sum(r["split"] == "train" for r in rows)
    n_ai = sum(int(r["label"]) == 1 for r in rows)
    print(f"\nmanifest: {len(rows)} rows (+{len(added)})  "
          f"train {n_tr}  test {len(rows) - n_tr}  |  real {len(rows) - n_ai}  ai {n_ai}")
    print("  kinds: " + ", ".join(f"{k}={v}" for k, v in
                                  sorted(Counter(r["kind"] for r in rows).items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
