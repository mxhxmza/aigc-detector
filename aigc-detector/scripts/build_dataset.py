"""Turn the raw SID_Set download into the training corpus.

`fetch_sid_set.py` leaves images under ``data/sid_set/`` in three folders:

    real/                    genuine photographs            -> label 0
    fake/full_synthetic/     fully AI-generated             -> label 1
    fake/tampered/           real photo, AI-edited region   -> label 0

This script makes a deterministic, stratified train/test split and *moves*
the images into their final home:

    data/train/real/   data/train/ai/
    data/test/real/    data/test/ai/

and writes ``data/manifest.csv`` (columns: ``image_path,label,kind,split``)
which is the single source of truth for every downstream step.

Tampered images are folded into ``real`` on disk: an edited photograph was
still taken by a person, so the detector treats it as authentic. The
original kind is kept in the manifest's ``kind`` column for error analysis.

Splitting by physically separating the files makes the "never trained on
test" guarantee structural -- an image is in exactly one folder -- rather
than something a hash check has to police after the fact.

Usage:
    python scripts/fetch_sid_set.py --out data/sid_set --per-class 10000 --include-tampered
    python scripts/build_dataset.py            # consumes data/sid_set/
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def classify(rel_parts: tuple[str, ...]) -> tuple[int, str, str] | None:
    """(label, kind, destination class dir) from a path relative to --raw.

    real/**            -> (0, real,      real)
    fake/tampered/**   -> (0, tampered,  real)   an edited photo is still a photo
    fake/**            -> (1, synthetic, ai)
    """
    lower = [p.lower() for p in rel_parts]
    if "tampered" in lower:
        return 0, "tampered", "real"
    if "real" in lower:
        return 0, "real", "real"
    if "fake" in lower:
        return 1, "synthetic", "ai"
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=Path("data/sid_set"),
                    help="raw download from fetch_sid_set.py")
    ap.add_argument("--out", type=Path, default=Path("data"),
                    help="parent of train/ and test/")
    ap.add_argument("--test-fraction", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    args = ap.parse_args()

    train_dir, test_dir = args.out / "train", args.out / "test"
    if not args.raw.is_dir():
        raise SystemExit(
            f"{args.raw} not found. Run scripts/fetch_sid_set.py first "
            "(with --include-tampered)."
        )

    rng = random.Random(args.seed)

    # group every image by (label, kind, class dir), then split each group so
    # the train/test class balance matches the whole corpus.
    groups: dict[tuple[int, str, str], list[Path]] = {}
    for path in sorted(args.raw.rglob("*")):
        if path.suffix.lower() not in IMAGE_EXTS or not path.is_file():
            continue
        hit = classify(path.relative_to(args.raw).parts)
        if hit:
            groups.setdefault(hit, []).append(path)

    rows: list[dict] = []
    for (label, kind, cls), files in sorted(groups.items()):
        rng.shuffle(files)
        n_test = round(len(files) * args.test_fraction)
        for i, path in enumerate(files):
            split = "test" if i < n_test else "train"
            dest_dir = (test_dir if split == "test" else train_dir) / cls
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / f"{kind}_{path.stem}{path.suffix.lower()}"
            shutil.move(str(path), dest)
            rows.append({"image_path": str(dest), "label": label,
                         "kind": kind, "split": split})
        print(f"{kind:12s} {len(files):6d}  ->  {len(files) - n_test} train / {n_test} test")

    if not rows:
        raise SystemExit(f"no images found under {args.raw} (expected real/ and fake/).")

    rows.sort(key=lambda r: r["image_path"])
    with args.manifest.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["image_path", "label", "kind", "split"])
        w.writeheader()
        w.writerows(rows)

    # tidy the now-empty raw tree
    shutil.rmtree(args.raw, ignore_errors=True)

    n_train = sum(r["split"] == "train" for r in rows)
    n_ai = sum(r["label"] == 1 for r in rows)
    print(f"\nwrote {len(rows)} rows to {args.manifest}")
    print(f"  train {n_train}   test {len(rows) - n_train}")
    print(f"  real {len(rows) - n_ai}   ai {n_ai}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
