"""Fetch a balanced subset of SID_Set to disk for scripts/build_dataset.py.

SID_Set (https://huggingface.co/datasets/saberzl/SID_Set) is 140 GB of
1024px images -- 210k train + 30k val, labelled:

    0 = real (photographs from OpenImages V7)
    1 = full synthetic (fully AI-generated)
    2 = tampered (real photo with an AI-edited region)

The full dataset is far too big to download, and row-level streaming off the
HF CDN is slow and drops connections. Instead this pulls whole parquet shards
(~490 MB / ~844 images each, resumable, cached by huggingface_hub) and reads
them locally until each kept class reaches --per-class. Images are saved as
PNG at <=512px under:

    <out>/real/<img_id>.png                          label 0
    <out>/fake/full_synthetic/<img_id>.png            label 1
    <out>/fake/tampered/<img_id>.png                  label 2  (--include-tampered)

`build_dataset.py --raw <out>` picks these up directly, splits them into
train/ and test/ folders, and folds `tampered` into the real class.

Two deliberate normalisations, both to stop the model learning the source
instead of the label:

  * everything is re-saved as PNG. SID real images arrive as JPEG and
    synthetic ones as PNG; a detector that reads JPEG history would otherwise
    get a shortcut that will not exist at deployment.
  * everything is downscaled to <=512px on the long side. The forensic branch
    analyses at 256px and CLIP at 224px, so this is lossless for the model
    and keeps 18k images near 6 GB instead of 40.

Resumable at two levels: HF caches partial shard downloads, and img_ids
already written are skipped.

Usage:
    python scripts/fetch_sid_set.py --out data/sid_set --per-class 10000 --include-tampered
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

from PIL import Image

N_SHARDS = {"train": 249, "validation": 34}

LABEL_DIRS = {
    0: "real",
    1: "fake/full_synthetic",
    2: "fake/tampered",
}


def _save(raw: bytes, path: Path, max_size: int) -> None:
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    w, h = img.size
    if max(w, h) > max_size:
        s = max_size / max(w, h)
        img = img.resize((max(1, round(w * s)), max(1, round(h * s))))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="PNG")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--split", default="train", choices=list(N_SHARDS))
    ap.add_argument("--per-class", type=int, default=9000,
                    help="target images for EACH kept class")
    ap.add_argument("--include-tampered", action="store_true",
                    help="also fetch label-2 images (real photo + AI-edited "
                         "region). Off by default: keeps the task 'fully "
                         "generated vs real'.")
    ap.add_argument("--max-size", type=int, default=512)
    ap.add_argument("--start-shard", type=int, default=0,
                    help="skip the first N shards (use a disjoint range for a "
                         "second, non-overlapping fetch)")
    args = ap.parse_args()

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    keep = {0, 1} | ({2} if args.include_tampered else set())
    target = {k: args.per_class for k in keep}

    have = {k: 0 for k in keep}
    seen: set[str] = set()
    for k in keep:
        d = args.out / LABEL_DIRS[k]
        if d.is_dir():
            for p in d.glob("*.png"):
                have[k] += 1
                seen.add(p.stem)
    if any(have.values()):
        print(f"resuming: already have {have}")

    n_total = N_SHARDS[args.split]
    t0 = time.time()
    written = 0

    for shard in range(args.start_shard, n_total):
        if all(have[k] >= target[k] for k in keep):
            break
        name = f"data/{args.split}-{shard:05d}-of-{n_total:05d}.parquet"
        try:
            local = hf_hub_download("saberzl/SID_Set", name, repo_type="dataset")
        except Exception as exc:  # noqa: BLE001
            print(f"  shard {shard}: download failed ({exc}); skipping", file=sys.stderr)
            continue

        table = pq.read_table(local, columns=["img_id", "image", "label"])
        rows = table.to_pylist()
        # free the pyarrow buffers before decoding a few hundred PNGs
        del table

        for row in rows:
            label = row["label"]
            if label not in keep or have[label] >= target[label]:
                continue
            img_id = row["img_id"]
            if img_id in seen:
                continue
            raw = row["image"]["bytes"]
            try:
                _save(raw, args.out / LABEL_DIRS[label] / f"{img_id}.png", args.max_size)
            except Exception as exc:  # noqa: BLE001
                print(f"  skip {img_id}: {exc}", file=sys.stderr)
                continue
            seen.add(img_id)
            have[label] += 1
            written += 1

        rate = written / max(time.time() - t0, 1e-6)
        print(f"shard {shard:3d}/{n_total} | {have} | {rate:.1f} img/s "
              f"| {(time.time() - t0) / 60:.1f} min", flush=True)

    ok = all(have[k] >= target[k] for k in keep)
    print(f"\n{'done' if ok else 'INCOMPLETE'}: {have} in "
          f"{(time.time() - t0) / 60:.1f} min -> {args.out}")
    if not ok:
        print("re-run to continue, or lower --per-class.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
