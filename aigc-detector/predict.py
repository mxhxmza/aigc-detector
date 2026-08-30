#!/usr/bin/env python3
"""Score a directory of images for AI-generated likelihood.

This is the graded deliverable. The brief specifies:

    "A script that takes an image directory as input and outputs a
     confidence score for each image, indicating the likelihood that it is
     AIGC-generated. The output should be a JSON file containing
     `image_path` and `pred` for each image."

Usage
-----
    python predict.py --image-dir path/to/images --out predictions.json

Output (default `--format list`):

    [
      {"image_path": "images/a.jpg", "pred": 0.9312},
      {"image_path": "images/b.png", "pred": 0.0417}
    ]

`pred` is a CALIBRATED probability in [0,1] that the image is AI-generated.
The brief says "confidence score ... indicating the likelihood", which reads
as a probability rather than a hard label, so that is the default. The
`--format dict` and `--binary` flags cover the alternate readings without a
code change.

Unreadable and non-image files are skipped and reported rather than crashing
the run -- a single corrupt file must never take down a judge's evaluation.

Stub mode
---------
    python predict.py --image-dir imgs --out preds.json --stub

Emits schema-correct output with random scores and NO dependency on torch or
trained weights, so the output contract can be checked before a model exists.
Never submit with it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def find_images(root: Path) -> list[Path]:
    if not root.exists():
        raise SystemExit(f"error: image directory does not exist: {root}")
    if root.is_file():
        return [root]
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def write_output(
    rows: list[tuple[str, float]],
    out_path: Path,
    fmt: str,
    binary: bool,
    threshold: float,
) -> None:
    def _value(score: float):
        return int(score >= threshold) if binary else round(float(score), 6)

    if fmt == "dict":
        payload = {path: _value(score) for path, score in rows}
    else:
        payload = [{"image_path": path, "pred": _value(score)} for path, score in rows]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_stub(paths: list[Path], seed: int) -> tuple[list[float], list[str]]:
    """Random scores, but the SAME skip behaviour as real inference.

    Stub mode must produce a row count identical to the real model's, or it
    is not actually validating the output contract -- which is its only job.
    So it still opens every file and drops the ones that fail.
    """
    import random

    from PIL import Image

    rng = random.Random(seed)
    scores: list[float] = []
    failed: list[str] = []
    ok_paths: list[Path] = []
    for path in paths:
        try:
            with Image.open(path) as img:
                img.verify()
        except Exception as exc:  # noqa: BLE001
            failed.append(f"{path}: {exc}")
            continue
        ok_paths.append(path)
        scores.append(rng.random())
    paths[:] = ok_paths
    return scores, failed


def run_model(
    paths: list[Path],
    checkpoint: Path,
    batch_size: int,
    device_arg: str,
) -> tuple[list[float], list[str]]:
    """Real inference: frozen backbone + forensic features + gated head.

    Thin wrapper over `src.inference.Scorer` -- the shared, single-source
    forward pass. This function only adds the file-level concern the CLI
    needs: open every path, skip and report the ones that fail, keep the
    scores aligned with the surviving paths.
    """
    from PIL import Image

    from src.inference import Scorer

    if not checkpoint.exists():
        raise SystemExit(
            f"error: checkpoint not found: {checkpoint}\n"
            "       train one with `python -m src.train`, or pass --stub to "
            "verify the output format only."
        )

    try:
        scorer = Scorer(checkpoint, device=device_arg)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}")

    counts = scorer.params
    print(f"parameters: {counts['total']:,} total "
          f"({counts['trainable']:,} trainable + {counts['frozen_backbone']:,} frozen) "
          f"| 2B cap: {'OK' if counts['compliant'] else 'VIOLATION'} "
          f"({counts['headroom_x']}x headroom)")

    images: list[Image.Image] = []
    failed: list[str] = []
    ok_paths: list[Path] = []
    for path in paths:
        try:
            images.append(Image.open(path).convert("RGB"))
        except Exception as exc:  # noqa: BLE001
            failed.append(f"{path}: {exc}")
            continue
        ok_paths.append(path)

    scores = scorer.score_many(images, batch_size=batch_size)

    # Re-align: only successfully read images produced scores.
    paths[:] = ok_paths
    return scores, failed


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Score images for AI-generated likelihood.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--image-dir", required=True, type=Path,
                    help="directory of images (searched recursively)")
    ap.add_argument("--out", required=True, type=Path,
                    help="path to write the JSON results")
    # Matches the `--tag full` run in the README's reproduction steps; train.py
    # names checkpoints after their tag, and nothing ever wrote a "best.pt".
    ap.add_argument("--checkpoint", type=Path, default=Path("checkpoints/full.pt"))
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--format", default="list", choices=["list", "dict"],
                    help="JSON shape: list of {image_path, pred} or {path: pred}")
    ap.add_argument("--binary", action="store_true",
                    help="emit 0/1 labels instead of probabilities")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--relative", action="store_true",
                    help="write paths relative to --image-dir")
    ap.add_argument("--stub", action="store_true",
                    help="random scores, no model; format check only")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    paths = find_images(args.image_dir)
    if not paths:
        print(f"warning: no images found under {args.image_dir}", file=sys.stderr)
        write_output([], args.out, args.format, args.binary, args.threshold)
        return 0

    print(f"found {len(paths)} images under {args.image_dir}")
    t0 = time.time()

    failed: list[str] = []
    if args.stub:
        print("STUB MODE -- random scores. Do not submit results from this.")
        scores, failed = run_stub(paths, args.seed)
    else:
        scores, failed = run_model(paths, args.checkpoint, args.batch_size, args.device)

    def render(p: Path) -> str:
        if args.relative:
            try:
                return str(p.relative_to(args.image_dir))
            except ValueError:
                return str(p)
        return str(p)

    rows = [(render(p), s) for p, s in zip(paths, scores)]
    write_output(rows, args.out, args.format, args.binary, args.threshold)

    elapsed = time.time() - t0
    rate = len(rows) / elapsed if elapsed > 0 else float("inf")
    print(f"wrote {len(rows)} predictions to {args.out}")
    print(f"{elapsed:.1f}s ({rate:.1f} img/s)")
    if failed:
        print(f"skipped {len(failed)} unreadable files:", file=sys.stderr)
        for line in failed[:10]:
            print(f"  {line}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
