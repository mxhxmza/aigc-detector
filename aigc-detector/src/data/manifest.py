"""The dataset manifest -- the single source of truth for everything
downstream (feature extraction, training, evaluation, error analysis).

``data/manifest.csv`` is written by ``scripts/build_dataset.py`` and has four
columns:

    image_path   path to the image on disk
    label        0 = real (a genuine or lightly-edited photograph),
                 1 = AI-generated (fully synthetic)
    kind         real | tampered | synthetic | cifake_real | cifake_fake --
                 finer-grained provenance, used only for error analysis;
                 ``tampered`` still has label 0
    split        train | test

Keeping this decoupled from the on-disk layout is deliberate: the rest of the
code never has to know how the raw dataset was organised, and the train/test
boundary is a column here plus a physical folder split, not a convention
someone has to remember.
"""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

COLUMNS = ("image_path", "label", "kind", "split")


@dataclass
class Record:
    image_path: str
    label: int
    kind: str
    split: str


def read(path: str | Path) -> list[Record]:
    with Path(path).open(newline="", encoding="utf-8") as fh:
        return [
            Record(row["image_path"], int(row["label"]), row["kind"], row["split"])
            for row in csv.DictReader(fh)
        ]


def write(records: Iterable[Record], out_path: str | Path) -> int:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for r in records:
            w.writerow({"image_path": r.image_path, "label": r.label,
                        "kind": r.kind, "split": r.split})
            n += 1
    return n


def summarise(records: list[Record]) -> str:
    lab = Counter(r.label for r in records)
    kind = Counter(r.kind for r in records)
    split = Counter(r.split for r in records)
    lines = [
        f"total images: {len(records)}",
        f"  real (0): {lab.get(0, 0)}   ai (1): {lab.get(1, 0)}",
        "  kinds: " + ", ".join(f"{k}={v}" for k, v in sorted(kind.items())),
        "  splits: " + ", ".join(f"{k}={v}" for k, v in sorted(split.items())),
    ]
    return "\n".join(lines)
