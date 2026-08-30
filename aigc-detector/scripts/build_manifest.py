"""Build the manifest from whatever datasets you managed to download.

Deliberately tolerant: any dataset root you pass is scanned, and missing
ones are skipped with a warning. On day 1 you will not have all three, and
the pipeline should not block on that.

The provided validation subset (COCO val2017 + DALL-E Advanced) is recorded
with split="provided_val" and is never reassigned by the splitter, so it can
never end up in training. The leakage check then verifies that claim rather
than trusting it.

Usage:
    python scripts/build_manifest.py \
        --wildfake data/wildfake --sid-set data/sid_set --cifake data/cifake \
        --provided-real data/val/coco_val2017 \
        --provided-aigc data/val/dalle_advanced \
        --out data/manifest.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _add_repo_root_to_path() -> Path:
    """Find the directory containing `src/` and put it on sys.path.

    Walks upward from this file rather than assuming a fixed depth, so the
    script works whether it lives in scripts/ or at the repo root, and
    whether it is invoked from the repo root or from anywhere else.
    """
    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents):
        if (candidate / "src" / "__init__.py").is_file():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return candidate
    raise SystemExit(
        f"error: could not locate the `src` package starting from {here}.\n"
        "       Run this from inside the project directory, and check that\n"
        "       src/__init__.py exists."
    )


REPO_ROOT = _add_repo_root_to_path()

from src.data import manifest as M           # noqa: E402
from src.data.splits import assign_generator_disjoint, check_leakage  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wildfake", type=Path)
    ap.add_argument("--sid-set", type=Path)
    ap.add_argument("--cifake", type=Path)
    ap.add_argument("--provided-real", type=Path,
                    help="COCO val2017 subset (all real, never trained on)")
    ap.add_argument("--provided-aigc", type=Path,
                    help="DALL-E Advanced subset (all AIGC, never trained on)")
    ap.add_argument("--out", type=Path, default=Path("data/manifest.csv"))
    ap.add_argument("--holdout-generators", nargs="*", default=[],
                    help="generators to hold out entirely; auto-picked if omitted")
    ap.add_argument("--no-holdout", action="store_true",
                    help="train on every generator (no cross-generator holdout). "
                         "Use when the goal is a model that works across all the "
                         "datasets passed, not measuring transfer to an unseen one.")
    ap.add_argument("--val-fraction", type=float, default=0.12)
    ap.add_argument("--sample-per-class", type=int, default=0,
                    help="randomly keep at most N real and N AIGC images "
                         "(0 = keep everything). CIFAKE is 120k images; a "
                         "subset gets a real number out in minutes, and the "
                         "full set is one re-run with this flag removed.")
    ap.add_argument("--skip-leakage-check", action="store_true",
                    help="NOT recommended -- C-4 is a disqualification rule")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    records: list[M.Record] = []
    for root, name in ((args.wildfake, "wildfake"),
                       (args.sid_set, "sid_set"),
                       (args.cifake, "cifake")):
        if root is None:
            continue
        if not root.exists():
            print(f"warning: {name} root not found, skipping: {root}")
            continue
        found = M.from_labelled_dirs(root, name)
        print(f"{name}: {len(found)} images")
        records += found

    provided: list[M.Record] = []
    if args.provided_real and args.provided_real.exists():
        provided += M.from_flat_dir(args.provided_real, 0, "real", "provided_val")
    if args.provided_aigc and args.provided_aigc.exists():
        provided += M.from_flat_dir(args.provided_aigc, 1, "dalle_advanced",
                                    "provided_val")
    if provided:
        print(f"provided validation subset: {len(provided)} images "
              "(split=provided_val, never trained on)")

    if not records:
        raise SystemExit(
            "no training images found. Pass at least one dataset root.\n"
            "Expected layout: <root>/**/{real,REAL,nature}/... and "
            "<root>/**/{fake,FAKE,ai}/<generator>/..."
        )

    if args.sample_per_class:
        import random
        from collections import defaultdict
        rng = random.Random(args.seed)
        # Sample per (source, label) so combining datasets keeps each one's
        # contribution balanced -- a global cap would let the largest dataset
        # (CIFAKE at 120k) crowd the others out entirely.
        groups: dict[tuple[str, int], list] = defaultdict(list)
        for r in records:
            groups[(r.source, r.label)].append(r)
        kept: list[M.Record] = []
        for key, recs in sorted(groups.items()):
            rng.shuffle(recs)
            kept += recs[: args.sample_per_class]
            print(f"  {key[0]} label={key[1]}: kept "
                  f"{min(len(recs), args.sample_per_class)} of {len(recs)}")
        records = kept
        print(f"subsampled to {len(records)} images")

    assign_generator_disjoint(
        records,
        holdout_generators=tuple(args.holdout_generators),
        val_fraction=args.val_fraction,
        seed=args.seed,
        no_holdout=args.no_holdout,
    )
    all_records = records + provided

    if provided and not args.skip_leakage_check:
        train_recs = [r for r in records if r.split == "train"]
        report = check_leakage(train_recs, provided)
        if not report["clean"]:
            raise SystemExit(
                "\nABORTED: provided validation images found in the training "
                "split.\nThis violates C-4 and would mean disqualification. "
                "Remove the duplicates and rebuild."
            )

    n = M.write(all_records, args.out)
    print(f"\nwrote {n} rows to {args.out}\n")
    print(M.summarise(all_records))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
