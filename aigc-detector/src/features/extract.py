"""One-time feature extraction -- the decision that makes this buildable
solo on 8GB.

For every image we compute K views (view 0 is always clean, views 1..K-1 are
randomly sampled degradations) and store, per view:

    semantic  frozen CLIP image embedding      float16
    forensic  hand-designed frequency features float16
    degrade   degradation descriptor           float16   <- free supervision
    label / generator / split / image_id

After this runs, training never touches a JPEG or a ViT again. An epoch
becomes a pass over a few hundred MB of vectors, so experiments take seconds
and the ablation table is actually achievable in a day.

Cost model (RTX 5060, ViT-L/14 @224 fp16):
    GPU  ~100-150 img/s   -> 240k views in roughly 30-40 min
    CPU  ~19 ms/img for forensic features, so it MUST be parallelised or it
         becomes the bottleneck at ~76 min single-threaded. We overlap it
         with the GPU work using a process pool.

Usage:
    python -m src.features.extract \
        --manifest data/manifest.csv \
        --out features/ \
        --views 4 --backbone ViT-L-14 --batch-size 64
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from ..data import manifest as M
from ..transforms.degradations import CLEAN, DegradationSampler
from . import frequency as FQ

# Backbones known to be public (C-2) and comfortably under the cap (C-1).
#
# The `-quickgelu` suffix is load-bearing, not cosmetic. OpenAI's CLIP weights
# were trained with the QuickGELU activation; open_clip's plain "ViT-L-14"
# config uses standard GELU. Loading the openai tag into the plain config
# succeeds, warns once, and then produces subtly wrong embeddings for every
# image. Key = the name used on the CLI and stored in checkpoints; value[0] =
# the open_clip config actually instantiated.
BACKBONES = {
    "ViT-B-16": ("ViT-B-16-quickgelu", "openai", 512),
    "ViT-L-14": ("ViT-L-14-quickgelu", "openai", 768),
}


def _forensic_worker(payload: tuple[bytes, int, int]) -> np.ndarray:
    """Runs in a separate process; takes raw pixels to avoid PIL pickling."""
    raw, w, h = payload
    img = Image.frombytes("RGB", (w, h), raw)
    return FQ.extract(img)


def _load_image(path: str) -> Image.Image | None:
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def build_views(
    img: Image.Image,
    n_views: int,
    sampler: DegradationSampler,
) -> tuple[list[Image.Image], list[np.ndarray], list[str]]:
    """View 0 is clean; the rest are sampled degradations."""
    images = [img]
    descriptors = [CLEAN.describe()]
    names = ["clean"]
    for _ in range(n_views - 1):
        aug, desc, name = sampler.apply(img)
        images.append(aug)
        descriptors.append(desc)
        names.append(name)
    return images, descriptors, names


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--views", type=int, default=4)
    ap.add_argument("--backbone", default="ViT-L-14", choices=list(BACKBONES))
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4,
                    help="processes for CPU forensic features")
    ap.add_argument("--limit", type=int, default=0,
                    help="only process N images (for a timing dry run)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    import open_clip

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no GPU visible. Extraction will be very slow.")

    model_name, pretrained, embed_dim = BACKBONES[args.backbone]
    print(f"loading {model_name} ({pretrained}) on {device} ...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    model = model.visual.to(device).eval()
    if device == "cuda":
        model = model.half()

    backbone_params = sum(p.numel() for p in model.parameters())
    print(f"frozen backbone parameters: {backbone_params:,} "
          f"({backbone_params / 2e9:.4%} of the 2B cap)")

    records = M.read(args.manifest)
    if args.limit:
        records = records[: args.limit]
    print(f"{len(records)} images x {args.views} views "
          f"= {len(records) * args.views} forward passes")

    args.out.mkdir(parents=True, exist_ok=True)
    sampler = DegradationSampler(seed=args.seed)

    sem_all: list[np.ndarray] = []
    for_all: list[np.ndarray] = []
    deg_all: list[np.ndarray] = []
    meta: list[dict] = []

    # ThreadPoolExecutor, not ProcessPoolExecutor: the forensic worker is scipy
    # FFT/DCT which releases the GIL, so threads parallelise it just as well,
    # and the process pool has deadlocked on Windows (spawn + large pickled
    # payloads) enough times to not be worth it.
    pool = ThreadPoolExecutor(max_workers=args.workers) if args.workers > 1 else None
    batch_tensors: list[torch.Tensor] = []
    pending_forensic = []
    n_skipped = 0

    def flush() -> None:
        """Run the GPU batch and collect the CPU results."""
        if not batch_tensors:
            return
        with torch.no_grad():
            x = torch.stack(batch_tensors).to(device)
            if device == "cuda":
                x = x.half()
            emb = model(x)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        sem_all.append(emb.float().cpu().numpy().astype(np.float16))

        if pool is not None:
            for_all.append(np.stack([f.result() for f in pending_forensic]).astype(np.float16))
        else:
            for_all.append(np.stack(pending_forensic).astype(np.float16))

        batch_tensors.clear()
        pending_forensic.clear()

    from time import time
    t0 = time()

    for i, rec in enumerate(records):
        img = _load_image(rec.image_path)
        if img is None:
            n_skipped += 1
            continue

        views, descriptors, names = build_views(img, args.views, sampler)
        for v, (view, desc, name) in enumerate(zip(views, descriptors, names)):
            batch_tensors.append(preprocess(view))
            payload = (view.tobytes(), view.size[0], view.size[1])
            pending_forensic.append(
                pool.submit(_forensic_worker, payload) if pool
                else _forensic_worker(payload)
            )
            deg_all.append(desc.astype(np.float16))
            meta.append({
                "image_id": i,
                "view": v,
                "view_name": name,
                "image_path": rec.image_path,
                "label": rec.label,
                "kind": rec.kind,
                "split": rec.split,
            })
            if len(batch_tensors) >= args.batch_size:
                flush()

        if i and i % 500 == 0:
            rate = (i + 1) * args.views / (time() - t0)
            eta = (len(records) - i) * args.views / max(rate, 1e-6) / 60
            print(f"  {i}/{len(records)} images | {rate:.0f} views/s | ETA {eta:.1f} min")

    flush()
    if pool is not None:
        pool.shutdown()

    semantic = np.concatenate(sem_all) if sem_all else np.empty((0, embed_dim), np.float16)
    forensic = np.concatenate(for_all) if for_all else np.empty((0, FQ.FEATURE_DIM), np.float16)
    degrade = np.stack(deg_all) if deg_all else np.empty((0, 7), np.float16)

    out_file = args.out / "features.npz"
    np.savez_compressed(
        out_file,
        semantic=semantic,
        forensic=forensic,
        degrade=degrade,
        image_id=np.array([m["image_id"] for m in meta], np.int32),
        view=np.array([m["view"] for m in meta], np.int16),
        label=np.array([m["label"] for m in meta], np.int8),
    )
    (args.out / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (args.out / "extract_info.json").write_text(json.dumps({
        "backbone": args.backbone,
        "backbone_params": backbone_params,
        "semantic_dim": int(semantic.shape[1]) if len(semantic) else embed_dim,
        "forensic_dim": FQ.FEATURE_DIM,
        "views": args.views,
        "n_images": len(records) - n_skipped,
        "n_rows": len(meta),
        "skipped": n_skipped,
        "seed": args.seed,
    }, indent=2), encoding="utf-8")

    mins = (time() - t0) / 60
    size_mb = out_file.stat().st_size / 1e6
    print(f"\nwrote {len(meta)} rows to {out_file} ({size_mb:.0f} MB) in {mins:.1f} min")
    if n_skipped:
        print(f"skipped {n_skipped} unreadable images")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
