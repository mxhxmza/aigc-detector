"""Generate a tiny synthetic dataset so the pipeline can be run end-to-end.

This is NOT training data. It exists so that `build_dataset -> extract ->
train -> evaluate -> predict` can be exercised in a couple of minutes on a
laptop, without downloading SID_Set first. The "real" class carries 1/f
spectral falloff plus per-pixel sensor noise; the "fake" class is generated
at low resolution and bicubically upsampled, which plants the periodic
spectral artifact the forensic branch is designed to find.

Any real result must come from the real dataset. Use this only to verify plumbing.

Usage:
    python scripts/make_smoke_data.py --out data/smoke_raw
    python scripts/build_dataset.py --raw data/smoke_raw --out data/smoke \
        --manifest data/smoke/manifest.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

GENERATORS = ("smoke_gan", "smoke_diffusion", "smoke_vae")


def _pink_noise(rng: np.random.Generator, size: int, beta: float) -> np.ndarray:
    """2D noise with a 1/f**beta radial power spectrum, normalised to [0,1]."""
    white = rng.normal(size=(size, size))
    fy = np.fft.fftfreq(size)[:, None]
    fx = np.fft.fftfreq(size)[None, :]
    radius = np.sqrt(fx**2 + fy**2)
    radius[0, 0] = 1.0
    shaped = np.fft.ifft2(np.fft.fft2(white) / radius**beta).real
    shaped -= shaped.min()
    return shaped / (shaped.max() + 1e-8)


def make_real(rng: np.random.Generator, size: int = 256) -> Image.Image:
    """Natural-ish: 1/f texture in each channel plus additive sensor noise."""
    channels = [_pink_noise(rng, size, beta=1.0) for _ in range(3)]
    arr = np.stack(channels, axis=-1) * 255.0
    arr += rng.normal(0.0, 4.0, arr.shape)          # sensor noise floor
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def make_fake(rng: np.random.Generator, size: int = 256, factor: int = 4) -> Image.Image:
    """Synthetic-ish: built small and upsampled, so it has no noise floor and
    carries the periodic replication artifact that upsampling leaves behind."""
    small = size // factor
    channels = [_pink_noise(rng, small, beta=1.4) for _ in range(3)]
    arr = np.stack(channels, axis=-1) * 255.0
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return img.resize((size, size), Image.BICUBIC)   # no noise added on purpose


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("data/smoke"))
    ap.add_argument("--n-real", type=int, default=120)
    ap.add_argument("--n-fake", type=int, default=120)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    real_dir = args.out / "real"
    real_dir.mkdir(parents=True, exist_ok=True)
    for i in range(args.n_real):
        make_real(rng, args.size).save(real_dir / f"real_{i:05d}.png")

    # Spread the fakes over several generator directories so the
    # generator-disjoint splitter has something to hold out.
    per_gen = args.n_fake // len(GENERATORS)
    for g_i, gen in enumerate(GENERATORS):
        gen_dir = args.out / "fake" / gen
        gen_dir.mkdir(parents=True, exist_ok=True)
        # Vary the upsampling factor per generator: different generators leave
        # different periodicities, which is what the holdout row tests.
        factor = (2, 4, 8)[g_i % 3]
        for i in range(per_gen):
            make_fake(rng, args.size, factor).save(gen_dir / f"{gen}_{i:05d}.png")

    print(f"wrote {args.n_real} real + {per_gen * len(GENERATORS)} fake "
          f"images under {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
