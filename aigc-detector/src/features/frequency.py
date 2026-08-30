"""Hand-designed forensic features (the low-level half of the hybrid).

These target the signal families named in the workshop:

  * Frequency artifacts   -- GAN/diffusion upsampling leaves periodic
                             structure in the Fourier spectrum that lenses
                             and sensors do not produce.
  * Noise fingerprints    -- real photos carry sensor noise (PRNU-like);
                             synthetic images lack it or fake it badly.
  * Texture / fine detail  -- captured in high-pass residual statistics.

Why fixed features rather than a small trainable CNN
----------------------------------------------------
1. They cache. Like the frozen backbone embeddings, these are computed once
   per (image, view) and never recomputed, which is what makes solo
   iteration affordable on an 8GB card.
2. No JIT. Nothing here compiles a CUDA kernel at runtime, which sidesteps
   the known PTX/JIT problems on sm_120.
3. They are interpretable. "The radial spectrum slope shifts under JPEG"
   is a sentence you can say on video with a chart behind it. A learned
   CNN's first layer is not.

Everything runs on CPU with numpy/scipy, in parallel with GPU extraction.

Feature layout (FEATURE_DIM total):
    radial FFT log-power profile      N_RADIAL_BINS
    radial profile slope + curvature  2
    per-position DCT log-magnitude    64
    high-pass residual moments        4 stats x 3 channels = 12
    cross-channel residual corr       3
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from scipy.fftpack import dct
from scipy.ndimage import gaussian_filter
from scipy.stats import kurtosis, skew

N_RADIAL_BINS = 48
ANALYSIS_SIZE = 256          # images are centre-resized to this before analysis
DCT_BLOCK = 8

FEATURE_NAMES: list[str] = (
    [f"radial_{i}" for i in range(N_RADIAL_BINS)]
    + ["radial_slope", "radial_curvature"]
    + [f"dct_{i}" for i in range(DCT_BLOCK * DCT_BLOCK)]
    + [f"resid_{s}_{c}" for c in "rgb" for s in ("std", "mad", "kurt", "skew")]
    + ["resid_corr_rg", "resid_corr_rb", "resid_corr_gb"]
)
FEATURE_DIM = len(FEATURE_NAMES)


def _to_analysis_array(img: Image.Image) -> np.ndarray:
    """Resize to a fixed square and return float32 RGB in [0,1].

    Fixed size matters: the radial spectrum is only comparable across images
    if they share a resolution, otherwise "frequency" means different things
    per image and the model learns image size instead of image origin.
    """
    if img.mode != "RGB":
        img = img.convert("RGB")
    if img.size != (ANALYSIS_SIZE, ANALYSIS_SIZE):
        img = img.resize((ANALYSIS_SIZE, ANALYSIS_SIZE), Image.BICUBIC)
    return np.asarray(img, dtype=np.float32) / 255.0


def radial_power_spectrum(gray: np.ndarray, n_bins: int = N_RADIAL_BINS) -> np.ndarray:
    """Radially averaged log power spectrum of a 2D array.

    Upsampling in generative models tends to leave periodic peaks; averaging
    over angle collapses the 2D spectrum into a compact profile that still
    shows those bumps as deviations from the natural 1/f falloff.
    """
    win = np.outer(np.hanning(gray.shape[0]), np.hanning(gray.shape[1]))
    spec = np.fft.fftshift(np.abs(np.fft.fft2(gray * win)) ** 2)

    h, w = spec.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    r_max = min(cy, cx)

    edges = np.linspace(0, r_max, n_bins + 1)
    idx = np.digitize(r.ravel(), edges) - 1
    flat = spec.ravel()

    profile = np.zeros(n_bins, dtype=np.float32)
    counts = np.bincount(np.clip(idx, 0, n_bins - 1), minlength=n_bins)
    sums = np.bincount(np.clip(idx, 0, n_bins - 1), weights=flat, minlength=n_bins)
    valid = counts > 0
    profile[valid] = (sums[valid] / counts[valid]).astype(np.float32)
    return np.log1p(profile)


def dct_signature(gray: np.ndarray, block: int = DCT_BLOCK) -> np.ndarray:
    """Mean log-magnitude per DCT coefficient position over 8x8 blocks.

    This is close to what JPEG itself quantises, which cuts both ways: it is
    sensitive to generative artifacts AND to compression history. That dual
    sensitivity is precisely why the degradation head exists -- the model
    needs to know how much compression happened before it can read this
    feature correctly.
    """
    h, w = gray.shape
    h, w = h - h % block, w - w % block
    g = gray[:h, :w] * 255.0 - 128.0

    blocks = (g.reshape(h // block, block, w // block, block)
               .transpose(0, 2, 1, 3)
               .reshape(-1, block, block))
    coeffs = dct(dct(blocks, axis=1, norm="ortho"), axis=2, norm="ortho")
    return np.log1p(np.abs(coeffs)).mean(axis=0).ravel().astype(np.float32)


def residual_stats(rgb: np.ndarray) -> np.ndarray:
    """Statistics of the high-pass residual, per channel plus cross-channel.

    residual = image - gaussian_blur(image). Real sensor noise is close to
    independent across channels after demosaicing artifacts are accounted
    for; generated images often show unnaturally correlated or unnaturally
    absent residuals.
    """
    stats: list[float] = []
    residuals = []
    for c in range(3):
        chan = rgb[..., c]
        resid = chan - gaussian_filter(chan, sigma=1.0)
        residuals.append(resid.ravel())
        flat = resid.ravel()
        stats.extend([
            float(np.std(flat)),
            float(np.mean(np.abs(flat))),
            float(kurtosis(flat, fisher=True, bias=False)),
            float(skew(flat, bias=False)),
        ])

    for a, b in ((0, 1), (0, 2), (1, 2)):
        ra, rb = residuals[a], residuals[b]
        denom = (np.std(ra) * np.std(rb)) + 1e-8
        stats.append(float(np.mean((ra - ra.mean()) * (rb - rb.mean())) / denom))

    return np.asarray(stats, dtype=np.float32)


def extract(img: Image.Image) -> np.ndarray:
    """Full forensic feature vector for one image. Shape: (FEATURE_DIM,)."""
    rgb = _to_analysis_array(img)
    gray = rgb.mean(axis=2)

    radial = radial_power_spectrum(gray)

    # Slope and curvature of the log-log spectral falloff. Natural images sit
    # near a 1/f power law; deviations are informative and, unlike the raw
    # profile, these two numbers are fairly robust to overall scaling.
    freqs = np.log1p(np.arange(1, N_RADIAL_BINS + 1, dtype=np.float32))
    slope, curv = np.polyfit(freqs, radial, deg=2)[:2][::-1]

    feats = np.concatenate([
        radial,
        np.asarray([slope, curv], dtype=np.float32),
        dct_signature(gray),
        residual_stats(rgb),
    ]).astype(np.float32)

    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


def extract_batch(images: list[Image.Image]) -> np.ndarray:
    return np.stack([extract(im) for im in images]) if images else np.empty((0, FEATURE_DIM), np.float32)
