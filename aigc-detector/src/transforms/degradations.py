"""The six real-world transformations specified in the problem statement.

Parameter values are taken verbatim from the track brief and must not be
changed -- evaluation is run against this grid.

    JPEG Compression   quality = 90, 70, 50, 30      social re-encode
    Gaussian Blur      sigma   = 0.5, 1.0, 2.0       out-of-focus
    Resize             scale   = 0.5, 0.25 (+upscale) thumbnail / CDN
    Gaussian Noise     sigma   = 0.02, 0.05, 0.10    low-light sensor
    Color Jitter       b/c/s   = +/- 20%             filter apps
    Center Crop        80%                           profile framing

Every function takes and returns a PIL RGB Image, and also returns the
parameters that were applied. Those parameters are the free supervision for
the degradation-estimation head (FR-13): we generated the damage, so we know
exactly how much there is.

Design notes
------------
* Noise sigma is expressed in [0,1] intensity units, so it is scaled by 255
  before being applied to uint8 pixels.
* `resize_down_up` returns the image at its ORIGINAL size -- the information
  loss is the point, not the size change.
* `center_crop` returns a smaller image. Downstream resizing to the model
  input size happens in preprocessing, which mirrors what a real pipeline
  does to a cropped profile picture.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

# --------------------------------------------------------------------------
# Canonical parameter grid. Do not edit without re-reading the brief.
# --------------------------------------------------------------------------

JPEG_QUALITIES = (90, 70, 50, 30)
BLUR_SIGMAS = (0.5, 1.0, 2.0)
RESIZE_SCALES = (0.5, 0.25)
NOISE_SIGMAS = (0.02, 0.05, 0.10)
JITTER_STRENGTH = 0.20          # +/- 20% on brightness, contrast, saturation
CROP_FRACTION = 0.80

# Ordering of the degradation descriptor vector produced by `describe`.
# Index 0 is a clean flag; the rest are severity in [0, 1].
DEGRADATION_KEYS = (
    "clean",
    "jpeg",
    "blur",
    "resize",
    "noise",
    "jitter",
    "crop",
)
DEGRADATION_DIM = len(DEGRADATION_KEYS)


# --------------------------------------------------------------------------
# Individual transforms
# --------------------------------------------------------------------------

def jpeg_compress(img: Image.Image, quality: int) -> Image.Image:
    """Re-encode through JPEG at the given quality, in memory."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def gaussian_blur(img: Image.Image, sigma: float) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=float(sigma)))


def resize_down_up(img: Image.Image, scale: float) -> Image.Image:
    """Downscale then upscale back to the original size (thumbnail round-trip)."""
    w, h = img.size
    small = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    return img.resize(small, Image.BICUBIC).resize((w, h), Image.BICUBIC)


def gaussian_noise(img: Image.Image, sigma: float, rng: np.random.Generator) -> Image.Image:
    """Additive Gaussian noise; sigma is in [0,1] intensity units."""
    arr = np.asarray(img, dtype=np.float32)
    noise = rng.normal(0.0, float(sigma) * 255.0, arr.shape).astype(np.float32)
    return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))


def color_jitter(
    img: Image.Image,
    brightness: float,
    contrast: float,
    saturation: float,
) -> Image.Image:
    """Multiplicative jitter. Each factor is around 1.0 (e.g. 0.8 - 1.2)."""
    out = ImageEnhance.Brightness(img).enhance(float(brightness))
    out = ImageEnhance.Contrast(out).enhance(float(contrast))
    out = ImageEnhance.Color(out).enhance(float(saturation))
    return out


def center_crop(img: Image.Image, fraction: float) -> Image.Image:
    w, h = img.size
    cw, ch = int(round(w * fraction)), int(round(h * fraction))
    left, top = (w - cw) // 2, (h - ch) // 2
    return img.crop((left, top, left + cw, top + ch))


# --------------------------------------------------------------------------
# Named, parameterised application
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Degradation:
    """A single named transform at a single parameter setting.

    `severity` is a normalised [0,1] measure used both for the degradation
    head's regression target and for ordering results in the robustness
    table. It is a design choice, not something the brief specifies.
    """

    kind: str
    params: dict
    severity: float

    @property
    def name(self) -> str:
        if self.kind == "clean":
            return "clean"
        bits = "_".join(f"{k}{v}" for k, v in self.params.items())
        return f"{self.kind}_{bits}"

    def apply(self, img: Image.Image, rng: np.random.Generator | None = None) -> Image.Image:
        rng = rng or np.random.default_rng(0)
        if self.kind == "clean":
            return img
        if self.kind == "jpeg":
            return jpeg_compress(img, self.params["q"])
        if self.kind == "blur":
            return gaussian_blur(img, self.params["sigma"])
        if self.kind == "resize":
            return resize_down_up(img, self.params["scale"])
        if self.kind == "noise":
            return gaussian_noise(img, self.params["sigma"], rng)
        if self.kind == "jitter":
            return color_jitter(
                img,
                self.params["brightness"],
                self.params["contrast"],
                self.params["saturation"],
            )
        if self.kind == "crop":
            return center_crop(img, self.params["fraction"])
        raise ValueError(f"unknown degradation kind: {self.kind}")

    def describe(self) -> np.ndarray:
        """Descriptor vector for the degradation-estimation head."""
        vec = np.zeros(DEGRADATION_DIM, dtype=np.float32)
        if self.kind == "clean":
            vec[0] = 1.0
        else:
            vec[DEGRADATION_KEYS.index(self.kind)] = float(self.severity)
        return vec


CLEAN = Degradation("clean", {}, 0.0)


def evaluation_grid() -> list[Degradation]:
    """Every (transform, parameter) cell the robustness table reports.

    Severity is normalised per-family so 1.0 is always the harshest setting
    in that family.
    """
    grid: list[Degradation] = [CLEAN]

    for q in JPEG_QUALITIES:
        # quality 90 -> mild, 30 -> harsh
        sev = (100 - q) / (100 - min(JPEG_QUALITIES))
        grid.append(Degradation("jpeg", {"q": q}, round(sev, 4)))

    for s in BLUR_SIGMAS:
        grid.append(Degradation("blur", {"sigma": s}, round(s / max(BLUR_SIGMAS), 4)))

    for sc in RESIZE_SCALES:
        # 0.25 is harsher than 0.5
        sev = (1.0 - sc) / (1.0 - min(RESIZE_SCALES))
        grid.append(Degradation("resize", {"scale": sc}, round(sev, 4)))

    for s in NOISE_SIGMAS:
        grid.append(Degradation("noise", {"sigma": s}, round(s / max(NOISE_SIGMAS), 4)))

    # Jitter has no natural severity ladder; the brief gives one magnitude.
    # Use the two extreme corners so the table shows both directions.
    grid.append(Degradation(
        "jitter",
        {"brightness": 1 - JITTER_STRENGTH, "contrast": 1 - JITTER_STRENGTH,
         "saturation": 1 - JITTER_STRENGTH},
        1.0,
    ))
    grid.append(Degradation(
        "jitter",
        {"brightness": 1 + JITTER_STRENGTH, "contrast": 1 + JITTER_STRENGTH,
         "saturation": 1 + JITTER_STRENGTH},
        1.0,
    ))

    grid.append(Degradation("crop", {"fraction": CROP_FRACTION}, 1.0))
    return grid


# --------------------------------------------------------------------------
# Random sampling for training (FR-4: "apply these randomly during training")
# --------------------------------------------------------------------------

@dataclass
class DegradationSampler:
    """Samples a random degradation, optionally chaining two of them.

    Chaining matters: a real reposted image has usually been resized AND
    re-encoded, not just one or the other. The brief only requires single
    transforms at evaluation, but training on compositions is free
    robustness and is explicitly permitted ("you may generate transformed
    samples from approved datasets").
    """

    clean_prob: float = 0.25
    chain_prob: float = 0.30
    seed: int = 0
    _rng: np.random.Generator = field(init=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def _sample_one(self) -> Degradation:
        kind = self._rng.choice(["jpeg", "blur", "resize", "noise", "jitter", "crop"])
        if kind == "jpeg":
            q = int(self._rng.integers(30, 96))
            sev = (100 - q) / 70.0
            return Degradation("jpeg", {"q": q}, float(np.clip(sev, 0, 1)))
        if kind == "blur":
            sigma = float(self._rng.uniform(0.3, 2.0))
            return Degradation("blur", {"sigma": round(sigma, 3)}, sigma / 2.0)
        if kind == "resize":
            scale = float(self._rng.uniform(0.25, 0.75))
            return Degradation("resize", {"scale": round(scale, 3)},
                               float(np.clip((1 - scale) / 0.75, 0, 1)))
        if kind == "noise":
            sigma = float(self._rng.uniform(0.01, 0.10))
            return Degradation("noise", {"sigma": round(sigma, 4)}, sigma / 0.10)
        if kind == "jitter":
            b, c, s = (float(self._rng.uniform(1 - JITTER_STRENGTH, 1 + JITTER_STRENGTH))
                       for _ in range(3))
            mag = max(abs(b - 1), abs(c - 1), abs(s - 1)) / JITTER_STRENGTH
            return Degradation("jitter",
                               {"brightness": round(b, 3), "contrast": round(c, 3),
                                "saturation": round(s, 3)}, mag)
        frac = float(self._rng.uniform(0.7, 0.95))
        return Degradation("crop", {"fraction": round(frac, 3)},
                           float(np.clip((1 - frac) / 0.30, 0, 1)))

    def sample(self) -> list[Degradation]:
        if self._rng.random() < self.clean_prob:
            return [CLEAN]
        chain = [self._sample_one()]
        if self._rng.random() < self.chain_prob:
            chain.append(self._sample_one())
        return chain

    def apply(self, img: Image.Image) -> tuple[Image.Image, np.ndarray, str]:
        """Returns (degraded image, descriptor vector, human-readable name)."""
        chain = self.sample()
        out = img
        for deg in chain:
            out = deg.apply(out, self._rng)
        # Descriptor of a chain is the element-wise max of its parts, so a
        # composition reads as "at least this damaged in each channel".
        vec = np.max(np.stack([d.describe() for d in chain]), axis=0)
        if len(chain) > 1:
            vec[0] = 0.0  # a chain is never clean
        return out, vec.astype(np.float32), "+".join(d.name for d in chain)


TRANSFORM_FN: dict[str, Callable] = {
    "jpeg": jpeg_compress,
    "blur": gaussian_blur,
    "resize": resize_down_up,
    "noise": gaussian_noise,
    "jitter": color_jitter,
    "crop": center_crop,
}
