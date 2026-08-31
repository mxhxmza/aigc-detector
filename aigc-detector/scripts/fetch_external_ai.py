"""Fetch DALL-E 3 and GAN images from sources that are NOT WildFake.

The WildFake benchmark shows the detector is weak on DALL-E 3 (recall ~0.72 at
0.5) and near-blind to GANs (GigaGAN recall ~0.04). SID_Set is diffusion-only
and contains neither, so the fix is training images of both -- but the
benchmark's own data is off limits (its README: "Do not train on any of
this ... the final test set comes from the same corpus"). So this pulls from
elsewhere:

    DALL-E 3   ProGamerGov/dalle-3-reddit-dataset   3,465 imgs, ~1024px, reddit
    GAN        frp94/progan_val                     ProGAN, ForenSynths layout,
                                                    400 real + 400 fake / class,
                                                    256x256

Two confounds are neutralised up front, both learned the hard way from an
earlier BigGAN attempt that regressed the benchmark:

  * Resolution. ProGAN images are 256x256; every SID image is ~512. Adding
    256px images as label 1 with nothing else lets the model separate the
    classes on size. The fix here is to take the *real* partner from the same
    ForenSynths zips -- LSUN photos, also 256x256, same pipeline -- so within
    the 256px population the classes are balanced.
  * Aspect ratio. SID synthetics are 100% square, SID reals 94% non-square, so
    "square" already predicts "AI" in the training set. DALL-E reddit images
    are square too. Each square AI image added is matched by a square real
    (an existing SID real, centre-cropped), so aspect ratio stays uninformative.

Every fetched image is perceptual-hash checked against all four WildFake eval
configs and dropped on a match, so nothing from the benchmark leaks in even by
coincidence (the DALL-E sources overlap in provenance).

Output, for scripts/add_external_ai.py:

    data/external_ai/fake_dalle3/   label 1
    data/external_ai/fake_progan/   label 1
    data/external_ai/real_lsun/     label 0   (256px partner for ProGAN)
    data/external_ai/real_square/   label 0   (square partner for DALL-E)

Usage:
    python scripts/fetch_external_ai.py --per-bucket 3000
"""

from __future__ import annotations

import argparse
import io
import sys
import tarfile
import zipfile
from pathlib import Path

import numpy as np

DALLE_REPO = "ProGamerGov/dalle-3-reddit-dataset"
DALLE_TAR = "dalle3-reddit-images.tar.gz"
PROGAN_REPO = "frp94/progan_val"
IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def dhash(img, s: int = 16) -> np.ndarray:
    g = np.asarray(img.convert("L").resize((s + 1, s)), dtype=np.int16)
    return (g[:, 1:] > g[:, :-1]).flatten()


def ahash(img, s: int = 16) -> np.ndarray:
    g = np.asarray(img.convert("L").resize((s, s)), dtype=np.float32)
    return (g > g.mean()).flatten()


def load_wildfake_hashes() -> tuple[np.ndarray, np.ndarray]:
    """dHash + aHash of every image in every WildFake eval config.

    Reads the parquet shards straight from the HF hub cache with pyarrow
    `iter_batches` -- going through `datasets.load_dataset` materialises the
    ~3 GB `default` config and OOM-kills the whole fetch, silently, on a busy
    machine. The shards are already on disk from the eval runs; if they are
    not, one `load_dataset(...)` call per config fetches them.
    """
    import io as _io
    import glob
    import pyarrow.parquet as pq
    from PIL import Image
    from huggingface_hub import snapshot_download

    root = snapshot_download("techjam-aigc/wildfake-eval-subset",
                             repo_type="dataset", allow_patterns=["*.parquet"])
    shards = sorted(glob.glob(f"{root}/**/*.parquet", recursive=True))
    dh, ah = [], []
    for sh in shards:
        n = 0
        for batch in pq.ParquetFile(sh).iter_batches(batch_size=64, columns=["image"]):
            for rec in batch.to_pylist():
                b = rec["image"]
                b = b["bytes"] if isinstance(b, dict) else b
                im = Image.open(_io.BytesIO(b))
                dh.append(dhash(im))
                ah.append(ahash(im))
                n += 1
        import os as _os; print(f"  hashed {_os.sep.join(sh.replace(chr(92),'/').split('/')[-2:])}: {n}", flush=True)
    return np.array(dh, dtype=bool), np.array(ah, dtype=bool)


class LeakFilter:
    def __init__(self, tol: float = 0.12, cache: Path = Path("data/external_ai/.wf_hashes.npz")):
        if cache.exists():
            z = np.load(cache)
            self.dh, self.ah = z["dh"], z["ah"]
            print(f"leak filter: loaded {len(self.dh)} cached WildFake hashes", flush=True)
        else:
            print("hashing the WildFake eval set for the leak filter ...", flush=True)
            self.dh, self.ah = load_wildfake_hashes()
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, dh=self.dh, ah=self.ah)
        self.tol = tol
        self.hits = 0

    def leaks(self, img) -> bool:
        for H, q in ((self.dh, dhash(img)), (self.ah, ahash(img))):
            if ((H ^ q).mean(axis=1).min()) <= self.tol:
                self.hits += 1
                return True
        return False


def _save(img, path: Path, max_size: int, square: bool = False) -> None:
    img = img.convert("RGB")
    if square:
        s = min(img.size)
        l, t = (img.width - s) // 2, (img.height - s) // 2
        img = img.crop((l, t, l + s, t + s))
    w, h = img.size
    if max(w, h) > max_size:
        r = max_size / max(w, h)
        img = img.resize((max(1, round(w * r)), max(1, round(h * r))))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, "PNG")


def fetch_dalle(out: Path, n: int, leak: LeakFilter, max_size: int) -> int:
    from huggingface_hub import hf_hub_download
    from PIL import Image
    tar = hf_hub_download(DALLE_REPO, DALLE_TAR, repo_type="dataset")
    dst_dir = out / "fake_dalle3"
    dst_dir.mkdir(parents=True, exist_ok=True)
    have = {p.stem for p in dst_dir.glob("dalle3_*.png")}
    got = len(have)
    if got:
        print(f"  dalle3: resuming, {got} already saved", flush=True)
    tf = tarfile.open(tar)
    members = sorted((m for m in tf.getmembers()
                      if m.isfile() and m.name.lower().endswith(IMG_EXTS)),
                     key=lambda m: m.name)
    for m in members:
        if got >= n:
            break
        if f"dalle3_{Path(m.name).stem}" in have:
            continue
        try:
            img = Image.open(io.BytesIO(tf.extractfile(m).read()))
            img.load()
        except Exception:
            continue
        if leak.leaks(img):
            continue
        _save(img, dst_dir / f"dalle3_{Path(m.name).stem}.png", max_size, square=True)
        got += 1
        if got % 250 == 0:
            print(f"  dalle3: {got}/{n}", flush=True)
    return got


def fetch_progan(out: Path, n: int, leak: LeakFilter, max_size: int) -> tuple[int, int]:
    """ProGAN fakes + LSUN reals from the ForenSynths category zips."""
    from huggingface_hub import HfApi, hf_hub_download
    from PIL import Image
    zips = sorted(s.rfilename for s in HfApi().dataset_info(PROGAN_REPO).siblings
                  if s.rfilename.endswith(".zip"))
    per_zip = max(1, n // max(1, min(len(zips), 10)) + 1)
    fake = len(list((out / "fake_progan").glob("*.png"))) if (out / "fake_progan").is_dir() else 0
    real = len(list((out / "real_lsun").glob("*.png"))) if (out / "real_lsun").is_dir() else 0
    if fake or real:
        print(f"  progan: resuming, fake {fake} real {real} already saved", flush=True)
    for zn in zips:
        if fake >= n and real >= n:
            break
        zp = hf_hub_download(PROGAN_REPO, zn, repo_type="dataset")
        z = zipfile.ZipFile(zp)
        cat = Path(zn).stem
        buckets = {"1_fake": ("fake_progan", "ext_progan"), "0_real": ("real_lsun", "lsun")}
        for key, (folder, tag) in buckets.items():
            names = [x for x in z.namelist()
                     if f"/{key}/" in x and x.lower().endswith(IMG_EXTS)
                     and "__MACOSX" not in x]
            names.sort()
            target = fake if key == "1_fake" else real
            dstd = out / folder
            have = {q.stem for q in dstd.glob("*.png")} if dstd.is_dir() else set()
            taken = 0
            for x in names:
                if target + taken >= n or taken >= per_zip:
                    break
                if f"{tag}_{cat}_{__import__('pathlib').Path(x).stem}" in have:
                    continue
                try:
                    img = Image.open(io.BytesIO(z.read(x)))
                    img.load()
                except Exception:
                    continue
                if leak.leaks(img):
                    continue
                _save(img, out / folder / f"{tag}_{cat}_{Path(x).stem}.png", max_size)
                taken += 1
            if key == "1_fake":
                fake += taken
            else:
                real += taken
        print(f"  progan[{cat}]: fake {fake}/{n}  real {real}/{n}", flush=True)
    return fake, real


def make_square_reals(out: Path, n: int, max_size: int) -> int:
    """Centre-crop existing SID reals to square -- the aspect-ratio partner
    for the (square) DALL-E images."""
    from PIL import Image
    pool = sorted(Path("data/train/real").glob("real_*.png"))
    rng = np.random.default_rng(0)
    pool = [pool[i] for i in rng.permutation(len(pool))]
    dstd = out / "real_square"; dstd.mkdir(parents=True, exist_ok=True)
    have = {q.stem for q in dstd.glob("realsq_*.png")}
    got = len(have)
    for p in pool:
        if got >= n:
            break
        if f"realsq_{p.stem}" in have:
            continue
        try:
            _save(Image.open(p), dstd / f"realsq_{p.stem}.png", max_size, square=True)
            got += 1
        except Exception:
            continue
    return got


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("data/external_ai"))
    ap.add_argument("--per-bucket", type=int, default=3000)
    ap.add_argument("--max-size", type=int, default=512)
    ap.add_argument("--leak-tol", type=float, default=0.12,
                    help="max normalised hamming distance to a WildFake image "
                         "before a fetched image is treated as a leak and dropped")
    args = ap.parse_args()

    leak = LeakFilter(args.leak_tol)

    print("\n[1/4] DALL-E 3 (reddit)")
    d = fetch_dalle(args.out, args.per_bucket, leak, args.max_size)
    print("\n[2/4] ProGAN + LSUN reals (ForenSynths)")
    gf, gr = fetch_progan(args.out, args.per_bucket, leak, args.max_size)
    print("\n[3/4] square real partners for DALL-E")
    sq = make_square_reals(args.out, d, args.max_size)

    print(f"\n[4/4] done -> {args.out}")
    print(f"  fake_dalle3 {d}  fake_progan {gf}  real_lsun {gr}  real_square {sq}")
    print(f"  dropped as WildFake leaks: {leak.hits}")
    if d + gf < args.per_bucket:
        print("WARNING: fewer AI images than requested; sources may be exhausted",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
