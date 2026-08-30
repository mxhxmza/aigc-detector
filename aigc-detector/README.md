# Robust Detection of AI-Generated Images Under Real-World Transformations

TikTok TechJam 2026 — Problem Statement #5 · **Solo submission**

> Section order follows the brief's deliverable list: overview → setup →
> reproduction → limitations → contributions.

---

## 1. Project Overview

### The problem
Detectors that score well on pristine images collapse on real ones. The most
discriminative generative fingerprints — periodic upsampling artifacts in the
Fourier spectrum, absent sensor noise, over-smooth micro-texture — live in the
high-frequency band that JPEG re-encoding, blur and downscaling attenuate
first. A social platform's redistribution pipeline is, incidentally, an almost
optimal attack on frequency-domain forensics. Every image a moderation system
sees has already been through it.

### The approach
Two branches with a **degradation-aware gate**:

```
   frozen CLIP embedding ────────► semantic head ──┐
                                                    ├─► gated fusion ─► calibrated p(AI)
   hand-designed frequency ──────► forensic head ──┘
   features            │                    ▲
                       └─► degradation estimator ───┘
                           (predicts how damaged the image is)
```

The forensic branch is precise on clean images and fragile under compression.
The semantic branch is coarser but survives. Rather than mixing them with
fixed weights, a small head **estimates how degraded the image is** and that
estimate sets the fusion weights — so the model can lean on frequency evidence
for a pristine PNG and on semantics for a q30 re-encode, without being told
which it is at test time.

The degradation estimator trains on free supervision: we applied the
degradations, so we know the answer.

Training adds a **consistency loss** between the clean and degraded views of
the same image. Plain augmentation says "also classify the damaged version
correctly". Consistency says "give it the *same* answer as the clean one",
which is strictly stronger and is the property the robustness table measures.

### The task
A binary classifier: **real vs fully AI-generated**. The data (SID_Set) also
contains *tampered* images — real photographs with an AI-edited region. A
tampered image was still taken by a person, so it is treated as real; only
fully synthetic images are flagged.

### Region-median inference
A whole-frame score can be dominated by one small region — a blurred
foreground, a hand, framing foliage — that the model reads as synthetic,
while the actual subject reads as clearly real. So inference (`predict.py`,
`app.py`, and the robustness eval) scores **six overlapping region crops and
takes the median**. A genuinely synthetic image looks generated everywhere,
so it is unaffected; a real photo with one odd region is no longer
misjudged. On the held-out set this cut false positives from 13 to 4 while
accuracy and F1 went up. `--no-multicrop` disables it.

### Relationship to prior work
This builds on, rather than reproduces, two known families: CLIP-feature
detectors and frequency-artifact detectors. The contribution is the
degradation-conditioned gating between them plus the consistency objective.
Neither branch alone is novel and we do not claim it is.

---

## 2. Setup

**Blackwell / RTX 50-series users read this first.** These GPUs are compute
capability `sm_120`. A PyTorch wheel built against older CUDA will report
`cuda.is_available() == True`, load models fine, and then fail on the first
real forward pass. Install from the CUDA 12.8 index:

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

python scripts/verify_env.py     # must print PASS before you go further
```

`verify_env.py` runs an actual fp16 matmul rather than just allocating a
tensor, because allocation succeeds on builds that cannot execute kernels.

---

## 3. Steps to Reproduce

The whole pipeline is one linear sequence. There is **one dataset** and **one
model** — no per-source variants.

### 3.0 Smoke test (optional, ~3 min, no download)

Verify the plumbing end-to-end on generated data before fetching anything.
The synthetic "real" class carries 1/f falloff plus a sensor-noise floor; the
"fake" class is built small and bicubically upsampled, planting the periodic
artifact the forensic branch looks for.

```bash
python scripts/make_smoke_data.py --out data/smoke_raw
python scripts/build_dataset.py --raw data/smoke_raw --out data/smoke \
    --manifest data/smoke/manifest.csv
python -m src.features.extract --manifest data/smoke/manifest.csv \
    --out features_smoke/ --views 4 --backbone ViT-B-16 --batch-size 64
python -m src.train --features features_smoke/ --out checkpoints_smoke/ \
    --epochs 8 --batch-size 32 --tag full
```

The smoke set is separable by construction, so it reports AUC ≈ 1.0. That
number means "the pipeline is wired correctly", **not** "the model works" —
never quote it.

### 3.1 The real run

[SID_Set](https://huggingface.co/datasets/saberzl/SID_Set) (CC-BY-4.0) is
~300k full-resolution images: real photos from OpenImages V7, fully-synthetic
images, and tampered images. It is 140 GB, so `fetch_sid_set.py` streams a
balanced subset (no full download), re-saving everything as PNG at ≤512 px.

```bash
pip install datasets

# 1. Download a balanced subset (~6 GB). Resumes if interrupted.
python scripts/fetch_sid_set.py --out data/sid_set --per-class 10000 --include-tampered

# 2. Split and organise into data/train/{real,ai} and data/test/{real,ai},
#    and write data/manifest.csv. The train/test boundary is a physical
#    folder split, so "never trained on test" is structural, not a promise.
python scripts/build_dataset.py

# 3. Extract cached features. One-time cost (~15 min); every run after is seconds.
python -m src.features.extract --manifest data/manifest.csv --out features/ \
    --views 4 --backbone ViT-B-16 --batch-size 128 --workers 4

# 4. Train. The ablation ladder -- run all four for the ablation table:
python -m src.train --features features/ --tag baseline --no-forensic --no-gate --no-consistency
python -m src.train --features features/ --tag aug      --no-forensic --no-gate
python -m src.train --features features/ --tag freq     --no-gate
python -m src.train --features features/ --tag full

# 5. Robustness evaluation -> results/robustness_table.md
python -m src.evaluate --manifest data/manifest.csv \
    --checkpoint checkpoints/full.pt --split test --out results/

# 6. Error analysis -> results/error_analysis.md
python -m src.error_analysis --manifest data/manifest.csv \
    --checkpoint checkpoints/full.pt --split test --out results/

# 7. Inference (the graded CLI)
python predict.py --image-dir path/to/images --out predictions.json
```

### Reference numbers

SID_Set subset (18,568 images, 2,228 held out in `data/test/`), ViT-B-16,
6-crop median inference, RTX 5060, seed 0.

Test-set classification report at threshold 0.5:

| | precision | recall | F1 |
|---|---|---|---|
| real (incl. tampered) | 0.997 | 0.997 | 0.997 |
| AI-generated | 0.995 | 0.993 | 0.994 |
| **overall** | | | **acc 0.996 · ROC-AUC 1.000 · PR-AUC 1.000** |

4 false positives (0.27% of real — 2 genuine photos, 2 tampered), 5 false
negatives (0.67% of AI). Multi-crop cut false positives from 13 to 4 versus
whole-frame scoring while raising accuracy and F1.

Robustness grid (`results/robustness_table.md`): re-run
`python -m src.evaluate --manifest data/manifest.csv --checkpoint
checkpoints/full.pt --split test --out results/` to refresh it for the
6-crop path. The whole-frame grid gave clean AUC 1.000, worst-cell AUC
0.999, mean AUC drop +0.0004, ECE ~0.007.

### 3.2 Web interface

`app.py` is a local single-page app: drag in an image, get the calibrated
probability with a verdict and a slider between "authentic" and
"AI-generated". It is a thin front-end over the exact same forward pass as
`predict.py` (both go through `src/inference.py::Scorer`), runs entirely on
your machine, and never writes uploads to disk.

```bash
pip install fastapi uvicorn python-multipart
python app.py                                  # then open http://127.0.0.1:8000
```

`--host 0.0.0.0` exposes it on the LAN; `--port`, `--device`, `--checkpoint`
work as expected. `GET /api/info` reports the loaded checkpoint and `POST
/api/predict` (multipart field `image`) returns the JSON score, so the server
doubles as a scoring API.

### Output format

```json
[
  {"image_path": "images/a.jpg", "pred": 0.9312},
  {"image_path": "images/b.png", "pred": 0.0417}
]
```

`pred` is a calibrated probability in [0,1] that the image is AI-generated.
`--format dict` and `--binary` cover the alternate readings of the spec.

### Design decisions that affect the numbers

- **Physical train/test folders.** `build_dataset.py` moves each image into
  `data/train/` or `data/test/`. An image is in exactly one folder, so
  training on a test image is impossible by construction — no hash-based
  leakage check to trust or disable.
- **Region-median inference** (six crops). See §1. It is applied identically
  in the robustness eval, so the table reflects deployed behaviour.
- **Model selection on the balanced objective** (clean + transformed AUC),
  not clean AUC. Selecting on clean accuracy is the failure mode this whole
  project is about.
- **Temperature scaling** fitted on the held-out split after selection. It
  cannot change ranking, so AUC is unaffected; it only makes the confidence
  mean something.

### Compliance

| Rule | Status |
|---|---|
| <2B parameters (C-1) | ViT-B/16: 86,756,364 total (563,724 trainable + 86,192,640 frozen backbone), 23.1× headroom; ViT-L/14 ≈ 0.32B. Printed at load by `count_parameters`. |
| Public backbones only (C-2) | OpenAI CLIP via `open_clip`; no custom or private weights. |
| Public/licensed data only (C-3) | SID_Set, CC-BY-4.0. |
| No test-label training (C-4) | Structural: train/ and test/ are separate directories. |
| Augmentation scripts committed (C-5) | `src/transforms/degradations.py` + seeded sampler. |
| Open-source (C-6) | MIT, see LICENSE. |

Measured on a consumer RTX 5060 Laptop (8GB, sm_120), ViT-B-16: feature
extraction ~75 img·view/s, inference through `predict.py` ~5 img/s with
6-crop median (~30 img/s with `--no-multicrop`), training ~4 s/epoch on
cached pairs. `--backbone ViT-L-14` is one flag away and still under the cap.

---

## 4. Limitations and What I'd Improve With More Time

**Known limitations**

1. **Frozen backbone.** The semantic branch never adapts to the detection
   task — a deliberate trade for iteration speed on a single 8GB GPU.
   Fine-tuning the top blocks would likely help.
2. **Finite augmentation views.** Training uses K=4 fixed views per image
   rather than fresh random augmentation each epoch, a consequence of caching
   features. Larger K trades disk for diversity.
3. **One dataset.** SID_Set covers real photos plus a fixed set of
   synthesis methods. A generator released next month is out of distribution;
   the honest way to measure that is to test on it, which is future work.
4. **Incidental degradation, not adversarial.** The six transforms model a
   redistribution pipeline, not an adversary deliberately evading detection.
5. **Tampered images are called real.** A photo with a small AI-edited region
   is treated as authentic. That is the right product call for "is this
   photo real", but it means localised manipulation is not surfaced.
6. **False positives are accusations.** Calling a real photograph synthetic
   has a human cost. At this accuracy the appropriate deployment is a
   human-review queue, not automated enforcement.

**With more time**

- Unfreeze the last transformer blocks and compare.
- Add a provenance signal (C2PA) as a third branch where present.
- Test against generators released after the training data was collected.
- Extend the gate to predict a *reliability* estimate, so the system can
  abstain on images too degraded to judge.

---

## 5. Team Member Contributions

**Solo submission.** All work — data pipeline, model, evaluation harness,
error analysis, and write-up — by the single participant listed on Devpost.

---

## Repository Layout

```
predict.py                      graded inference CLI  (-> predictions.json)
app.py                          local web interface (upload -> probability)
configs/default.yaml            committed hyperparameters
scripts/verify_env.py           sm_120 / CUDA check -- run first
scripts/fetch_sid_set.py        stream a balanced SID_Set subset to disk
scripts/build_dataset.py        split + organise -> data/{train,test} + manifest.csv
scripts/make_smoke_data.py      tiny synthetic dataset for the plumbing test
src/data/manifest.py            the manifest: image_path, label, kind, split
src/transforms/degradations.py  the six specified transforms (C-5)
src/features/frequency.py       hand-designed forensic features
src/features/extract.py         one-time cached feature extraction
src/models/detector.py          two-branch model + degradation gate + loss
src/models/calibration.py       temperature scaling
src/train.py                    training + ablation switches
src/evaluate.py                 robustness table
src/error_analysis.py           false-positive / false-negative analysis
src/metrics.py                  AUC, ECE, TPR@FPR, final-score formula
```
