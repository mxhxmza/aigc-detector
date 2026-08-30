# Robust Detection of AI-Generated Images Under Real-World Transformations

TikTok TechJam 2026 — Problem Statement #5 · **Solo submission**

> Section order follows the brief's deliverable list (§5.5) so each required
> item is where a judge expects it: overview → setup → reproduction →
> limitations → contributions.

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

### Relationship to prior work
This builds on, rather than reproduces, two known families: CLIP-feature
detectors and frequency-artifact detectors. The contribution is the
degradation-conditioned gating between them plus the consistency objective.
Neither branch alone is novel and we do not claim it is.

---

## 2. Setup and Installation

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

### 3.0 Smoke test (optional, ~3 minutes, no datasets required)

Before downloading tens of GB, verify the plumbing end-to-end on generated
data. The "real" class carries 1/f falloff plus a sensor-noise floor; the
"fake" class is built small and bicubically upsampled, planting the periodic
artifact the forensic branch looks for.

```bash
python scripts/make_smoke_data.py --out data/smoke
python scripts/build_manifest.py --wildfake data/smoke --out data/smoke_manifest.csv
python -m src.features.extract --manifest data/smoke_manifest.csv \
    --out features_smoke/ --views 4 --backbone ViT-B-16 --batch-size 64
python -m src.train --features features_smoke/ --out checkpoints_smoke/ \
    --epochs 8 --batch-size 32 --tag full
python predict.py --image-dir data/smoke/fake --out out_smoke/preds.json \
    --checkpoint checkpoints_smoke/full.pt
```

The smoke set is separable by construction, so it reports AUC ≈ 1.0. That
number means "the pipeline is wired correctly", **not** "the model works" —
never quote it. Only the real datasets produce a reportable result.

### 3.1 CIFAKE quickstart (real data, ~25 minutes end-to-end)

CIFAKE (C-3 approved) is the fastest real dataset to train on: 120k images,
public, one command to fetch. It is a single generator (Stable Diffusion 1.4)
at CIFAR resolution (32×32), so it cannot measure cross-generator transfer —
the `unseen_generator` row is correctly absent — and its numbers are an
easier problem than WildFake. Use it to get a working, calibrated model
today; add WildFake / SID_Set for the submission result.

```bash
pip install kagglehub
python -c "import kagglehub; print(kagglehub.dataset_download('birdy654/cifake-real-and-ai-generated-synthetic-images'))"
# prints e.g.  ~/.cache/kagglehub/.../versions/3   -- use that as $CIFAKE

python scripts/build_manifest.py --cifake "$CIFAKE" \
    --sample-per-class 6000 --out data/cifake_manifest.csv        # omit --sample-per-class for all 120k
python -m src.features.extract --manifest data/cifake_manifest.csv \
    --out features_cifake/ --views 4 --backbone ViT-B-16 --batch-size 128 --workers 4
python -m src.train --features features_cifake/ --out checkpoints_cifake/ \
    --epochs 30 --batch-size 256 --tag full
python -m src.evaluate --manifest data/cifake_manifest.csv \
    --checkpoint checkpoints_cifake/full.pt --split val --out results_cifake/
python -m src.error_analysis --manifest data/cifake_manifest.csv \
    --checkpoint checkpoints_cifake/full.pt --split val --out results_cifake/
python predict.py --image-dir <any image dir> --out preds.json \
    --checkpoint checkpoints_cifake/full.pt
```

Reference numbers from the 6000/class subset (ViT-B-16, RTX 5060, seed 0):
clean AUC **0.996**, per-family robust score **0.979**, mean AUC drop under
transformation **+0.018**, ECE **0.012**. On a fresh 300-image slice of the
CIFAKE test split: AUC **0.993**, accuracy **95.3%**. Worst cells are heavy
blur (σ=2.0, AUC 0.93) and aggressive downscale (0.25×, AUC 0.93), which is
expected — both destroy the high-frequency forensic signal, and there is
little of it to begin with at 32px.

### 3.1b SID_Set (realistic 1024px data, C-3 approved)

CIFAKE is 32px toy data — a model trained only on it makes obvious mistakes
on real photographs. [SID_Set](https://huggingface.co/datasets/saberzl/SID_Set)
is the fix: 300k full-resolution images (real photos from OpenImages V7, plus
fully-synthetic and tampered), CC-BY-4.0. It is 140 GB, so `fetch_sid_set.py`
streams a balanced subset (no full download), re-saves everything as PNG at
≤512px, and lays it out for `build_manifest`.

SID_Set ships three labels — real, fully AI-generated, and tampered (a real
photo with an AI-edited region). The detector is a **binary** real-vs-AI
classifier (`--n-classes 1`, the default): a tampered image was still taken
by a person, so `src/train.py` folds those rows into class 0 (real). Only
fully AI-generated images are flagged. The `--n-classes 3` softmax head
(real / AI / tampered as distinct classes) is still available if wanted.

```bash
pip install datasets
python scripts/fetch_sid_set.py --out data/sid_set --split train \
    --per-class 10000 --include-tampered
python scripts/build_manifest.py --sid-set data/sid_set --no-holdout \
    --out data/sid_manifest.csv
python -m src.features.extract --manifest data/sid_manifest.csv \
    --out features_sid/ --views 4 --backbone ViT-B-16 --batch-size 128 --workers 4
python -m src.train --features features_sid/ --out checkpoints_sid/ \
    --epochs 30 --batch-size 256 --tag full          # binary head; tampered -> real
python -m src.evaluate --manifest data/sid_manifest.csv \
    --checkpoint checkpoints_sid/full.pt --split val --out results_sid/
python app.py --checkpoint checkpoints_sid/full.pt        # point the UI at the better model
```

`--include-tampered` on the fetch pulls the label-2 images. They still help
the binary head (harder "real" examples), and are only kept as a separate
class under `--n-classes 3`. `--no-holdout` keeps every generator in
train/val — the goal here is a model that works across all of SID_Set, not a
cross-generator transfer measurement.
Combine sources by passing both `--cifake` and `--sid-set` to one
`build_manifest` call.

```bash
# 1. Build the manifest. Any subset of datasets works; missing ones are skipped.
#    The provided validation subset is marked and never trained on (C-4).
python scripts/build_manifest.py \
    --wildfake data/wildfake --sid-set data/sid_set \
    --provided-real data/val/coco_val2017 \
    --provided-aigc data/val/dalle_advanced \
    --out data/manifest.csv

# 2. Extract cached features. One-time cost; everything after is seconds.
#    Time it on a small slice first: add --limit 500
python -m src.features.extract \
    --manifest data/manifest.csv --out features/ \
    --views 4 --backbone ViT-L-14 --batch-size 64 --workers 4

# 3. Train. The ablation ladder (run all four for the ablation table):
python -m src.train --features features/ --tag baseline   --no-forensic --no-gate --no-consistency
python -m src.train --features features/ --tag aug        --no-forensic --no-gate
python -m src.train --features features/ --tag freq       --no-gate
python -m src.train --features features/ --tag full

# 4. Robustness evaluation -> results/robustness_table.md   (deliverable D4)
python -m src.evaluate --manifest data/manifest.csv \
    --checkpoint checkpoints/full.pt --split val --out results/

# 5. Error analysis -> results/error_analysis.md            (deliverable D5)
python -m src.error_analysis --manifest data/manifest.csv \
    --checkpoint checkpoints/full.pt --split val --out results/

# 6. Inference (the graded CLI)
python predict.py --image-dir path/to/images --out predictions.json
```

### 3.2 Web interface

`app.py` is a local single-page app: drag in an image, get the calibrated
probability with a verdict and a slider showing where it lands between
"authentic" and "AI-generated". It is a thin front-end over the exact same
forward pass as `predict.py` (both go through `src/inference.py::Scorer`),
runs entirely on your machine, and never writes uploads to disk.

```bash
pip install fastapi uvicorn python-multipart
python app.py --checkpoint checkpoints_cifake/full.pt      # then open http://127.0.0.1:8000
```

`--host 0.0.0.0` exposes it on the LAN; `--port` and `--device` work as
expected. `GET /api/info` reports the loaded checkpoint and `POST
/api/predict` (multipart field `image`) returns the JSON score, so the same
server doubles as a scoring API.

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

- **Generator-disjoint splits.** Whole generators are held out rather than
  splitting randomly. A random split lets the same generator appear in train
  and test, which measures memorisation of a fingerprint rather than
  detection. This lowers the headline number and makes it mean something.
- **Automatic leakage guard.** Every manifest build verifies that no image
  from the provided validation subset appears in training (C-4). Two stages:
  loose perceptual-hash recall, then pixel-level confirmation. Measured
  separation on natural images is wide — true duplicates at 0.03–0.14,
  distinct images above 1.0 — so the check catches hard JPEG re-encodes and
  resize round-trips without false alarms.
- **Model selection on the balanced objective**, not clean AUC. Selecting on
  clean accuracy is the failure mode this whole project is about.
- **Temperature scaling** fitted on held-out data after selection. It cannot
  change ranking, so AUC is unaffected; it only makes the confidence mean
  something.

### Compliance

| Rule | Status |
|---|---|
| <2B parameters (C-1) | ViT-B/16 run: 86.76M total (563.7k trainable + 86.19M frozen backbone), 23× headroom; ViT-L/14 ≈ 0.32B. Printed at load by `count_parameters` |
| Public backbones only (C-2) | OpenAI CLIP via `open_clip`; no custom or private weights |
| Public/licensed data only (C-3) | WildFake, SID_Set, CIFAKE |
| No test-label training (C-4) | Enforced automatically; report printed at manifest build |
| Augmentation scripts committed (C-5) | `src/transforms/degradations.py` + seeded sampler |
| Open-source (C-6) | MIT, see LICENSE |

Measured on a consumer RTX 5060 Laptop (8GB, sm_120), ViT-B-16:
feature extraction **157 img·view/s**, inference through `predict.py`
**~15 img/s** end-to-end (CLIP + forensic features + head), training
**~4 s/epoch** on 10.5k cached pairs. The ViT-L-14 backbone is one flag away
(`--backbone ViT-L-14`) and still well under the cap.

---

## 4. Limitations and What I'd Improve With More Time

**Known limitations**

1. **Frozen backbone.** The semantic branch never adapts to the detection
   task. This was a deliberate trade for iteration speed on a single 8GB GPU;
   fine-tuning the top blocks would likely help and was not affordable in 72
   hours.
2. **Finite augmentation views.** Training uses K=4 fixed views per image
   rather than fresh random augmentation each epoch, a direct consequence of
   caching features. Larger K trades disk for diversity.
3. **Cross-generator generalisation is the weakest result**, as expected. The
   unseen-generator row is the honest measure of how this would behave against
   a model released next month.
4. **Incidental degradation, not adversarial.** The six transforms model a
   redistribution pipeline, not an adversary deliberately evading detection.
   An attacker who knows the detector could do much better than JPEG.
5. **Dataset shortcut risk is mitigated, not eliminated.** Generator-disjoint
   splits and source-matching reduce it; the error analysis checks whether
   false positives concentrate in one source dataset, which is the signature.
6. **False positives are accusations.** Calling a real photograph synthetic
   has a human cost. At this accuracy the appropriate deployment is a
   human-review queue, not automated enforcement.

**With more time**

- Unfreeze the last transformer blocks and compare.
- Add a provenance signal (C2PA) as a third branch where present.
- Test against generators released after the training data was collected.
- Extend the gate to predict a *reliability* estimate, so the system can
  abstain on images too degraded to judge — more useful in a moderation
  pipeline than a forced answer.

---

## 5. Team Member Contributions

**Solo submission.** All work — data pipeline, model, evaluation harness,
error analysis, and write-up — by the single participant listed on Devpost.

---

## Repository Layout

```
predict.py                    graded inference CLI
app.py                        local web interface (upload -> probability)
scripts/verify_env.py         sm_120 / CUDA check -- run first
scripts/build_manifest.py     dataset adapters + leakage guard
scripts/make_smoke_data.py    synthetic dataset for the plumbing smoke test
src/inference.py              Scorer -- the single shared forward pass
src/transforms/degradations.py  the six transforms (C-5)
src/features/frequency.py     hand-designed forensic features
src/features/extract.py       one-time cached feature extraction
src/models/detector.py        two-branch model + gate + loss
src/models/calibration.py     temperature scaling
src/train.py                  training + ablation switches
src/evaluate.py               robustness table (D4)
src/error_analysis.py         FP/FN analysis (D5)
src/metrics.py                AUC, ECE, TPR@FPR, final-score formula
configs/default.yaml          committed hyperparameters
```
