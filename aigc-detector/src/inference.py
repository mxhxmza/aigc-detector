"""Single shared inference path: load a checkpoint once, score PIL images.

`predict.py` and `app.py` both score images through `Scorer`, so there is
one forward pass to keep correct. Before a shared path existed, each caller
reimplemented it, which is how the forensic-normalisation step silently went
missing from inference for a while -- the fix had to land in three files and
one was missed. `evaluate.py` / `error_analysis.py` still carry their own
`score_images` (they apply a fixed transform grid to already-loaded images,
a different access pattern) but follow the same contract below.

The contract that must not drift:

  1. semantic  = L2-normalised frozen-CLIP image embedding
  2. forensic  = FQ.extract(img), THEN standardised with the train-split
     (mu, sd) stored in the checkpoint -- the model only ever saw
     standardised forensic features during training
  3. probability = model.predict_proba(...) which also applies the fitted
     temperature

Usage:
    from src.inference import Scorer
    scorer = Scorer("checkpoints_cifake/full.pt")
    p = scorer.score_one(pil_image)          # float in [0, 1]
    ps = scorer.score_many([img1, img2])     # list[float]
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .features import frequency as FQ
from .features.extract import BACKBONES
from .models.detector import Detector, count_parameters


class Scorer:
    """A loaded detector. Construct once (it is not cheap), reuse for every
    image -- constructing it loads the CLIP backbone into memory / VRAM."""

    def __init__(self, checkpoint: str | Path, device: str = "auto"):
        import torch
        import open_clip

        self.checkpoint = Path(checkpoint)
        if not self.checkpoint.exists():
            raise FileNotFoundError(
                f"checkpoint not found: {self.checkpoint}. Train one with "
                "`python -m src.train`."
            )

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        ckpt = torch.load(self.checkpoint, map_location=device, weights_only=False)
        self.config = ckpt["config"]

        model_name, pretrained, _ = BACKBONES[self.config["backbone"]]
        clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.backbone = clip_model.visual.to(device).eval()
        if device == "cuda":
            self.backbone = self.backbone.half()

        # Legacy checkpoints (CIFAKE, early SID) have no n_classes key and are
        # binary sigmoid heads; SID_Set 3-class checkpoints store n_classes=3.
        self.n_classes = self.config.get("n_classes", 1)
        self.model = Detector(
            semantic_dim=self.config["semantic_dim"],
            forensic_dim=self.config["forensic_dim"],
            hidden=self.config.get("hidden", 256),
            use_forensic=self.config.get("use_forensic", True),
            use_gate=self.config.get("use_gate", True),
            n_classes=self.n_classes,
        ).to(device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

        mu = ckpt.get("forensic_mu")
        sd = ckpt.get("forensic_sd")
        if mu is None or sd is None:
            raise ValueError(
                f"{self.checkpoint} has no forensic_mu/forensic_sd -- it "
                "predates the forensic-normalisation fix. Retrain with the "
                "current src/train.py."
            )
        self.forensic_mu = mu.to(device)
        self.forensic_sd = sd.to(device)

        self.params = count_parameters(
            self.model, sum(p.numel() for p in self.backbone.parameters())
        )
        self.temperature = float(self.model.temperature.item())

    def _score(self, images: list[Image.Image], batch_size: int, full: bool):
        import torch

        out: list = []
        buf_t, buf_f = [], []

        def flush():
            if not buf_t:
                return
            with torch.no_grad():
                x = torch.stack(buf_t).to(self.device)
                if self.device == "cuda":
                    x = x.half()
                emb = self.backbone(x)
                emb = emb / emb.norm(dim=-1, keepdim=True)
                f = torch.from_numpy(np.stack(buf_f).astype(np.float32)).to(self.device)
                f = (f - self.forensic_mu) / self.forensic_sd
                if full:
                    p = self.model.predict_proba_full(emb.float(), f)
                else:
                    p = self.model.predict_proba(emb.float(), f)
            out.extend(p.cpu().numpy().tolist())
            buf_t.clear()
            buf_f.clear()

        for img in images:
            rgb = img.convert("RGB")
            buf_t.append(self.preprocess(rgb))
            buf_f.append(FQ.extract(rgb))
            if len(buf_t) >= batch_size:
                flush()
        flush()
        return out

    def score_many(self, images: list[Image.Image], batch_size: int = 32) -> list[float]:
        """Calibrated p(AI-generated) for each image, in order.

        For a 3-class checkpoint this is p(fully synthetic); tampered images
        score low here on purpose -- an edited photo is still a photo.
        """
        return self._score(images, batch_size, full=False)

    def score_many_detailed(self, images: list[Image.Image],
                            batch_size: int = 32) -> list[dict]:
        """Per-class calibrated probabilities plus the collapsed verdict split.

        Each entry: {p_real, p_ai_generated, p_tampered, p_authentic}.
        `p_authentic = p_real + p_tampered` -- the number the website shows.
        For a binary checkpoint p_tampered is 0 and p_real = 1 - p_ai.
        """
        rows = self._score(images, batch_size, full=True)
        result = []
        for r in rows:
            p_real = float(r[0])
            p_ai = float(r[1])
            p_tam = float(r[2]) if len(r) > 2 else 0.0
            result.append({
                "p_real": p_real,
                "p_ai_generated": p_ai,
                "p_tampered": p_tam,
                "p_authentic": p_real + p_tam,
            })
        return result

    def score_one(self, image: Image.Image) -> float:
        return self.score_many([image])[0]
