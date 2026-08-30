"""Train the detector on cached features.

Because features are precomputed (§6.5), an epoch is a pass over a few
hundred MB of vectors. Expect seconds per epoch, which is the whole point --
it turns the ablation table from an aspiration into an afternoon.

Each training example is a PAIR: the clean view of an image and one of its
degraded views. That pairing is what the consistency loss needs, and it is
why the sampler groups rows by `image_id` rather than shuffling rows freely.

Ablation switches (these produce the four rows of the ablation table):
    --no-forensic --no-gate --no-consistency    semantic baseline
    --no-forensic --no-gate                     + augmentation only
    --no-gate                                   + frequency branch
    (none)                                      + degradation gate

Usage:
    python -m src.train --features features/ --out checkpoints/ --epochs 30
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_features(feat_dir: Path):
    data = np.load(feat_dir / "features.npz")
    meta = json.loads((feat_dir / "meta.json").read_text(encoding="utf-8"))
    info = json.loads((feat_dir / "extract_info.json").read_text(encoding="utf-8"))
    return data, meta, info


def build_pairs(meta: list[dict], split: str) -> list[tuple[int, list[int]]]:
    """Group row indices by image, returning (clean_row, [augmented_rows])."""
    by_image: dict[int, dict] = defaultdict(lambda: {"clean": None, "aug": []})
    for row_idx, m in enumerate(meta):
        if m["split"] != split:
            continue
        if m["view"] == 0:
            by_image[m["image_id"]]["clean"] = row_idx
        else:
            by_image[m["image_id"]]["aug"].append(row_idx)
    return [
        (v["clean"], v["aug"])
        for v in by_image.values()
        if v["clean"] is not None and v["aug"]
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=Path("checkpoints"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lambda-consistency", type=float, default=1.0)
    ap.add_argument("--lambda-degradation", type=float, default=0.1)
    ap.add_argument("--no-forensic", action="store_true")
    ap.add_argument("--no-gate", action="store_true")
    ap.add_argument("--no-consistency", action="store_true")
    ap.add_argument("--n-classes", type=int, default=1,
                    help="1 = binary real-vs-AI sigmoid head; tampered rows are "
                         "folded into class 0 (real). 3 = {real, AI, tampered} "
                         "softmax head.")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader, Dataset

    from .metrics import auc, expected_calibration_error
    from .models.detector import Detector, count_parameters, detector_loss

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data, meta, info = load_features(args.features)
    semantic = torch.from_numpy(data["semantic"].astype(np.float32))
    forensic = torch.from_numpy(data["forensic"].astype(np.float32))
    degrade = torch.from_numpy(data["degrade"].astype(np.float32))
    # int64 labels: {0 real, 1 AI-generated, 2 tampered}. detector_loss casts
    # to float internally for the legacy binary head.
    labels = torch.from_numpy(data["label"].astype(np.int64))

    if args.n_classes == 1:
        # Binary detector: real vs fully AI-generated only. A tampered image
        # (real photo with an AI-edited region) is still a real photograph, so
        # its rows are relabelled to class 0. No re-extraction needed -- the
        # cached features are identical; only the target changes.
        n_tam = int((labels == 2).sum())
        if n_tam:
            labels[labels == 2] = 0
            print(f"binary head: folded {n_tam} tampered rows into class 0 (real)")

    # Standardise forensic features once, globally. Stats come from TRAIN
    # rows only -- computing them over everything would leak val statistics
    # into training, which is a small but real form of cheating.
    train_rows = [i for i, m in enumerate(meta) if m["split"] == "train"]
    mu = forensic[train_rows].mean(0, keepdim=True)
    sd = forensic[train_rows].std(0, keepdim=True).clamp(min=1e-6)
    forensic = (forensic - mu) / sd

    class PairDataset(Dataset):
        def __init__(self, pairs, rng_seed: int):
            self.pairs = pairs
            self.rng = np.random.default_rng(rng_seed)

        def __len__(self) -> int:
            return len(self.pairs)

        def __getitem__(self, i):
            clean_row, aug_rows = self.pairs[i]
            aug_row = aug_rows[self.rng.integers(len(aug_rows))]
            return clean_row, aug_row

    train_pairs = build_pairs(meta, "train")
    val_pairs = build_pairs(meta, "val")
    if not train_pairs:
        raise SystemExit("no training pairs found -- check manifest splits")
    print(f"train pairs: {len(train_pairs)}   val pairs: {len(val_pairs)}")

    loader = DataLoader(
        PairDataset(train_pairs, args.seed),
        batch_size=args.batch_size, shuffle=True, drop_last=True,
    )

    model = Detector(
        semantic_dim=semantic.shape[1],
        forensic_dim=forensic.shape[1],
        hidden=args.hidden,
        dropout=args.dropout,
        use_forensic=not args.no_forensic,
        use_gate=not args.no_gate,
        n_classes=args.n_classes,
    ).to(device)

    counts = count_parameters(model, info.get("backbone_params", 0))
    print(f"parameters: {counts['total']:,} total | 2B cap "
          f"{'OK' if counts['compliant'] else 'VIOLATION'} "
          f"({counts['headroom_x']}x headroom)")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    semantic, forensic = semantic.to(device), forensic.to(device)
    degrade, labels = degrade.to(device), labels.to(device)

    lam_c = 0.0 if args.no_consistency else args.lambda_consistency
    lam_d = 0.0 if args.no_forensic else args.lambda_degradation

    def evaluate(pairs) -> dict:
        model.eval()
        rows_clean = torch.tensor([p[0] for p in pairs], device=device)
        rows_aug = torch.tensor([p[1][0] for p in pairs], device=device)
        with torch.no_grad():
            oc = model(semantic[rows_clean], forensic[rows_clean])
            oa = model(semantic[rows_aug], forensic[rows_aug])
        pc = oc["prob"].cpu().numpy()          # p(AI-generated), clean view
        pa = oa["prob"].cpu().numpy()          # p(AI-generated), aug view
        y = labels[rows_clean].cpu().numpy()
        # The headline task is "is this fully AI-generated?" -- class 1 vs the
        # rest. Real and tampered are both negatives here.
        y_ai = (y == 1).astype(int)
        out = {
            "auc_clean": auc(y_ai, pc),
            "auc_aug": auc(y_ai, pa),
            "ece_clean": expected_calibration_error(y_ai, pc),
        }
        if args.n_classes > 1:
            pred_c = oc["probs"].argmax(-1).cpu().numpy()
            out["acc3_clean"] = float((pred_c == y).mean())
            # Of the tampered images, how many are (correctly) NOT flagged AI?
            tam = y == 2
            out["tampered_as_authentic"] = (
                float((pred_c[tam] != 1).mean()) if tam.any() else float("nan"))
        return out

    best_score, best_state = -1.0, None
    history = []

    for epoch in range(args.epochs):
        model.train()
        agg = defaultdict(float)
        n_batches = 0
        for clean_rows, aug_rows in loader:
            clean_rows = clean_rows.to(device)
            aug_rows = aug_rows.to(device)

            out_clean = model(semantic[clean_rows], forensic[clean_rows])
            out_aug = model(semantic[aug_rows], forensic[aug_rows])

            loss, parts = detector_loss(
                out_clean, out_aug, labels[clean_rows],
                degrade[clean_rows], degrade[aug_rows],
                lambda_consistency=lam_c, lambda_degradation=lam_d,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            for k, v in parts.items():
                agg[k] += v
            n_batches += 1
        sched.step()

        val = evaluate(val_pairs) if val_pairs else {"auc_clean": float("nan"),
                                                    "auc_aug": float("nan"),
                                                    "ece_clean": float("nan")}
        # Select on the balanced objective, not clean AUC. Selecting on clean
        # AUC is exactly the failure the workshop warned about. For the 3-class
        # head, fold in clean 3-class accuracy so a model that nails AI
        # detection but muddles real vs tampered is not picked.
        score = 0.5 * val["auc_clean"] + 0.5 * val["auc_aug"]
        if "acc3_clean" in val:
            score = 0.4 * val["auc_clean"] + 0.4 * val["auc_aug"] + 0.2 * val["acc3_clean"]
        history.append({"epoch": epoch, **{k: v / max(n_batches, 1) for k, v in agg.items()},
                        **val, "select_score": score})

        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        extra = ""
        if "acc3_clean" in val:
            extra = (f" | acc3 {val['acc3_clean']:.3f} "
                     f"tamp->auth {val['tampered_as_authentic']:.3f}")
        print(f"epoch {epoch:3d} | loss {agg['loss']/max(n_batches,1):.4f} "
              f"| cons {agg['consistency']/max(n_batches,1):.4f} "
              f"| val auc clean {val['auc_clean']:.4f} aug {val['auc_aug']:.4f} "
              f"| gap {val['auc_clean']-val['auc_aug']:+.4f}{extra}"
              f"{'  *' if score == best_score else ''}")

    model.load_state_dict(best_state)

    # Temperature scaling on the val split (§FR-6). Fitted after selection so
    # it calibrates the model we are actually shipping.
    from .models.calibration import fit_temperature
    rows_val = torch.tensor([p[0] for p in val_pairs], device=device) if val_pairs else None
    if rows_val is not None:
        with torch.no_grad():
            out_val = model(semantic[rows_val], forensic[rows_val])
        # binary head -> 1-D logit + float label; 3-class head -> (N,3) logits
        # + int label. fit_temperature dispatches on shape.
        if args.n_classes == 1:
            temp = fit_temperature(out_val["logit"].cpu(),
                                   labels[rows_val].float().cpu())
        else:
            temp = fit_temperature(out_val["logits"].cpu(),
                                   labels[rows_val].cpu())
        model.temperature.fill_(temp)
        print(f"fitted temperature: {temp:.4f}")

    args.out.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.out / f"{args.tag}.pt"
    torch.save({
        "model": model.state_dict(),
        "config": {
            "backbone": info["backbone"],
            "semantic_dim": semantic.shape[1],
            "forensic_dim": forensic.shape[1],
            "hidden": args.hidden,
            "n_classes": args.n_classes,
            "use_forensic": not args.no_forensic,
            "use_gate": not args.no_gate,
            "lambda_consistency": lam_c,
            "lambda_degradation": lam_d,
            "seed": args.seed,
        },
        "forensic_mu": mu, "forensic_sd": sd,
        "history": history,
        "best_select_score": best_score,
        "parameters": counts,
    }, ckpt_path)
    print(f"saved {ckpt_path} (best select score {best_score:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
