"""Robust detection of AI-generated images under real-world transformations.

TikTok TechJam 2026, Problem Statement #5.

Package layout:
    data/        the manifest (single source of truth: image_path, label, split)
    transforms/  the six specified degradations + random sampler
    features/    frozen-backbone extraction, hand-designed forensic features
    models/      two-branch detector, degradation gate, calibration
    metrics      AUC / ECE / TPR@FPR and the final-score formula
"""

__version__ = "0.1.0"
