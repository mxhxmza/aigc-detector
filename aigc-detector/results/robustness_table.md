# Robustness Evaluation

Split: `test` | checkpoint: `full.pt` | 1600 images x 16 conditions

| Condition | n | Acc | AUC | TPR@1%FPR | ECE | ΔAUC vs clean |
|---|---|---|---|---|---|---|
| clean | 1600 | 0.999 | 1.000 | 1.000 | 0.003 | — |
| jpeg_q90 | 1600 | 0.997 | 1.000 | 1.000 | 0.004 | -0.000 |
| jpeg_q70 | 1600 | 0.992 | 1.000 | 0.998 | 0.005 | -0.000 |
| jpeg_q50 | 1600 | 0.990 | 1.000 | 0.991 | 0.008 | -0.000 |
| jpeg_q30 | 1600 | 0.989 | 1.000 | 0.993 | 0.006 | -0.000 |
| blur_sigma0.5 | 1600 | 0.998 | 1.000 | 1.000 | 0.003 | +0.000 |
| blur_sigma1.0 | 1600 | 0.998 | 1.000 | 1.000 | 0.004 | -0.000 |
| blur_sigma2.0 | 1600 | 0.991 | 1.000 | 0.995 | 0.006 | -0.000 |
| resize_scale0.5 | 1600 | 0.998 | 1.000 | 1.000 | 0.003 | -0.000 |
| resize_scale0.25 | 1600 | 0.991 | 1.000 | 0.990 | 0.008 | -0.000 |
| noise_sigma0.02 | 1600 | 0.989 | 1.000 | 0.994 | 0.006 | -0.000 |
| noise_sigma0.05 | 1600 | 0.989 | 0.999 | 0.985 | 0.009 | -0.001 |
| noise_sigma0.1 | 1600 | 0.991 | 0.999 | 0.990 | 0.009 | -0.001 |
| jitter_brightness0.8_contrast0.8_saturation0.8 | 1600 | 0.991 | 1.000 | 0.995 | 0.010 | -0.000 |
| jitter_brightness1.2_contrast1.2_saturation1.2 | 1600 | 0.994 | 1.000 | 0.995 | 0.004 | -0.000 |
| crop_fraction0.8 | 1600 | 0.996 | 1.000 | 1.000 | 0.003 | -0.000 |

## Headline

- **AUC (clean): 1.0000**
- **Mean AUC drop under transformation: +0.0002**
- Final Score = 0.5*AUC_clean + 0.5*AUC_robust:
  - `mean`: **0.9999**
  - `worst`: **0.9997**
  - `per_family`: **0.9998**

AUC_robust has three readings; `per_family` is the primary figure because it is not biased by how many parameter settings each transform family happens to contribute.
