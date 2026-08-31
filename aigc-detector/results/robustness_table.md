# Robustness Evaluation

Split: `test` | checkpoint: `full.pt` | 3627 images x 16 conditions

| Condition | n | Acc | AUC | TPR@1%FPR | ECE | ΔAUC vs clean |
|---|---|---|---|---|---|---|
| clean | 3627 | 0.996 | 1.000 | 0.999 | 0.004 | — |
| jpeg_q90 | 3627 | 0.995 | 1.000 | 0.999 | 0.003 | -0.000 |
| jpeg_q70 | 3627 | 0.992 | 1.000 | 0.996 | 0.004 | -0.000 |
| jpeg_q50 | 3627 | 0.988 | 1.000 | 0.989 | 0.010 | -0.000 |
| jpeg_q30 | 3627 | 0.988 | 0.999 | 0.988 | 0.008 | -0.000 |
| blur_sigma0.5 | 3627 | 0.996 | 1.000 | 0.999 | 0.003 | +0.000 |
| blur_sigma1.0 | 3627 | 0.996 | 1.000 | 0.999 | 0.003 | -0.000 |
| blur_sigma2.0 | 3627 | 0.993 | 1.000 | 0.993 | 0.004 | -0.000 |
| resize_scale0.5 | 3627 | 0.996 | 1.000 | 0.999 | 0.005 | -0.000 |
| resize_scale0.25 | 3627 | 0.990 | 1.000 | 0.992 | 0.008 | -0.000 |
| noise_sigma0.02 | 3627 | 0.991 | 1.000 | 0.993 | 0.006 | -0.000 |
| noise_sigma0.05 | 3627 | 0.989 | 0.999 | 0.987 | 0.011 | -0.001 |
| noise_sigma0.1 | 3627 | 0.988 | 0.999 | 0.988 | 0.011 | -0.001 |
| jitter_brightness0.8_contrast0.8_saturation0.8 | 3627 | 0.990 | 1.000 | 0.996 | 0.012 | -0.000 |
| jitter_brightness1.2_contrast1.2_saturation1.2 | 3627 | 0.993 | 1.000 | 0.995 | 0.006 | -0.000 |
| crop_fraction0.8 | 3627 | 0.996 | 1.000 | 1.000 | 0.003 | -0.000 |

## Headline

- **AUC (clean): 1.0000**
- **Mean AUC drop under transformation: +0.0002**
- Final Score = 0.5*AUC_clean + 0.5*AUC_robust:
  - `mean`: **0.9998**
  - `worst`: **0.9997**
  - `per_family`: **0.9998**

AUC_robust has three readings; `per_family` is the primary figure because it is not biased by how many parameter settings each transform family happens to contribute.
