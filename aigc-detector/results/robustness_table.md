# Robustness Evaluation

Split: `test` | checkpoint: `full.pt` | 2255 images x 16 conditions

| Condition | n | Acc | AUC | TPR@1%FPR | ECE | ΔAUC vs clean |
|---|---|---|---|---|---|---|
| clean | 2255 | 0.996 | 1.000 | 1.000 | 0.005 | — |
| jpeg_q90 | 2255 | 0.993 | 1.000 | 0.999 | 0.004 | -0.000 |
| jpeg_q70 | 2255 | 0.989 | 0.999 | 0.991 | 0.008 | -0.000 |
| jpeg_q50 | 2255 | 0.991 | 0.999 | 0.992 | 0.008 | -0.001 |
| jpeg_q30 | 2255 | 0.986 | 0.999 | 0.981 | 0.011 | -0.001 |
| blur_sigma0.5 | 2255 | 0.996 | 1.000 | 1.000 | 0.004 | +0.000 |
| blur_sigma1.0 | 2255 | 0.992 | 1.000 | 0.997 | 0.004 | -0.000 |
| blur_sigma2.0 | 2255 | 0.990 | 1.000 | 0.995 | 0.007 | -0.000 |
| resize_scale0.5 | 2255 | 0.994 | 1.000 | 0.996 | 0.004 | -0.000 |
| resize_scale0.25 | 2255 | 0.989 | 0.999 | 0.988 | 0.006 | -0.000 |
| noise_sigma0.02 | 2255 | 0.986 | 0.999 | 0.983 | 0.009 | -0.001 |
| noise_sigma0.05 | 2255 | 0.984 | 0.999 | 0.973 | 0.010 | -0.001 |
| noise_sigma0.1 | 2255 | 0.983 | 0.999 | 0.968 | 0.009 | -0.001 |
| jitter_brightness0.8_contrast0.8_saturation0.8 | 2255 | 0.992 | 1.000 | 0.996 | 0.007 | -0.000 |
| jitter_brightness1.2_contrast1.2_saturation1.2 | 2255 | 0.991 | 0.999 | 0.995 | 0.006 | -0.000 |
| crop_fraction0.8 | 2255 | 0.996 | 1.000 | 0.997 | 0.004 | -0.000 |

## Headline

- **AUC (clean): 0.9999**
- **Mean AUC drop under transformation: +0.0004**
- Final Score = 0.5*AUC_clean + 0.5*AUC_robust:
  - `mean`: **0.9997**
  - `worst`: **0.9994**
  - `per_family`: **0.9997**

AUC_robust has three readings; `per_family` is the primary figure because it is not biased by how many parameter settings each transform family happens to contribute.
