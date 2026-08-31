# Robustness Evaluation

Split: `test` | checkpoint: `full.pt` | 2000 images x 16 conditions

| Condition | n | Acc | AUC | TPR@1%FPR | ECE | ΔAUC vs clean |
|---|---|---|---|---|---|---|
| clean | 2000 | 0.990 | 0.999 | 0.997 | 0.008 | — |
| jpeg_q90 | 2000 | 0.988 | 0.999 | 0.986 | 0.009 | -0.000 |
| jpeg_q70 | 2000 | 0.981 | 0.999 | 0.971 | 0.013 | -0.001 |
| jpeg_q50 | 2000 | 0.977 | 0.998 | 0.968 | 0.015 | -0.001 |
| jpeg_q30 | 2000 | 0.974 | 0.997 | 0.939 | 0.017 | -0.003 |
| blur_sigma0.5 | 2000 | 0.991 | 0.999 | 0.994 | 0.007 | -0.000 |
| blur_sigma1.0 | 2000 | 0.988 | 0.999 | 0.986 | 0.007 | -0.000 |
| blur_sigma2.0 | 2000 | 0.973 | 0.997 | 0.925 | 0.015 | -0.003 |
| resize_scale0.5 | 2000 | 0.987 | 0.999 | 0.985 | 0.010 | -0.001 |
| resize_scale0.25 | 2000 | 0.973 | 0.996 | 0.945 | 0.018 | -0.003 |
| noise_sigma0.02 | 2000 | 0.980 | 0.998 | 0.964 | 0.012 | -0.002 |
| noise_sigma0.05 | 2000 | 0.971 | 0.996 | 0.941 | 0.020 | -0.003 |
| noise_sigma0.1 | 2000 | 0.964 | 0.993 | 0.896 | 0.026 | -0.007 |
| jitter_brightness0.8_contrast0.8_saturation0.8 | 2000 | 0.989 | 0.999 | 0.989 | 0.007 | -0.000 |
| jitter_brightness1.2_contrast1.2_saturation1.2 | 2000 | 0.986 | 0.999 | 0.982 | 0.009 | -0.001 |
| crop_fraction0.8 | 2000 | 0.993 | 0.999 | 0.992 | 0.006 | -0.000 |

## Headline

- **AUC (clean): 0.9993**
- **Mean AUC drop under transformation: +0.0017**
- Final Score = 0.5*AUC_clean + 0.5*AUC_robust:
  - `mean`: **0.9985**
  - `worst`: **0.9961**
  - `per_family`: **0.9980**

AUC_robust has three readings; `per_family` is the primary figure because it is not biased by how many parameter settings each transform family happens to contribute.
