# WildFake Reference Benchmark

`techjam-aigc/wildfake-eval-subset` | checkpoint: `full.pt` | **evaluation only — never trained on**

The track's demonstration subset. Which config a number comes from matters:
`default` is separable by image size alone (every COCO real is exactly
200x200, no DALL-E fake is), so its AUC is close to meaningless on its own.
`laion_matched` is the honest comparison — both classes natively >=1024px,
put through one identical downscale.

| Config | n | real/fake | AUC | Balanced acc @0.5 | Balanced acc @best | F1 | TPR@1%FPR | ECE | size-cue AUC |
|---|---|---|---|---|---|---|---|---|---|
| `default` | 13841 | 4998/8843 | **0.9616** | 0.8276 | 0.8993 (t=0.002) | 0.7937 | 0.734 | 0.216 | 1.000 |
| `normalized` | 13841 | 4998/8843 | **0.9398** | 0.7707 | 0.8749 (t=0.001) | 0.7042 | 0.655 | 0.289 | 0.500 |
| `laion_matched` | 7652 | 3826/3826 | **0.9098** | 0.8020 | 0.8480 (t=0.005) | 0.7605 | 0.461 | 0.183 | 0.500 |
| `cross_generator` | 5494 | 1500/3994 | **0.7922** | 0.7089 | 0.7545 (t=0.003) | 0.6117 | 0.342 | 0.399 | 0.500 |

`size-cue AUC` is the no-model rule *real iff exactly 200x200*, scored on the
same rows: 1.000 means the config is fully winnable without looking at the
image, 0.500 means that cue carries no information.

`Balanced acc @best` sweeps the decision threshold. The gap from `@0.5` is
*calibration* drift, not a failure to separate the classes: the model's
temperature was fitted on SID_Set, and this is a different distribution, so
the scores are ranked well but shifted. A high AUC with a much lower
balanced accuracy at 0.5 means the operating point is wrong, not the model.

## Per generator — `default`

Each generator scored against the same real images.

| Generator | n | AUC | Recall @0.5 | mean p(AI) |
|---|---|---|---|---|
| dalle3_advanced | 8843 | 0.9616 | 0.660 | 0.659 |

## Per generator — `normalized`

Each generator scored against the same real images.

| Generator | n | AUC | Recall @0.5 | mean p(AI) |
|---|---|---|---|---|
| dalle3_advanced | 8843 | 0.9398 | 0.544 | 0.546 |

## Per generator — `laion_matched`

Each generator scored against the same real images.

| Generator | n | AUC | Recall @0.5 | mean p(AI) |
|---|---|---|---|---|
| dalle3_advanced | 3826 | 0.9098 | 0.629 | 0.625 |

## Per generator — `cross_generator`

Each generator scored against the same real images.

| Generator | n | AUC | Recall @0.5 | mean p(AI) |
|---|---|---|---|---|
| dalle3 | 1000 | 0.9233 | 0.666 | 0.666 |
| midjourney_v5 | 999 | 0.9217 | 0.648 | 0.648 |
| sdxl | 1000 | 0.8227 | 0.445 | 0.442 |
| gigagan | 995 | 0.4998 | 0.020 | 0.025 |

## False positives by real source

| Config | Source | n | flagged as AI | mean p(AI) |
|---|---|---|---|---|
| `default` | coco_val2017 | 4998 | 0.46% | 0.006 |
| `normalized` | coco_val2017 | 4998 | 0.28% | 0.004 |
| `laion_matched` | laion5b | 3826 | 2.46% | 0.028 |
| `cross_generator` | laion5b | 1500 | 2.73% | 0.029 |
