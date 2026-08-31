# WildFake Reference Benchmark

`techjam-aigc/wildfake-eval-subset` | checkpoint: `full.pt` | **evaluation only — never trained on**

The track's demonstration subset. Which config a number comes from matters:
`default` is separable by image size alone (every COCO real is exactly
200x200, no DALL-E fake is), so its AUC is close to meaningless on its own.
`laion_matched` is the honest comparison — both classes natively >=1024px,
put through one identical downscale.

| Config | n | real/fake | AUC | Balanced acc @0.5 | Balanced acc @best | F1 | TPR@1%FPR | ECE | size-cue AUC |
|---|---|---|---|---|---|---|---|---|---|
| `default` | 13841 | 4998/8843 | **0.9985** | 0.9762 | 0.9808 (t=0.055) | 0.9777 | 0.965 | 0.022 | 1.000 |
| `normalized` | 13841 | 4998/8843 | **0.9974** | 0.9652 | 0.9736 (t=0.041) | 0.9662 | 0.941 | 0.036 | 0.500 |
| `laion_matched` | 7652 | 3826/3826 | **0.9889** | 0.9476 | 0.9502 (t=0.714) | 0.9480 | 0.831 | 0.036 | 0.500 |
| `cross_generator` | 5494 | 1500/3994 | **0.8078** | 0.7642 | 0.7724 (t=0.026) | 0.7255 | 0.463 | 0.301 | 0.500 |

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
| dalle3_advanced | 8843 | 0.9985 | 0.961 | 0.959 |

## Per generator — `normalized`

Each generator scored against the same real images.

| Generator | n | AUC | Recall @0.5 | mean p(AI) |
|---|---|---|---|---|
| dalle3_advanced | 8843 | 0.9974 | 0.939 | 0.937 |

## Per generator — `laion_matched`

Each generator scored against the same real images.

| Generator | n | AUC | Recall @0.5 | mean p(AI) |
|---|---|---|---|---|
| dalle3_advanced | 3826 | 0.9889 | 0.955 | 0.953 |

## Per generator — `cross_generator`

Each generator scored against the same real images.

| Generator | n | AUC | Recall @0.5 | mean p(AI) |
|---|---|---|---|---|
| dalle3 | 1000 | 0.9862 | 0.931 | 0.930 |
| midjourney_v5 | 999 | 0.9664 | 0.870 | 0.867 |
| sdxl | 1000 | 0.8228 | 0.482 | 0.488 |
| gigagan | 995 | 0.4543 | 0.036 | 0.047 |

## False positives by real source

| Config | Source | n | flagged as AI | mean p(AI) |
|---|---|---|---|---|
| `default` | coco_val2017 | 4998 | 0.86% | 0.012 |
| `normalized` | coco_val2017 | 4998 | 0.88% | 0.012 |
| `laion_matched` | laion5b | 3826 | 6.01% | 0.067 |
| `cross_generator` | laion5b | 1500 | 5.20% | 0.060 |
