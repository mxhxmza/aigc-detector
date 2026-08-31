# Error Analysis Note

Split `test` | 5072 images | threshold 0.5

- False positives (real called AI): **20** (0.6% of real images)
- False negatives (AI called real): **16** (0.8% of AI images)

## Error rate by image kind

| Kind | total | false pos | false neg | error rate |
|---|---|---|---|---|
| synthetic | 1200 | 0 | 3 | 0.2% |
| real | 1200 | 11 | 0 | 0.9% |
| tampered | 1200 | 4 | 0 | 0.3% |
| ext_progan | 406 | 0 | 7 | 1.7% |
| lsun | 399 | 5 | 0 | 1.3% |
| ext_dalle3 | 322 | 0 | 6 | 1.9% |
| real_square | 318 | 0 | 0 | 0.0% |
| hard_real | 27 | 0 | 0 | 0.0% |

![false positives](fp_grid.png)

![false negatives](fn_grid.png)

## Interpretation

> Write this by hand after looking at the grids. Questions to answer:
>
> 1. Do the false positives share a visual property (heavy texture,
>    shallow depth of field, smooth skin, low resolution)? Real photos
>    with smooth regions getting flagged means the model reads 'lack of
>    high-frequency detail' as 'synthetic' -- a real artifact signal
>    misfiring, not a bug.
> 2. Do the tampered images fail more often than genuine photos? That
>    is the AI-edited region leaking a detectable signal.
> 3. What is the cost asymmetry? Calling a real photograph synthetic is
>    an accusation against a person; missing a synthetic image is a gap
>    in coverage. State which error the threshold favours.