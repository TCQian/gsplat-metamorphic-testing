"""Library for metamorphic testing of gsplat's forward RGB rasterizer.

- `scene`: random Gaussian scenes and cameras, and the stress modes (scene types)
- `metrics`: image comparison (D_max, D_MAE, SSIM), tolerances and verdicts
- `relations`: the five metamorphic relations MR1-MR5, with planted faults for MR3-MR5
- `harness`: builds one scene and runs every relation on it (one CSV row)
- `sampling`: the per-scene random configuration of the large-scale run, and scene replay
- `repro_issues`: reproductions of gsplat issues #764 and #630
"""
