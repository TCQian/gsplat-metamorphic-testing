"""The five metamorphic relations. Each `run_mrN(scene, cameras, ...)` renders the same scene two
ways through `gsplat.rasterization`, over every camera in one batched call, and returns the image
differences (or an error string, since a crash is itself a finding).

- MR1 `mr1_packed`: packed vs unpacked intermediate layout
- MR2 `mr2_batch`: all cameras in one call vs one call per camera
- MR3 `mr3_covariance`: precomputed covariance vs scale + rotation (planted fault: `xyzw_bug`)
- MR4 `mr4_world_scale`: scene, camera translation and near/far scaled by k (planted fault: `no_translation`)
- MR5 `mr5_permutation`: Gaussians shuffled (planted fault: `misaligned`)
"""
