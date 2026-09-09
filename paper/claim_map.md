# GS-DroneGym Preprint Claim Map

Paper type: systems/tool + dataset-factory technical report

Target venue/style: arXiv preprint, ML/robotics audience

Audience: robotics, robot learning, VLA, sim-to-real, and benchmark-tooling researchers

One-sentence claim:
GS-DroneGym provides an open-source Python stack for drone VLA research that combines quadrotor simulation, waypoint control, 3D Gaussian Splatting rendering, benchmark tasks, and VLA-AN-like synthetic waypoint-supervised dataset generation.

Problem:
Drone VLA research needs environments that connect language-conditioned visual observations to physically meaningful aerial actions while reducing the visual gap between simulation and real deployment spaces.

Why now:
Recent aerial VLA papers motivate waypoint-style actioning, staged data generation, safety correction, and photorealistic 3DGS data, but reusable open infrastructure for generating and evaluating such data remains limited.

Main contribution 1:
A Gymnasium-compatible drone environment with 6-DOF quadrotor dynamics, waypoint and direct action modes, task library, and live visualization.

Evidence:
Implemented package on PyPI as `gs-dronegym==0.3.0`; tests currently pass with `38 passed`.
Local CPU/mock throughput measured at 51.7 environment steps/s on a Ryzen 7 7735HS laptop.

Main contribution 2:
A rendering interface that supports `gsplat` Gaussian scenes and a deterministic CPU mock renderer for testing and CI.

Evidence:
Renderer modules, mock path, import-gated GPU path, and tests are implemented.
Mock renderer measured at 134.5 FPS for 224x224 RGB/depth observations. Built-in public scene handles now resolve to NerfBaselines Gaussian Splatting zip archives, but real `gsplat` throughput is not yet reported because no local GPU run on those large archives is included in the draft.

Main contribution 3:
A synthetic VLA-AN-like dataset factory that exports RGB/depth/state/instruction/expert-waypoint/safety-label data as Parquet shards plus external media.

Evidence:
Implemented CLI tools for generation, preview, validation, and reload through the shared dataset layer.
Small baseline study generated 24 mock episodes / 593 steps, validated the dataset, trained three-epoch BC, measured validation action MSE, and evaluated the checkpoint through the benchmark API.

Strongest baseline or comparison:
Existing VLA systems such as VLA-AN and RaceVLA motivate the problem, but GS-DroneGym is infrastructure rather than a new VLA model.

Known limitations:
No physical drone deployment stack; real 3DGS scene generation depends on external tools; built-in public scene downloads are multi-gigabyte external assets; no bundled small GS-DroneGym-owned Gaussian scene asset yet; current paper should not claim benchmark superiority or sim-to-real deployment success.

Likely reviewer objection:
The system is useful infrastructure but needs more real-scene experiments and baseline training results before making strong empirical claims.

What not to claim:
Do not claim SOTA, exact VLA-AN dataset reproduction, real-drone deployment readiness, or solved sim-to-real transfer.

## Validity of Behavior-Cloning Results Reported Before the Encoder Fix

Every behavior-cloning number produced before the deterministic instruction
encoder landed must be treated as unreliable. The pre-fix encoder bucketed
instruction tokens with Python's built-in `hash`, which is salted per
interpreter, so instruction features were not reproducible across processes or
across runs. Two tiers apply.

**Invalid — training and inference used different encoders.** Any result where
the checkpoint was written by one process and loaded by another, which is the
Appendix workflow of `gs-dronegym-train-bc` followed by
`gs-dronegym-evaluate --policy`. The recorded artifact of this path,
`outputs/paper_smoke_eval.json` (0/3 success, collision rate 1.0), measured a
policy whose instruction features at inference time did not correspond to the
features it was trained on. It is not evidence about the baseline and must not
be cited.

**Irreproducible — internally consistent but not repeatable.** The results in
Tables `tab:smoke` and `tab:small-baselines` were produced by
`paper/run_paper_experiments.py`, which trains and evaluates in a single
process. Within one process `hash` is self-consistent, so training and
inference agreed and these numbers were not corrupted. They were, however, not
reproducible: each rerun drew a fresh hash salt, so `--seed 42` did not pin the
instruction features. Treat them as single-sample observations, not as
reproducible measurements.

**Post-fix rerun.** `paper/run_paper_experiments.py` was rerun on commit
`f97d964` plus the encoder fix, with the published configuration, writing to
`outputs/postfix_baseline_dataset`, `outputs/postfix_baseline_policy.pt`, and
`outputs/postfix_experiment_results.json`:

| Policy | Episodes | Success | Collision | SPL |
|---|---|---|---|---|
| Zero action | 5 | 0/5 | 0/5 | 0.0 |
| Random action | 5 | 0/5 | 3/5 | 0.0 |
| Behavior cloning | 5 | 1/5 | 4/5 | 0.20 |

Training: 393 train examples, 3 epochs, final train loss 0.0170. Offline
imitation error: train MSE 0.0142, val MSE 0.0626.

**Do not read this rerun as an effect of the encoder fix.** It is not a
controlled comparison against the published table. `paper/paper_experiment_results.json`
was written at 2026-04-15 13:13, and the PointNav control fix in commit
`5a9297e` landed at 13:57 the same day, so the published numbers predate that
fix. The same 24-episode configuration now yields 1327 dataset steps instead of
593 because the corrected controller flies longer trajectories. The step-count
change, not the encoder, dominates the difference between the two tables.

What the encoder fix does establish is reproducibility. Evaluating
`outputs/postfix_baseline_policy.pt` in two fresh interpreters with
`PYTHONHASHSEED=1` and `PYTHONHASHSEED=98765` now returns bit-identical core and
benchmark metrics, including `mean_path_length` 18.14236068725586 in both. The
regression test `test_instruction_encoding_is_stable_across_process_hash_seeds`
in `tests/test_bc.py` guards this.

Checkpoints written before the fix carry no `instruction_encoder_version` and
raise a `RuntimeWarning` on load. That includes `outputs/paper_smoke_policy.pt`
and `outputs/paper_baseline_policy.pt`. Retrain rather than reusing them.

## Real Gaussian Renderer: Status and Verified Evidence

Statements about real-scene rendering made before 2026-09-10 are withdrawn.
Three independent defects meant the `gsplat` path had never produced a
geometrically correct image since the renderer shipped in `b21712d`:

1. Gaussian scales were passed to the rasterizer as stored. The PLY format
   holds the logarithm of the scale, so the rasterizer received negative
   scales, for example -2.590 where 0.075 m was intended, giving every
   Gaussian in the scene a degenerate covariance.
2. The camera frame was never converted between the drone body convention
   (x forward, y left, z up) and the vision convention the intrinsics and the
   rasterizer both assume (x right, y down, z forward). A point 8 m directly
   ahead mapped to camera-frame z = -2.06, behind the camera.
3. Depth used render mode `RGB+D`, which is an alpha-weighted accumulation
   rather than metric depth. Treating it as metric and clipping to the near
   plane collapsed the whole image to roughly 0.10 m.

None of these were reachable on the development workstation, where `gsplat`
cannot build its CUDA extension and the renderer silently falls back to the
mock path. The mock renderer reads only the camera height out of the inverse
transform, so the full test suite passed throughout with the camera facing the
wrong way.

**Verified after the fixes.** Rendering a synthetic Gaussian room with real
`gsplat` on an NVIDIA T4 (CUDA 12.4, torch 2.6.0+cu124, 9468 Gaussians) and
comparing rendered depth against the occupancy grid derived from the same
Gaussians, over 851 rays:

| Quantity | Value |
|---|---|
| Median absolute error | 0.0728 m |
| Mean absolute error | 0.1237 m |
| 90th percentile error | 0.1765 m |
| Within one 0.1 m voxel | 60.0% |
| Within two voxels | 93.9% |
| Rendered depth median | 2.527 m |
| Accumulated alpha median | 0.9942 |

Median error is below one voxel, which is the resolution limit of the
comparison: occupancy splats each Gaussian across its own extent, so a ray
registers a hit up to one voxel before the true surface. This validates the
renderer and the derived geometry against each other on the real GPU path. It
is not a claim about reconstructed real-world scenes, coordinate alignment of
third-party scans, or sim-to-real transfer.

Reproduce with `modal run tools/modal_validate_occupancy.py`. The job refuses
to report a result if the renderer falls back to the mock path.
