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
