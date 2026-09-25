# Research findings

All facts were checked against the linked sources on 2026-09-25.

## 1. Training machine

| Item | Value |
|---|---|
| OS | Windows 11 Home 10.0.26200 |
| GPU | NVIDIA GeForce RTX 4060 Laptop, **8 GB VRAM**, 65 W power cap, driver 610.88 (CUDA 13.3) |
| CPU | Intel Core i9-14900HX, 24 cores / 32 threads |
| RAM | 32 GB |
| Disk | C: 294 GB free, D: 320 GB free |
| Tools | Python 3.12.10 on PATH, Git 2.51.1, VS Code. Not on PATH: uv, conda (Anaconda folders exist). No WSL, no Docker. |

The GPU also drives the display, and other apps use it even when idle. Close GPU-using apps during training, keep the laptop plugged in, and use the best-performance power mode.

## 2. Robot: Unitree G1, model `g1_23dof_rev_1_0`

- **Degrees of freedom:** 12 legs (per leg: hip pitch, hip roll, hip yaw, knee, ankle pitch, ankle roll), 1 waist yaw, 10 arms (per arm: shoulder pitch, roll, yaw, elbow, wrist roll). No hands.
- **Unitree's model table** ([source](https://github.com/unitreerobotics/unitree_ros/blob/master/robots/g1_description/README.md)): mode 4, hip gear ratios 14.3 / 22.5, no wrist motor, status "Up-to-date".
- **MJCF:** `src/assets/robots/unitree_g1/xmls/g1_23dof.xml` in `unitree_rl_mjlab` (declares `<mujoco model="g1_23dof_rev_1_0">`), scene file `scene_g1_23dof.xml`.
- **Joint order in the MJCF:** `left_hip_pitch_joint, left_hip_roll_joint, left_hip_yaw_joint, left_knee_joint, left_ankle_pitch_joint, left_ankle_roll_joint`, the same six for the right leg, `waist_yaw_joint`, then `left_shoulder_pitch_joint, left_shoulder_roll_joint, left_shoulder_yaw_joint, left_elbow_joint, left_wrist_roll_joint`, and the same five for the right arm.
- **Feet:** 7 collision capsules per foot (`left_foot1_collision` … `left_foot7_collision`, same for the right), and sites `left_foot` / `right_foot`.
- **Actuators** (`src/assets/robots/unitree_g1/g1_23dof_constants.py`): position actuators. Gains come from each motor's reflected inertia, tuned to a 10 Hz natural frequency with damping ratio 2.

  | Motor | Joints | Torque limit | Speed limit |
  |---|---|---|---|
  | 5020 | shoulders, elbow, wrist roll | 25 N·m | 37 rad/s |
  | 7520-14 | hip pitch, hip yaw, waist yaw | 88 N·m | 32 rad/s |
  | 7520-22 | hip roll, knee | 139 N·m | 20 rad/s |
  | 2 × 5020 | ankles | 50 N·m | — |

  The action scale per joint is 0.25 × torque limit / stiffness.
- **Starting pose** (`HOME_KEYFRAME`): base height 0.8 m; hip pitch −0.1, knee 0.3, ankle pitch −0.2, shoulder pitch 0.35, elbow 0.87, shoulder roll ±0.18 (radians). Collision config `FULL_COLLISION`: feet use condim 3, priority 1, friction 0.6.
- **Sensors on the real robot's head:** Livox MID-360 LiDAR (360° horizontal, 59° vertical field of view) and Intel RealSense D435i depth camera. Onboard computer: Jetson Orin NX.
- **Motor order on the real robot** (23-DOF `LowCmd.motor_cmd`, [source](https://support.unitree.com/home/en/G1_developer/joint_motor_sequence)): 29 slots.
  - 0–11: legs, left first (hip pitch, roll, yaw, knee, ankle pitch, ankle roll), then right
  - 12: waist yaw
  - 13–14: empty
  - 15–19: left arm (shoulder pitch, roll, yaw, elbow, wrist roll)
  - 20–21: empty
  - 22–26: right arm
  - 27–28: empty

## 3. Stack

### 3.1 `unitree_rl_mjlab`

- **Repo:** <https://github.com/unitreerobotics/unitree_rl_mjlab>, Unitree's official MuJoCo RL repo, Apache-2.0. Last commit 2026-04-13 (`1425b15`, "Fix the warnings during rough-terrain training"). Local copy: `external/unitree_rl_mjlab`.
- **Setup** ([guide](https://github.com/unitreerobotics/unitree_rl_mjlab/blob/main/doc/setup_en.md)): recommends Ubuntu 22.04 and NVIDIA driver 550+, conda with Python 3.11, then `pip install -e .`. Its apt packages are only for the C++ simulator and deploy code.
- **Version pins** in `setup.py`: `mjlab==1.2.0`, `mujoco-warp==3.5.0`.
- **G1 23-DOF tasks:** `Unitree-G1-23Dof-Rough`, `Unitree-G1-23Dof-Flat`.
- **Commands:**
  - Train: `python scripts/train.py Unitree-G1-23Dof-Rough --env.scene.num-envs=4096`
  - Play: `python scripts/play.py <task> --checkpoint_file=logs/rsl_rl/<experiment>/<run>/model_<n>.pt`
  - Also `scripts/visualize_terrain.py` and `scripts/list_envs.py`.
- **Code layout:**
  - `src/tasks/velocity/velocity_env_cfg.py`: shared task factory
  - `src/tasks/velocity/config/g1_23dof/{__init__,env_cfgs,rl_cfg}.py`: the 23-DOF tasks
  - `src/tasks/velocity/mdp/`: rewards, observations, curriculums, terminations, velocity command
  - `src/tasks/velocity/rl/runner.py`: the training runner

### 3.2 The `Unitree-G1-23Dof-Rough` task

- **Simulation:** 0.005 s physics step (200 Hz), policy every 4 steps (50 Hz), 20 s episodes. Solver iterations 10, line-search iterations 20. The Rough task adds CCD iterations 500, `nconmax` 48, `contact_sensor_maxmatch` 500.
- **Terrain:** mjlab's `ROUGH_TERRAINS_CFG` (v1.2.0), a grid of 10 × 20 tiles of 8 × 8 m with a 20 m border.

  | Terrain type | Share | Parameters |
  |---|---|---|
  | flat | 20% | |
  | pyramid stairs (up) | 20% | risers 0.0–0.1 m, treads 0.3 m, 3 m top platform |
  | inverted pyramid stairs (down) | 20% | same |
  | pyramid slope | 10% | |
  | inverted slope | 10% | |
  | random rough | 10% | |
  | wave | 10% | |

  The curriculum (`terrain_levels_vel`) is on, starting at level 5 at most. **The stairs only reach 10 cm, so the project must raise them.**
- **Height scan (the "LiDAR"):** a `RayCastSensorCfg` named `terrain_scan`, attached to the pelvis and turning with the robot's heading. It casts a 1.6 × 1.0 m grid at 0.1 m spacing (17 × 11 = 187 rays), up to 5 m. The policy gets it with ±0.1 noise; the critic gets it clean.
- **Policy (actor) observations:** IMU angular velocity, projected gravity, the velocity command, a gait-clock `phase` (fixed 0.6 s period), joint positions relative to default, joint velocities, the last action, and the height scan.
- **Critic-only extras:** IMU linear velocity, the clean height scan, foot height, foot air time, foot contact, and foot contact forces.
- **Actions:** position targets for all 23 joints, with per-joint scale and default offset.
- **Commands:** velocity forward, sideways and turning, plus a heading target. Resampled every 3–8 s, with 5% of robots told to stand still. A command curriculum starts at forward −0.5 to 1.0 m/s, sideways ±0.5 m/s, turning ±1 rad/s. After 5000 × 24 steps it widens to forward −1 to 2 m/s and sideways ±1 m/s.
- **Rewards:**

  | Term | Weight | Notes |
  |---|---|---|
  | `track_linear_velocity` | 1.0 | std 0.5 |
  | `track_angular_velocity` | 1.0 | |
  | `body_orientation_l2` | −1.0 | on the torso |
  | `pose` | 1.0 | per-joint posture tolerance, separate for standing, walking and running |
  | `body_ang_vel` | −0.05 | |
  | `angular_momentum` | −0.025 | |
  | `is_terminated` | −200 | |
  | `joint_acc_l2` | −2.5e-7 | |
  | `joint_pos_limits` | −10 | |
  | `action_rate_l2` | −0.05 | |
  | `foot_gait` | 0.5 | 0.6 s period, feet half a cycle apart, stance threshold 0.56 |
  | `foot_clearance` | −1.0 | fixed 0.10 m swing-height target |
  | `foot_slip` | −0.25 | |
  | `soft_landing` | −1e-3 | |
  | `stand_still` | −1.0 | |
  | `self_collisions` | −1.0 | |
- **Episode ends:** on time-out, or when the robot tilts more than 70°.
- **Randomization:** start position ±0.5 m with any heading; pushes every 5–6 s; foot friction 0.3–1.6; joint encoder bias ±0.015 rad; torso centre-of-mass offset ±5 cm.
- **Play mode:** endless episodes, no noise or pushes, randomized terrain on a 5 × 5 tile grid.
- **PPO settings** (rsl_rl 5.0.1 through mjlab): actor and critic networks 512-256-128 with ELU and observation normalization. Learning rate 1e-3 with an adaptive schedule (KL target 0.01), gamma 0.99, lambda 0.95, clip 0.2, entropy 0.01, 5 epochs × 4 mini-batches, 24 steps per env per iteration. **10,001 max iterations**, saving every 100. Experiment name `g1_23dof_velocity`.
- **Hooks for the project's task:**
  - The gait clock gives step length ≈ speed × period / 2, e.g. 0.30 m at 1 m/s with the 0.6 s period. So a commanded period or step length is a small change to `phase` and `foot_gait`.
  - `foot_clearance` has a fixed target height. A clearance measured relative to the ground under each foot would help on stairs.
  - mjlab's command manager accepts extra command terms, e.g. a gait or step command next to the velocity command.

### 3.3 mjlab

- **What it is:** <https://github.com/mujocolab/mjlab>. A manager-based task API (managers for observations, rewards, events, curricula) on top of MuJoCo Warp, using PyTorch and rsl_rl. **v1.2.0 released 2026-03-06.** Local copy of the v1.2.0 tag: `external/mjlab`.
- **Official platform support** ([install docs](https://mujocolab.github.io/mjlab/main/source/installation.html)): training on Linux with an NVIDIA GPU (CUDA 12.4+ recommended); evaluation on Linux, macOS, or Windows via WSL.
- **What the FAQ says about Windows:** "We have performed preliminary testing on **Windows** and **WSL**, but some workflows are not guaranteed to be stable… Windows support may lag behind Linux."
- **Evidence it works on native Windows:** [PR #283](https://github.com/mujocolab/mjlab/pull/283), merged 2025-11. Its author ran examples and tests on Windows 11 and WSL: "core functionality seems to work on both platforms, including hardware acceleration". Only the `nan_guard` tests failed on Windows, due to file permissions. [PR #132](https://github.com/mujocolab/mjlab/pull/132) fixed Windows path handling.
- **No Linux-only blockers found:**
  - v1.2.0's `__init__.py` does not force a Linux rendering backend (no `MUJOCO_GL=egl`).
  - GPU selection only reads `CUDA_VISIBLE_DEVICES`.
  - The multi-GPU launcher `torchrunx` is imported only for multi-GPU runs.
  - The `required-environments` setting in mjlab's `pyproject.toml` affects `uv sync` inside the mjlab repo, not `pip install mjlab`.
- **Dependencies of mjlab 1.2.0:** torch ≥ 2.7, warp-lang ≥ 1.12, mujoco-warp ≥ 3.5, mujoco ≥ 3.5, rsl-rl-lib == 5.0.1, tyro, viser, mediapy, imageio-ffmpeg, tensordict, tensorboard, onnxscript, wandb, trimesh, torchrunx.
- **Versions in mjlab 1.2.0's lock file:** mujoco 3.5.1.dev, mujoco-warp 3.5.0.2, warp-lang 1.12.0, torch 2.9.0, numpy 2.2.6.
- **Windows builds confirmed for Python 3.12:**

  | Package | Build |
  |---|---|
  | `mujoco==3.5.0` | win_amd64 |
  | `mujoco-warp==3.5.0` | pure Python |
  | `warp-lang==1.12.0` | win_amd64 |
  | `rsl-rl-lib==5.0.1` | pure Python |
  | `torch==2.9.0+cu128` | win_amd64, from <https://download.pytorch.org/whl/cu128> |
- **Install warnings:**
  - The default `pip install torch` on Windows is CPU-only. Install torch from the cu128 index first.
  - Unpinned installs pull mujoco / mujoco-warp 3.11, which is newer than mjlab 1.2.0 was built against. Pin them.

### 3.4 MuJoCo Warp

- **What it is:** <https://github.com/google-deepmind/mujoco_warp>. MuJoCo rewritten in NVIDIA Warp to run on the GPU.
- **Platforms:** Windows and Linux on x86-64. It needs an NVIDIA GPU; a CPU mode exists for debugging.
- **Supported:** heightfields, meshes, ray casting, batch rendering.
- **Not supported:** the IMPLICITFAST integrator, PGS and noslip solvers, plugins.

## 4. Speed estimates (to be measured)

- **Published speed:** [mjlab's nightly benchmark](https://mujocolab.github.io/mjlab/nightly/) runs 4096 envs on an RTX 5090. `Mjlab-Tracking-Flat-Unitree-G1` reached ≈ 249k env-steps/s (2026-09-24). `Mjlab-Velocity-Flat-Unitree-G1` reached ≈ 140k env-steps/s (Dec 2025).
- **Rough guess for the laptop:** the RTX 4060 Laptop (65 W, 8 GB) is several times slower, perhaps 5–10×, giving about 15–50k env-steps/s on flat ground. Stairs are slower again, because of ray casting and extra contacts.
- **What that means per iteration:** one PPO iteration at 4096 envs × 24 steps is about 98k env-steps. A few thousand iterations takes hours.
- **Benchmark before committing.** Measure env-steps/s and VRAM in a smoke test before setting `num_envs` and the iteration budget. 8 GB may cap `num_envs`; try 4096, then 2048.

## 5. Path to the real robot

- **Deploy code:** `unitree_rl_mjlab` exports ONNX policies and includes a C++ controller for the 23-DOF G1 in `deploy/robots/g1_23dof`: `config/config.yaml`, `config/policy/velocity/v0/params/deploy.yaml`, and `src/State_RLBase.cpp`.
- **It is Linux-only:** it builds with CMake against `unitree_sdk2` on Linux, with ONNX Runtime included for Linux x64 and aarch64.
  - Sim2sim: run `./simulate/build/unitree_mujoco`, then `./g1_ctrl --network=lo`.
  - Real robot: `./g1_ctrl --network=<interface>`.
- **Outside the simulation pipeline.** It is the next step after it.
- **A sim2sim check that works on Windows:** run the exported policy in plain CPU MuJoCo (the `mujoco` Python package) on the same `g1_23dof_rev_1_0` model with a stairs scene.
- **Perception on hardware:** build a height map from the MID-360 point cloud (elevation mapping). That gives the same observation as the training height scan.

## 6. Sources

- Unitree G1 docs: [joint motor sequence](https://support.unitree.com/home/en/G1_developer/joint_motor_sequence), [about G1](https://support.unitree.com/home/en/G1_developer/about_G1), [LiDAR routine](https://support.unitree.com/home/en/G1_developer/lidar_Instructions)
- Unitree repos: [unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab), [unitree_ros G1 model table](https://github.com/unitreerobotics/unitree_ros/blob/master/robots/g1_description/README.md)
- mjlab: [repo](https://github.com/mujocolab/mjlab), [installation](https://mujocolab.github.io/mjlab/main/source/installation.html), [FAQ](https://github.com/mujocolab/mjlab/blob/main/docs/source/faq.rst), [nightly benchmark](https://mujocolab.github.io/mjlab/nightly/), [paper](https://arxiv.org/abs/2601.22074)
- MuJoCo Warp: [repo](https://github.com/google-deepmind/mujoco_warp), [docs](https://mujoco.readthedocs.io/en/latest/mjwarp/)
