# G1 stair climbing with step-length control: RL in MuJoCo

**Goal:** a reinforcement-learning pipeline that trains a simulated **Unitree G1 humanoid, model `g1_23dof_rev_1_0`**, to **climb stairs while controlling its step length**. Training runs in **MuJoCo** on a Windows laptop GPU.

Detailed research, with numbers and sources, is in `docs/research/findings.md`. The base code is in `external/unitree_rl_mjlab` (main) and `external/mjlab` (tag v1.2.0).

## Deadline

- Results are needed by **Sat 26 Sep 2026, evening**. Not yet confirmed.
- The main training window is overnight on the laptop.

## Decisions

| Topic | Decision |
|---|---|
| Simulator | MuJoCo, running thousands of robots in parallel on the GPU via **MuJoCo Warp**, through **mjlab** |
| Base code | **`unitreerobotics/unitree_rl_mjlab`**, Unitree's official MuJoCo RL repo. It already has G1 23-DOF Flat and Rough tasks. |
| Robot | `g1_23dof_rev_1_0`. Its MJCF is already in `unitree_rl_mjlab`. |
| Task | **One single policy** handles both **step length** and **step height (stair climbing)**, combined **through reward design**. No separate flat-ground and stairs policies. |
| Perception | No cameras or depth images. A **LiDAR-style height scan** (a grid of ray casts measuring the ground, as in the existing Rough task) is enough. |
| Compute | RTX 4060 Laptop 8 GB, i9-14900HX, 32 GB RAM, Windows 11 Home. Runs **natively on Windows**: no WSL, no Docker. |
| Reward variants | Teammates will train alternative reward functions in parallel, so reward terms must be easy to swap and compare. |
| Out of scope | Depth-camera or vision policies; fallback plans. |

## What `unitree_rl_mjlab` already provides

- **Tasks** `Unitree-G1-23Dof-Rough` and `Unitree-G1-23Dof-Flat`: `src/tasks/velocity/config/g1_23dof/env_cfgs.py`, built on the shared factory `src/tasks/velocity/velocity_env_cfg.py`.
- **Height scan:** a `RayCastSensorCfg` named `terrain_scan` on the pelvis. It casts a 1.6 × 1.0 m grid at 0.1 m spacing (187 rays) and turns with the robot's heading. The policy sees it with noise; the critic sees it clean.
- **Gait clock:** a `phase` observation with a **fixed 0.6 s period**, plus a `foot_gait` reward for alternating foot contacts, half a cycle apart. **Step length ≈ speed × period / 2**, so 0.30 m at 1 m/s. This is the natural hook for step-length control.
- **Other rewards:** `foot_clearance` (fixed 0.10 m swing-height target), `foot_slip`, `soft_landing`, `self_collisions`, `pose` (per-joint posture tolerance), velocity tracking, orientation, action rate, and more. The full list with weights is in findings §3.2.
- **Terrain curriculum** (`terrain_levels_vel`) on mjlab's `ROUGH_TERRAINS_CFG`. **Those stairs only reach 0–10 cm** in mjlab 1.2.0, so the project needs its own stairs-heavy terrain with realistic risers.
- **PPO training via rsl_rl:** 512-256-128 networks, 24 steps per env per iteration, **10,001 max iterations**. That is more than the time budget allows; the benchmark sets the real budget.
- **Scripts:** `scripts/train.py`, `scripts/play.py`, `scripts/visualize_terrain.py`, `scripts/list_envs.py`, and ONNX export. A C++ controller for the real 23-DOF G1 is in `deploy/robots/g1_23dof`. It is Linux-only and outside the simulation pipeline.

## Environment facts (checked 2026-09-25)

- **Version pins:** `mjlab==1.2.0`, `mujoco-warp==3.5.0`, `mujoco==3.5.0`, `warp-lang==1.12.0`, `rsl-rl-lib==5.0.1`, `torch==2.9.0+cu128`. All have Windows builds for Python 3.12.
- **Install torch first, from `https://download.pytorch.org/whl/cu128`.** The default Windows torch is CPU-only.
- **Unpinned installs break things:** they pull mujoco / mujoco-warp 3.11, which is newer than mjlab 1.2.0 was built for.
- **Python:** 3.12.10 is installed. mjlab supports 3.10–3.13; Unitree's guide uses 3.11. `uv` and `conda` are not on PATH.
- **Windows support:** mjlab officially trains on Linux, and its FAQ says Windows "may lag behind Linux". A contributor reported native Windows 11 works, including GPU acceleration (mjlab PR #283). Nothing Linux-only was found in the single-GPU training path. Expect small Windows fixes (paths, file encodings).
- **Speed:** mjlab reports roughly 140k–250k env-steps/s for the G1 on an RTX 5090 with 4096 robots. The laptop is several times slower. A smoke test must measure speed and VRAM before `num_envs` and the iteration budget are set. 8 GB of VRAM may cap `num_envs`; try 4096, then 2048.
- **Laptop:** keep it plugged in, set the power mode to best performance, and close GPU-using apps during training.

## Open design questions

1. **Deadline:** the confirmed hand-in time, and how many hours the laptop can train.
2. **Step-length interface:** how is step length specified?
   - (a) A commanded step length L, with the gait period derived as T = 2L / speed and clipped to a sane range.
   - (b) A commanded gait frequency, with step length following from it.
   - (c) A commanded L, rewarded on the distance between the feet at each touchdown, with free cadence.

   Also: behaviour on stairs. Either the step-length reward is relaxed or masked on stair tiles, or the command is sampled to match the tread depth.
3. **Step height and stairs:**
   - Swing height: adapted to the terrain (clearance measured above the ground under each foot, using the height scan), or commanded?
   - Stair-specific penalties: toes hitting risers, feet placed on stair edges, base height measured relative to the terrain?
4. **Stair dimensions:** riser height range (common stairs are about 15–20 cm; the G1's leg length suggests at most about 18 cm), tread depth (e.g. 25–35 cm), up only or up and down, and the curriculum starting from 0 cm.
5. **Success metrics:** candidates are the share of robots reaching the top platform, the highest terrain level reached, falls per episode, step-length tracking error, velocity error, toe-stubbing count, and energy use.
6. **Code structure:** fork `unitree_rl_mjlab`, or install it as a dependency and keep the project's task in its own package.

## Expected outputs

- A trained policy: checkpoint and ONNX export.
- Task code: commands, observations and reward terms for the combined step-length and stairs task.
- Evaluation results on a fixed stairs course, using the metrics from question 5.
- Simulation videos recorded in play mode.
- Technical documentation: the design, the results, and why it works (or why it doesn't).
- Optional: a sim2sim check of the exported policy in plain CPU MuJoCo on a stairs scene.

## Design constraints

- Runs natively on Windows, as `python ...` commands from this folder.
- Each reward term is its own function, registered with a weight in config. A variant is then a new task ID and doesn't touch shared code.
- One evaluation script scores every variant on the same fixed stairs course.
- Work starts with a smoke test: install → train 20–50 iterations → measure env-steps/s and VRAM. Only then comes the long overnight run.

## Folder layout

- `docs/research/findings.md`: research notes with sources.
- `external/unitree_rl_mjlab/` (main) and `external/mjlab/` (tag v1.2.0): the base code, as read-only copies ignored by git.
- Git is initialized on `main`. Nothing is committed yet.
