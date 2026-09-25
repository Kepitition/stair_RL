# Shared environment for G1 stair climbing: design

Date: 2026-09-25. Status: approved in design review; spec awaiting review.

## 1. Purpose and scope

The team trains different reward functions, perception setups and network architectures in parallel. They all need the **same scene**, so that results can be compared: the same robot, terrain, sensors and physics. This project builds that shared base as a repo that teammates clone, install with one command, and extend without editing shared code.

**In scope**

1. Install: pinned requirements and Windows fixes.
2. Scene: robot, stairs terrain, sensors and physics settings.
3. Plumbing: registering a task per variant, plus the train, play, scene-viewer and task-list scripts.
4. One runnable baseline task (`G1-Stairs-Baseline`). It is a template and a sanity check, not a design decision.
5. Fast tests, a training acceptance check, and a README.

**Out of scope (decided later, on top of this base)**

- The step-length command and how the policy uses it
- Reward design
- Perception variants
- The fixed evaluation course and its metrics
- Sim2sim

## 2. Measured facts behind the design

These come from a smoke-test spike on the target laptop (RTX 4060 Laptop, 8 GB, Windows 11, native Python 3.12.10) on 2026-09-25.

| Configuration @ 4096 envs | Result | Env-steps/s | Peak GPU memory |
|---|---|---|---|
| Unitree `Unitree-G1-23Dof-Rough`, stock (`ccd_iterations=500`) | out of memory at startup | — | — |
| Same, `ccd_iterations=50` | ok | 17.1k | 6.8 GB |
| **Box stairs (risers up to 20 cm) + flat** | **ok** | **14.7k (6.7 s/iteration)** | **6.7 GB** |
| Box stairs + box-grid rough tile | out of memory at startup | — | — |
| Box stairs + heightfield rough tile, `ccd_iterations=50` | CUDA launch failure after iteration 0 | 13.2k | 7.7 GB |

What drives memory:

- **Heightfield tiles** send capsule-vs-heightfield contacts through MuJoCo Warp's general convex collider. Its buffer grows with `ccd_iterations` and reaches 1.97 GB at 500. All robot collision geoms are capsules or spheres, and capsule-vs-box contacts use cheap dedicated routines, so box-only terrain never runs the general collider.
- **Geom count** drives the ray-cast sensor's acceleration structure, which is copied per world. The box-grid rough tile makes 13,302 geoms, and that structure alone takes 1.96 GB.
- Anything peaking above about 7.6 GB fails, because the display also uses the GPU.

At 14.7k env-steps/s, one PPO iteration (4096 × 24 steps) takes 6.7 s, so about 540 iterations per hour.

Install findings:

- rsl-rl-lib pulls in `torchvision`. Left unpinned, it installs torchvision 0.29, which silently replaces CUDA torch with CPU torch 2.14. It must be pinned to `torchvision==0.24.0` from the cu128 index.
- mjlab 1.2.0 imports `scipy` without declaring it, so `scipy` must be listed explicitly.

## 3. Repo layout

```
g1-stairs-rl/
  pyproject.toml            package g1_stairs (setuptools), Python >= 3.10
  requirements.txt          one-command install, including "-e ."
  requirements.lock.txt     pip freeze from the passing acceptance run
  NOTICE                    attribution for code and assets from unitree_rl_mjlab (Apache-2.0)
  LICENSES/                 copy of unitree_rl_mjlab's Apache-2.0 license
  .gitattributes            marks meshes as binary
  scripts/
    train.py  play.py  list_envs.py  view_scene.py
  g1_stairs/
    __init__.py             package root (REPO_ROOT); scripts import g1_stairs.tasks to register tasks
    runtime.py              script helpers: log paths, render backend, viewer choice, dummy policies
    runner.py               Unitree's VelocityOnPolicyRunner (ONNX export on save)
    assets/g1_23dof/        g1_23dof.xml, meshes used by it, constants.py
    scene/
      robot.py              robot entity config and action scale
      terrain.py            STAIRS_TERRAIN_CFG plus a play-mode variant
      sensors.py            the five scene sensors
      scene.py              make_scene_cfg(), SIM_CFG, memory-budget constants
    mdp/                    shared term library (Unitree's velocity mdp, copied)
    tasks/
      __init__.py           auto-imports every subpackage (no central list)
      baseline/
        __init__.py         registers G1-Stairs-Baseline
        env_cfg.py          baseline_env_cfg(play: bool = False)
        rl_cfg.py           baseline_ppo_cfg()
  tests/
  docs/
```

## 4. Install

`requirements.txt`:

```
--extra-index-url https://download.pytorch.org/whl/cu128
torch==2.9.0+cu128
torchvision==0.24.0+cu128
mjlab==1.2.0
mujoco==3.5.0
mujoco-warp==3.5.0
warp-lang==1.12.0
rsl-rl-lib==5.0.1
scipy
pytest
-e .
```

Steps, run from the repo root:

1. `python -m venv .venv`
2. `.venv\Scripts\activate`
3. `pip install -r requirements.txt`

Linux works the same, with `source .venv/bin/activate`. `pyproject.toml` lists the same pins without the torch index, so `pip install -e .` alone never picks the CPU torch over an existing CUDA install. `requirements.lock.txt` records the exact versions from the acceptance run.

## 5. Scene (fixed for every variant)

### 5.1 Robot

- **Model:** `g1_23dof_rev_1_0`, copied unchanged from `unitree_rl_mjlab`: MJCF, meshes, actuator gains (10 Hz, damping ratio 2), `HOME_KEYFRAME` and `FULL_COLLISION`.
- **Robot interface:** position targets for all 23 joints at 50 Hz, with Unitree's per-joint action scale (`0.25 × torque limit / stiffness`) and the default pose as offset. Every policy trained on this base therefore uses the same action space.

### 5.2 Terrain: `STAIRS_TERRAIN_CFG`

- **Grid:** 8 × 8 m tiles, 10 rows × 20 columns, 20 m outer border, curriculum mode. Rows are difficulty: row `k` gets difficulty uniform in `[k/10, (k+1)/10)`.
- **Generators:** mjlab's own box generators only. Stair tiles use a 3.0 m platform and a 1.0 m border.

| Sub-terrain | Share | Generator | Parameters |
|---|---|---|---|
| `up_25`, `up_30`, `up_35` | 13% / 14% / 13% (40%) | `BoxInvertedPyramidStairsTerrainCfg` (robot starts in the pit and climbs out) | treads 0.25 / 0.30 / 0.35 m (6 / 5 / 4 steps), risers 0.02 → 0.20 m by difficulty |
| `down_25`, `down_30`, `down_35` | 8% / 9% / 8% (25%) | `BoxPyramidStairsTerrainCfg` (robot starts on top and descends) | same |
| `flat` | 35% | `BoxFlatTerrainCfg` | |

- **Starting level:** `max_init_terrain_level = 2`.
- **Fixed seed:** `seed = 42`. mjlab's default draws a random seed on every run, which would give each run and each teammate a different terrain.
- **No heightfield or box-grid tiles** (see §2). Robustness to uneven ground comes from pushes, friction randomization and height-scan noise.
- **Play mode:** a random 5 × 5 tile grid, no curriculum.
- **Scene viewer:** the full curriculum grid, with robots spread over all rows.

### 5.3 Sensors

Each variant chooses which sensors its policy, critic and rewards use.

| Name | Type | Configuration | Purpose |
|---|---|---|---|
| `terrain_scan` | ray cast | pelvis, `ray_alignment="yaw"`, grid 1.6 × 1.0 m at 0.1 m (187 rays), max 5 m | default perception (as in Unitree) |
| `foot_scan_left`, `foot_scan_right` | ray cast | site `left_foot` / `right_foot`, `ray_alignment="yaw"`, grid 0.2 × 0.0 m at 0.1 m (3 rays: toe, middle, heel), max 1 m | height of each foot above the ground under it; detects a foot hanging over a stair edge |
| `feet_ground_contact` | contact | subtree of each `*_ankle_roll_link` vs terrain; `found`, `force`, air time | as in Unitree |
| `body_terrain_contact` | contact | every robot body except the `*_ankle_roll_link` subtrees, vs terrain; `found`, `force`, one slot per body | shins, knees, torso or hands striking stairs |
| `self_collision` | contact | pelvis subtree vs itself, history 4 | as in Unitree |

- **Built into the robot MJCF:** `imu_ang_vel`, `imu_lin_vel`, `imu_lin_acc` and `root_angmom`.
- **Ray filtering:** all ray-cast sensors use `include_geom_groups=(0,)`, so they see terrain only. The robot's visual meshes are geom group 2, and Unitree's default `(0, 1, 2)` lets the pelvis scan hit the robot's own legs. A test (§8) checks that a scan over flat ground returns ground height under every ray.

### 5.4 Physics: `SIM_CFG`

| Setting | Value |
|---|---|
| Physics timestep | 0.005 s |
| Decimation | 4 (policy runs at 50 Hz) |
| Episode length | 20 s |
| Solver iterations | 10 |
| Line-search iterations | 20 |
| `ccd_iterations` | 50 (no effect with box-only terrain; guards against someone adding a heightfield) |
| `nconmax` | 48 |
| `njmax` | 1500 |
| `contact_sensor_maxmatch` | 500 |

**Memory budget:** at 4096 envs, peak GPU memory is about 6.7 GB of 8 GB. The training scene has 3,462 geoms (58 of them the robot). Each extra geom costs about 150–250 KB of GPU memory at 4096 worlds, through the per-world ray-cast structure. `scene.py` records these numbers and a budget of **4,000 geoms**, which a test enforces. Extra sensors, geoms or envs may push a variant down to 3072 envs.

## 6. Tasks and variants

### 6.1 Baseline: `G1-Stairs-Baseline`

- **Environment:** `baseline_env_cfg()` puts Unitree's `Unitree-G1-23Dof-Rough` MDP on this scene, unchanged: commands, observations, rewards and weights, events and domain randomization, terminations, terrain and command curricula. Rewards are not tuned.
- **Training defaults:** `num_envs = 4096`.
- **PPO:** `baseline_ppo_cfg()` uses Unitree's settings: actor and critic 512-256-128, ELU, observation normalization, learning rate 1e-3 adaptive, 24 steps per env, and so on. Changes:
  - `logger="tensorboard"`
  - `experiment_name="g1_stairs_baseline"`
  - `max_iterations=5000`, overridable with `--agent.max-iterations`
- **Runner:** Unitree's `VelocityOnPolicyRunner`, which exports `policy.onnx` on every save.
- **Velocity command:** mjlab's `UniformVelocityCommandCfg`, exactly what Unitree's Rough task uses. Unitree's own copy in their `mdp/velocity_command.py` is never used by that task, so we don't copy it.
- **Play mode:** Unitree's play overrides on the play terrain.

### 6.2 Adding a variant

1. Copy `g1_stairs/tasks/baseline/` to `g1_stairs/tasks/<name>/`.
2. Change the task ID (`G1-Stairs-<Name>`) and `experiment_name`.
3. Edit rewards, observations, network or PPO settings. Each reward term is a function registered with a weight in the config.

`g1_stairs/tasks/__init__.py` imports every subpackage automatically, so a variant needs no edits elsewhere. That works both with one folder per variant on `main` and with branches. Terms that other variants may reuse go into `g1_stairs/mdp/`; terms for one variant stay in its folder.

## 7. Scripts

The scripts are adapted from `unitree_rl_mjlab/scripts` and run as `python scripts/<name>.py` from the repo root.

- **`train.py <task>`:** tyro CLI as upstream (`--env.*`, `--agent.*` overrides), single GPU. It sets `MUJOCO_GL=egl` only on Linux, and only if the user hasn't set it (upstream sets it unconditionally). Logs go to `<repo>/logs/rsl_rl/` wherever the script is run from. The YAML config dumps need no encoding fix, because PyYAML escapes non-ASCII characters by default.
- **`play.py <task> --checkpoint-file <path>`:** `--agent zero|random` needs no checkpoint. `--viewer` defaults to `native` on Windows (upstream's `auto` checks `DISPLAY`, which Windows lacks, and picks Viser). `--video` records to the run folder.
- **`view_scene.py [--num-envs 64] [--viewer native|viser]`:** the full training terrain, robots on every row, zero policy.
- **`list_envs.py`:** prints the registered `G1-Stairs-*` tasks.

## 8. Verification

**Fast tests** (`pytest`, CPU only, a few seconds):

1. `STAIRS_TERRAIN_CFG` compiles into a scene with at most 4,000 geoms. Its share of stair tiles is 65%, and its maximum riser is 0.20 m.
2. The compiled scene contains all five sensors. `foot_scan_*` have 3 rays each and `terrain_scan` has 187.
3. On flat ground with the robot at `HOME_KEYFRAME`, every hit point of `terrain_scan` and `foot_scan_*` lies on the ground (z = 0 ± 1 cm). This checks that no ray hits the robot.
4. `G1-Stairs-Baseline` is registered, and its train and play configs build.
5. There are no heightfield geoms in the training terrain.

**Acceptance check before the first push:** `python scripts/train.py G1-Stairs-Baseline --agent.max-iterations 30` finishes. The README records env-steps/s and peak GPU memory, and `requirements.lock.txt` is written from that venv.

## 9. Known issues to discuss next

These are starting points for the reward and command discussion. The baseline deliberately leaves them unfixed.

1. **`foot_clearance`** targets 0.10 m of absolute world height. On stairs it penalizes a foot for standing on a higher step. The `foot_scan_*` sensors provide the height above the ground under each foot.
2. **Gait clock:** the `phase` observation and the `foot_gait` reward use a fixed 0.6 s period, so step length is tied to speed (about 0.30 m at 1 m/s). There is no step-length command yet.
3. **Terrain curriculum:** `terrain_levels_vel` promotes robots by distance walked (more than 4 m), not by stairs climbed.
4. **Command ranges:** velocity commands (up to 2 m/s after the curriculum) are not tuned for stairs.

## 10. Attribution

`g1_stairs/assets/g1_23dof/`, `g1_stairs/mdp/`, the baseline configs and the scripts are derived from `unitreerobotics/unitree_rl_mjlab` (Apache-2.0). `NOTICE` lists them, and each copied file keeps a header naming its source.
