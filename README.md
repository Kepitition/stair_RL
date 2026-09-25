# g1-stairs-rl

A shared MuJoCo environment for training the **Unitree G1 humanoid (`g1_23dof_rev_1_0`) to climb stairs** with reinforcement learning. It is built on [mjlab](https://github.com/mujocolab/mjlab) (MuJoCo Warp + RSL-RL) and runs natively on Windows or Linux with an NVIDIA GPU.

The **scene** is identical for everyone: robot, stairs terrain, sensors and physics. You add your own **variant** (rewards, observations, network, PPO settings) as a separate task, so results stay comparable. Design and reasoning: [docs/superpowers/specs/2026-09-25-shared-environment-design.md](docs/superpowers/specs/2026-09-25-shared-environment-design.md).

## Install

You need an NVIDIA GPU with at least 8 GB (driver with CUDA 12.8 support), Python 3.10–3.13 and git.

```bash
git clone https://github.com/Kepitition/stair_RL.git
cd stair_RL
python -m venv .venv
.venv\Scripts\activate            # Linux: source .venv/bin/activate
pip install -r requirements.txt   # CUDA torch + pinned mjlab stack + this package
python -m pytest -q               # about 30 s; all tests should pass
```

Always install through `requirements.txt`. A plain `pip install -e .` may give you CPU-only torch, and the tests will tell you so.

## Commands

Run these from the repo root.

| What | Command |
|---|---|
| List tasks | `python scripts/list_envs.py` |
| Look at the scene | `python scripts/view_scene.py` |
| Train | `python scripts/train.py G1-Stairs-Baseline` |
| Short test run | `python scripts/train.py G1-Stairs-Baseline --agent.max-iterations 30` |
| Watch a policy | `python scripts/play.py G1-Stairs-Baseline --checkpoint-file logs/rsl_rl/g1_stairs_baseline/<run>/model_<n>.pt` |
| Record a video | add `--video --video-length 300` to the play command |
| TensorBoard | `tensorboard --logdir logs/rsl_rl` |

- **Overrides:** any config field can be overridden on the command line, e.g. `--env.scene.num-envs 2048` or `--agent.max-iterations 3000`.
- **Checkpoints:** saved every 100 iterations, each with a `policy.onnx` export.
- **Viewer:** `view_scene.py` and `play.py` open the native MuJoCo window. The keys are:
  - `.` and `,`: next and previous robot
  - `A`: show all robots
  - `R`: height-scan rays
  - `Space`: pause
  - `Enter`: reset

  Use `--viewer viser` for a browser view instead.

## The shared scene

| Part | What |
|---|---|
| Robot | Unitree G1 23-DOF, Unitree's actuator model; 23 joint position targets at 50 Hz |
| Terrain | 10 × 20 tiles of 8 × 8 m, seeded. Stairs up 40%, stairs down 25%, flat 35%. Risers 2 → 20 cm by difficulty row; treads 25, 30 and 35 cm. |
| `terrain_scan` | 187-ray height grid (1.6 × 1.0 m) under the pelvis |
| `foot_scan_left/right` | 3 rays under each foot (heel, middle, toe) |
| `feet_ground_contact` | per-foot contact, force, air time |
| `body_terrain_contact` | shins, knees, torso, arms touching the terrain |
| `self_collision` | self-contacts |
| Physics | 200 Hz, 20 s episodes |

### GPU memory and speed (RTX 4060 Laptop, 8 GB, 4096 envs)

Training itself uses about **6.2 GB** of GPU memory. Whatever else is running on the GPU (browser, IDE, display) comes on top of that:

| Other apps on the GPU | Peak total | Speed |
|---|---|---|
| ~0.5 GB (apps closed)* | 6.7 GB | 14.7k env-steps/s, 6.7 s per iteration |
| 1.6 GB (browser and IDE open) | 7.9 GB of 8.2 | 10.7k env-steps/s, 9.2 s per iteration |

\* Measured in the design spike, before `foot_scan_*` and `body_terrain_contact` were added; these add only 6 rays and one contact sensor.

When the GPU is nearly full, Windows spills GPU memory into system RAM, and the PPO update slows down about 8×. Before long runs, close GPU-heavy apps and check `nvidia-smi`. If that isn't possible, or you add sensors or geoms, use `--env.scene.num-envs 3072`.

The terrain uses box geometry only, because heightfield or box-grid tiles do not fit in 8 GB (spec §2). A test enforces a budget of at most 4,000 geoms.

## Adding your variant

1. Copy `g1_stairs/tasks/baseline/` to `g1_stairs/tasks/<your_name>/`.
2. In `<your_name>/__init__.py`, set `TASK_ID = "G1-Stairs-<YourName>"`. Two folders with the same ID stop every script with `Task '...' is already registered`.
3. In `<your_name>/rl_cfg.py`, set `experiment_name="g1_stairs_<your_name>"`, so your logs get their own folder.
4. Edit `env_cfg.py`: rewards, observations, commands; and `rl_cfg.py`: network, PPO. Each reward term is a function with a weight. Put terms other variants may reuse in `g1_stairs/mdp/`; keep your own terms in your folder.
5. Run `python scripts/list_envs.py`. Your task shows up without any other edits.

Do not change `g1_stairs/scene/` in a variant. If the scene needs to change, propose it to the team, because everyone's results depend on it.

## Known issues (starting points for reward design)

The baseline is Unitree's flat and rough-terrain setup, unchanged except for the height-scan noise. Unitree adds ±0.1 m per ray, as large as a riser; the baseline uses the model BeamDojo deployed on a real G1 with the MID-360 LiDAR: ±0.03 m per ray and step, plus one ±0.03 m offset per episode shared by all rays. The actor also sees the scan 0 or 1 policy step (0-20 ms) late. Other known issues:

- `foot_clearance` targets 0.10 m of absolute height, so on stairs it penalizes a foot for standing on a higher step. Use `foot_scan_*` for the height above the ground under each foot.
- The gait clock has a fixed 0.6 s period, and there is no step-length command yet.
- The terrain curriculum promotes robots by distance walked, not stairs climbed.
- Velocity commands go up to 2 m/s, which is not tuned for stairs.

## Layout

| Path | Contents |
|---|---|
| `g1_stairs/scene/` | the shared scene: `robot.py`, `terrain.py`, `sensors.py`, `scene.py` |
| `g1_stairs/tasks/` | one folder per variant; `baseline/` is the template |
| `g1_stairs/mdp/` | shared reward, observation, curriculum and termination terms |
| `scripts/` | train, play, view_scene, list_envs |
| `tests/` | fast CPU tests: `python -m pytest -q` |
| `docs/` | design spec, plans, research notes |

## Attribution

The robot model, baseline MDP and scripts are derived from [unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab) (Apache-2.0); see `NOTICE`.
