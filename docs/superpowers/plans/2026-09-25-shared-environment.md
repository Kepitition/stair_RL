# Shared G1 Stairs Environment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `g1_stairs` package: a shared, pip-installable MuJoCo (mjlab) scene for the Unitree G1 23-DOF on stairs, a runnable `G1-Stairs-Baseline` task, and Windows-safe scripts. Teammates clone it and add their own reward, perception and network variants.

**Architecture:** `g1_stairs/scene/` holds everything that must be identical across variants: robot, terrain, sensors and physics. `g1_stairs/tasks/<variant>/` folders build full environment configs on top of it and register a task ID; every subpackage of `g1_stairs.tasks` is imported automatically. The scripts in `scripts/` are thin tyro CLIs over `g1_stairs.runtime`.

**Tech Stack:**
- Python 3.12 (3.10+ supported)
- torch 2.9.0+cu128, mjlab 1.2.0, mujoco 3.5.0, mujoco-warp 3.5.0, warp-lang 1.12.0, rsl-rl-lib 5.0.1
- pytest

**Spec:** `docs/superpowers/specs/2026-09-25-shared-environment-design.md`. Read it first: §2 explains the memory limits behind several choices.

## Global Constraints

- **Setup:**
  - Work on branch `shared-environment` in this checkout (`C:\Users\ayber\g1-stairs-rl`). The venv at `.venv/` is bound to this path, so do not use a separate worktree.
  - Run every command from the repo root with the venv's Python: `.venv/Scripts/python.exe` on Windows (written `python` below after activating: `.venv\Scripts\activate`).
- **Pinned versions:** `torch==2.9.0+cu128`, `torchvision==0.24.0+cu128` (cu128 index), `mjlab==1.2.0`, `mujoco==3.5.0`, `mujoco-warp==3.5.0`, `warp-lang==1.12.0`, `rsl-rl-lib==5.0.1`, `scipy`.
- **Platform:** runs natively on Windows 11 (no WSL, no Docker) and on Linux.
- **Code style:**
  - 2-space indentation and double quotes, matching mjlab and Unitree.
  - Every file copied or derived from `external/unitree_rl_mjlab` starts with a header naming its source file and "Apache-2.0".
- **Terrain:** box generators only, no heightfields. Scene geom budget: at most 4,000 geoms (currently 3,462).
- **Ray-cast sensors:** `include_geom_groups=(0,)`, so they see terrain only.
- **Baseline behavior:** Unitree's `Unitree-G1-23Dof-Rough` MDP, unchanged: same reward terms and weights, observations, events, terminations and curricula.
- **Baseline training defaults:** `num_envs=4096`, `ccd_iterations=50`, `logger="tensorboard"`, `experiment_name="g1_stairs_baseline"`, `max_iterations=5000`.
- **Reference copies:** `external/unitree_rl_mjlab` and `external/mjlab` are read-only references and are never edited.
- **Commits:** every commit message ends with the line `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.

## Review Focus

1. **Running from another directory** (a script launched from `scripts/`, or pytest from elsewhere): assets still load, and logs still land in `<repo>/logs/rsl_rl/`. Tested in Task 2 (robot compiles after `chdir`) and Task 6 (`list_envs.py` from a temp cwd, `LOG_ROOT` inside the repo).
2. **A teammate copies `tasks/baseline/` and forgets to rename the task ID:** the import must fail loudly with the duplicated ID in the message. Tested in Task 5.
3. **An install that ends up with CPU-only torch** (`pip install -e .` alone, or torchvision left unpinned): the test suite must fail with a clear message. Tested in Task 1.
4. **Windows specifics:** `MUJOCO_GL` must not be forced to `egl`, and `--viewer auto` must pick the native window rather than Viser. Tested in Task 6.
5. **Two runs or two teammates getting different terrains:** the generator must be seeded so the compiled terrain is identical. Tested in Task 3 (seed) and Task 4 (two compiles give identical geoms).

---

### Task 0: Branch

- [ ] **Step 1: Create the branch**

```bash
git checkout -b shared-environment
git status --short
```
Expected: `Switched to a new branch 'shared-environment'` and a clean status.

---

### Task 1: Package skeleton and install

**Files:**
- Create: `pyproject.toml`, `requirements.txt`, `.gitattributes`, `NOTICE`, `LICENSES/unitree_rl_mjlab-LICENSE`
- Create: `g1_stairs/__init__.py`, `g1_stairs/tasks/__init__.py`
- Test: `tests/test_install.py`

**Interfaces:**
- Produces: `g1_stairs.PACKAGE_ROOT: Path` and `g1_stairs.REPO_ROOT: Path`. Importing `g1_stairs.tasks` imports every subpackage of `g1_stairs/tasks/`.

- [ ] **Step 1: Write the failing test**

`tests/test_install.py`:
```python
"""The install must give a CUDA torch and the pinned simulator stack (spec §4)."""

from importlib.metadata import version

import torch

import g1_stairs


def test_torch_is_a_cuda_build():
  # CPU torch means requirements.txt was bypassed or torchvision was unpinned (spec §2).
  assert torch.version.cuda is not None, (
    f"torch {torch.__version__} is a CPU-only build; reinstall with "
    "`pip install -r requirements.txt`"
  )


def test_pinned_versions():
  assert torch.__version__.startswith("2.9.0")
  assert version("torchvision").startswith("0.24.0")
  assert version("mjlab") == "1.2.0"
  assert version("mujoco") == "3.5.0"
  assert version("mujoco-warp") == "3.5.0"
  assert version("warp-lang") == "1.12.0"
  assert version("rsl-rl-lib") == "5.0.1"


def test_repo_root_points_at_the_checkout():
  assert (g1_stairs.REPO_ROOT / "pyproject.toml").is_file()
  assert g1_stairs.PACKAGE_ROOT == g1_stairs.REPO_ROOT / "g1_stairs"
```

- [ ] **Step 2: Run it and watch it fail**

Run: `.venv/Scripts/python.exe -m pip install pytest` and then `.venv/Scripts/python.exe -m pytest tests/test_install.py -v`
Expected: collection error `ModuleNotFoundError: No module named 'g1_stairs'`.

- [ ] **Step 3: Create the package and install files**

`pyproject.toml`:
```toml
[build-system]
requires = ["setuptools>=64"]
build-backend = "setuptools.build_meta"

[project]
name = "g1_stairs"
version = "0.1.0"
description = "Shared MuJoCo (mjlab) environment for Unitree G1 23-DOF stair climbing"
requires-python = ">=3.10"
# Same pins as requirements.txt. Install through requirements.txt so torch
# comes from the CUDA 12.8 index; "==2.9.0" also accepts "2.9.0+cu128".
dependencies = [
  "torch==2.9.0",
  "torchvision==0.24.0",
  "mjlab==1.2.0",
  "mujoco==3.5.0",
  "mujoco-warp==3.5.0",
  "warp-lang==1.12.0",
  "rsl-rl-lib==5.0.1",
  "scipy",
]

[tool.setuptools.packages.find]
include = ["g1_stairs*"]

[tool.setuptools.package-data]
g1_stairs = ["assets/**/*.xml", "assets/**/*.STL"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

`requirements.txt`:
```
# One-command install: pip install -r requirements.txt
# torch/torchvision must come from the CUDA 12.8 index; unpinned torchvision
# silently replaces CUDA torch with CPU torch (see the design spec, section 2).
--extra-index-url https://download.pytorch.org/whl/cu128
torch==2.9.0+cu128
torchvision==0.24.0+cu128
mjlab==1.2.0
mujoco==3.5.0
mujoco-warp==3.5.0
warp-lang==1.12.0
rsl-rl-lib==5.0.1
# mjlab 1.2.0 imports scipy without declaring it.
scipy
pytest
-e .
```

`.gitattributes`:
```
* text=auto
*.STL binary
*.stl binary
*.onnx binary
*.pt binary
```

`NOTICE`:
```
g1-stairs-rl

This project contains code and assets derived from unitree_rl_mjlab
(https://github.com/unitreerobotics/unitree_rl_mjlab), Copyright Unitree
Robotics, licensed under the Apache License, Version 2.0. A copy of that
license is in LICENSES/unitree_rl_mjlab-LICENSE.

Derived files (each starts with a header naming its source):
- g1_stairs/assets/g1_23dof/  (MJCF model, meshes, constants.py)
- g1_stairs/mdp/              (rewards, observations, curriculums, terminations)
- g1_stairs/runner.py
- g1_stairs/tasks/baseline/   (environment and PPO configuration)
- scripts/train.py, scripts/play.py, scripts/list_envs.py
```

Copy the license:
```bash
mkdir -p LICENSES
cp external/unitree_rl_mjlab/LICENCE LICENSES/unitree_rl_mjlab-LICENSE
```

`g1_stairs/__init__.py`:
```python
"""Shared MuJoCo environment for Unitree G1 23-DOF stair climbing.

Import ``g1_stairs.tasks`` to register every G1-Stairs-* task.
"""

from pathlib import Path

PACKAGE_ROOT: Path = Path(__file__).resolve().parent
REPO_ROOT: Path = PACKAGE_ROOT.parent
```

`g1_stairs/tasks/__init__.py`:
```python
"""Task registry.

Every subpackage of g1_stairs.tasks is imported here, and each one registers its
own task IDs. To add a variant, copy tasks/baseline/ to tasks/<name>/ and rename
its task ID; no other file needs editing.
"""

from mjlab.utils.lab_api.tasks.importer import import_packages

import_packages(__name__, blacklist_pkgs=[])
```

- [ ] **Step 4: Install**

Run: `.venv/Scripts/python.exe -m pip install -r requirements.txt`
Expected: ends with `Successfully installed g1_stairs-0.1.0` (other packages already satisfied). Then `.venv/Scripts/python.exe -m pip check` prints `No broken requirements found.`

- [ ] **Step 5: Run the tests and watch them pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_install.py -v`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml requirements.txt .gitattributes NOTICE LICENSES g1_stairs tests/test_install.py
git commit -m "Add g1_stairs package skeleton and pinned install

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Robot asset and robot interface

**Files:**
- Create: `g1_stairs/assets/__init__.py`, `g1_stairs/assets/g1_23dof/__init__.py`
- Create (copied): `g1_stairs/assets/g1_23dof/g1_23dof.xml`, `g1_stairs/assets/g1_23dof/meshes/*.STL` (27 files), `g1_stairs/assets/g1_23dof/constants.py`
- Create: `g1_stairs/scene/__init__.py`, `g1_stairs/scene/robot.py`
- Test: `tests/test_robot.py`

**Interfaces:**
- Produces, in `g1_stairs.scene.robot`:
  - `ROBOT_NAME = "robot"`, `PELVIS_BODY = "pelvis"`, `TORSO_BODY = "torso_link"`
  - `FOOT_SITE_NAMES: tuple[str, str]`, `FOOT_BODY_PATTERN: str`, `FOOT_GEOM_NAMES: tuple[str, ...]`
  - `NON_FOOT_CONTACT_BODIES: tuple[str, ...]`, `ACTION_SCALE: dict[str, float]`
  - `get_robot_cfg() -> EntityCfg`
  - `make_actions() -> dict[str, JointPositionActionCfg]`
- Produces, in `g1_stairs.assets.g1_23dof.constants`: `G1_23DOF_XML: Path`, `HOME_KEYFRAME`.

- [ ] **Step 1: Write the failing test**

`tests/test_robot.py`:
```python
"""The robot is Unitree's g1_23dof_rev_1_0, unchanged (spec §5.1)."""

import re

import mujoco
import pytest
from mjlab.entity import Entity

from g1_stairs.scene import robot

# MJCF joint order (docs/research/findings.md §2).
EXPECTED_JOINTS = [
  "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
  "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
  "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
  "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
  "waist_yaw_joint",
  "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
  "left_elbow_joint", "left_wrist_roll_joint",
  "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
  "right_elbow_joint", "right_wrist_roll_joint",
]


@pytest.fixture(scope="module")
def model() -> mujoco.MjModel:
  return Entity(robot.get_robot_cfg()).spec.compile()


def test_model_is_g1_23dof_rev_1_0():
  from g1_stairs.assets.g1_23dof.constants import G1_23DOF_XML

  assert mujoco.MjSpec.from_file(str(G1_23DOF_XML)).modelname == "g1_23dof_rev_1_0"


def test_joint_order_and_actuators(model):
  hinges = [
    model.joint(j).name
    for j in range(model.njnt)
    if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
  ]
  assert hinges == EXPECTED_JOINTS
  assert model.nu == 23


def test_action_scale_covers_every_joint():
  for joint in EXPECTED_JOINTS:
    assert any(re.fullmatch(p, joint) for p in robot.ACTION_SCALE), joint


def test_named_parts_exist(model):
  for site in robot.FOOT_SITE_NAMES:
    model.site(site)  # raises KeyError if missing
  for geom in robot.FOOT_GEOM_NAMES:
    model.geom(geom)
  for body in robot.NON_FOOT_CONTACT_BODIES:
    bid = model.body(body).id
    owned = [g for g in range(model.ngeom) if model.geom_bodyid[g] == bid and model.geom_group[g] == 3]
    assert owned, f"{body} has no collision geom"


def test_actions_are_joint_position_targets():
  term = robot.make_actions()["joint_pos"]
  assert term.entity_name == robot.ROBOT_NAME
  assert term.actuator_names == (".*",)
  assert term.use_default_offset is True
  assert term.scale == robot.ACTION_SCALE


def test_robot_compiles_from_any_working_directory(tmp_path, monkeypatch):
  monkeypatch.chdir(tmp_path)
  assert Entity(robot.get_robot_cfg()).spec.compile().nu == 23
```

- [ ] **Step 2: Run it and watch it fail**

Run: `python -m pytest tests/test_robot.py -v`
Expected: collection error `ModuleNotFoundError: No module named 'g1_stairs.scene'`.

- [ ] **Step 3: Copy the model and meshes**

```bash
SRC=external/unitree_rl_mjlab/src/assets/robots/unitree_g1
DST=g1_stairs/assets/g1_23dof
mkdir -p $DST/meshes
cp $SRC/xmls/g1_23dof.xml $DST/g1_23dof.xml
for f in $(grep -oE 'file="[^"]+"' $DST/g1_23dof.xml | sed 's/file="//;s/"//'); do
  cp "$SRC/xmls/assets/$f" "$DST/meshes/"
done
ls $DST/meshes | wc -l
sed -i 's/meshdir="assets"/meshdir="meshes"/' $DST/g1_23dof.xml
sed -i '1i <!-- Derived from unitree_rl_mjlab src/assets/robots/unitree_g1/xmls/g1_23dof.xml (Apache-2.0). Change: meshdir="meshes". -->' $DST/g1_23dof.xml
cp $SRC/g1_23dof_constants.py $DST/constants.py
```
Expected: `27` meshes listed.

- [ ] **Step 4: Edit `g1_stairs/assets/g1_23dof/constants.py`**

Make these edits, and no others:

(a) Replace the first line `"""Unitree G1_23DOF constants."""` with:
```python
"""Unitree G1 23-DOF (g1_23dof_rev_1_0): MJCF path, actuators, keyframes, collisions.

Derived from unitree_rl_mjlab src/assets/robots/unitree_g1/g1_23dof_constants.py
(Apache-2.0). Changes: paths are relative to this file, meshes live in meshes/,
and the __main__ viewer block is removed.
"""
```

(b) Delete the line `from src import SRC_PATH`.

(c) Replace
```python
G1_23DOF_XML: Path = (
  SRC_PATH / "assets" / "robots" / "unitree_g1" / "xmls" / "g1_23dof.xml"
)
```
with
```python
G1_23DOF_XML: Path = Path(__file__).resolve().parent / "g1_23dof.xml"
```

(d) In `get_assets`, replace `G1_23DOF_XML.parent / "assets"` with `G1_23DOF_XML.parent / "meshes"`.

(e) Delete the final `if __name__ == "__main__":` block through the end of the file.

`g1_stairs/assets/__init__.py`:
```python
"""Robot assets."""
```

`g1_stairs/assets/g1_23dof/__init__.py`:
```python
"""Unitree G1 23-DOF (g1_23dof_rev_1_0) model, derived from unitree_rl_mjlab."""
```

- [ ] **Step 5: Write the robot interface**

`g1_stairs/scene/__init__.py`:
```python
"""The shared scene: robot, terrain, sensors and physics. Identical for every variant."""
```

`g1_stairs/scene/robot.py`:
```python
"""The robot and its control interface (spec §5.1). Fixed for every variant."""

from mjlab.entity import EntityCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg

from g1_stairs.assets.g1_23dof.constants import (
  G1_23DOF_ACTION_SCALE,
  get_g1_23dof_robot_cfg,
)

ROBOT_NAME = "robot"
PELVIS_BODY = "pelvis"
TORSO_BODY = "torso_link"

FOOT_SITE_NAMES: tuple[str, str] = ("left_foot", "right_foot")
FOOT_BODY_PATTERN = r"^(left_ankle_roll_link|right_ankle_roll_link)$"
FOOT_GEOM_NAMES: tuple[str, ...] = tuple(
  f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
)

# Every body that owns a collision geom, except the two feet (ankle_roll links).
NON_FOOT_CONTACT_BODIES: tuple[str, ...] = (
  "pelvis",
  "torso_link",
  "left_hip_roll_link",
  "right_hip_roll_link",
  "left_hip_yaw_link",
  "right_hip_yaw_link",
  "left_knee_link",
  "right_knee_link",
  "left_shoulder_yaw_link",
  "right_shoulder_yaw_link",
  "left_elbow_link",
  "right_elbow_link",
  "left_wrist_roll_rubber_hand",
  "right_wrist_roll_rubber_hand",
)

# Per-joint action scale: 0.25 * torque limit / stiffness (Unitree).
ACTION_SCALE: dict[str, float] = dict(G1_23DOF_ACTION_SCALE)


def get_robot_cfg() -> EntityCfg:
  """A fresh copy of Unitree's G1 23-DOF entity config."""
  return get_g1_23dof_robot_cfg()


def make_actions() -> dict[str, JointPositionActionCfg]:
  """Position targets for all 23 joints around the default pose."""
  return {
    "joint_pos": JointPositionActionCfg(
      entity_name=ROBOT_NAME,
      actuator_names=(".*",),
      scale=dict(ACTION_SCALE),
      use_default_offset=True,
    )
  }
```

- [ ] **Step 6: Run the tests and watch them pass**

Run: `python -m pytest tests/test_robot.py -v`
Expected: 6 passed.

- [ ] **Step 7: Commit**

```bash
git add g1_stairs/assets g1_stairs/scene tests/test_robot.py
git commit -m "Add G1 23-DOF asset and robot interface

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Stairs terrain

**Files:**
- Create: `g1_stairs/scene/terrain.py`
- Test: `tests/test_terrain.py`

**Interfaces:**
- Produces, in `g1_stairs.scene.terrain`:
  - `RISER_RANGE = (0.02, 0.20)`, `TREADS = (0.25, 0.30, 0.35)`, `TERRAIN_SEED = 42`, `MAX_INIT_TERRAIN_LEVEL = 2`
  - `STAIRS_TERRAIN_CFG: TerrainGeneratorCfg`
  - `make_terrain_cfg(play: bool = False) -> TerrainEntityCfg`, which returns a fresh copy on every call

- [ ] **Step 1: Write the failing test**

`tests/test_terrain.py`:
```python
"""Training terrain: box stairs up/down plus flat, seeded (spec §5.2)."""

import mjlab.terrains as tg
import pytest

from g1_stairs.scene import terrain

UP = tg.BoxInvertedPyramidStairsTerrainCfg
DOWN = tg.BoxPyramidStairsTerrainCfg


def _share(kind) -> float:
  subs = terrain.STAIRS_TERRAIN_CFG.sub_terrains.values()
  total = sum(s.proportion for s in subs)
  return sum(s.proportion for s in subs if type(s) is kind) / total


def test_tile_shares():
  assert _share(UP) == pytest.approx(0.40)
  assert _share(DOWN) == pytest.approx(0.25)
  assert _share(tg.BoxFlatTerrainCfg) == pytest.approx(0.35)


def test_risers_treads_and_platforms():
  stairs = [s for s in terrain.STAIRS_TERRAIN_CFG.sub_terrains.values() if type(s) in (UP, DOWN)]
  assert len(stairs) == 6
  for s in stairs:
    assert s.step_height_range == (0.02, 0.20)
    assert s.platform_width == 3.0
    assert s.border_width == 1.0
  assert sorted({s.step_width for s in stairs}) == [0.25, 0.30, 0.35]


def test_only_box_generators():
  # Heightfield tiles route foot contacts through the memory-hungry EPA collider (spec §2).
  for name, s in terrain.STAIRS_TERRAIN_CFG.sub_terrains.items():
    assert type(s).__name__.startswith("Box"), f"{name}: {type(s).__name__}"


def test_grid_and_seed():
  gen = terrain.STAIRS_TERRAIN_CFG
  assert (gen.size, gen.num_rows, gen.num_cols) == ((8.0, 8.0), 10, 20)
  assert gen.curriculum is True
  assert gen.seed == 42  # every run and every teammate gets the same terrain


def test_training_terrain_cfg():
  cfg = terrain.make_terrain_cfg()
  assert cfg.terrain_type == "generator"
  assert cfg.max_init_terrain_level == 2
  assert cfg.terrain_generator.curriculum is True


def test_play_terrain_cfg():
  gen = terrain.make_terrain_cfg(play=True).terrain_generator
  assert (gen.curriculum, gen.num_rows, gen.num_cols, gen.border_width) == (False, 5, 5, 10.0)


def test_make_terrain_cfg_returns_independent_copies():
  a = terrain.make_terrain_cfg()
  b = terrain.make_terrain_cfg()
  a.terrain_generator.sub_terrains["flat"].proportion = 0.9
  assert b.terrain_generator.sub_terrains["flat"].proportion == 0.35
  assert terrain.STAIRS_TERRAIN_CFG.sub_terrains["flat"].proportion == 0.35
```

- [ ] **Step 2: Run it and watch it fail**

Run: `python -m pytest tests/test_terrain.py -v`
Expected: `ImportError: cannot import name 'terrain' from 'g1_stairs.scene'`.

- [ ] **Step 3: Implement**

`g1_stairs/scene/terrain.py`:
```python
"""Training and play terrain (spec §5.2): box stairs up and down, plus flat ground.

Only box generators are used. Heightfields route foot contacts through MuJoCo
Warp's EPA collider, and box grids add thousands of geoms; either one pushes 4096
envs past 8 GB of GPU memory (spec §2).
"""

from dataclasses import replace

import mjlab.terrains as terrain_gen
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeneratorCfg

RISER_RANGE: tuple[float, float] = (0.02, 0.20)
"""Riser height in meters, interpolated over the 10 difficulty rows."""
TREADS: tuple[float, ...] = (0.25, 0.30, 0.35)
"""Tread depths in meters; each gets its own columns."""
TERRAIN_SEED = 42
MAX_INIT_TERRAIN_LEVEL = 2

_UP_SHARES = (0.13, 0.14, 0.13)  # 40 %: start in the pit, climb out
_DOWN_SHARES = (0.08, 0.09, 0.08)  # 25 %: start on the top platform, descend
_FLAT_SHARE = 0.35


def _stairs(up: bool, tread: float, share: float) -> SubTerrainCfg:
  cls = (
    terrain_gen.BoxInvertedPyramidStairsTerrainCfg
    if up
    else terrain_gen.BoxPyramidStairsTerrainCfg
  )
  return cls(
    proportion=share,
    step_height_range=RISER_RANGE,
    step_width=tread,
    platform_width=3.0,
    border_width=1.0,
  )


def _sub_terrains() -> dict[str, SubTerrainCfg]:
  subs: dict[str, SubTerrainCfg] = {}
  for tread, share in zip(TREADS, _UP_SHARES):
    subs[f"up_{round(tread * 100)}"] = _stairs(True, tread, share)
  for tread, share in zip(TREADS, _DOWN_SHARES):
    subs[f"down_{round(tread * 100)}"] = _stairs(False, tread, share)
  subs["flat"] = terrain_gen.BoxFlatTerrainCfg(proportion=_FLAT_SHARE)
  return subs


STAIRS_TERRAIN_CFG = TerrainGeneratorCfg(
  size=(8.0, 8.0),
  border_width=20.0,
  num_rows=10,
  num_cols=20,
  curriculum=True,
  seed=TERRAIN_SEED,
  sub_terrains=_sub_terrains(),
  add_lights=True,
)


def make_terrain_cfg(play: bool = False) -> TerrainEntityCfg:
  """A fresh terrain config. Play mode: a random 5 x 5 grid without curriculum."""
  generator = replace(STAIRS_TERRAIN_CFG, sub_terrains=_sub_terrains())
  if play:
    generator = replace(
      generator, curriculum=False, num_rows=5, num_cols=5, border_width=10.0
    )
  return TerrainEntityCfg(
    terrain_type="generator",
    terrain_generator=generator,
    max_init_terrain_level=MAX_INIT_TERRAIN_LEVEL,
  )
```

- [ ] **Step 4: Run the tests and watch them pass**

Run: `python -m pytest tests/test_terrain.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add g1_stairs/scene/terrain.py tests/test_terrain.py
git commit -m "Add seeded box-stairs training and play terrain

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Sensors, scene assembly and physics

**Files:**
- Create: `g1_stairs/scene/sensors.py`, `g1_stairs/scene/scene.py`
- Test: `tests/test_scene.py`, `tests/test_ray_filter.py`

**Interfaces:**
- Consumes: `g1_stairs.scene.robot` (Task 2), `g1_stairs.scene.terrain.make_terrain_cfg` (Task 3).
- Produces, in `g1_stairs.scene.sensors`:
  - name constants `TERRAIN_SCAN`, `FOOT_SCAN_LEFT`, `FOOT_SCAN_RIGHT`, `FEET_GROUND_CONTACT`, `BODY_TERRAIN_CONTACT`, `SELF_COLLISION`, and `SENSOR_NAMES: tuple[str, ...]`
  - `RAY_GEOM_GROUPS = (0,)`, `TERRAIN_SCAN_MAX_DISTANCE = 5.0`
  - `make_sensors() -> tuple[SensorCfg, ...]`
- Produces, in `g1_stairs.scene.scene`:
  - `DEFAULT_NUM_ENVS = 4096`, `PLAY_NUM_ENVS = 16`, `DECIMATION = 4`, `EPISODE_LENGTH_S = 20.0`
  - `MEASURED_SCENE_GEOMS = 3462`, `GEOM_BUDGET = 4000`
  - `make_scene_cfg(play: bool = False, num_envs: int | None = None) -> SceneCfg`
  - `make_sim_cfg() -> SimulationCfg`

- [ ] **Step 1: Write the failing tests**

`tests/test_scene.py`:
```python
"""Compiled scene: sensors, geom budget, no heightfields, determinism (spec §5.3–5.4)."""

import numpy as np
import pytest
from mjlab.scene import Scene

from g1_stairs.scene import scene as scene_mod
from g1_stairs.scene import sensors


def _compile(play: bool = False):
  s = Scene(scene_mod.make_scene_cfg(play=play, num_envs=1), device="cpu")
  return s, s.compile()


@pytest.fixture(scope="module")
def training_scene():
  return _compile()


def test_geom_budget(training_scene):
  _, m = training_scene
  assert m.ngeom <= scene_mod.GEOM_BUDGET, (
    f"{m.ngeom} geoms exceed the {scene_mod.GEOM_BUDGET} budget: 4096 envs will not fit in 8 GB"
  )


def test_no_heightfields(training_scene):
  _, m = training_scene
  assert m.nhfield == 0


def test_all_sensors_present(training_scene):
  s, _ = training_scene
  assert set(sensors.SENSOR_NAMES) <= set(s.sensors)


@pytest.mark.parametrize(
  "name,rays",
  [(sensors.TERRAIN_SCAN, 187), (sensors.FOOT_SCAN_LEFT, 3), (sensors.FOOT_SCAN_RIGHT, 3)],
)
def test_ray_counts(training_scene, name, rays):
  s, _ = training_scene
  offsets, _ = s.sensors[name].cfg.pattern.generate_rays(None, "cpu")
  assert offsets.shape[0] == rays


def test_all_ray_sensors_see_terrain_only(training_scene):
  s, _ = training_scene
  for name in (sensors.TERRAIN_SCAN, sensors.FOOT_SCAN_LEFT, sensors.FOOT_SCAN_RIGHT):
    assert s.sensors[name].cfg.include_geom_groups == (0,)


def test_terrain_is_identical_across_builds(training_scene):
  _, m1 = training_scene
  _, m2 = _compile()
  assert m1.ngeom == m2.ngeom
  np.testing.assert_array_equal(m1.geom_pos, m2.geom_pos)
  np.testing.assert_array_equal(m1.geom_size, m2.geom_size)


def test_play_scene_compiles():
  _, m = _compile(play=True)
  assert m.nhfield == 0


def test_defaults():
  assert scene_mod.make_scene_cfg().num_envs == 4096
  assert scene_mod.make_scene_cfg(play=True).num_envs == 16
  sim = scene_mod.make_sim_cfg()
  assert sim.mujoco.timestep == 0.005
  assert sim.mujoco.ccd_iterations == 50
  assert (sim.nconmax, sim.njmax, sim.contact_sensor_maxmatch) == (48, 1500, 500)
```

`tests/test_ray_filter.py`:
```python
"""Ray-cast sensors must hit the terrain, never the robot (spec §5.3).

Uses CPU MuJoCo's mj_ray with each sensor's own geom-group mask, on flat ground
with the robot upright and 1 m above the ground.
"""

import mujoco
import numpy as np
import pytest
from mjlab.scene import Scene
from mjlab.terrains import TerrainEntityCfg

from g1_stairs.scene import scene as scene_mod
from g1_stairs.scene import sensors


@pytest.fixture(scope="module")
def flat_world():
  cfg = scene_mod.make_scene_cfg(num_envs=1)
  cfg.terrain = TerrainEntityCfg(terrain_type="plane")
  s = Scene(cfg, device="cpu")
  m = s.compile()
  d = mujoco.MjData(m)
  free = next(j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE)
  adr = m.jnt_qposadr[free]
  d.qpos[adr : adr + 7] = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]  # upright, zero yaw
  mujoco.mj_forward(m, d)
  return s, m, d


def _hit_heights(m, d, cfg, origin, exclude_body, groups) -> np.ndarray:
  offsets, directions = cfg.pattern.generate_rays(None, "cpu")
  mask = np.zeros(6, dtype=np.uint8)
  mask[list(groups)] = 1
  geomid = np.zeros(1, dtype=np.int32)
  heights = []
  for offset, direction in zip(offsets.numpy().astype(np.float64), directions.numpy().astype(np.float64)):
    start = origin + offset  # zero yaw: yaw-aligned offsets equal world offsets
    dist = mujoco.mj_ray(m, d, start, direction, mask, 1, exclude_body, geomid)
    assert dist >= 0, f"ray from {start} hit nothing"
    heights.append(start[2] + dist * direction[2])
  return np.array(heights)


def test_terrain_scan_hits_only_ground(flat_world):
  s, m, d = flat_world
  cfg = s.sensors[sensors.TERRAIN_SCAN].cfg
  pelvis = m.body("robot/pelvis").id
  z = _hit_heights(m, d, cfg, d.xpos[pelvis].copy(), pelvis, cfg.include_geom_groups)
  np.testing.assert_allclose(z, 0.0, atol=0.01)


@pytest.mark.parametrize(
  "name,site",
  [(sensors.FOOT_SCAN_LEFT, "robot/left_foot"), (sensors.FOOT_SCAN_RIGHT, "robot/right_foot")],
)
def test_foot_scan_hits_only_ground(flat_world, name, site):
  s, m, d = flat_world
  cfg = s.sensors[name].cfg
  sid = m.site(site).id
  z = _hit_heights(m, d, cfg, d.site_xpos[sid].copy(), m.site_bodyid[sid], cfg.include_geom_groups)
  np.testing.assert_allclose(z, 0.0, atol=0.01)


def test_unitree_default_groups_would_hit_the_robot(flat_world):
  """Documents why RAY_GEOM_GROUPS is (0,): Unitree's (0, 1, 2) includes the robot's visual meshes.

  If this fails, the rationale in spec §5.3 is wrong: stop and report instead of deleting it.
  """
  s, m, d = flat_world
  cfg = s.sensors[sensors.TERRAIN_SCAN].cfg
  pelvis = m.body("robot/pelvis").id
  z = _hit_heights(m, d, cfg, d.xpos[pelvis].copy(), pelvis, (0, 1, 2))
  assert (z > 0.05).any()
```

- [ ] **Step 2: Run them and watch them fail**

Run: `python -m pytest tests/test_scene.py tests/test_ray_filter.py -v`
Expected: `ImportError: cannot import name 'scene' from 'g1_stairs.scene'`.

- [ ] **Step 3: Implement the sensors**

`g1_stairs/scene/sensors.py`:
```python
"""Scene sensors (spec §5.3). Every variant gets all of them; each decides what to use.

The robot's IMU (robot/imu_ang_vel, robot/imu_lin_vel, robot/imu_lin_acc) and
robot/root_angmom come from the MJCF and are not listed here.
"""

from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  GridPatternCfg,
  ObjRef,
  RayCastSensorCfg,
  SensorCfg,
)

from g1_stairs.scene.robot import (
  FOOT_BODY_PATTERN,
  FOOT_SITE_NAMES,
  NON_FOOT_CONTACT_BODIES,
  PELVIS_BODY,
  ROBOT_NAME,
)

TERRAIN_SCAN = "terrain_scan"
FOOT_SCAN_LEFT = "foot_scan_left"
FOOT_SCAN_RIGHT = "foot_scan_right"
FEET_GROUND_CONTACT = "feet_ground_contact"
BODY_TERRAIN_CONTACT = "body_terrain_contact"
SELF_COLLISION = "self_collision"
SENSOR_NAMES: tuple[str, ...] = (
  TERRAIN_SCAN,
  FOOT_SCAN_LEFT,
  FOOT_SCAN_RIGHT,
  FEET_GROUND_CONTACT,
  BODY_TERRAIN_CONTACT,
  SELF_COLLISION,
)

# Terrain geoms are in group 0. The robot's visual meshes are group 2 and its
# collision geoms group 3, so rays restricted to group 0 never see the robot.
RAY_GEOM_GROUPS: tuple[int, ...] = (0,)
TERRAIN_SCAN_MAX_DISTANCE = 5.0


def _terrain_scan() -> RayCastSensorCfg:
  """1.6 x 1.0 m grid at 0.1 m (187 rays) under the pelvis, turning with heading."""
  return RayCastSensorCfg(
    name=TERRAIN_SCAN,
    frame=ObjRef(type="body", name=PELVIS_BODY, entity=ROBOT_NAME),
    ray_alignment="yaw",
    pattern=GridPatternCfg(size=(1.6, 1.0), resolution=0.1),
    max_distance=TERRAIN_SCAN_MAX_DISTANCE,
    exclude_parent_body=True,
    include_geom_groups=RAY_GEOM_GROUPS,
    debug_vis=True,
    viz=RayCastSensorCfg.VizCfg(show_normals=True),
  )


def _foot_scan(name: str, site: str) -> RayCastSensorCfg:
  """3 downward rays along the sole (heel, middle, toe; 0.1 m apart)."""
  return RayCastSensorCfg(
    name=name,
    frame=ObjRef(type="site", name=site, entity=ROBOT_NAME),
    ray_alignment="yaw",
    pattern=GridPatternCfg(size=(0.2, 0.0), resolution=0.1),
    max_distance=1.0,
    exclude_parent_body=True,
    include_geom_groups=RAY_GEOM_GROUPS,
  )


def make_sensors() -> tuple[SensorCfg, ...]:
  feet_ground_contact = ContactSensorCfg(
    name=FEET_GROUND_CONTACT,
    primary=ContactMatch(mode="subtree", pattern=FOOT_BODY_PATTERN, entity=ROBOT_NAME),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  body_terrain_contact = ContactSensorCfg(
    name=BODY_TERRAIN_CONTACT,
    primary=ContactMatch(
      mode="body",
      pattern=tuple(f"^{body}$" for body in NON_FOOT_CONTACT_BODIES),
      entity=ROBOT_NAME,
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
  )
  self_collision = ContactSensorCfg(
    name=SELF_COLLISION,
    primary=ContactMatch(mode="subtree", pattern=PELVIS_BODY, entity=ROBOT_NAME),
    secondary=ContactMatch(mode="subtree", pattern=PELVIS_BODY, entity=ROBOT_NAME),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  return (
    _terrain_scan(),
    _foot_scan(FOOT_SCAN_LEFT, FOOT_SITE_NAMES[0]),
    _foot_scan(FOOT_SCAN_RIGHT, FOOT_SITE_NAMES[1]),
    feet_ground_contact,
    body_terrain_contact,
    self_collision,
  )
```

- [ ] **Step 4: Implement the scene and physics**

`g1_stairs/scene/scene.py`:
```python
"""Scene assembly and physics settings shared by every task (spec §5)."""

from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg

from g1_stairs.scene.robot import ROBOT_NAME, get_robot_cfg
from g1_stairs.scene.sensors import make_sensors
from g1_stairs.scene.terrain import make_terrain_cfg

DEFAULT_NUM_ENVS = 4096
PLAY_NUM_ENVS = 16
DECIMATION = 4  # 200 Hz physics, 50 Hz policy
EPISODE_LENGTH_S = 20.0

# Memory budget, measured on an RTX 4060 Laptop (8 GB) at 4096 envs on 2026-09-25:
# peak about 6.7 GB; anything above about 7.6 GB fails. The ray-cast structure
# copies every geom into every world (about 150-250 KB per geom at 4096 envs),
# so the scene's geom count is capped.
MEASURED_SCENE_GEOMS = 3462
GEOM_BUDGET = 4000


def make_scene_cfg(play: bool = False, num_envs: int | None = None) -> SceneCfg:
  """Robot + stairs terrain + sensors. ``num_envs`` defaults to 4096 (train) or 16 (play)."""
  if num_envs is None:
    num_envs = PLAY_NUM_ENVS if play else DEFAULT_NUM_ENVS
  return SceneCfg(
    num_envs=num_envs,
    terrain=make_terrain_cfg(play=play),
    entities={ROBOT_NAME: get_robot_cfg()},
    sensors=make_sensors(),
    extent=2.0,
  )


def make_sim_cfg() -> SimulationCfg:
  """Unitree's rough-terrain settings, except ccd_iterations 500 -> 50 (spec §5.4)."""
  return SimulationCfg(
    nconmax=48,
    njmax=1500,
    contact_sensor_maxmatch=500,
    mujoco=MujocoCfg(
      timestep=0.005,
      iterations=10,
      ls_iterations=20,
      ccd_iterations=50,
    ),
  )
```

- [ ] **Step 5: Run the tests and watch them pass**

Run: `python -m pytest tests/test_scene.py tests/test_ray_filter.py -v`
Expected: 14 passed. `test_geom_budget` should compile about 3,462 geoms; if it reports a different count below the budget, update `MEASURED_SCENE_GEOMS` to the reported number.

- [ ] **Step 6: Commit**

```bash
git add g1_stairs/scene/sensors.py g1_stairs/scene/scene.py tests/test_scene.py tests/test_ray_filter.py
git commit -m "Add scene sensors, scene assembly and physics settings

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Shared MDP library, runner and the baseline task

**Files:**
- Create (copied): `g1_stairs/mdp/rewards.py`, `g1_stairs/mdp/observations.py`, `g1_stairs/mdp/curriculums.py`, `g1_stairs/mdp/terminations.py`
- Create: `g1_stairs/mdp/__init__.py`, `g1_stairs/runner.py`
- Create: `g1_stairs/tasks/baseline/__init__.py`, `g1_stairs/tasks/baseline/env_cfg.py`, `g1_stairs/tasks/baseline/rl_cfg.py`
- Test: `tests/test_tasks.py`

**Interfaces:**
- Consumes: `g1_stairs.scene.robot`, `g1_stairs.scene.sensors`, `g1_stairs.scene.scene` (Tasks 2–4).
- Produces:
  - `g1_stairs.mdp`, which re-exports `mjlab.envs.mdp.*` plus Unitree's terms
  - `g1_stairs.runner.VelocityOnPolicyRunner`
  - `g1_stairs.tasks.baseline.TASK_ID = "G1-Stairs-Baseline"`
  - `baseline_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg`
  - `baseline_ppo_cfg() -> RslRlOnPolicyRunnerCfg`

- [ ] **Step 1: Write the failing test**

`tests/test_tasks.py`:
```python
"""G1-Stairs-Baseline: Unitree's Rough MDP on the shared scene (spec §6.1)."""

import pytest
from mjlab.tasks.registry import (
  list_tasks,
  load_env_cfg,
  load_rl_cfg,
  load_runner_cls,
  register_mjlab_task,
)

import g1_stairs.tasks  # noqa: F401  (registers the tasks)
from g1_stairs.runner import VelocityOnPolicyRunner

TASK = "G1-Stairs-Baseline"

UNITREE_REWARD_WEIGHTS = {
  "track_linear_velocity": 1.0,
  "track_angular_velocity": 1.0,
  "body_orientation_l2": -1.0,
  "pose": 1.0,
  "body_ang_vel": -0.05,
  "angular_momentum": -0.025,
  "is_terminated": -200.0,
  "joint_acc_l2": -2.5e-7,
  "joint_pos_limits": -10.0,
  "action_rate_l2": -0.05,
  "foot_gait": 0.5,
  "foot_clearance": -1.0,
  "foot_slip": -0.25,
  "soft_landing": -1e-3,
  "stand_still": -1.0,
  "self_collisions": -1.0,
}
UNITREE_ACTOR_TERMS = [
  "base_ang_vel", "projected_gravity", "command", "phase",
  "joint_pos", "joint_vel", "actions", "height_scan",
]


def test_registered():
  assert TASK in list_tasks()


def test_training_config():
  cfg = load_env_cfg(TASK)
  assert cfg.scene.num_envs == 4096
  assert cfg.sim.mujoco.ccd_iterations == 50
  assert cfg.decimation == 4
  assert cfg.episode_length_s == 20.0
  assert cfg.scene.terrain.terrain_generator.curriculum is True
  assert set(cfg.curriculum) == {"terrain_levels", "command_vel"}
  assert "push_robot" in cfg.events


def test_unitree_rewards_and_observations_unchanged():
  cfg = load_env_cfg(TASK)
  assert {k: v.weight for k, v in cfg.rewards.items()} == UNITREE_REWARD_WEIGHTS
  assert list(cfg.observations["actor"].terms) == UNITREE_ACTOR_TERMS
  assert cfg.commands["twist"].ranges.lin_vel_x == (-1.0, 2.0)


def test_play_config():
  cfg = load_env_cfg(TASK, play=True)
  assert cfg.scene.num_envs == 16
  assert cfg.scene.terrain.terrain_generator.curriculum is False
  assert cfg.observations["actor"].enable_corruption is False
  assert "push_robot" not in cfg.events
  assert "randomize_terrain" in cfg.events
  assert cfg.curriculum == {}


def test_ppo_config():
  rl = load_rl_cfg(TASK)
  assert rl.logger == "tensorboard"
  assert rl.experiment_name == "g1_stairs_baseline"
  assert rl.max_iterations == 5000
  assert rl.num_steps_per_env == 24
  assert rl.actor.hidden_dims == (512, 256, 128)
  assert load_runner_cls(TASK) is VelocityOnPolicyRunner


def test_duplicate_task_id_fails_loudly():
  # A copied baseline folder that keeps the old task ID must not be silently ignored.
  with pytest.raises(ValueError, match=TASK):
    register_mjlab_task(
      task_id=TASK,
      env_cfg=load_env_cfg(TASK),
      play_env_cfg=load_env_cfg(TASK, play=True),
      rl_cfg=load_rl_cfg(TASK),
    )
```

- [ ] **Step 2: Run it and watch it fail**

Run: `python -m pytest tests/test_tasks.py -v`
Expected: `ModuleNotFoundError: No module named 'g1_stairs.runner'`.

- [ ] **Step 3: Copy the MDP library**

```bash
mkdir -p g1_stairs/mdp
for f in rewards observations curriculums terminations; do
  cp external/unitree_rl_mjlab/src/tasks/velocity/mdp/$f.py g1_stairs/mdp/$f.py
  sed -i "1i # Derived from unitree_rl_mjlab src/tasks/velocity/mdp/$f.py (Apache-2.0)." g1_stairs/mdp/$f.py
done
sed -i 's/^from \.velocity_command import UniformVelocityCommandCfg/from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg/' g1_stairs/mdp/curriculums.py
grep -n "UniformVelocityCommandCfg" g1_stairs/mdp/curriculums.py | head -2
```
Expected: the grep shows `from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg`.

`g1_stairs/mdp/__init__.py`:
```python
"""Shared MDP term library: rewards, observations, curriculums, terminations.

Derived from unitree_rl_mjlab src/tasks/velocity/mdp (Apache-2.0). Also re-exports
mjlab.envs.mdp. Put terms that several variants may use here; keep terms used by
one variant in that variant's folder.
"""

from mjlab.envs.mdp import *  # noqa: F401, F403

from .curriculums import *  # noqa: F403
from .observations import *  # noqa: F403
from .rewards import *  # noqa: F403
from .terminations import *  # noqa: F403
```

- [ ] **Step 4: Add the runner**

`g1_stairs/runner.py`:
```python
# Derived from unitree_rl_mjlab src/tasks/velocity/rl/runner.py (Apache-2.0).
"""PPO runner that also exports policy.onnx (with metadata) on every save."""

import os

import wandb
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
from mjlab.rl.runner import MjlabOnPolicyRunner


class VelocityOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper

  def save(self, path: str, infos=None):
    super().save(path, infos)
    policy_path = path.split("model")[0]
    filename = "policy.onnx"
    self.export_policy_to_onnx(policy_path, filename)
    run_name: str = (
      wandb.run.name if self.logger.logger_type == "wandb" and wandb.run else "local"
    )  # type: ignore[assignment]
    onnx_path = os.path.join(policy_path, filename)
    metadata = get_base_metadata(self.env.unwrapped, run_name)
    attach_metadata_to_onnx(onnx_path, metadata)
    if self.logger.logger_type in ["wandb"]:
      wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))
```

- [ ] **Step 5: Write the baseline PPO config**

`g1_stairs/tasks/baseline/rl_cfg.py`:
```python
# Derived from unitree_rl_mjlab src/tasks/velocity/config/g1_23dof/rl_cfg.py (Apache-2.0).
"""PPO settings for G1-Stairs-Baseline: Unitree's, with local logging and 5000 iterations."""

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


def baseline_ppo_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="g1_stairs_baseline",
    logger="tensorboard",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=5000,
  )
```

- [ ] **Step 6: Write the baseline environment config**

`g1_stairs/tasks/baseline/env_cfg.py`:
```python
# Derived from unitree_rl_mjlab src/tasks/velocity/velocity_env_cfg.py and
# src/tasks/velocity/config/g1_23dof/env_cfgs.py (Apache-2.0).
"""G1-Stairs-Baseline: Unitree's Unitree-G1-23Dof-Rough MDP on the shared stairs scene.

Rewards, weights, observations, events, terminations and curricula are Unitree's,
unchanged (spec §6.1). Only the scene comes from g1_stairs.scene. This is the
template for variants: copy this folder and change what you need.
"""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

import g1_stairs.mdp as mdp
from g1_stairs.scene.robot import (
  FOOT_GEOM_NAMES,
  FOOT_SITE_NAMES,
  ROBOT_NAME,
  TORSO_BODY,
  make_actions,
)
from g1_stairs.scene.scene import (
  DECIMATION,
  EPISODE_LENGTH_S,
  make_scene_cfg,
  make_sim_cfg,
)
from g1_stairs.scene.sensors import (
  FEET_GROUND_CONTACT,
  SELF_COLLISION,
  TERRAIN_SCAN,
  TERRAIN_SCAN_MAX_DISTANCE,
)

# Unitree's per-joint posture tolerances for the G1 23-DOF.
_POSE_STD_WALKING = {
  r".*hip_pitch.*": 0.5,
  r".*hip_roll.*": 0.15,
  r".*hip_yaw.*": 0.15,
  r".*knee.*": 0.5,
  r".*ankle_pitch.*": 0.15,
  r".*ankle_roll.*": 0.1,
  r".*waist_yaw.*": 0.15,
  r".*shoulder_pitch.*": 0.15,
  r".*shoulder_roll.*": 0.1,
  r".*shoulder_yaw.*": 0.1,
  r".*elbow.*": 0.1,
  r".*wrist.*": 0.1,
}
_POSE_STD_RUNNING = {
  r".*hip_pitch.*": 0.5,
  r".*hip_roll.*": 0.25,
  r".*hip_yaw.*": 0.25,
  r".*knee.*": 0.5,
  r".*ankle_pitch.*": 0.25,
  r".*ankle_roll.*": 0.1,
  r".*waist_yaw.*": 0.25,
  r".*shoulder_pitch.*": 0.25,
  r".*shoulder_roll.*": 0.1,
  r".*shoulder_yaw.*": 0.1,
  r".*elbow.*": 0.1,
  r".*wrist.*": 0.1,
}


def _observations() -> dict[str, ObservationGroupCfg]:
  actor_terms = {
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "command": ObservationTermCfg(
      func=mdp.generated_commands,
      params={"command_name": "twist"},
    ),
    "phase": ObservationTermCfg(
      func=mdp.phase,
      params={"period": 0.6, "command_name": "twist"},
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-1.5, n_max=1.5),
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
    "height_scan": ObservationTermCfg(
      func=envs_mdp.height_scan,
      params={"sensor_name": TERRAIN_SCAN},
      noise=Unoise(n_min=-0.1, n_max=0.1),
      scale=1 / TERRAIN_SCAN_MAX_DISTANCE,
    ),
  }
  critic_terms = {
    **actor_terms,
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
      noise=Unoise(n_min=-0.5, n_max=0.5),
    ),
    "height_scan": ObservationTermCfg(
      func=envs_mdp.height_scan,
      params={"sensor_name": TERRAIN_SCAN},
      scale=1 / TERRAIN_SCAN_MAX_DISTANCE,
    ),
    "foot_height": ObservationTermCfg(
      func=mdp.foot_height,
      params={"asset_cfg": SceneEntityCfg(ROBOT_NAME, site_names=FOOT_SITE_NAMES)},
    ),
    "foot_air_time": ObservationTermCfg(
      func=mdp.foot_air_time,
      params={"sensor_name": FEET_GROUND_CONTACT},
    ),
    "foot_contact": ObservationTermCfg(
      func=mdp.foot_contact,
      params={"sensor_name": FEET_GROUND_CONTACT},
    ),
    "foot_contact_forces": ObservationTermCfg(
      func=mdp.foot_contact_forces,
      params={"sensor_name": FEET_GROUND_CONTACT},
    ),
  }
  return {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
      history_length=1,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
      history_length=1,
    ),
  }


def _commands() -> dict[str, CommandTermCfg]:
  twist = UniformVelocityCommandCfg(
    entity_name=ROBOT_NAME,
    resampling_time_range=(3.0, 8.0),
    rel_standing_envs=0.05,
    heading_command=True,
    heading_control_stiffness=0.5,
    debug_vis=True,
    ranges=UniformVelocityCommandCfg.Ranges(
      lin_vel_x=(-1.0, 2.0),
      lin_vel_y=(-1.0, 1.0),
      ang_vel_z=(-1.0, 1.0),
      heading=(-math.pi, math.pi),
    ),
  )
  twist.viz.z_offset = 1.15
  return {"twist": twist}


def _events() -> dict[str, EventTermCfg]:
  return {
    "reset_base": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "z": (0.0, 0.0),
          "yaw": (-3.14, 3.14),
        },
        "velocity_range": {},
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.0, 0.0),
        "velocity_range": (-0.0, 0.0),
        "asset_cfg": SceneEntityCfg(ROBOT_NAME, joint_names=(".*",)),
      },
    ),
    "push_robot": EventTermCfg(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(5.0, 6.0),
      params={
        "velocity_range": {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "z": (-0.4, 0.4),
          "roll": (-0.52, 0.52),
          "pitch": (-0.52, 0.52),
          "yaw": (-0.78, 0.78),
        },
      },
    ),
    "foot_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg(ROBOT_NAME, geom_names=FOOT_GEOM_NAMES),
        "operation": "abs",
        "ranges": (0.3, 1.6),
        "shared_random": True,
      },
    ),
    "encoder_bias": EventTermCfg(
      mode="startup",
      func=dr.encoder_bias,
      params={
        "asset_cfg": SceneEntityCfg(ROBOT_NAME),
        "bias_range": (-0.015, 0.015),
      },
    ),
    "base_com": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg(ROBOT_NAME, body_names=(TORSO_BODY,)),
        "operation": "add",
        "ranges": {0: (-0.05, 0.05), 1: (-0.05, 0.05), 2: (-0.05, 0.05)},
      },
    ),
  }


def _rewards() -> dict[str, RewardTermCfg]:
  feet = SceneEntityCfg(ROBOT_NAME, site_names=FOOT_SITE_NAMES)
  torso = SceneEntityCfg(ROBOT_NAME, body_names=(TORSO_BODY,))
  return {
    "track_linear_velocity": RewardTermCfg(
      func=mdp.track_linear_velocity,
      weight=1.0,
      params={"command_name": "twist", "std": math.sqrt(0.25)},
    ),
    "track_angular_velocity": RewardTermCfg(
      func=mdp.track_angular_velocity,
      weight=1.0,
      params={"command_name": "twist", "std": math.sqrt(0.5)},
    ),
    "body_orientation_l2": RewardTermCfg(
      func=mdp.body_orientation_l2,
      weight=-1.0,
      params={"asset_cfg": torso},
    ),
    "pose": RewardTermCfg(
      func=mdp.variable_posture,
      weight=1.0,
      params={
        "asset_cfg": SceneEntityCfg(ROBOT_NAME, joint_names=".*"),
        "command_name": "twist",
        "std_standing": {".*": 0.05},
        "std_walking": _POSE_STD_WALKING,
        "std_running": _POSE_STD_RUNNING,
        "walking_threshold": 0.1,
        "running_threshold": 1.5,
      },
    ),
    "body_ang_vel": RewardTermCfg(
      func=mdp.body_angular_velocity_penalty,
      weight=-0.05,
      params={"asset_cfg": SceneEntityCfg(ROBOT_NAME, body_names=(TORSO_BODY,))},
    ),
    "angular_momentum": RewardTermCfg(
      func=mdp.angular_momentum_penalty,
      weight=-0.025,
      params={"sensor_name": "robot/root_angmom"},
    ),
    "is_terminated": RewardTermCfg(func=mdp.is_terminated, weight=-200.0),
    "joint_acc_l2": RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.5e-7),
    "joint_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-10.0),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.05),
    "foot_gait": RewardTermCfg(
      func=mdp.feet_gait,
      weight=0.5,
      params={
        "period": 0.6,
        "offset": [0.0, 0.5],
        "threshold": 0.56,
        "command_threshold": 0.1,
        "command_name": "twist",
        "sensor_name": FEET_GROUND_CONTACT,
      },
    ),
    # Known issue (spec §9): target is absolute world height, wrong on stairs.
    "foot_clearance": RewardTermCfg(
      func=mdp.feet_clearance,
      weight=-1.0,
      params={
        "target_height": 0.10,
        "command_name": "twist",
        "command_threshold": 0.1,
        "asset_cfg": feet,
      },
    ),
    "foot_slip": RewardTermCfg(
      func=mdp.feet_slip,
      weight=-0.25,
      params={
        "sensor_name": FEET_GROUND_CONTACT,
        "command_name": "twist",
        "command_threshold": 0.1,
        "asset_cfg": SceneEntityCfg(ROBOT_NAME, site_names=FOOT_SITE_NAMES),
      },
    ),
    "soft_landing": RewardTermCfg(
      func=mdp.soft_landing,
      weight=-1e-3,
      params={
        "sensor_name": FEET_GROUND_CONTACT,
        "command_name": "twist",
        "command_threshold": 0.1,
      },
    ),
    "stand_still": RewardTermCfg(
      func=mdp.stand_still,
      weight=-1.0,
      params={
        "command_name": "twist",
        "command_threshold": 0.1,
        "asset_cfg": SceneEntityCfg(ROBOT_NAME, joint_names=".*"),
      },
    ),
    "self_collisions": RewardTermCfg(
      func=mdp.self_collision_cost,
      weight=-1.0,
      params={"sensor_name": SELF_COLLISION, "force_threshold": 10.0},
    ),
  }


def _terminations() -> dict[str, TerminationTermCfg]:
  return {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(
      func=mdp.bad_orientation,
      params={"limit_angle": math.radians(70.0)},
    ),
  }


def _curriculum() -> dict[str, CurriculumTermCfg]:
  return {
    "terrain_levels": CurriculumTermCfg(
      func=mdp.terrain_levels_vel,
      params={"command_name": "twist"},
    ),
    "command_vel": CurriculumTermCfg(
      func=mdp.commands_vel,
      params={
        "command_name": "twist",
        "velocity_stages": [
          {"step": 0, "lin_vel_x": (-0.5, 1.0), "lin_vel_y": (-0.5, 0.5), "ang_vel_z": (-1.0, 1.0)},
          {"step": 5000 * 24, "lin_vel_x": (-1.0, 2.0), "lin_vel_y": (-1.0, 1.0)},
        ],
      },
    ),
  }


def baseline_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = ManagerBasedRlEnvCfg(
    scene=make_scene_cfg(play=play),
    observations=_observations(),
    actions=make_actions(),
    commands=_commands(),
    events=_events(),
    rewards=_rewards(),
    terminations=_terminations(),
    curriculum=_curriculum(),
    metrics={"mean_action_acc": MetricsTermCfg(func=mdp.mean_action_acc)},
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name=ROBOT_NAME,
      body_name=TORSO_BODY,
      distance=3.0,
      elevation=-5.0,
      azimuth=90.0,
    ),
    sim=make_sim_cfg(),
    decimation=DECIMATION,
    episode_length_s=EPISODE_LENGTH_S,
  )
  if play:
    cfg.episode_length_s = int(1e9)  # effectively endless
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )
  return cfg
```

`g1_stairs/tasks/baseline/__init__.py`:
```python
"""G1-Stairs-Baseline: registers the task. Copy this folder to start a variant,
then change TASK_ID here and experiment_name in rl_cfg.py."""

from mjlab.tasks.registry import register_mjlab_task

from g1_stairs.runner import VelocityOnPolicyRunner

from .env_cfg import baseline_env_cfg
from .rl_cfg import baseline_ppo_cfg

TASK_ID = "G1-Stairs-Baseline"

register_mjlab_task(
  task_id=TASK_ID,
  env_cfg=baseline_env_cfg(),
  play_env_cfg=baseline_env_cfg(play=True),
  rl_cfg=baseline_ppo_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
```

- [ ] **Step 7: Run the tests and watch them pass**

Run: `python -m pytest tests/test_tasks.py -v`
Expected: 6 passed.

- [ ] **Step 8: Run the whole suite**

Run: `python -m pytest -q`
Expected: 36 passed (all tests from Tasks 1–5).

- [ ] **Step 9: Commit**

```bash
git add g1_stairs/mdp g1_stairs/runner.py g1_stairs/tasks/baseline tests/test_tasks.py
git commit -m "Add shared MDP library, ONNX-exporting runner and G1-Stairs-Baseline task

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Runtime helpers and scripts

**Files:**
- Create: `g1_stairs/runtime.py`
- Create: `scripts/train.py`, `scripts/play.py`, `scripts/list_envs.py`, `scripts/view_scene.py`
- Test: `tests/test_runtime.py`

**Interfaces:**
- Consumes: `g1_stairs.REPO_ROOT` (Task 1), `g1_stairs.tasks` (Task 5), `g1_stairs.scene.terrain.make_terrain_cfg` (Task 3).
- Produces, in `g1_stairs.runtime`:
  - `LOG_ROOT: Path`
  - `configure_render_backend() -> None`, `default_viewer() -> str`, `resolve_device(device: str | None) -> str`
  - `zero_policy(env)`, `random_policy(env)`, `run_viewer(env, policy, viewer: str) -> None`

- [ ] **Step 1: Write the failing test**

`tests/test_runtime.py`:
```python
"""Script helpers: Windows-safe rendering, viewer choice, cwd-independent paths (spec §7)."""

import os
import subprocess
import sys

from g1_stairs import REPO_ROOT, runtime


def test_render_backend_is_not_forced_on_windows(monkeypatch):
  monkeypatch.setattr(sys, "platform", "win32")
  monkeypatch.delenv("MUJOCO_GL", raising=False)
  runtime.configure_render_backend()
  assert "MUJOCO_GL" not in os.environ


def test_render_backend_is_egl_on_linux(monkeypatch):
  monkeypatch.setattr(sys, "platform", "linux")
  monkeypatch.delenv("MUJOCO_GL", raising=False)
  runtime.configure_render_backend()
  assert os.environ["MUJOCO_GL"] == "egl"


def test_render_backend_respects_user_choice(monkeypatch):
  monkeypatch.setattr(sys, "platform", "linux")
  monkeypatch.setenv("MUJOCO_GL", "osmesa")
  runtime.configure_render_backend()
  assert os.environ["MUJOCO_GL"] == "osmesa"


def test_default_viewer(monkeypatch):
  monkeypatch.setattr(sys, "platform", "win32")
  monkeypatch.delenv("DISPLAY", raising=False)
  assert runtime.default_viewer() == "native"
  monkeypatch.setattr(sys, "platform", "linux")
  monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
  assert runtime.default_viewer() == "viser"
  monkeypatch.setenv("DISPLAY", ":0")
  assert runtime.default_viewer() == "native"


def test_resolve_device():
  assert runtime.resolve_device("cpu") == "cpu"
  assert runtime.resolve_device(None) in ("cuda:0", "cpu")


def test_logs_live_in_the_repo():
  assert runtime.LOG_ROOT == REPO_ROOT / "logs" / "rsl_rl"


def test_list_envs_script_from_another_directory(tmp_path):
  result = subprocess.run(
    [sys.executable, str(REPO_ROOT / "scripts" / "list_envs.py")],
    cwd=tmp_path,
    capture_output=True,
    text=True,
    timeout=300,
  )
  assert result.returncode == 0, result.stderr
  assert "G1-Stairs-Baseline" in result.stdout
```

- [ ] **Step 2: Run it and watch it fail**

Run: `python -m pytest tests/test_runtime.py -v`
Expected: `ImportError: cannot import name 'runtime' from 'g1_stairs'`.

- [ ] **Step 3: Implement the runtime helpers**

`g1_stairs/runtime.py`:
```python
"""Helpers shared by scripts/*.py: log paths, rendering backend, viewer, dummy policies."""

import os
import sys
from pathlib import Path

import torch

from g1_stairs import REPO_ROOT

LOG_ROOT: Path = REPO_ROOT / "logs" / "rsl_rl"


def configure_render_backend() -> None:
  """EGL for headless rendering on Linux. Windows and macOS keep MuJoCo's default.

  Upstream unitree_rl_mjlab sets MUJOCO_GL=egl on every platform, which breaks
  rendering on Windows. A value the user already set always wins.
  """
  if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")


def default_viewer() -> str:
  """'native' (MuJoCo window) on Windows or with a display, else 'viser' (browser)."""
  if sys.platform == "win32":
    return "native"
  has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
  return "native" if has_display else "viser"


def resolve_device(device: str | None) -> str:
  return device or ("cuda:0" if torch.cuda.is_available() else "cpu")


def zero_policy(env):
  """Policy that always outputs 0: the robot holds its default pose."""
  shape = env.unwrapped.action_space.shape
  device = env.unwrapped.device

  def policy(obs):
    del obs
    return torch.zeros(shape, device=device)

  return policy


def random_policy(env):
  """Policy that outputs uniform noise in [-1, 1]."""
  shape = env.unwrapped.action_space.shape
  device = env.unwrapped.device

  def policy(obs):
    del obs
    return 2 * torch.rand(shape, device=device) - 1

  return policy


def run_viewer(env, policy, viewer: str) -> None:
  from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

  if viewer == "native":
    NativeMujocoViewer(env, policy).run()
  elif viewer == "viser":
    ViserPlayViewer(env, policy).run()
  else:
    raise ValueError(f"Unknown viewer {viewer!r}; use 'native' or 'viser'.")
```

- [ ] **Step 4: Write the scripts**

`scripts/list_envs.py`:
```python
# Derived from unitree_rl_mjlab scripts/list_envs.py (Apache-2.0).
"""List registered tasks. Usage: python scripts/list_envs.py [--keyword G1-Stairs]"""

import mjlab
import tyro
from mjlab.tasks.registry import list_tasks
from prettytable import PrettyTable

import g1_stairs.tasks  # noqa: F401  (registers the G1-Stairs-* tasks)


def list_environments(keyword: str = "G1-Stairs") -> int:
  """Print registered task IDs containing ``keyword`` (use "" for all)."""
  table = PrettyTable(["#", "Task ID"])
  table.title = "Registered tasks"
  table.align["Task ID"] = "l"
  tasks = [t for t in list_tasks() if keyword.lower() in t.lower()]
  for i, task in enumerate(tasks, start=1):
    table.add_row([i, task])
  print(table)
  return len(tasks)


def main() -> None:
  tyro.cli(list_environments, config=mjlab.TYRO_FLAGS)


if __name__ == "__main__":
  main()
```

`scripts/train.py`:
```python
# Derived from unitree_rl_mjlab scripts/train.py (Apache-2.0).
"""Train a task with RSL-RL PPO on one GPU.

Usage:
  python scripts/train.py G1-Stairs-Baseline
  python scripts/train.py G1-Stairs-Baseline --agent.max-iterations 30 --env.scene.num-envs 2048

Changes from upstream: single GPU only, no motion-tracking tasks, logs under
<repo>/logs/rsl_rl/<experiment>/<timestamp>, MUJOCO_GL=egl only on Linux.
"""

import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import mjlab
import tyro
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.os import dump_yaml, get_checkpoint_path
from mjlab.utils.torch import configure_torch_backends

import g1_stairs.tasks  # noqa: F401  (registers the G1-Stairs-* tasks)
from g1_stairs.runtime import LOG_ROOT, configure_render_backend, resolve_device


@dataclass(frozen=True)
class TrainConfig:
  env: ManagerBasedRlEnvCfg
  agent: RslRlBaseRunnerCfg
  device: str | None = None
  """Device, e.g. 'cuda:0'. Default: cuda:0 if available, else cpu."""
  enable_nan_guard: bool = False

  @staticmethod
  def from_task(task_id: str) -> "TrainConfig":
    return TrainConfig(env=load_env_cfg(task_id), agent=load_rl_cfg(task_id))


def run_train(task_id: str, cfg: TrainConfig) -> Path:
  configure_render_backend()
  configure_torch_backends()
  device = resolve_device(cfg.device)
  cfg.env.seed = cfg.agent.seed
  if cfg.enable_nan_guard:
    cfg.env.sim.nan_guard.enabled = True

  experiment_dir = LOG_ROOT / cfg.agent.experiment_name
  run_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  if cfg.agent.run_name:
    run_dir_name += f"_{cfg.agent.run_name}"
  log_dir = experiment_dir / run_dir_name
  print(f"[INFO] Training {task_id} on {device}, logging to {log_dir}")

  env = ManagerBasedRlEnv(cfg=cfg.env, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)
  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(cfg.agent), str(log_dir), device)
  runner.add_git_repo_to_log(__file__)
  if cfg.agent.resume:
    resume_path = get_checkpoint_path(
      experiment_dir, cfg.agent.load_run, cfg.agent.load_checkpoint
    )
    print(f"[INFO] Resuming from {resume_path}")
    runner.load(str(resume_path))
  dump_yaml(log_dir / "params" / "env.yaml", asdict(cfg.env))
  dump_yaml(log_dir / "params" / "agent.yaml", asdict(cfg.agent))
  runner.learn(
    num_learning_iterations=cfg.agent.max_iterations, init_at_random_ep_len=True
  )
  env.close()
  return log_dir


def main() -> None:
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(list_tasks()),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )
  args = tyro.cli(
    TrainConfig,
    args=remaining_args,
    default=TrainConfig.from_task(chosen_task),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  run_train(chosen_task, args)


if __name__ == "__main__":
  main()
```

`scripts/play.py`:
```python
# Derived from unitree_rl_mjlab scripts/play.py (Apache-2.0).
"""Run a policy in the viewer, optionally recording a video.

Usage:
  python scripts/play.py G1-Stairs-Baseline --checkpoint-file logs/rsl_rl/<exp>/<run>/model_<n>.pt
  python scripts/play.py G1-Stairs-Baseline --checkpoint-file <...> --video --video-length 300
  python scripts/play.py G1-Stairs-Baseline --agent zero

Changes from upstream: no tracking tasks or W&B downloads; --viewer auto picks the
native window on Windows; MUJOCO_GL=egl only on Linux.
"""

import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import mjlab
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder

import g1_stairs.tasks  # noqa: F401  (registers the G1-Stairs-* tasks)
from g1_stairs.runtime import (
  configure_render_backend,
  default_viewer,
  random_policy,
  resolve_device,
  run_viewer,
  zero_policy,
)


@dataclass(frozen=True)
class PlayConfig:
  agent: Literal["zero", "random", "trained"] = "trained"
  checkpoint_file: str | None = None
  num_envs: int | None = None
  device: str | None = None
  video: bool = False
  """Record the first video_length steps to <run>/videos/play (trained agent only)."""
  video_length: int = 200
  viewer: Literal["auto", "native", "viser"] = "auto"
  no_terminations: bool = False


def run_play(task_id: str, cfg: PlayConfig) -> None:
  configure_render_backend()
  configure_torch_backends()
  device = resolve_device(cfg.device)
  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)
  if cfg.no_terminations:
    env_cfg.terminations = {}
  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs

  checkpoint: Path | None = None
  if cfg.agent == "trained":
    if cfg.checkpoint_file is None:
      raise SystemExit("--checkpoint-file is required for a trained agent (or use --agent zero).")
    checkpoint = Path(cfg.checkpoint_file).resolve()
    if not checkpoint.is_file():
      raise SystemExit(f"Checkpoint not found: {checkpoint}")
  if cfg.video and checkpoint is None:
    print("[WARN] --video needs a trained checkpoint; recording disabled.")
  record = cfg.video and checkpoint is not None

  env = ManagerBasedRlEnv(
    cfg=env_cfg, device=device, render_mode="rgb_array" if record else None
  )
  if record:
    assert checkpoint is not None
    env = VideoRecorder(
      env,
      video_folder=checkpoint.parent / "videos" / "play",
      step_trigger=lambda step: step == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  if checkpoint is None:
    policy = random_policy(env) if cfg.agent == "random" else zero_policy(env)
  else:
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)

  run_viewer(env, policy, default_viewer() if cfg.viewer == "auto" else cfg.viewer)
  env.close()


def main() -> None:
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(list_tasks()),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )
  args = tyro.cli(
    PlayConfig,
    args=remaining_args,
    default=PlayConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  run_play(chosen_task, args)


if __name__ == "__main__":
  main()
```

`scripts/view_scene.py`:
```python
"""Show the full training terrain with robots on every difficulty row (zero policy).

Usage: python scripts/view_scene.py [--num-envs 64] [--viewer auto|native|viser]
"""

from dataclasses import dataclass
from typing import Literal

import mjlab
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends

import g1_stairs.tasks  # noqa: F401  (registers the G1-Stairs-* tasks)
from g1_stairs.runtime import (
  configure_render_backend,
  default_viewer,
  resolve_device,
  run_viewer,
  zero_policy,
)
from g1_stairs.scene.terrain import make_terrain_cfg


@dataclass(frozen=True)
class ViewConfig:
  num_envs: int = 64
  viewer: Literal["auto", "native", "viser"] = "auto"
  device: str | None = None
  task: str = "G1-Stairs-Baseline"


def main() -> None:
  cfg = tyro.cli(ViewConfig, config=mjlab.TYRO_FLAGS)
  configure_render_backend()
  configure_torch_backends()
  env_cfg = load_env_cfg(cfg.task, play=True)
  env_cfg.scene.terrain = make_terrain_cfg(play=False)  # full 10 x 20 curriculum grid
  env_cfg.scene.terrain.max_init_terrain_level = None  # robots on every row
  env_cfg.events.pop("randomize_terrain", None)
  env_cfg.scene.num_envs = cfg.num_envs
  env = ManagerBasedRlEnv(cfg=env_cfg, device=resolve_device(cfg.device))
  env = RslRlVecEnvWrapper(env, clip_actions=load_rl_cfg(cfg.task).clip_actions)
  run_viewer(env, zero_policy(env), default_viewer() if cfg.viewer == "auto" else cfg.viewer)
  env.close()


if __name__ == "__main__":
  main()
```

- [ ] **Step 5: Run the tests and watch them pass**

Run: `python -m pytest tests/test_runtime.py -v`
Expected: 7 passed. (`test_list_envs_script_from_another_directory` takes up to a minute.)

- [ ] **Step 6: Check the CLIs parse**

Run: `python scripts/train.py G1-Stairs-Baseline --help` and `python scripts/play.py G1-Stairs-Baseline --help`
Expected: both print tyro help (train shows `--agent.max-iterations`, play shows `--checkpoint-file`) and exit 0.

- [ ] **Step 7: Commit**

```bash
git add g1_stairs/runtime.py scripts tests/test_runtime.py
git commit -m "Add Windows-safe train, play, list_envs and view_scene scripts

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Acceptance run, lock file and README

**Files:**
- Create: `requirements.lock.txt`
- Modify: `README.md` (full rewrite)

**Interfaces:**
- Consumes: everything above.

- [ ] **Step 1: Full test suite**

Run: `python -m pytest -q`
Expected: 43 passed.

- [ ] **Step 2: Acceptance training run (GPU)**

Close GPU-heavy apps. Sample GPU memory in the background while training:
```bash
: > gpu_accept.csv
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> gpu_accept.csv; sleep 3; done ) & MON=$!
python scripts/train.py G1-Stairs-Baseline --agent.max-iterations 30 > train_accept.log 2>&1; echo "exit $?"
kill $MON
grep -E "Steps per second|Iteration time" train_accept.log | tail -2
sort -n gpu_accept.csv | tail -1
ls logs/rsl_rl/g1_stairs_baseline/*/
```
Expected:
- `exit 0`
- steps per second of about 14–15k, and peak memory at most about 7,000 MiB
- the run folder contains `model_29.pt` (or the final `model_<n>.pt`), `policy.onnx` and `params/env.yaml`

Note the numbers for Step 5. Delete `gpu_accept.csv` and `train_accept.log` afterwards; they are not committed.

- [ ] **Step 3: Play and scene checks (manual, needs a person at the screen)**

Run: `python scripts/play.py G1-Stairs-Baseline --checkpoint-file <path to model_29.pt from Step 2> --num-envs 4`
Expected: a MuJoCo window opens with 4 robots on play terrain. Close it; the process exits 0.

Run: `python scripts/view_scene.py`
Expected: a window with the full stairs grid and 64 robots. Close it.

- [ ] **Step 4: Lock file**

```bash
( echo "# Exact versions from the passing acceptance run ($(date +%F)). Install with:"
  echo "#   pip install -r requirements.lock.txt && pip install -e . --no-deps"
  echo "--extra-index-url https://download.pytorch.org/whl/cu128"
  python -m pip freeze --exclude-editable ) > requirements.lock.txt
grep -E "^(torch|torchvision|mjlab|mujoco|mujoco-warp|warp-lang|rsl-rl-lib)==" requirements.lock.txt
```
Expected: shows `torch==2.9.0+cu128`, `torchvision==0.24.0+cu128`, `mjlab==1.2.0`, `mujoco==3.5.0`, `mujoco-warp==3.5.0`, `warp-lang==1.12.0` and `rsl-rl-lib==5.0.1`.

- [ ] **Step 5: Rewrite README.md**

Replace `README.md` with the text below. Fill in `<STEPS>`, `<ITER_S>` and `<PEAK_GB>` with the numbers measured in Step 2.

````markdown
# g1-stairs-rl

A shared MuJoCo environment for training the **Unitree G1 humanoid (`g1_23dof_rev_1_0`) to climb stairs** with reinforcement learning. It is built on [mjlab](https://github.com/mujocolab/mjlab) (MuJoCo Warp + RSL-RL) and runs natively on Windows or Linux with an NVIDIA GPU.

The **scene** is identical for everyone: robot, stairs terrain, sensors and physics. You add your own **variant** (rewards, observations, network, PPO settings) as a separate task, so results stay comparable. Design and reasoning: [docs/superpowers/specs/2026-09-25-shared-environment-design.md](docs/superpowers/specs/2026-09-25-shared-environment-design.md).

## Install

You need an NVIDIA GPU with at least 8 GB (driver with CUDA 12.8 support), Python 3.10–3.13 and git.

```bash
git clone <repo-url> g1-stairs-rl
cd g1-stairs-rl
python -m venv .venv
.venv\Scripts\activate            # Linux: source .venv/bin/activate
pip install -r requirements.txt   # CUDA torch + pinned mjlab stack + this package
python -m pytest -q               # about a minute; all tests should pass
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

Measured on an RTX 4060 Laptop (8 GB) with 4096 envs: **<STEPS> env-steps/s** (<ITER_S> s per PPO iteration) and **<PEAK_GB> GB** peak GPU memory. About 1 GB is left over, so extra sensors, geoms or envs may need `--env.scene.num-envs 3072`. The terrain uses box geometry only, because heightfield or box-grid tiles do not fit in 8 GB (spec §2). A test enforces a budget of at most 4,000 geoms.

## Adding your variant

1. Copy `g1_stairs/tasks/baseline/` to `g1_stairs/tasks/<your_name>/`.
2. In `<your_name>/__init__.py`, set `TASK_ID = "G1-Stairs-<YourName>"`. Two folders with the same ID stop every script with `Task '...' is already registered`.
3. In `<your_name>/rl_cfg.py`, set `experiment_name="g1_stairs_<your_name>"`, so your logs get their own folder.
4. Edit `env_cfg.py`: rewards, observations, commands; and `rl_cfg.py`: network, PPO. Each reward term is a function with a weight. Put terms other variants may reuse in `g1_stairs/mdp/`; keep your own terms in your folder.
5. Run `python scripts/list_envs.py`. Your task shows up without any other edits.

Do not change `g1_stairs/scene/` in a variant. If the scene needs to change, propose it to the team, because everyone's results depend on it.

## Known issues (starting points for reward design)

The baseline is Unitree's flat and rough-terrain setup, unchanged:

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
````

- [ ] **Step 6: Commit**

```bash
git add README.md requirements.lock.txt
git commit -m "Add README, measured performance and lock file from acceptance run

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: Merge and push (needs the user)

- [ ] **Step 1: Merge to main**

```bash
git checkout main
git merge --ff-only shared-environment
git log --oneline | head -12
```

- [ ] **Step 2: Show the user what will be pushed, and ask for the remote**

List the commits and `git ls-files | wc -l`. Ask the user for the URL of an empty GitHub repository they created. The `gh` CLI is not installed, so it can't be created from here. Do not push before the user confirms.

- [ ] **Step 3: Put the clone URL in the README**

Replace `<repo-url>` in `README.md` with the URL from Step 2, then:
```bash
git add README.md
git commit -m "Add clone URL to README

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 4: Push**

```bash
git remote add origin <URL from the user>
git push -u origin main
```
Expected: `branch 'main' set up to track 'origin/main'`.
