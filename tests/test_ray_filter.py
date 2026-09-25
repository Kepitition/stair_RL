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
