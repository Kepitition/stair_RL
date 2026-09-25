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
