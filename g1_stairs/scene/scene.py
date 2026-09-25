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
