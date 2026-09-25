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
