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
