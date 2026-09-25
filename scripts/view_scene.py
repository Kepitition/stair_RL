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
  configure_utf8_output,
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
  configure_utf8_output()
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
