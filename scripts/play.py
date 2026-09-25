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
  configure_utf8_output,
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
  configure_utf8_output()
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
