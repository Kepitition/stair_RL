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
from g1_stairs.runtime import (
  LOG_ROOT,
  configure_render_backend,
  configure_utf8_output,
  resolve_device,
)


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
  configure_utf8_output()
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
