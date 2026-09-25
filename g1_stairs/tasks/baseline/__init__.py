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
