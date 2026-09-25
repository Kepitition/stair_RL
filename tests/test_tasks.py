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
