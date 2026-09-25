"""G1-Stairs-Baseline: Unitree's Rough MDP on the shared scene (spec §6.1)."""

import copy

import pytest
import torch
from mjlab.tasks.registry import (
  list_tasks,
  load_env_cfg,
  load_rl_cfg,
  load_runner_cls,
  register_mjlab_task,
)

from mjlab.utils.noise import UniformNoiseCfg as Unoise

import g1_stairs.tasks  # noqa: F401  (registers the tasks)
from g1_stairs.runner import VelocityOnPolicyRunner
from g1_stairs.tasks.baseline.env_cfg import HEIGHT_SCAN_OFFSET

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


def test_height_scan_noise_matches_real_lidar():
  # Per-ray noise plus a per-episode offset, both U(-0.03, 0.03) m (BeamDojo, G1 + MID-360).
  cfg = load_env_cfg(TASK)
  noise = cfg.observations["actor"].terms["height_scan"].noise
  assert (noise.noise_cfg.n_min, noise.noise_cfg.n_max) == (-0.03, 0.03)
  assert (noise.bias_noise_cfg.n_min, noise.bias_noise_cfg.n_max) == (-0.03, 0.03)
  assert cfg.observations["critic"].terms["height_scan"].noise is None


def test_height_scan_lag_is_at_most_one_policy_step():
  cfg = load_env_cfg(TASK)
  actor = cfg.observations["actor"].terms["height_scan"]
  assert (actor.delay_min_lag, actor.delay_max_lag) == (0, 1)
  assert cfg.observations["critic"].terms["height_scan"].delay_max_lag == 0


def test_height_scan_offset_is_shared_and_does_not_drift():
  noise_cfg = copy.deepcopy(load_env_cfg(TASK).observations["actor"].terms["height_scan"].noise)
  noise_cfg.noise_cfg = Unoise(n_min=0.0, n_max=1e-9)  # isolate the offset
  model = noise_cfg.class_type(noise_cfg, num_envs=64, device="cpu")
  for _ in range(200):
    model.reset()
  out = model(torch.zeros(64, 187))
  # One offset per env, shared by every ray, still inside its range after many resets.
  assert torch.allclose(out, out[:, :1].expand_as(out), atol=1e-6)
  assert out.abs().max() <= HEIGHT_SCAN_OFFSET + 1e-6
  assert out.std() > 0.005


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
