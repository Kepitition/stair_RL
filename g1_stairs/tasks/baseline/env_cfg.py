# Derived from unitree_rl_mjlab src/tasks/velocity/velocity_env_cfg.py and
# src/tasks/velocity/config/g1_23dof/env_cfgs.py (Apache-2.0).
"""G1-Stairs-Baseline: Unitree's Unitree-G1-23Dof-Rough MDP on the shared stairs scene.

Rewards, weights, observations, events, terminations and curricula are Unitree's,
unchanged (spec §6.1), except the height-scan noise (see _height_scan_noise). Only
the scene comes from g1_stairs.scene. This is the template for variants: copy this
folder and change what you need.
"""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.utils.noise import NoiseModelWithAdditiveBiasCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

import g1_stairs.mdp as mdp
from g1_stairs.scene.robot import (
  FOOT_GEOM_NAMES,
  FOOT_SITE_NAMES,
  ROBOT_NAME,
  TORSO_BODY,
  make_actions,
)
from g1_stairs.scene.scene import (
  DECIMATION,
  EPISODE_LENGTH_S,
  make_scene_cfg,
  make_sim_cfg,
)
from g1_stairs.scene.sensors import (
  FEET_GROUND_CONTACT,
  SELF_COLLISION,
  TERRAIN_SCAN,
  TERRAIN_SCAN_MAX_DISTANCE,
)

# Height-scan noise on the actor, in meters (mjlab adds noise before scaling).
# Values from BeamDojo (arXiv:2502.10363, Table IX), deployed on a real G1 with the
# head-mounted Livox MID-360 and an elevation map: U(-0.03, 0.03) m per ray and
# step, plus one U(-0.03, 0.03) m vertical offset per episode shared by all rays.
# The MID-360 datasheet gives a range precision (1 sigma) of 2-3 cm. Unitree's
# 0.1 m per ray is as large as a riser and hides the step edges.
HEIGHT_SCAN_NOISE = 0.03
HEIGHT_SCAN_OFFSET = 0.03
# Lag of the actor's height scan, in policy steps (20 ms each), sampled per env and
# step. Covers the pose-estimation and processing delay of cropping the map around
# the robot. The map itself refreshes at ~10 Hz on hardware, but stairs are static,
# so that slower refresh is not modeled as lag.
HEIGHT_SCAN_MAX_LAG = 1


def _height_scan_noise() -> NoiseModelWithAdditiveBiasCfg:
  return NoiseModelWithAdditiveBiasCfg(
    noise_cfg=Unoise(n_min=-HEIGHT_SCAN_NOISE, n_max=HEIGHT_SCAN_NOISE),
    # "abs" draws a fresh offset on reset; mjlab's default "add" would sum it onto
    # the previous episode's offset, so the offset would drift over training.
    bias_noise_cfg=Unoise(
      n_min=-HEIGHT_SCAN_OFFSET, n_max=HEIGHT_SCAN_OFFSET, operation="abs"
    ),
    sample_bias_per_component=False,
  )

# Unitree's per-joint posture tolerances for the G1 23-DOF.
_POSE_STD_WALKING = {
  r".*hip_pitch.*": 0.5,
  r".*hip_roll.*": 0.15,
  r".*hip_yaw.*": 0.15,
  r".*knee.*": 0.5,
  r".*ankle_pitch.*": 0.15,
  r".*ankle_roll.*": 0.1,
  r".*waist_yaw.*": 0.15,
  r".*shoulder_pitch.*": 0.15,
  r".*shoulder_roll.*": 0.1,
  r".*shoulder_yaw.*": 0.1,
  r".*elbow.*": 0.1,
  r".*wrist.*": 0.1,
}
_POSE_STD_RUNNING = {
  r".*hip_pitch.*": 0.5,
  r".*hip_roll.*": 0.25,
  r".*hip_yaw.*": 0.25,
  r".*knee.*": 0.5,
  r".*ankle_pitch.*": 0.25,
  r".*ankle_roll.*": 0.1,
  r".*waist_yaw.*": 0.25,
  r".*shoulder_pitch.*": 0.25,
  r".*shoulder_roll.*": 0.1,
  r".*shoulder_yaw.*": 0.1,
  r".*elbow.*": 0.1,
  r".*wrist.*": 0.1,
}


def _observations() -> dict[str, ObservationGroupCfg]:
  actor_terms = {
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "command": ObservationTermCfg(
      func=mdp.generated_commands,
      params={"command_name": "twist"},
    ),
    "phase": ObservationTermCfg(
      func=mdp.phase,
      params={"period": 0.6, "command_name": "twist"},
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-1.5, n_max=1.5),
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
    "height_scan": ObservationTermCfg(
      func=envs_mdp.height_scan,
      params={"sensor_name": TERRAIN_SCAN},
      noise=_height_scan_noise(),
      scale=1 / TERRAIN_SCAN_MAX_DISTANCE,
      delay_min_lag=0,
      delay_max_lag=HEIGHT_SCAN_MAX_LAG,
    ),
  }
  critic_terms = {
    **actor_terms,
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
      noise=Unoise(n_min=-0.5, n_max=0.5),
    ),
    "height_scan": ObservationTermCfg(
      func=envs_mdp.height_scan,
      params={"sensor_name": TERRAIN_SCAN},
      scale=1 / TERRAIN_SCAN_MAX_DISTANCE,
    ),
    "foot_height": ObservationTermCfg(
      func=mdp.foot_height,
      params={"asset_cfg": SceneEntityCfg(ROBOT_NAME, site_names=FOOT_SITE_NAMES)},
    ),
    "foot_air_time": ObservationTermCfg(
      func=mdp.foot_air_time,
      params={"sensor_name": FEET_GROUND_CONTACT},
    ),
    "foot_contact": ObservationTermCfg(
      func=mdp.foot_contact,
      params={"sensor_name": FEET_GROUND_CONTACT},
    ),
    "foot_contact_forces": ObservationTermCfg(
      func=mdp.foot_contact_forces,
      params={"sensor_name": FEET_GROUND_CONTACT},
    ),
  }
  return {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
      history_length=1,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
      history_length=1,
    ),
  }


def _commands() -> dict[str, CommandTermCfg]:
  twist = UniformVelocityCommandCfg(
    entity_name=ROBOT_NAME,
    resampling_time_range=(3.0, 8.0),
    rel_standing_envs=0.05,
    heading_command=True,
    heading_control_stiffness=0.5,
    debug_vis=True,
    ranges=UniformVelocityCommandCfg.Ranges(
      lin_vel_x=(-1.0, 2.0),
      lin_vel_y=(-1.0, 1.0),
      ang_vel_z=(-1.0, 1.0),
      heading=(-math.pi, math.pi),
    ),
  )
  twist.viz.z_offset = 1.15
  return {"twist": twist}


def _events() -> dict[str, EventTermCfg]:
  return {
    "reset_base": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "z": (0.0, 0.0),
          "yaw": (-3.14, 3.14),
        },
        "velocity_range": {},
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.0, 0.0),
        "velocity_range": (-0.0, 0.0),
        "asset_cfg": SceneEntityCfg(ROBOT_NAME, joint_names=(".*",)),
      },
    ),
    "push_robot": EventTermCfg(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(5.0, 6.0),
      params={
        "velocity_range": {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "z": (-0.4, 0.4),
          "roll": (-0.52, 0.52),
          "pitch": (-0.52, 0.52),
          "yaw": (-0.78, 0.78),
        },
      },
    ),
    "foot_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg(ROBOT_NAME, geom_names=FOOT_GEOM_NAMES),
        "operation": "abs",
        "ranges": (0.3, 1.6),
        "shared_random": True,
      },
    ),
    "encoder_bias": EventTermCfg(
      mode="startup",
      func=dr.encoder_bias,
      params={
        "asset_cfg": SceneEntityCfg(ROBOT_NAME),
        "bias_range": (-0.015, 0.015),
      },
    ),
    "base_com": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg(ROBOT_NAME, body_names=(TORSO_BODY,)),
        "operation": "add",
        "ranges": {0: (-0.05, 0.05), 1: (-0.05, 0.05), 2: (-0.05, 0.05)},
      },
    ),
  }


def _rewards() -> dict[str, RewardTermCfg]:
  feet = SceneEntityCfg(ROBOT_NAME, site_names=FOOT_SITE_NAMES)
  torso = SceneEntityCfg(ROBOT_NAME, body_names=(TORSO_BODY,))
  return {
    "track_linear_velocity": RewardTermCfg(
      func=mdp.track_linear_velocity,
      weight=1.0,
      params={"command_name": "twist", "std": math.sqrt(0.25)},
    ),
    "track_angular_velocity": RewardTermCfg(
      func=mdp.track_angular_velocity,
      weight=1.0,
      params={"command_name": "twist", "std": math.sqrt(0.5)},
    ),
    "body_orientation_l2": RewardTermCfg(
      func=mdp.body_orientation_l2,
      weight=-1.0,
      params={"asset_cfg": torso},
    ),
    "pose": RewardTermCfg(
      func=mdp.variable_posture,
      weight=1.0,
      params={
        "asset_cfg": SceneEntityCfg(ROBOT_NAME, joint_names=".*"),
        "command_name": "twist",
        "std_standing": {".*": 0.05},
        "std_walking": _POSE_STD_WALKING,
        "std_running": _POSE_STD_RUNNING,
        "walking_threshold": 0.1,
        "running_threshold": 1.5,
      },
    ),
    "body_ang_vel": RewardTermCfg(
      func=mdp.body_angular_velocity_penalty,
      weight=-0.05,
      params={"asset_cfg": SceneEntityCfg(ROBOT_NAME, body_names=(TORSO_BODY,))},
    ),
    "angular_momentum": RewardTermCfg(
      func=mdp.angular_momentum_penalty,
      weight=-0.025,
      params={"sensor_name": "robot/root_angmom"},
    ),
    "is_terminated": RewardTermCfg(func=mdp.is_terminated, weight=-200.0),
    "joint_acc_l2": RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.5e-7),
    "joint_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-10.0),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.05),
    "foot_gait": RewardTermCfg(
      func=mdp.feet_gait,
      weight=0.5,
      params={
        "period": 0.6,
        "offset": [0.0, 0.5],
        "threshold": 0.56,
        "command_threshold": 0.1,
        "command_name": "twist",
        "sensor_name": FEET_GROUND_CONTACT,
      },
    ),
    # Known issue (spec §9): target is absolute world height, wrong on stairs.
    "foot_clearance": RewardTermCfg(
      func=mdp.feet_clearance,
      weight=-1.0,
      params={
        "target_height": 0.10,
        "command_name": "twist",
        "command_threshold": 0.1,
        "asset_cfg": feet,
      },
    ),
    "foot_slip": RewardTermCfg(
      func=mdp.feet_slip,
      weight=-0.25,
      params={
        "sensor_name": FEET_GROUND_CONTACT,
        "command_name": "twist",
        "command_threshold": 0.1,
        "asset_cfg": SceneEntityCfg(ROBOT_NAME, site_names=FOOT_SITE_NAMES),
      },
    ),
    "soft_landing": RewardTermCfg(
      func=mdp.soft_landing,
      weight=-1e-3,
      params={
        "sensor_name": FEET_GROUND_CONTACT,
        "command_name": "twist",
        "command_threshold": 0.1,
      },
    ),
    "stand_still": RewardTermCfg(
      func=mdp.stand_still,
      weight=-1.0,
      params={
        "command_name": "twist",
        "command_threshold": 0.1,
        "asset_cfg": SceneEntityCfg(ROBOT_NAME, joint_names=".*"),
      },
    ),
    "self_collisions": RewardTermCfg(
      func=mdp.self_collision_cost,
      weight=-1.0,
      params={"sensor_name": SELF_COLLISION, "force_threshold": 10.0},
    ),
  }


def _terminations() -> dict[str, TerminationTermCfg]:
  return {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(
      func=mdp.bad_orientation,
      params={"limit_angle": math.radians(70.0)},
    ),
  }


def _curriculum() -> dict[str, CurriculumTermCfg]:
  return {
    "terrain_levels": CurriculumTermCfg(
      func=mdp.terrain_levels_vel,
      params={"command_name": "twist"},
    ),
    "command_vel": CurriculumTermCfg(
      func=mdp.commands_vel,
      params={
        "command_name": "twist",
        "velocity_stages": [
          {"step": 0, "lin_vel_x": (-0.5, 1.0), "lin_vel_y": (-0.5, 0.5), "ang_vel_z": (-1.0, 1.0)},
          {"step": 5000 * 24, "lin_vel_x": (-1.0, 2.0), "lin_vel_y": (-1.0, 1.0)},
        ],
      },
    ),
  }


def baseline_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = ManagerBasedRlEnvCfg(
    scene=make_scene_cfg(play=play),
    observations=_observations(),
    actions=make_actions(),
    commands=_commands(),
    events=_events(),
    rewards=_rewards(),
    terminations=_terminations(),
    curriculum=_curriculum(),
    metrics={"mean_action_acc": MetricsTermCfg(func=mdp.mean_action_acc)},
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name=ROBOT_NAME,
      body_name=TORSO_BODY,
      distance=3.0,
      elevation=-5.0,
      azimuth=90.0,
    ),
    sim=make_sim_cfg(),
    decimation=DECIMATION,
    episode_length_s=EPISODE_LENGTH_S,
  )
  if play:
    cfg.episode_length_s = int(1e9)  # effectively endless
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )
  return cfg
