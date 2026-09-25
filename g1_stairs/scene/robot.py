"""The robot and its control interface (spec §5.1). Fixed for every variant."""

from mjlab.entity import EntityCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg

from g1_stairs.assets.g1_23dof.constants import (
  G1_23DOF_ACTION_SCALE,
  get_g1_23dof_robot_cfg,
)

ROBOT_NAME = "robot"
PELVIS_BODY = "pelvis"
TORSO_BODY = "torso_link"

FOOT_SITE_NAMES: tuple[str, str] = ("left_foot", "right_foot")
FOOT_BODY_PATTERN = r"^(left_ankle_roll_link|right_ankle_roll_link)$"
FOOT_GEOM_NAMES: tuple[str, ...] = tuple(
  f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
)

# Every body that owns a collision geom, except the two feet (ankle_roll links).
NON_FOOT_CONTACT_BODIES: tuple[str, ...] = (
  "pelvis",
  "torso_link",
  "left_hip_roll_link",
  "right_hip_roll_link",
  "left_hip_yaw_link",
  "right_hip_yaw_link",
  "left_knee_link",
  "right_knee_link",
  "left_shoulder_yaw_link",
  "right_shoulder_yaw_link",
  "left_elbow_link",
  "right_elbow_link",
  "left_wrist_roll_rubber_hand",
  "right_wrist_roll_rubber_hand",
)

# Per-joint action scale: 0.25 * torque limit / stiffness (Unitree).
ACTION_SCALE: dict[str, float] = dict(G1_23DOF_ACTION_SCALE)


def get_robot_cfg() -> EntityCfg:
  """A fresh copy of Unitree's G1 23-DOF entity config."""
  return get_g1_23dof_robot_cfg()


def make_actions() -> dict[str, JointPositionActionCfg]:
  """Position targets for all 23 joints around the default pose."""
  return {
    "joint_pos": JointPositionActionCfg(
      entity_name=ROBOT_NAME,
      actuator_names=(".*",),
      scale=dict(ACTION_SCALE),
      use_default_offset=True,
    )
  }
