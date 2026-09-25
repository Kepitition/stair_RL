"""The robot is Unitree's g1_23dof_rev_1_0, unchanged (spec §5.1)."""

import re

import mujoco
import pytest
from mjlab.entity import Entity

from g1_stairs.scene import robot

# MJCF joint order (docs/research/findings.md §2).
EXPECTED_JOINTS = [
  "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
  "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
  "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
  "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
  "waist_yaw_joint",
  "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
  "left_elbow_joint", "left_wrist_roll_joint",
  "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
  "right_elbow_joint", "right_wrist_roll_joint",
]


@pytest.fixture(scope="module")
def model() -> mujoco.MjModel:
  return Entity(robot.get_robot_cfg()).spec.compile()


def test_model_is_g1_23dof_rev_1_0():
  from g1_stairs.assets.g1_23dof.constants import G1_23DOF_XML

  assert mujoco.MjSpec.from_file(str(G1_23DOF_XML)).modelname == "g1_23dof_rev_1_0"


def test_joint_order_and_actuators(model):
  hinges = [
    model.joint(j).name
    for j in range(model.njnt)
    if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
  ]
  assert hinges == EXPECTED_JOINTS
  assert model.nu == 23


def test_action_scale_covers_every_joint():
  for joint in EXPECTED_JOINTS:
    assert any(re.fullmatch(p, joint) for p in robot.ACTION_SCALE), joint


def test_named_parts_exist(model):
  for site in robot.FOOT_SITE_NAMES:
    model.site(site)  # raises KeyError if missing
  for geom in robot.FOOT_GEOM_NAMES:
    model.geom(geom)
  for body in robot.NON_FOOT_CONTACT_BODIES:
    bid = model.body(body).id
    owned = [g for g in range(model.ngeom) if model.geom_bodyid[g] == bid and model.geom_group[g] == 3]
    assert owned, f"{body} has no collision geom"


def test_actions_are_joint_position_targets():
  term = robot.make_actions()["joint_pos"]
  assert term.entity_name == robot.ROBOT_NAME
  assert term.actuator_names == (".*",)
  assert term.use_default_offset is True
  assert term.scale == robot.ACTION_SCALE


def test_robot_compiles_from_any_working_directory(tmp_path, monkeypatch):
  monkeypatch.chdir(tmp_path)
  assert Entity(robot.get_robot_cfg()).spec.compile().nu == 23
