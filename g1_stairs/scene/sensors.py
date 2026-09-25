"""Scene sensors (spec §5.3). Every variant gets all of them; each decides what to use.

The robot's IMU (robot/imu_ang_vel, robot/imu_lin_vel, robot/imu_lin_acc) and
robot/root_angmom come from the MJCF and are not listed here.
"""

from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  GridPatternCfg,
  ObjRef,
  RayCastSensorCfg,
  SensorCfg,
)

from g1_stairs.scene.robot import (
  FOOT_BODY_PATTERN,
  FOOT_SITE_NAMES,
  NON_FOOT_CONTACT_BODIES,
  PELVIS_BODY,
  ROBOT_NAME,
)

TERRAIN_SCAN = "terrain_scan"
FOOT_SCAN_LEFT = "foot_scan_left"
FOOT_SCAN_RIGHT = "foot_scan_right"
FEET_GROUND_CONTACT = "feet_ground_contact"
BODY_TERRAIN_CONTACT = "body_terrain_contact"
SELF_COLLISION = "self_collision"
SENSOR_NAMES: tuple[str, ...] = (
  TERRAIN_SCAN,
  FOOT_SCAN_LEFT,
  FOOT_SCAN_RIGHT,
  FEET_GROUND_CONTACT,
  BODY_TERRAIN_CONTACT,
  SELF_COLLISION,
)

# Terrain geoms are in group 0. The robot's visual meshes are group 2 and its
# collision geoms group 3, so rays restricted to group 0 never see the robot.
RAY_GEOM_GROUPS: tuple[int, ...] = (0,)
TERRAIN_SCAN_MAX_DISTANCE = 5.0


def _terrain_scan() -> RayCastSensorCfg:
  """1.6 x 1.0 m grid at 0.1 m (187 rays) under the pelvis, turning with heading."""
  return RayCastSensorCfg(
    name=TERRAIN_SCAN,
    frame=ObjRef(type="body", name=PELVIS_BODY, entity=ROBOT_NAME),
    ray_alignment="yaw",
    pattern=GridPatternCfg(size=(1.6, 1.0), resolution=0.1),
    max_distance=TERRAIN_SCAN_MAX_DISTANCE,
    exclude_parent_body=True,
    include_geom_groups=RAY_GEOM_GROUPS,
    debug_vis=True,
    viz=RayCastSensorCfg.VizCfg(show_normals=True),
  )


def _foot_scan(name: str, site: str) -> RayCastSensorCfg:
  """3 downward rays along the sole (heel, middle, toe; 0.1 m apart)."""
  return RayCastSensorCfg(
    name=name,
    frame=ObjRef(type="site", name=site, entity=ROBOT_NAME),
    ray_alignment="yaw",
    pattern=GridPatternCfg(size=(0.2, 0.0), resolution=0.1),
    max_distance=1.0,
    exclude_parent_body=True,
    include_geom_groups=RAY_GEOM_GROUPS,
  )


def make_sensors() -> tuple[SensorCfg, ...]:
  feet_ground_contact = ContactSensorCfg(
    name=FEET_GROUND_CONTACT,
    primary=ContactMatch(mode="subtree", pattern=FOOT_BODY_PATTERN, entity=ROBOT_NAME),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  body_terrain_contact = ContactSensorCfg(
    name=BODY_TERRAIN_CONTACT,
    primary=ContactMatch(
      mode="body",
      pattern=tuple(f"^{body}$" for body in NON_FOOT_CONTACT_BODIES),
      entity=ROBOT_NAME,
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
  )
  self_collision = ContactSensorCfg(
    name=SELF_COLLISION,
    primary=ContactMatch(mode="subtree", pattern=PELVIS_BODY, entity=ROBOT_NAME),
    secondary=ContactMatch(mode="subtree", pattern=PELVIS_BODY, entity=ROBOT_NAME),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  return (
    _terrain_scan(),
    _foot_scan(FOOT_SCAN_LEFT, FOOT_SITE_NAMES[0]),
    _foot_scan(FOOT_SCAN_RIGHT, FOOT_SITE_NAMES[1]),
    feet_ground_contact,
    body_terrain_contact,
    self_collision,
  )
