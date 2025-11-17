"""Unitree G1 flat terrain locomanipulation configuration.

This module provides factory functions that create complete ManagerBasedRlEnvCfg
instances for the G1 robot locomanipulation task on flat terrain.
"""

from pathlib import Path
import mujoco
from copy import deepcopy

from mjlab.asset_zoo.robots import (
  G1_ACTION_SCALE,
  get_g1_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.locomanipulation.locomanipulation_env_cfg import create_locomanipulation_env_cfg
from mjlab.utils.retval import retval
from mjlab.entity import EntityCfg


def infer_object_cfg_from_motion_file(motion_file: str) -> dict[str, EntityCfg] | None:
  """Infer interactive asset (largebox / chair / climb terrain) from the motion name.

  This mirrors the naming conventions used in `omniretarget/visualize.py`:
  - robot-object:      `sub*_largebox_...`                 → `models/largebox/largebox.xml`
  - robot-object-terr: `scene_*_chair_scaled_*.npz`        → `models/chair/chair_scaled_*.xml`
  - robot-object-terr: `scene_*_original.npz`              → `models/chair/chair.xml`
  - robot-terrain:     `climb_XX_z_scale_Y.npz`            → `models/terrain/climb_XX/multi_boxes_z_scale_Y.xml`
  """
  path = Path(motion_file)
  candidate = path.parent.name or path.stem
  # Strip optional WandB alias suffix, e.g. "sub3_largebox_000_original:v0"
  candidate = candidate.split(":", 1)[0]

  base_models_dir = Path(__file__).parents[6] / "omniretarget" / "models"

  # Large box motions (robot-object scenes).
  if "largebox" in candidate:
    xml_path = base_models_dir / "largebox" / "largebox.xml"

    def spec_fn(xml_path=xml_path) -> mujoco.MjSpec:
      return mujoco.MjSpec.from_file(str(xml_path))

    return {"object": EntityCfg(spec_fn=spec_fn)}

  # Chair motions (robot-object-terrain scenes).
  if "chair" in candidate or ("scene_" in candidate and "original" in candidate):
    chair_dir = base_models_dir / "chair"

    # Scaled chair: extract "chair_scaled_X.Y" from the filename.
    if "chair_scaled_" in candidate:
      start = candidate.index("chair_scaled_")
      # Everything after "chair_scaled_...", up to "_z_scale" if present.
      rest = candidate[start:]
      end = rest.find("_z_scale")
      if end != -1:
        scale_tag = rest[:end]
      else:
        scale_tag = rest
      xml_name = f"{scale_tag}.xml"
    else:
      # "scene_XX_original" → default chair.xml
      xml_name = "chair.xml"

    xml_path = chair_dir / xml_name

    def spec_fn(xml_path=xml_path) -> mujoco.MjSpec:
      return mujoco.MjSpec.from_file(str(xml_path))

    return {"object": EntityCfg(spec_fn=spec_fn)}

  # Climb motions (robot-terrain scenes).
  if candidate.startswith("climb_"):
    # Example: "climb_08_z_scale_1.1"
    # - terrain folder is "models/terrain/climb_08"
    # - xml is       "multi_boxes_z_scale_1.1.xml"
    climb_id = candidate[:8]  # "climb_08"
    z_tag: str
    if "_z_scale" in candidate:
      idx = candidate.index("_z_scale")
      z_tag = candidate[idx:]  # "_z_scale_1.1"
    else:
      z_tag = "_z_scale_1.0"

    xml_path = (
      base_models_dir
      / "terrain"
      / climb_id
      / f"multi_boxes{z_tag}.xml"
    )

    def spec_fn(xml_path=xml_path) -> mujoco.MjSpec:
      return mujoco.MjSpec.from_file(str(xml_path))

    return {"object": EntityCfg(spec_fn=spec_fn)}

  # No recognized object for this motion.
  return None


@retval
def UNITREE_G1_FLAT_LOCOMANIPULATION_ENV_CFG() -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain locomanipulation configuration."""
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )

  return create_locomanipulation_env_cfg(
    robot_cfg=get_g1_robot_cfg(),
    action_scale=G1_ACTION_SCALE,
    viewer_body_name="torso_link",
    motion_file="",
    anchor_body_name="torso_link",
    body_names=(
      "pelvis",
      "left_hip_roll_link",
      "left_knee_link",
      "left_ankle_roll_link",
      "right_hip_roll_link",
      "right_knee_link",
      "right_ankle_roll_link",
      "torso_link",
      "left_shoulder_roll_link",
      "left_elbow_link",
      "left_wrist_yaw_link",
      "right_shoulder_roll_link",
      "right_elbow_link",
      "right_wrist_yaw_link",
    ),
    ee_body_names=(
      "left_ankle_roll_link",
      "right_ankle_roll_link",
      "left_wrist_yaw_link",
      "right_wrist_yaw_link",
    ),
    base_com_body_name="torso_link",
    sensors=(self_collision_cfg,),
    pose_range={
      "x": (-0.05, 0.05),
      "y": (-0.05, 0.05),
      "z": (-0.01, 0.01),
      "roll": (-0.1, 0.1),
      "pitch": (-0.1, 0.1),
      "yaw": (-0.2, 0.2),
    },
    velocity_range={
      "x": (-0.5, 0.5),
      "y": (-0.5, 0.5),
      "z": (-0.2, 0.2),
      "roll": (-0.52, 0.52),
      "pitch": (-0.52, 0.52),
      "yaw": (-0.78, 0.78),
    },
    joint_position_range=(-0.1, 0.1),
    objects={},
  )


@retval
def UNITREE_G1_FLAT_LOCOMANIPULATION_NO_STATE_ESTIMATION_ENV_CFG() -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain locomanipulation config without state estimation.

  This variant disables motion_anchor_pos_b and base_lin_vel observations,
  simulating the lack of state estimation.
  """
  cfg = deepcopy(UNITREE_G1_FLAT_LOCOMANIPULATION_ENV_CFG)
  assert "policy" in cfg.observations
  cfg.observations["policy"].terms.pop("motion_anchor_pos_b")
  cfg.observations["policy"].terms.pop("base_lin_vel")
  return cfg
