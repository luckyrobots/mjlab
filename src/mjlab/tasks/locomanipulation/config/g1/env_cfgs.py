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
  """Infer object configuration from motion file or directory.

  Follows OmniRetarget artifact naming conventions:
    - robot-object:      sub*_largebox_*        → largebox/largebox.xml
    - robot-object-terr: scene_*_chair_scaled_* → chair/chair_scaled_*.xml
    - robot-object-terr: scene_*_original       → chair/chair.xml
    - robot-terrain:     climb_XX_z_scale_Y     → terrain/climb_XX/multi_boxes_z_scale_Y.xml

  Args:
    motion_file: Path to .npz file or directory containing .npz files
  """
  path = Path(motion_file)
  base_models_dir = Path(__file__).parents[6] / "omniretarget" / "models"

  # If directory provided, use first .npz file for inference
  if path.is_dir():
    npz_files = sorted(path.glob("*.npz"))
    if not npz_files:
      return None
    path = npz_files[0]

  # Extract filename stem, removing extension and WandB suffix (e.g., ":v0")
  stem = path.stem.split(":", 1)[0]

  def create_entity(xml_path: Path) -> dict[str, EntityCfg]:
    """Helper to create EntityCfg from XML path."""
    return {"object": EntityCfg(spec_fn=lambda p=xml_path: mujoco.MjSpec.from_file(str(p)))}

  # Pattern matching: largebox
  if "largebox" in stem:
    return create_entity(base_models_dir / "largebox" / "largebox.xml")

  # Pattern matching: chair (scaled or default)
  if "chair" in stem or ("scene_" in stem and "original" in stem):
    xml_name = "chair.xml"  # default
    
    if "chair_scaled_" in stem:
      start = stem.index("chair_scaled_")
      rest = stem[start:]
      end_idx = rest.find("_z_scale")
      scale_tag = rest[:end_idx] if end_idx != -1 else rest
      xml_name = f"{scale_tag}.xml"
    
    return create_entity(base_models_dir / "chair" / xml_name)

  # Pattern matching: climb terrain
  if stem.startswith("climb_"):
    climb_id = stem[:8]  # e.g., "climb_08"
    z_tag = stem[stem.index("_z_scale"):] if "_z_scale" in stem else "_z_scale_1.0"
    xml_path = base_models_dir / "terrain" / climb_id / f"multi_boxes{z_tag}.xml"
    return create_entity(xml_path)

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
