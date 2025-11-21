"""Locomanipulation task configuration.

This module defines the base configuration for locomanipulation tasks.
Robot-specific configurations are located in the config/ directory.

This is a re-implementation of OmniRetarget (https://omniretarget.github.io/).
"""

from copy import deepcopy

from mjlab.entity.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.manager_term_config import (
  ActionTermCfg,
  CommandTermCfg,
  EventTermCfg,
  ObservationGroupCfg,
  ObservationTermCfg,
  RewardTermCfg,
  TerminationTermCfg,
  CurriculumTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.locomanipulation import mdp
from mjlab.tasks.locomanipulation.mdp import MotionCommandCfg
from mjlab.tasks.locomanipulation.config.g1 import rl_cfg as g1_rl_cfg
from mjlab.terrains import TerrainImporterCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

VELOCITY_RANGE = {
  "x": (-0.3, 0.3),
  "y": (-0.3, 0.3),
  "z": (-0.0, 0.0),
  "roll": (-0.0, 0.0),
  "pitch": (-0.0, 0.0),
  "yaw": (-0.78, 0.78),
}

SCENE_CFG = SceneCfg(terrain=TerrainImporterCfg(terrain_type="plane"), num_envs=1)

VIEWER_CONFIG = ViewerConfig(
  origin_type=ViewerConfig.OriginType.ASSET_BODY,
  asset_name="robot",
  body_name="",  # Override in robot cfg.
  distance=3.0,
  elevation=-5.0,
  azimuth=90.0,
)

SIM_CFG = SimulationCfg(
  nconmax=35,
  njmax=250,
  mujoco=MujocoCfg(
    timestep=0.005,
    iterations=10,
    ls_iterations=20,
  ),
)

EPISODE_LENGTH_S = 10.0
DECIMATION = 4
EPISODE_STEPS = int(
  EPISODE_LENGTH_S / (SIM_CFG.mujoco.timestep * DECIMATION)
)


def iters_to_steps(iters: int) -> int:
    """Convert PPO iterations to env.common_step_counter units."""
    return iters * g1_rl_cfg.UNITREE_G1_LOCOMANIPULATION_PPO_RUNNER_CFG.num_steps_per_env


def create_locomanipulation_env_cfg(
  robot_cfg: EntityCfg,
  action_scale: float | dict[str, float],
  viewer_body_name: str,
  motion_file: str,
  anchor_body_name: str,
  body_names: tuple[str, ...],
  ee_body_names: tuple[str, ...],
  base_com_body_name: str,
  sensors: tuple[ContactSensorCfg, ...],
  pose_range: dict[str, tuple[float, float]],
  velocity_range: dict[str, tuple[float, float]],
  joint_position_range: tuple[float, float],
  objects: dict[str, EntityCfg] | None = None,
) -> ManagerBasedRlEnvCfg:
  """Create a locomanipulation task configuration for locomanipulation.

  Args:
    robot_cfg: Robot configuration.
    action_scale: Action scaling factor(s).
    viewer_body_name: Body for camera tracking.
    motion_file: Path to motion capture data file.
    anchor_body_name: Root body for motion tracking.
    body_names: List of body names to track.
    ee_body_names: End-effector body names for termination.
    base_com_body_name: Body for COM randomization.
    sensors: Sensor configurations to add to the scene.
    pose_range: Position/orientation randomization ranges.
    velocity_range: Velocity randomization ranges.
    joint_position_range: Joint position randomization range.
    objects: Dictionary of object entities to add to the scene.

  Returns:
    Complete ManagerBasedRlEnvCfg for locomanipulation task.
  """

  scene = deepcopy(SCENE_CFG)
  scene.entities = {"robot": robot_cfg, **(objects or {})}
  scene.sensors = sensors

  viewer = deepcopy(VIEWER_CONFIG)
  viewer.body_name = viewer_body_name

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      asset_name="robot",
      actuator_names=(".*",),
      scale=action_scale,
      use_default_offset=True,
    )
  }

  commands: dict[str, CommandTermCfg] = {
    "motion": MotionCommandCfg(
      asset_name="robot",
      resampling_time_range=(1.0e9, 1.0e9),
      debug_vis=True,
      pose_range=pose_range,
      velocity_range=velocity_range,
      joint_position_range=joint_position_range,
      motion_file=motion_file,
      anchor_body_name=anchor_body_name,
      body_names=body_names,
    )
  }

  policy_terms = {
    "command": ObservationTermCfg(
      func=mdp.generated_commands, params={"command_name": "motion"}
    ),
    "motion_anchor_pos_b": ObservationTermCfg(
      func=mdp.motion_anchor_pos_b,
      params={"command_name": "motion"},
      noise=Unoise(n_min=-0.25, n_max=0.25),
    ),
    "motion_anchor_ori_b": ObservationTermCfg(
      func=mdp.motion_anchor_ori_b,
      params={"command_name": "motion"},
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
      noise=Unoise(n_min=-0.5, n_max=0.5),
    ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01)
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5)
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }

  critic_terms = {
    "command": ObservationTermCfg(
      func=mdp.generated_commands, params={"command_name": "motion"}
    ),
    "motion_anchor_pos_b": ObservationTermCfg(
      func=mdp.motion_anchor_pos_b, params={"command_name": "motion"}
    ),
    "motion_anchor_ori_b": ObservationTermCfg(
      func=mdp.motion_anchor_ori_b, params={"command_name": "motion"}
    ),
    "body_pos": ObservationTermCfg(
      func=mdp.robot_body_pos_b, params={"command_name": "motion"}
    ),
    "body_ori": ObservationTermCfg(
      func=mdp.robot_body_ori_b, params={"command_name": "motion"}
    ),
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_lin_vel"}
    ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_ang_vel"}
    ),
    "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
    "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }

  observations = {
    "policy": ObservationGroupCfg(
      terms=policy_terms,
      concatenate_terms=True,
      enable_corruption=True,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  events: dict[str, EventTermCfg] = {
    "push_robot": EventTermCfg(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(1.0, 3.0),
      params={"velocity_range": velocity_range},
    ),
    "base_com": EventTermCfg(
      mode="startup",
      func=mdp.randomize_field,
      domain_randomization=True,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=(base_com_body_name,)),
        "operation": "add",
        "field": "body_ipos",
        "ranges": {
          0: (-0.025, 0.025),
          1: (-0.05, 0.05),
          2: (-0.075, 0.075),
        },
      },
    ),
    "add_joint_default_pos": EventTermCfg(
      mode="startup",
      func=mdp.randomize_field,
      domain_randomization=True,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "operation": "add",
        "field": "qpos0",
        "ranges": (-0.01, 0.01),
      },
    ),
    "object_mass": EventTermCfg(
      mode="startup",
      func=mdp.randomize_field,
      domain_randomization=True,
      params={
        "asset_cfg": SceneEntityCfg("object"),
        "field": "body_mass",
        "ranges": (0.1, 2.0),
        "operation": "abs",
      },
    ),
    "object_com": EventTermCfg(
      mode="startup",
      func=mdp.randomize_field,
      domain_randomization=True,
      params={
        "asset_cfg": SceneEntityCfg("object"),
        "field": "body_ipos",
        "operation": "add",
        "ranges": {
          0: (-0.08, 0.08),
          1: (-0.08, 0.08),
          2: (-0.08, 0.08),
        },
      },
    ),
    "object_inertia": EventTermCfg(
      mode="startup",
      func=mdp.randomize_field,
      domain_randomization=True,
      params={
        "asset_cfg": SceneEntityCfg("object"),
        "field": "body_inertia",
        "ranges": (0.5, 1.5),
        "operation": "scale",
      },
    ),
    "object_shape": EventTermCfg(
      mode="startup",
      func=mdp.randomize_field,
      domain_randomization=True,
      params={
        "asset_cfg": SceneEntityCfg("object"),
        "field": "geom_size",
        "ranges": (0.9, 1.1),
        "operation": "scale",
      },
    ),
  }

  rewards: dict[str, RewardTermCfg] = {
    "motion_global_root_pos": RewardTermCfg(
      func=mdp.motion_global_anchor_position_error_exp,
      weight=0.5,
      params={"command_name": "motion", "std": 0.3},
    ),
    "motion_global_root_ori": RewardTermCfg(
      func=mdp.motion_global_anchor_orientation_error_exp,
      weight=0.5,
      params={"command_name": "motion", "std": 0.4},
    ),
    "motion_body_pos": RewardTermCfg(
      func=mdp.motion_relative_body_position_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 0.3},
    ),
    "motion_body_ori": RewardTermCfg(
      func=mdp.motion_relative_body_orientation_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 0.4},
    ),
    "motion_body_lin_vel": RewardTermCfg(
      func=mdp.motion_global_body_linear_velocity_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 1.0},
    ),
    "motion_body_ang_vel": RewardTermCfg(
      func=mdp.motion_global_body_angular_velocity_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 3.14},
    ),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-1e-1),
    "joint_limit": RewardTermCfg(
      func=mdp.joint_pos_limits,
      weight=-10.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "self_collisions": RewardTermCfg(
      func=mdp.self_collision_cost,
      weight=-10.0,
      params={"sensor_name": "self_collision"},
    ),
    "object_pos": RewardTermCfg(
      func=mdp.object_global_position_error_exp,
      weight=0.5,
      params={"command_name": "motion", "std": 0.3},
    ),
    "object_ori": RewardTermCfg(
      func=mdp.object_global_orientation_error_exp,
      weight=0.5,
      params={"command_name": "motion", "std": 0.4},
    ),
  }

  terminations: dict[str, TerminationTermCfg] = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "anchor_pos": TerminationTermCfg(
      func=mdp.bad_anchor_pos_z_only,
      params={"command_name": "motion", "threshold": 0.25},
    ),
    "anchor_ori": TerminationTermCfg(
      func=mdp.bad_anchor_ori,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "command_name": "motion",
        "threshold": 0.8,
      },
    ),
    "ee_body_pos": TerminationTermCfg(
      func=mdp.bad_motion_body_pos_z_only,
      params={
        "command_name": "motion",
        "threshold": 0.25,
        "body_names": ee_body_names,
      },
    ),
    "object_pos": TerminationTermCfg(
      func=mdp.bad_object_pos,
      params={"command_name": "motion", "threshold": 1.0},
    ),
    "object_ori": TerminationTermCfg(
      func=mdp.bad_object_ori,
      params={"command_name": "motion", "threshold": 0.8},
    ),
  }

  curriculum = {
      "object_terminations": CurriculumTermCfg(
          func=mdp.object_termination_curriculum,
          params={
              "stages": [
                  # Stage 0: Extended Warm-up (0 - 15,000 Iterations)
                  # Goal: Let all body tracking and velocity errors fully converge.
                  {"step": 0, "pos_threshold": 1e6, "ori_threshold": 1e6},

                  # Stage 1: The Binary Switch (Strict Enforcement)
                  # Apply the final paper-defined constraint after maximum body stability.
                  {"step": iters_to_steps(15_000), "pos_threshold": 1.0, "ori_threshold": 0.78}, 
              ]
          }
      )
  }

  return ManagerBasedRlEnvCfg(
    scene=scene,
    observations=observations,
    actions=actions,
    commands=commands,
    rewards=rewards,
    terminations=terminations,
    curriculum=curriculum,
    events=events,
    sim=SIM_CFG,
    viewer=viewer,
    decimation=DECIMATION,
    episode_length_s=EPISODE_LENGTH_S,
  )