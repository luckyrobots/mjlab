from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .env_cfgs import (
  unitree_g1_flat_tracking_env_cfg_default,
  unitree_g1_flat_tracking_env_cfg_demo,
  unitree_g1_flat_tracking_env_cfg_play,
  unitree_g1_flat_tracking_no_state_estimation_env_cfg,
  unitree_g1_flat_tracking_no_state_estimation_env_cfg_play,
)
from .rl_cfg import UNITREE_G1_TRACKING_PPO_RUNNER_CFG

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Unitree-G1",
  env_cfg=unitree_g1_flat_tracking_env_cfg_default,
  rl_cfg=UNITREE_G1_TRACKING_PPO_RUNNER_CFG,
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Unitree-G1-Play",
  env_cfg=unitree_g1_flat_tracking_env_cfg_play,
  rl_cfg=UNITREE_G1_TRACKING_PPO_RUNNER_CFG,
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Unitree-G1-Demo",
  env_cfg=unitree_g1_flat_tracking_env_cfg_demo,
  rl_cfg=UNITREE_G1_TRACKING_PPO_RUNNER_CFG,
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation",
  env_cfg=unitree_g1_flat_tracking_no_state_estimation_env_cfg,
  rl_cfg=UNITREE_G1_TRACKING_PPO_RUNNER_CFG,
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation-Play",
  env_cfg=unitree_g1_flat_tracking_no_state_estimation_env_cfg_play,
  rl_cfg=UNITREE_G1_TRACKING_PPO_RUNNER_CFG,
  runner_cls=MotionTrackingOnPolicyRunner,
)
