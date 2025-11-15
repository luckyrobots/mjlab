from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  unitree_g1_flat_env_cfg_default,
  unitree_g1_flat_env_cfg_play,
  unitree_g1_rough_env_cfg_default,
  unitree_g1_rough_env_cfg_play,
)
from .rl_cfg import UNITREE_G1_PPO_RUNNER_CFG

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-Unitree-G1",
  env_cfg=unitree_g1_rough_env_cfg_default,
  rl_cfg=UNITREE_G1_PPO_RUNNER_CFG,
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-Unitree-G1-Play",
  env_cfg=unitree_g1_rough_env_cfg_play,
  rl_cfg=UNITREE_G1_PPO_RUNNER_CFG,
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Unitree-G1",
  env_cfg=unitree_g1_flat_env_cfg_default,
  rl_cfg=UNITREE_G1_PPO_RUNNER_CFG,
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Unitree-G1-Play",
  env_cfg=unitree_g1_flat_env_cfg_play,
  rl_cfg=UNITREE_G1_PPO_RUNNER_CFG,
  runner_cls=VelocityOnPolicyRunner,
)
