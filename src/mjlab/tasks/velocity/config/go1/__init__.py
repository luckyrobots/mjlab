from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  unitree_go1_flat_env_cfg_default,
  unitree_go1_flat_env_cfg_learned_default,
  unitree_go1_flat_env_cfg_learned_play,
  unitree_go1_flat_env_cfg_play,
  unitree_go1_rough_env_cfg_default,
  unitree_go1_rough_env_cfg_play,
)
from .rl_cfg import UNITREE_GO1_PPO_RUNNER_CFG

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-Unitree-Go1",
  env_cfg=unitree_go1_rough_env_cfg_default,
  rl_cfg=UNITREE_GO1_PPO_RUNNER_CFG,
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-Unitree-Go1-Play",
  env_cfg=unitree_go1_rough_env_cfg_play,
  rl_cfg=UNITREE_GO1_PPO_RUNNER_CFG,
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Unitree-Go1",
  env_cfg=unitree_go1_flat_env_cfg_default,
  rl_cfg=UNITREE_GO1_PPO_RUNNER_CFG,
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Unitree-Go1-Play",
  env_cfg=unitree_go1_flat_env_cfg_play,
  rl_cfg=UNITREE_GO1_PPO_RUNNER_CFG,
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Unitree-Go1-ActuatorNet",
  env_cfg=unitree_go1_flat_env_cfg_learned_default,
  rl_cfg=UNITREE_GO1_PPO_RUNNER_CFG,
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Unitree-Go1-ActuatorNet-Play",
  env_cfg=unitree_go1_flat_env_cfg_learned_play,
  rl_cfg=UNITREE_GO1_PPO_RUNNER_CFG,
  runner_cls=VelocityOnPolicyRunner,
)
