"""Task registry system for managing environment registration and creation."""

from collections.abc import Callable

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg

# Private module-level registry: task_id -> (env_cfg, rl_cfg, runner_cls)
EnvRlCfgPair = tuple[
  ManagerBasedRlEnvCfg | Callable[[], ManagerBasedRlEnvCfg],
  RslRlOnPolicyRunnerCfg,
  type | None,
]
_REGISTRY: dict[str, EnvRlCfgPair] = {}


def register_mjlab_task(
  task_id: str,
  env_cfg: ManagerBasedRlEnvCfg | Callable[[], ManagerBasedRlEnvCfg],
  rl_cfg: RslRlOnPolicyRunnerCfg,
  runner_cls: type | None = None,
) -> None:
  """Register an environment task.

  Args:
    task_id: Unique task identifier (e.g., "Mjlab-Velocity-Rough-Unitree-Go1").
    env_cfg: Environment configuration (instance or callable that returns one).
    rl_cfg: RL runner configuration.
    runner_cls: Optional custom runner class. If None, uses OnPolicyRunner.
  """
  if task_id in _REGISTRY:
    raise ValueError(f"Task '{task_id}' is already registered")
  _REGISTRY[task_id] = (env_cfg, rl_cfg, runner_cls)


def list_tasks() -> list[str]:
  """List all registered task IDs."""
  return sorted(_REGISTRY.keys())


def load_env_cfg(task_name: str) -> ManagerBasedRlEnvCfg:
  """Load environment configuration for a task."""
  env_cfg, _, _ = _REGISTRY[task_name]

  if callable(env_cfg):
    return env_cfg()
  return env_cfg


def load_rl_cfg(task_name: str) -> RslRlOnPolicyRunnerCfg:
  """Load RL configuration for a task."""
  _, rl_cfg, _ = _REGISTRY[task_name]
  return rl_cfg


def load_runner_cls(task_name: str) -> type | None:
  """Load the runner class for a task.

  If None, the default OnPolicyRunner will be used.
  """
  _, _, runner_cls = _REGISTRY[task_name]
  return runner_cls
