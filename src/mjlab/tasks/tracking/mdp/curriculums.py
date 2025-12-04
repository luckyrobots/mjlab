"""Curriculum functions for the tracking task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def motion_weights_curriculum(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  weights_schedule: dict[int, list[float]],
  command_name: str = "motion",
) -> None:
  """Update motion weights based on the current step count.

  Args:
    env: The environment.
    env_ids: The environment ids (unused, as weights are global).
    weights_schedule: A dictionary mapping step counts to weight lists.
      The keys must be integers (step count) and values must be lists of floats
      summing to 1.0.
    command_name: The name of the command term to update.
  """
  current_step = env.common_step_counter

  # Find the active weights based on the schedule
  sorted_steps = sorted(weights_schedule.keys())
  
  # Default to the first schedule if we haven't reached any threshold yet
  # (though usually 0 is included)
  if not sorted_steps:
    return

  active_weights = weights_schedule[sorted_steps[0]]
  
  for step in sorted_steps:
    if current_step >= step:
      active_weights = weights_schedule[step]
    else:
      break

  try:
    command = env.command_manager.get_term(command_name)
    if hasattr(command, "set_motion_weights"):
        command.set_motion_weights(active_weights)
  except (KeyError, AttributeError):
    raise ValueError(f"Command '{command_name}' does not support motion weights curriculum.")