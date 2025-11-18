"""Script to train RL agent with RSL-RL."""

import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg as TrackingMotionCommandCfg
from mjlab.tasks.locomanipulation.mdp import MotionCommandCfg as LocomotionMotionCommandCfg
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner
from mjlab.tasks.locomanipulation.rl import LocomanipulationOnPolicyRunner
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
from mjlab.utils.os import dump_yaml, get_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder


@dataclass(frozen=True)
class TrainConfig:
  env: Any
  agent: RslRlOnPolicyRunnerCfg
  registry_name: str | None = None
  motion_file: str | None = None
  device: str = "cuda:0"
  video: bool = False
  video_length: int = 200
  video_interval: int = 2000
  enable_nan_guard: bool = False


def run_train(cfg: TrainConfig) -> None:
  configure_torch_backends()

  registry_name: str | None = None

  # Check if this env uses a motion command, and whether it is a tracking or
  # locomanipulation-style motion command (mirrors `play.py`).
  has_motion_command = (
    cfg.env.commands is not None and "motion" in cfg.env.commands
  )
  is_tracking_task = (
    has_motion_command
    and isinstance(cfg.env.commands["motion"], TrackingMotionCommandCfg)
  )
  is_locomanipulation_task = (
    has_motion_command
    and isinstance(cfg.env.commands["motion"], LocomotionMotionCommandCfg)
  )

  if is_tracking_task:
    if not cfg.registry_name:
      raise ValueError("Must provide --registry-name for tracking tasks.")

    # Check if the registry name includes alias, if not, append ":latest".
    registry_name = cast(str, cfg.registry_name)
    if ":" not in registry_name:
      registry_name = registry_name + ":latest"
    import wandb

    api = wandb.Api()
    artifact = api.artifact(registry_name)

    assert cfg.env.commands is not None
    motion_cmd = cfg.env.commands["motion"]
    assert isinstance(motion_cmd, TrackingMotionCommandCfg)
    motion_cmd.motion_file = str(Path(artifact.download()) / "motion.npz")

  # For non-tracking tasks that still use a "motion" command term (e.g.,
  # locomanipulation), allow overriding the motion source via --motion-file.
  # This can be either a single .npz file or a directory of .npz files; the
  # MotionCommand implementation for that task is responsible for handling
  # directories as multi-motion datasets.
  if (not is_tracking_task) and cfg.motion_file is not None:
    if not has_motion_command:
      raise ValueError(
        "`--motion-file` was provided but the selected task has no 'motion' command."
      )
    assert cfg.env.commands is not None
    motion_cmd = cfg.env.commands["motion"]
    motion_cmd.motion_file = cfg.motion_file

    # For locomanipulation-style motion-command tasks, infer the interactive
    # object/terrain asset from the motion file name and add it to the scene
    # *before* constructing the environment, so object-based rewards and
    # terminations are active.
    if is_locomanipulation_task:
      try:
        from mjlab.tasks.locomanipulation.config.g1.env_cfgs import (
          infer_object_cfg_from_motion_file,
        )
      except Exception:
        infer_object_cfg_from_motion_file = None

      if infer_object_cfg_from_motion_file is not None:
        obj_cfgs = infer_object_cfg_from_motion_file(motion_cmd.motion_file)
        if obj_cfgs is not None:
          cfg.env.scene.entities.update(obj_cfgs)

  # Enable NaN guard if requested.
  if cfg.enable_nan_guard:
    cfg.env.sim.nan_guard.enabled = True
    print(f"[INFO] NaN guard enabled, output dir: {cfg.env.sim.nan_guard.output_dir}")

  # Specify directory for logging experiments.
  log_root_path = Path("logs") / "rsl_rl" / cfg.agent.experiment_name
  log_root_path.resolve()
  print(f"[INFO] Logging experiment in directory: {log_root_path}")
  log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  if cfg.agent.run_name:
    log_dir += f"_{cfg.agent.run_name}"
  log_dir = log_root_path / log_dir

  env = ManagerBasedRlEnv(
    cfg=cfg.env, device=cfg.device, render_mode="rgb_array" if cfg.video else None
  )

  resume_path = (
    get_checkpoint_path(log_root_path, cfg.agent.load_run, cfg.agent.load_checkpoint)
    if cfg.agent.resume
    else None
  )

  if cfg.video:
    env = VideoRecorder(
      env,
      video_folder=Path(log_dir) / "videos" / "train",
      step_trigger=lambda step: step % cfg.video_interval == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )
    print("[INFO] Recording videos during training.")

  env = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)

  agent_cfg = asdict(cfg.agent)
  env_cfg = asdict(cfg.env)

  if is_tracking_task:
    runner = MotionTrackingOnPolicyRunner(
      env, agent_cfg, str(log_dir), cfg.device, registry_name
    )
  elif is_locomanipulation_task:
    runner = LocomanipulationOnPolicyRunner(
      env, agent_cfg, str(log_dir), cfg.device
    )
  else:
    runner = VelocityOnPolicyRunner(env, agent_cfg, str(log_dir), cfg.device)

  runner.add_git_repo_to_log(__file__)
  if resume_path is not None:
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner.load(str(resume_path))

  dump_yaml(log_dir / "params" / "env.yaml", env_cfg)
  dump_yaml(log_dir / "params" / "agent.yaml", agent_cfg)

  runner.learn(
    num_learning_iterations=cfg.agent.max_iterations, init_at_random_ep_len=True
  )

  env.close()


def main():
  # Parse first argument to choose the task.
  # Import tasks to populate the registry.
  import mjlab.tasks  # noqa: F401

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
  )

  # Parse the rest of the arguments + allow overriding env_cfg and agent_cfg.
  env_cfg = load_env_cfg(chosen_task)
  agent_cfg = load_rl_cfg(chosen_task)
  assert isinstance(agent_cfg, RslRlOnPolicyRunnerCfg)

  args = tyro.cli(
    TrainConfig,
    args=remaining_args,
    default=TrainConfig(env=env_cfg, agent=agent_cfg),
    prog=sys.argv[0] + f" {chosen_task}",
    config=(
      tyro.conf.AvoidSubcommands,
      tyro.conf.FlagConversionOff,
    ),
  )
  del env_cfg, agent_cfg, remaining_args

  run_train(args)


if __name__ == "__main__":
  main()
