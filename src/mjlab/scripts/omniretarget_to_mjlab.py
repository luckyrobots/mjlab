from typing import Any

import numpy as np
import torch
import tyro
from tqdm import tqdm

from mjlab.entity import Entity
from mjlab.scene import Scene
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.tasks.locomanipulation.config.g1.env_cfgs import UNITREE_G1_FLAT_LOCOMANIPULATION_ENV_CFG
from mjlab.third_party.isaaclab.isaaclab.utils.math import (
    axis_angle_from_quat,
    quat_conjugate,
    quat_mul,
    quat_slerp,
)
from mjlab.viewer.offscreen_renderer import OffscreenRenderer
from mjlab.viewer.viewer_config import ViewerConfig


class MotionLoader:
    def __init__(
        self,
        motion_file: str,
        output_fps: int,
        device: torch.device | str,
        frame_range: tuple[int, int] | None = None,
    ):
        self.motion_file = motion_file
        self.input_fps = None  # will be set in _load_motion
        self.output_fps = output_fps
        self.input_dt = None  # will be set in _load_motion
        self.output_dt = 1.0 / self.output_fps
        self.current_idx = 0
        self.device = device
        self.frame_range = frame_range
        self.has_object = False  # will be set in _load_motion
        self._load_motion()
        self._interpolate_motion()
        self._compute_velocities()

    def _load_motion(self):
        """Loads the motion from the npz file (OmniRetarget format)."""
        data = np.load(self.motion_file)
        
        # Load qpos: [qw, qx, qy, qz, x, y, z, 29 joints, (optional 7D object)]
        qpos = data["qpos"]  # (T, D) where D=36 or D=43
        
        fps = float(data["fps"])
        self.input_fps = fps
        self.input_dt = 1.0 / self.input_fps

        # Check if object data is present
        self.has_object = qpos.shape[1] == 43

        # Apply frame range if specified
        if self.frame_range is not None:
            start_idx = self.frame_range[0]
            end_idx = self.frame_range[1] + 1
            qpos = qpos[start_idx:end_idx]
        
        # Extract robot base and joints (first 36 dimensions)
        base_quat = qpos[:, 0:4]    # (T, 4) [qw, qx, qy, qz]
        base_pos = qpos[:, 4:7]     # (T, 3) [x, y, z]
        joint_pos = qpos[:, 7:36]   # (T, 29)
        
        # Convert to torch tensors
        self.motion_base_poss_input = torch.from_numpy(base_pos).to(torch.float32).to(self.device)
        self.motion_base_rots_input = torch.from_numpy(base_quat).to(torch.float32).to(self.device)
        self.motion_dof_poss_input = torch.from_numpy(joint_pos).to(torch.float32).to(self.device)

        # Extract object data if present
        if self.has_object:
            object_quat = qpos[:, 36:40]  # (T, 4) [qw, qx, qy, qz]
            object_pos = qpos[:, 40:43]   # (T, 3) [x, y, z]
            
            self.motion_object_poss_input = torch.from_numpy(object_pos).to(torch.float32).to(self.device)
            self.motion_object_rots_input = torch.from_numpy(object_quat).to(torch.float32).to(self.device)

        self.input_frames = qpos.shape[0]
        self.duration = (self.input_frames - 1) * self.input_dt
        
        print(f"Loaded motion from OmniRetarget format:")
        print(f"  Frames: {self.input_frames} at {self.input_fps} FPS")
        print(f"  qpos shape: {qpos.shape}")
        print(f"  Has object: {self.has_object}")
        print(f"  Base position shape: {self.motion_base_poss_input.shape}")
        print(f"  Base rotation shape: {self.motion_base_rots_input.shape}")
        print(f"  DOF position shape: {self.motion_dof_poss_input.shape}")
        if self.has_object:
            print(f"  Object position shape: {self.motion_object_poss_input.shape}")
            print(f"  Object rotation shape: {self.motion_object_rots_input.shape}")

    def _interpolate_motion(self):
        """Interpolates the motion to the output fps."""
        times = torch.arange(
            0, self.duration, self.output_dt, device=self.device, dtype=torch.float32
        )
        self.output_frames = times.shape[0]
        index_0, index_1, blend = self._compute_frame_blend(times)
        
        # Interpolate robot motion
        self.motion_base_poss = self._lerp(
            self.motion_base_poss_input[index_0],
            self.motion_base_poss_input[index_1],
            blend.unsqueeze(1),
        )
        self.motion_base_rots = self._slerp(
            self.motion_base_rots_input[index_0],
            self.motion_base_rots_input[index_1],
            blend,
        )
        self.motion_dof_poss = self._lerp(
            self.motion_dof_poss_input[index_0],
            self.motion_dof_poss_input[index_1],
            blend.unsqueeze(1),
        )
        
        # Interpolate object motion if present
        if self.has_object:
            self.motion_object_poss = self._lerp(
                self.motion_object_poss_input[index_0],
                self.motion_object_poss_input[index_1],
                blend.unsqueeze(1),
            )
            self.motion_object_rots = self._slerp(
                self.motion_object_rots_input[index_0],
                self.motion_object_rots_input[index_1],
                blend,
            )
        
        print(
            f"Motion interpolated, input frames: {self.input_frames}, "
            f"input fps: {self.input_fps}, "
            f"output frames: {self.output_frames}, "
            f"output fps: {self.output_fps}"
        )

    def _lerp(
        self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor
    ) -> torch.Tensor:
        """Linear interpolation between two tensors."""
        return a * (1 - blend) + b * blend

    def _slerp(
        self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor
    ) -> torch.Tensor:
        """Spherical linear interpolation between two quaternions."""
        slerped_quats = torch.zeros_like(a)
        for i in range(a.shape[0]):
            slerped_quats[i] = quat_slerp(a[i], b[i], float(blend[i]))
        return slerped_quats

    def _compute_frame_blend(
        self, times: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes the frame blend for the motion."""
        phase = times / self.duration
        index_0 = (phase * (self.input_frames - 1)).floor().long()
        index_1 = torch.minimum(index_0 + 1, torch.tensor(self.input_frames - 1))
        blend = phase * (self.input_frames - 1) - index_0
        return index_0, index_1, blend

    def _compute_velocities(self):
        """Computes velocities using backward finite differences to match inference."""
        # Robot velocities
        self.motion_base_lin_vels = torch.zeros_like(self.motion_base_poss)
        self.motion_base_lin_vels[1:] = (
            self.motion_base_poss[1:] - self.motion_base_poss[:-1]
        ) / self.output_dt
        self.motion_base_lin_vels[0] = self.motion_base_lin_vels[1]

        self.motion_dof_vels = torch.zeros_like(self.motion_dof_poss)
        self.motion_dof_vels[1:] = (
            self.motion_dof_poss[1:] - self.motion_dof_poss[:-1]
        ) / self.output_dt
        self.motion_dof_vels[0] = self.motion_dof_vels[1]

        self.motion_base_ang_vels = self._so3_derivative_backward(
            self.motion_base_rots, self.output_dt
        )

        # Object velocities if present
        if self.has_object:
            self.motion_object_lin_vels = torch.zeros_like(self.motion_object_poss)
            self.motion_object_lin_vels[1:] = (
                self.motion_object_poss[1:] - self.motion_object_poss[:-1]
            ) / self.output_dt
            self.motion_object_lin_vels[0] = self.motion_object_lin_vels[1]

            self.motion_object_ang_vels = self._so3_derivative_backward(
                self.motion_object_rots, self.output_dt
            )

    def _so3_derivative_backward(
        self, rotations: torch.Tensor, dt: float
    ) -> torch.Tensor:
        """Backward finite difference for SO3."""
        q_prev, q_curr = rotations[:-1], rotations[1:]
        q_rel = quat_mul(q_curr, quat_conjugate(q_prev))
        omega = axis_angle_from_quat(q_rel) / dt
        # Pad with first velocity
        omega = torch.cat([omega[:1], omega], dim=0)
        return omega

    def get_next_state(
        self,
    ) -> tuple[
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor | None,
            torch.Tensor | None,
        ],
        bool,
    ]:
        """Gets the next state of the motion."""
        object_pos = None
        object_rot = None
        
        if self.has_object:
            object_pos = self.motion_object_poss[self.current_idx : self.current_idx + 1]
            object_rot = self.motion_object_rots[self.current_idx : self.current_idx + 1]
        
        state = (
            self.motion_base_poss[self.current_idx : self.current_idx + 1],
            self.motion_base_rots[self.current_idx : self.current_idx + 1],
            self.motion_base_lin_vels[self.current_idx : self.current_idx + 1],
            self.motion_base_ang_vels[self.current_idx : self.current_idx + 1],
            self.motion_dof_poss[self.current_idx : self.current_idx + 1],
            self.motion_dof_vels[self.current_idx : self.current_idx + 1],
            object_pos,
            object_rot,
        )
        self.current_idx += 1
        reset_flag = False
        if self.current_idx >= self.output_frames:
            self.current_idx = 0
            reset_flag = True
        return state, reset_flag


def run_sim(
    sim: Simulation,
    scene: Scene,
    joint_names,
    input_file,
    output_fps,
    output_name,
    render,
    frame_range,
    renderer: OffscreenRenderer | None = None,
):
    motion = MotionLoader(
        motion_file=input_file,
        output_fps=output_fps,
        device=sim.device,
        frame_range=frame_range,
    )

    robot: Entity = scene["robot"]
    robot_joint_indexes = robot.find_joints(joint_names, preserve_order=True)[0]

    if motion.has_object:
        object: Entity = scene["object"]

    log: dict[str, Any] = {
        "fps": [output_fps],
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }
    
    # Add object logging if object is present
    if motion.has_object:
        log["object_pos_w"] = []
        log["object_quat_w"] = []
    
    file_saved = False

    frames = []
    scene.reset()

    print(f"\nStarting simulation with {motion.output_frames} frames...")
    if render:
        print("Rendering enabled - generating video frames...")

    # Create progress bar
    pbar = tqdm(
        total=motion.output_frames,
        desc="Processing frames",
        unit="frame",
        ncols=100,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )

    frame_count = 0
    while not file_saved:
        (
            (
                motion_base_pos,
                motion_base_rot,
                motion_base_lin_vel,
                motion_base_ang_vel,
                motion_dof_pos,
                motion_dof_vel,
                motion_object_pos,
                motion_object_rot,
            ),
            reset_flag,
        ) = motion.get_next_state()

        root_states = robot.data.default_root_state.clone()
        root_states[:, 0:3] = motion_base_pos
        root_states[:, :2] += scene.env_origins[:, :2]
        root_states[:, 3:7] = motion_base_rot
        root_states[:, 7:10] = motion_base_lin_vel
        root_states[:, 10:] = motion_base_ang_vel
        robot.write_root_state_to_sim(root_states)

        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, robot_joint_indexes] = motion_dof_pos
        joint_vel[:, robot_joint_indexes] = motion_dof_vel
        robot.write_joint_state_to_sim(joint_pos, joint_vel)

        if motion.has_object and motion_object_pos is not None:
            object_root_states = object.data.default_root_state.clone()  # pyright: ignore[reportPossiblyUnboundVariable]
            object_root_states[:, 0:3] = motion_object_pos
            object_root_states[:, :2] += scene.env_origins[:, :2]
            object_root_states[:, 3:7] = motion_object_rot  # pyright: ignore
            object.write_root_state_to_sim(object_root_states)  # pyright: ignore[reportPossiblyUnboundVariable]

        sim.forward()
        scene.update(sim.mj_model.opt.timestep)
        if render and renderer is not None:
            renderer.update(sim.data)
            frames.append(renderer.render())

        if not file_saved:
            log["joint_pos"].append(robot.data.joint_pos[0, :].cpu().numpy().copy())
            log["joint_vel"].append(robot.data.joint_vel[0, :].cpu().numpy().copy())
            log["body_pos_w"].append(
                robot.data.body_link_pos_w[0, :].cpu().numpy().copy()
            )
            log["body_quat_w"].append(
                robot.data.body_link_quat_w[0, :].cpu().numpy().copy()
            )
            log["body_lin_vel_w"].append(
                robot.data.body_link_lin_vel_w[0, :].cpu().numpy().copy()
            )
            log["body_ang_vel_w"].append(
                robot.data.body_link_ang_vel_w[0, :].cpu().numpy().copy()
            )

            # Log object data if present
            if motion.has_object and motion_object_pos is not None:
                # Store sim-measured object pose as flat (3,) and (4,) arrays.
                log["object_pos_w"].append(
                    object.data.body_link_pos_w[0, 0].cpu().numpy().copy()  # pyright: ignore[reportPossiblyUnboundVariable]
                )
                log["object_quat_w"].append(
                    object.data.body_link_quat_w[0, 0].cpu().numpy().copy()  # pyright: ignore[reportPossiblyUnboundVariable]
                )

            torch.testing.assert_close(
                robot.data.body_link_lin_vel_w[0, 0], motion_base_lin_vel[0]
            )
            torch.testing.assert_close(
                robot.data.body_link_ang_vel_w[0, 0], motion_base_ang_vel[0]
            )

            frame_count += 1
            pbar.update(1)

            if frame_count % 100 == 0:  # Update every 100 frames to avoid spam
                elapsed_time = frame_count / output_fps
                pbar.set_description(f"Processing frames (t={elapsed_time:.1f}s)")

            if reset_flag and not file_saved:
                file_saved = True
                pbar.close()

                print("\nStacking arrays and saving data...")
                keys_to_stack = [
                    "joint_pos",
                    "joint_vel",
                    "body_pos_w",
                    "body_quat_w",
                    "body_lin_vel_w",
                    "body_ang_vel_w",
                ]
                if motion.has_object:
                    keys_to_stack.extend([
                        "object_pos_w",
                        "object_quat_w",
                    ])
                
                for k in keys_to_stack:
                    log[k] = np.stack(log[k], axis=0)

                print("Saving to /tmp/motion.npz...")
                np.savez("/tmp/motion.npz", **log)  # type: ignore[arg-type]

                print("Uploading to Weights & Biases...")
                import wandb

                COLLECTION = output_name
                run = wandb.init(
                    project="omniretarget_to_mjlab", name=COLLECTION, entity="mjlab"
                )
                print(f"[INFO]: Logging motion to wandb: {COLLECTION}")
                REGISTRY = "motions"
                logged_artifact = run.log_artifact(
                    artifact_or_path="/tmp/motion.npz", name=COLLECTION, type=REGISTRY
                )
                run.link_artifact(
                    artifact=logged_artifact,
                    target_path=f"wandb-registry-{REGISTRY}/{COLLECTION}",
                )
                print(f"[INFO]: Motion saved to wandb registry: {REGISTRY}/{COLLECTION}")

                if render:
                    from moviepy import ImageSequenceClip

                    print("Creating video...")
                    clip = ImageSequenceClip(frames, fps=output_fps)
                    clip.write_videofile("./motion.mp4")

                    print("Logging video to wandb...")
                    wandb.log({"motion_video": wandb.Video("./motion.mp4", format="mp4")})

                wandb.finish()


def main(
    input_file: str,
    output_name: str,
    output_fps: float = 50.0,
    device: str = "cuda:0",
    render: bool = False,
    frame_range: tuple[int, int] | None = None,
):
    """Convert OmniRetarget format (qpos) to mjlab format (body_pos_w, joint_pos, etc).

    Args:
        input_file: Path to the input npz file (OmniRetarget format with 'qpos' and 'fps' keys).
        output_name: Name for the output (used in wandb).
        output_fps: Desired output frame rate.
        device: Device to use.
        render: Whether to render the simulation and save a video.
        frame_range: Range of frames to process from the npz file (start_idx, end_idx).
    """
    sim_cfg = SimulationCfg()
    sim_cfg.mujoco.timestep = 1.0 / output_fps

    scene = Scene(UNITREE_G1_FLAT_LOCOMANIPULATION_ENV_CFG.scene, device=device)
    model = scene.compile()

    sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)

    scene.initialize(sim.mj_model, sim.model, sim.data)

    renderer = None
    if render:
        viewer_cfg = ViewerConfig(
            height=480,
            width=640,
            origin_type=ViewerConfig.OriginType.ASSET_ROOT,
            distance=2.0,
            elevation=-5.0,
            azimuth=20,
        )
        renderer = OffscreenRenderer(
            model=sim.mj_model,
            cfg=viewer_cfg,
            scene=scene,
        )
        renderer.initialize()

    run_sim(
        sim=sim,
        scene=scene,
        joint_names=[
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ],
        input_file=input_file,
        output_fps=output_fps,
        output_name=output_name,
        render=render,
        frame_range=frame_range,
        renderer=renderer,
    )


if __name__ == "__main__":
    tyro.cli(main)