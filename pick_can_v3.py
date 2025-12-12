from typing import Any, Dict, Union

import numpy as np
import sapien
import torch
import os

import mani_skill.envs.utils.randomization as randomization
from mani_skill.agents.robots import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose

PICK_CAN_V3_DOC_STRING = """**Task Description:**
A modification of the PickCan task (v3). The objective is to grasp a Coke can and move it to a target goal position while avoiding 
Pepsi cans placed around the Coke can as obstacles.

**Randomizations:**
- The Coke can and Pepsi cans are randomized on the table.
- A safety margin ensures the Pepsi cans do not spawn inside the Coke can.

**Success Conditions:**
- The Coke can position is within `goal_thresh` of the goal position.
- The robot is static.

**Penalties:**
- Penalty if the robot hits any of the Pepsi cans.
- Large penalty if any of the Pepsi cans tips over/falls.
"""

@register_env("PickCan-v3", max_episode_steps=200)
class PickCanEnv3(BaseEnv):
    SUPPORTED_ROBOTS = ["panda"]
    agent: Union[Panda]
    goal_thresh = 0.025
    can_spawn_half_size = 0.05
    can_spawn_center = (0, 0)

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.can_half_size = 0.03
        self.goal_thresh = 0.025
        self.can_spawn_half_size = 0.1
        self.can_spawn_center = (0, 0)
        self.max_goal_height = 0.25

        # Obstacle setup
        self.num_obstacles = 4 # Number of Pepsi cans surrounding the Coke can as obstacles
        
        # Camera setups
        self.sensor_cam_eye_pos = [0.3, 0, 0.6]
        self.sensor_cam_target_pos = [-0.1, 0, 0.1]
        self.human_cam_eye_pos = [0.6, 0.7, 0.6]
        self.human_cam_target_pos = [0.0, 0.0, 0.35]

        # Curriculum Learning Setup
        # period = number of steps / num_envs
        self.total_step_calls = 0
        self.curriculum_period = 10_000_000 // 1024
        
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(
            eye=self.sensor_cam_eye_pos, target=self.sensor_cam_target_pos
        )
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(
            eye=self.human_cam_eye_pos, target=self.human_cam_target_pos
        )
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # --- Common Properties ---
        self.can_scale = 0.03 
        self.can_radius = 0.03 
        self.can_height = 0.12 
        self.scaled_can_half_height = (self.can_height * self.can_scale) / 2
        mass = 0.4
        I_z = 0.5 * mass * self.can_radius**2 
        I_xy = mass * (2 * self.can_height)**2 / 3 
        inertia = np.array([I_xy, I_xy, I_z], dtype=np.float32)

        # --- Load Coke Can ---
        coke_builder = self.scene.create_actor_builder()
        coke_builder.set_initial_pose(sapien.Pose(p=[0, 0, 0]))
        coke_path = os.path.join(os.path.dirname(__file__), "/home/jypaulsung/Sapien/Shared/models/coke/coke.obj")
        coke_builder.add_visual_from_file(coke_path, scale=[self.can_scale]*3)
        coke_builder.add_convex_collision_from_file(coke_path, scale=[self.can_scale]*3)
        coke_builder.set_mass_and_inertia(mass, sapien.Pose([0, 0, self.scaled_can_half_height*(-0.1)],[1, 0, 0, 0]), inertia)
    
        self.coke_cans: list[sapien.Actor] = []
        for i in range(1):
            can = coke_builder.build(name=f"coke{i+1}")
            self.coke_cans.append(can)

        # --- Load Pepsi Can (Obstacle) ---
        pepsi_builder = self.scene.create_actor_builder()
        pepsi_builder.set_initial_pose(sapien.Pose(p=[0, 0, 0]))
        pepsi_path = os.path.join(os.path.dirname(__file__), "/home/jypaulsung/Sapien/Shared/models/pepsi/pepsi.obj")
        
        # Rotation of 90 degrees (pi/2) around X-axis
        # Quaternion formula: [cos(theta/2), sin(theta/2) * x, sin(theta/2) * y, sin(theta/2) * z]
        theta = np.pi / 2
        half_theta = theta / 2
        q_rotate = [np.cos(half_theta), np.sin(half_theta), 0, 0]

        pepsi_builder.add_visual_from_file(pepsi_path, pose=sapien.Pose(q=q_rotate), scale=[self.can_scale]*3)
        pepsi_builder.add_convex_collision_from_file(pepsi_path, pose=sapien.Pose(q=q_rotate), scale=[self.can_scale]*3)
        pepsi_builder.set_mass_and_inertia(mass, sapien.Pose([0, 0, self.scaled_can_half_height*(-0.1)],[1, 0, 0, 0]), inertia)

        self.pepsi_cans = []
        for i in range(self.num_obstacles):
            p_can = pepsi_builder.build(name=f"pepsi{i+1}")
            self.pepsi_cans.append(p_can)
        

        # --- Goal Site ---
        self.goal_site = actors.build_sphere(
            self.scene,
            radius=self.goal_thresh,
            color=[0, 1, 0, 1],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        self._hidden_objects.append(self.goal_site)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # --- 1. Initialize Coke Position (Vectorized) ---
            xyz = torch.zeros((b, 3), device=self.device)
            xyz[:, :2] = (
                torch.rand((b, 2), device=self.device) * self.can_spawn_half_size * 2
                - self.can_spawn_half_size
            )
            xyz[:, 0] += self.can_spawn_center[0]
            xyz[:, 1] += self.can_spawn_center[1]
            xyz[:, 2] = 0 
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.coke_cans[0].set_pose(Pose.create_from_pq(xyz, qs))

            # --- 2. Initialize Goal (Vectorized) ---
            goal_min_dist = 0.12  # Minimum distance from Coke
            goal_xyz = torch.zeros((b, 3), device=self.device)
            
            # Initialize with random values
            goal_valid = torch.zeros(b, dtype=torch.bool, device=self.device)
            
            # Rejection sampling for Goal
            for _ in range(10): # Max attempts
                if goal_valid.all(): break
                needed = ~goal_valid
                count = needed.sum()
                
                temp_goal_xy = (
                    torch.rand((count, 2), device=self.device) * self.can_spawn_half_size * 2
                    - self.can_spawn_half_size
                )
                temp_goal_xy[:, 0] += self.can_spawn_center[0]
                temp_goal_xy[:, 1] += self.can_spawn_center[1]
                
                # Check distance to Coke
                dist = torch.linalg.norm(temp_goal_xy - xyz[needed, :2], axis=1)
                valid_goals = dist > goal_min_dist
                
                # Update valid goals
                indices = torch.nonzero(needed).flatten()[valid_goals]
                goal_xyz[indices, :2] = temp_goal_xy[valid_goals]
                goal_valid[indices] = True

            # Fallback: Force offset if still invalid
            if not goal_valid.all():
                needed = ~goal_valid
                goal_xyz[needed, :2] = xyz[needed, :2]
                goal_xyz[needed, 0] += 0.15 
                goal_xyz[needed, 1] += 0.15 

            goal_xyz[:, 2] = torch.rand((b), device=self.device) * self.max_goal_height + xyz[:, 2]
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))

            # --- 3. Initialize Pepsi Cans (Vectorized Clutter) ---
            # Configs
            min_dist_coke_goal = self.can_radius * 2.2  # ~9cm from Coke/Goal
            min_dist_inter_obs = self.can_radius * 2.2  # ~6.6cm between Pepsis
            spawn_radius_min = self.can_radius * 2.5
            spawn_radius_max = 0.15
            
            # Storage for all Pepsi positions: (Batch, Num_Obstacles, 3)
            pepsi_xyz = torch.zeros((b, self.num_obstacles, 3), device=self.device)
            
            # We generate obstacles one by one (k=0, k=1, ...) for the whole batch
            for k in range(self.num_obstacles):
                # Track which envs in the batch have successfully placed obstacle 'k'
                placed_k = torch.zeros(b, dtype=torch.bool, device=self.device)
                
                # Max retry attempts for this specific obstacle index
                for _ in range(20):
                    if placed_k.all(): break
                    
                    needed_mask = ~placed_k
                    count = needed_mask.sum()
                    
                    # 1. Generate Candidates (Polar around Coke)
                    theta = torch.rand(count, device=self.device) * 2 * np.pi
                    r = torch.rand(count, device=self.device) * (spawn_radius_max - spawn_radius_min) + spawn_radius_min
                    
                    # Convert to Cartesian relative to Coke
                    cand_xy = torch.zeros((count, 2), device=self.device)
                    coke_pos = xyz[needed_mask, :2]
                    cand_xy[:, 0] = coke_pos[:, 0] + r * torch.cos(theta)
                    cand_xy[:, 1] = coke_pos[:, 1] + r * torch.sin(theta)

                    # 2. Validation Checks
                    # A. Check distance to Goal (Critical Fix)
                    goal_pos = goal_xyz[needed_mask, :2]
                    dist_goal = torch.linalg.norm(cand_xy - goal_pos, axis=1)
                    valid_geo = dist_goal > min_dist_coke_goal

                    # B. Check distance to PREVIOUS Pepsi cans (0 to k-1)
                    if k > 0 and valid_geo.any():
                        # Get previously placed pepsis for the 'needed' envs
                        # Shape: (Count, k, 2) vs Candidate: (Count, 1, 2)
                        prev_pepsis = pepsi_xyz[needed_mask, :k, :2] 
                        curr_cand = cand_xy.unsqueeze(1) # (Count, 1, 2)
                        
                        dists = torch.linalg.norm(prev_pepsis - curr_cand, dim=2) # (Count, k)
                        # Valid if ALL previous distances > min_inter_obs
                        valid_inter = (dists > min_dist_inter_obs).all(dim=1)
                        valid_geo = valid_geo & valid_inter

                    # 3. Update Valid Entries
                    valid_indices = torch.nonzero(needed_mask).flatten()[valid_geo]
                    
                    # Extract only the valid candidates from the temp batch
                    valid_candidates = cand_xy[valid_geo]
                    
                    pepsi_xyz[valid_indices, k, :2] = valid_candidates
                    placed_k[valid_indices] = True
                
                # Fallback for this obstacle if we couldn't place it safely after retries
                # Just place it far away (off the table or edge) to prevent physics explosion
                if not placed_k.all():
                    failed = ~placed_k
                    # Place effectively "out of play" or at a fixed safe offset
                    # Here we place it at max radius + small buffer
                    fallback_r = spawn_radius_max + 0.05
                    fallback_xy = xyz[failed, :2] # Coke pos
                    fallback_xy[:, 0] += fallback_r # Simply offset X
                    pepsi_xyz[failed, k, :2] = fallback_xy

            # Apply Poses
            pepsi_qs = randomization.random_quaternions(b * self.num_obstacles, lock_x=True, lock_y=True)
            pepsi_qs = pepsi_qs.reshape(b, self.num_obstacles, 4)
            
            for k, p_can in enumerate(self.pepsi_cans):
                p_can.set_pose(Pose.create_from_pq(pepsi_xyz[:, k], pepsi_qs[:, k]))

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            is_grasped=info["is_grasped"],
            tcp_pose=self.agent.tcp_pose.raw_pose,
            goal_pos=self.goal_site.pose.p,
        )
        if "state" in self.obs_mode:
            # Stack poses of all obstacles: Shape (B, Num_Obstacles, 7)
            pepsi_poses = torch.stack([can.pose.raw_pose for can in self.pepsi_cans], dim=1)
            # Flatten to (B, Num_Obstacles * 7) so it fits in a fixed-size MLP input
            pepsi_poses = pepsi_poses.flatten(start_dim=1)

            # Stack relative positions: Shape (B, Num_Obstacles, 3) -> Flatten to (B, Num_Obstacles * 3)
            tcp_to_pepsi_pos = torch.stack(
                [can.pose.p - self.agent.tcp_pose.p for can in self.pepsi_cans], dim=1
            ).flatten(start_dim=1)

            obs.update(
                obj_pose=self.coke_cans[0].pose.raw_pose,
                pepsi_poses=pepsi_poses, 
                tcp_to_obj_pos=self.coke_cans[0].pose.p - self.agent.tcp_pose.p,
                obj_to_goal_pos=self.goal_site.pose.p - self.coke_cans[0].pose.p,
                tcp_to_pepsi_pos=tcp_to_pepsi_pos
            )
        return obs

    def evaluate(self):
        # 1. Check Goal Success (Vectorized)
        # Shape: (B,)
        is_obj_placed = (
            torch.linalg.norm(self.goal_site.pose.p - self.coke_cans[0].pose.p, axis=1)
            <= self.goal_thresh
        )
        
        # 2. Check Grasp/Static (Vectorized)
        # Shape: (B,)
        is_grasped = self.agent.is_grasping(self.coke_cans[0])
        is_robot_static = self.agent.is_static(0.2)

        # 3. Check Obstacles (Fully Vectorized - No Loop)
        # Gather all Pepsi orientations into a single tensor
        # [can1_q, can2_q, ...] -> Stack -> Shape: (B, Num_Obstacles, 4)
        pepsi_qs = torch.stack([can.pose.q for can in self.pepsi_cans], dim=1)
        
        # Extract x and y components (columns 1 and 2 of the quaternion)
        # Shape: (B, Num_Obstacles)
        x = pepsi_qs[:, :, 1]
        y = pepsi_qs[:, :, 2]
        
        # Calculate alignment for all cans at once
        # Shape: (B, Num_Obstacles)
        up_alignment = 1 - 2 * (x**2 + y**2)

        # Check conditions across the 'Num_Obstacles' dimension (dim=1)
        # is_pepsi_fallen: True if ANY can in the row is < 0.5
        is_pepsi_fallen = (up_alignment < 0.5).any(dim=1)
        
        # is_pepsi_upright: True if ALL cans in the row are > 0.9
        all_pepsi_upright = (up_alignment > 0.9).all(dim=1)

        # 4. Final Success Metric
        success = is_obj_placed & is_robot_static & all_pepsi_upright

        return {
            "success": success,
            "is_obj_placed": is_obj_placed,
            "is_robot_static": is_robot_static,
            "is_grasped": is_grasped,
            "is_pepsi_fallen": is_pepsi_fallen,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # --- POSITIVE REWARD SHAPING ---
        # Reaching Reward (Max 1.0)
        tcp_to_obj_dist = torch.linalg.norm(
            self.coke_cans[0].pose.p - self.agent.tcp_pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        reward = reaching_reward

        # --- NEW: NEAR-GRASP INCENTIVE ---
        # 1. Get Gripper Width (Panda's last 2 joints are the fingers)
        # Summing them gives approx width in meters 
        qpos = self.agent.robot.get_qpos()
        gripper_width = qpos[:, -2:].sum(dim=1)
        
        # 2. Define "Near" (e.g., within 15cm)
        is_near_obj = tcp_to_obj_dist < 0.15

        # 3. Reward Calculation
        # Reward = (How Open 0-1) * (Is Near?) * (Not Already Grasping?)
        # We assume max width is ~0.08m
        openness_ratio = torch.clamp(gripper_width / 0.08, max=1.0)
        
        # We add a small bonus (e.g., 0.5) for approaching with an open hand
        near_grasp_reward = 0.5 * openness_ratio * is_near_obj.float() * (~info["is_grasped"]).float()
        reward += near_grasp_reward

        # Grasp Reward (Max 2.0)
        is_grasped = info["is_grasped"]
        reward += is_grasped * 2.0

        # Placing Reward (Max 1.0)
        obj_to_goal_dist = torch.linalg.norm(
            self.goal_site.pose.p - self.coke_cans[0].pose.p, axis=1
        )
        place_reward = 1 - torch.tanh(5 * obj_to_goal_dist)
        reward += place_reward * is_grasped

        # Static Reward (Max 2.0)
        qvel = self.agent.robot.get_qvel()
        qvel = qvel[..., :-2]
        static_reward = 1 - torch.tanh(5 * torch.linalg.norm(qvel, axis=1))
        reward += static_reward * info["is_obj_placed"] * 2.0

        # Action damping
        reward -= 0.05 * torch.sum(action**2, dim=1)

        # Success Bonus (Max 5.0)
        reward[info["success"]] = 5

        # --- Calculate Curriculum Factor ---
        curriculum_factor = min(self.total_step_calls / self.curriculum_period, 1.0)

        # --- NEGATIVE PENALTIES (Modified) ---
        total_collision_penalty = 0.0
        any_pepsi_fallen = torch.zeros_like(reward, dtype=torch.bool)

        for p_can in self.pepsi_cans:
            # Velocity Penalty Calculation
            lin_vel = p_can.linear_velocity
            speed = torch.linalg.norm(lin_vel, axis=1)
            speed = torch.where(speed < 0.02, torch.zeros_like(speed), speed)
            total_collision_penalty += speed * 2.0

            # Fallen Check
            pepsi_q = p_can.pose.q
            x, y = pepsi_q[:, 1], pepsi_q[:, 2]
            up_alignment = 1 - 2 * (x**2 + y**2)
            is_fallen = up_alignment < 0.5
            any_pepsi_fallen = any_pepsi_fallen | is_fallen

        # Clamp the velocity penalty to max 1.0.
        clipped_velocity_penalty = torch.clamp(total_collision_penalty, max=1.0)
        reward -= clipped_velocity_penalty * curriculum_factor

        # Catastrophic penalty (Falling) remains large and unclipped
        # because this is a failure state we want to avoid at all costs.
        reward -= 5.0 * any_pepsi_fallen

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 5
    
    def step(self, action):
        # 1. Increment internal counter
        self.total_step_calls += 1
        
        # 2. Call the parent's step function to handle physics/simulation
        # This is CRITICAL. If you forget super().step(), the robot won't move.
        obs, reward, terminated, truncated, info = super().step(action)

        # 3. Vectorized Termination (Sudden Death)
        # Update the 'terminated' flag if any Pepsi can has fallen
        terminated = terminated | info["is_pepsi_fallen"]
        
        return obs, reward, terminated, truncated, info

PickCanEnv3.__doc__ = PICK_CAN_V3_DOC_STRING.format(robot_id="Panda")