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

PICK_CAN_V2_DOC_STRING = """**Task Description:**
A modification of the PickCan task (v2). The objective is to grasp a coke can and move it to a target goal position while avoiding a pepsi can obstacle.

**Randomizations:**
- The coke can and pepsi can are randomized on the table.
- A safety margin ensures the pepsi can does not spawn inside the coke can.

**Success Conditions:**
- The coke can position is within `goal_thresh` of the goal position.
- The robot is static.

**Penalties:**
- Penalty if the robot gets too close (touches) to the pepsi can.
- Large penalty if the pepsi can tips over/falls.
"""

@register_env("PickCan-v2", max_episode_steps=200)
class PickCanEnv2(BaseEnv):
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
        
        # Camera setups
        self.sensor_cam_eye_pos = [0.3, 0, 0.6]
        self.sensor_cam_target_pos = [-0.1, 0, 0.1]
        self.human_cam_eye_pos = [0.6, 0.7, 0.6]
        self.human_cam_target_pos = [0.0, 0.0, 0.35]
        
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

        self.pepsi_cans: list[sapien.Actor] = []
        for i in range(1):
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

            # --- 1. Initialize Coke Position ---
            xyz = torch.zeros((b, 3))
            xyz[:, :2] = (
                torch.rand((b, 2)) * self.can_spawn_half_size * 2
                - self.can_spawn_half_size
            )
            xyz[:, 0] += self.can_spawn_center[0]
            xyz[:, 1] += self.can_spawn_center[1]
            xyz[:, 2] = 0 
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.coke_cans[0].set_pose(Pose.create_from_pq(xyz, qs))

            # --- 2. Initialize Goal (With Min Distance Check) ---
            # We want the goal to be at least 10cm away from start to force movement
            goal_min_dist = 0.10 
            goal_xyz = torch.zeros((b, 3))
            
            # Simple rejection sampling for goal
            goal_valid = torch.zeros(b, dtype=torch.bool, device=self.device)
            for _ in range(5): # Quick checks
                if goal_valid.all(): break
                needed = ~goal_valid
                count = needed.sum()
                
                temp_goal_xy = (
                    torch.rand((count, 2), device=self.device) * self.can_spawn_half_size * 2
                    - self.can_spawn_half_size
                )
                temp_goal_xy[:, 0] += self.can_spawn_center[0]
                temp_goal_xy[:, 1] += self.can_spawn_center[1]
                
                dist = torch.linalg.norm(temp_goal_xy - xyz[needed, :2], axis=1)
                valid_goals = dist > goal_min_dist
                
                # Update valid goals
                indices = torch.nonzero(needed).flatten()[valid_goals]
                goal_xyz[indices, :2] = temp_goal_xy[valid_goals]
                goal_valid[indices] = True

            # Fallback for goal: if still too close, just push it +0.15 in X and +0.15 in Y
            if not goal_valid.all():
                goal_xyz[~goal_valid, :2] = xyz[~goal_valid, :2]
                goal_xyz[~goal_valid, 0] += 0.15 # Force offset
                goal_xyz[~goal_valid, 1] += 0.15 # Force offset

            goal_xyz[:, 2] = torch.rand((b)) * self.max_goal_height + xyz[:, 2]
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))

            # --- 3. Initialize Pepsi (Obstacle) ---
            min_dist_safe = self.can_radius * 3 # Minimum safe distance from Coke and Goal
            
            pepsi_xyz = torch.zeros((b, 3))
            pepsi_xyz[:, 2] = 0
            
            valid_mask = torch.zeros(b, dtype=torch.bool, device=self.device)
            max_loops = 10
            loop_count = 0
            
            while not valid_mask.all() and loop_count < max_loops:
                not_valid = ~valid_mask
                count = not_valid.sum()
                
                if count == 0: break

                temp_xy = (
                    torch.rand((count, 2), device=self.device) * self.can_spawn_half_size * 2
                    - self.can_spawn_half_size
                )
                temp_xy[:, 0] += self.can_spawn_center[0]
                temp_xy[:, 1] += self.can_spawn_center[1]
                
                # Check 1: Distance to Coke
                dist_to_coke = torch.linalg.norm(temp_xy - xyz[not_valid, :2], axis=1)
                
                # Check 2: Distance to Goal (XY plane only)
                dist_to_goal = torch.linalg.norm(temp_xy - goal_xyz[not_valid, :2], axis=1)
                
                # Valid only if FAR from Coke AND FAR from Goal
                newly_valid = (dist_to_coke > min_dist_safe) & (dist_to_goal > min_dist_safe)
                
                indices_to_update = torch.nonzero(not_valid).flatten()[newly_valid]
                pepsi_xyz[indices_to_update, :2] = temp_xy[newly_valid]
                
                valid_mask[indices_to_update] = True
                loop_count += 1
            
            # --- IMPROVED FALLBACK ---
            # If we couldn't find a spot, place Pepsi roughly "behind" the Coke 
            # relative to the table center, or just offset safely
            # Here we calculate vector from Coke -> Center, and place Pepsi along that
            if not valid_mask.all():
                failed_indices = ~valid_mask
                
                # Center of spawn area
                center = torch.tensor([self.can_spawn_center], device=self.device)
                
                # Vector from Coke to Center
                vec_to_center = center - xyz[failed_indices, :2]
                
                # Normalize and scale by safe distance
                # Add a small epsilon to avoid division by zero
                vec_len = torch.linalg.norm(vec_to_center, axis=1, keepdim=True) + 1e-6
                offset = (vec_to_center / vec_len) * min_dist_safe
                
                # Place Pepsi towards the center (prevent falling off table)
                pepsi_xyz[failed_indices, :2] = xyz[failed_indices, :2] + offset

            pepsi_qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.pepsi_cans[0].set_pose(Pose.create_from_pq(pepsi_xyz, pepsi_qs))

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            is_grasped=info["is_grasped"],
            tcp_pose=self.agent.tcp_pose.raw_pose,
            goal_pos=self.goal_site.pose.p,
        )
        if "state" in self.obs_mode:
            obs.update(
                obj_pose=self.coke_cans[0].pose.raw_pose,
                pepsi_pose=self.pepsi_cans[0].pose.raw_pose, 
                tcp_to_obj_pos=self.coke_cans[0].pose.p - self.agent.tcp_pose.p,
                obj_to_goal_pos=self.goal_site.pose.p - self.coke_cans[0].pose.p,
                tcp_to_pepsi_pos=self.pepsi_cans[0].pose.p - self.agent.tcp_pose.p
            )
        return obs

    def evaluate(self):
        is_obj_placed = (
            torch.linalg.norm(self.goal_site.pose.p - self.coke_cans[0].pose.p, axis=1)
            <= self.goal_thresh
        )
        is_grasped = self.agent.is_grasping(self.coke_cans[0])
        is_robot_static = self.agent.is_static(0.2)

        pepsi_q = self.pepsi_cans[0].pose.q
        x, y = pepsi_q[:, 1], pepsi_q[:, 2]
        up_alignment = 1 - 2 * (x**2 + y**2)

        # Check if Pepsi can is upright (not tipped over)
        is_pepsi_fallen = up_alignment < 0.5
        is_pepsi_upright = up_alignment > 0.9

        success = is_obj_placed & is_robot_static & is_pepsi_upright

        return {
            "success": success,
            "is_obj_placed": is_obj_placed,
            "is_robot_static": is_robot_static,
            "is_grasped": is_grasped,
            "is_pepsi_fallen": is_pepsi_fallen,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # Base Reward
        tcp_to_obj_dist = torch.linalg.norm(
            self.coke_cans[0].pose.p - self.agent.tcp_pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        reward = reaching_reward

        is_grasped = info["is_grasped"]
        reward += is_grasped

        obj_to_goal_dist = torch.linalg.norm(
            self.goal_site.pose.p - self.coke_cans[0].pose.p, axis=1
        )
        place_reward = 1 - torch.tanh(5 * obj_to_goal_dist)
        reward += place_reward * is_grasped

        qvel = self.agent.robot.get_qvel()
        qvel = qvel[..., :-2]
        static_reward = 1 - torch.tanh(5 * torch.linalg.norm(qvel, axis=1))
        reward += static_reward * info["is_obj_placed"]

        reward[info["success"]] = 5

        # --- PENALTIES FOR HITTING THE PEPSI CAN ---
        # Penalty for hitting the Pepsi can (based on its velocity)
        # directly access the linear velocity (returns [vx, vy, vz])
        pepsi_lin_vel = self.pepsi_cans[0].linear_velocity
        pepsi_speed = torch.linalg.norm(pepsi_lin_vel, axis=1)
        
        # Apply Deadzone
        # Ignore any movement slower than 2.0cm/s to filter out simulation noise
        # This ensures the penalty is exactly 0.0 when the can is just sitting there
        noise_threshold = 0.02
        pepsi_speed = torch.where(
            pepsi_speed < noise_threshold, 
            torch.zeros_like(pepsi_speed), 
            pepsi_speed
        )

        # Calculate Scaled Penalty
        # 0.1 m/s * 5.0 = -0.5 reward (light hit)
        # 1.0 m/s * 5.0 = -5.0 reward (hard hit)
        collision_penalty_scale = 5.0
        raw_penalty = pepsi_speed * collision_penalty_scale

        # Clamp the Penalty
        # Ensure the penalty never exceeds a certain amount 
        # Prevent a physics explosion from ruining the training
        clipped_penalty = torch.clamp(raw_penalty, max=5.0)
        
        reward -= clipped_penalty

        # Large penalty if Pepsi can falls over
        reward -= 5.0 * info["is_pepsi_fallen"]

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 5

PickCanEnv2.__doc__ = PICK_CAN_V2_DOC_STRING.format(robot_id="Panda")