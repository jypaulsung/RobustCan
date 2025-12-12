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

PICK_CAN_DOC_STRING = """**Task Description:**
A simple task where the objective is to grasp a coke can with the {robot_id} robot and move it to a target goal position. 

**Randomizations:**
- the can's xy position is randomized on top of a table in the region [0.1, 0.1] x [-0.1, -0.1]. It is placed flat on the table
- the can's z-axis rotation is randomized to a random angle
- the target goal position (marked by a green sphere) of the can has its xy position randomized in the region [0.1, 0.1] x [-0.1, -0.1] and z randomized in [0, 0.3]

**Success Conditions:**
- the can position is within `goal_thresh` (default 0.025m) euclidean distance of the goal position
- the robot is static (q velocity < 0.2)
"""


@register_env("PickCan-v1", max_episode_steps=50)
class PickCanEnv(BaseEnv):
    SUPPORTED_ROBOTS = [
        "panda"
    ]
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

        self.can_scale = 0.03 # scale factor for the can URDF model
        self.can_radius = 0.03 # radius of the can (meters)
        self.can_height = 0.12 # height of the can (meters)

        self.scaled_can_half_height = (self.can_height * self.can_scale) / 2

        mass = 0.4 # mass of the can (kg)
        I_z = 0.5 * mass * self.can_radius**2 # moment of inertia around z-axis
        I_xy = mass * (2 * self.can_height)**2 / 3 # moment of inertia around x and y axes

        can_builder = self.scene.create_actor_builder()
        can_builder.set_initial_pose(sapien.Pose(p=[0, 0, 0]))
        model_path = os.path.join(os.path.dirname(__file__), "/home/jypaulsung/Sapien/Shared/models/coke/coke.obj") # path to the can model
        can_builder.add_visual_from_file(
            model_path,
            scale=[self.can_scale]*3
        )
        can_builder.add_convex_collision_from_file(model_path, scale=[self.can_scale]*3)
        can_builder.set_mass_and_inertia(mass, sapien.Pose([0, 0, self.scaled_can_half_height*(-0.1)],[1, 0, 0, 0]), np.array([I_xy, I_xy, I_z], dtype=np.float32))
    
        self.cans: list[sapien.Actor] = []
        for i in range(1):
            can = can_builder.build(name=f"coke{i+1}")
            self.cans.append(can)

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
            xyz = torch.zeros((b, 3))
            xyz[:, :2] = (
                torch.rand((b, 2)) * self.can_spawn_half_size * 2
                - self.can_spawn_half_size
            )
            xyz[:, 0] += self.can_spawn_center[0]
            xyz[:, 1] += self.can_spawn_center[1]

            # xyz[:, 2] = self.can_half_size
            xyz[:, 2] = 0
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.cans[0].set_pose(Pose.create_from_pq(xyz, qs))

            goal_xyz = torch.zeros((b, 3))
            goal_xyz[:, :2] = (
                torch.rand((b, 2)) * self.can_spawn_half_size * 2
                - self.can_spawn_half_size
            )
            goal_xyz[:, 0] += self.can_spawn_center[0]
            goal_xyz[:, 1] += self.can_spawn_center[1]
            goal_xyz[:, 2] = torch.rand((b)) * self.max_goal_height + xyz[:, 2]
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))

    def _get_obs_extra(self, info: Dict):
        # in reality some people hack is_grasped into observations by checking if the gripper can close fully or not
        obs = dict(
            is_grasped=info["is_grasped"],
            tcp_pose=self.agent.tcp_pose.raw_pose,
            goal_pos=self.goal_site.pose.p,
        )
        if "state" in self.obs_mode:
            obs.update(
                obj_pose=self.cans[0].pose.raw_pose,
                tcp_to_obj_pos=self.cans[0].pose.p - self.agent.tcp_pose.p,
                obj_to_goal_pos=self.goal_site.pose.p - self.cans[0].pose.p,
            )
        return obs

    def evaluate(self):
        is_obj_placed = (
            torch.linalg.norm(self.goal_site.pose.p - self.cans[0].pose.p, axis=1)
            <= self.goal_thresh
        )
        is_grasped = self.agent.is_grasping(self.cans[0])
        is_robot_static = self.agent.is_static(0.2)
        return {
            "success": is_obj_placed & is_robot_static,
            "is_obj_placed": is_obj_placed,
            "is_robot_static": is_robot_static,
            "is_grasped": is_grasped,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        tcp_to_obj_dist = torch.linalg.norm(
            self.cans[0].pose.p - self.agent.tcp_pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        reward = reaching_reward

        is_grasped = info["is_grasped"]
        reward += is_grasped

        obj_to_goal_dist = torch.linalg.norm(
            self.goal_site.pose.p - self.cans[0].pose.p, axis=1
        )
        place_reward = 1 - torch.tanh(5 * obj_to_goal_dist)
        reward += place_reward * is_grasped

        qvel = self.agent.robot.get_qvel()
        qvel = qvel[..., :-2]
        static_reward = 1 - torch.tanh(5 * torch.linalg.norm(qvel, axis=1))
        reward += static_reward * info["is_obj_placed"]

        reward[info["success"]] = 5
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 5


PickCanEnv.__doc__ = PICK_CAN_DOC_STRING.format(robot_id="Panda")

