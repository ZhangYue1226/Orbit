# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
# 使用RMPFlow，计算并返回关节的下一个动作（即：下一个位置，下一个速度），rmpflow算法中需要输入机器人碰撞路径模型描述：collision_file。

import torch
from dataclasses import MISSING
from typing import Tuple

import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.simulation_context import SimulationContext # 
from omni.isaac.motion_generation import ArticulationMotionPolicy
from omni.isaac.motion_generation.lula import RmpFlow

from omni.isaac.orbit.utils import configclass


@configclass
class RmpFlowControllerCfg:
    """Configuration for RMP-Flow controller (provided through LULA library)."""

    config_file: str = MISSING
    """Path to the configuration file for the controller."""
    urdf_file: str = MISSING
    """Path to the URDF model of the robot."""
    collision_file: str = MISSING
    """Path to collision model description of the robot."""
    """机器人碰撞路径模型描述。"""
    frame_name: str = MISSING
    """Name of the robot frame for task space (must be present in the URDF)."""
    evaluations_per_frame: int = MISSING
    """Number of substeps during Euler integration inside LULA world model."""
    ignore_robot_state_updates: bool = False
    """If true, then state of the world model inside controller is rolled out. (default: False)."""
    """MPiNets:输入：a robot configuration(qt)（各关节初始位姿）, a segmented calibrated point cloud (zt)(相当于路径描述？)."""

class RmpFlowController:
    """Wraps around RMP-Flow from IsaacSim for batched environments."""
    """ 包装来自IsaacSim的rm - flow，用于批处理环境。"""

    def __init__(self, cfg: RmpFlowControllerCfg, prim_paths_expr: str, device: str):
        """Initialize the controller. 初始化控制器

        Args:
            cfg (RmpFlowControllerCfg): The configuration for the controller.
            prim_paths_expr (str): The expression to find the articulation prim paths. 寻找机械手路径的语句
            device (str): The device to use for computation. 用于计算的设备(cuda/gpu)

        Raises:
            NotImplementedError: When the robot name is not supported.
        """
        # store input 存储输入
        self.cfg = cfg
        self._device = device

        print(f"[INFO]: Loading controller URDF from: {self.cfg.urdf_file}")
        # obtain the simulation time
        physics_dt = SimulationContext.instance().get_physics_dt()
        # find all prims
        self._prim_paths = prim_utils.find_matching_prim_paths(prim_paths_expr)
        self.num_robots = len(self._prim_paths)
        # create all franka robots references and their controllers 创建机器人的参考和其控制器
        self.articulation_policies = list()
        for prim_path in self._prim_paths:
            # add robot reference 添加机器人参考
            robot = Articulation(prim_path)
            robot.initialize()
            # add controller 添加控制器（包括了collision碰撞路径模型描述）
            rmpflow = RmpFlow(
                rmpflow_config_path=self.cfg.config_file,
                urdf_path=self.cfg.urdf_file,
                robot_description_path=self.cfg.collision_file,
                end_effector_frame_name=self.cfg.frame_name,
                evaluations_per_frame=self.cfg.evaluations_per_frame,
                ignore_robot_state_updates=self.cfg.ignore_robot_state_updates,
            )
            # wrap rmpflow to connect to the Franka robot articulation 包装rmpflow，以便于和Franka机器人关节连接
            articulation_policy = ArticulationMotionPolicy(robot, rmpflow, physics_dt)
            self.articulation_policies.append(articulation_policy)
        # get number of active joints 获取活动关节数量
        self.active_dof_names = self.articulation_policies[0].get_motion_policy().get_active_joints()
        self.num_dof = len(self.active_dof_names)
        # create buffers
        # -- for storing command 存储命令
        self._command = torch.zeros(self.num_robots, self.num_actions, device=self._device) # torch.zeros():输出指定格式的0张量
        # -- for policy output  输出
        self.dof_pos_target = torch.zeros((self.num_robots, self.num_dof), device=self._device)
        self.dof_vel_target = torch.zeros((self.num_robots, self.num_dof), device=self._device)

    """
    Properties.
    """

    @property
    def num_actions(self) -> int:
        """Dimension of the action space of controller."""
        return 7

    """
    Operations.
    """

    def reset_idx(self, robot_ids: torch.Tensor = None):
        """Reset the internals."""
        pass

    def set_command(self, command: torch.Tensor):
        """Set target end-effector pose command."""
        """设置EE目标姿态的命令"""
        # store command 保存命令
        self._command[:] = command

    def compute(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs inference with the controller. 使用控制器执行推理。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The target joint positions and velocity commands.
            返回：目标关节位置和速度命令
        """
        # convert command to numpy
        command = self._command.cpu().numpy()
        # compute control actions 计算控制动作
        for i, policy in enumerate(self.articulation_policies):
            # enable type-hinting
            policy: ArticulationMotionPolicy
            # set rmpflow target to be the current position of the target cube.将rmpflow目标设置为目标数据集的当前位置。
            policy.get_motion_policy().set_end_effector_target(
                target_position=command[i, 0:3], target_orientation=command[i, 3:7]
            )
            # command 0:3 是目标位置target position，3:7是目标方向target orientation
            # apply action on the robot 将动作应用到机器人上（获得关节的下一个动作）
            action = policy.get_next_articulation_action()  
           
            # ！！！policy.get_next_articulation_action()这个函数在哪里能看？想看是如何得到joint_positions和joint_velocities的！！！
            # ！！！能否使用这个函数，将mpinets的输出（qt点或者qt+1）转化成orbit中的关节的下一个位置，关节的下一个速度，这样可以直接将这些输出作为orbit接下来可视化或是其他模块的输入了！！！
            
            # copy actions into buffer
            # TODO: Make this more efficient?
            for dof_index in range(self.num_dof):
                self.dof_pos_target[i, dof_index] = action.joint_positions[dof_index] #关节的下一个位置
                self.dof_vel_target[i, dof_index] = action.joint_velocities[dof_index] #关节的下一个速度

        return self.dof_pos_target, self.dof_vel_target
        # 返回关节的下一个动作（即：下一个位置，下一个速度）
