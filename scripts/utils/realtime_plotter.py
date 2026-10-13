# =============================================================================
# Copyright (c) 2026 SCUTRobotLab
# SPDX-License-Identifier: MIT
#
# Part of the wheeled-legged_RL project.
# See LICENSE for full license terms.
#
# Authors:
#     Zhang Zhirui <2231625449@qq.com>
#     Cui Yu       <ctty694@gmail.com>
# =============================================================================

"""
实时数据可视化工具（性能优化版）
用于在play模式下查看跳跃轨迹、轮电机功率、关节力矩等数据
"""

import matplotlib
matplotlib.use('TkAgg')  # 使用轻量级backend

import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import torch


class RealtimePlotter:
    """实时数据可视化器（性能优化版）"""
    
    def __init__(self, max_points=200, num_leg_joints=6, update_interval=5):
        """
        Args:
            max_points: 显示的最大数据点数（减少以提高性能）
            num_leg_joints: 腿部关节数量
            update_interval: 更新间隔（步数），越大性能越好
        """
        plt.ion()  # 交互模式
        
        # 创建子图：4行
        self.fig, self.axes = plt.subplots(4, 1, figsize=(10, 10))
        self.fig.suptitle('Real-time Jump Data Visualization', fontsize=12, fontweight='bold')
        
        self.max_points = max_points
        self.num_leg_joints = num_leg_joints
        self.update_interval = update_interval
        self.update_counter = 0
        
        # 数据缓冲区
        self.time_data = deque(maxlen=max_points)
        self.target_height = deque(maxlen=max_points)
        self.actual_height = deque(maxlen=max_points)
        self.jump_phase_sin = deque(maxlen=max_points)
        self.wheel_power_left = deque(maxlen=max_points)
        self.wheel_power_right = deque(maxlen=max_points)
        self.spring_force_left = deque(maxlen=max_points)
        self.spring_force_right = deque(maxlen=max_points)
        self.leg_torques = [deque(maxlen=max_points) for _ in range(num_leg_joints)]
        
        # === 第1行：高度曲线 ===
        ax0 = self.axes[0]
        self.line_target = ax0.plot([], [], 'r-', linewidth=1.5, label='Target', alpha=0.8)[0]
        self.line_actual = ax0.plot([], [], 'b-', linewidth=1.5, label='Actual')[0]
        
        # 创建第二个y轴用于jump_phase_sin
        self.ax0_twin = ax0.twinx()
        self.line_phase = self.ax0_twin.plot([], [], 'g--', linewidth=1, label='Phase', alpha=0.6)[0]
        
        ax0.set_ylabel('Height (m)', fontsize=9)
        ax0.set_title('Jump Trajectory', fontsize=10, fontweight='bold')
        ax0.grid(True, alpha=0.3, linewidth=0.5)
        ax0.legend(loc='upper left', fontsize=8)
        ax0.set_ylim(0, 1.0)  # 固定y轴范围
        
        self.ax0_twin.set_ylabel('Phase', fontsize=9, color='g')
        self.ax0_twin.tick_params(axis='y', labelcolor='g')
        self.ax0_twin.legend(loc='upper right', fontsize=8)
        self.ax0_twin.set_ylim(-0.1, 1.1)
        
        # === 第2行：轮电机功率 ===
        ax1 = self.axes[1]
        self.line_wheel_left = ax1.plot([], [], 'g-', linewidth=1.5, label='Left')[0]
        self.line_wheel_right = ax1.plot([], [], 'm-', linewidth=1.5, label='Right')[0]
        
        ax1.set_ylabel('Power (W)', fontsize=9)
        ax1.set_title('Wheel Motor Power', fontsize=10, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=8)
        ax1.grid(True, alpha=0.3, linewidth=0.5)
        ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
        ax1.set_ylim(-500, 500)  # P = torque * joint_velocity, signed mechanical power
        
        # === 第3行：腿部关节力矩 ===
        ax2 = self.axes[2]
        colors = plt.cm.tab10(np.linspace(0, 1, num_leg_joints))
        self.leg_torque_lines = []
        joint_names = ['L_Front1', 'L_Rear1', 'R_Front1', 'R_Rear1']  # 指定主动关节
        
        for i in range(num_leg_joints):
            line = ax2.plot([], [], color=colors[i], linewidth=1, 
                          label=joint_names[i] if i < len(joint_names) else f'J{i}',
                          alpha=0.7)[0]
            self.leg_torque_lines.append(line)
        
        ax2.set_xlabel('Time (steps)', fontsize=9)
        ax2.set_ylabel('Torque (Nm)', fontsize=9)
        ax2.set_title('Leg Joint Torques', fontsize=10, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=7, ncol=3)
        ax2.grid(True, alpha=0.3, linewidth=0.5)
        ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
        ax2.set_ylim(-80, 80)  # 固定y轴范围
        
        # === 第4行：弹簧力 ===
        ax3 = self.axes[3]
        self.line_spring_left = ax3.plot([], [], 'c-', linewidth=1.5, label='Spring Left')[0]
        self.line_spring_right = ax3.plot([], [], 'y-', linewidth=1.5, label='Spring Right')[0]
        
        ax3.set_ylabel('Force (N)', fontsize=9)
        ax3.set_title('Spring Forces', fontsize=10, fontweight='bold')
        ax3.legend(loc='upper right', fontsize=8)
        ax3.grid(True, alpha=0.3, linewidth=0.5)
        ax3.set_ylim(0, 500)  # 弹簧力通常为正且较大 (e.g., 240N)
        
        plt.tight_layout()
        
        # 初始绘制一次
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        # 保存背景用于blit（性能优化）
        self.backgrounds = [
            self.fig.canvas.copy_from_bbox(ax.bbox) for ax in self.axes
        ]
        self.background_twin = self.fig.canvas.copy_from_bbox(self.ax0_twin.bbox)
        
        self.step = 0
        
    def update(self, target_h, actual_h, phase_sin, power_left, power_right, leg_torques, 
               spring_left=0.0, spring_right=0.0):
        """
        更新图表数据（性能优化版）
        """
        self.step += 1
        
        # 转换为Python类型
        if isinstance(target_h, torch.Tensor):
            target_h = target_h.item()
        if isinstance(actual_h, torch.Tensor):
            actual_h = actual_h.item()
        if isinstance(phase_sin, torch.Tensor):
            phase_sin = phase_sin.item()
        if isinstance(power_left, torch.Tensor):
            power_left = power_left.item()
        if isinstance(power_right, torch.Tensor):
            power_right = power_right.item()
        
        # 添加数据
        self.time_data.append(self.step)
        self.target_height.append(target_h)
        self.actual_height.append(actual_h)
        self.jump_phase_sin.append(phase_sin)
        self.wheel_power_left.append(power_left)
        self.wheel_power_right.append(power_right)
        self.spring_force_left.append(spring_left)
        self.spring_force_right.append(spring_right)
        
        # 添加腿部关节力矩
        for i in range(self.num_leg_joints):
            if i < len(leg_torques):
                torque_val = leg_torques[i]
                if isinstance(torque_val, torch.Tensor):
                    torque_val = torque_val.item()
                self.leg_torques[i].append(torque_val)
            else:
                self.leg_torques[i].append(0.0)
        
        # 性能优化：只在一定间隔后更新显示
        self.update_counter += 1
        if self.update_counter < self.update_interval:
            return
        
        self.update_counter = 0
        
        # 更新曲线数据
        self.line_target.set_data(self.time_data, self.target_height)
        self.line_actual.set_data(self.time_data, self.actual_height)
        self.line_phase.set_data(self.time_data, self.jump_phase_sin)
        self.line_wheel_left.set_data(self.time_data, self.wheel_power_left)
        self.line_wheel_right.set_data(self.time_data, self.wheel_power_right)
        self.line_spring_left.set_data(self.time_data, self.spring_force_left)
        self.line_spring_right.set_data(self.time_data, self.spring_force_right)
        
        for i in range(self.num_leg_joints):
            self.leg_torque_lines[i].set_data(self.time_data, self.leg_torques[i])
        
        # 只更新x轴范围（y轴已固定）
        if len(self.time_data) > 1:
            x_min = min(self.time_data)
            x_max = max(self.time_data)
            for ax in [self.axes[0], self.axes[1], self.axes[2], self.axes[3]]:
                ax.set_xlim(x_min, x_max)
            self.ax0_twin.set_xlim(x_min, x_max)
        
        # 使用blit技术快速重绘（性能优化）
        try:
            # 重绘每个axes
            for i, ax in enumerate(self.axes):
                self.fig.canvas.restore_region(self.backgrounds[i])
                ax.draw_artist(ax.get_lines()[0])
                if len(ax.get_lines()) > 1:
                    ax.draw_artist(ax.get_lines()[1])
                self.fig.canvas.blit(ax.bbox)
            
            # 重绘twin axis
            self.fig.canvas.restore_region(self.background_twin)
            self.ax0_twin.draw_artist(self.line_phase)
            self.fig.canvas.blit(self.ax0_twin.bbox)
            
            self.fig.canvas.flush_events()
        except:
            # 如果blit失败，使用标准绘制
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
    
    def close(self):
        """关闭绘图窗口"""
        plt.ioff()
        plt.close(self.fig)
    
    def save_figure(self, filename='jump_visualization.png'):
        """保存当前图表"""
        self.fig.savefig(filename, dpi=120, bbox_inches='tight')
        print(f"[INFO] Figure saved to: {filename}")
