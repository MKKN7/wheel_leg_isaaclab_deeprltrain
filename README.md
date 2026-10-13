# RMUC Wheel-Leg Deep RL Training

RMUC（RoboMaster University Championship）轮腿步兵强化学习训练与部署仓库。基于 **Isaac Sim + Isaac Lab + RSL-RL** 的端到端轮腿机器人 RL 训练代码，包含 MuJoCo sim2sim 验证与实机底层部署核心。

## 项目概述

面向 RMUC 闭链并联轮腿机器人：轮端 RM3508（C620 电调，268:17 减速箱），腿部 DM-J8009P（9:1），带气弹簧辅助。策略为 25 维观测、6 维动作（4 腿位置 + 2 轮力矩）的端到端控制器。

### 执行器契约（当前）

| 项 | 值 |
| --- | --- |
| 轮端峰值 / 持续 | 5.5 N·m 瞬态峰值 / 3 N·m 持续热参考 |
| 轮力矩环 | 直接力矩（非速度环），动作 ×5.5 N·m，电磁限幅 ±5.5 N·m |
| 被动项 | 机械粘性 b=0.002 N·m·s/rad + 库仑 c=0.001 N·m，均在电磁限幅之后 |
| 反馈阻尼 D | 未辨识（当前 0），部署端按训练值全额补偿 |
| 腿 PD | 60/2/40 N·m（2 ms 更新） |
| 气弹簧 | 每腿 active_link 输出侧 −10.775 N·m 恒等效，机身等大反作用 |
| 时序 | 物理 1 ms / 腿 PD 2 ms / 策略 8 ms |

## 目录结构

```
wheel_leg_isaaclab_deeprltrain/
├── configs/                         # RMUC 训练与硬件目标 YAML
│   ├── rmuc_wheel_leg.yaml          # 主训练配置
│   └── rmuc_hardware_target.yaml    # 实机硬件目标（schema-3）
├── source/
│   ├── agent_world/                 # 资产、执行器、地形
│   │   └── agent_world/
│   │       ├── actuators/           # M3508 / 差速 / 学习速度执行器
│   │       ├── assets/              # rmuc_wheel_leg.py + USD 模型
│   │       └── terrains/            # 高度场地形
│   ├── agent_tasks/                 # 环境与任务
│   │   └── agent_tasks/
│   │       ├── direct/rmuc_wheel_leg/   # RMUC 环境（env / env_cfg / 被动项 / 气弹簧 / recovery）
│   │       └── manager/             # MDP 观测 / 奖励 / 事件 / 课程
│   └── agent_rl/                    # RL 算法与 runner（RSL-RL）
├── scripts/
│   ├── rsl_rl/                      # train.py / play.py 入口
│   ├── utils/                       # 可视化工具
│   └── rmuc_*.py                    # RMUC 审计 / 探针 / 评估脚本
├── sim2real_RL/
│   ├── deploy/                      # 实机底层 C 执行器核心（rmuc_low_level.c/h）
│   └── sim2sim/                     # MuJoCo 验证（接口 / 拓扑 / 腿 PD / 气弹簧）
├── tests/                           # test_rmuc_* 单元测试
└── docs/                            # RMUC 契约与迁移进度文档
```

## 快速开始

```bash
# 可编辑安装（依赖 Isaac Lab 2.x + Isaac Sim）
python -m pip install -e source/agent_world
python -m pip install -e source/agent_tasks
python -m pip install -e source/agent_rl

# 环境可视化
python scripts/view_robot.py --task=Robotics-RMUC-WheelLeg-Command-v0 --num_envs=1 --device=cpu

# 训练
python scripts/rsl_rl/train.py --task=Robotics-RMUC-WheelLeg-Command-v0 --num_envs=128 --max_iterations=2000 --device=cuda:0 --headless

# Play
python scripts/rsl_rl/play.py --task=Robotics-RMUC-WheelLeg-Command-v0 --num_envs=1 --checkpoint=<model.pt> --keyboard
```

> 需在仓库根目录运行（`env.py` 内部依赖 `scripts.utils` 路径）。

## Sim2Sim（MuJoCo）

`sim2real_RL/sim2sim/` 提供 USD → MuJoCo 的接口 / 拓扑 / 腿 PD / 气弹簧等价验证：

- 默认模型 `rmuc_peak55.xml`（轮 5.5 N·m 峰值 / 腿 40 N·m，gear=1）。
- 39 刚体质量 / COM / 惯量 / 关节轴 / 四闭链与 USD 数值对照。
- 接口 parity：25D 观测 + authority 动作 consumer，ONNX↔TorchScript 误差 ~1e-7。
- `run_mujoco_probe.py` 运行闭环探针，需显式指定 policy、damping 与气弹簧参数。

## Sim2Real（部署）

`sim2real_RL/deploy/rmuc_low_level.c` 是可移植的 25D/6D 执行器边界（不跑 ONNX、不读 CAN）：

- 腿：`clip(60·(q_t−q) − 2·qd, ±40 N·m)`，2 ms 节拍。
- 轮：`clip(5.5·sign·clip(action, ±1), ±5.5 N·m)`，1 ms 节拍。
- 3 N·m 仅为持续热参考；10.775 N·m 气弹簧与轮 b/c 损耗为 plant 项。

主机侧自检：

```sh
gcc -std=c11 -Wall -Wextra -Werror -pedantic \
  sim2real_RL/deploy/rmuc_low_level.c tests/test_rmuc_low_level.c \
  -lm -o /tmp/rmuc_low_level_test && /tmp/rmuc_low_level_test
```

## 作者

MKKN7
