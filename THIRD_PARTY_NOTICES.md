# Third-Party Notices（第三方代码声明）

本仓库部分代码派生/移植自下列第三方开源项目。为履行各项目的许可要求，此处集中声明其出处、许可协议与版权信息。各源文件的文件头也有对应的局部声明。

> 说明：本文件仅如实记录上游项目的许可状态，不构成法律意见。如涉及商用，请自行（或咨询律师）核验并取得必要授权，尤其是带 **非商业（NonCommercial）** 限制的协议。

## 1. HIMLoco

- **仓库**：<https://github.com/InternRobotics/HIMLoco>
- **许可协议**：CC BY-NC-SA 4.0（署名-非商业性使用-相同方式共享）
- **版权**：Copyright (c) 2024 Junfeng Long, Zirui Wang
- **许可全文**：<https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode>
- **许可要点**：
  - **署名**：须保留原始版权声明与协议链接；
  - **非商业**：**禁止用于商业用途**；
  - **相同方式共享**：衍生作品须以相同协议（BY-NC-SA）授权，**不得**以 MIT 等方式再授权。

  受影响的文件：
  - `source/agent_rl/agent_rl/rsl_rl/algorithms/ppo_him.py`
  - `source/agent_rl/agent_rl/rsl_rl/modules/actor_critic_him.py`
  - `source/agent_rl/agent_rl/rsl_rl/modules/him_estimator.py`
  - `source/agent_rl/agent_rl/rsl_rl/runners/on_policy_runner_him.py`

> ⚠️ 该协议与仓库根目录的 MIT 声明**不兼容**。若需商业发布，应移除/替换上述 HIM 相关文件，或另行取得上游作者授权。

## 2. DreamWaQ

- **仓库**：<https://github.com/Manaro-Alpha/DreamWaQ>
- **许可协议**：仓库**未在顶层提供 LICENSE**，DreamWaQ 算法代码本身未声明明确许可协议。其内置的 `legged_gym` 与 `rsl_rl-1.0.2` 为 **BSD-3-Clause**（Copyright (c) 2021, ETH Zurich / Nikita Rudin；NVIDIA CORPORATION & AFFILIATES）。
- **版权**：无明确声明（DreamWaQ 算法部分）。
- **许可要点**：算法代码未附带 LICENSE，此处仅作学术引用与致谢；使用前建议与上游作者确认授权。

  受影响的文件：
  - `source/agent_rl/agent_rl/rsl_rl/algorithms/ppo_dreamwaq.py`
  - `source/agent_rl/agent_rl/rsl_rl/modules/actor_critic_dreamwaq.py`
  - `source/agent_rl/agent_rl/rsl_rl/modules/nn/vqvae.py`
  - `source/agent_rl/agent_rl/rsl_rl/runners/on_policy_runner_dreamwaq.py`

## 3. ddt_rl_isaacgym（NP3O）

- **仓库**：<https://github.com/DDTRobot/ddt_rl_isaacgym>
- **许可协议**：仓库**未提供 LICENSE 文件**，未声明明确许可协议。其 README 声明使用了 [legged_gym](https://github.com/leggedrobotics/legged_gym) 与 [LocomotionWithNP3O](https://github.com/zeonsunlightyu/LocomotionWithNP3O) 的代码。
- **版权**：无明确声明。
- **许可要点**：未附带 LICENSE，此处作学术引用与致谢；NP3O 思路参考 legged_gym 与 LocomotionWithNP3O。

  受影响的文件：
  - `source/agent_rl/agent_rl/rsl_rl/algorithms/np3o.py`
  - `source/agent_rl/agent_rl/rsl_rl/modules/actor_critic_balowtwins.py`
  - `source/agent_rl/agent_rl/rsl_rl/runners/on_constraint_policy_runner.py`
  - `source/agent_rl/agent_rl/rsl_rl/storage/rollout_storage.py`

## 4. rsl_rl / legged_gym（基座框架）

- **仓库**：<https://github.com/leggedrobotics/rsl_rl>、<https://github.com/leggedrobotics/legged_gym>
- **许可协议**：BSD-3-Clause
- **版权**：Copyright (c) 2021, ETH Zurich / Nikita Rudin；Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES

  上述所有受影响文件均整合于 `rsl_rl` 框架内，遵循该框架的 BSD-3-Clause 许可（要求保留版权声明与免责声明）。
