# RMUC 专家混合与残差训练接续审阅

审阅日期：2026-09-21。主交接文件为 `../inclusion.md`，本文补充实现细节与代码核对结果。
本次仅审阅代码、配置、已有日志和 checkpoint，并做 CPU 权重/路由/导出包装器检查；没有启动、停止或修改训练，没有重新做物理仿真认证。
下述运行数据是快照；后续操作应重新检查进程、日志和有效配置。

## 1. 实际运行链路

`scripts/rsl_rl/train.py` → Gym RMUC task → `adapter.py` → `env_cfg.py`/`env.py` → `RMUC_WHEEL_LEG_CFG` → closed-loop USD。

- `Robotics-Wheelbipe-V14-RMUC-Command-v0` 是 RMUC task alias，不使用旧 V14 环境的关节拓扑。
- 主配置是 `configs/rmuc_wheel_leg.yaml`；某次实验实际使用的参数应核对该 run 的 `params/env.yaml`、`params/agent.yaml` 和启动 CLI。
- 当前 stock `OnPolicyRunner + PPO`：旧式 `ActorCritic` 配置经 `_prepare_stock_rsl_cfg()` 转成实际的 actor/critic `MLPModel`。
- Actor 为 `25 → 128 → 64 → 32 → 6`，critic 为 `25 → 256 → 128 → 64 → 1`，ELU。
- v139 有效参数：128 envs、128 steps/env、LR `2e-6`、1 epoch、4 minibatches、gamma `0.99`、lambda `0.95`、clip `0.1`、entropy `0`。
- 物理步长 `0.002 s`，decimation 4，控制周期 `0.008 s`；12 s episode 约 1500 control steps；每 PPO rollout 约 1.024 s/env。

## 2. Observation、专家路由与动作

25-D observation 的零基索引：

| 索引 | 内容 |
|---|---|
| 0:3 | canonical angular velocity × 0.25 |
| 3:6 | projected gravity |
| 6 | `(target_vx - heading_vx) × 2.1` |
| 7 | `target_vx × 0.5` |
| 8 | `target_yaw - actual_yaw_rate` |
| 9:13 | 四个受控腿关节的位置偏移 |
| 13:19 | 六个受控关节速度 × 0.05 |
| 19:25 | 上一步实际执行的 normalized policy action，包含环境反馈 |

command slots 另有 ±2 clip。当前 obs normalization 关闭。路由从 obs[7]/0.5 还原目标速度，从 obs[2]/0.25 + obs[8] 还原 yaw；改变这些缩放会同时改变专家路由。
DR 对整个 observation 加噪声、施加延迟，所以路由使用的 command 还原值也会受影响。重建 yaw 在 command-error clipping 饱和时也不再严格等于真实 command。

actor forward 顺序见 `scripts/rsl_rl/rmuc_command_adapter.py`：

1. Base MLP 与低中速 turn expert 按 command gate 混合；turn expert 的速度混合区间为 1.5–1.9 m/s。
2. 加原有 turn adapter；其 support 限制在转弯且速度大于 2.0 m/s。
3. 累加 stage turn residual bank。16 个 stage module 各有公共 net 和正/负 yaw correction；仅对应符号 correction 生效。
4. 2.9 m/s straight expert 与上述结果混合：速度 2.80–2.88 渐入，2.88–2.92 全开，2.92–3.00 渐出；yaw 0.02–0.06 渐出。
5. 加 domain residual bank：straight、positive/negative small yaw、positive/negative large yaw 五个 head；总修正为 `0.04*tanh(output)`。
6. 如安装 actuator bridge adapter，再加其输出；v139 checkpoint 未安装它。
7. Gaussian distribution 在训练时采样，deterministic inference 使用均值。

注意：domain heads 在平滑过渡区可以同时有非零 gate，并非所有输入下都严格只激活一个 head。stage residual 隔离的是 command support，不保证不同速度段之间完全无影响。

v139 `model_375.pt` 中 actor 共 57 个输出宽度为 6 的 Linear 层：base/turn expert/speed expert/旧 turn adapter 各 1，stage bank 48，domain bank 5；最终动作仍只有一个六维向量。

## 3. 冻结、防遗忘与实际权重证据

`set_rmuc_turn_delta_trainable()` 先冻结整个 stage bank，再只打开当前 level 的正/负 correction；directional 模式下公共 net 保持冻结。
Base、turn expert、speed expert 和旧 turn adapter 保持冻结。Domain bank 参与训练，通过限幅约束其影响。

CPU 比较结果：

- v130 `2026-09-21_15-19-35_level28_strict_to29_v130/model_125.pt` → v139 `2026-09-21_19-15-18_level29_progressive_flat_dr_domain_bank_v139/model_375.pt`，共有的 actor tensors 中仅 `distribution.std_param` 和 `rmuc_turn_delta_adapters.1.positive_net/negative_net` 变化。
- base、turn expert、其他已有 stage heads 逐 tensor 一致。
- v139 内嵌 speed expert 与 v98b `2026-09-19_20-50-50_command_level21_v87_survival_refine_v98b/model_125.pt` 的 base MLP 逐 tensor 一致。
- 新增 domain bank 20 个参数 tensors，speed expert 8 个参数 tensors。
- 完整 v139 actor CPU `strict=True` 恢复成功。

Rehearsal 在 reset 时维持全局 population quota；v139 约 40% envs 执行旧 stage，priority fraction 0.85，priority stage 按权重采样。yaw 方向按各 stage 的累计 episode 配额自适应分配。
旧参数冻结、command routing、真实 on-policy rehearsal、retention gate 四者共同构成防遗忘机制；冻结参数不等于环境变化后行为必然保留。

Teacher distillation 是可选辅助，v139 CLI 未启用。它使用同一 25-D 输入和 deterministic actions，SmoothL1 loss；转弯区分 wheel differential/common mode 与 leg loss；权重和独立 Adam LR 同步 cosine 退火，可选 speed/turn gradient projection。它不是 off-policy PPO replay。

## 4. 网络之外的控制链

`env.py::_compute_policy_action()` 还会添加 bounded speed/pitch/yaw feedback，然后腿动作映射为 position targets，轮动作映射为 effort targets。
四腿通道顺序为 action[0,1,3,4]；轮为 action[2,5]；轮符号 [-1,-1]。
Yaw differential 使用剩余 action headroom，尽量保持 wheel common mode 不变。

训练 `_configure_env_action_credit()` 缓存补偿后的动作供 PhysX 使用一次，但 PPO buffer 保留原始 actor sample/log-prob。不能把补偿动作伪装成 Gaussian 样本。
部署不仅需要完整 actor，还需要一致的 observation/state/filter、环境反馈、action scale/sign 和 actuator contract。

**Active 与 rehearsal 存在控制差异**：pitch feedback、propulsion、部分高速补偿逻辑受 `~_rehearsal_mask` 限制；部分逻辑还依赖全局 command level。固定 Level23 evaluation 会关闭 rehearsal，所以它与 Level29 中的 Level23 rehearsal 不天然等价。
这些差异是后续 A/B 必须控制的变量，不是已经证明的 Level23 根因。

## 5. Gate 与 Level23 排查的准确边界

关键实现：`env.py::_pre_physics_step()`、`_update_rehearsal_gate_stats()`、`_update_rehearsal_survival_gate()`、`_reset_idx()`。

- rolling window 长度 300，但传入 retention 的条件是 `window_ready & steady_state_episode`。
- `steady_state_episode` 使用 env 自有 reset counter，阈值为 `max(command_gate_steps, command_survival_gate_steps)`；v139 实际是 **1200 steps / 9.6 s**。
- 因此 tracking sample=0 不仅可能意味着未活过 300 steps，也可能意味着未到 1200-step steady-state 条件；不能仅凭零样本推出提前在 300 steps 前终止。
- completed rehearsal episode 只要已开始且步数大于零就可计数；survival 要达到 stage horizon 且结束时没有 unsafe reason。即使活过 horizon，之后摔倒仍计失败。
- straight stage 的 tracking 已兼容两个 slots；survival 仍按分配的 turn-sign slot 统计，实际 command yaw 保持 0。
- active/retention tracking、双向 survival、posture、contact、ready/stable、transition+hold 同时满足，再连续确认 125 control steps 后进阶。
- global env steps 每次增加 num_envs，与单 env 的 control steps 不同。

日志 iteration 490/1500 快照：仍 Level29，retention ready `0.8333`；Level23 两个 tracking count 和 survival 都是 0，而 completed episode counts 约 327/326。Level8 positive yaw error 约 0.098；Level28 negative yaw error约 0.069、survival约 0.934。Level28 positive complete episodes 约42，尚少于64。终端这些数是 rollout 聚合日志，不是逐时刻原始计数。

下一轮诊断须记录 stage-local：reset age、termination reason、rolling-ready 与 steady-ready 分开计数、vx/pitch/root-z/effort/slip 轨迹、raw/expert/domain/feedback action 分量。
A/B 必须同时区分 nominal/DR、active/rehearsal feedback、确定性/训练 action noise；只比较 evaluator 与 trainer 的汇总均值不足以定位。

## 6. DR 与 sim2real

DR strength 随 command level 和 transition 增长；Level29 transition 完成后为 2/19≈0.1053。Reset 时重采样 root mass/inertia、robot friction、wheel damping/gain、obs noise/delay。
它依据全局训练 level，所以 Level29 的低速/直行 rehearsal 也承受 Level29 DR。
`play=True` 且 `apply_in_play=False` 时关闭物理/观测 DR，但 checkpoint 的 domain residual 仍参与 inference。

正式训练 USD 为 `source/agent_world/agent_world/assets/usd_files/rmuc_wheel_leg/rmuc_robot_closed_loop.usd`；`RMUC_WHEEL_LEG_CFG` 明确加载它。保持闭环拓扑、frame/reset、25-D/6-D contract。
当前 explicit wheel actuator center 是 5.5 Nm/0.1；USD authored 3 Nm 不会自动把 explicit actuator 限制到3 Nm。
最终 bridge 应同步迁移 torque scale/effort limit 至3 Nm，damping center 来自实测。

## 7. 接续前已发现的工程缺口

### 7.1 导出不能直接交付：已通过 CPU 对照确认

`play.py` 当前调用 installed runner 的 `export_policy_to_jit/onnx()`。
installed `rsl_rl.models.mlp_model` 的 export wrappers 只复制 `obs_normalizer` 与 `mlp`，没有完整专家/残差 forward。
对 v139 model_375 重建 actor 后，用7组合成25-D输入比较：`actor.as_onnx(False)(x)` 与 base MLP 完全相等，却与完整 actor 不同，最大 action 差约0.01850。
这是导出包装器的确定性缺陷证据，不是重新完成 ONNX Runtime 或仿真验收。
交付前需实现完整 composition 的可导出 wrapper，并做 PyTorch/ONNX parity，再验证外部控制链；不能只检查导出文件存在。

### 7.2 后续直行学习容量

CPU 路由检查：Level29/30/38/42 分别选中对应 turn head；Level39/40/41 返回 none。
这些直行阶段仍可更新 domain bank 和按配置允许的 std，但没有专用 straight acquisition head。是否足够要用行为证据判断；不能假设3.0/3.1/3.2 m/s自然会学会。
Level46 是 terminal duplicate，没有 turn head 属于预期的认证语义。

### 7.3 Bridge 推进检查尚不等于交接目标

`_update_wheel_damping_anneal_progress()` 当前检查 active tracking/posture/contact/survival，并在转弯时检查双向 survival；没有调用 retention gate。
因此“只有 active 与 retention 均稳定才推进 bridge”是接续实现要求，目前不能宣称已完整实现。当前 v139 bridge 禁用。

### 7.4 独立 evaluator 不等于最终验收矩阵

现有 evaluator 支持固定 level、双向 temporal tracking、tilt termination rate、mean height 等；没有与训练等价的所有 unsafe reasons 的 full-episode survival gate，也没有始终 root-z>0.20 的最小值检查。
当前物理 root-low termination 配置为0.08，command safe/gate要求0.20；最终“始终>0.20”仍需单独严格验证。
Evaluator 没有直接提供当前训练DR强度的CLI override；`play=True` 默认关闭DR。完整 nominal/DR multi-seed 验收需要补充能力。
当前47个 command stages 的 vx 都为正；倒车、零速保持、起停/减速和连续键盘命令不能由这些离散正向认证直接推出。

## 8. 续训/交付顺序

1. 先补齐 Level23 stage-local 证据，按上面变量做小范围 A/B；根因定位前不修改 gate 来跳过它。
2. 修复证实的问题，再验证 Level23 与 Level8/28 弱方向 retention。已有名义技能应保持冻结，新增能力通过受控 residual 获取。
3. 使用明确 checkpoint lineage 逐级完成 Level30–38；每个 entry checkpoint 做独立验证。Level39–41 前检查直行学习容量，随后完成42–46。
4. 名义技能完成后，补齐 bridge retention 推进约束，执行3 Nm和实测 damping 的渐进迁移。
5. 完成 nominal/DR、多seed、双yaw方向、零速起步/运动状态、正反向与连续命令验收；修复完整 actor 导出后再更新 controller。

续训语义：默认 `--checkpoint` 是 finetune，iteration归零、optimizer新建；`--load_critic_for_finetune` 才加载critic。`--resume_training` 恢复optimizer/iteration，但这不等于恢复物理状态、gate buffers和DR episode状态。
安装对应 expert/residual modules 必须先于加载权重。当前 finetune loader 会报告跳过/零填充 tensors，应审阅日志，不能把部分兼容加载等同于完整恢复。
`--allow_curriculum_override` 绕过的是启动前的 sidecar provenance检查，不会取消环境内的进阶gate。
v139 `--highspeed_regression_guard_warn_only` 只告警，不应指望它自动停止退化训练。
`control.py` 默认仍为Level18旧checkpoint；后续应同时检查 checkpoint与command level等控制配置，而非只换文件路径。
