# 长时间停滞审查：v154

> 最新执行为 **v157**（见本文末尾 critic 审查），v154/v155为历史运行，均已保存退出。

2026-09-22。用户指出长时间没有进阶后，保存并停止了 v151，保留 `model_requested_stop_624.pt`。不能将此前反复微调、探针和小样本复测当成课程收尾成功。

## 已确认的问题

1. **控制一致，奖励却不一致。** `_get_rewards()` 原本用 `~_rehearsal_mask` 屏蔽旧技能的滤波速度误差、速度获取/回退、严重欠速、pitch 跟踪及转弯专用项；部分资格又依赖当前全局 Level。同一状态与 command 仅改变 cohort 或全局 Level 就会改变优化目标。rehearsal 中的局部 domain head 仍在学习，因此 frozen base 本身未变并不能阻止组合行为被削弱。
2. **局部残差确实削弱了一个旧技能。** model600 的 2.9 m/s 直行、seed43、64 个 nominal episodes：原策略 speed error=0.168342、stable=0.026129；仅关闭该速度直行的新增局部 domain 输出后，speed error=0.150587、stable=0.382316。两者 survival=1，但两者都没完整通过。该 A/B 支持局部残差有害，不能证明所有停滞都由它造成。
3. **执行流程反复中断。** 多次短运行重启会重新积累 gate 的完整 episode 证据；此前续训命令又均设置 `--stop_at_command_level 27`，即使通过 Level26 也不会连续训练到31。这是执行流程的问题，不应继续用状态播报代替推进。
4. **精细 tracking 奖励被截断。** 此前将 active bank shaping scale提高到2时，没有配套核对总奖励上界，这是执行中的遗漏。当前正奖励项保守上界14.85，原cap却是10。在真实稳态姿态、实际训练配置的counterfactual yaw扫描中，误差0.06（超过0.05 gate）时仍有31.25%状态到达cap，误差0时81.25%状态到达cap。部分状态的精确/不合格tracking得到相同截断奖励，损失了精细学习信号。不能把这一证据扩大成所有失败的唯一原因。

证据：`artifacts/continuation_v153/no_local_straight64_nominal.json`、`artifacts/continuation_v151/model600_straight64_nominal.json`、历史 launch.json 和训练日志。

## 修复与验证

- 增加 opt-in `env.command_reward_contract_by_command=True`：奖励资格由 command 对应的速度区间决定，不再因 active/rehearsal 身份或全局课程等级改变。同一 command 的 bank 增强系数也一致。低速仍不启用高速 pitch 项。
- CPU 测试直接抽取执行实际 `_get_rewards()` 的奖励计算，不是另写一个参考公式：旧逻辑可复现差异；新逻辑在 active/rehearsal、Level26/31之间完全相等。连同 std、bank gate、残差审计、layout 测试，共19项通过。
- 真实16环境、1.5/2.9 m/s、直行/负转弯的1600-step物理评估后，在相同物理状态下切换 cohort/Level：旧逻辑最大差异为1.933624（cohort）、5.093191（全局Level），新逻辑全为0。报告 `artifacts/continuation_v153/reward_contract_physics_audit.json`。这证明奖励一致性修复，不代表技能已通过。
- 恢复已有 acquired-skill regression stop，取消 warn-only。该保护只有达到其资格条件后才会触发，不能声称它能发现所有未认证技能的退化。

## 当前运行

v154：`logs/rsl_rl/rmuc_wheel_leg_course/2026-09-22_19-24-07_command_reward_consistency_to31_v154`。

从 v151/624 恢复 actor、critic、optimizer，先做5轮 critic-only warmup 适应新奖励，再恢复原残差学习；LR、PPO epochs、USD、25-D/6-D、控制反馈、DR进度、gate均不改变。保留固定 v149/271 L2 锚点与实时权重审计。没有采用此前固定偏置候选或 pitch-rate 增益反转。

训练经各级 gate 连续推进，停止点改为 Level32 entry，即须实际通过 Level31；这是中间执行里程碑，全速度、完整DR与3 Nm/实测 damping 的部署目标仍保留。完整命令见 `artifacts/continuation_v154/launch.json`。

## 不应再做的事

- 不因一次小样本改善或一个近阈值均值就改基线、重启或宣布通过。
- 不把权重冻结、有限动作或代码测试通过等同于行为防遗忘。
- 不放宽 gate、删掉失败环境，或跳过 Level23/旧技能 retention 来制造进阶。
- 不承诺修复后必然一小时达到 Level31；以进阶 checkpoint 和独立行为证据验证。

尚待证明：这次目标一致性修复能否恢复旧技能并带来实际课程进阶。学习率偏保守可能影响速度，但尚无将其认定为唯一根因的证据，本次不同时改多个优化器参数。

## 19:39 补充修复：v155

`configs/rmuc_wheel_leg.yaml`的reward上限由10改为20，高于14.85的保守正项上界；负奖励下限和所有行为gate不变。新增奖励排序回归测试，直接执行实际奖励计算，复现旧cap使yaw误差0与0.06同得10，并验证新cap恢复正确排序。相关测试共20项通过。

证据：`artifacts/continuation_v154/reward_clip_actual_config_audit.json`。该实验使用真实物理姿态，但以反事实yaw sensor值扫描奖励函数，不是该命令的独立技能证书。最初未带训练override的诊断保留在`reward_clip_physical_state_audit.json`，实际结论以带bank scale=2、turn yaw start=26的前一报告为准。

v154已STOP保存`model_requested_stop_765.pt`，v155从766恢复，run为`2026-09-22_19-39-50_reward_signal_unclipped_to31_v155`，启动记录`artifacts/continuation_v155/launch.json`。只再修正reward cap，保留一致奖励、固定271锚点、全部学习率/epoch及控制/physics/gate，5轮critic warmup后继续。仍须以实际Level进阶与完整验收证明修复有效。

修复后真实物理姿态复核已完成：`artifacts/continuation_v155/reward_clip20_physics_audit.json`使用同model675、同16环境、同seed及1400steps，原始奖励与cap10对照逐扫描点一致；cap20下所有扫描点high_clip_fraction=0，clipped=raw，yaw误差0到0.10对应奖励严格递减。v155第784轮实时日志同样reward_clip_high_fraction=0、actor参数更新非零、immutable变化0。上述证明学习信号截断已修复，不是Level26/31技能认证。

## 20:20补充：critic 量纲与 value clipping

model900自动验证已完成：2.9直行64回合survival1，但speed error0.200698、stable0，未恢复。2.2双向DR中负向通过本次阈值（yaw0.04912、stable0.48896、survival0.984375、minroot0.20469）；正向仍有一次倾倒和coverage失败，整体未通过。这些结果反驳了“前两项代码修复就已经恢复技能”的判断。

新增critic诊断使用checkpoint的随机动作分布、与训练相同的奖励配置和gamma0.99。对MC回报显式计算未观测尾项误差上界，区分真实termination与timeout，只分析上界≤5的样本。2.9直行32环境轨迹得到28192个有效样本：critic均值-1098.37、估计回报均值-1657.19，偏差+558.82、RMSE587.30。EV虽为0.8136，不能掩盖均值偏差。样本主要来自加速段；obs[6]有clip，不能从饱和obs反推精确实际速度，`critic_error_by_speed.json`已明确区分被截断的信号。

源码与运行配置确认：`use_clipped_value_loss=True`，critic复用了actor的`clip_param=0.1`。这不是value每轮变化的硬限制，但会让已改善超过0.1的样本选择常数clipped-loss分支，失去进一步正确修正的梯度。直接执行安装版本PPO的value-loss分支，预测从-1000朝目标-1500移到-1001时，旧loss梯度为0，普通MSE梯度为998。

独立、同初值、同LR1e-4、同4 minibatches/update、同gradient clipping的100轮固定数据校准：留出8个环境不参与训练，开启value clipping后RMSE439.47、偏差381.93；关闭后RMSE159.11、偏差16.00。数据与报告：`artifacts/continuation_v156/critic_clipping_comparison.json`。这仅是单command诊断，不是多技能策略认证，离线拟合critic没有用于正式训练。

`rsl_rl_ppo_cfg.py`已关闭critic value clipping，actor PPO clipping仍0.1。v155通过STOP保存`model_requested_stop_1152.pt`，v157从1153恢复；使用50轮真实多任务critic-only校准后再恢复actor，全部LR/epoch、physics、gate、L2锚点保持不变。启动记录`artifacts/continuation_v157/launch.json`。同时修复warmup期间跨Level可能提前解冻/结束后恢复过时head的问题；这是防止未来路由错误的修复，尚无证据说明此前发生过该路径。相关测试共24项通过。

实跑校准复查：采集冻结policy1152自身的独立2.9直行随机策略轨迹，采用28192个尾项界≤5样本。原critic RMSE467.04/bias429.45；v157/1175为384.05/339.69，1200为374.97/328.56。两个checkpoint全部actor_state_dict tensors与1152完全相同。报告`artifacts/continuation_v157/live_critic_calibration_check.json`。说明真实多任务校准有改善，但价值偏差仍明显，不能宣称已彻底校准或推论到所有速度。50轮warmup结束后恢复残差训练，不能导入单任务离线critic替代。
