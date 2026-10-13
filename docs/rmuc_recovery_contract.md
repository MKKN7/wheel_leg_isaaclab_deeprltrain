# 独立平地复位策略契约（原型）

用户确认平地倒地自主复位为近期交付目标。独立任务`Robotics-RMUC-WheelLeg-Recovery-v0`与未来`recovery.onnx`不覆盖行走checkpoint。结构保持同一closed-loop USD，25D观测、6D动作及腿部60/2关节输出侧PD律不改。

## 输入与动作

观测沿用当前训练环境`_get_observations`及`tracking_error_with_target`语义。复位阶段command为零，但状态误差槽不一定为零：slot6仍含速度误差、slot8仍含yaw速度误差。历史动作必须是复位策略实际执行的归一化动作，不能直接使用行走历史或关节Nm。

动作顺序为右active、右hip、右wheel、左active、左hip、左wheel。复位腿位置与行走尺度不同：

```
a = clip(raw_action, -1, 1)
lower = joint_lower + 0.02 rad
upper = joint_upper - 0.02 rad
q_des = (lower+upper)/2 + (upper-lower)/2 * a_leg
```

目标位置交给原IdealPD，速度目标和前馈为零，关节力矩限制40Nm。轮端当前使用`-3*a_wheel`作为目标力矩，显式damping0.0225，最终限幅3Nm。左右两侧`active_link_pitch`各施加输出侧`-10.775 N·m`等效辅助力矩，机身施加两侧合力的等大反作用力矩；该被动项不进入电机PD或策略action。旧10.775N径向力仅保留为历史诊断模型。默认DR0只是原型起点，所有这些能力仍需最高DR和跨引擎验收。

这一映射不能交给行走ONNX的consumer。后续部署必须为每个ONNX附带独立动作契约，由同一PD与执行器底层消费最终关节目标/轮端Nm。

## 课程、终止与成功

当前四个固定90度倒地姿态仅用于环境/学习接线。还须扩展关节姿态、倒地方向扰动、跌倒动能及静止/动态倒地分布，初始不穿地、闭链不破坏。不通过修改拓扑或调用reset来完成起身。

原型允许机体倾斜和合理触地，仍保留非有限、关节限位、越界和过大速度终止；不修改行走环境的保护。原型起稳检测：倾角<12度、root高度>0.28m、平移速度<0.15m/s、角速度<0.3rad/s、两轮各>2N，持续0.5s。此检测只表示满足本任务候选交接范围，**不是行走接管证书**。

最终必须测试：从各倒地分布起身→持续起稳→映射关节目标和历史动作→切换行走策略→执行操作手命令，过程中无跳变失稳。交接不盲目把归一化动作当同单位：腿需根据目标角反算目标策略动作，轮需根据各自尺度/阻尼契约处理，并在训练/验证中覆盖。

## 当前证据

第260条：recovery_v1已在v277完成3次实际PPO更新及完整JIT/ONNX导出一致性验证，真实params和export contract均记录奖励profile/weights；只是训练链路验证，没有自主起身成功结论。正在运行的v274仍是legacy_v0奖励的独立探索对照，不因源码新增新方案而自动切换。

第259条用户要求独立自复位reward：新增`recovery_rewards.py`，`env.recovery_reward_profile=recovery_v1`显式启用，可配置权重（upright3、upright_height2、upright_progress2、established5、action_rate-0.1、leg_effort-0.05、time_cost-0.05）。height项乘max(upright,0)，水平/倒置没有高度奖励；progress在reset首帧置0，避免伪进展。起稳门槛保持，不使用行走速度追踪reward。4项逻辑测试与v276四env/128步原生执行通过，尚无正式学习效果证据。legacy_v0保留当前v274实验条件，评估从checkpoint参数恢复reward版本，不隐式切换。未来越障/跳跃可用独立模块，不在本轮修改其环境。

第257条：首回合实际控制步数按Isaac `max_episode_length-1` 为999（7.992s），导出记录first_episode_control_steps。MuJoCo初始t0不累计ready，在每个已执行控制间隔末更新计数；terminal帧只记录不生成未执行动作。trace区分previous_executed_action与next_action，避免历史槽/动作时序歧义。四姿态已重跑并检查终端时间与历史一致，仍未起身。

第256条勘误：累计reward和episode长度受初始化随机回合年龄、每步正奖励影响，不能推导已经学会避免危险终止。原始奖励在零effort/动作变化的倒置0.07m姿态仍有0.35/step，999步约349.65；这是源码反事实，不是实际return拆解。model100四方向未起身是唯一对应行为结论。新独立PPO配置经v273三更新原生验证，探索std0.15/lr3e-4/epochs4确实生效，但训练效果仍待预算对照。

第254条：v269 `model_100.pt` 的完整四姿态评估仍全部未起身，最长 ready 为0，最终高度约0.07m、倾角约180度；TensorBoard快照显示平均回合长度56→999、reward约72→442。曲线提升不能代替起身成功，继续训练前需检查动作饱和、腿目标范围、轮腿分工和奖励塑形。

第253条：导出契约现包含recovery_criteria（连续保持、倾角、高度、平移/角速度和双轮净法向力阈值）、termination_limits（14关节源USD上下限及root/速度/越界）、episode_length_s。MuJoCo入口按契约读取，而非另写一套固定数字；轮接触使用各接触法向力转世界系后的合向量模长。接触求解器/速度约束和末端采样时序尚需原生对应验证，不能因此标记整个sim2sim验收器等价。

独立导出入口为`scripts/export_rmuc_recovery.py --checkpoint <recovery-run>/model_N.pt --output <directory>`。要求run的params明确recovery_prototype；拒绝含行走残差的checkpoint。导出仍是原始归一化动作，物理目标映射及PD在controller执行，必须读取同目录action_contract.json。契约记录policy_kind、USD/params/checkpoint hash、四腿目标范围、25D观测模式、PD/轮力矩参数与周期，明确不能使用行走consumer。v268短训练模型仅已验证导出及真实倒地观测的数值一致性，不能作为已掌握复位的部署策略。

- v267原生4env/128步：倒地不立即终止，观测和奖励有限，PD60/2与轮端3Nm读回正确。
- v268为独立新网络3更新PPO接线验证；状态以`artifacts/continuation_v268/status.json`及checkpoint/审计为准。
- 尚无已通过自主起身、策略交接或最高DR的recovery checkpoint，不能部署为已掌握技能。

第249条更新：v268完整8秒确定性评估四种倒地方向均未起稳（最长ready0）；`evaluate_rmuc_recovery.py`在自动reset前保存终止状态，避免误统计。v269已实际启动1000更新独立初期复位训练，128env/128rollout、四固定初态和DR0，从新网络开始；完成后自动独立评估，状态以artifacts/continuation_v269/status.json与实际PID为准。固定初态训练只是第一课程，不代表最终泛化或交还通过。

### Recovery motion intensity audit (2026-09-30)

User requests controlled recovery action intensity. Evaluate physical leg-target increments (including the reset-to-first-action jump), measured joint speed, applied leg torque and body net contact force; normalized action-rate alone is insufficient because recovery uses the full leg range. The evaluator now emits `motion_audit` per first episode. These are control-boundary maxima, not physics-substep peak guarantees. Do not infer impact thresholds from total body net force alone. Any future target slew limiter must be implemented consistently in Isaac, export contract and MuJoCo/real consumers; do not add a deployment-only filter. Current v279 training and PD remain unchanged while diagnosing numerical joint-limit termination and action intensity.

## 2026-09-30 用户最新要求

下一正式恢复训练必须渐进初态并在训练内限制目标变化率，不能直接90度大幅甩腿。recovery_v3取消侧倒静止姿态正收益；可选progress替换恢复恒零目标vx槽index7（25D维度不变但语义变更），部署必须同步且禁止混用旧模型。当前仅原生接线验证；课程和部署consumer同步仍待实现。

## 反思落实与下一轮必要检查（2026-09-30）

- 旧侧倒姿态正收益、近成功才触发的高度奖励，使目标回报提升不能证明学会起身。v3已取消倾角>=60度时的静态姿态正收益；但站立高度项仍有12度/双轮门槛，过程塑形的可触达性尚未完全解决，不能把此项写成已修复。
- 直接90度倒地＋全关节范围动作使探索跨度过大。下一训练从30度开始并启用训练内目标slew约束，按真实扶正/保持/动作强度验收再晋级；低难度通过不是最终倒地恢复通过。
- 动作限速首次实现用错时间尺度且绕过首动作，v291无效。现以physics dt并从实测关节状态初始化，v300实际512子步写入PD检查最大增量0.008rad/2ms，含反向动作和中途reset，无终止。该结论只保证目标变化率，不保证实际腿速/冲击都受相同上限。
- 再跌倒惩罚先armed、持续越界、单次处罚，避免初态误罚。仍需在真实起身回合检查奖励贡献，不能以函数测试当行为有效。
- progress只表示测得恢复阶段，允许回退，不能按时间宣布完成；index7语义变更必须同步export/MuJoCo/controller。当前consumer同步未完成，禁止将模型按旧25D语义交付。
- 不再把奖励版本smoke、导出一致性、阶段编号当技能完成；每次长训练之前需确认有可观测的过程信号，检查动作峰值与重置；长训练保留已承诺预算，结束必须独立验收。
- 行走中旧参数冻结不等于行为保留；共享物理分支已发现局部梯度冲突。必须保留完整行为retention，不能靠加epochs或放宽gate解决。

## 第302条更新

恢复新progress/slew契约已支持导出和MuJoCo consumer，参见artifacts/continuation_v302；前文“待同步/拒绝progress导出”属于历史状态。实机controller和行为鲁棒性未通过。新policy_kind=rmuc_recovery_v2与reward profile recovery_v3是不同版本字段。
