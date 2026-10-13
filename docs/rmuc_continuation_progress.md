# RMUC 本轮接续记录
## 2026-09-23 v163已结束，Level28最终复测未通过

v163完成3753轮并保存最终checkpoint，当前无正式训练。3400正/负生存0.81944/0.96875；最终3753同seed/grid为0.80556/0.89063，coverage失败，虽steady tracking/bank通过仍不能验收。证据artifacts/continuation_v164/final3753_large_turns.json。当前诊断仅关闭level28新增head的输出层，保留其他所有张量，做同条件对照；head_ablation_launch.json记录进程，不能把诊断候选替换正式策略。
## 后期自转目标更新（2026-09-23）

用户要求先完成全速度行走/压弯，再将原地小陀螺自转逐步提高到双向10rad/s；0–3rad/s为中间能力，不再是最终自转上限。自转奖励按目标方向比较：目标至目标+0.5rad/s允许超速带不罚yaw，转慢强罚；零目标仍要求停转，不允许以该宽限带持续自转。姿态/生存/位置漂移/3Nm轮端持续上限及最高DR验收仍保留。

已有默认关闭的spin_asymmetric_reward_enabled实现此非对称奖励；欠速0.5rad/s默认额外扣10（有界），不是已验证最优权重。当前v163行驶训练未启用，不改变原行驶gate。后期旋转还需单独课程、足够yaw观测范围（±10换向至少覆盖20.5误差）、启停换向及相应验收指标。
> 原地±3rad/s基础支持已加入兼容选项：command_yaw_observation_error_clip可单独设6，sensor/延迟同步；command_yaw_overspeed_relative_to_target可避免合法高速旋转被旧1rad/s规则惩罚。默认保留旧行为，v163未启用；9项相关测试通过，不代表原地旋转已学会。详见improvement.md第54条。
## 用户补充最终能力：原地双向旋转0–3rad/s与广泛速度下转弯

原地vx=0时yaw必须可从0连续调到±3rad/s，包含启停/换向且限制位置漂移；行进状态须适应大部分速度下的转弯/压弯，需明确并实测速度—yaw能力边界。必须由同一最终策略在最高DR及最终动力配置下验收，不能用原±0.15/±0.30课程替代。当前尚无该能力证书。
> 用户确认采用低DR逐步推进最高DR（2026-09-23）。当前v163实际配置start29/full46、strength_override=-1：Level29约5.26%起步，阶段transition内升至10.53%，后续逐级增加，到46达到100%；环境reset才重采样。阶段进阶由技能/retention gates约束，但阶段内部强度爬升按transition时间，不是完整的性能自适应强度控制。后续若出现持续退化，应在有证据的强度区间巩固而非跳过gate；最终最高DR验收不变。
> 最新最高DR实测已完成且未通过：entry3051在2.4m/s、yaw±0.15，同seed nominal通过而strength1下survival0.5676/0.4337，yaw0.0701/0.0749、stable0.1800/0.0388均不达标。报告确认观测与指令延迟各实际采到1/2/3周期，并记录新IMU/编码器配置与source hash；见artifacts/continuation_v163/level27_skill_full_latest_dr.json。仍是5.5Nm迁移前动力、气弹簧关闭，不是最终硬件验收。v163继续Level28 nominal训练，最高DR通过仍为强制交付要求。
## 2026-09-23 当前Level28；Level27新技能独立通过

v163在3051保存model_level_28_entry_3051.pt，进入2.4m/s、±0.30大转弯；对应2p4_large head可训练，其他已认证head冻结。进阶日志retention speed max0.143/yaw max0.047/stable min0.666/survival min0.958。

该entry checkpoint的2.4m/s、±0.15独立复测各64回合全部通过：survival1，linear0.02159/0.02394、yaw0.02185/0.01991、bank2.12960°/0.47836°、stable0.87202/0.85708。证据artifacts/continuation_v163/level27_skill_2p4_small_turns.json。3064权重/冻结head/路由buffer审计全零变化。

训练不停；已安排首次Level29 entry或3400 checkpoint的2.4×±0.30独立验收，worker325742，level28_skill_status.json。最高DR新传感器延迟尚未训练完成，3Nm/径向气弹簧与部署迁移仍待完成。
> Level27 entry2866固定checkpoint已在seed43与seed45/layout2027两轮独立nominal网格通过2.0/2.2×±0.3各64回合检查；seed45的2.2负向survival0.96875，其余三桶1。证据汇总artifacts/continuation_v163/level26_repeated_nominal_evidence.json，不代表fullDR/全速度/部署通过。当前继续Level27，已安排首次Level28 entry或3200检查点独立验证2.4×±0.15（level27_skill_status.json）。
## Level27 entry独立四桶验收通过（seed43，nominal）

model_level_27_entry_2866.pt固定策略在2.0/2.2m/s、yaw±0.30四桶均完成64回合并全部checks通过，survival均1。2.0正向bank2.0033°、yaw0.04924、stable0.47053，余量小；负向yaw0.04305/stable0.53595。2.2正负yaw0.02553/0.02722、stable0.87291/0.79693。证据artifacts/continuation_v163/milestone_turns_nominal_seed43.json。

同checkpoint seed45/layout2027重复验证已启动，尚无结论；不把单seed当充分鲁棒。v163继续Level27，未重启。最高DR、全速度、3Nm径向气弹簧迁移/部署目标仍未完成。
## 2026-09-23 已通过Level26，当前Level27

v163在iteration2866实际进阶到Level27：bank_2p4_turn_0p15，保存model_level_27_entry_2866.pt。进阶日志active双向yaw max0.035、stable min0.708、bank max1.936°、survival min0.971；retention speed max0.144、yaw max0.046、stable min0.595、survival min0.985。进阶后日志计数归零是新阶段重置，不是丢失checkpoint或训练回退。

自动worker已针对该entry checkpoint启动2.0/2.2双向独立64回合复测，尚未判通过；训练PID314868继续Level27，不重启。关键base/expert/修正buffer共32个张量与初始化逐位一致，见artifacts/continuation_v163/level27_entry_integrity.json；完整运行审计继续覆盖其他冻结head。当前仍未最高DR/3Nm/气弹簧/部署验收。
> v163校准在2763结束，actor恢复更新；完整组合策略导出数值一致性验证通过。首次Level27 entry或model3000的独立双向验证worker已启动等待（artifacts/continuation_v163/milestone_validation_status.json）。早期2.0正向bank改善不等于完整进阶，仍以真实checkpoint与完整回合证据为准。
## 2026-09-23 当前v163：局部双向修正初始化后续训

从v161最终2753及v162验证过的修正方向构造both_direction_seed.pt，模型参数未变；正向2.0active差分-.12/+.12、负向wheel差分+.02/-.02只在speed(1.75,2.2)、对应大yaw平滑生效。负向64回合通过；正向单seed64通过但seed45网格2.0/.3 yaw/stable失败，所以这是学习初始化而非稳健证书。v163已实际运行，启动与run目录见artifacts/continuation_v163/launch.json。L2参考显式固定到该初始化，10critic校准后续训，其他LR/gate/physics不变；最新1–3周期sensor/action DR配置按课程启用（当前26仍DR0），气弹簧仍关闭、5.5Nm为迁移前阶段。
> 2026-09-23 用户更新DR契约：观测上行与控制指令下行分别随机1–3控制周期（当前8–24ms），取代先前5–25ms方案。最高DR全部环境分别reset采样并固定回合内延迟，低强度按strength提高启用概率。两条链路可叠加为2–6步总滞后；不是总链路最多3步。新增按物理单位IMU零偏/随机游走、加速度计对gravity估计的等效噪声、编码器角度/速度噪声。25-D/6-D不变。既有旧DR通过记录不认证新传感器/通信契约；新契约仅完成接入测试，尚未训练通过。
> 2026-09-23 当前无正式训练进程：v161已完成2753，最终复测2.0正向bank/负向yaw-stable仍失败。定向差分0.12经seed44/64回合诊断逐桶通过，已构造所有权重不变、仅2.0正向大yaw附近平滑固定修正的候选；完整候选物理验证进行中。详情improvement.md第32–36条、artifacts/continuation_v162。不是全目标完成或新训练已启动。
## 当前运行：v161，定向加强2.0转向奖励

v160已保存model_requested_stop_1753.pt退出；v161从该checkpoint接续，唯一奖励资格变化为turn_yaw_tracking_start_level26→18（速度阈值2.2→2.0），相同L2锚点、nominal直行保护、rehearsal0.60、学习率、gate与物理。10轮critic校准已结束，actor正在更新。启动/目录见artifacts/continuation_v161/launch.json。自动复测等待首次Level27 entry或model2050，状态milestone_validation_status.json，尚未出新证书。

v160/model1700四桶各64回合全存活：2.2双向全部通过；2.0正向bank3.4466°失败，负向yaw0.05926/stable0.31637失败。已有奖励代码counterfactual显示加强起点移至2.0增强该速度yaw/bank纠偏信号而抽查1.5/2.2/2.5/2.9不变，并非已证明训练成功。
> 历史口径（已于2026-10-01修正）：当前生效模型为左右 active_link_pitch 输出侧各 `-10.775 N·m`，机身施加等大反作用力矩。旧径向力探针保留为历史诊断，不作为最终执行器契约。
> 气弹簧接入进展：RMUC已新增默认关闭的独立机械力矩对（gas_spring_assist_torque_nm=0保持旧动力），45测试及短时PhysX/composer实测通过；尚未迁移训练或证明稳定性提升。v160继续原动力，model1700独立双向复测已自动启动。详见improvement.md第22条。
> 2026-09-22 电机/气弹簧参考已核对：用户确认RM3508持续最大3Nm是减速器之后的轮端扭矩。参考/home/kellen/wheel_leg_mjrl-lqr/wheeled_infantry.xml把±3再乘gear15.7647，MuJoCo实际前向得到±47.294Nm，不能照搬。保留gear时输入限幅折算±0.1902985Nm，内存修正模型已验证轮端±3Nm。参考气弹簧在lqr_deploy.py外加左右active_link各-10.775Nm，非XML弹簧；当前RMUC缺少对应辅助力路径，必须纳入后续动力迁移和最高DR验收。详见docs/rmuc_motor_spring_reference.md；原XML/USD/运行物理参数未修改。
> 硬件参数最终澄清：RM3508 的 **3 Nm 为持续最大扭矩**，表示持续输出上限，不表示任何工况均输出3 Nm。最终策略不得依赖超过此值的持续输出；短时峰值能力未提供，不自行假定。
> 2026-09-22 用户确认：轮电机为RM3508，3 Nm是持续输出扭矩，不是短时峰值。最终actuator/deployment按这一用户确认参数建立验收；不能再称其仅为USD元数据。当前5.5 Nm仿真训练仍为迁移前阶段。real_hardware配置0.8比例是另一个软件限幅，需在最终控制契约中显式核对；不得把用户确认3 Nm误当成该限幅已自动修改。
> v160固定里程碑独立验证已安排：首次Level27 entry或model1700，验证2.0/2.2双向各至少64完整回合。worker状态见artifacts/continuation_v160/milestone_validation_status.json；当前waiting_checkpoint，不能当作通过。保持训练连续，不改基线。
## 最终启用策略的硬条件（2026-09-22 用户明确）

最终control/controller及部署启用的必须是**同一份完整专家混合残差组合策略**，对应唯一checkpoint hash与导出制品；必须经过最高DR（strength=1.0）训练/验证并具备足够鲁棒性，同时覆盖全速度直行、双向压弯行进和转弯。不能只认证最高速度一个点、只验一个转向、只用nominal/早期DR结果，也不能拼接不同checkpoint的通过记录作为最终证书。

稳定掌握需在同一最终策略上重复独立seed、连续速度/yaw变化和持续行驶，逐点检查tracking、bank、survival、接触/姿态与用户认可的高度口径；完成后续actuator迁移后需在最终部署控制契约和最终动力参数下重新验证最高DR。全程保持25-D/6-D、closed-loop USD与frozen skill graph/rehearsal/retention。导出需完整组合图及数值一致性校验，control默认策略只能在上述证据齐全后切换。

若随机化范围与动力能力存在物理矛盾，必须明确诊断、解决并验证；不得静默降低DR范围、放宽性能gate或掩盖失败来交付。当前候选尚未满足这些条件，不能作为最终启用策略。
> 2026-09-22 稳定掌握要求更新：同一策略须通过重复/双向/fullDR/持续命令及部署验证。固定v158候选fullDR seed45已实测未通过（69完成回合、33tilt、survival52.17%、speed error0.3356），不是高度拒绝，也不能单凭动力估算归因。v160保持连续训练，未变基线或削减DR。详见improvement.md第14条与artifacts/continuation_v160/full_dr_straight64_seed45.json。
## 2026-09-22 当前v160：补齐retention bank，原checkpoint接续

v159完成50轮critic warmup并保存model_requested_stop_1409.pt正常退出。真实checkpoint比较确认actor恢复更新后，2.9straight四个受保护head切片全部逐位不变。v160从1409继续actor/critic/optimizer，不重复warmup，固定candidate L2锚点、奖励、学习率、物理参数不变；加载retention双向bank EMA/3°gate、进阶与actuator推进检查、bank deficit方向采样。实际迭代与新bank日志已出现，启动记录artifacts/continuation_v160/launch.json。初期出现一次2.2正向tilt终止（约85.5°），不能把之前deterministic64回合通过当成随机探索100%安全，继续以完整计数判定，不因单次事件再改基线。
## 2026-09-22 当前v159：恢复直行路由并修复retention

v157已STOP保存model_requested_stop_1643.pt退出。v158候选2.9直行nominal/DR2/19通过；2.2双向nominal各64回合通过；2.0正向bank和负向yaw/stable未通过。v159从已测candidate(model1350来源)续训，启用nominal2.9straight局部head/Adam切片保护，rehearsal0.60，用户高度口径0.15，固定L2参考显式改为候选；50critic warmup，保持原LR与全部其他gate。已看到真实1353迭代，PID216377，启动与run路径见artifacts/continuation_v159/launch.json。尚未通过Level26/全目标；详细证据及反思见improvement.md第10/11条。
> 2026-09-22 用户新增要求：每次瓶颈、反思、攻克方法及证据更新到 [improvement.md](../improvement.md)。已补录本轮8类问题，区分代码修复、局部行为通过与未闭环；后续持续追加，不覆盖历史证据。
## 2026-09-22 直行候选DR2/19通过，转向复测进行中

候选exact_base_2p9_candidate.pt在seed44、DR2/19、64环境/64完整回合通过2.9直行：survival1.0、linear_error0.146764、stable0.505539、root_min0.291888。证据为artifacts/continuation_v158/exact_base_straight64_dr_seed44.json。不代表fullDR通过。观察期间曾误把1000步进度当作结束；重复启动被输出文件防覆盖检查拒绝，JSON保持完整，但原.log被重复命令重定向覆盖成拒绝信息，日志不再代表原运行。

官方v157 iteration1556仍Level26：turn yaw0.0693、stable0.4357；此前1544日志还明确显示retention未通过（Level23 speed error0.2746/stable0，另有2.0负向yaw/stable失败）。所以此前只列当前stage yaw/stable为阻塞不完整。已启动候选2.0/2.2、yaw±0.30、seed43、nominal、每bucket8环境/12000步、64回合要求的独立验证，输出exact_base_turns_nominal_seed43.json。官方训练不变。
## 2026-09-22 2.9直行恢复候选通过单seed nominal复测

`artifacts/continuation_v158/exact_base_2p9_candidate.pt`恢复2.9直行精确base路由，332个受保护tensor不变，44个其他锚点输出不变（过渡区仍改变）。seed43/64回合 nominal：survival1.0、linear_error0.146924、stable0.604515、root_min0.294759，全部checks通过；与v130同条件结果一致，不依赖root门槛降至0.15。完整JIT/ONNX导出通过，24项相关测试通过。详见`artifacts/continuation_v158/exact_base_straight64_nominal_height015.json`和`control_contract_audit.md`。

官方v157保持运行且仍在Level26，未采用此候选。seed44/DR2/19复测已启动（输出`exact_base_straight64_dr_seed44.json`），完成与否以文件和进程为准。候选尚缺过渡区、跨速度、双向、fullDR和部署验收。
## 2026-09-22 用户更新：允许低姿态行驶

撤销行驶 root z >0.20 m 的硬要求，暂用 >=0.15 m，保持速度、yaw、bank、接触和生存 gate。奖励 minimum height 从0.36改至0.15 m；站立任务的目标高度和行驶中的软姿态偏好未删除。0.15 m 并非已验证的导轮触地高度，需实测姿态/接触；不追溯改写旧报告。v157 已启动进程仍使用旧0.20 m配置，新配置从下一次启动/复测生效。


## 2026-09-22 critic 审查：当前v157

v157真实校准复核已完成：policy1152的独立MC样本上RMSE467.04→374.97、bias429.45→328.56；model1175/1200的actor tensors与1152完全一致。50轮warmup已结束，actor恢复更新。仍有明显价值偏差，后续以学习和完整行为验证继续判断，报告`artifacts/continuation_v157/live_critic_calibration_check.json`。

**当前v157**从v155保存的`model_requested_stop_1152.pt`恢复，启动记录`artifacts/continuation_v157/launch.json`。关闭critic absolute value clipping，保留actor PPO clipping0.1及全部LR/epoch，先做50轮critic-only多任务校准。24项相关测试通过，实际启动参数确认use_clipped_value_loss=false。详见[停滞审查末尾](rmuc_stall_audit_v154.md)。

model900两组自动验证已完成且整体均未通过：2.9直行speed0.200698/stable0；2.2双向DR负向通过本次阈值，但正向一次倾倒且coverage不足。不能把两项奖励修复等同于技能已恢复。

critic MC诊断仅采用未观测尾项误差上界≤5的样本。其value平均偏差约+559，EV仍0.814；独立留出环境固定数据校准100轮，旧0.1 value clipping RMSE439.47，关闭后159.11。离线单command critic不用于训练，v157直接恢复原checkpoint再校准全部当前任务。新增Monte Carlo工具、实际PPO梯度测试及warmup跨Level冻结/正确路由测试。训练仍需实际进阶与多seed/full-speed/DR/部署验收。

## 2026-09-22 19:39 奖励截断修复：当前v155

下一次独立验证已启动等待worker：`artifacts/continuation_v155/validate_model900.py`，实时状态`model900_validation_status.json`。它等待当前v155的model900完整落盘，之后顺序执行2.9直行nominal64回合、2.2正负yaw各64回合DR2/19（seed43、layout2026）。不重启或修改训练；应以status和最终JSON判断结果，不能将waiting_checkpoint当作验证已完成。训练若提前退出且无checkpoint，worker会记录失败并退出。

此前bank shaping scale提升到2时未核对总正奖励上界，导致cap10压平部分精细tracking奖励。synthetic实际奖励代码测试与真实物理姿态counterfactual扫描均复现；按实际训练参数，yaw误差0.06仍有31.25%姿态到达cap，误差0时81.25%。当前正奖励项保守上界14.85，因此将reward cap改为20；所有behavior gate不变。新增排序测试，20项相关测试通过。细节和限定见[停滞审查](rmuc_stall_audit_v154.md)。

v154已保存`model_requested_stop_765.pt`停止，**当前v155** `2026-09-22_19-39-50_reward_signal_unclipped_to31_v155`从766续训；命令`artifacts/continuation_v155/launch.json`。固定原L2锚点、5轮critic warmup，保持原LR/epochs/控制/physics/DR/验收条件，仅修正reward上限。仍目标通过Level31到Level32 entry，再继续全目标，不宣称已恢复进阶。

## 2026-09-22 19:24 停滞审查后启动v154

**当前运行v154** `2026-09-22_19-24-07_command_reward_consistency_to31_v154`，启动记录`artifacts/continuation_v154/launch.json`。v151已保存`model_requested_stop_624.pt`停止。用户指出长时间未推进后，本轮暂停了反复调参，确认并修复active/rehearsal/global-Level奖励不一致；实际同物理状态的cohort差异1.933624、Level差异5.093191修正为0，19项测试通过。完整证据、流程问题与修复边界见[停滞审查](rmuc_stall_audit_v154.md)。

v154从625轮恢复原actor/critic/optimizer，固定原L2锚点，5轮critic warmup，启用`env.command_reward_contract_by_command=True`，取消regression warn-only，停止点改为32（须通过31）。不改USD、25-D/6-D、控制反馈、DR或gate，不采用诊断偏置候选。奖励逻辑一致性已验证，技能恢复与课程进阶尚待实测，不能提前宣称成功。

## 2026-09-22 19:09 pitch-rate 符号诊断：不采用增益反转

当前canonical映射`(-source_z, source_x, source_y)`行列式为-1。对已有真实失败轨迹的357个低roll/pitch样本，pitch数值导数与记录的angular_y相关系数-0.93553；取负后的导数误差中位数0.05770，原符号为0.25789。`artifacts/continuation_v152/pitch_rate_sign_check.json`支持运动学上的反号关系，但不能据此直接认定闭环gain错误或最终根因；25-D观测约定不变。

评估器增加仅诊断的`--pitch-rate-feedback`（范围[-0.2,0.2]），报告实际gain及diagnostic_feedback_override，有override时总体passed强制false，不能作为策略证书。原gain+0.04与-0.04冻结策略A/B均seed43、DR2/19、64环境：
- 原model500：+0.04 yaw0.054153/stable0.363556/survival1；-0.04 yaw0.054518/stable0.371446/survival1，无实质tracking改善。
- +0.01 wheel-diff候选：+0.04 stable0.442987、maxpitch79.03°、survival1；-0.04 stable0.426794、maxpitch86.00°、survival0.984375，变差。

对照见`pitch_rate_feedback_ab_comparison.json`及四份原始报告。**不反转正式反馈增益，不改运行策略或观测轴；v151继续。**脚本编译及两次真实64episode诊断执行通过；技能验收仍未通过。此结果不能写成“已找到最终根因”，也不能因坐标推导覆盖实测反证。

## 2026-09-22 19:00 小幅轮差动对照，正式训练仍v151

`artifacts/continuation_v151/model375_negative_small_diff_probe.json`为shuffle layout2026、DR2/19、每case32episode的诊断：wheel diff0/+0.005仍yaw/stable失败，+0.01/+0.02均survival1、tracking/stable及高度通过。仍是diagnostic_action_bias，不能直接认证。

创建独立候选`artifacts/continuation_v152/model500_negative_diff_0p01_candidate.pt`：仅对model500的`rmuc_turn_delta_adapters.15.negative_net.2.bias`轮动作2/5分别加+0.01/-0.01，移除optimizer，明确candidate_only。CPU检查旧≤2.0速度锚点、正yaw、小yaw与2.9straight输出不变；2.2及以上negative large yaw的输出相应改变。来源/hash/支持域见`seed_provenance.json`。未替换正式训练。

新seed43、64episode、DR2/19严格A/B（`seed43_ab_comparison.json`）表明：baseline500 survival1，yaw0.054153、stable0.363556、minroot0.291156；候选survival1，yaw0.047889、stable0.442987、minroot0.211660。候选仍stable<0.45，且高度余量缩小，不能采用为认证策略或声称所有目标完成。该结果支持后续更有针对性的学习，但不能用均值改善掩盖姿态代价。

正式v151继续，不因未通过的候选重启；最新汇总`artifacts/continuation_v151/latest_audit_summary.json`至537轮共232次update，冻结权重违规0、动作均有限，相对固定271锚点最大动作偏移0.012245。后续应继续看active/retention和正式checkpoint独立评估。

## 2026-09-22 18:50 可复现随机排列验收

`scripts/evaluate_rmuc_grid.py`新增`--layout-seed`，由`scripts/rmuc_eval_layout.py`使用独立Python RNG打乱command/bias-case到环境的映射，不消费仿真RNG；阈值、episode计数、各项统计及failure-trace bucket均按实际映射归组。报告保存`layout_seed`和`environment_bucket_ids`，未指定时保持旧排列。`tests/test_rmuc_eval_layout.py`三项测试通过，验证固定旧顺序、随机数隔离、分桶计数/阈值一致性；脚本编译通过。

真实128环境验证`artifacts/continuation_v151/model375_both64_layout2026.json`：model375、nominal、仿真seed42、layout-seed2026；正负各64episode均survival1。正向speed0.02978/yaw0.02653/bank2.26777°/stable0.85859/minroot0.29334，全部本次条件通过；负向speed0.03202/yaw0.051747/bank0.72831°/stable0.45537/minroot0.28225，仅yaw失败。已核对保存的环境分配每桶恰64个，评估未整体通过。

正在以同model375做小幅负向轮差动0/+0.005/+0.01/+0.02对照，DR2/19、独立layout-seed2026、每case32环境。该实验为diagnostic_action_bias，永不作为策略证书；正式训练v151继续、不变更运行策略。

## 2026-09-22 18:45 model375 同布局复测（v151继续）

`artifacts/continuation_v151/model375_both64_nominal.json`沿用model300的checkpoint外所有评估条件（2.2 m/s、正负0.3、每方向64环境、nominal、seed42、原排列）。正向survival由0.91176升至0.95455，重复提前失败从env7/28减少到env7；minroot0.200033m虽略过高度检查，但env7三次tilt终止、coverage失败，整体未通过。负向64episodes survival1、minroot0.29180m、stable0.53904，yaw0.051669仍超过0.05；不能四舍五入成通过。对照见`model300_375_comparison.json`。

v151已连续审计93次update至398轮，immutable与frozen-step权重变化始终0，最大锚点动作偏移0.006986；checkpoint actor没有normalizer状态键，未发现归一化状态暗中变化的证据。仍须以实际行为证明防遗忘。保持当前训练、不重启，后续在充分学习后复验。此次仅新增评估产物与文档记录，未修改策略/控制代码。

## 2026-09-22 18:41 固定排列敏感性对照（训练仍v151）

同一model300、seed42、nominal、128环境，仅将正/负yaw的前后64环境分配交换。`artifacts/continuation_v150/model300_nominal_swapped_layout.json`结果：正向64episodes survival1、minroot0.29554m、speed0.02261、yaw0.02478、bank2.19228°、stable0.88420，全部本次条件通过；负向64episodes survival0.984375、minroot0.19583m、yaw0.06200、stable0.28480，未通过。此时失败为env63在1482步，原排列反复提前失败的env7/28换成负向后未复现早期失败。

对照报告`model300_layout_comparison.json`。说明当前表现依赖command到cloned-env的分配，不能仅用单一固定排列或单个方向均值作为证书；也不足以断言是PhysX故障或纯粹残差失控。已从代码及当前配置确认评估无速度bootstrap、无额外terrain friction随机化；根因仍需分离初始状态/位置相关数值差异与策略鲁棒性。后续不能删除env7/28来伪造通过。保持v151训练和实时审计，无本轮策略/控制/门控修改。

## 2026-09-22 18:32 审计覆盖修正，v151

model300的同checkpoint/seed/128环境nominal与DR2/19对照已完成，见`artifacts/continuation_v150/model300_both64_comparison.json`及两份完整报告。nominal正向68 completed episodes survival0.91176、最低root0.19162m；env7/28各重复提前倾倒，尽管有steady样本环境的speed/yaw/bank/stable均过。DR正向66episodes survival0.95455但最低root0.18938m、coverage不足；env7三次失败。负向nominal64episodes survival1、root0.29043m、yaw0.050186略超0.05；DR survival0.984375、root0.19751m、yaw0.05450，均未整体通过。nominal复现证明DR不是正向提前失稳的必要条件，尚不能认定具体根因，更不能把均值通过或权重冻结当作全行为通过。v151保留正在进行的训练，避免仅因小样本或单项改善再次换基线；继续积累full-episode与retention证据。

**当前训练 v151**：`logs/rsl_rl/rmuc_wheel_leg_course/2026-09-22_18-31-41_level26_stratified_residual_audit_v151`，日志`train_level26_stratified_residual_audit_v151.log`，启动参数`artifacts/continuation_v151/launch.json`。v150通过STOP保存`model_requested_stop_305.pt`后退出，v151从306继续。固定L2锚点仍为v149 `model_requested_stop_271.pt`，通过新参数`--residual_anchor_checkpoint`显式保持，避免重启时重置锚点；actor/critic/optimizer恢复，系数1.0及奖励/验收条件不变。

发现并修复诊断抽样aliasing：对(T=128,N=128)展平后按stride32抽512点，实际反复选中少数env，某轮仅覆盖2.2负yaw。该问题只影响diagnostic probe，不影响PPO minibatch正则化或全参数权重审计。现在`rollout_probe`按time与env分层、覆盖全部128环境，512点来自4个时间截面；domain diagnostics也复用这一方法。14项测试通过，包括覆盖所有环境、不消耗RNG、固定外部锚点续训和梯度一致性。

v151真实运行第308轮日志已验证512样本、16个command桶，包含正负Level26、低速rehearsal和2.9直行；immutable/frozen-step参数变化均0、动作有限、每轮4个PPO minibatch接入正则化。相对原271锚点最大动作偏移约0.00297。实时`residual_safety.jsonl`位于当前run目录。v150前期action probe有覆盖局限，不能用于宣称当时全部技能行为未变。

model300双方向DR验证已启动，输出目标`artifacts/continuation_v150/model300_both64_dr.json`，2.2/+0.3与-0.3每方向64环境、seed42、DR2/19。结果必须以最终JSON为准，当前不宣称通过。

## 2026-09-22 18:28 用户要求：实时残差审计与正则化，v150

**当前训练 v150**：`logs/rsl_rl/rmuc_wheel_leg_course/2026-09-22_18-27-24_level26_residual_anchor_audit_v150`，PID与完整参数见`artifacts/continuation_v150/launch.json`。v149已通过STOP保存`model_requested_stop_271.pt`后退出；v150恢复actor/critic/optimizer，从272轮继续225轮，仍为Level26，尚未认证27。

新增`scripts/rsl_rl/rmuc_residual_safety.py`与`--residual_anchor_l2`：当前系数1.0，以本次续训checkpoint的完整策略作为固定锚点，约束同一rollout observation上新增输出偏移的L2；不把已学会的专家和压弯修正强行归零。均值gradient hook加入的梯度严格等价于`lambda * mean((mu - frozen_anchor_mu)^2)`，在现有PPO gradient clipping之前合并；不改变raw action/logprob/reward或推理输出。仅支持feed-forward、无symmetry augmentation、非AMP PPO。该项限制续训新增偏移，不证明锚点自身已合格，也不是对所有历史残差的绝对幅度限制。

每次完整PPO update写入`residual_safety.jsonl`并flush：固定base/旧expert权重偏移、本轮冻结参数偏移、正则损失、相对锚点最大动作变化、完整composition与base的RMS差，以及按vx/yaw分桶的动作偏移/饱和比例。composition-minus-base包含合法专家混合，不能仅凭其非零认定base被篡改。冻结权重实际变化或非有限输出会保存`model_residual_safety_violation.pt`并停止训练。PPO日志/TensorBoard同步记录正则和权重指标。

真实运行首两轮272/273：每轮4个minibatch均接入正则，immutable与frozen-step权重最大变化均0，策略相对锚点最大动作偏移约0.000282/0.000541。11项测试通过，包括显式Gaussian损失+L2与注入梯度等价、推理不变、冻结权重篡改触发保存/停训。后续仍需观察训练曲线与行为认证。

本轮新增行为证据：
- v149 model175 frozen check通过，base/旧expert及冻结stage heads逐tensor不变，`artifacts/continuation_v149/freeze_check_model175.json`。
- 同checkpoint、seed42、每格3episode的nominal/DR2/19全35格对照均17格通过，但格子不同：DR下2.0/-0.3恢复stable，2.9/+0.15新增linear/stable失败。报告`model175_grid_comparison.json`。仅诊断，不能当作完成率。
- 1.5/-0.3的64episode nominal复测实际通过：speed error0.18732、yaw0.03366、stable0.58058、survival1。不能把3episode网格失败直接归因为遗忘。
- 相同64环境/seed条件，关闭该速度/负大yaw的局部domain新增输出后，speed error0.16972、stable0.69779、survival1（yaw0.03486）。证明该局部残差在本次对照中降低速度/stable表现，但不足以全局关闭残差。`ablation_no_local_1p5_negative.pt`明确是diagnostic_only，仅改变该局部输出及邻近插值，不用于正式训练。初次CPU接口调用错误已修正为policy observation字典，支持域检查通过。

## 2026-09-22 18:12 接续：v149 与完整 DR 失败轨迹

**当前正式训练为 v149**：`logs/rsl_rl/rmuc_wheel_leg_course/2026-09-22_18-11-08_level26_turn_yaw_shaping_resume_v149`；日志 `train_level26_turn_yaw_shaping_resume_v149.log`；启动记录 `artifacts/continuation_v149/launch.json`。v147 已通过 STOP 保存 `model_requested_stop_145.pt` 后退出。v149 恢复 actor/critic/optimizer，从146轮继续350轮，Level26通过才保存Level27 entry。唯一奖励变更为 `env.command_turn_yaw_tracking_start_level=26`，提前启用已有的turn yaw专用惩罚；pitch-rate专用惩罚仍从28开始。base/旧expert冻结、局部残差、bank门控及retention条件保持原约定。当前训练DR仍为0，独立评估DR为2/19，不能混淆。

独立评估证据：
- `artifacts/continuation_v147/model100_negative64_dr_trace.json`：model100、2.2/-0.3、seed42、DR2/19，64 episodes survival1、最低root0.29128m、bank0.71789°；yaw error0.05485、stable0.36857未过。仍未认证Level26。
- `artifacts/continuation_v148/straight64_dr_conservative.json`：保守偏置候选2.9直行，64episodes中63存活，speed0.13250、stable0.71466，但最低root0.19413m且coverage不足，整体失败。
- `artifacts/continuation_v148/negative64_dr_conservative.json`：相同候选2.2负向，64episodes中62存活，yaw0.03930、stable0.57110，但最低root0.18945m，整体失败。不能作为正式训练基线或证书。

`scripts/evaluate_rmuc_grid.py` 新增可选 `--failure-traces`：保存最多16个不安全episode、最近500控制步的状态/动作（每10步取样）、首次高度违规或termination的完整状态及该episode实际DR参数。仅诊断记录，不修改动作、physics、阈值。`negative64_dr_failure_trace.json`真实复测成功捕获两个事件，env23/15在1341/1421步首次跌破0.2m；此前出现超速（约2.69/2.80m/s）和pitch增长，违规时pitch约83/84°。两者delay均0，因此不能将这两例归因为观测延迟；其余随机化项的独立因果影响仍未确定。

验证：评估脚本编译通过，真实64episode评估包含无失败和有失败两种trace路径；action std与bank gate共6个测试通过。v149启动确认resume146及原learning rates，后续仍需独立DR、多seed、旧技能及全速度复验。全目标未完成，不提供虚构完成百分比。

Level23直行细扫DR结果见`artifacts/continuation_v148/straight_fine_dr_probe.json`（每格16episode，strength2/19）：baseline speed error0.1582/stable0.399未过；共模-0.005 stable0.4486仍未过；-0.010 speed0.1230/stable0.6612/survival1/minroot0.2960，-0.015 speed0.1175/stable0.8788/survival1，均通过本轮条件；-0.020 survival0.875/minroot0.1932失败。不能只按speed最小选偏置。已创建较小修正候选`directional_speed_seed_conservative.pt`，将局部2.9straight domain修正由约-0.02降至约-0.01；正进行64episode DR验证。正式训练仍v147，未替换为未验证候选。

2026-09-22轮动作诊断：`artifacts/continuation_v147/wheel_probe.json`显示Level26 negative差模+0.04及2.9straight共模-0.02能改善tracking，但只是每格3episode。进一步候选`artifacts/continuation_v148/directional_speed_seed.pt`把前者写入Level26 negative head、后者写入2.9straight局部domain bias，CPU支持域隔离检查通过。每格16episode验证`nominal_candidate_grid.json`未整体通过：2.2/-0.3 survival1、yaw0.0493通过；2.9straight speed0.1037但survival15/16、出现低root，不能采用为认证基线；2.2正向也出现1/16失败。尚未切换正式训练到该候选，v147继续运行。正在对2.9straight共模0/-0.005/-0.01/-0.015/-0.02做DR=2/19、每格16episode细扫，不以速度改善掩盖survival失败。

**当前运行：v147** `level26_bank_seed_stage_lr_v147`，日志`train_level26_bank_seed_stage_lr_v147.log`，run目录`2026-09-22_17-51-19_level26_bank_seed_stage_lr_v147`，启动记录`artifacts/continuation_v147/launch.json`。v146已通过STOP保存`model_requested_stop_577.pt`后退出。v147从下面的probe初始化候选finetune（加载critic、新optimizer、5轮critic warmup），stage residual LR=2e-5，domain/critic LR=1e-4，base LR=2e-6且base/旧expert冻结。当前仍是Level26严格认证，不是已通过Level27。

初始化候选的邻近grid结果：`artifacts/continuation_v147/seed_neighbor_grid.json`（25格、每格4episode），19格通过小样本条件。2.2/2.4/2.5 m/s正向0.3yaw的bank误差2.274/2.661/2.932°，均survival=1且tracking通过；2.0正向bank仍3.468°，2.2负向yaw仍0.0963未通过。候选没有掩盖旧失败项。64环境正向2.2/0.3、DR=2/19的独立验证正在运行，结果未出前不能宣称随机化认证。

该64环境验证已完成并通过全部本次条件，报告`artifacts/continuation_v147/positive64_dr.json`：64 completed episodes，survival=1，linear error0.02976m/s，yaw error0.02889rad/s，bank error2.2996°，stable0.83884，physical min root0.29420m。仅覆盖2.2/+0.3、seed42、DR strength2/19；不是全速度、双方向、多seed的认证。验证对象是初始化候选，后续v147训练checkpoint仍需单独复验。

2026-09-22 17:48诊断更新：v146连续训练至550+轮仍未通过Level26，正向bank误差约3.83°。固定model475单腿±0.06偏置主要改变速度，没有明显侧倾；稳态raw action无±1饱和（`artifacts/continuation_v146/action_saturation_probe.json`）。成对±0.20诊断中，两个hip同加0.20产生roll≈-1.536°，bank误差2.313°、速度误差0.0234、yaw误差0.0283、4episode survival=1；其他一些偏置导致倾倒，不能泛化采用。

候选 `artifacts/continuation_v147/level26_positive_bank_seed.pt` 只将当前Level26正向head的action[1,4]输出bias各加0.20，base/旧专家不变；删除optimizer state并明确标记candidate_only。CPU对照确认≤2.0m/s、负yaw和小yaw输出不变，正向大yaw≥2.2m/s会受影响，因此正在做邻近速度grid。此文件只是probe支持的初始化候选，不是PPO训练完成或认证产物；来源、旧/新bias与hash见`seed_provenance.json`。正式训练仍为v146，未切换到该候选。

另准备了`--stage_residual_learning_rate`开关，独立改变stage bank optimizer group，冻结的旧heads仍无梯度；CPU隔离验证通过，尚未启用，等待上述物理证据指导。

最新运行：v145在iteration236通过STOP marker保存后退出，v146 `level26_bank_gate_active_shaping_v146`从237恢复；日志 `train_level26_bank_gate_active_shaping_v146.log`，launch记录 `artifacts/continuation_v146/launch.json`。训练门控此前只记录bank误差，不要求通过；现已增加双方向≤3° bank条件，与grid检查一致。6个回归测试通过（直行不受bank条件影响、任一turn方向bank失败不能通过、wheel std floor）。正向bank误差持续约3.82°，本轮仅通过Hydra override把active stage的bank reward/penalty乘2，rehearsal bank权重仍为1；对应隔离检查通过。尚无改善结论，需等待轨迹与独立验证；不因增加门控/奖励而宣称技能完成。

最新运行更新：v144通过STOP marker保存 `2026-09-22_16-56-10_level26_unified_contract_resume_v144/model_requested_stop_199.pt` 后退出；v145 `level26_std_floor_resume_v145` 从iteration200恢复，日志 `train_level26_std_floor_resume_v145.log`，启动记录 `artifacts/continuation_v145/launch.json`。下文v144状态为历史快照。

本轮修复wheel std floor实际未写回的问题：高级索引产生副本，原`std_parameter[list(indices)].clamp_()`没有修改Gaussian参数；改为索引赋值。model175实测轮std约0.0059585/0.0059818，低于0.006；v145启动日志确认实际floor=0.006。`tests/test_rmuc_action_std.py`两个测试覆盖scalar/log参数、optimizer更新后重新clamp及腿梯度隔离，均通过。修复影响的是探索下限，不构成技能已通过的证据。

model175独立复测（每格8 episodes）见 `artifacts/continuation_v144/weakpoints_model175.json`：2.2/+0.3 bank误差3.831°；2.2/-0.3 yaw误差0.0602；2.9直行速度误差0.1543；这些格子仍未通过。2.9/+0.3无稳态样本且survival=0。必须继续学习并复验，不得把小样本或训练aggregate作为正式证书。

2026-09-21 开始。目标：保持现有专家/残差体系，完成全速度直行、双向压弯和 flat DR 验收；不放宽进阶条件。

当前学习轮（2026-09-22恢复）：`level26_unified_contract_resume_v144`，日志 `train_level26_unified_contract_resume_v144.log`，run目录 `logs/rsl_rl/rmuc_wheel_leg_course/2026-09-22_16-56-10_level26_unified_contract_resume_v144`。v143已无存活进程，最后日志为83轮，最新可用checkpoint为model75；v144使用resume恢复其actor/critic/optimizer，从iteration76继续。128 envs，RTX4080，名义Level26再认证，目标通过后保存Level27 entry；DR课程仍从Level29开始。物理episode与gate统计重新收集，不能将启动时的零计数当作旧技能丢失。启动参数与PID保存在 `artifacts/continuation_v144/launch.json`，需以实际进程确认存活。

## 已保留的基线

- v139 已在 iteration 566 附近停止；最新落盘 checkpoint 是
  `logs/rsl_rl/rmuc_wheel_leg_course/2026-09-21_19-15-18_level29_progressive_flat_dr_domain_bank_v139/model_500.pt`。
- 没有得到 manual-stop checkpoint，因此不能声称 iteration566 状态已保存。
- v130 `model_125.pt` 的 `infos=None`，原 run 日志仅有 Level28、没有 Level29 entry 文件。交接文件将它称为 Level29 净基线不能作为严格进阶证据；本轮需先补齐 Level28 认证。

## 冻结策略因果对照

所有下列对照使用 v139 model500、128 envs、35×128 control steps，actor 和 action std 冻结。已逐 tensor 比较每轮 model0/model34，确认 actor 未变化。日志数值为最后 rollout 聚合值，计数可能显示小数。

| 对照 | Level23 mean episode steps | Level23 tilt count | retention ready |
|---|---:|---:|---:|
| nominal、原 rehearsal 控制 | 529.90 | 53.84 | 0.8333 |
| DR、原 rehearsal 控制 | 529.22 | 32 | 0.8333 |
| nominal、恢复 rehearsal 控制 | 1487.95 | 1 | 1.0 |
| DR、恢复 rehearsal 控制 | 1483.45 | 1 | 1.0 |

证据：`artifacts/retention_control_parity_ablation.json`、`train_level29_stage23_frozen_*.log`。

原控制下，Level23 所有被计入的完整 episode 都达到过300-step window，但未达到1200-step steady-state，并因tilt终止。零 tracking sample不是遗漏straight bucket；本checkpoint的主要失败路径是约530步倾倒。关闭DR不能消除此现象。
恢复feedback说明 rehearsal 屏蔽propulsion/pitch/部分speed控制是一个主要因果因素。这里恢复的是一个控制组合，尚不能把所有效果归功于某一个反馈项。短对照还不足以宣称通过64 episodes/方向的正式认证。

**修复之间存在冲突**：对全部rehearsal恢复反馈，使Level26 negative yaw error由约0.052变为0.118。应限定修复到高速straight rehearsal，保留旧turn expert的控制约定；不是给全部专家统一换控制器。

低速teacher历史yaw feedback clip为0.3，当前全局clip0.1来自高速安全调参。只恢复低速clip的nominal冻结对照中，Level8 positive yaw error约0.034、stable约0.966；高速度继续保持0.1。

## 已实施的支撑工作

- `env.py`：Level23完整episode长度、300/1200步资格与各termination reason计数；训练logger接入。
- 独立opt-in flags：rehearsal feedback parity、高速straight-only parity、low-speed yaw clip、command-consistent DR noise、bounded domain bank LR。未通过验证的控制开关没有写成全局默认。
- `run_retention_ablation.py`：顺序运行冻结actor对照、保存日志/summary/source snapshot，检查actor权重不变。
- DR sensor noise helper：保持command target/previous action精确，用同一noisy yaw sensor重算yaw error，避免full DR下专家路由被独立噪声改变。CPU校验通过；物理接入仍需实验。
- evaluator新增fixed DR strength、strict full-episode survival分方向计数、reset之前的物理最低root height；新增路径尚待仿真验证。
- 训练按CLI环境数检查/扩展flat plane，避免提高并行度后机器人离开地面。

## 导出验证

已修复默认导出只保留base MLP的问题。`rmuc_policy_export.py`捕获完整actor forward，`play.py`对RMUC composition使用该路径。
v139 model500在247组synthetic command/state inputs、batch1/7/247上：JIT max error=0，ONNX reference max error≤2.3842e-7；旧base-only wrapper最大偏差约0.0873。
报告：`artifacts/rmuc_export_checks/v139_model500/export_parity.json`。
这仅是raw normalized actor输出的数值一致性，不是最终策略或部署控制器认证；环境反馈、执行器映射仍需在部署端保持一致。

## 当前顺序

1. 完成低速clip恢复的DR对照。
2. 验证straight-only反馈修复，复查Level23/8/26/28，严格区分样本不足与真实失败。
3. 从现有checkpoint补齐Level28认证，再推进Level29及更高阶段；扩大并行env数量用于完整episode样本收集，不降低gate阈值。
4. 增强DR时保持command路由一致、正负yaw容量隔离、旧skill graph冻结；持续做nominal与DR独立验收，不能只看domain均值或全局mean reward。

## 并行规模与后续DR物理可达性

v140尝试2048 envs、256 m flat plane后，active contact约0.28、旧技能yaw和Level23 survival均退化。随后相同规模的冻结actor对照也复现退化，因此不能直接归因于domain LR或PPO更新。2048规模尚未通过仿真等价性验证，后续学习回到128 envs/128 m平地；更大规模暂不作为推进依据。

`artifacts/continuation_v140/dr_feasibility.json`记录了steady/no-slip/零额外负载近似下的力矩预算：当前action target先clip，再由IdealPD减去damping*qdot。最弱gain和最大damping组合在Level41的no-load速度上界约2.96 m/s，终局full DR约2.85 m/s，低于3.2-0.15。这是基于当前控制代码的物理可达性预检，不是实测速度认证。
在到达高速阶段前必须验证并解决执行器/DR工作点矛盾；不能靠放宽tracking gate或仅看随机化均值来宣称全范围鲁棒。

## v141–v143 的后续改进与验证

- v141（128 envs）contact恢复到约0.63–0.64，修正后的Level23多数完整episode可存活；但shared domain head跨速度修正仍可能互相影响。保存于 `2026-09-21_21-07-56_level28_recert_scoped_contract_v141/model_requested_stop_155.pt`。
- v142加入按signed target speed和五种yaw regime隔离的vectorized domain heads；旧shared domain heads冻结，新增head零初始化，最终总修正仍由`0.04*tanh`限幅。CPU验证：零初始化输出完全一致；2.2 m/s negative-large-yaw样本只向对应speed/regime产生梯度；其他离散锚点输出不变；checkpoint strict restore和ONNX parity通过。
- v142使用均衡完整episode取样修复：任一方向不足64 episodes时先50/50取样；两方向均通过后保持均衡；只有真失败bucket才自适应聚焦。旧实现只看300 window samples，并错误地给rehearsal套用active stage的survival deficit。均衡与stage-local survival测试通过。
- v142保存于 `2026-09-21_21-24-10_level28_speed_isolated_domain_v142/model_requested_stop_113.pt`。截至停止仍未通过Level28，不是认证checkpoint。
- scoped feedback parity是诊断过渡方案，不能作为最终控制约定：同一command在active和rehearsal使用不同控制器，部署时没有rehearsal标记。v143通过`--feedback_contract_by_command`使用一致的command-based控制，借助隔离后的residual逐项适配旧技能。CPU测试确认相同state/command在active和rehearsal输出完全相同。
- v143先修复当前薄弱的Level26，再认证27/28/29，不把v130未实际进阶的文件名当作证书。actor base LR仍为2e-6；critic独立1e-4，domain bank独立1e-4；保存critic并warmup 10轮，避免长期使用失配value估计。
- Level39/40/41新增3个speed extension heads，append-only保持旧checkpoint indices；在此前速度锚点严格为零，并将新速度能力传递给后续turn heads。旧checkpoint按其实际bank长度恢复；新训练可零初始化扩展。身份映射、锚点隔离、路由、ONNX一致性测试通过。
- 手动command的leg action scale按训练stage速度锚点连续插值；匹配所有已有正速度stage，并对反向使用对应绝对速度scale。这仅修正控制映射，不意味着倒车技能已通过验证。
- strict evaluator真实仿真smoke：`eval_strict_dr_smoke_v139.log`，4 envs/1800 steps/Level29/DR=2/19，正负各2个完整episode，survival=1，物理最低root z≈0.29688 m，所有smoke checks通过。样本量不足正式认证，未覆盖全速度。
- 后续run可在其目录创建`STOP`文件，训练在完整PPO update结束后保存`model_requested_stop_<iteration>.pt`并退出；已在v141/v142真实验证。

下一步必须等待并审阅v143的active/retention和critic曲线；所有nominal、DR、完整episode和最终部署验收仍未完成，不能把上述代码修复/短对照当作项目完成。

2026-09-22补充：actuator bridge增加retention全条件保护，ready/linear/yaw/stable/survival任一失败都不推进；CPU构造检查覆盖各条件单独失败及全部通过。当前v144没有启用bridge，本改动仅供之后迁移使用。新的grid evaluator正在做实际smoke验证，结果未出前不宣称全网格验收通过。

v144后续现场证据：已保存model100，比较v143/model75，旧base/experts/已认证heads逐tensor不变，仅允许的Level26方向head、局部domain bank和std更新；结果见`artifacts/continuation_v144/freeze_check_model100.json`。grid smoke已完成，报告`artifacts/continuation_v144/grid_smoke.json`：2.2 m/s、±0.3，每方向2个episode，survival=1、min root约0.30m；正yaw bank error3.83°未达本次3°检查，负yaw stable0.420未达0.45，因此overall=false。该小样本只验证评估器路径及暴露失败项，不构成正式技能证书。

model100的35格nominal基线已完成（1.0/1.5/2.0/2.5/2.9/3.0/3.2 × 0/±0.15/±0.30）：每格3个episode，19格通过本轮条件。1.0/1.5全部格子通过；2.0+正向大yaw仍有压弯角不足；2.9+高速负向存在tracking问题；3.0/3.2直行无法完整生存。报告为 `artifacts/continuation_v144/nominal_grid_baseline_model100.json` 及同名md，不能把19/35解释为正式认证完成率。旧报告零tracking样本时error为0，coverage/survival检查已拒绝其通过；新评估器已把这种error显示为null，md表格标明无样本。同checkpoint同seed的DR=2/19全格对照已启动。

上述DR全格对照已完成，报告 `artifacts/continuation_v144/dr_grid_baseline_model100.json`，比较见 `grid_nominal_dr_comparison.json`：17/35通过本次小样本条件。相对nominal新增失败为1.5 m/s、yaw=-0.30（linear/stable）与2.9 m/s、yaw=+0.15（linear），两格survival仍为1。这是单seed、每格3episode的诊断证据；后续DR适应必须包含这些弱点，并用多seed足量样本重验。v144仍在运行，model150已保存。
