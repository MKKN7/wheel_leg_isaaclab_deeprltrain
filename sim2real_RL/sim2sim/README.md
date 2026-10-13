# RMUC MuJoCo sim2sim

## 本地模型入口与对齐要求

腿部控制遵循[LEG_PD_CONTRACT.md](LEG_PD_CONTRACT.md)：60/2输出侧显式PD、目标速度0/前馈0、40Nm裁剪；同一`leg_pd.py`用于MuJoCo。真实Isaac源码1024组对照误差0，实机固件尚未接入，不能声称实机已一致。

**当前默认执行模型是 [rmuc_peak55.xml](rmuc_peak55.xml)**：轮端5.5Nm瞬态峰值、腿关节40Nm，gear=1直接使用关节输出Nm；3Nm仅作为持续热参考，不是默认瞬时裁剪。`run_mujoco_probe.py`默认选择此模型，动作尺度和输出上限默认5.5Nm，必须显式传入本次策略对应的 damping 与气弹簧等效力矩。`rmuc_usd_limits.xml`保留为历史3Nm降额诊断。请求限幅超过加载模型实际能力会拒绝运行。原wheeled_infantry.xml是未改的来源参考。

用户明确要求 **MJCF 与当前训练 USD 对齐**，该要求是后续修改和验收约束。

用户进一步要求保留腿轮拓扑：不得为调通策略改连杆层级、关节连接/轴/范围或四闭链约束。`topology_verification.json`已对原副本及两个生成配置逐项验证一致，配置生成器也加入结构不变检查。

新增显式执行器配置：`rmuc_training_v261.xml`为旧5.5Nm训练对照；`rmuc_usd_limits.xml`从USD读取轮3Nm/腿40Nm限制。两者gear均1，ctrl直接为输出轴Nm；生成脚本`build_mjcf_profiles.py`读取USD与v261参数并保留质量/惯量/几何，正负大输入饱和探针确认实际限幅。来源、hash及范围见`mjcf_profiles.json`。3Nm配置10秒闭环仍未过yaw门槛（速度/yaw误差0.211308/0.151626），不是全动力对齐证书。原`wheeled_infantry.xml`保持来源逐字节一致。

- 可直接打开的模型：[wheeled_infantry.xml](wheeled_infantry.xml)。
- 相对路径网格：`meshes_mj/`（21个实际引用文件）。模型及网格均已复制到本目录，不依赖外部目录或符号链接。
- 来源及22个文件的SHA-256：`model_source_manifest.json`，复制时逐文件校验一致。
- `run_mujoco_probe.py`默认使用本地`rmuc_peak55.xml`；历史审计和`run_dynamics_ablation.py`保留显式旧模型选择，保持原实验可复现。下文外部路径仅代表早期审计来源。
- 对齐状态：39刚体质量/初态位置/COM/完整惯量/关节轴已做数值对照；运行时接触、驱动及最终3Nm/气弹簧尚未全部对齐。不得因复制或加载成功就宣称完整sim2sim通过。

本次复制不静默改动物理属性：XML仍保留旧gear/ctrlrange和摩擦；诊断runner的力矩换算与显式runtime overrides另行记录。后续需要按USD authored值和训练runtime覆盖分别整理配置，不能将两者混为一个“USD参数”。

2026-09-29：用户指定部署根目录为 `sim2real_RL`，MuJoCo 验证入口放在本目录 `sim2real_RL/sim2sim`。现有空目录 `sim2real_RL/sim2sim_RL` 保留，不作为新入口。

当前状态：goal 已由用户恢复，按已确认部署优先级推进；尚未启动新训练。已实现并验证独立于 ROS 的 RMUC 25D 观测与 authority 动作 consumer，尚未移植、构建或运行 RMUC MuJoCo 控制器。本目录不是已完成部署的证明。

## 已运行的接口对照

`rmuc_policy_interface.py` 实现 nominal 25D 输入及归一化动作限幅；不含状态估计、执行器、传感器/通信延迟或物理闭环。输入必须使用训练 body frame 和关节符号，关节顺序为右 active/hip/wheel、左 active/hip/wheel。

运行 `/home/kellen/anaconda3/envs/robot/bin/python sim2real_RL/sim2sim/verify_interface.py`：从真实训练源码 AST 提取观测构造前缀，对照 1024 组状态；观测和 authority consumer 最大误差均为 0，ONNX Runtime 与完整 TorchScript 最大误差约 7.15e-7。结果保存在 `interface_verification.json`。使用现有 robot 环境（Torch 2.6.0、ONNX Runtime 1.18.1、MuJoCo 3.11.0），没有改动训练环境；这些版本不是上游指定依赖版本的等价认证。

## 上游参考

- 仓库：https://github.com/scutrobotlab/wheelbipe_ros2_sim2sim
- 固定版本：`dd367bf78c7e393d811595edeb4affe253156c95`
- 已阅读：README.md、DEPLOY.md、state_rl.cpp、robot_state.cpp、wheelbipe_V14.yaml。
- 上游链路：操作输入 → Controller/FSM → ONNX Runtime → ros2_control → MuJoCo 或 RealBridge。
- 公开版为 normal-only；FSM 为 INIT/IDLE/PREPARE/RL。不能据此声称已有复位、跳跃多 ONNX 切换。

版本化来源和源文件链接见 `upstream_reference.json`。后续引入源码时保留上游 LICENSE、第三方及模型/资产许可。

## 可以复用与必须适配

| 项目 | 上游 | RMUC 要求 |
|---|---|---|
| 观测 | float32 [1,35]，normal 模式字段固定 | 当前完整 actor 为 25D；按训练端顺序、单位、坐标、延迟及历史动作重建，不能补零冒充 35D |
| 动作 | 6D；文档主线为腿位置 PD、轮速度 PD，源码含输出模式分支 | 保持腿位置、轮力矩；逐项检查底层分支和单位，不能直接沿用轮速度 PD |
| 动作限制 | 上游自身缩放与滤波 | 复现 RMUC authority consumer；局部腿动作可能超过 ±1，不能统一 clip 到 ±1；不擅加滤波 |
| 周期 | ros2_control 配置 500 Hz；策略推理另有频率调度 | 当前训练控制步为 8 ms，必须分开核对物理、PD、策略及命令周期 |
| 模型 | V14 MJCF、关节顺序及 IMU | RMUC 闭链模型、惯量、碰撞、关节符号与被动自由度；不能用 V14 模型替代 |
| 模式 | INIT/IDLE/PREPARE/RL | 近期行走与复位策略独立；后续越障；交接条件、历史状态与失败退出需验收 |
| 环境 | 文档为 Ubuntu 22.04 / ROS 2 Humble | 本机实查 Ubuntu 24.04；本次未发现 /opt/ros，尚未完成 ROS 环境验证 |

最终实机目标为每轮输出轴持续 3 Nm、左右 active_link_pitch 各施加 10.775 N·m 等效辅助力矩并由机身承受等大反作用。旧径向 10.775 N 实验仅作历史诊断；v261 旧动力标称结果不能当作该目标已完成。DR 范围需版本化，独立电机响应不能用通信 FIFO 代替。

## 适配顺序

2026-09-29 模型初审：`audit_model.py --xml /home/kellen/wheel_leg_sim2sim/assets/wheeled_infantry.xml` 在 robot 环境实际编译 MJCF，并调用 issaclab 的 USD API 读取训练资产，结果见 `model_audit.json`。39个刚体名称匹配，总质量分别19.22113998/19.22113937kg；初始四闭链连接误差小于1.2e-16m。此结果不证明动力学等价：COM/惯量方向、关节轴、碰撞与运行时驱动仍待对照。

重要单位差异：候选MJCF轮执行器gear=15.7647058824，ctrl=3会给出约47.294Nm关节力矩；输出轴3Nm应先除gear，ctrl约0.1902985。六执行器逐个用mj_forward/qfrc_actuator验证输出力矩换算通过，未积分、未修改原模型。训练动力尺度5.5Nm与最终3Nm必须分开记录。

接口验证已加严为直接执行env.py的实际authority限幅分支，包括相邻float32边界。部署consumer相应改成与环境一致的直接条件判断；不再仅用平滑门控helper充当执行端参考。1024组验证仍通过。

1. 核对 RMUC MJCF 来源、闭链约束、质量惯量、执行器、初始姿态和坐标系。
2. 固定当前策略及 action_contract，建立训练端与部署端观测/动作离线对照。
3. 接入 MuJoCo 后端，先验证原契约下的站立、基础行走，再推进最终动力参数。历史基线和最终硬件配置分开记录。
4. 复用 ROS 输入、ONNX Runtime、状态管理和后端抽象，保持 RL 直接输出轮力矩，不加入任务级速度/yaw反馈。
5. 增加独立 recovery ONNX 与交接训练；恢复成功要求自行起稳、切回行走并执行命令，不能调用仿真 reset 冒充复位。
6. 对同一发布模型做最高 DR、连续操纵、切换和 MuJoCo 回归；保留模型 hash、配置、失败统计及复现命令。

近期优先两个 ONNX（locomotion/recovery），跳跃与飞坡后续作为 traversal。原地 ±4 rad/s 自转与旋转中平移分开验收。上游框架减少后端搭建工作，但不替代 RMUC 模型和闭环适配验证。

## 当前候选定位

复位ONNX独立MuJoCo入口（第252条）：

```bash
/home/kellen/anaconda3/envs/robot/bin/python sim2real_RL/sim2sim/run_recovery_onnx.py \
  --export artifacts/continuation_v268/recovery_export \
  --output artifacts/continuation_v268/mujoco_recovery_diagnostic.json
```

该入口检查recovery policy_kind、ONNX hash、25D观测、关节顺序、输出侧PD及周期/腿范围，使用共享leg_pd与径向弹簧，拒绝行走contract。v268未学会模型四姿态实际各8秒、longest_ready全0，不能称起身成功。此处contact和终止保护尚未完全与Isaac等价，明确标为diagnostic/certified=false，不作为正式恢复验收器；v269当前训练尚未完成。

复位准备（第246条）：`audit_recovery_starts.py`生成四类前/后/两侧倒地初态，通过地面距离自动确定root高度，保留3mm间隙和原闭链/关节几何；`recovery_start_audit.json`记录qpos/qvel。`settle_recovery_starts.py`做每姿态无弹簧/每腿10.775N两种2秒无电机驱动落地，共8例，全部finite、闭链峰值<=1.502mm、主动腿峰值<=12.747rad/s，见`recovery_settle_audit.json`。部分轮仍自转，不能将所有状态称为静止；这是CPU物理初始化检查，没有训练复位、更没有用reset冒充起身。

第239条damping敏感性探针：新增--wheel-damping，单位Nm/(rad/s)。3Nm/每腿10.775N下原暂定区间0.005、0.0425、0.08分别在4.344秒失败、6秒失败、10秒存活但几乎不前进，见probes/damping_sensitivity.json。所有值仍未实测，未据此缩小或认证DR。Isaac迁移helper已支持独立动作尺度与3Nm上限（18项测试通过），旧target配置未自动改变，下一步可显式选5.5尺度避免突改已学单位。

第237条：`radial_spring.py`沿hip关节锚点到wheel轴心施加径向伸长力，根机体承受相反反力，以mj_applyFT作用于实际位置点，不增加关节或改变拓扑。`--radial-force 10.775`表示每腿10.775N，默认0用于旧策略对照。`verify_radial_spring.py`对训练函数512组力对照最大误差3.55e-15，MuJoCo广义力功率与腿长有限差分一致，根自由度净力接近零；报告radial_spring_verification.json。它仍是用户给定恒力的虚拟径向模型，不是已辨识的真实力—行程曲线。

在rmuc_usd_limits.xml/3Nm上限/旧5.5动作尺度中，开启每腿10.775N运行10秒，后半程速度/yaw误差0.258780/0.111332，实际轮力矩峰值1.35752/1.19864Nm。相对无弹簧0.211308/0.151626，速度变差而yaw改善，仍未通过。完整轨迹probes/v261_radial10775.json。未修改策略、训练USD或源MJCF。

第234条区分动作尺度和物理力矩上限：probe新增--wheel-action-scale、--wheel-effort-limit，保持训练的“feedforward先限幅→减显式阻尼→最终力矩限幅”顺序，逐物理步读取qfrc_actuator并检查轮端上限。三组同ONNX/初态/摩擦0.5的证据见probes/torque_migration_comparison.json（含复现命令）：5.5尺度+5.5限幅与5.5尺度+3限幅完整trace逐值相同，两轮实际最大1.34233/1.19844Nm，未触及限幅；3尺度+3限幅在1.984秒倾角45.85°停止，不能直接把旧ONNX乘3。保持5.5动作尺度并最终限幅3Nm不等于允许输出5.5Nm；该组合仍仅为诊断候选，未通过全技能、DR、最终气弹簧与有效damping验证。

第233条硬件口径：用户再次确认两台RM3508轮电机、四台DM8009腿电机（此前完整型号DM-J8009P-2EC，9:1）。USD轮joint drive maxForce实际为3Nm、四腿为40Nm；Isaac训练运行时轮IdealPD覆盖为5.5Nm，当前诊断probe也明确按旧5.5Nm运行。候选MJCF ctrlrange±3乘gear15.7647不是轮端±3Nm。最终每轮持续3Nm目标保持，四腿40Nm只是当前配置值，不冒充已确认持续扭矩。

`compare_initial_observation.py`读取v261实际Isaac reset记录，与真实MuJoCo probe首帧比较：同命令16个Isaac环境的25槽最大差1.1921e-7，仅投影重力浮点误差；同一ONNX输出最大差3.5763e-7。报告probes/initial_observation_comparison.json；只证明记录初态的obs/actor输入输出一致，不等于整段物理轨迹或实机估计器已对齐。

第232条扩展模型审计：在根坐标系比较39刚体位置、COM、完整惯量张量，最大分量误差分别5.87e-7m、5.87e-7m、3.69e-7kg m²；转动关节轴最大误差2.22e-16。只是初始配置及authored USD对照，不替代PhysX运行时属性读回。发现有据可查的摩擦差异：v261 final_turns.json碰撞形状读回0.5，训练地面配置0.5，候选XML轮地为1.1。新增--sliding-friction参数，在模型内存设置0.5后实际接触读回0.5；10秒实验probes/v261_friction05.json仍存活，但后半段速度/yaw误差增至0.212264/0.152709，不能为了成绩把1.1宣称为训练匹配值。后续继续接触求解与匹配初态诊断。

第231条四组10秒对照已实际执行，汇总 `probes/dynamics_ablation.json`，可用 `run_dynamics_ablation.py` 复现。基线速度/yaw误差0.157622/0.102968；2ms步长0.156987/0.103340；被动阻尼0.01为0.157502/0.102955；两者同时修改0.156872/0.103327。均跑满且未通过yaw门槛，不用这两项的微小变化解释主要缺口。补充源码/数值核查：Isaac root_lin_vel_w是root COM速度；MuJoCo mj_objectVelocity(BODY)与mj_jacBodyCom*qvel一致（误差<6e-17），不同于body原点速度，当前速度取点相符。

已新增 `run_mujoco_probe.py` 并完成真实 ONNX→MuJoCo 10秒诊断闭环：

```bash
/home/kellen/anaconda3/envs/robot/bin/python sim2real_RL/sim2sim/run_mujoco_probe.py \
  --xml /home/kellen/wheel_leg_sim2sim/assets/wheeled_infantry.xml \
  --seconds 10 --speed 1 --yaw -1.4 \
  --output sim2real_RL/sim2sim/probes/v261_turn_10s.json
```

运行达到10秒，最大倾角5.048度，最大腿速度2.557rad/s。后5秒平均前进速度0.84238m/s、yaw -1.50297rad/s；速度/yaw平均绝对误差0.15762/0.10297，未达到既有固定点yaw门槛，不认定通过。策略未改变，尚未加入ROS；该脚本用于定位动力差异。

诊断profile使用旧5.5Nm/0.1轮显式阻尼、腿60/2 PD、0.22位置尺度和40Nm限幅，按gear换算ctrl。模型内存中关闭六主动自由度原阻尼以避免与显式PD重复，腿armature设置0.01，原XML未改。被动阻尼、接触、17rad/s PhysX硬速度限制和reset等仍未全部对齐。输出完整记录profile、模型/ONNX hash、每策略步25D输入及6D动作，不是最高DR或最终3Nm验收。每策略步先mj_forward刷新派生状态再构造观测。

仓库根目录下 `artifacts/continuation_v261/export_final/` 提供完整 `policy.onnx`、`action_contract.json`、`export_parity.json`。它是研究候选，已通过八个标称固定转弯点，不是最高 DR 或 sim2sim 发布版本。暂不复制到正式发布模型路径，以免误用。
