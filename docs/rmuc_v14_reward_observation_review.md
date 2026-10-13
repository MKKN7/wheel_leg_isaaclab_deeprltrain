# RMUC与原仓库V14观测、奖励、物理契约复盘
> 2026-09-26后续修复：下文“当前方向错误”描述的是v210/v211诊断时快照。第174条已统一bank_reference为world_heading_inward_v2，保留旧25D观测。v212同5084四点各64回合速度/yaw/生存精确不变，正确bank门槛全部拒绝外倾。策略尚未适应新物理目标，不能将代码修复当技能完成。


日期：2026-09-26。原仓库对照基线：本地HEAD b8ff79f（完整hash见v210/v14_source_provenance.json），origin为scutrobotlab/wheeled-legged_RL。读取了V14→V13→wheelbipe25_v3继承链；这些相关文件归一化换行后的工作区内容与HEAD一致。不是对网上最新版本的推测，也未修改原V14。

## 结论与优先级

当前主要问题不能简化为“奖励不够大”或“训练轮数不够”。先修坐标与验收语义，再完善观测/critic与探索，再做姿态和高度奖励消融。此前反复按固定点加seed虽有局部速度收益，却把原有符号假设当成正确物理，不能证明向更高速度/角速度扩展。v208已完整跑100updates/90策略学习；没有足够改善不是漏更新，冻结图、残差路径和导出均已核验。

**最优先：RMUC bank目标在当前资产坐标下鼓励向弯外倾。** 必须冻结“已通过物理压弯”的结论，保留历史报告作为旧指标结果；速度、生存、yaw原始测量仍有效。

## V14实际代码与RMUC的差异

| 项目 | 本地原V14 flat实现 | 当前RMUC |
|---|---|---|
| 继承 | V14 env继承V13，基础_get_observations/_get_rewards在wheelbipe25_v3/env.py | 独立RmucWheelLegEnv |
| 策略输入 | 原始3维command、height_cmd、gyro、gravity、关节位置/速度、actions，以及配置选定的task/mode扩展；基础维度28，具体变体可能加维和history | 固定25D：gyro3、gravity3、速度误差/目标vx/yaw误差3、腿位置4、关节速度6、执行actions6 |
| 横向状态 | 基础actor也没有直接root_lin_vel；不能说“V14 actor一直直接观测侧滑” | 没有直接vy、接触或真实COM；当前MLP也没有循环状态 |
| critic | 另加root_lin_vel_b三分量、obs_height，以及选定privileged扩展；有history/estimator变体 | 实际agent.yaml的actor和critic都读policy，25维一样，未保留上述特权状态 |
| 命令范围 | command分量clip100，直接目标，不会在2rad/s折叠 | yaw误差clip2；零gyro时2/2.5/3/10目标观测相同 |
| 轮动作 | V14Flat use_wheel_vel_control=True，目标轮速经限幅后送速度执行器 | 归一化轮力矩→5.5Nm动作契约、IdealPD阻尼，再叠速度/pitch/yaw外环 |
| 速度/yaw奖励 | 指数tracking与平方error并存；虽然名为track_lin_vel_xy，当前基础代码实际主要跟踪水平前向x | 多个速度误差、欠速、超速、进展、bank与yaw耦合项；额外细yaw资格目前仅按目标速度判断 |
| roll目标 | 使用projected_gravity_b的水平分量向直立约束，部分权重随速度变化；未发现解析atan(v*yaw/g)转弯bank目标 | 显式desired_bank=-atan(v_cmd*yaw_cmd/g)，但与自定义坐标映射组合后方向错误 |
| 高度 | height_cmd进入输入，tight exponential + squared height error；flat默认高度命令0.22，范围0.20–0.42，变体/地形可覆盖 | 按此前用户要求已去除0.4m/0.34m吸引，阈值0.15m以上无主动高度目标；不能把这个阈值当COM目标 |
| 奖励尺度 | 每项先乘step_dt，再加权求和 | 每控制步直接累计后clip[-10,20]；不能把V14原权重直接照搬，critic量纲也不同 |

定位：
- V14/env.py:26、530（继承和观测钩子）。
- wheelbipe25_v3/env.py:3100、3198、3230、3280（actor/critic拼接）；3474、3584、3707、3829、4002（奖励）。
- V14/env_cfg.py:468、568、838、869（flat开关、控制和奖励）；cfg_utils.py:25、76（基础维度、clip）。
- RMUC env.py:2156、2259、2364、2752（坐标、观测、奖励）；v208/params/agent.yaml:6（actor/critic输入组）。
这些是本地源码比较，未假称已运行V14原训练或获得其性能证书。

## 已执行核验

### 1. 坐标与压弯目标方向错误

RMUC canonical映射为[-source_z, source_x, source_y]，变换行列式−1，属于左手坐标排列。这种约定本身可保留，但不能直接套用另一套右手roll/gyro符号公式。

使用当前env._state()及安装IsaacLab的quat_mul/quat_apply/quat_apply_inverse原函数，按资产初始+90°X姿态做世界几何核验：heading世界+Y，正世界yaw应向世界−X内倾。

- 真正向弯内倾5.8204°：当前reported roll为+5.8204°，目标−5.8204°，错误罚为11.6409°。
- 向弯外倾5.8204°：reported roll为−5.8204°，当前误差接近0。
- 负yaw同样反向。

证据：v210/audit_frame_bank_sign.py、frame_bank_sign.json。此前v177“正确bank奖励更高”的测试生成状态时直接沿用−atan目标，只有奖励内部一致性，缺少独立世界几何oracle；不能作为物理正确性证据。

v211后续原生8replicas/1500step确认（相同5084）：+1方向world heading-left倾角−6.1805°，−1方向+3.3069°；乘实际世界yaw方向后的physical_inward_lean分别−6.1805°/−3.3069°，均为向弯外。实际初始四元数[0.7071068,0,0,0.7071068]也与几何审计假设一致。证据v211/turn_dynamics.json，诊断小样本只验证物理符号，不认证技能。

修复方向：保留旧25D数值及原checkpoint，不全局重排观测；以world heading-left和body-up明确测量物理内倾，修正reward/gate/evaluator三处同一契约，并标记迁移。不能把改评价门槛后的通过冒充旧策略学会了内倾，也不能直接将旧bank reward翻大。

### 2. 侧滑指标混入正常转弯轮速差

旧evaluate_rmuc_grid的steady_wheel_slip_speed_mps比较每轮速度与root中心前向速度，正常内外轮的v±yaw*半轮距差也被计入。纯滚动合成测试：轮速0.8/1.2、车体1.0、半轮距0.2、yaw1，旧值0.2，按各轮中心速度比较则0。

v210原生同5084/1m/s：
- +1rad：旧指标0.2597 m/s；每轮局部纵向比较0.0487 m/s。
- −1rad：旧指标0.2274；局部比较0.0190。
- heading侧向速度绝对均值约0.0148/0.0123 m/s。

因此不能把0.26当成实际侧滑严重并直接加大惩罚。新量仍是root-heading投影的轮中心纵向滚动误差，并非精确轮接触点轮胎模型。当前reward有body-vy penalty，但无直接轮纵向滑移罚；日志中的slip量不等于已进入奖励。

### 3. 真实COM比root高约8.6cm

新诊断读body_com_pos_w和body_mass做质量加权，没有用root代替COM。
- 1m/s、+1：steady root约0.3170m，COM约0.4034m。
- 1m/s、−1：root约0.3116m，COM约0.3982m。
- 2.5m/s、±1：root约0.302–0.304m，COM约0.389m。

root_mean旧报告是全步平均，与新steady值范围不同，不能相减当运动变化。上述COM/root对比使用相同steady样本。

2.5/−1的弱轮normal-load fraction约0.156，明显小于1m/s时约0.37–0.38，提示高侧向加速度下要关注卸载与支撑裕量。当前错误外倾会使该问题更值得优先修复。姿态与实际COM应配合，但不能无条件奖励降高或要求静态左右载荷各50%。

重要勘误：ContactSensor.net_forces_w在当前安装版本中是NORMAL force，非含切向摩擦的总力。v210首份JSON字段contact_tangential_normal_ratio名称不准确，禁止将其解读为摩擦利用率。当前代码已改名normal_force_horizontal_vertical_ratio；要测真实摩擦需另启track_friction_forces并验证contact过滤。旧报告保留，解释以此勘误为准。

### 4. 25D观测确有信息损失，但不是所有问题都由它导致

执行实际_get_observations AST证明：
- 相同其余状态、vy=0与0.6m/s，25D观测完全相同。
- 零gyro下yaw目标2、2.5、3、10均解码成2。
- 当前1rad初始误差没有触及yaw clip；不能用大yaw裁剪解释当前1rad失败。
- gyro_z是body投影，目标和gate也沿用此值，不能直接等同世界heading yaw rate；随大倾角差别会增长。
- 当前DR0没有传感器噪声/延迟，不能把本次nominal失败归咎DR延迟。

证据：observation_aliasing.json。修复可优先使用训练期privileged critic补速度三分量、COM/载荷，不改部署actor25D；这能改善critic可见性，但不能声称自动解决actor侧滑不可观测。actor若采用历史估计器需要明确时序/导出/部署状态契约，不能暗改槽位破坏旧图。

### 5. 原怠速/yaw策略未完整适配“低速大转弯”

用v208实际params/env.yaml标量和真实reward AST重新审计：
- 1m/s的额外turn_yaw_tracking_active=0，bank scale=1；2.5m/s对应1和2。配置写bank scale2不代表所有命令实际使用2。
- yaw>1时command_yaw_overspeed_relative_to_target仍False，合法2.5rad/s也被旧绝对1rad/s阈值罚；此项对未来大yaw与自转不一致。
- body-vy=0.2m/s只增加0.07的单步罚；这是权重事实，尚不能由此决定应增大多少。
- root0.32→0.20且其他量不变，奖励差0：取消旧高度吸引没有新增主动降COM机制。
- 当前bank“目标姿态”确实获得更高reward，但目标物理方向错，这比权重不足更优先。
- 原高速度怠速项有速度资格门槛；1m/s仍有通用speed tracking/underspeed项，不能说完全没有速度引导。实际AST中0.8→0.81的速度改善能增加reward。

### 6. 学习/探索结构也限制了新技能

v208将全部旧actor冻结，只允许一个小门控残差学习；旧left-leg action3 steady均值约1.081，执行clip为1。它处在已有动作饱和盆地，新层不能被当作从头训练的大范围姿态控制器。

checkpoint的腿std0.004、轮std约0.00614，且freeze_action_std。实际诊断有效轮偏置约0.05–0.06，即单步约8–10个轮std；这不证明PPO绝对学不到，也不能将稀有单步概率等同学习概率，但说明“给了新分支和100轮”并不等于充分探索到持续修正。高频独立噪声对需要持续动作/姿态变化的任务可能效率低。

最终value loss约4941，旧MC审计曾确认critic偏差；本轮没有同样MC校准，不能仅凭value loss大小断言critic坏了。V14的privileged critic和按dt缩放奖励都没有等价保留，值得在修正坐标契约后优先对齐。上限clip本轮末高clip比例0，不能再把此前clip10问题当作当前主因。

## 后续顺序

1. 先修正物理bank方向，统一world几何验证、reward与gate；保留旧checkpoint和旧指标报告，重新定义物理压弯证据，不热改旧25D输入。
2. 修正大yaw合法命令罚项与范围，并使新增低速大转弯获得适当fine-yaw资格；明确命令局部适用范围，不擅自改所有旧技能奖励。
3. 建立可用的privileged critic/学习信号审计；在新分支域内给足探索和既定训练预算，保留旧参数隔离。
4. 使用真实COM、heading侧速、各轮局部滚动误差与normal载荷做受控消融。必要时对新技能加入平滑、与实际横向加速度和接触裕量关联的高度/倾角引导，避免瞬间要求命令终点倾角，避免把正常转弯运动罚零。
5. 重验同完整图多seed、连续命令、最终3Nm/径向弹簧/最高联合DR。当前所有旧bank“通过”只代表旧符号gate，不能直接当物理向内压弯已掌握。

当前只增加了诊断和证据，未改训练reward、部署25D、USD、马达或已学策略权重。原V14源码保持原样。
