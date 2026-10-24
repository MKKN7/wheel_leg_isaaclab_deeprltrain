# 可命令变腿长：单一策略接口与训练设计（2026-10-22，尚未训练）

用户明确：需要下发高低腿长目标；能力在当前教师接续、最终同一小MLP/ONNX中完成，不另做腿长策略或FSM。新接口版本必须独立命名，旧25D ONNX不修改。不增加vy导航命令。

## 接口

候选v2为26D observation / 6D action。索引0–24及其normalization、history含义全部不变，索引25追加 `(L_cmd_m-0.34)/0.10`，float32。L_cmd为左右髋轴至轮轴在腿运动平面内距离的平均值；不是root z，不是单腿的世界竖直高度。前后向和压弯命令保留旧槽。命令范围随课程写入模型manifest，越界拒绝/上层显式处理，不静默裁剪成另一个目标。

高低开关先生成连续L_cmd；限速器属于命令整形、训练和部署必须同构，不是腿长反馈外环。策略输入必须包含实际送入策略的目标，不能只在reward偷偷平滑。速度和yaw阶跃不因此自动平滑。最终也须训练/评估高低直接切换；先小幅切换再扩大幅度。

raw root z不加入actor以避免依赖实机缺失的绝对高度；训练使用root相对地面高度、底盘最低点、轮支撑、整机质量加权COM等作privileged奖励/诊断。现有env stand_metrics使用root至轮轴距离，必须先与髋轴参考点转换后才可用作L测量，不能直接冒名替换。

## 无变化迁移与同一ONNX

源候选adapt36 SHA cb2c3e972b6f87b5e277a55a998b12d3164d8c68a0101412c1e982e8cd14f733（不是已认证部署策略）。现有25D组合actor冻结并仅接收obs[:25]，新增26D条件化残差末层零初始化；不是简单把所有专家硬塞26维。最初对任意合法L命令必须输出逐值等于旧actor；loader/export/normalizer分别验证。原25维normalizer不重估，新维用上述显式scale，不可受旧running stats污染。训练后全域、各腿长和切换一起采数，蒸馏为单个26→hidden→hidden→6学生；旧25输入CubeAI工程不能直接复用，重做Analyze和IO对拍。

6D仍腿位置目标+轮直接力矩，PD60/2，轮最终4.89Nm/动作scale5.5。第一阶段保留固定零位和原consumer，不偷偷加IK位置外环。若低腿长目标被action范围卡住，单独版本化consumer扩展或目标基准迁移，确保在L=.34处零变化，训练/部署/ONNX契约一起更新，不能只在部署平移q0。

## 可达范围证据

传统源码l_xx限0.16–0.36m，RL接管升到.35；这是传统控制范围，不是RL已具备范围。对传统l1=l4=.21,l2=l3=.25,l5=0直立支路的解析逆解：.34对应训练零位，.30在低速scale.19下需要abs(action)约.759，.28约1.122，.16约3.375。只说明名义零位/consumer余量，未认证USD闭链/关节限位/碰撞/电机负载。初始候选长度课程 .34→.33→.32→.30 与 .34→.35→.36，范围须先通过USD几何/力矩可达扫描，不能先承诺.16。

## 奖励设计

保留速度/yaw/存活/直行角差奖励与现有DR。增加平均腿长跟踪cost：`sqrt(1+((.5*(L_l+L_r)-L_cmd)/.02)^2)-1`；.02m为初始shaping尺度，不是验收放宽。不得把左右腿分别强拉到同一长度：压弯需要差动。直行/待机继续原active/hip角差约束，转弯不套并腿惩罚。

root高度只做单侧净空不足惩罚及有限弱姿态辅助，不奖励无限降低z。机身俯仰、侧翻、轮离地、蹭地不能伪装成缩腿达标。安全接触/净空/关节限位单独gate；量化真实COM高度，不能直接声称root下降=COM同比下降或高速必然更稳。

加有限竖向速度/腿长速度、关节速度与action变化正则，防止反复蹲起/谐振；这些正则不可让静止不响应比跟踪L_cmd更优。切换时保持非零误差梯度，不用超窄exp奖励导致远处全零。先做姿态/速度/长度reward排序、权重贡献与clip比例审计，避免现有reward总clip[-25,20]再压平新项；不随意加大weight。不得reward上升就晋级。

L_cmd=.34是旧技能rehearsal锚点，但不能假称旧策略本来严格跟踪.34；须实测旧腿长分布。扩长度期间保留每个已通过长度与速度域的回放，按“长度×vx×yaw”分层采样，避免高频简单点淹没难点。

## 课程与验收

先验证名义长度几何→站立/低速小幅切换→每一长度复制已掌握静态速度/双向压弯域→该长度内平滑变速/起停→行驶与压弯中高低切换→随机L/vx/yaw组合与多seed。每新增长度都要保留旧长度域，不允许只训低腿长快跑或某个速度。最终目标域继续包含2.0常驻/3.2冲刺及原yaw包络，但当前教师未掌握点列为能力缺口，不从历史拼接认证。

既有物理/DR保持当前真实契约：Dfb L/R .0494515/.05133043，机械b .0025485/.00066957，C .054699/.052843，Cff .053699/.051843；弹簧阶段4.31Nm/腿，radial0；DR=.1，质量/独立惯量各±2%，COM±1/1/.5mm，地胶迁移progress.1，轮transport0/2ms，motor tau .2–3ms分开。新增长度不是同时升级DR或弹簧幅值。弹簧真实分配差异尚未解决，不把此专项称sim2real认证。

沿用速度/yaw门槛；直行摆角strict2deg，用户授权近线<=2.1deg单列waiver；阶跃按已有放宽档，平滑跟踪不放宽。每长度至少32副本、多seed、每格有效10秒；圆弧偏差<.1m、原安全/接触检查继续。L跟踪的建议门槛先列为候选：稳态MAE<=.01m、P95<=.015m，切换响应/超调需结合净空与执行器可达性定标，不宣称用户已批准或已通过。必须报告实际L轨迹及root/COM高度，不能仅看root z。

## 当前状态

已实现独立CPU接口原型scripts/rmuc_leg_length_command.py与6项测试；尚未接入现行env、policy、consumer或部署，未启动26D训练。几何审计artifacts/leg_length_command_design_20261022/legacy_reachability.json。接下来做USD几何可达/离地检查，再接入独立版本26D任务；保留现有25D训练与部署资产。


### 资产几何复核补记
当前USD读回hip平面原点相对root仅约28微米，参考点差异在此资产很小；定义仍显式使用hip pivot。闭链MJCF几何交叉检查26点均可解且在关节限位内，但不是USD接触认证；.30需要低速action约.817、.29约1.001、.28约1.183。初始.30–.36保留，.34不强称资产精确零位，未扩consumer。见artifacts/leg_length_command_design_20261022/README.md。

### 26D教师组件实现
scripts/rsl_rl/rmuc_leg_length_actor.py提供独立deterministic wrapper：冻结完整旧actor及normalizer，source仅读前25维，26D残差零初始化、显式预算、eval固定旧running stats；它不是PPO actor API。12项CPU测试通过；真实adapt36的4096输入初始输出逐值不变且保存重载一致，证据teacher_migration_audit.json。teacher26_initial_component.pt明确为组件，不得作为train.py既有checkpoint或CubeAI部署文件。候选残差预算腿.2/轮.03尚需课程可达与retention验证，不代表已能覆盖全部腿长。

### 独立26D环境原型
新增leg_length_env.py与leg_length_terms.py，未注册Gym task。显式set_leg_length_command、原25D后追加目标、USD读取root侧hip anchors后测量平面腿长；原25D运行不变。奖励权重默认0；原型采用base已clip reward后追加cost再clip，此顺序须在正式非零奖励前完成排序/饱和审计，不能认为已完成reward设计验证。whole-observation history depth>0明确拒绝，当前2ms轮transport与此不同。15项CPU测试通过（参考点/横向轮距排除、bank差动、梯度方向、接口及冻结），仅py_compile验证环境文件，未宣称Isaac实例化/接触/训练通过。

### 奖励与PPO接口接入补记
低腿长奖励改为env._additional_reward_before_clip扩展点，在base所有项汇总后、统一clip前添加；旧25D默认hook为同tensor identity，取消原型double-clip顺序。22项CPU测试过。rmuc_leg_length_ppo.py新增显式26D RSL MLPModel actor，完整教师冻结，mean为25D教师+26D残差，独立Gaussian探索参数冻结；旧normalizer不改变，禁止通用25D导出接口误导出。真实教师初始parity+合成PPO ratio更新/冻结/KL验证通过，ppo_actor_audit.json保存。这不等于真实PPO训练启动；critic迁移、runner创建、rollout存储与真实多步优化、USD净空负载验证仍待接通。
