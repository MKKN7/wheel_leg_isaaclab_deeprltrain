# 已交付H723小MLP v8：传感器、电机与网络I/O对齐契约

日期：2026-10-09。只适用于桌面RMUC_H723_smallMLP_v8_20261009/policy.onnx。
ONNX SHA256：b866a53ecdb186e3adb3db38c04e971ec9bbec1360605ffbb97dd86f7027713f。
网络25→128→128→6、20614参数，float32输入[1,25]、输出[1,6]。
本文件没有修改网络、控制器或训练。训练仍暂停。它说明软件/模型坐标，不等于实机电流/编码器标定完成。

## 1. 先统一物理朝向，不能直接按CAD的left/right名字接线

固定定义：人站在车后，朝策略规定的前进方向看；前F、左L、上U构成常规右手FLU坐标。
模型策略前方为robot刚体局部-Z；上方为局部+Y。因此模型局部+X位于物理右侧。
当前模型名字发生镜像：right_*位于上述物理左侧，left_*位于物理右侧。这里的“物理侧”相对于策略前方；必须先确认实机选定车头对应这一方向。
此结论来自MJCF初态实际关节锚点和轴的变换，不是电机安装符号猜测。

|六关节数组槽位|模型名|按策略前方定义的物理侧|正关节轴在FLU中|
|---|---|---|---|
|0|right_active_link_pitch|左|−L|
|1|right_hip_pitch|左|−L|
|2|right_wheel_spin|左|−L|
|3|left_active_link_pitch|右|−L|
|4|left_hip_pitch|右|+L|
|5|left_wheel_spin|右|−L|

所有joint_position、joint_velocity、action、history均用这个槽位顺序。四腿数组另取[0,1,3,4]。
从物理右侧向左看、车头在画面右边：槽0/1/2/3/5正转为逆时针，槽4正转为顺时针。
这描述关节输出轴，不是驱动器正电流或编码器计数正向。

## 2. 训练零位：CAD装配坐标q=0，不是竖直，也不是上电清零

训练资产initial_joint_pos与standing_joint_pos_profile均为全关节0。
观测减的是固定default_joint_pos；本模型四个受控腿关节q0=[0,0,0,0] rad。
MJCF初态qpos0的六个受控关节也均为0。角度0代表CAD已装配姿态，连杆本身已经倾斜。

active是“机身→主动短杆→long_link”的基座关节；hip是“机身→upper_leg→lower_leg→wheel”的基座关节。两者均主动控制。

模型零位的连杆中心线投影（只看F-U平面，起点/终点均取铰接轴中心）：

|杆件|起点→终点|前向差F|向上差U|零位几何角|
|---|---|---:|---:|---|
|active短杆|机身active轴→短杆与long_link连接轴|+69.234mm|−64.318mm|从向前水平线顺时针42.8919°，即向前下方|
|upper_leg上腿杆|机身hip轴→上腿与下腿连接轴|−153.840mm|−142.940mm|向后下方，与向后水平线夹42.8966°|

左右两侧侧视投影相同。不要用杆外轮廓、气弹簧端点或轮胎接地点替代这两个铰接中心。
推荐在断使能、可靠支撑下用装配基准/定位工装确定这两个相对机身角度，记录对应编码器原始读数。
若不能摆到零位，可在任意已知装配角q_cal处标定：q=q_cal+s_enc*(theta-theta_cal)/N。

也可以借侧视几何核对（角度均先连续展开）：
alpha=atan2(U_end-U_start,F_end-F_start)。
active两侧：q_active=alpha−(−42.891915°)。
模型right_hip：q_hip=alpha−(−137.103388°)。
模型left_hip：q_hip=−[alpha−(−137.103388°)]。
上述角差转换为rad。机身不水平时，F/U必须跟随机身，不能直接量相对地面角度。

重要：q0只是动作和观测的参考；零动作不是“站稳指令”，实际平衡时网络会输出非零动作。
闭链其余被动关节也必须满足相同装配闭合分支，不能只使两个电机编码器显示0就认为几何正确。

## 3. 每个电机需要填的硬件映射表

|槽位|实机电机/CAN ID|编码器零位theta0|s_enc=±1|电机转速/关节转速比N>0|s_tau=±1|每Nm输出端力矩对应电流或计数|
|---|---|---|---|---|---|---|
|0 right_active|待确认|待标定|待确认|待确认|待确认|待确认|
|1 right_hip|待确认|待标定|待确认|待确认|待确认|待确认|
|2 right_wheel|待确认|轮角零位任意，不输入轮位置|待确认|待确认|待确认|待确认|
|3 left_active|待确认|待标定|待确认|待确认|待确认|待确认|
|4 left_hip|待确认|待标定|待确认|待确认|待确认|待确认|
|5 left_wheel|待确认|轮角零位任意，不输入轮位置|待确认|待确认|待确认|待确认|

若theta是电机转子弧度：
q=s_enc*(unwrap(theta)-theta0)/N；dq=s_enc*omega_rotor/N。
若测输出端：N=1，不能再除一次减速比。
RPM转rad/s：omega=RPM*2*pi/60；度转rad：angle=degrees*pi/180。
单圈编码器跨零须连续展开；不能每周期对(q−q0)随意wrap而制造跳变。
正电流是否产生正q方向的力矩由s_tau确认，不应仅因为s_enc为正就盲目填写s_tau为正。
q、dq必须用同一方向约定；输出力矩也必须落到同一关节坐标。

## 4. IMU：先对齐常规FLU，再构造网络分量

设R_BS将传感器S坐标的向量旋转到机身FLU，omega_B=R_BS*omega_S。先转换单位rad/s。
本策略实际采用的三个分量是前/右/上，所以：
omega_policy=[omega_B.x, -omega_B.y, omega_B.z]。
此“前右上”分量排列不是常规右手姿态坐标，不能拿它直接建立标准Euler/quaternion旋转库；姿态解算保持右手坐标，最后只做分量映射。

|网络角速度|正向物理动作|
|---|---|
|omega_policy[0]|右侧下降、左侧上升（右倾）|
|omega_policy[1]|车头上升（抬头）|
|omega_policy[2]|左转，俯视逆时针|

重力是世界向下单位向量的机身投影。若R_WB为FLU机身到Z-up世界的旋转矩阵：
g_B=R_WB.T*[0,0,-1]；g_policy=[g_B.x,-g_B.y,g_B.z]。
若使用NED姿态库，先按库约定将世界重力和机身坐标转到一致坐标，不可原样套Z-up公式。
旋转矩阵方向、四元数wxyz/xyzw及安装旋转必须显式确认；不能给传感器四元数盲套Isaac的排列。

|纯姿态|g_policy|
|---|---|
|直立|[0,0,-1]|
|低头30°|[+.5,0,-.866025]|
|抬头30°|[−.5,0,-.866025]|
|右倾30°|[0,+.5,-.866025]|
|左倾30°|[0,−.5,-.866025]|

重力分量连续取值，模长约1，不是±1开关，不带9.81系数。运动时不能直接归一化加速度计替代姿态投影重力。

## 5. 25维观测的完整顺序

所有数值float32。q/dq均为上表模型关节坐标，角度rad、角速度rad/s。
v为沿车头在水平面投影方向的实际速度，m/s；c_v为原body前向命令m/s，c_w为左转正的yaw命令rad/s。
若有世界速度v_W：f_W=R_WB*[1,0,0]，v=dot(v_W.xy,f_W.xy)/norm(f_W.xy)。
轮速可辅助估计，但有打滑/转弯/腿运动时不等于真实车体速度；不能宣称直接轮速平均与训练观测严格等价。

|索引|精确构造|正号说明|
|---|---|---|
|0|.25*omega_policy[0]|右倾角速度|
|1|.25*omega_policy[1]|抬头角速度|
|2|.25*omega_policy[2]|左转角速度|
|3|g_policy[0]|低头时正|
|4|g_policy[1]|右倾时正|
|5|g_policy[2]|直立为−1|
|6|clip(2.1*(c_v−v),−2,2)|速度落后于命令时正|
|7|clip(.5*c_v,−2,2)|前进命令正|
|8|clip(c_w−omega_policy[2],−2,2)|需增加左转角速度时正|
|9|q[0]−q0[0]|模型right_active正关节方向|
|10|q[1]−q0[1]|模型right_hip正关节方向|
|11|q[3]−q0[3]|模型left_active正关节方向|
|12|q[4]−q0[4]|模型left_hip正关节方向|
|13|.05*dq[0]|同q[0]正向|
|14|.05*dq[1]|同q[1]正向|
|15|.05*dq[2]|轮子后退滚动方向正|
|16|.05*dq[3]|同q[3]正向|
|17|.05*dq[4]|同q[4]正向|
|18|.05*dq[5]|轮子后退滚动方向正|
|19..24|上一周期executed_action[0..5]|归一化动作，不是力矩/电流|

omega_policy[2]在索引8中必须使用未乘.25的rad/s值。输入没有轮位置，没有独立vy命令，没有直接Euler角输入。
0.25/0.05/2.1/0.5是观测定义的一部分，必须执行。ONNX只已折叠学生Normalizer，不能再次减mean/除std，也不能省略物理观测缩放。

## 6. 输出、动作历史与腿位置PD

ONNX输出原始action[0..5]，顺序见第1节。没有tanh兜底，必须经过交付包consume_action。
通常各槽clip到±1；当.8<c_v<1.2且abs(c_w)>.4501时，仅active槽0/3允许±(1+.15/.22)=±1.681818，其他仍±1。
精确参考使用obs[7]/.5及obs[2]/.25+obs[8]重建命令，与训练保持一致；边界按float32判断。
历史记录此消费后动作，不是输出电流，不是最终关节限位后的反算动作。初始history全0，每8ms决策后更新一次。

腿槽0/1/3/4：
q_target=clip(q0+leg_scale(c_v)*executed_action, joint_lower+.02, joint_upper−.02)。
tau_leg=clip(60*(q_target−q)−2*dq,−40,+40) Nm，关节输出端，目标角速度为0。
positive action提高模型q目标；不等于统一伸腿/缩腿。
模型限位active±2.60rad、hip±3.10rad，.02rad余量；实机安全限位若更紧会改变可执行策略，需要重新检查，不可擅自扩大实机限位。

leg_scale按abs(c_v)在线性插值，不能固定一直用.22：

|abs(vx_cmd) m/s|leg_scale rad/action|
|---:|---:|
|0.0|0.18|
|1.0|0.22|
|1.2|0.24|
|1.35|0.24|
|1.5|0.24|
|1.6|0.242|
|1.75|0.245|
|2.0|0.25|
|2.2|0.255|
|2.4|0.262|
|2.5|0.265|
|2.6|0.268|
|2.7|0.27|
|2.8|0.273|
|2.9|0.276|
|3.0|0.28|
|3.1|0.283|
|3.2|0.286|

例如vx_cmd=1、active action=.5时，q_target−q0=.11rad=6.3025°。

## 7. 轮直接力矩及实机电流符号

两轮的policy_action_sign均为−1，已在网络动作→模型关节力矩这一步使用。
部署包C参考默认建议D_comp=.052 Nm·s/rad，硬件参数结构体需显式填入，宏本身不会自动完成标定。
对槽2/5分别执行：

tau_drive_joint=−5.5*clip(executed_action,−1,+1)
tau_motor_joint=clip(tau_drive_joint−D_comp*dq_joint,−5.5,+5.5)
current_counts=s_tau*counts_per_output_Nm*tau_motor_joint

这是关节坐标写法：轮action正→前进驱动力矩，轮dq负→前进转动。
若固件采用“前进为正”的轮坐标，可令omega_forward=−dq_joint，tau_forward=−tau_motor_joint，等价写成：
tau_forward=clip(5.5*a−D_comp*omega_forward,±5.5)。
两种写法选一种且映射一致，禁止既使用负关节轴写法又额外重复取反。

检查例子（D_comp=.052）：
- dq_joint=0、a=+.2 → tau_motor_joint=−1.1Nm，轮向前驱动。
- dq_joint=−30rad/s、a=0 → tau_motor_joint=+1.56Nm，抵抗前转。
- dq_joint=−30、a=+1 → tau_motor_joint=−3.94Nm，仍前驱但阻尼占用预算。
- dq_joint=+30、a=+1 → 未限幅−7.06，限幅后−5.5Nm。

物理轮端最大瞬态峰值5.5Nm；3Nm只作持续热负载参考，不能将输出直接硬裁到±3。
训练名义轮模型为clip(−5.5*a−.05*dq,±5.5)−.002*dq−.001*tanh(dq/.1)。
实机机械黏性/库仑损耗由实体提供，部署代码不能再额外计算这两项电流“模拟机械”。
部署D_comp=.052与训练Dfb=.05+b=.002只是既有近似补偿路线，尤其饱和区不严格等价；不能把这个接口契约说成sim2real保证。

counts_per_output_Nm必须包含实际输出端力矩/电流关系及传动效率；5.5动作scale是轮端Nm，绝不能再把5.5乘减速比。
若用Kt(A→电机轴Nm)换算，电机电流I=tau_joint/(N*eta*Kt)，最后应用电流正方向；具体Kt/N/eta/驱动计数不在此臆造。
实机如果已有同功能速度阻尼反馈，只保留一个施加点，不能在上下位机重复扣D*omega。
腿同理：PD给的是输出关节Nm，接入电机电流/扭矩API前完成正确输出端换算；不要再叠一套未知内部位置PD。

## 8. 时间与执行顺序

- 网络推理/动作更新：8ms，125Hz。
- 腿PD：2ms，500Hz，两个腿PD时刻之间保持上次算出的目标和力矩。
- 部署参考轮执行：1ms，1kHz；每次用最新同源、同符号轮速度重算阻尼，8ms间保持policy动作。
- 训练物理/轮更新2ms；1ms部署参考尚未通过真实板测，不能称完全时序等价。
- IMU/编码器/速度估计按时间戳对齐；更高采样频率本身不解决不同滤波延迟或符号差异。

每8ms：读取对齐后的状态→构造25D(history为上周期)→ONNX→consume_action→提交控制器→history=executed_action。
不要在每1ms把history反复更新成力矩。传感器延迟不能通过复制历史动作槽替代。

## 9. 不开网络闭环时的逐项对齐验收

1. 身体朝向/左右：确认实机策略前方；根据连杆连接、CAD对应关系填6个槽位与CAN ID，特别确认模型right在物理左侧。
2. IMU静态：机身直立g≈[0,0,−1]；分别手动低头/抬头/左倾/右倾，核对第4节正负。静止时omega应接近0。
3. IMU动态：单独右倾/抬头/左转，各自对应obs0/1/2为正。pitch滤波符号不能凭姿态库字段名字猜。
4. 腿编码器：固定机身，逐关节沿第1节正方向移动约5°，对应q−q0应增加约.087266rad，运动中对应dq为正；其余槽不要串接。初始零位按铰接中心线或工装核对。
5. 轮编码器：支撑机身使轮离地，手转每轮向前滚动方向，两轮dq均为负；此时obs15/18也为负。
6. 力矩映射：禁用网络，可靠支撑并用受限小力矩逐轴确认正关节力矩方向，与正dq方向一致；轮a正的驱动力矩应向前。悬空轮不能用来证明整车平衡。
7. 腿PD方向：小正q_target扰动应让q增加，且dq正时D项产生负关节力矩；若反向，先修映射，不允许靠RL适应符号错误。
8. 单元对拍：用交付golden_100.npz中的observations直接输入CubeAI，与onnx数组比较；该数据已构造好25D，不要再对其中的角速度重复乘.25。这个对拍只验证推理，不验证传感器构造。
9. 整链离线回放：同一真实状态同时输入PC build_observation和MCU构造器，逐项比较25个数；再比较ONNX输出、consume后动作、PD目标/力矩及轮电流换算。
10. 符号、零偏、单位、周期都验证后才启用实机闭环。当前网络低速/转弯sim2sim仍有缺口，建议首测支撑/限域逐级推进，不能直接当最终鲁棒控制器。

## 10. 还需要实机人员明确填写，软件不能替你推断

IMU安装旋转R_BS、姿态四元数排列及方向；六个CAN ID和机械连接；四腿编码器零偏；六轴编码器正向和减速比；六轴电流正向与Nm换算；实机机械限位；轮速滤波及速度估计延迟。
现有交付包hardware_sign_verified/current_mapping_verified等字段为false，真实验证前保持false。

该小MLP训练弹簧阶段为2.424375Nm/腿，不能用现在教师的3.5附近阶段口径替换。它未获最高DR、最终弹簧、完整sim2sim、CubeAI板测或实机认证。

## 来源与本次核对范围

- 桌面交付包policy.onnx哈希、README、training_controller_contract_reference.json。
- 桌面reference_interface/rmuc_policy_interface.py与当前工作区同文件逐字一致。
- 桌面controller_reference/rmuc_low_level.c/.h与reference_interface/leg_pd.py。
- 工作区configs/rmuc_wheel_leg.yaml：initial_joint_pos/standing_joint_pos_profile全0。
- 工作区sim2real_RL/sim2sim/rmuc_peak55.xml：关节轴、初态铰接中心线与左右位置。
- 关节零位几何已通过MuJoCo静态mj_forward读取验证；不是实机机械测量。
