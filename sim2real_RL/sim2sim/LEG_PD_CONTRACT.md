# RMUC 腿部 PD 一致性契约

用户要求训练、MuJoCo、实机使用相同腿部PD律。当前参考实现为`leg_pd.py`，对应Isaac显式IdealPDActuator，禁止根据仿真表现另调一组部署增益。

## 当前契约

四腿关节采用减速器输出侧rad、rad/s、Nm。策略中的腿顺序为右active、右hip、左active、左hip（6D动作槽0/1/3/4）。电机实际通信顺序、方向和零位必须通过显式映射处理，不能只按数组长度接线。

```
q_des = clip(q_default + leg_scale * action_leg,
             joint_lower + 0.02, joint_upper - 0.02)
tau_computed = 60 * (q_des - q) + 2 * (qd_des - qd) + tau_ff
tau_output = clip(tau_computed, -40, 40)   # output-shaft Nm
```

- 当前位置测量误差直接用关节角差，不增加wrap、积分、速度环或位置差分求导。
- `qd`是训练所用语义的关节速度；当前目标速度`qd_des=0`，前馈`tau_ff=0`。不能擅自把位置目标的差分当目标速度。
- Kp=60Nm/rad，Kd=2Nm/(rad/s)，40Nm是当前仿真限制，未确认为电机持续额定扭矩。
- 当前1m/s转弯probe的leg_scale=0.22rad/action；完整课程含其他尺度，不得把该值外推为全部技能常数。authority允许对应active槽扩展至约±1.6818，先按既有consumer执行，再映射位置及物理限位。
- 默认对齐模型物理/PD更新周期2ms（500Hz），策略8ms（125Hz），期间保持动作目标，PD每2ms使用新状态重算。旧1ms诊断模型不属于周期已对齐的证据。
- 输出侧Nm到电流/驱动协议值仅在硬件适配层转换。9:1不能在已是关节侧的目标或增益上再次乘除。若驱动字段使用电机侧单位，必须先完成明确换算。
- 只执行一份PD。若下位机实现此律，上位机传相应目标；若上位机计算力矩，下位机不得再叠加未经建模的位置/速度闭环。驱动内部滤波、速度估计、限幅、延迟和电流响应仍需确认。
- 气弹簧属于外部被动力，不加入电机PD前馈或重复计入40Nm限幅。电机输出限幅不等于总关节受力限制。

## 已验证与未验证

`verify_leg_pd.py`直接提取本机Isaac IdealPD.compute和ActuatorBase._clip_effort源码AST，对1024组四关节输入（包含正负饱和、非零目标速度/前馈及当前零值模式）验证，最大误差0；源hash与结果在`leg_pd_verification.json`。MuJoCo runner已调用同一参考实现。

当前未连接DM8009下位机，故**实机PD一致性未验证**。实机交付前必须以相同输入记录对比输出力矩/电流、实际周期、角速度滤波和限幅行为；仅配置相同Kp/Kd不能替代验证。若硬件实际限制与仿真40Nm或17rad/s不同，更新共同动力契约并复验，不在部署端静默修改。
