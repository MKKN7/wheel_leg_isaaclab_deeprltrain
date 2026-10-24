# 每批训练的桌面曲线与双腿不对称审计

2026-10-13用户要求：每批末期速度tracking和双腿摆角差图保存桌面，优先解决实机明显的前后不对称。后续实验生成时必须：

1. evaluation_command.json加入`--record-leg-tracking`；不重复GPU评估，仅搭载既有回归采数。
2. report.py之后调用`python scripts/rmuc_finish_training_charts.py --experiment <实验目录>`，CPU读取TensorBoard末三分之一真实训练趋势，以及最终checkpoint评估的末6秒物理轨迹；桌面目录为`/home/kellen/桌面/RMUC训练曲线/<实验名>`，禁止覆盖旧结果。
3. 三张PNG分别为真实训练末期EMA、最终策略速度轨迹、最终策略两侧连杆差；manifest带checkpoint/trace hash、所有条件统计。没有训练中关节轨迹不能把评估图冒称训练末期腿曲线；下一版采集须明确补训练内真实joint时序。

当前物理侧定义见h723_v8_hardware_io_alignment.md：模型right为物理左。active同轴方向差为q[0]-q[3]，hip镜像方向差为q[1]+q[4]，rad转deg。不直接减两个hip原始编码器；图显示绕机身轴的连杆姿态差，不是机身到轮心的虚拟腿角。后续需同步根部/轮心几何，验证整腿一前一后。

本次strict_crawl_negative01_20261013实际仿真直行稳态差：vx=.5时active−9.831°/hip−9.776°；vx1时−8.707°/−8.702°；vx1.5时−7.680°/−7.802°。这是源数据中的持续不对称，不能全归因实机符号。与实机约8°量级接近不等于同一根因。转弯应结合载荷/内倾，不强迫所有动作差=0；直行检查持续前后错位、限幅、姿态、欠速；另外补同幅值倒车对照。

默认门槛仍严格，图像不能替代32回合、多seed、动态切换、世界轨迹、地胶及sim2real认证。当前腿谐振/异常瞬态未定量认证。
