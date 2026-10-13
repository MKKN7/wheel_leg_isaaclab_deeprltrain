# 电机与气弹簧参考核对（2026-09-22）
> 当前生效口径（用户于2026-10-01确认）：左右两侧`active_link_pitch`输出侧各施加`-10.775 N·m`等效辅助力矩，机身施加两侧合力的等大反作用力矩。该项是被动机械辅助，不进入策略action或腿部PD。训练、recovery export和MuJoCo recovery consumer必须使用同一语义。以下旧径向力/安装点讨论保留为历史诊断，不代表当前执行器契约。

> 历史口径（已被2026-10-01用户确认修正）：此前曾要求按小腿径向力及作用点/Jacobian建模，并将固定active-link力矩限制为play诊断；该段只解释旧实验为何不能外推，当前恢复训练与MuJoCo consumer按上方10.775 N·m等效力矩契约执行。

用户提供 `/home/kellen/wheel_leg_mjrl-lqr/wheeled_infantry.xml`，确认实机有气弹簧、轮电机RM3508、持续最大扭矩3Nm。参考文件只读，现有训练USD与物理配置未改。

| 项目 | 参考工程事实 | 当前RMUC与处理要求 |
|---|---|---|
| 轮电机 | XML motor ctrl/force ±3，gear=268/17≈15.7647 | 当前轮关节直接作用5.5Nm，不是同一扭矩口径 |
| 编译验证 | MuJoCo mj_forward输入3得到actuator_force3、轮关节qfrc_actuator47.294Nm，两侧一致 | 用户已确认3Nm是减速器后；XML重复放大，不导入47.294Nm |
| 轮轴阻尼 | XML 0.01 | 当前explicit actuator 0.1；参考模型参数并非实测辨识结果，需迁移验证 |
| 腿电机 | 注释DM-J8009P-2EC，每侧两个、gear1、±40Nm | 当前四个腿关节position PD、40Nm；保留6-D动作，不能直接改成另一套策略接口 |
| 气弹簧 | XML没有气弹簧弹性参数；lqr_deploy.py外加qfrc_applied，在两侧active_link_pitch上各-10.775Nm | 当前环境没有对应辅助力实现，电机PD刚度不能当作气弹簧 |
| 气弹簧变体 | warp_flat_controller配置开启，还按hip剩余合力矩裕量裁剪 | 不能仅抄常数，需明确采用哪种力学语义 |
| XML tendon | 仅测量髋轴到轮心腿长 | 不应误认成已生效的弹簧 |

气弹簧参考是恒定广义力矩近似，不是实际安装点/力臂/力—行程模型。后续应在现有USD闭链上加入明确的被动辅助项，先验证伸腿方向、力矩符号、虚功及motor/passive力矩是否重复限幅，再渐进迁移并复测全速度最高DR。气弹簧属于机械模型，不增加策略输出层。

证据：`artifacts/continuation_v160/reference_motor_spring_audit.json`，包含XML hash、MuJoCo版本、实际前向力矩探针。此前3Nm轮端动力包络结论保留，但适用前提明确为3Nm确实是轮端；用户已确认该轮端前提成立。

用户已确认减速器之后3Nm。若保留XML gear=15.7647，motor ctrlrange和forcerange应折算为约±0.1902985Nm；已在内存修正模型并执行mj_forward验证，正负方向两侧轮端均限到±3Nm。原XML与当前USD未修改。证据wheel_output_torque_conversion.json。

## 可选实现与实际验证

2026-09-22已在RMUC环境加入默认关闭的`gas_spring_assist_torque_nm`机械辅助。`gas_spring.py`验证USD关节轴/父子关系，对两个连杆施加负向力矩，对机身施加等大反作用力矩；使用逐子步瞬时wrench，不修改motor命令或策略维度。参考上限10.775Nm是当前近似模型边界，不是实机力—行程辨识。

45项相关测试通过；8环境/500步零命令探针无终止，400子步实际composer输出误差<2e-5Nm，净力矩误差约2.2e-5Nm。开关辅助的短时root高度差很小，不据此宣称稳定性提升。原训练进程没有启用气弹簧，下一次采用须单独明确迁移与验证。诊断报告在artifacts/continuation_v160/gas_spring_*probe.json，均不是行为通过证书。

纯径向近似现已有可测试几何函数`radial_spring_force_pair`：将正向伸长力沿髋到轮心轴线施于两端，净力/净力矩守恒，功率仅对应腿长变化、不对纯摆动做功。该函数尚未接入运行环境，未确定力值/DR范围；不能与已跑过的恒关节力矩诊断混为一谈。允许使用DR覆盖合理近似的不确定性，但须最终实测验证。

径向力函数已接入默认关闭的环境机械项：`gas_spring_radial_force_n`与`gas_spring_radial_dr_fraction`。作用点从USD读取，左右reset独立随机，DR强度依现有课程；不与恒关节力矩叠加。独立8环境500步fullDR探针（40.93N±50%实验假设）出现7次倾倒，不能直接作为正式训练/部署动力基线。该幅度不是实测范围。详见improvement第27条及radial_spring_dr_probe.json。
