# 腿电机速度与armature核查

用户确认四个腿电机为DM-J8009P-2EC，减速比9:1。型号/减速比已确认，具体输出轴极限转速和转子惯量尚未取得本型号实测或厂家数据。本次外部文档检索接口返回404，结论以本地资产、安装的Isaac Lab源码与原生读回为依据。

- RMUC YAML写velocity_limit_rad_s=60，asset factory只传IdealPD的velocity_limit。实际IdealPD.compute只按PD计算力矩并按effort裁剪，没有速度—扭矩包络。安装的ActuatorBaseCfg说明真正solver速度限制使用velocity_limit_sim。原生v225四个主动腿关节读回solver约1.7e4rad/s，故仅把60文本改17不足以形成实际硬限速。
- RMUC未显式写armature会继承USD；原生表读回四关节armature均0.010kg·m²，并非0。V14_2腿使用17及DM8009_ARMATURE=1.95e-4×9²=0.015795。这个值是原代码参考，不冒充本型号实测，也不再次给输出轴惯量乘减速比。
- armature直接加入关节空间惯量，影响瞬态加速度、振荡和既有PD闭环；不会凭空产生期望内倾。速度硬限幅也不等价于电机真实力矩—转速曲线，后续辨识仍需核对。

新增显式leg profile：nominal solver limit17、armature0.015795；暂定DR速度[12,17]rad/s、armature[0.008,0.024]kg·m²，均标工程先验。随机化随现有DR strength从nominal连续插值，每reset独立采样四电机并写入真实solver。零强度精确保持nominal且不消耗随机数。参考armature profile仍需显式选择。完成原生限速对照后，默认YAML已改真正solver限速17，并显式保留原armature0.010；腿DR默认随现有DR strength启用，零强度仍为17/0.010。USD、轮执行器和腿PD/力矩参数不改。历史无限速报告保留，旧DR条件不能冒充新联合DR验收。

asset factory现支持velocity_limit_sim_rad_s与armature_kg_m2显式键；也可在env config的leg_motors上明确设值。profile helper与诊断CLI负责完整应用，不能只改velocity_limit。训练使用profile同值的Hydra actuator overrides以及env.leg_actuator_dr_enabled=True时，需在新动力下重验，不把原证书带过来。

v226串行诊断：同v223 checkpoint、1m/s±1、seed71、每点4replicas。baseline、limit17、limit17+reference armature各1500步；最后128步全DR只检查腿参数实际写入。每physics compute统计腿速峰值、1200步后稳态峰值与超过17次数，保存solver读回。DR检查含其他已有DR，不作为“只改腿DR”的行为因果试验。最终结论待对应JSON完成。


## 已完成结果

基线每physics步6001次compute记录：四腿全程最大2.7784rad/s，1200控制步后稳态最大0.8952，超过17次数为0。实际solver旧限速17453.293，armature0.010。仅限速17后，初态与两命令的速度/yaw/bank/动作/关节位置统计逐值相同。因此需要修正限速配置，但没有证据支持它是这组压弯失败的直接原因。

参考armature0.015795读回正确，最高腿速2.7156。+1/−1的bank误差12.0734/9.1397°，基线11.9485/9.0482°；8回合均存活，未解决外倾。样本只够诊断，不替代完整任务验收。

128步DR写入探针实际读回：速度限值12.0291–16.9346rad/s，armature0.008412–0.023568，范围有效。该短探针没有完整回合，报告survival0是没有完成样本，不能说全部摔倒。默认17/0.010配置另做原生短验证。所有原始证据在artifacts/continuation_v226。
