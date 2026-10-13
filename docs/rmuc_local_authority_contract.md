# 失败命令局部修正原型

最新扩展（第218条）：v253从v2515684追加±1.4中心，索引3/4；旧中心及专家0/1/2保留。center向量唯一原buffer变化为尾部追加，原前缀不变，half_width及总budget保持。严格恢复按保存center数量构造，旧三专家checkpoint仍能恢复。zero_extended.pt及人工非零扩展已通过完整导出与原生同场景对照，六保护命令完整结果不变；尚未新训练，下一步仅选择indices3/4。原生接线每点4回合不等于新技能验收。

最新（第214条）：v247完成后五点在seed67/75均通过，仅-0.8未过；v251从该5584仅训练索引0，显式--local_authority_train_indices 0。optimizer持久化索引列表并校验恢复来源；源图和专家1/2冻结。v250短验证只改变专家0的20张量、200保护观测输出相同、导出通过。v251正式100预算仍在进行，五点通过证据不等于全DR/动态指令认证。下方v243/v245是本模块历史起点。

状态（第210条）：actor恢复/导出/原生对照及独立optimizer接线通过。v246原生3更新（1critic、2策略）只改变局部专家60张量，整个源图不变、训练后完整导出通过；v247已从零局部候选重新启动正式100更新预算。源候选为v243完整5484，sha d310f4dd4d5bf006d7eb0338e01671ee70b9adb4b9f184f6fcef145d1bb3012e。

v245零候选zero_local.pt sha01088f29f94a3cf4a6fbaee6682e8103332d107cf7f89bcddf0049aa5ce0576d，源615张量保留，150点CPU完整actor严格恢复一致；人工非零模块在132个支持域外样本输出与源逐值一致。零/非零JIT、ONNX完整导出通过。原生seed75六点各8回合：零候选所有行逐值等于v244源策略；人工非零仅改变三个目标命令，+0.8/+1/-1.2完整行（含物理极值）仍逐值一致。证据artifacts/continuation_v245/{initialization,native_comparison}.json及导出报告；人工非零文件仅诊断，不作训练起点。

v243/v244两个seed一致：1m/s的+0.8/+1/-1.2通过当前跟踪检查，-0.8/-1过度转向，+1.2速度不足。继续更新共享positive/negative网络会影响同方向其他命令，故原型准备冻结整个源图，在失败命令邻域内增加可学习修正。单seed局部成功仍不等于完成全部技能。

`scripts/rsl_rl/rmuc_local_authority.py`包含三个零输出初始化专家，中心yaw为[-0.8,-1.0,+1.2]rad/s。以交付的25D观测重建command，速度沿用原authority的0.8–1.2m/s平滑支持域；每个yaw中心使用半宽0.15rad/s的平滑紧支撑。这是用于隔离命令的网络结构选择，不是物理安全阈值。源策略输出与可学习补偿仍构成一张策略图，不加入速度/yaw控制律。

修正作用在源authority correction上，合计裁剪到原budget，active腿位置余量、wheel预算、hip零余量及最终6D动作/consumer限制均不扩大。+0.8/+1/-1.2处所有局部gate为零，模块逐值返回源correction；这个保证针对实际交付观测重建的命令，不能扩大解释成观测噪声、连续命令过渡或整场景轨迹已验证。

独立训练使用--train_position_authority --train_local_authority，optimizer组为local_authority/critic，只解冻local experts；缺局部checkpoint或误用共享optimizer会拒绝。审计每更新检查源参数与全部buffer不变、实际合计correction不超预算，记录三个局部gate采样数。无样本专家在step前清grad，CPU验证先产生Adam动量再离开支持域仍无漂移。12项相关测试通过。v246三个专家都实际更新，192个保护命令观测的完整actor输出保持逐值一致。PPO原样本与consumer路径未改。v247结束后仍需独立保护点/失败点、多seed及动态指令验收，原生短更新不是技能通过证明。
