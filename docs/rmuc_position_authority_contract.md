# 局部位置余量原型契约

状态：v234已接入完整actor可选分支、checkpoint重建和完整图导出；环境/训练执行链尚未接入，未启动PPO。禁止仅挂上head就训练。

依据：v231左active动作长期饱和而力矩未满；v232额外关节目标使bank改善但损失速度/yaw。需要可学习的腿轮协同，不能将常数诊断偏移保留作外环。

原型输入原25D和完整旧图raw6D输出，在正向速度0.8..1.2、yaw幅值>0.4501的平滑支持域内，以旧clip[-1,1]的执行动作为基底，再加双向独立新分支修正。active legs槽0/3预算0.15/0.22动作单位，轮槽2/5预算0.06，hip槽1/4为0。轮和hip最终限幅仍1；active局部最大1+0.15/0.22，因此当前1m/s/scale0.22时目标绝对范围可到0.37rad。实际joint limit margin始终保留。

域外head保留旧raw值且consumer用旧clip。域内零分支时确定性执行动作精确等于旧图，避免旧超限值因放宽clip突然生效。该结构不是新任务级外环，最终若采用必须包含在完整策略导出及明确的执行映射中。

已验证：真实5284策略1131固定探针执行动作逐值相同，1061域外raw逐值相同；head trace batch1/7/1131一致。3项测试覆盖边界、正负分支隔离、饱和处新参数非零梯度。

待完成：
- checkpoint显式重建与完整图JIT/ONNX导出已完成（v234），含零/非零分支测试；仍需原生consumer验证。
- RSL wrapper统一clip=1会吞掉新增余量，必须与env逐动作/局部限幅一并迁移，外域保持旧契约。
- previous-action保持实际执行的legacy action units（位置单位换算不变），不能暗改25D槽位。
- 当前actor mean局部先clip再加head会改变随机策略分布；零初始化只证明确定性执行一致，不能声称PPO采样/logprob相同。新训练必须reset optimizer、校准critic并按实际采样分布记logprob，禁止将限幅后动作当原Gaussian样本。
- 原生同初态零初始化对照、带学习分支的真实控制边界读回后，才进入完整预算训练。
- 0.8..1.2速度的原leg_scale约0.22但不能只假设所有时序/课程均为0.22；明确验证命令过渡时的实际target上界。

已生成zero_authority.pt候选，保留旧图573张量，1131状态确定性执行一致，1061域外raw一致。另存diagnostic_nonzero_head.pt仅用于验证导出路径，不是训练策略。两者导出均JIT误差0/ONNX最大2.38e−7，action_contract.json明确新局部限幅与未验证状态。旧统一clip评估器暂时拒绝新head，避免静默吞掉新增位置余量。

更新v235：eval和env显式consumer已接通，缓存实际actor观测决定限幅；DR0双向各4完整回合零head原生行为与物理目标/力矩逐值相同。生产trainer尚未接通，仍不得启动PPO。非零head物理读回、随机分布与延迟场景仍待验收。

## v236更新：非零余量与PPO样本语义通过

人工非零head原生目标到0.365rad，证明执行链未吞余量；不是行为认证。CPU执行真实PPO.act/仓库credit wrapper，边界采样仍保留raw logprob且ratio1。6测试通过。待接trainer新head优化器/consumer及原生少量更新验证；未启动长训练，无后台任务。见improvement第198条及artifacts/continuation_v236/summary.json。

2026-09-27 v237/v238更新：trainer consumer/新head optimizer已接通，3次原生更新旧图隔离与导出通过。正式100次预算已启动，运行/结果见v238；此前“trainer尚未接通”是历史阶段说明。技能、随机DR及部署依然未验收。
