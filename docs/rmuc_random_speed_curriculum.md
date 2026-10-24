# 回合内随机速度课程（2026-10-15）

状态：默认关闭，34项相关CPU测试通过；尚未GPU验证随机模式。当前姿态奖励训练不变。固定起停通过后首档0–0.5m/s，再逐步覆盖0–1、0–1.5及高速度域；不自动晋级。

使用原vx/yaw/25D/6D接口，当前随机模式仅直行vx，vy/yaw保持0。动态cohort沿用command_continuous_sampling_fraction，75%旧技能rehearsal及固定环境不变。课程区间以外的旧技能由rehearsal保留，不宣称随机模式已覆盖最终全域。

每次reset只给被选动态环境生成新序列；同seed可复现。回合内读取无随机数、无状态推进；观测/动作尺度/奖励按相同episode-age取值。保持时间2–4s，50%阶跃、50%0.4–0.8s斜坡，斜坡之后仍留稳态平台。目标在可达区间均匀采样并加入两端锚点；相邻目标差≤1.5m/s。不是全域严格均匀分布，需检查实际曝光。reset初态目标不是回合内阶跃，不用它证明响应时间。

首档配置：
```
env.command_training_sequence=[]
env.command_random_speed_range=[0.0,0.5]
env.command_random_hold_s=[2.0,4.0]
env.command_random_ramp_s=[0.4,0.8]
env.command_random_step_fraction=0.5
env.command_random_max_delta_mps=0.5
```

固定sequence与随机sequence互斥，禁止play模式、混合turn box或acquisition ramp，必须command一致feedback/reward。预检检查范围内65个速度的可训练gate，实际执行仍须GPU时序/cohort/奖励/参数读回审计。

验收三套证据独立：固定点恒速retention；固定阶跃（≤1.5m/s，急停≤1.5→0）；独立seed随机轨迹。阶跃到达<0.5s、超调<20%不放宽；斜坡跟踪单独报告，不能当阶跃通过。保留整个速度/角差瞬态与reset断点，不用平台片段掩盖切换失败。转弯随机化在后续单独加入，不改接口或增加任务外环。

## 2026-10-15 最新阶跃响应修订（覆盖以上旧0.5s口径）

用户允许放宽阶跃到达时间，当前采用|Δvx|≤0.5m/s时≤1.5s，0.5<|Δvx|≤1.5时≤2s；仍须进入原误差带并连续保持0.2s。超调仍<20%，停车反向峰值以本次Δvx的20%为界，不放宽稳态速度/角差/存活。更大阶跃不在本验收范围，不自动判过。新报告保存legacy_05s_passed及acceptance_profile，历史原报告不覆盖。斜坡跟踪独立评估，不冒充阶跃通过。

随机时序复现证据：启动保存initial_random_speed_schedule.pt；开启PPO捕获时逐步保存random_speed_schedule_per_step及physical_commands/cohort/episode_age，支持reset后不同时间表复放。初始表不能代替全程时间表。CPU相关测试现44项通过，随机模式GPU验证仍未完成。

2026-10-15 GPU接入补验完成：random_speed_gpu_capture_20261015中128步时间表复放与13056动态样本actualcommand误差0，actor/critic冻结、runtime通过。状态升级为时序接入通过，行为/训练效果仍未认证。

## 平滑变速严格门槛（2026-10-15用户补充）

只放宽阶跃，平滑斜坡/连续变速不得使用1.5/2s阶跃门槛。按原时间戳实时目标计算非零速度误差<10%，零目标绝对误差≤.01m/s；响应延迟<.5s，不做时间平移掩盖误差。姿态/双腿角差/存活单独验证。smooth_tracking_metrics目前支持完整单调斜坡，逐样本报告严格误差带及10–90%速度阈值首次越过延迟；延迟诊断不能代替误差门槛，非单调随机序列需先按完整斜坡拆段，reset或截断段不得认证。该严格逐样本检查不把允许的延迟当作误差豁免。
