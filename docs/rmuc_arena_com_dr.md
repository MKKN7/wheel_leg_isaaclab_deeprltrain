# 地胶与基座重心DR：渐进接入

这两项是独立课程变量，不用训练轮数自动晋级。默认关闭；2026-10-14当前运行的straight0275_balanced教师批次不热改配置。

## 地胶接触课程

训练显式设置：

```text
env.arena_ground_severity=-1
env.arena_ground_migration_progress=0.1
env.arena_ground_randomization_enabled=true
env.arena_ground_severity_range=[0,1]
env.friction_randomization_enabled=false
env.domain_randomization_max_friction_fraction=0
```

`migration_progress`是旧有效接触系数到目标地胶的插值比例；`severity`表示新地胶到磨损地胶的状态，每个环境reset从64个有限材质桶独立采样。每个环境两轮使用同一severity，静/动摩擦相关采样，避免构造动摩擦大于静摩擦的无效组合。

目标工程先验：新地胶有效静/动摩擦`.65/.55`、磨损`.45/.38`，回弹`.02`。这不是实测标定，也不靠3mm厚度虚构接触刚度。当前旧有效系数`.5/.5/0`，第一档progress=.1得到静摩擦`.495–.515`、动摩擦`.488–.505`、回弹`.002`。后续建议progress `.1→.25→.5→.75→1`，失败则插更小档。

共享terrain保持原物理材质`.5/.5/0`，仅两驱动轮采用multiply摩擦/max回弹，并把目标有效摩擦除以实际ground系数作为轮材料系数；这只是轮—地接触对分解，不能把轮材料值1.3宣传成地胶摩擦1.3。其他碰撞体接触保持原有模型。

绑定必须覆盖所有克隆环境，不能只给env0绑定。初始化时核对全部驱动轮材料；reset只更新选定env的驱动轮shape，关闭旧friction scaling防重复；保存实际PhysX系数、组合有效值、severity、shape索引和reset计数。此前仅检查源环境USD绑定不能证明整个克隆批次一致。

## 基座COM课程

```text
env.base_com_randomization_enabled=true
env.base_com_max_offset_flu_m=[0.01,0.01,0.005]
env.base_com_randomization_strength=0.1
```

最大范围为物理车体前/左/上坐标±10/10/5mm，属于可调整工程先验，非用户实测范围；第一档strength=.1实际为±1/1/.5mm。偏移作用于`robot`基座刚体COM，**不是宣称整机总COM移动同样数值**，其他刚体质量加权后总COM变化另算。

当前模型局部坐标换算为 `body_xyz=(-left,+up,-forward)`；API接收link-local COM pose，因此不再额外旋转资产90°初始姿态。每次reset从不可变标称COM加新偏移，不在上次偏移基础上累加。只改COM位置，保留COM四元数、质量、惯量和几何；质量/惯量DR仍独立运行。

GPU集成验证必须核对请求偏移与实际COMpose一致、四元数完全保留、只有选定基座写入，不能用配置值或CPU单测替代运行读回。

## 验证与启用

`evaluate_rmuc_grid.py --arena-ground-progress .1 --base-com-dr-strength .1` 显式启用首档，其他物理/策略参数沿同checkpoint契约。运行证据目录为 `artifacts/arena_com_dr_20261014`；启动失败单独归档，不计通过。主线仍执行逐副本速度<10%与直行/待机角差P95≤2°、无观察倾倒、旧技能retention；转弯不强制双腿对称。短集成测试仅证明参数生效，不等于地胶或COM鲁棒性完成。

接续教师训练时可同时采用这两个很小的首档，避免与惯量/弹簧/大延迟迁移同批叠加。总延迟最终上限12ms由独立课程负责，不再要求三个policy周期。旧v8不继续训练，待新教师验收后按用户部署需求蒸馏小MLP。
