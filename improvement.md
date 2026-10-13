# RMUC 瓶颈与工程复盘

## 2026-10-11 扩域7950训练完成，56点闭环启动

positive_bank_ppo620秒正常exit0，60更新，model7950 SHA94be4f4f7e64e554d8213e68110b6482c0a3ede8c7ad7dd9eaa83e12a9250e59，freeze/runtime通过；训练终报和freeze另存保留。已启动旧33+新23共56点/112env各>=4快筛，固定4.31/DR.1，单列旧域retention、新域指标和倾倒。尚无扩域行为结论，不宣传训练完成即学会，不升弹簧。

## 2026-10-11 扩域7910参数与旧域输出隔离审计

中段model7910 SHA a66b865eb85de1c9c5213f94273c6d95da0d545665fa40ab1f8eaca888704ffc确认仅指定扩域network及预声明std变化，所有anchor逐位未变；mid_freeze_audit通过。首次直接改command因部分yaw_error裁剪无法重建目标而拒绝；审计改用rollout状态并将yaw_rate置0以构造不裁剪的旧33命令，整actor输出差均0，mid_output_audit通过；这是门控隔离证明，不替代闭环retention。原60更新训练继续，固定4.31/DR.1，不以中段参数审计宣称新技能成功。

## 2026-10-11 扩域PPO实际采样与梯度通过

原positive_bank_ppo持续运行，连续采样实际.0996。更新11快照目标(1.25,+.6)14080条，抽样梯度463.30，域外梯度0，actual_rollout_gradient_audit通过；证明新速度下分支有效但非行为成功。56点验收入口与汇总已准备，分开报告旧33 retention和新增23通过数/倾倒，避免用总分掩盖旧技能丢失。固定4.31/DR.1，未扩弹簧。

## 2026-10-11 首个连续低速压弯扩域训练启动

positive_bank_ppo source fd036ec02c4ca6c2ab70a55e6ed76bb36de3bc7f5ab8676c169d4db4f6cff3c4，目标1.25/+.6，固定4.31/DR.1；只训最深network的显式signed更新域，冻结anchor及其余网络，原预算/consumer限幅不变。512env/60更新/10critic预热，50%旧域rehearsal+10%已开放域连续采样；完整数组保留，新增stage23目标，未改变vx+yaw接口。后续旧33+新增23共56点快筛命令已存，正式验收与无倾倒/严格速度仍需实证。当前仅启动，不宣称扩域学会，不扩弹簧。

## 2026-10-11 低速压弯扩域隔离入口实现

新增PhysicsAdaptation可选network_update_domain+冻结network_anchor，仅影响显式新checkpoint；恢复器与physics选择/Adam gate同步支持。16项physics/selection测试通过，保留原残差预算和25D/6D，未扩大consumer腿限幅。positive_bank_seed由7890构造SHA fd036ec02c4ca6c2ab70a55e6ed76bb36de3bc7f5ab8676c169d4db4f6cff3c4，局部vx[1,1.1,1.5,1.6]/yaw[.45,.55,1,1.1]，旧网络anchor冻结；8192真实obs初始与重载输出逐位不变，合成(1.25,+.6)有效梯度422.96。该可训练分支绕开旧authority速度门控但保持原动作/电机契约，尚未训练/闭环成功；先正向后独立负向，不把准备视作全域掌握。

## 2026-10-11 首批低速扩域23点结果与门控瓶颈

冻结7890/4.31/DR.1，23新点7个旧阶段passed，其中6个同时满足严格速度10%，全存活；其余主要yaw/速度/stable失败，不能称已掌握。典型1.25/±1实际yaw近0。routing_audit确认原1m/s position_authority/local专家速度support在>=1.2为0、1.1已衰减；既有局部训练无法覆盖这些新命令，下一步需显式扩域可训练分支并保留旧域，不盲目增加旧expert更新数。连续采样器只在已开放锚点间采样，原vx+yaw接口不改。新命令诊断retention审计明确not_evaluated，不能把空对照当通过。

## 2026-10-11 4.31正式全过，固定弹簧转入低速23点扩域

7890正式33/min32/runtime通过、全存活；三处速度.013009/.031178/.092597严格通过，原慢转倒退0。按最新用户指令暂停弹簧扩档，4.579375不启动，技术next_spring_allowed不覆盖用户暂停。已启动low_speed_domain_20261011/frontier_screen：固定SHA647c4ab1与4.31/DR.1，仅23个新增vx<=1.5中间速度/双向转弯命令，各>=4回合；旧33基线冻结，诊断不等同全域认证，后续训练需旧域retention。

## 2026-10-10 局部PPO正式严格与异常轨迹对照

正式33/33，runtime=True；严格速度门槛[{'command': (0.25, 0), 'mae': 0.013008801036056467, 'limit': 0.025, 'passed': True}, {'command': (0.5, -0.1), 'mae': 0.031178408357098064, 'limit': 0.05, 'passed': True}, {'command': (1.5, -0.1), 'mae': 0.0925969152546248, 'limit': 0.15, 'passed': True}]。旧稳态倒退回合1，新0；next_spring_allowed=True。证据artifacts/teacher_course_20261009/spring431_positive_slow_ppo/strict_retention_audit.json。仅同布局单seed阶段结果，未自动启动下一档。

## 2026-10-10 低速隔离候选完整正式结果

33/33，每点至少32回合，runtime=True；严格低速10%=True，指标{'linear_error': 0.013008801036056467, 'yaw_error': 0.006834658491970304, 'stable_fraction': 1.0, 'survival': 1.0}。证据artifacts/teacher_course_20261009/spring431_positive_slow_ppo/formal_combined.json。spring4.31/DR.1，最终10.775及全域最高DR未完成；未自动启动下一档。

## 2026-10-11 最新用户优先级：暂停扩弹簧，先扩1.5以下命令域

用户要求本批结束后不再往上扩弹簧，先把1.5m/s以下（含1.5边界）全速度双向压弯适应。当前4.31正式A/B继续；通过则冻结当前弹簧/Dfb/b/C/DR档，先补连续中间速度、慢转双向和起停，再逐级扩到既定1m/s±1–2.5rad/s、1.5m/s±.8–1.8rad/s及中间命令，旧33项持续retention。若当前候选不合格，保留最后合格基线并修复，不自动升物理档。此指令覆盖“先完成10.775才扩当前低速命令域”的旧顺序；最终弹簧/更高速/最高DR长期目标不删除。仍不新增vy/导航command，不启动常规学生或sim2sim。下一4.579375入口已加入暂停断言，未启动。计划证据low_speed_domain_priority_20261011.json。

## 2026-10-11 下一弹簧4.579375条件入口准备

7890 formal_a/b原进程存活，未重复启动。spring4579375_from7890仅准备唯一spring4.31→4.579375，绑定SHA647c4ab1，入口读取当前formal_combined与strict_retention_audit并校验同SHA/min32/全33/runtime/严格速度/全存活/无倒退；脚本编译通过，尚未启动，当前4.31正式结果未出。

## 2026-10-11 4.31候选7890快筛全过，正式验收启动

647c4ab1快筛33/33、runtime通过、全存活tilt0，(1,+.1)yaw.015526/stable1，三处速度.006973/.030646/.077435严格通过。已启动formal_a/b同checkpoint各>=32，A记录轨迹、零观察倾倒及严格速度门槛保持。快筛不等同正式或最高DR，通过前不继续弹簧升档。

## 2026-10-11 4.31训练收尾恢复与7890快筛

原trainer/launcher均不存在，日志完成7890/7891与SimulationContext cleared，model7890存在；旧status备份，恢复终态exit_code=null，不编造退出0、不重训。最终SHA647c4ab1f20df4c5f001a38edb23c6657dfca02bea165ebc1167160c619a4d5d冻结/runtime审计通过，仅指定reference及预声明std变化。quick_final全33各>=4已启动，阶段4.31/DR.1，严格速度/零倾倒要求保持，尚无行为结论、未晋升。

## 2026-10-11 4.31正向慢转7850中段冻结审计

model7850 SHA98819b47a44171a9edd7cd2433861746b14b59a528b8cd468f3aadcbd7a30014确有frozen_source.reference_network更新，除预声明std外无其他actor参数变化，mid_actor_update_audit通过。原PID1095951正常运行，60更新预算不变；仅参数隔离证据，不等同新动力行为通过，尚未评估或晋升。

## 2026-10-11 4.31正向慢转运行与实际梯度通过

原PID1095951正常，runtime实测4.31/Dfb/b/C/5.5/周期全通过。更新11实际PPO快照目标(1,+.1)10112条，抽样gradient279.47、域外gradient0，actual_rollout_gradient_audit通过。quick_final全33入口准备，三处严格速度及零观察倾倒保持，尚未评估或晋升。源scope广于单命令，75%既有域rehearsal保持。

## 2026-10-11 7830正式严格全过，接续4.31正向慢转适应

4.1753125正式33/min32/runtime通过、全存活，三处速度.012413/.025770/.086427严格通过，慢转倒退0，next_spring_allowed=true。原A/B正常terminal。启动spring431_positive_slow_ppo，source54f6f7ec，仅frozen_source.reference_network，60更新/512env/75%既有域rehearsal覆盖宽正向慢转域，其余actor冻结；阶段弹簧4.31、DR.1不变。不重复已知+0.1失败快筛，不按训练轮数晋升，最终10.775/全速度/最高DR仍未完成。

## 2026-10-10 局部PPO正式严格与异常轨迹对照

正式33/33，runtime=True；严格速度门槛[{'command': (0.25, 0), 'mae': 0.012412608666563513, 'limit': 0.025, 'passed': True}, {'command': (0.5, -0.1), 'mae': 0.025769830331014095, 'limit': 0.05, 'passed': True}, {'command': (1.5, -0.1), 'mae': 0.0864274972258603, 'limit': 0.15, 'passed': True}]。旧稳态倒退回合1，新0；next_spring_allowed=True。证据artifacts/teacher_course_20261009/spring41753125_crawl_ppo/strict_retention_audit.json。仅同布局单seed阶段结果，未自动启动下一档。

## 2026-10-10 低速隔离候选完整正式结果

33/33，每点至少32回合，runtime=True；严格低速10%=True，指标{'linear_error': 0.012412608666563513, 'yaw_error': 0.007108323948837841, 'stable_fraction': 1.0, 'survival': 1.0}。证据artifacts/teacher_course_20261009/spring41753125_crawl_ppo/formal_combined.json。spring4.1753125/DR.1，最终10.775及全域最高DR未完成；未自动启动下一档。

## 2026-10-11 下一档正向慢转弯分支定位

当前7830正式A/B原进程存活。CPU历史真实256obs(1,+.1)梯度审计：最近有效可训练分支为rmuc_physics_adaptation.frozen_source.reference_network，gradient287.46；更深旧network也有梯度，不优先解冻。读取checkpoint实际reference域vx[.3,.5,1.5,1.7]/yaw[0,.05,.35,.45]，覆盖多个正向慢转命令，后续适应不能误当单点expert，需该域rehearsal/retention及全部33验收。branch与buffer证据spring431_positive01_branch_audit，未启动下一档。

## 2026-10-11 下一档避免重复已知失败筛查

当前7830 formal_a/b原进程存活。next_stage_scope_audit对实际PPO512条(1,+.1)观测验证7830与7770输出逐位相同，crawl局部门控不覆盖该命令；4.31既有证据此点yaw.052463/stable.149666未过。当前正式若通过，下一步应针对4.31该已知缺口适应，不再全33快筛只为重新发现同一失败。修复后完整33/min32/严格速度/零倾倒验收仍保留；未启动新GPU任务。

## 2026-10-11 7830快筛33/33零倾倒，正式验收启动

54f6f7ec候选在spring4.1753125/DR.1快筛全33通过、runtime通过、全存活tilt0，三处速度.012324/.025239/.071195全部严格达标。已启动同checkpoint formal_a/b各>=32，A记录world轨迹，零倾倒和严格速度条件保持；不能从4回合快筛预设正式成功，未晋升。

## 2026-10-11 crawl60更新完成，7830独立快筛

880秒正常exit0，model7830 SHA54f6f7eca4d7f7899b569ed9b000a5a8efdc950adb05644e2964c554241de30f，freeze/runtime通过。quick_final完整33各>=4已启动，spring4.1753125/DR.1保持，三处严格速度+显式零倾倒门槛不变。仅训练完成，未声明低速跟踪成功；正式>=32全域前不晋升。

## 2026-10-11 crawl中段7790冻结范围审计

model7790 SHA3521aec8a7ed1c9946359ec599aa55d40b2b9051abaa5068b42660c08476a83c存在实际outer.frozen_reference_network参数更新；除预声明std外无域外actor参数变化，mid_actor_update_audit通过。原训练继续60更新预算，未据此声明速度修复、未追加GPU评估或升档。

## 2026-10-11 crawl实际rollout覆盖与梯度审计通过

原PID1069907运行，runtime读回通过。更新11快照目标(.25,0)33787条、负速度366条；抽样目标gradient414.51、负速度384.90、域外梯度0，actual_rollout_gradient_audit通过；负速度可含启动瞬态，不宣称持续倒退或已修复。quick_final全33入口准备，保留三处严格速度与零观察倾倒门槛，尚未启动评估。记录真实快照hash77c76522…，不重复训练、不晋升。

## 2026-10-11 半增量正式33/33但crawl严格速度失败，接续局部PPO

4.1753125正式33/min32/runtime通过、全存活，无慢转稳态倒退；crawl误差.031787>严格.025，其他两处速度.025770/.086427通过，因此不晋升。新spring41753125_crawl_ppo source7770 SHA e48a91a76329d0e6bd8696c24bd789395c26a687b04588505fa52d635a1e5d49，仅outer.frozen_reference_network，256真实历史obs门控1/gradient362.22；512env/60更新/10预热/50%既有域rehearsal，原动力及DR.1保持。旧A/B均terminal0，未重复评估；最终10.775前不扩高速域。

## 2026-10-10 局部PPO正式严格与异常轨迹对照

正式33/33，runtime=True；严格速度门槛[{'command': (0.25, 0), 'mae': 0.03178719613065688, 'limit': 0.025, 'passed': False}, {'command': (0.5, -0.1), 'mae': 0.025769830331014095, 'limit': 0.05, 'passed': True}, {'command': (1.5, -0.1), 'mae': 0.0864274972258603, 'limit': 0.15, 'passed': True}]。旧稳态倒退回合1，新0；next_spring_allowed=False。证据artifacts/teacher_course_20261009/spring41753125_from7770/strict_retention_audit.json。仅同布局单seed阶段结果，未自动启动下一档。

## 2026-10-10 低速隔离候选完整正式结果

33/33，每点至少32回合，runtime=True；严格低速10%=False，指标{'linear_error': 0.03178719613065688, 'yaw_error': 0.006820552723862255, 'stable_fraction': 1.0, 'survival': 1.0}。证据artifacts/teacher_course_20261009/spring41753125_from7770/formal_combined.json。spring4.1753125/DR.1，最终10.775及全域最高DR未完成；未自动启动下一档。

## 2026-10-10 spring4.1753125快筛全过，正式验收启动

同7770半增量快筛33/33、runtime通过、全存活tilt0，三处速度.022169/.025239/.071195均达严格线；eligible_for_formal=true。已启动formal_a/b同checkpoint全33每点>=32，A记录world轨迹，严格速度与显式倾倒检查保持。固定DR.1及Dfb/b/C，不按快筛升级，最终10.775未完成。

## 2026-10-10 spring4.31快筛32/33，半增量4.1753125接续

同7770在4.31 runtime通过，全部存活、tilt0，但(1,+.1)yaw.052463/stable.149666失败；crawl误差.032449超严格.025，不能晋升。(.5,-.1)速度.030646、(1.5,-.1).077435/stable1保留。原评估正常terminal0；已启动spring41753125_from7770同66env/seed/原33命令，只把从4.040625的弹簧增量减半到4.1753125，无网络更新。报告显式检查三处严格速度+零倾倒，若仍失败做定向适应，不反复缩步掩盖问题。

## 2026-10-10 expert6最终正式全存活，进入spring4.31

7770 SHA e48a91a76329d0e6bd8696c24bd789395c26a687b04588505fa52d635a1e5d49正式33/33、min32、runtime通过，all_survived=true、fall_commands为空；三处严格速度.023132/.020251/.079735通过，慢转旧稳态倒退1→0，next_spring_allowed=true。本轮核实原A/B均正常terminal，运行CPU汇总完成证据。已启动spring431_from7770_survival，唯一弹簧4.040625→4.31，网络/DR.1/Dfb/b/C/原33命令冻结，快筛各>=4，不代表正式或最高DR认证，最终10.775未完成。

## 2026-10-10 局部PPO正式严格与异常轨迹对照

正式33/33，runtime=True；严格速度门槛[{'command': (0.25, 0), 'mae': 0.02313153002174403, 'limit': 0.025, 'passed': True}, {'command': (0.5, -0.1), 'mae': 0.020250884528599796, 'limit': 0.05, 'passed': True}, {'command': (1.5, -0.1), 'mae': 0.07973506378888287, 'limit': 0.15, 'passed': True}]。旧稳态倒退回合1，新0；next_spring_allowed=True。证据artifacts/teacher_course_20261009/spring4040625_positive1_survival_ppo/strict_retention_audit.json。仅同布局单seed阶段结果，未自动启动下一档。

## 2026-10-10 低速隔离候选完整正式结果

33/33，每点至少32回合，runtime=True；严格低速10%=True，指标{'linear_error': 0.02313153002174403, 'yaw_error': 0.0070438026185817145, 'stable_fraction': 1.0, 'survival': 1.0}。证据artifacts/teacher_course_20261009/spring4040625_positive1_survival_ppo/formal_combined.json。spring4.040625/DR.1，最终10.775及全域最高DR未完成；未自动启动下一档。

## 2026-10-10 expert6训练60更新完成，7770快筛启动

575秒正常exit0，model7770 SHA e48a91a76329d0e6bd8696c24bd789395c26a687b04588505fa52d635a1e5d49，freeze/runtime通过，仅expert6更新。quick_final全33各>=4已启动，显式倾倒/存活及三处严格速度门槛保持；正式匹配128env/32回合前不能称+1倾倒修复，不自动推进4.31。

## 2026-10-10 expert6中段真实更新冻结审计

model7720为10次critic预热端点，actor未变化符合配置；后续model7740 SHA d809476b245e67c2d5aff4adb46d34875e00b056e37629aa443569140f43e2ed确有expert6参数更新，unexpected为空，其他actor/physics未变。mid_actor_update_audit通过，原训练继续60预算；仅参数审计，不声明倾倒修复或行为通过。

## 2026-10-10 expert6运行与显式倾倒验收

positive1_survival_ppo原PID339707存活，runtime实际Dfb/b/C/4.040625/5.5及周期全部通过，进入7712/7771。quick_final入口准备并绑定最终哈希；报告新增all_survived、total_tilt_terminations、fall_commands，eligible_for_formal要求全33/runtime/三处严格速度且零观察倾倒、全部存活。不将旧>=95%阶段passed等同无倾倒，不将有限回合零倾倒等同实机保证。未启动评估、未晋升。

## 2026-10-10 +1存活缺口启动隔离expert6适应

当前b577候选慢转倒退已在匹配正式布局消失、三处严格速度通过，但+1rad/s仍31/32存活/yaw.049375/stable.511040，禁止下一弹簧。checkpoint确认expert6中心+1，近期实际rollout256样本gradient255.89。新spring4040625_positive1_survival_ppo仅expert6，source b57703a16b93df5f9c2c443bb469c204eb080022166283faef73c2fc03a7bfe3，60更新/512env/10critic预热/直行与双向.8 rehearsal，其他physics/anchor/expert冻结，动力4.040625/DR.1不变。首选A轨迹不含+1命令，CPU预检拒绝后改用真实PPO快照，未产生GPU失败实验。后续以实际存活和全33保留为准，不按reward或轮数晋升。

## 2026-10-10 局部PPO正式严格与异常轨迹对照

正式33/33，runtime=True；严格速度门槛[{'command': (0.25, 0), 'mae': 0.023136169216704607, 'limit': 0.025, 'passed': True}, {'command': (0.5, -0.1), 'mae': 0.020250884528599796, 'limit': 0.05, 'passed': True}, {'command': (1.5, -0.1), 'mae': 0.07982521950201844, 'limit': 0.15, 'passed': True}]。旧稳态倒退回合1，新0；next_spring_allowed=False。证据artifacts/teacher_course_20261009/spring4040625_slow_turn_local_ppo/strict_retention_audit.json。仅同布局单seed阶段结果，未自动启动下一档。

## 2026-10-10 局部PPO正式严格与异常轨迹对照

正式33/33，runtime=True；严格速度门槛[{'command': (0.25, 0), 'mae': 0.023136169216704607, 'limit': 0.025, 'passed': True}, {'command': (0.5, -0.1), 'mae': 0.020250884528599796, 'limit': 0.05, 'passed': True}, {'command': (1.5, -0.1), 'mae': 0.07982521950201844, 'limit': 0.15, 'passed': True}]。旧稳态倒退回合1，新0；next_spring_allowed=True。证据artifacts/teacher_course_20261009/spring4040625_slow_turn_local_ppo/strict_retention_audit.json。仅同布局单seed阶段结果，未自动启动下一档。

## 2026-10-10 低速隔离候选完整正式结果

33/33，每点至少32回合，runtime=True；严格低速10%=True，指标{'linear_error': 0.023136169216704607, 'yaw_error': 0.0070311799894607185, 'stable_fraction': 1.0, 'survival': 1.0}。证据artifacts/teacher_course_20261009/spring4040625_slow_turn_local_ppo/formal_combined.json。spring4.040625/DR.1，最终10.775及全域最高DR未完成；未自动启动下一档。

## 2026-10-10 更正4.040625正式“全部存活”说法

逐行核验发现combined_candidate、combined_v2及本轮localPPO的正式B中(1,+1)均survival31/32=.96875、tilt1；旧阶段survival>=.95故仍passed，非统计漏计、非本轮新增退化。此前概括“全部存活”不准确，以原evaluation JSON为准。known_fall_audit显式保留此事实；后续严格汇总单列all_survived和fall_commands，新增未处理倾倒阻止下一弹簧。A批仍运行，不能宣传当前为无倾倒/交付鲁棒通过。

## 2026-10-10 局部PPO候选快筛33/33及严格门槛全过

b57703a1快筛33/33、runtime通过，(.5,-.1)速度误差.019159/yaw.007346/stable1/survival1；crawl.011886、1.5/-.1 .064241均低于各自10%门槛。已启动formal_a/b同checkpoint每点>=32，A复用119env曾出现倒退的布局且记录全程world_trace，不能仅凭快筛宣称异常消失。原快筛terminal0，未重复训练，未升档。

## 2026-10-10 慢转局部PPO60更新完成，独立快筛启动

原训练600秒正常exit0，model7710 SHA b57703a16b93df5f9c2c443bb469c204eb080022166283faef73c2fc03a7bfe3，freeze/runtime通过，仅指定signed reference及预声明std变化。quick_final完整33各>=4已启动，三处严格速度门槛保持，尚无行为结论。后续正式同119env异常回合及完整33保留验收必需，不以训练完成晋升。

## 2026-10-10 局部PPO后续快筛严格门槛固化

原训练PID317480正常推进至7679/7711，预算不变。quick_final/report.py新增strict_retention_audit和eligible_for_formal：全33、runtime以及crawl<.025、(.5,-.1)<.05、(1.5,-.1)<.15同时通过才具备正式验收资格。编译检查通过，尚未运行评估，不以reward或更新数晋升。

## 2026-10-10 慢转局部PPO实际rollout梯度通过

原PID317480正常运行，更新11捕获实际rollout：目标(.5,-.1)18070条、负速度205条；目标gradient_norm826.78，负速度642.89，域外gradient0。actual_rollout_gradient_audit通过，证明选中分支实际可学习且参数域外隔离；负速度样本不等同持续倒退异常完整重现，也不等同修复成功。quick_final独立33快筛入口已准备，训练完成/freeze/runtime后才可执行，未提前晋升。

## 2026-10-10 慢转异常状态局部PPO启动

source c18d8915，19条记录倒退状态preflight门控1、gradient_norm54.02，选定第二层frozen_source.reference_network可达；未直接回灌状态，在线异常覆盖仍需rollout核对。启动spring4040625_slow_turn_local_ppo，512env/60更新/10critic预热/lr1e-5/50%既有域rehearsal，除该局部reference及critic外冻结，spring4.040625/DR.1及Dfb/b/C不变；完整33>=32和三处严格速度门槛保持，不升4.31。

## 2026-10-10 慢转失败集中于少数回合，停止固定偏置搜索

轨迹滑窗重建误差约1.3e-9，原MAE.052582确认。回合间均值偏差占方差97.92%；env3两个异常回合，最差稳态实际约-.122m/s，而命令+.5，其余环境多数误差.02-.04。branch_attribution显示异常回合门控1、实际修复轮增量约.09793/.05874接近目标.1/.06，并非修复头失效；腿姿态/轮输出进入另一稳定分支。后续针对真实异常状态用现有signed reference局部PPO适应，不继续固定偏置搜索，不删除异常样本。最终DR读回不能代表先前回合，不宣称具体延迟因果。当前无新GPU任务、未晋升；next_action.json保留下一步约束。

## 2026-10-10 慢转完整轨迹误差分解

同正式A布局采集完成，滑窗误差重建通过；原报告MAE=0.05258247，瞬时MAE=0.04948156，signed=0.00730669。回合内方差=0.00026192、回合间均值方差=0.01233418。详见artifacts/teacher_course_20261009/spring4040625_slow_turn_trace_retry/speed_dispersion_report.json；该分解不单独证明DR因素因果，不改变验收线或晋升。

## 2026-10-10 速度离线归因对齐原验收定义

确认evaluate_rmuc_grid原速度门槛为物理heading速度post-step绝对误差滑窗均值，signed_mean是原始物理误差，二者不同不直接证明超速。analyze_speed已改post_age>=warmup、排除done、逐回合重建command_gate_steps滑窗，要求重建MAE与原报告差<1e-5后才出归因。完整trace原进程继续，未改控制或验收线。

## 2026-10-10 慢转严格速度缺口改做离散/振荡归因

正式(.5,-.1)MAE.052582而signed_mean仅+.007307，不能按单一超速继续加固定偏置。启动同c18d8915/119env A布局/seed/4.040625/DR.1的只读完整world_trace，区分回合间偏差和回合内波动，未改动作。首次startup_capture8000被CLI上限500拒绝，未仿真；独立slow_turn_trace_retry修正为500，world recorder仍完整记录。新增离线speed_dispersion脚本按环境/回合分析，结果尚待，4.31禁止启动。

## 2026-10-10 4.040625组合v2正式33/33，严格慢转速度仍差0.00258

同c18d8915候选完整33每点>=32、runtime通过、无失败项。+1.4 yaw.037908/stable.755748，1.5/-.1速度.079825/yaw.021982/stable1；crawl误差.023136<.025。(.5,-.1)正式速度.052582（10.52%）略超严格.05门槛，yaw.018372/stable.950884/survival1。strict_retention_audit标记next_spring_allowed=false，4.31不启动；保留当前33通过候选但不放宽新增门槛，下一步仅补慢转速度余量。

## 2026-10-10 低速隔离候选完整正式结果

33/33，每点至少32回合，runtime=True；严格低速10%=True，指标{'linear_error': 0.023136169216704607, 'yaw_error': 0.0070311799894607185, 'stable_fraction': 1.0, 'survival': 1.0}。证据artifacts/teacher_course_20261009/spring4040625_combined_v2/formal_combined.json。spring4.040625/DR.1，最终10.775及全域最高DR未完成；未自动启动下一档。

## 2026-10-10 下一4.31草案绑定组合v2

formal_a/b原进程正常运行。独立spring431_from_combined_v2准备同c18d8915候选、唯一spring4.040625→4.31；硬要求当前组合正式33/min32/runtime及三处修复速度门槛crawl<.025、(.5,-.1)<.05、(1.5,-.1)<.15。旧bb25aa36下一档草案不启用。脚本编译通过，未启动、未晋升。

## 2026-10-10 +1.4快筛通过，组合v2正式启动

positive14快筛33/33、runtime通过，(1,+1.4)yaw.027484/stable.857860/survival1。与d13f21c4隔离1.5/-.1候选合并仅expert3，merge audit允许范围和同输入输出差0；新SHA c18d8915a3a20a7cd55d9b5facacc8dd92c8998e587609e9cda38e601e781f5d。spring4040625_combined_v2正式A/B同checkpoint全33各>=32已启动，动力4.040625/DR.1保持，尚无组合正式结论，未晋升。

## 2026-10-10 +1.4训练完成，独立快筛启动

positive14原训练正常exit0、60更新，checkpoint7710 SHA1bc112bb01b93c8d7a024652a78511111f0e488afab8ca9213ed4ac3c73edc56，freeze/runtime通过。quick_final全33快筛已启动，spring4.040625/DR.1不变；尚未行为认证。combined_v2仅准备merge脚本，必须目标各自通过、全存活并权重审计/同输入对拍；尚未执行合并，后续同checkpoint正式33仍必需。

## 2026-10-10 1.5负向隔离候选快筛33/33

候选d13f21c4无偏置完整33项各>=4快筛通过，runtime通过；(1.5,-.1)速度误差.064241（4.28%）、yaw.022205、stable1/survival1。+1.4快筛通过不替代此前正式128env失败，仍需原训练与正式验收。positive14原PID42042存活，quick_final准备但未启动，入口需训练完成/freeze/runtime并绑定final哈希到command与contract。未合并、未晋升。

## 2026-10-10 1.5负向隔离候选启动闭环及边界检查

fit600完成，轮holdout MAE .000064/.000026、保留样本差0、reload exact；candidate d13f21c4d6d21337c1217d7e8a3cb9ee38541ee0e6cda99e84aa049f5490dc4f。无偏置全33快筛已启动，spring4.040625/DR.1固定。210组真实观测构造边界审计通过，域外输出完全一致、域内有限；不替代中间命令行为验收。+1.4原训练PID42042仍运行，尚未合并或晋升。

## 2026-10-10 1.5负向慢转有效诊断与隔离拟合

9组诊断runtime通过、全存活。common+.02/diff+.005速度误差.078744（5.25%）、yaw.026152、stable1；负diff破坏yaw/stable，排除。已CPU启动spring4040625_v15_negative01_fit，source bb25aa36，第二层frozen_source.frozen_reference_network局部拟合，冻结anchor和其他模块，域vx[1.35,1.45,1.55,1.65]/yaw[-.16,-.12,-.08,-.04]，原残差预算不变，600更新。原+1.4训练继续，拟合与闭环未完成，不晋升。

## 2026-10-10 低速共同轮偏置诊断完成

冻结bb25aa36、spring4.040625/DR.1，仅(1.5,-.1)九组偏置含零对照；未改部署闭环。逐组速度/yaw/stable及严格10%速度门槛见artifacts/teacher_course_20261009/spring4040625_v15_negative01_probe/final_report.json。诊断不代表正式通过，不晋升；正向expert训练独立进行。

## 2026-10-10 1.5负向慢转归因并行启动

positive14训练原进程存活且runtime通过。新spring4040625_v15_negative01_probe冻结bb25aa36，仅(1.5,-.1)共同[0,.01,.02]×差分[-.005,0,.005]9组含零对照，各4回合/3200步/600秒。历史256obs梯度审计定位有效模块为第二层frozen_source的frozen_reference_network及更深旧network，当前+1.4expert不覆盖；不盲目解冻深层旧网，等待敏感性方向后做局部隔离。最终正式33前不晋升。

## 2026-10-10 4.040625组合正式31/33，继续+1.4局部适应

完整33每点>=32，runtime通过、全部存活；失败(1,+1.4)yaw.052550/stable.445757、(1.5,-.1)yaw.050647/stable.493311；两处原负向修复保留。failed_scope_audit证实这两个命令在组合前后同输入输出精确相同，同布局3.9059375→4.040625性能下降，证据指向动力适应不足，不是组合直接改输出。4.31禁止启动。已启动spring4040625_positive14_ppo，source bb25aa36，仅expert3，60更新/512env，其余physics/anchor/expert冻结，原rehearsal保留。1.5/-.1单独处理，未晋升。

## 2026-10-10 低速隔离候选完整正式结果

31/33，每点至少32回合，runtime=True；严格低速10%=True，指标{'linear_error': 0.023136169216704607, 'yaw_error': 0.0070311799894607185, 'stable_fraction': 1.0, 'survival': 1.0}。证据artifacts/teacher_course_20261009/spring4040625_combined_candidate/formal_combined.json。spring4.040625/DR.1，最终10.775及全域最高DR未完成；未自动启动下一档。

## 2026-10-10 组合候选后续严格慢转弯门槛

组合正式A/B原进程正常推进。下一4.31入口除完整33/min32/runtime和crawl<.025外，追加(.5,-.1)速度误差<.05且行为passed，逐批核对同checkpoint SHA；避免新修复只通过旧.2速度宽线便晋升。脚本编译通过，未启动下一档。

## 2026-10-10 慢速隔离边界通过与后续条件入口

慢速reference修复210组真实观测构造边界审计通过，域外与冻结源逐位一致、域内有限；不等同中间命令闭环认证。组合formal_a/b原进程运行中。spring430_from_combined仅准备spring4.040625→4.31，source bb25aa36固定，入口要求组合正式33/min32/runtime及strictcrawl通过，未启动；最终10.775前不扩高速域。

## 2026-10-10 两负向修复组合，启动同checkpoint正式33验收

7710独立快筛32/33，(1,-1)速度误差.008331/yaw.009567/stable1/survival1，唯一(.5,-.1)原缺口。慢速a383候选修复另一缺口，均无存活退化。组合仅从7710复制expert1到慢速候选，merge_audit确认允许权重范围及同输入两作用域输出差0；组合SHA bb25aa367fadfdc843c5029a6df54e35ede025a488ccfe2e511e0b2c67a4c1f9。已启动formal_a/b同组合全33每点>=32，spring4.040625/DR.1，无外部偏置；尚无组合闭环结论，未晋升。

## 2026-10-10 慢速负向闭环修复，7710独立验收

慢速隔离候选a38308a5全33快筛32/33，(.5,-.1)误差.027331（5.47%）、yaw.011537、stable1/survival1；剩(1,-1)原失败，未新增失败。单expert训练60更新完成，model7710 SHA03b8510d11fa01a114715a5bfc44f0306cf237813d5f6bf494afc54152a787c3，freeze/runtime通过；quick_final独立33快筛已启动。两个修复尚未合并，静态或分开通过不当组合通过，最终同checkpoint正式33>=32仍必需。

## 2026-10-10 慢速负向隔离候选进入无偏置闭环

CPU600更新完成，轮action holdout MAE .000310/.000239，保留域外样本输出精确不变、reload exact；candidate SHA a38308a567e5b81cf3b91d376c55592427debacee2acbcb62db9020a18c831ea。已启动spring4040625_slow_negative_isolated_fit全33快筛，spring4.040625/DR.1，无外加偏置；负向expert训练原PID37552仍存活，尚未合并，静态拟合不能当闭环成功。完整正式验收前不晋升。

## 2026-10-10 慢速负向有效目标与隔离拟合

refine诊断runtime通过：common+.08/diff+.02速度误差.033996（6.8%）、yaw.017685、stable1/survival1；+.10两组倾倒，不用于拟合。已启动CPU600步spring4040625_slow_negative_isolated_fit，在第二层frozen_source新增原network克隆reference，旧network冻结，signed域vx[.4,.45,.55,.6]/yaw[-.16,-.12,-.08,-.04]，复用现有门控实现、残差预算不改。source d8a3f329，目标为真实轨迹动作，不在部署加偏置；拟合/闭环尚未证明，原负向expert训练继续。

## 2026-10-10 低速共同轮偏置诊断完成

冻结d8a3f329、spring4.040625/DR.1，仅(.5,-.1)七组细化偏置含零对照；未改部署闭环。逐组速度/yaw/stable及严格10%速度门槛见artifacts/teacher_course_20261009/spring4040625_slow_negative_probe_refine/final_report.json。诊断不代表正式通过，不晋升；正向expert训练独立进行。

## 2026-10-10 慢速负向共同/差分归因结果

9组诊断runtime通过且全部存活；common+.04/diff+.01速度误差.113956、yaw.016590、stable1；common主要降低超速，正diff修正yaw，负diff均stable0。尚不达最终10%速度线(.05m/s)，未拟合或晋升。已启动refine零对照+common[.06,.08,.10]×diff[.01,.02]7组各4回合，其他契约不变；负向expert原训练继续。

## 2026-10-10 低速共同轮偏置诊断完成

冻结d8a3f329、spring4.040625/DR.1，仅(.5,-.1)九组偏置含零对照；未改部署闭环。逐组速度/yaw/stable及严格10%速度门槛见artifacts/teacher_course_20261009/spring4040625_slow_negative_probe/final_report.json。诊断不代表正式通过，不晋升；正向expert训练独立进行。

## 2026-10-10 负向expert训练与慢速归因并行

negative1_ppo_retry原训练PID37552存活，runtime实测Dfb/b/C/5.5/4.040625及周期通过。另启动spring4040625_slow_negative_probe，冻结d8a3f329，仅(.5,-.1)，common[0,.02,.04]×diff[-.01,0,.01]共9组含零对照，各4回合/3200步/600秒；判别速度与yaw需要共同还是差分轮动作，诊断不改部署、不晋升。实际学习修复仍须隔离和完整33保留验收。

## 2026-10-10 负向训练采样覆盖修正

首次spring4040625_negative1_ppo启动前coverage检查拒绝：level49不能rehearse未来50/51，未产生训练更新，保留incomplete报告。独立_retry修正rehearsal为[2,48]、gate[2,48,49]，直行/双向.8与目标双向1；高位expert冻结、最终完整33验收不减。已重新启动，其他source/动力/DR/60更新预算不变。

## 2026-10-10 3.9059375正式33/33通过，接续4.040625负向expert

formal_combined确认同d8a3f329候选33/33、min32、runtime通过，(.25,0)误差.016771/stable1且严格10%通过。4.040625已知快筛31/33不重复；checkpoint centers确认expert1=-1rad/s。启动spring4040625_negative1_ppo：仅expert1、全部physics/两个隔离anchor/其他expert冻结，512env/60更新/10critic预热/每10保存，当前DR.1/Dfb/b/C保持；(.5,-.1)独立缺口不能由该训练修复，正式全33前不晋升，最终10.775未完成。

## 2026-10-10 低速隔离候选完整正式结果

33/33，每点至少32回合，runtime=True；严格低速10%=True，指标{'linear_error': 0.01677135958719413, 'yaw_error': 0.007272476336629096, 'stable_fraction': 1.0, 'survival': 1.0}。证据artifacts/teacher_course_20261009/spring39059375_from_crawl_candidate_retry/formal_combined.json。spring3.9059375/DR.1，最终10.775及全域最高DR未完成；未自动启动下一档。

## 2026-10-10 负向慢转弯有效分支审计

3.9059375正式A/B运行中，未重复启动。冻结当前d8a3f329候选，使用历史真实256条(.5,-.1)观测检查网络梯度：outer/inner现有局部修复头未生效，仅更深frozen_source.frozen_source.network及再深一层network有有效梯度。不能把正向expert训练当该技能修复；未解冻旧分支、未修改网络。证据spring39059375_branch_audit/gradient_inventory.json，历史观测审计不代替当前动力闭环。

## 2026-10-10 spring3.9059375快筛33/33，正式接续

同d8a3f329候选、同66env/seed4567对照：3.77125→3.9059375→4.040625，两项负向转弯yaw/stable余量逐渐减小；matched_spring_comparison.json保留具体指标。半增量3.9059375快筛33/33，(.25,0)误差.006698/stable1；(.5,-.1)yaw.049737/stable.561037，(1,-1)yaw.048789/stable.743311，余量仍小。已启动该档formal_a/b每项>=32，无新训练，尚未晋升；最终10.775/全域/最高DR未完成。

## 2026-10-10 spring4.040625快筛31/33，减半增量核对

全存活、runtime通过；(.5,-.1)yaw.050022/stable.437291失败，(1,-1)yaw.050289失败；低速(.25,0)误差.011886/stable1保留。固定d8a3f329候选，将增量减半到3.9059375，仍全33快筛，不升档。首次新目录命令因JSON转义路径替换未更新output，防覆盖保护退出2、未进入仿真、旧证据未改；已保存failed_before_simulation报告，修正并在spring39059375_from_crawl_candidate_retry独立目录启动，预算/契约不变。

## 2026-10-10 spring3.77125完整33/33通过，接续4.040625

隔离候选SHA d8a3f329abe9c57e2cd39cdd9511f9d1f2164d0b848198a494d30de113d04dc3正式A/B共33/33，每点至少32，runtime通过；低速(.25,0)误差.014375（5.75%）、yaw.007622、stable1/survival1。原正式进程均terminal exit0。已启动spring4040625_from_crawl_candidate当前33点各>=4快筛，唯一spring3.77125→4.040625，Dfb/b/C/DR.1保持；最终10.775/最高DR/全速度及圆弧阶跃仍未完成，不将阶段通过写成部署认证。

## 2026-10-10 低速隔离候选完整正式结果

33/33，每点至少32回合，runtime=True；严格低速10%=True，指标{'linear_error': 0.014375259246315844, 'yaw_error': 0.007621895908113308, 'stable_fraction': 1.0, 'survival': 1.0}。证据artifacts/teacher_course_20261009/spring377125_crawl_isolated_fit/formal_combined.json。spring3.77125/DR.1，最终10.775及全域最高DR未完成；未自动启动下一档。

## 2026-10-10 隔离边界核验与下一弹簧条件入口

当前formal_a/b原进程存活，未重复评估。boundary_audit使用真实观测构造187组vx/yaw边界输入，域外输出逐位不变、域内有限，最大动作变化.094878；仅代码隔离证明，不能替代中间速度闭环。新spring4040625_from_crawl_candidate准备唯一spring3.77125→4.040625，固定SHA d8a3f329abe9c57e2cd39cdd9511f9d1f2164d0b848198a494d30de113d04dc3；入口硬要求formal_combined33/33、min32、runtime及严格crawl<.025，同SHA。尚未启动、未晋升；最终10.775前不扩高速域。

## 2026-10-10 低速隔离候选33/33快筛通过，正式A/B启动

无外加偏置，全部33当前命令各>=4通过，runtime通过、零技能丢失/存活退化；(.25,0)速度误差.013212（5.28%）、yaw.007610、stable1/survival1。候选SHA d8a3f329abe9c57e2cd39cdd9511f9d1f2164d0b848198a494d30de113d04dc3。同候选formal_a/b GPU0并行正式每点>=32已启动，尚未晋升；禁止将快筛当正式通过。physics/DR保持3.77125/.1，未扩高速域。

## 2026-10-10 低速隔离候选完成CPU拟合，启动无偏置闭环

600次CPU更新22.9秒，轮动作holdout MAE .000831/.001271，域外保留样本逐位相同、reload exact；candidate SHA d8a3f329abe9c57e2cd39cdd9511f9d1f2164d0b848198a494d30de113d04dc3。仅outer.frozen_reference_network更新，新增冻结anchor限制vx[.15,.20,.30,.35]/yaw[-.08,-.02,.02,.08]，残差上限不改。当前33项无外加偏置快筛已启动，各>=4回合；静态拟合不是行为成功，完整正式>=32尚待，promoted=false。目录artifacts/teacher_course_20261009/spring377125_crawl_isolated_fit。

## 2026-10-10 7650两正向缺口正式修复，低速进入隔离拟合

final7650_b正式15/16，每点>=32、runtime通过、全存活；+1.2 yaw误差.028690/stable.935410，+1.4 .044860/.637019均通过，唯一(.25,0)速度.200098/stable.440008仍失败。低速边界诊断+.09速度误差.018485/stable1/survival1，+.10误差.0550，+.11存活.5，不把非单调边界外推。15000真实obs核对7550/7650低速输出完全相同。已CPU启动spring377125_crawl_isolated_fit，source7650，仅outer.frozen_reference_network局部域vx[.15,.20,.30,.35]/yaw[-.08,-.02,.02,.08]，冻结anchor、残差上限不变、600更新，拟合及后续闭环未完成，不晋升。

## 2026-10-10 低速共同轮偏置诊断完成

冻结7550、spring3.77125/DR.1，仅(.25,0)边界细化四组含零对照；未改部署闭环。逐组速度/yaw/stable及严格10%速度门槛见artifacts/teacher_course_20261009/spring377125_crawl_probe_bracket/final_report.json。诊断不代表正式通过，不晋升；正向expert训练独立进行。

## 2026-10-10 低速可行偏置边界证据

冻结7550、spring3.77125，wide诊断+.04速度误差.12940，+.08误差.05695，二者survival1/stable1；+.12和+.16稳态前倾倒，survival0，无稳态数据。最终10%尚未满足，不拟合失败目标。已启动bracket仅0/.09/.10/.11四组各8回合细化，固定其余契约；final7650_b正式评估原PID继续，无重复训练。finish脚本增加None稳态样本处理，失败不伪造数值。

## 2026-10-10 低速共同轮偏置诊断完成

冻结7550、spring3.77125/DR.1，仅(.25,0)扩大偏置五组含零对照；未改部署闭环。逐组速度/yaw/stable及严格10%速度门槛见artifacts/teacher_course_20261009/spring377125_crawl_probe_wide/final_report.json。诊断不代表正式通过，不晋升；正向expert训练独立进行。

## 2026-10-10 7650训练完成并启动正式B批

正向expert2/3完成100更新，model7650 SHAffa66679132cd87b8c17c99f14753b3a1daef0fada4614a34aafce53cfedb942，freeze/runtime通过；final7650_b按同布局16项每点>=32验收已启动，尚无行为结论。spring3.77125、DR.1及Dfb/b/C不变。低速五组轮共同偏置诊断全部存活：零偏置误差.20110/stable.43186，+.02误差.16582/stable1；最终10%速度门槛全部未过。诊断finish脚本误比较源码hash字典与checkpointhash，已修复并重算runtime通过，原evaluation未改、未重复仿真。偏置不是部署策略，不晋升；后续仍需低速隔离学习及完整33项保留验收。

## 2026-10-10 低速共同轮偏置诊断完成

冻结7550、spring3.77125/DR.1，仅(.25,0)五组偏置含零对照；未改部署闭环。逐组速度/yaw/stable及严格10%速度门槛见artifacts/teacher_course_20261009/spring377125_crawl_probe/final_report.json。诊断不代表正式通过，不晋升；正向expert训练独立进行。

## 2026-10-10 低速共同轮偏置诊断完成

冻结7550、spring3.77125/DR.1，仅(.25,0)五组偏置含零对照；未改部署闭环。逐组速度/yaw/stable及严格10%速度门槛见artifacts/teacher_course_20261009/spring377125_crawl_probe/final_report.json。诊断不代表正式通过，不晋升；正向expert训练独立进行。

## 2026-10-10 0.25直行缺口单独归因

当前3.77125正式(.25,0)signed_speed_error+.200183，即实测约.45018m/s，survival1，非倾倒；动作槽3超1后被consumer裁剪。对7550历史真实256obs梯度审计发现outer.frozen_reference_network生效，默认outer.network及inner.scoped/reference均gate0，当前转弯expert训练不会修复它。branch证据spring377125_crawl_audit/gradient_inventory.json，已保存同布局3.501875/3.77125对照；后续优先局部隔离低速直行修正，禁止盲目解冻最老physics分支。正向训练原进程继续。

## 2026-10-10 3.77125正式30/33，接续两正向expert适应

完整33项每点>=32、runtime通过、全部存活；失败(1,+1.2)yaw.057439/stable.409072，(1,+1.4)yaw.053675/stable.472826，(.25,0)speed_error.200098/stable.440008。快筛33/33不能代替正式通过，4.040625不启动。新spring377125_positive12_14_ppo source7550 SHA583e5252ce6a4dcd48ff40c01368507dfb680558ceb984dd53c59829d7fc380c，仅expert2/3更新，physics/隔离修复全冻结；512env/100轮/10预热，当前DR.1/Dfb/b/C不变。低速直行缺口单独跟踪，最终全33验收不得遗漏。跨日launcher改按唯一run_name查找日志，避免日期过滤漏读runtime。

## 2026-10-09 恢复主线：3.77125隔离候选正式结果

同SHA583e5252ce6a4dcd48ff40c01368507dfb680558ceb984dd53c59829d7fc380c，30/33，每点至少32回合，runtime通过。见artifacts/teacher_course_20261009/spring377125_formal_combined.json；仅当前阶段单seed保留，promoted=false，最终10.775及更高DR/全域仍未完成。

## 2026-10-10 3.77125正式运行中，下一档4.040625仅准备

原A/B进程PID22465/22529均确认存活，未重启。新spring4040625_from7550保留同7550 source、当前33命令、Dfb/b/C和DR.1，唯一拟议变化3.77125→4.040625Nm/腿；evaluate.py硬检查spring377125_formal_combined33/33、min32、runtime及SHA一致。脚本编译通过，尚未启动、无行为结论；最终10.775前不扩高速域。

## 2026-10-10 3.77125快筛33/33，直接正式保留验收

spring377125_from7550快筛33/33、runtime通过、无丢失/存活退化。保留冻结7550 SHA583e5252ce6a4dcd48ff40c01368507dfb680558ceb984dd53c59829d7fc380c，不追加无依据训练；已GPU0并行启动spring377125_formal_a/b，每点>=32，spring3.77125/DR.1/Dfb/b/C不变。完整33点结果未出，不升下一档，最终10.775/最高DR/全域仍未完成。

## 2026-10-09 spring3.77125当前域快筛结果

33/33，旧技能丢失0项、存活退化0项，runtime=True。仅4回合，promoted=false。66环境与前正式布局不同，不直接将差异归因成权重退化；按失败点修复或缩小弹簧步长。证据artifacts/teacher_course_20261009/spring377125_from7550/final_report.json；未扩高速域。

## 2026-10-10 7550完整33/33通过，接续3.77125

formal7550_combined.json确认同SHA583e5252ce6a4dcd48ff40c01368507dfb680558ceb984dd53c59829d7fc380c在spring3.501875/DR.1完整33/33每点>=32、runtime通过。两批原进程均已结束，无重复训练。已启动spring377125_from7550当前33项各>=4快筛，唯一物理变化spring3.501875→3.77125，Dfb/b/C、DR.1、原vx+yaw不变；最高10.775、全速度/新增圆弧阶跃和最终DR未完成，promoted=false。

## 2026-10-09 spring3.501875 model7550完整正式结果

33/33，每点最少32回合，runtime=True。见artifacts/teacher_course_20261009/spring3501875_positive14_ppo/formal7550_combined.json；仍未达到最终10.775弹簧、全域或最高DR。

## 2026-10-10 下一弹簧草案绑定7550，等待正式完整保留

A批PID20900已确认运行，B16/16通过。新spring377125_from7550仅准备，将spring3.501875→3.77125，固定SHA583e5252ce6a4dcd48ff40c01368507dfb680558ceb984dd53c59829d7fc380c与当前33命令，Dfb/b/C/DR不变；启动入口要求formal7550_combined33/33、min32、runtime及sourceSHA一致，旧trainer终态。脚本编译通过，尚未启动/未晋升，避免复用旧候选的下一档草案。

## 2026-10-10 7550正式B16/16通过，A17已接续

final7550_b每点>=32、runtime通过、无丢失或存活退化；+1.4 yaw_error.0433504/stable.621551、survival1。+1.2虽通过但stable.496028/yaw_error.0481555余量偏小，不能称全域高鲁棒。formal7550_a同SHA583e5252ce6a4dcd48ff40c01368507dfb680558ceb984dd53c59829d7fc380c原队列已启动PID20900；完整33点尚待A批，spring3.501875/DR.1不升档。

## 2026-10-09 spring3.501875 model7550中段正式结果

16/16，每点>=32，runtime=True；目标[{'vx': 1.0, 'yaw': 0.8, 'passed': True, 'survival': 1.0, 'yaw_error': 0.0185523814580911, 'stable_fraction': 0.986517558528428}, {'vx': 1.0, 'yaw': 1.0, 'passed': True, 'survival': 1.0, 'yaw_error': 0.028668483364143498, 'stable_fraction': 0.8909908026755853}, {'vx': 1.0, 'yaw': 1.2, 'passed': True, 'survival': 1.0, 'yaw_error': 0.04815546884185893, 'stable_fraction': 0.4960284280936455}, {'vx': 1.0, 'yaw': 1.4, 'passed': True, 'survival': 1.0, 'yaw_error': 0.043350385582965355, 'stable_fraction': 0.6215510033444817}]。详见artifacts/teacher_course_20261009/spring3501875_positive14_ppo/final7550_b/final_report.json。完整33点、最终10.775和最高DR尚未完成，promoted=false。

## 2026-10-10 7550正式验收条件接续准备

final7550_b原PID20086已确认存活，未重启；formal7550_a已配置仅当B16/16、runtime和retention通过才运行，固定同SHA583e5252ce6a4dcd48ff40c01368507dfb680558ceb984dd53c59829d7fc380c，A17×至少32回合、8000步/20分钟。当前spring3.501875/DR.1/Dfb/b/C保持，完整33项完成前不晋升，旧7480失败队列不启动。

## 跨日接续：7480未过，7550正式B批已启动

原训练/评估PID均不存在，日志完整；训练100更新、SimulationContext cleared，收尾status未更新，已备份旧status并据实补terminal/exit_code=null，不编造退出0。7480正式B15/16，唯一+1.4 yaw_error.052712/stable.485890，全部存活；A队列标记不启动。最终7550 SHA583e5252ce6a4dcd48ff40c01368507dfb680558ceb984dd53c59829d7fc380c冻结审计通过，仅expert3变化，final7550_b同B16/32回合验收已启动，spring3.501875/DR.1/Dfb/b/C不变。最终报告未完成，不升档、不重复训练。

## 2026-10-09 7480正式保留验收条件接续

mid7480_b原进程正常，formal7480_a准备同SHA91017763e66b1b3e7911e895dad48ca5183fecf655c27fae7f4428ce6eab7c71的A17×至少32回合，只有B16全过且runtime/retention通过才启动；合并严格检查同checkpoint和33项覆盖。spring3.501875/DR.1/Dfb/b/C保持，无升档。运行中不代表通过。

## 2026-10-09 3.501875单expert7480正式B批启动

30更新checkpoint7480 SHA91017763e66b1b3e7911e895dad48ca5183fecf655c27fae7f4428ce6eab7c71已冻结存档；参数核对只有expert3及预声明std变化，physics和隔离负向修复保持冻结。mid7480_b同B16布局/128env/每点>=32/8000步，Dfb/b/C、DR.1和spring3.501875保持。训练继续，无行为通过结论，B通过仍须A17全域保留验收。

## 2026-10-09 隔离候选正式32/33，接续仅+1.4 expert修复

同db663956…在spring3.501875/DR.1完整33项每点>=32、runtime通过，32/33全部存活；唯一vx1/yaw+1.4 yaw_error.058624/stable.376359失败，低速和隔离1.5/-.2修复保留。两批正常结束，启动spring3501875_positive14_ppo：仅expert3可训，冻结全部physics含新增anchor/局部门控及其他expert，sourceSHA db6639561ed90f80620c9098f8a893f4c314b7ef996a50c103ea4ee7342fd3c1；512env/100轮上限/10预热/每10保存，保留已有rehearsal。完整正式33点通过前不升弹簧/DR。

## 2026-10-09 恢复主线：3.501875隔离候选正式结果

同SHAdb6639561ed90f80620c9098f8a893f4c314b7ef996a50c103ea4ee7342fd3c1，32/33，每点至少32回合，runtime通过。见artifacts/teacher_course_20261009/spring3501875_isolated_formal_combined.json；仅当前阶段单seed保留，promoted=false，最终10.775及更高DR/全域仍未完成。

## 2026-10-09 恢复后正式验收source一致性核查

两批原进程已到1000步，尚未完成。CPU核验同db663956…checkpoint、隔离domain及冻结anchor均存在，A17+B16命令去重覆盖33项；源代码/策略哈希已存spring3501875_isolated_source_audit.json。此项只证明来源和覆盖配置一致，不代表行为通过，未启动下一档。

## 2026-10-09 用户明确恢复教师训练主线

用户“回归训练主线，接续任务”解除此前暂停。继续以最新用户修正为准：无vy/导航/75格，原vx+yaw/25D/6D，先当前域迁移弹簧至10.775并保留，再高速扩域。冻结隔离候选db6639561ed90f80620c9098f8a893f4c314b7ef996a50c103ea4ee7342fd3c1在3.501875/DR.1快筛33/33，现已GPU0并行启动A17/B16每点>=32正式验收，完整命令及契约分别在spring3501875_isolated_formal_a/b；无重复训练，无学生蒸馏/sim2sim。正式完成前不升档。

## 2026-10-09 已交付小MLP v8实机I/O对齐契约

按用户要求新增docs/h723_v8_hardware_io_alignment.md，锁定ONNX b866a53e…，详述模型q0=0的CAD姿态、active向前下42.8919°/hip向后下42.8966°铰接中心线、模型right/left与策略前方下物理左右镜像、25D/6D顺序/单位/极性、IMU FLU转换、编码器零位/减速比、电流与PD映射及首测逐项对拍。MuJoCo静态轴/零位核对及接口数值示例通过，证据artifacts/h723_io_alignment_20261009；不代表实机标定，未改部署权重/控制代码，训练保持暂停。

## 2026-10-09 局部隔离候选33/33快筛通过，按用户要求暂停

slow_negative_isolated_fit candidate SHA db6639561ed90f80620c9098f8a893f4c314b7ef996a50c103ea4ee7342fd3c1；冻结旧共享分支快照，平滑局部命令域内学习差值，保持原残差预算/25D/6D，无外部动作偏置。32项相关测试通过，600CPU拟合目标MAE最大.000113、保留样本输出差0、checkpoint重载一致；spring3.501875/DR.1同33项各>=4快筛33/33全部存活，无丢失、runtime通过。仅快筛，正式32回合/多seed/最终10.775/全域等仍未完成，promoted=false。用户明确“验证完后先停”，本次验证结束后不再启动任何训练或评估，等待问答。

## 2026-10-09 唯一慢负转向诊断可行，共享拟合未达线

slow_negative_probe中common+.010/diff+.005案例4回合survival1、speed_error.133196/yaw_error.031327/stable1，选为监督目标；其他旧门槛通过但speed_error>.2的案例不选。slow_negative_fit CPU600更新、20倍rehearsal仅训inner.frozen_reference_network；目标轮MAE.005646超过预设.003，retention平均最大.000536，冻结通过但fit_passed=false，未输出candidate，不做GPU盲测。32/33回退候选保留；下一步须隔离目标命令更新，避免广泛负向分支干扰。最终弹簧/全域/DR未完成。

## 2026-10-09 共享分支回退恢复32/33，只剩1.5/-0.2

rollback_reference_screen SHA6387a95cdef306042e743c99d0e961f80c01c2475e51da1aa534ace5be4b587f同3.501875快筛32/33、全部存活，原负向退化恢复且正向4项改善保留。唯一(1.5,-.2)速度偏高signed+.16276、yaw偏正+.04972，stable.3846。已启动slow_negative_probe12组微小轮共模/差模常值动作扰动含零对照，48env/每组>=4/捕获真实obs-action，诊断用于选择可行训练修正，不作为部署外环。原32项保留在source，最终无偏置候选仍需全33项验收，promoted=false。

## 2026-10-09 spring3.501875当前域快筛结果

32/33，旧技能丢失1项、存活退化0项，runtime=True。仅4回合，promoted=false。66环境与前正式布局不同，不直接将差异归因成权重退化；按失败点修复或缩小弹簧步长。证据artifacts/teacher_course_20261009/spring3501875_physics_ppo/rollback_reference_screen/final_report.json；未扩高速域。

## 2026-10-09 physics负向保留退化已停训并定位共享分支

7450快筛30/33，较7420的31/33新增(1.5,-.1)失败；(.5,-.1)/(1.5,-.2)仍失败，全部存活。本轮已SIGINT正常结束exit0、冻结审计通过，保留全部checkpoint。真实rollout单分支回退动作归因：三个负向缺口仅shared inner.frozen_reference_network发生输出变化；正向慢转弯/直行由其他分支承担，(.25,+.1)耦合两分支。新rollback_reference_screen仅将7450此分支恢复7390，其他改善保留，SHA6387a95cdef306042e743c99d0e961f80c01c2475e51da1aa534ace5be4b587f；33项同条件快筛已启动，未预设通过或晋升。

## 2026-10-09 spring3.501875当前域快筛结果

30/33，旧技能丢失3项、存活退化0项，runtime=True。仅4回合，promoted=false。66环境与前正式布局不同，不直接将差异归因成权重退化；按失败点修复或缩小弹簧步长。证据artifacts/teacher_course_20261009/spring3501875_physics_ppo/mid7450_screen/final_report.json；未扩高速域。

## 2026-10-09 physics7420快筛31/33，检查60更新趋势

7420同3.501875/66env快筛31/33全部存活，相对7390恢复(.5,0)、(.5,+.1)、(1,+.1)、(.25,+.1)，新增(.5,-.1)退化，(1.5,-.2)仍失败，promoted=false。7450(60更新)SHA909d9f66887185bad407461d619b774aac4c33289e06b66dcdb3c1fd2d753660已固定且冻结审计通过，mid7450_screen同条件33项快筛已启动。若负向退化扩大则停止共享分支当前更新并改保护，不因reward继续长训。

## 2026-10-09 spring3.501875当前域快筛结果

31/33，旧技能丢失2项、存活退化0项，runtime=True。仅4回合，promoted=false。66环境与前正式布局不同，不直接将差异归因成权重退化；按失败点修复或缩小弹簧步长。证据artifacts/teacher_course_20261009/spring3501875_physics_ppo/mid7420_screen/final_report.json；未扩高速域。

## 2026-10-09 真实rollout梯度通过，7420开始全33筛查

spring3501875_physics_ppo/attempt3第11轮当前动力rollout的5个失败命令各128obs均对指定分支有非零梯度，实际网络路径与预检一致。7420 SHA936656ebffb3ff3825a4ee286544a76712b72c2e3db1766cabb9d975e733598f仅4指定physics网络变化，冻结通过。mid7420_screen同66env/33项每项>=4/3.501875/DR.1已启动，检查多分支更新的全域retention；正式32回合尚未完成。

## 2026-10-09 spring3.501875嵌套physics适应已实际启动

spring3501875_physics_ppo/attempt3 PID1404591，source7390 SHAe07a3d1c88bdbbf50cb458581164973ca03fe67b584dd39a8d85229dbd41b1cb；仅4个预检有效physics子网络可训，其余冻结，512env/100轮/10critic预热/每10保存，第11更新留rollout证据。原33项及对称rehearsal均实际覆盖，runtime通过、冻结无异常。前两次初始化失败分别是课程数组长度不齐和奖励旧索引越界，均未更新，日志保留；修复保留完整82级数组，仅采样前23当前域阶段。Dfb/b/C与DR.1不变，弹簧3.501875，无高速扩域。尚未行为认证。

## 2026-10-09 嵌套physics优化入口修复及真实梯度预检

新增--physics_train_modules显式选择已有子网络，必须physics模式+reset_optimizer，拒绝任意base路径；修复外层gate0时错误清除内层有效梯度的审计分支逻辑，保留原残差限幅和结构。19项相关测试通过、编译通过。真实历史rollout每个缺口256obs梯度检查发现(.5,0)实际需inner.scoped_network，原inner.network被scope完全覆盖，已纠正；选outer.network、inner.scoped_network、inner.reference_network、inner.frozen_reference_network覆盖5项，最老物理分支/基座不选。证据spring3501875_physics_ppo/selected_branch_preflight.json；尚未启动新GPU训练，真实新阶段rollout梯度/retention仍需验证。

## 2026-10-09 同布局对照确认弹簧导致低速跟踪余量不足

同7390/66env/seed/4回合，spring3.2325为33/33，3.501875为28/33，3.77125为27/33，均存活；本次新增缺口确由弹簧增量暴露，非布局混淆。(.5,0)原速度误差.191已接近旧.2门槛，不能称10%精度通过。失败5项跨outer.physics.network、inner.network、inner.reference_network、inner.frozen_reference_network；默认仅outer.network的physics optimizer在多项gate为0。已保存branch_gate_inventory.json和adaptation_decision.json，下一轮要显式选择生效分支并先验证实际梯度/冻结/rehearsal，不继续无效大转弯expert更新；暂不升弹簧/DR/高速域。

## 2026-10-09 同66环境弹簧基线归因完成

7390在spring3.2325的同布局快筛33/33。与3.501875/3.77125逐项比较见artifacts/teacher_course_20261009/spring32325_layout66_baseline/matched_comparison.json；不把换环境数量导致的采样差异直接归因成弹簧或网络退化，promoted=false。

## 2026-10-09 spring3.501875当前域快筛结果

28/33，旧技能丢失5项、存活退化0项，runtime=True。仅4回合，promoted=false。66环境与前正式布局不同，不直接将差异归因成权重退化；按失败点修复或缩小弹簧步长。证据artifacts/teacher_course_20261009/spring3501875_from7390/final_report.json；未扩高速域。

## 2026-10-09 3.77125缺口扩散，减半弹簧增量

spring377125_from7390快筛27/33，全部存活；失败(.5,0)、(.5,+.1)、(1,+.1)、(1,+1.4)、(1.5,-.2)、(.25,+.1)。不把广泛失败归结单expert，也不盲目扩训；同7390/33布局/seed下改测spring3.501875(从通过3.2325的增量减半)。Dfb/b/C和DR.1不变，spring3501875_from7390已启动4回合筛查，正式晋升仍需完整32回合。最终10.775前不扩高速域。

## 2026-10-09 spring3.77125当前域快筛结果

27/33，旧技能丢失6项、存活退化0项，runtime=True。仅4回合，promoted=false。66环境与前正式布局不同，不直接将差异归因成权重退化；按失败点修复或缩小弹簧步长。证据artifacts/teacher_course_20261009/spring377125_from7390/final_report.json；未扩高速域。

## 2026-10-09 spring3.2325完整33/33，推进3.77125

7390同SHAe07a3d1c88bdbbf50cb458581164973ca03fe67b584dd39a8d85229dbd41b1cb的A17+B16完整33/33、每点>=32、runtime通过。对话中断后收尾父进程缺失，A子进程已写完整JSON并退出；人工重跑CPU审计合并，status标记recovered/exit_code=null，未编造退出0。原正向trainer112更新后请求SIGINT且PID消失，保留最近7470及冻结审计，后期checkpoint未行为认证；下一档只用已通过7390。spring377125_from7390已启动33项4回合快筛，唯一物理变化spring3.2325→3.77125，Dfb/b/C与DR.1不变，不扩高速。原7360下一档草案继续禁用。

## 2026-10-09 7390下一档独立草案已备，未启动

当前B16/16通过，A17原进程正在验收；PPO66更新冻结检查零异常。为减少完成后的空档，新建spring377125_from7390，sourceSHAe07a3d1c88bdbbf50cb458581164973ca03fe67b584dd39a8d85229dbd41b1cb；启动硬检查formal7390完整33/33、min32、runtime、sourceSHA和旧trainer终态。仅当前33项快筛、不扩高速；旧spring377125(7360)仍禁止启动。新脚本编译通过，暂无本档完整通过结论。

## 2026-10-09 spring3.2325 model7390中段正式结果

16/16，每点>=32，runtime=True；目标[{'vx': 1.0, 'yaw': 0.8, 'passed': True, 'survival': 1.0, 'yaw_error': 0.024468267243044035, 'stable_fraction': 0.9346780936454849}, {'vx': 1.0, 'yaw': 1.0, 'passed': True, 'survival': 1.0, 'yaw_error': 0.022846888538985748, 'stable_fraction': 0.9688545150501672}, {'vx': 1.0, 'yaw': 1.2, 'passed': True, 'survival': 1.0, 'yaw_error': 0.0387340150150567, 'stable_fraction': 0.6212374581939799}, {'vx': 1.0, 'yaw': 1.4, 'passed': True, 'survival': 1.0, 'yaw_error': 0.04800457778981697, 'stable_fraction': 0.551316889632107}]。详见artifacts/teacher_course_20261009/spring32325_positive_ppo/mid7390_b/final_report.json。完整33点、最终10.775和最高DR尚未完成，promoted=false。

## 2026-10-09 Dfb/b/C口径核对

当前runtime实际采样Dfb[.0490011,.0509987]、b[.00190049,.00229936]、C[.000950226,.00139997]，分别位于声明[.049,.051]/[.0019,.0023]/[.00095,.0014]；无配置变化。此前“机械D”就是b，简报省略C不表示删除库仑项。后续统一Dfb/b/C，证据spring32325_positive_ppo/damping_triplet_readback.json。

## 2026-10-09 7390同checkpoint保留验收已配置条件接续

mid7390_b原进程运行，formal7390_a队列只在B16/16、runtime通过且无retention/存活回退时启动。A17每点>=32，固定同SHAe07a3d1c88bdbbf50cb458581164973ca03fe67b584dd39a8d85229dbd41b1cb，spring3.2325/DR.1不变；合并检查33项唯一覆盖。暂无行为新结论，3.77125仍禁止启动旧草案。

## 2026-10-09 正向修复7390开始B批正式验收

source7390 SHAe07a3d1c88bdbbf50cb458581164973ca03fe67b584dd39a8d85229dbd41b1cb，CPU检查只有expert3/5及预声明exploration std变化，负向修复冻结。mid7390_b优先16项/128env/每点>=32/8000步/20分钟，spring3.2325、DR.1/D不变，PPO继续。B通过后再补A17，现阶段不晋升弹簧。完整命令和冻结审计已存档。

## 2026-10-09 退役失效source的下一弹簧草案

先前spring377125准备绑定7360，但该source在3.2325正式31/33，不能继续使用其草案。已显式launch_allowed=false并在evaluate.py加启动拒绝；不是取消最终目标，而是下一档必须从本轮正向修复后新通过的checkpoint新建实验。当前正向PPO原进程继续，固定30更新候选先B16(+.8/+1.4)、通过后A17的计划已保存。

## 2026-10-09 正向修复采样覆盖确认

spring32325_positive_ppo实际rollout包含vx1的0、±.8/±1/±1.2/±1.4，无遗漏，冻结检查正常。GPU0另有进程1339380共享，当前轮时约20s，未操作其他任务。下一中段优先B16(+.8/+1.4所在批次)，固定>=30更新checkpoint后再验收；当前预热不产生行为通过结论。

## 2026-10-09 spring3.2325正式31/33，转正向局部修复

7360完整33项每点>=32，31/33通过，全部存活；失败vx1/yaw+.8误差.051328/stable.5324，+1.4误差.058655/stable.3772。负向-1.2已正式通过，不继续用负向expert盲训。旧训练113更新后SIGINT正常结束；保留未验收后期权重但接续用7360 SHAec9410cbae0e14a95a30ea05356c85e6224cb333cd786f457df395d5754ce3f8。新spring32325_positive_ppo只更新expert3(+1.4)/5(+.8)，冻结其他分支，增加+.8 rehearsal；D/DR/弹簧3.2325不变，gate预检通过。3.77125未启动，必须本档完整通过才推进。

## 2026-10-09 spring3.2325 model7360完整正式结果

31/33，每点最少32回合，runtime=True。见artifacts/teacher_course_20261009/spring32325_ppo/formal7360_combined.json；仍未达到最终10.775弹簧、全域或最高DR。

## 2026-10-09 准备spring3.77125，前置条件未满足不启动

当前spring3.2325的A17通过，B16原进程仍运行。下一档spring377125仅准备当前33命令4回合筛查，source固定7360 SHAec9410cbae0e14a95a30ea05356c85e6224cb333cd786f457df395d5754ce3f8，唯一动力变化spring3.2325→3.77125，D/DR/接口不变。evaluate.py启动前要求formal7360_combined33/33、min32、runtime通过、同SHA及旧trainer终态；尚未启动。编译检查通过。

## 2026-10-09 spring3.2325 model7360中段正式结果

17/17，每点>=32，runtime=True；目标[{'vx': 1.0, 'yaw': -1.4, 'passed': True, 'survival': 1.0, 'yaw_error': 0.044714107595444634, 'stable_fraction': 0.6595317725752509}, {'vx': 1.0, 'yaw': -1.2, 'passed': True, 'survival': 1.0, 'yaw_error': 0.018765709335246954, 'stable_fraction': 1.0}, {'vx': 1.0, 'yaw': -1.0, 'passed': True, 'survival': 1.0, 'yaw_error': 0.030313814603365383, 'stable_fraction': 1.0}]。详见artifacts/teacher_course_20261009/spring32325_ppo/mid7360_a/final_report.json。完整33点、最终10.775和最高DR尚未完成，promoted=false。

## 2026-10-09 spring3.2325同checkpoint正式验收自动接续

mid7360_a原进程正在验收；准备formal7360_b条件队列，只有A17全部通过、runtime与retention无回退才启动B16，合并严格校验同SHA及33项唯一覆盖。固定7360 SHAec9410cbae0e14a95a30ea05356c85e6224cb333cd786f457df395d5754ce3f8，B每项>=32、128env、8000步/20分钟，不改动力/命令。未完成报告前不晋升。

## 2026-10-09 spring3.2325 model7360开始A批正式验收

source7360 SHAec9410cbae0e14a95a30ea05356c85e6224cb333cd786f457df395d5754ce3f8；约30更新后CPU核对仅负向expert7改变，冻结检查通过。mid7360_a优先含失败-1.2的17点/119env/每点>=32/8000step，spring3.2325/DR.1/D不变，PPO继续。A通过再补B16，当前不可声称33项通过。

## 2026-10-09 spring3.2325验收顺序按失败点调整

负向-1.2位于formal_a的17点批次，下一中段先验A而非沿用正向修复的B；A通过后才补B16并合并同一checkpoint33项。计划保存在spring32325_ppo/evaluation_order.json，尚未启动新评估。当前训练7更新仍预热，runtime/冻结/实际命令覆盖通过，不能宣称行为改善。

## 2026-10-09 spring3.2325仅负向-1.2缺口，启动定点PPO

当前33项快筛32/33，全部survival1；唯一vx1/yaw-1.2 yaw_error.0567259/stable0，旧正向修复保留。启动spring32325_ppo，source7330 SHA2a3dab780ea0c876fbb3bbe7f1ee3b7f7476755a9015c899a377169c02f109c7，spring3.2325/DR.1/D不变，只更新existing expert7(-1.2)，gate预检正向0/负向1，其他actor冻结。512env/最多200更新/10预热/每10保存，正式33点通过可提前结束，不按轮数晋升。高速域等待10.775弹簧适应规则不变。

## 2026-10-09 spring3.2325当前域快筛结果

32/33，旧技能丢失1项、存活退化0项，runtime=True。仅4回合，promoted=false。66环境与前正式布局不同，不直接将差异归因成权重退化；按失败点修复或缩小弹簧步长。证据artifacts/teacher_course_20261009/spring32325/final_report.json；未扩高速域。

## 2026-10-09 spring2.69375正式通过，推进3.2325当前域筛查

model7330完整33/33每点>=32且runtime通过，SHA2a3dab780ea0c876fbb3bbe7f1ee3b7f7476755a9015c899a377169c02f109c7。训练在117更新后SIGINT正常结束exit0，冻结检查通过；后期checkpoint不替代已验收中段候选。下一档仅改spring3.2325，当前33命令/66env/每点>=4/DR.1，未扩高速。增量从最终弹簧2.5%试探到5%，失败则修复或缩步，不自动晋升。原10.775前禁止高速扩训规则保持。

## 2026-10-09 spring2.69375 model7330完整正式结果

33/33，每点最少32回合，runtime=True。见artifacts/teacher_course_20261009/spring269375_ppo/formal7330_combined.json；仍未达到最终10.775弹簧、全域或最高DR。

## 2026-10-09 spring2.69375 model7330中段正式结果

16/16，每点>=32，runtime=True；目标[{'vx': 1.0, 'yaw': 1.0, 'passed': True, 'survival': 1.0, 'yaw_error': 0.020781068897566268, 'stable_fraction': 0.9902801003344481}, {'vx': 1.0, 'yaw': 1.2, 'passed': True, 'survival': 1.0, 'yaw_error': 0.0278412522280894, 'stable_fraction': 0.9275710702341137}]。详见artifacts/teacher_course_20261009/spring269375_ppo/mid7330_b/final_report.json。完整33点、最终10.775和最高DR尚未完成，promoted=false。

## 2026-10-09 弹簧适应30更新后开始正式中段验收

spring269375_ppo已生成model7330，SHA2a3dab780ea0c876fbb3bbe7f1ee3b7f7476755a9015c899a377169c02f109c7；CPU权重核对仅指定expert2/6变化，冻结检查通过。mid7330_b复用16点/128env/seed4567/每点>=32，spring2.69375、DR.1/D不变；PPO继续，行为仍待报告。修正两个未运行的全33点备用命令模板：33×4=132超过128场景上限，改为33×3=99及18000步；实际正式评估继续使用17/16两批。

## 2026-10-09 model7300完整33/33，转弹簧适应训练

formal7300_combined.json证实33/33、每点>=32、runtime通过，SHA3792d5a1c3a3851af7da31c669cd930fdddf3045938a710a1e5ecc7fdd4a947c。为优先弹簧，当前单点PPO在139更新后SIGINT正常退出0并保留全部checkpoint；后期final checkpoint未行为验收，不替代已通过的7300。新spring269375_ppo从7300接续，512env/最多200更新/10预热，弹簧2.69375，只更新+1.0/+1.2现有expert，D/DR保持。进入高速域必须先适应10.775并通过retention；当前圆弧/动态阶跃/最高DR/多seed未完成。

## 2026-10-09 model7300正式33项结果

33/33，最少32回合，runtime=True；见artifacts/teacher_course_20261009/ppo_yaw12/formal7300_combined.json。当前弹簧2.424375/DR.1，单seed原行为门槛，新增圆弧/阶跃/最终动力尚未认证。

## 2026-10-09 最新顺序纠正：最终弹簧先于高速域

用户明确：进入高速域之前，应已适应10.775N·m/腿弹簧。执行顺序为当前命令域稳定→在当前命令域内逐级迁移弹簧至10.775并通过retention→再扩展高速域。撤销先前“不等10.775就拓展高速”的解释；当前2.69375候选适应继续，D/DR暂保持当前档，禁止以低弹簧阶段结果启动高速域扩训。原vx+yaw/25D/6D及撤销vy/导航/75格要求不变。

## 2026-10-09 弹簧下一档适应范围已定位

spring2.69375快筛31/61，旧33仅丢失vx1/yaw+1.0、+1.2，均survival1；无存活退化。已准备spring269375_ppo，只更新existing expert6(+1.0)和2(+1.2)，提高+1.0 rehearsal权重，目标gate预检均有效，其他分支冻结。尚未启动：先等model7300完整33项正式验收及当前PPO终态，禁止重复训练；当前D/DR不扩大。

## 2026-10-09 弹簧2.69375快筛结果

31/61通过，旧33快筛技能丢失2项，存活退化0项；runtime=True。仅4回合筛查，promoted=false。见artifacts/teacher_course_20261009/spring269375/final_report.json；不得将此结果当正式弹簧适应完成。

## 2026-10-09 PPO中段闭环16/16，+1.2得到实际改善

model7300 SHA3792d5a1c3a3851af7da31c669cd930fdddf3045938a710a1e5ecc7fdd4a947c；mid7300同128环境/seed4567/每点>=32回合16/16通过，runtime通过，无旧技能/存活回退。目标vx1/yaw+1.2 yaw_error .053812→.029743、stable .432483→.882839，survival1。只说明当前子集改善，剩余17点formal7300_a排在spring269375后补齐；PPO200轮仍运行。世界轨迹640段完整性错误0，圆弧门槛仅9段通过，固定命令无动态step段，不得将旧门槛16/16当新增圆弧/阶跃通过。

## 2026-10-09 用户要求加快弹簧迁移：与命令拓展交错

不等待最终10.775弹簧才启动高速域。保留当前PPO独立+1.2修复，排队spring269375在mid7300评估结束后运行：v597冻结、spring2.424375→2.69375唯一物理变量，DR.1/D不变，全61命令各至少4回合检查旧33及存活。排队不等于已通过或已完成迁移；正式32回合和多seed要求不变。已保存完整命令、契约、进程依赖及结果自动记录脚本；同时最多本主线训练+一项评估。

## 2026-10-09 中段checkpoint闭环对照已启动

当前PPO继续运行，累计36更新时冻结/命令覆盖检查通过。冻结model_7300.pt副本SHA3792d5a1c3a3851af7da31c669cd930fdddf3045938a710a1e5ecc7fdd4a947c，teacher_course_20261009/ppo_yaw12/mid7300并行复用formal_b的16命令/128环境/seed4567/每点32回合，动力和DR.1不变。开启只读world trajectory记录，进程已进入物理rollout；真实轨迹完整性与行为结果仍待最终JSON，不能把运行成功算指标通过。完整33点及后续checkpoint仍需验收。

## 2026-10-09 当前有效范围确认：撤销vy及导航扩展

本次新增的独立vy、世界二维速度command、导航/航点/路径规划、vx×vy×yaw的75格验收均已撤销，不进入训练、接口或验收。保留原body vx+yaw、25D/6D、全速度压弯和6–8rad/s自转；新增仅保留纵向稳态/阶跃响应、世界轨迹记录与圆弧横向偏差等只读行为评估，原survival/retention/多seed/DR门槛继续。世界坐标仅用于测量，不构成导航或外环。当前训练command与所有网格已审计；旧自动goal文本中的vy/75格条目已被本次用户指令明确废止，不得执行。证据artifacts/teacher_course_20261009/scope_correction.json。

## 2026-10-09 PPO已产生实际actor更新，轨迹记录通道接入

model_7280.pt SHA4a71e62a262ca7ca2651c2a7c1a5d60fd486ef49aa7304d7aa62716daa9e746e；CPU比对确认expert2权重已改变，其余actor仅预先声明的exploration std覆盖，冻结检查通过。训练继续200轮预算，不以reward或新checkpoint宣称通过。evaluate_rmuc_grid新增可选--record-world-trajectory，记录pre-step与pre-reset终端世界xy/heading/速度/age/原vx+yaw，保存哈希NPZ，按reset和命令变化分段；仅实际同episode命令阶跃才算step，固定网格不会冒充动态阶跃。新rmuc_trajectory_trace.py和测试合计8项通过，编译通过；真实GPU记录路径尚待下一评估验证。未改command/obs/action或训练进程。

## 2026-10-09 对照排除策略退化，启动鲁棒性PPO

同128环境/seed4567/32回合，v588与v597在vx1/yaw+1.2结果完全一致：survival1、yaw_error.0538120844、stable.432483278；不是新增负向修复造成回退，旧通过批次为120环境且DR采样不同。停止重复对照，启动teacher_course_20261009/ppo_yaw12：source v597 SHA925c3ec2e1f673cd8c3b60433166499da632adc7bf9beebcfbd6549cce9442f7，512环境/200更新/10critic预热/每10更新保存，只训local expert2，其余网络冻结。保持DR.1、Dfb.05、b.002及各自当前采样范围，弹簧2.424375、5.5峰值/3持续，旧命令rehearsal。命令/契约/完整33点32回合验收模板已保存；运行中不代表技能改善，promoted=false。

## 2026-10-09 matched baseline完成

旧v588同16命令/seed/32回合对照完成；相对v588的v597失败回退点[]。详见artifacts/teacher_course_20261009/baseline_b/paired_comparison.json；promoted=false，下一步Investigate common sampling-sensitive boundary; do not promote on quick screen。

## 2026-10-09 正式retention失败定位，暂停升DR

v597正式32/33，vx1/yaw+1.2存活100%但yaw误差.053812、stable .432483失败；快速4回合曾通过，不能用快速结果晋升。准备baseline_b：仅checkpoint换v588，复用formal_b全部16命令/8副本/seed4567/32回合/相同DR.1与动力，区分策略回退与采样敏感性；不启动无依据长训。新轨迹指标仍未行为认证。

## 2026-10-09 teacher主线33项正式结果

32/33，最少32回合，见artifacts/teacher_course_20261009/combined_report.json；新增世界速度/阶跃/圆弧/姿态指标尚未认证。

## 2026-10-09 补充纠正：保留轨迹/圆弧/阶跃验收，不新增command

用户明确世界轨迹、圆弧偏差、阶跃响应仍需验收；仅撤回新增横向命令与75格扩展，不改变现有body vx+yaw、25D/6D。上一节撤回全部扩展评估的表述过宽，以本节为准。新scripts/rmuc_trajectory_metrics.py基于原vx/yaw解析参考直线/圆弧；几何径向偏差与时间对齐误差分别记录，避免将轨迹相位滞后误作横向误差。阶跃记录稳态误差/超调/到达时间；零目标.05m/s和连续保持.2s为显式暂定定义。5项测试通过，尚未实际接入评估，不代表行为通过；正在跑的teacher正式33项command无变化。

## 2026-10-09 用户纠正：保持原command与原课程

撤回本轮新提出的横向/世界轨迹及扩展验收草案，相关未接入模块、测试和75格文件已删除；不修改command，25D/6D及原body vx+yaw语义保持。继续原全速度压弯、渐进D/DR→命令域→弹簧课程，保留两周截至10/23和双向持续6–8rad/s自转目标，不再要求10。常规蒸馏/sim2sim仍暂停，用户要求部署新版再验收学生。正式v59733项两批原进程未改command，不重启；旧技能retention与真实行为门槛保持。goal工具先前文本包含已撤回要求，执行必须以用户本次纠正及本节为准。

## 2026-10-09 用户授权先集成，桌面小MLP与最快已测直行demo交付

新目录/home/kellen/桌面/RMUC_H723_smallMLP_v8_20261009，policy.onnx SHA b866a53ecdb186e3adb3db38c04e971ec9bbec1360605ffbb97dd86f7027713f，20614参数/82456B权重，正式Isaac31x32通过。默认canonical MJCF筛12个直行命令，实测最快且未倒选1.5m/s；20秒连续视频实测末10秒vx1.38426/yaw-.02921，全程未触发tilt/height终止，最大tilt9.65deg。1.6未倒但仅.914m/s，1.7及以上测试倾倒，.5亦失败，不能外推全速度。文件夹含视频/ONNX/PT/100输入/控制参考/验收JSON/README/哈希清单，截图检查正常。用户先部署授权不等于sim2real认证，stage spring2.424375/DR.1，完整sim2sim/CubeAI/板测未完成，未修改默认MJCF或旧桌面包。

## 2026-10-09 摩擦锥诊断结束：局部改善，未晋升

elliptic六项均12秒未倒；.5/+.3实测vx.47753/yaw.34844，1/-.8实测.98498/-.85215，负向/低速部分改善。但1/+.8 yaw1.11122、1/+1.4 yaw1.73739，仍绝对跟踪失败。decision.json明确passed=false/promoted=false；不能按单点成绩选择solver掩盖模型差异。所有诊断已结束，默认MJCF未改，小MLP正式Isaac31/31x32证据保留，跨引擎验收仍未通过。下一步聚焦正负倾斜下接触和闭环约束响应。

## 2026-10-09 sim2sim全模型审计与接触数值收敛

full_model_inventory.json：刚体全对应，最大质量差5.67e-7kg，主惯量差2.9e-7，关节轴差2.22e-16、锚点差5.03e-7m，未发现左右轴/姿态建模错误。已改善eq/contact阻抗下1ms复测6项12秒均未倒，速度最大变化.00175m/s；小yaw变化约.012rad/s，大转弯仍未合格，不能由时序解释。接续独立mujoco_elliptic只切摩擦锥pyramidal→elliptic，摩擦系数/几何/质量不变，原MJCF不覆盖；等待结果后判断摩擦求解近似影响。v8正式Isaac31x32通过保持，sim2sim尚未通过，不交付。

## 2026-10-09 sim2sim接触阻抗及稳态响应归因

速度口径核对：Isaac root_lin_vel_w是COM速度，MuJoCo mj_objectVelocity(BODY)数值微分验证同为COM，排除此项。主轮USD/MJCF半径.06/宽.033一致；名义摩擦.5一致。独立mujoco_contact只改geom solimp=.999，保持已诊断equality设置；17类mass/inertia/geometry/joint/friction/actuator数组与父诊断精确一致，六项均12s未倒，.5/+.3 vx.47346/yaw.38209，但+1.4 yaw1.75813仍不合格。未改canonical。steady_state_comparison.json显示正转弯MJ速度下降伴两轮转速降低及hip差约.03rad，需继续动力归因；不能只改yaw外环掩盖。v8正式Isaac31x32通过不变，sim2sim尚未通过，不交付。

## 2026-10-09 闭环约束阻抗诊断：修复反向倾倒，yaw仍超调

独立MJCF仅eq_solimp改.999，其余mass/inertia/geom/joint/actuator数组精确一致。v8 .5/+.3从反向倾倒变12s未倒、vx.46978/yaw.41313，闭环间隙最大.07477mm（原1.41mm；Isaac.11461mm）。其他条件速度多数改善但yaw仍超调，六项均12s未倒不等于行为通过。证据h723_feedback_audit/mujoco_constraint/comparison_report.json。未替换canonical MJCF或桌面产物，不能通过改solver隐藏问题；需继续审计其余接触/动力差异。v8正式Isaac31/31x32已通过，压缩门槛与跨引擎验收分开记录。

## 2026-10-09 v8正式31/31通过；sim2sim物理归因推进

v8正式31项各32回合全通过，动力及mass/inertia/motor读回审计通过；标准20614参数学生Isaac阶段验收达标。accepted export仅Isaac已验收，deployable=false，CubeAI/板测未完成。DR0名义Isaac六点全通过，第一拍动作/力矩与MuJoCo吻合，但2ms后关节速度分离；机身质量/惯量及抽查各连杆质量/惯量吻合。闭环约束误差Isaac最大.115mm/MuJoCo1.41mm，独立MJCF只改eq_solimp至.999，保持几何/质量/惯量/关节/执行器数值逐数组精确一致，诊断6点运行中。canonical rmuc_peak55.xml未修改，不能先认定归因或成功。

## 2026-10-09 v8正式闭环结果

保留31/31，每点至少32回合，存活退化0项，Isaac正式gate=True。实际动力/mass/inertia/motor读回检查通过；报告h723_distill_v8/formal_acceptance_report.json。单seed、阶段弹簧/DR.1稀疏延迟仍非最终DR认证；学生MuJoCo/CubeAI/实机板测必须独立通过。

## 2026-10-09 sim2sim归因：接口通过，自接触/时序非已证实主因

v8 MuJoCo5项12秒未倒，.5/+.3反向加速后倾倒，跟踪未过。Isaac self-collision=false/MJCF显式pair差异确实存在，但失败轨迹774个policy快照无自接触，不能归因为自碰撞。1ms与2ms对照稳定条件速度差<.001，失稳复现；1ms运行actual dt/scheduler为1ms，但源contract low_level_dt元数据遗留2ms已注明，不作部署证明。正式Isaac32回合原进程仍运行；启动DR0名义Isaac6点早期物理/观测捕捉，与MuJoCo名义对照，找首次响应偏差。无几何/质量/惯量改动，不调参掩盖差异。

## 2026-10-09 v8首次31/31快筛，正式32回合及小MLP MuJoCo启动

v8全61点4回合exit0，旧31全部保留，无存活退化；反馈邻域标签改善产生闭环证据。正式formal_a/b同SHA9bfd53023d339722993b07525d551f8c360feb20ddedf5e5c45a1edc0dd2ba57、31项各>=32回合并行已启动，finish_formal.py自动审计/合并。静态ONNX3Gemm+2Relu、100输入误差1.52e-6，诊断非部署文件；同契约MuJoCo六项正在跑。接口audit1024组观测与实际训练源码差0、consumer差0、JIT/ONNX2.38e-6通过；该证据不等于物理一致或行为认证。最终弹簧/最高DR/正式sim2sim/CubeAI/板测未完成。

## 2026-10-08 H723 v8闭环快筛结果

保留31/31，存活退化0项，报告h723_distill_v8/acceptance_report.json。仅4回合，不是32回合正式验收；小MLP的MuJoCo对照与CubeAI板测仍未完成，promoted=false。

## 2026-10-09 反馈律误差归因与v8重标邻域训练

反馈扰动审计artifacts/h723_feedback_audit/feedback_response.json：首版倾斜响应约教师一半，历史动作响应偏强；v7改善但腿速度扰动轮响应误差仍明显。仅支持局部反馈拟合不足，不声明唯一根因。v8用Isaac状态生成548864条小扰动教师重标标签，保留全部旧数据/20,614参数/lr1e-4/12000步，不使用MuJoCo训练数据，改善反馈而非继续只堆rollout。controller OnnxReplayController同样残留obs/actions硬编码已修复为读取实际端口；100输入ONNX动作差0，10项controller测试通过。v7仍未认证，v8训练后需全域闭环，不提前交付。

## 2026-10-09 v7实际验收30/31拒绝，修复汇总教师路径

v7评估exit0/131.94秒，自动汇总因广义字符串版本替换误指continuation_v788失败；已修复所有受影响finish_evaluation.py为固定教师v588，并直接重算原结果，无重复仿真。v7保留30/31(96.77%)、总通过31，不能把新通过点抵消旧技能丢失；1/-.3 yaw.0500709/stable.5失败，静止存活.5较教师退化，passed=false。当前无运行训练/评估。v5仍30/31无存活退化但sim2sim跟踪失败，未有可交付策略。正式32回合、MuJoCo绝对行为、CubeAI/板测均未完成。

## 2026-10-08 v6仍30/31，无生存退化；v7低lr收敛

v6快筛exit0/133.5秒，30/31，无存活率退化；+1已修复，+1.2 yaw.04648但stable.44064未过，拒绝晋升。失败点在相邻命令移动，停止重复定点采数。v7从v6 SHA433cf57d44ba58f2201fc85a0599504b5a0c219d1ad5e4ba3087b745e62990b0，用相同3427840条数据/网络/12000更新，仅lr1e-3→1e-4做收敛；命令training_command.json，未行为验证不认定成功。小MLP正式sim2sim/CubeAI/板测仍未通过，最终目标未完成。

## 2026-10-08 DAgger v6训练完成待闭环

学生rollout冻结教师同状态标签278528条，与首轮教师数据合并，学生结构20614参数不变，12000更新完成；最佳留出MSE=2.6943323810871928e-05。必须读取h723_distill_v6训练报告再作闭环复验；promoted=false，未交付合格网络。

## 2026-10-08 v5 MuJoCo六项存活但跟踪仍拒绝，v6针对剩余yaw1

v5六项12秒未倒，.5/+.3实测vx=-.35560仍反向，1m/s各项实际vx约.65–.83，未通过绝对行为验收；teacher_comparison.json已保存。不以MSE或存活替代跟踪。v6准备1m/s yaw+.8/+1/+1.2及-.3保护DAgger，278528条，合并所有旧数据，固定标准MLP与动力，随后全61及正式回合。不宣称教师已有跨引擎缺口能由模仿自动消除。

## 2026-10-08 v5闭环30/31无存活退化，MuJoCo对照接续

v5全61点4回合exit0/129.39秒，30/31(96.77%)，全域无存活率退化。仅1/+1未过：yaw误差.05355/stable.54933/存活1，前轮两慢转恢复。仍未满足全部必需技能/32回合门槛。v5静态ONNX100输入对拍误差1.4454e-6通过，diagnostic_only=true/deployable=false；同契约小MLP六项MuJoCo对照启动。正式sim2sim/CubeAI/板测未认证，未交付。

## 2026-10-08 H723 v5闭环快筛结果

保留30/31，存活退化0项，报告h723_distill_v5/acceptance_report.json。仅4回合，不是32回合正式验收；小MLP的MuJoCo对照与CubeAI板测仍未完成，promoted=false。

## 2026-10-08 v5训练完成并接续闭环；跨引擎状态差异审计

v5采数249.5秒/训练8.38秒exit0，合计3149312条，留出MSE3.2032e-5，学生SHAab217f727c40175142207ef1ec59ebfbe134d811da39ae36f0411e48b9032f8c；全61点4回合已启动，原31项与全域存活rate闸门保持。另在v4 MuJoCo轨迹同观测重算教师，部分动作差异最大.23977，远大于Isaac留出平均；证据v4/mujoco/same_state_teacher_audit.json。不能据此断言唯一根因，教师本身同契约跟踪也未过。需同时解决压缩泛化及跨引擎绝对行为；正式sim2sim/CubeAI/板测均未认证。

## 2026-10-08 DAgger v5训练完成待闭环

学生rollout冻结教师同状态标签278528条，与首轮教师数据合并，学生结构20614参数不变，12000更新完成；最佳留出MSE=3.203207183163547e-05。必须读取h723_distill_v5训练报告再作闭环复验；promoted=false，未交付合格网络。

## 2026-10-08 v4小MLP MuJoCo六项未倒，但跟踪拒绝

v4六探针全12秒未倒，含此前失败1/-1.4；非行为验收通过。1m/s实测约.66–.82，.5/+.3平均vx=-.34347(反向)，yaw.19444。teacher_comparison.json保存同契约逐项比较，必须保留绝对跟踪缺口。v5准备余下.5/-.1、1/-.3及历史退化保护1/+1.2、1.5/+.1 DAgger，共278528条，合并之前全部数据续训，固定20614参数；随后全61点检查。Isaac29/31仍仅4回合，正式32回合/全域MuJoCo/CubeAI/板测均未完成，未交付。

## 2026-10-08 H723 v4保留29/31、无存活退化，MuJoCo诊断启动

v4 Isaac61点4回合exit0，29/31(93.55%)，无存活退化；.5/-.1 yaw.05194 stable.1112和1/-.3 yaw.05284 stable.25未过，均存活1。尚非正式32回合或交付认证。v4诊断ONNX静态Gemm/Relu、100输入误差1.2815e-6通过，deployable=false；同契约MuJoCo六项准备对照教师，重点检查前版1/-1.4倾倒。正式sim2sim、CubeAI/实机时序仍待验收。

## 2026-10-08 H723 v4闭环快筛结果

保留29/31，存活退化0项，报告h723_distill_v4/acceptance_report.json。仅4回合，不是32回合正式验收；小MLP的MuJoCo对照与CubeAI板测仍未完成，promoted=false。

## 2026-10-08 v2小MLP MuJoCo对照：5/6未倒仍失败

修复端口后六项正常执行：1/+.8、1/-.8、1/+1.4、.25/+.1、.5/+.3跑满12秒；1/-1.4在5.312秒倾倒，教师该项12秒未倒，故学生sim2sim拒绝。相比教师.5/+.3倾倒，学生该项未倒，但不能抵消另一项退化。详细速度/yaw对照h723_distill_v2/mujoco_retry/teacher_comparison.json；不是5/6行为通过。v4 Isaac快筛仍运行。

## 2026-10-08 v4闭环运行，小MLP sim2sim接口修复

v4采数263.73秒/续训8.41秒均exit0，学生SHA3a4e14fa8eb02f00564cb5c82bcc47cc6e8c9e8e98bf02579344917028551004，61点4回合快筛已启动，finish_evaluation.py自动审计。v2诊断ONNX静态1x25、3Gemm+2Relu、100输入最大误差1.43e-6通过，明确deployable=false。首轮MuJoCo六项均因硬编码obs/actions与学生observation/action端口不一致报错，非行为失败；run_mujoco_probe.py改为读取模型端口名，mujoco_retry新目录重跑，原证据不覆盖。训练环境无onnxruntime，使用已有robot环境导出对拍未安装依赖。正式32回合/学生sim2sim/CubeAI/板测未通过，禁止交付。

## 2026-10-08 DAgger v4训练完成待闭环

学生rollout冻结教师同状态标签530944条，与首轮教师数据合并，学生结构20614参数不变，12000更新完成；最佳留出MSE=3.481732848459452e-05。必须读取h723_distill_v4训练报告再作闭环复验；promoted=false，未交付合格网络。

## 2026-10-08 学生v3闭环27/31拒绝，v4回到全域DAgger

v3 exit0/130.84秒，27/31(87.1%)，低于v2的28/31；+1.4恢复，+1.2 yaw改善但存活.75，新增1/-.3、1.5/+.1跟踪退化，静止存活.5。未晋升，不做正式sim2sim验收。v4准备当前v3学生全61点4352步530944条DAgger，保留之前全部数据自动续训，固定20614参数/原动力，不继续只针对旧失败点采样。小MLP尚未可交付；后续sim2sim只验收部署小MLP并对照教师基线。

## 2026-10-08 H723 v3闭环快筛结果

保留27/31，存活退化2项，报告h723_distill_v3/acceptance_report.json。仅4回合，不是32回合正式验收；小MLP的MuJoCo对照与CubeAI板测仍未完成，promoted=false。

## 2026-10-08 部署迭代sim2sim只验收小MLP；v3快筛启动

用户明确后续MuJoCo常规验收对象为部署小MLP，需与同契约教师表现比较；教师历史基线可复用，缺少同契约/相同命令或物理契约变化才补测。不得以教师未倒替代学生通过，须同时报告绝对速度/yaw/存活指标和相对教师差异。v3采数252.5秒/续训8.28秒均exit0，合计2339840条，最佳MSE4.9493e-5；学生SHAa799f14742d807a48669bd87a15a611a83668acabb0c39931fd09f116f53bf2a，61点4回合复验已启动。最终32回合/学生MuJoCo/CubeAI板测仍待办，未交付。

## 2026-10-08 DAgger v3训练完成待闭环

学生rollout冻结教师同状态标签278528条，与首轮教师数据合并，学生结构20614参数不变，12000更新完成；最佳留出MSE=4.949294131176091e-05。必须读取h723_distill_v3训练报告再作闭环复验；promoted=false，未交付合格网络。

## 2026-10-08 DAgger v2闭环提升至28/31，仍未交付

v2完整61点4回合exit0，保留28/31(90.32%)，从2/31显著改善；1m/s yaw+1/+1.2/+1.4存活1但跟踪未过，另2.4/.3存活.75较教师退化。acceptance_report.json明确passed=false，不能仅按比例交付。已建立静态Gemm/Relu导出闸门及严格闭环验收脚本；fold100真实输入误差1.31e-6通过，正式ONNX未导出。v3准备4缺口×16副本4352步278528条targeted DAgger，保留首轮和v2全部数据，固定网络与物理，采数审计后自动12000步续训，最终仍须全域/32回合/MuJoCo/CubeAI。

## 2026-10-08 DAgger v2训练完成待闭环

学生rollout冻结教师同状态标签1030656条，与首轮教师数据合并，学生结构20614参数不变，12000更新完成；最佳留出MSE=8.0350047136014e-05。必须读取h723_distill_v2训练报告再作闭环复验；promoted=false，未交付合格网络。

## 2026-10-08 H723失败排查与DAgger v2启动

四个分片抽样教师标签重算最大差9.54e-7；DR/弹簧/观测契约/sensor/peak配置一致。归一化训练/推理同公式，尚无证据是接口错位；主要待验证假设为教师状态分布覆盖不足。evaluator新增学生rollout+冻结教师同观测标签（绝不教师接管），v2采103万条，原物理契约不变。continue_pipeline.py等待成功采数及SHA/finite/样本数审计后自动接12000步学生续训，合并旧数据并保留旧归一化；防止前轮采完未接训练。静态holdout仍可能同episode相关，不能称独立泛化；闭环验收未完成，禁止交付，标准20614参数冻结。

## 2026-10-08 H723首轮学生闭环失败，禁止部署

student_evaluation exit0/134.8秒，61点每点>=4回合，教师31项仅保留2项(6.45%)，丢失29项，32条件存活率低于教师，未过90%闸门。student SHA752c92663b1668dfbb4abac3b5f7247c88125c890b6e5ea92ac18fe6f66a90be，报告h723_distill_v1/student_acceptance_report.json，promoted=false。静态MSE7.18e-6不能证明闭环，根因未定；需核对标签/归一化/consumer并考虑学生轨迹教师纠偏，禁止宣称小MLP验收完成或提供作部署合格版。

## 2026-10-08 H723学生训练完成，闭环快筛启动

12000更新exit0，训练wrapper8.71秒；留出MSE最佳7.18366477e-6。标准20614参数checkpoint约84KiB，SHA752c92663b1668dfbb4abac3b5f7247c88125c890b6e5ea92ac18fe6f66a90be。新增rmuc_student_runtime.py确定性适配，evaluator显式--distilled-student并校验teacher SHA，教师仅用于物理/观测契约，不参与学生动作。100输入形状/finite smoke通过。h723_distill_v1/evaluate_student.py已启动61点4回合/3200步/900秒，固定spring2.424375/DR.1；闭环90%尚无结论，不能交付为已验收网络。静态ONNX/100组跨runtime对拍/MuJoCo/CubeAI尚未完成。

## 2026-10-08 H723采数完成，学生训练接续

collection exit0/500.85秒，33分片1030656条，SHA/25D/6D/finite审计通过。此前学生尚未启动，本轮启动标准20614参数学生12000步/Adam1e-3/batch4096/GPU0，20分钟超时，完整命令h723_distill_v1/training_command.json，状态training_status.json。闭环90%与安全门槛、静态ONNX/100输入对拍、MuJoCo/CubeAI均未验收，无部署证书。

## 2026-10-08 H723蒸馏管线开始，尚无验收学生

用户冻结标准25→128→128→6，精确20614参数/82456字节FP32。未发现声称15:45已启动的蒸馏进程或产物，现新建artifacts/h723_distill_v1。正式教师v588 SHA66fe36e3ac4c6811e9eb6d4ec83f3841960a3472e02801e8291d69543a0c3679，当前spring2.424375/DR.1正式31项通过，v597仅33项快筛。采数1030656条、61静态点/current reset/current DR，1800秒上限；不宣称动态指令/最高DR覆盖。scripts/evaluate_rmuc_grid.py新增可选CPU流式分片，scripts/rsl_rl/rmuc_student_distill.py标准学生MSE/轻噪声Adam1e-3/batch4096，按整分片留出验证。ONNX/闭环/FSM/CubeAI未完成，当前无可交付验收小MLP；学生90%门槛不能替代必需技能生存/硬件契约验收。Normalizer fold不免除25D观测构造。

## 2026-10-08 用户确认部署MCU为STM32H723

完整后缀及板载外部Flash/RAM仍待确认。ST官方H723/733系列页：最高550MHz，512KB–1MB片上Flash，564KB SRAM；不能承诺当前完整组合策略直接片上部署。基础MLP13862参数与完整策略493492参数必须区分；当前v588 ONNX约2.5MB。需CubeAI实际Analyze/Validate、权重及activation内存报告、125Hz策略与1kHz轮控并发板测。外存或压缩路径须按板卡配置决定，训练及行为认证与MCU推理认证独立。来源https://www.st.com/en/microcontrollers-microprocessors/stm32h723-733.html。

## 2026-10-08 v598无偏置快筛最终结果

33/61通过；旧31丢失0项，存活退化0项，runtime=True。spring2.424375/DR.1，报告artifacts/continuation_v598/final_report.json；仅4回合快筛，promoted=false，未作32回合晋级或最终交付认证。冻结审计保留训练实际变化；下一步按真实失败项决定正式复验或继续定点适应。

## 2026-10-08 v596新增两项但crawl退化，v597隔离恢复拟合通过

v596 exit0/runtime通过32/61，新增1/-.45及1.5/-.2通过，无存活退化，但旧.25/+.1 yaw.07302/stable0丢失，拒绝晋升。v597仅outer positive crawl network600CPU步恢复v588同输入动作，最大MAE.000494<.001，域外精确保护/冻结通过；候选SHA925c3ec2e1f673cd8c3b60433166499da632adc7bf9beebcfbd6549cce9442f7。v598准备同spring2.424375/DR.1全61无偏置4回合，不把拟合当行为通过，正式基线仍v588。

## 2026-10-08 v596无偏置快筛最终结果

32/61通过；旧31丢失1项，存活退化0项，runtime=True。spring2.424375/DR.1，报告artifacts/continuation_v596/final_report.json；仅4回合快筛，promoted=false，未作32回合晋级或最终交付认证。冻结审计保留训练实际变化；下一步按真实失败项决定正式复验或继续定点适应。

## 2026-10-08 v595汇总拟合过门槛，v596行为快筛启动

v595仅fit损失权重1→2，600步/seed594保持，汇总fit与rehearsal/保护通过，候选SHAc52614203327964d3d7d314b39e1a9962a67145f193e82148b8c432af311c011。分命令1/-.45最大MAE.00201，1.5/-.2仍.00820，不能把汇总通过称两个技能通过。v596同spring2.424375/DR.1，无偏置61点4回合/3200步/600秒，对照v588原31和存活。准备时grid路径错误已修正，首次未启动GPU；目前查evaluation_status及PID接续。promoted=false，最终目标active。

## 2026-10-08 v594分层可实现性已核对，辅助拟合未过

对v592两个教师的当前嵌套门控/限幅实际前向检查，6000动作分量超预算0，浮点最大越界1.79e-7。v594保持600CPU步加入.001 pre-tanh辅助目标仍MAE.0053998>.005，rehearsal约.000352/保护约.000687通过；无候选、不GPU评估。当前证据支持总拟合/保护目标折中未解决，不支持力矩预算不足；不自动扩大预算或放宽验收。v588正式31项基线不变，最终目标active。

## 2026-10-08 v592三命令有教师，v593短训拒绝

v592 exit0，.25/-.3、1/-.45、1.5/-.2各有存活1/stable1的偏移通过轨迹；.5/-.3无满足条件的教师。v593现有共享reference600步CPU拟合后未通过fit/rehearsal门槛，无候选生成、不做GPU无效评估。报告v593/final_report.json，源v588保持；下一步必须检查分层authority/gate与目标可实现性，不盲加更新数。目标active。

## 2026-10-08 v592负向命令扩展诊断结束

诊断完成，满足通过/存活1/stable>=.9的偏移条件共4条（非去重命令数），详见v592/feasible_teacher_summary.json及final_report.json。源v588，spring2.424375/DR.1；偏移非部署策略，必须检查各残差层可实现预算后再拟合，无偏置retention仍必需，promoted=false。

## 2026-10-08 v592启动：正式弹簧迁移通过后扩展命令域

v589/v590已exit0，联合31/31每点>=32回合、runtime与质量惯量motor审计通过，无失败/存活退化；接受v588在spring2.424375/DR.1的阶段31项结果，不是全域认证。v592原源SHA66fe36e3ac4c6811e9eb6d4ec83f3841960a3472e02801e8291d69543a0c3679，固定动力/DR测试4个失败负向慢转命令×5轮差分偏移，1600步/300秒/至少2回合，零训练。实际status接续，完成自动report；不重复启动，不把偏移加入部署。MuJoCo仍未通过，最终目标active。

## 2026-10-08 v589/v590正式复验最终结果

v588同源spring2.424375/DR.1正式31/31，各至少32回合；失败0项、存活退化0项。报告artifacts/continuation_v589/combined_report.json，单seed静态子集，不等于全命令域/最高DR/正式sim2sim。下一步需读取真实失败条件再决策，目标active。

## 2026-10-08 v591 MuJoCo结果与下一命令域准备

v588同源/spring2.424375名义MuJoCo六探针：1m/s ±.8/±1.4与.25/+.1均12s未倒，但速度/yaw跟踪未过；.5/+.3约5.52s倾倒。报告v591/final_report.json，非sim2sim成功，不替换桌面。v589/v590原PID162350/162351仍运行。v592仅准备4个失败负向慢转条件的轮差分诊断（.25/-.3、.5/-.3、1/-.45、1.5/-.2），必须正式31项保留后才能启动；固定新spring/DR不升档，偏置不部署，目标active。

## 2026-10-08 v588保留31项，v589/v590正式复验运行

v588 exit0、runtime通过，31/61快筛通过、原31无丢失/存活退化；1/+.8 yaw.02803/stable.75/存活1。v589/v590同v588 SHA66fe36e3ac4c6811e9eb6d4ec83f3841960a3472e02801e8291d69543a0c3679、spring2.424375/DR.1，各32回合正式31项复验已启动，PID162350/162351，未结果不晋级。v588导出CPU完成；v591准备同契约六项MuJoCo探针，不以运行完替代行为验收。目标active，桌面包不替换。

## 2026-10-08 v588无偏置快筛最终结果

31/61通过；旧31丢失0项，存活退化0项，runtime=True。spring2.424375/DR.1，报告artifacts/continuation_v588/final_report.json；仅4回合快筛，promoted=false，未作32回合晋级或最终交付认证。冻结审计保留训练实际变化；下一步按真实失败项决定正式复验或继续定点适应。

## 2026-10-08 v588验收运行，v589/v590仅准备

v588实际PID153326存活、1000步；日志2m/s yaw.3有倾倒，须完整61点对照基线才能判定退化，未给成功结论。v589/v590正式31点每点32回合脚本已准备未启动，evaluate.py要求v588 runtime通过且无旧技能丢失/存活退化。v588/finish.py等待原作业结束自动报告并记入两文档；禁止重启重复作业。部署参考D_comp=.052已核对，最终目标active。

## 2026-10-08 v587有效轨迹，v588已有专家适应完成待retention

v587四偏移均存活，-.02右active动作下1/+.8 yaw误差.02164、stable1，两回合；仅诊断不晋级。v588 CPU400步仅拟合已有+.8专家，冻结及域外输出精确保护通过，源v579 SHA2a9ac3c7bcca93ad55b6c78187e0eb3d4d68cdab02c923b3c89fbac75aa75130，候选SHA66fe36e3ac4c6811e9eb6d4ec83f3841960a3472e02801e8291d69543a0c3679。拟合通过不等于行为通过；准备无偏置全61点>=4回合与旧31 retention，spring2.424375/DR.1不变。正式接受基线仍v579/spring2.155/DR.1，最终目标未完成。

## 2026-10-08 v586新弹簧仅+.8退化，v587定点适应数据采集中

v586 spring2.424375/DR.1全61点30通过，原31仅1/+.8丢失：speed error.05960、yaw.06550、stable.25167；无生存退化，runtime通过。新弹簧不接受，不升DR。旧正式基线仍v579/spring2.155/DR.1/31点各32回合。

v587固定新弹簧/DR，仅1/+.8测试右active动作0/-.005/-.01/-.02，原consumer限幅，1600步/300秒、采500帧、leg audit，零训练。source SHA2a9ac3c7bcca93ad55b6c78187e0eb3d4d68cdab02c923b3c89fbac75aa75130。此前负向有效不代表本档必然有效，先看真实轨迹后仅拟合已有+.8专家；不新增网络/预算，不将偏移加入部署，随后全61点retention。状态/命令在v587，promoted=false，最终目标未完成。

## 2026-10-08 DR.1正式31/31通过，v586下一弹簧档运行

v584/v585均exit0，同v579 SHA2a9ac3c7bcca93ad55b6c78187e0eb3d4d68cdab02c923b3c89fbac75aa75130，spring2.155/DR.1下31/31各32回合、无失败/生存退化、runtime及质量惯量motor读回通过，报告v584/combined_report.json。仍单seed静态/命令子集，延迟两通道并未各条件覆盖，非最高DR或全域认证。

按用户循环启动v586，唯一变化spring2.155→2.424375Nm/腿；DR.1/原执行器/延迟概率5%/2%固定1周期保持。61点>=4回合/3200步/600秒、零训练，与同DR的v582比较，旧31丢失或存活退化拒绝新档。命令/status在v586，未结果不晋级；最终spring10.775/全域/最高DR/正式sim2sim仍未完成，桌面网络不替换。

## 2026-10-08 v584/v585等待，v586下一弹簧档仅准备

真实评估PID121868/121869（session64213/2554）约2分钟/1000步，正常运行，未完成不晋级。按用户递归顺序准备v586：同v579、DR.1不变，仅spring2.155→2.424375Nm/腿；径向0、执行器/延迟概率5%/2%固定1周期保持。完整command/contract/report已生成，evaluate.py要求v584/combined_report.json同SHA31/31、>=32回合、无失败/存活退化才允许启动。当前prepared_not_launched，不把准备当适应成果。

当前31点是命令子集，质量/摩擦±2%/总惯量约±4%仍不是最终±20%全联合DR；真实弹簧与全域/最高DR/部署均未认证，桌面包不覆盖，目标active。

## 2026-10-08 v582 DR.1快筛保留31项，v584/v585正式回合启动

v582原PID已结束、wrapper明确exit0，旧session失效不是失败；report与joint readback audit核对通过。全61点31通过，原31无损失/存活退化，观测/通信暴露17996/10497步，仍非固定延迟全域认证。实际质量scale[.980176,1.018414]、总惯量[.964481,1.036065]、motor约.203–2.996ms，范围符合DR.1阶段契约。

同v579 SHA2a9ac3c7bcca93ad55b6c78187e0eb3d4d68cdab02c923b3c89fbac75aa75130，v584/v585正式31点分16/15×8副本GPU0并行，每点>=32回合、各6500步/900秒，零训练，仅增加回合。spring2.155、DR.1及延迟5%/2%固定1周期保持。结束v584/combine.py联合检查，未结果不升档；最终弹簧10.775/全域/最高DR/sim2sim仍未完成，桌面包不替换。

## 2026-10-08 v580/v581正式31/31通过，v582 DR.1筛查启动

v579在spring2.155/DR.05正式31条件各32回合全部通过，无失败/生存退化；两批exit0、同hash/runtime及质量惯量motor读回审计通过，报告v580/combined_report.json。仍单seed静态与稀疏1周期延迟，不是最高DR/全域/正式sim2sim。

v582已满足启动门槛并启动，source v579 SHA2a9ac3c7bcca93ad55b6c78187e0eb3d4d68cdab02c923b3c89fbac75aa75130，唯一区别物理/传感器幅度DR.05→.1，spring2.155与延迟概率.05/.02/1周期不变。实际质量/摩擦±2%、总惯量倍率[.9604,1.0404]、Dfb[.049,.051]/b[.0019,.0023]/C[.00095,.0014]、motor.2–3ms；peak5.5/continuous3/scale5.5/2ms保持。全61点4回合/3200步/600秒，无训练；旧31损失或生存退化不晋级，未报告promoted=false。结束report.py与joint readback audit后决定适应或正式复验，最终目标未完成。

## 2026-10-08 v583同契约MuJoCo回归：新增两慢转存活，半速转弯仍倾倒

v579同源SHA2a9ac3c7bcca93ad55b6c78187e0eb3d4d68cdab02c923b3c89fbac75aa75130、spring2.155/名义Dfb.05/b.002/C.001、canonical rmuc_peak55.xml的6项CPU12秒探针结束。1/+.2与1/+.3未倒，实测vx.77860/.76497、yaw.16655/.28154，yaw误差.03345/.01846，但速度误差.2214/.2350；.5/+.3约5.032s触发tilt45deg阈值。三个保护条件(.25,+.1),(1,-.2),(1.5,-.1)未倒，不能将5个duration_reached称跟踪/DR验收通过。报告v583/final_report.json，失败证据完整保留，未替换桌面包。

GPU v580/v581仍在正式31点复验，原进程活着，继续等待；v582 DR.1仅准备，正式门槛未过不启动。0.5m/s附近跨引擎闭环缺口已覆盖直行及正转弯，需在后续sim2sim定位，不能用Isaac通过掩盖；目标active。

## 2026-10-08 v579导出通过，v580/v581等待，v582仅准备

v579导出6254输入JIT差0/ONNX最大3.576e-7通过，export_hashes.json已保存，不代表MuJoCo或部署验收。v580/v581原PID102452/102451、session94482/48975仍活，最新日志1000步，继续原作业不重启。

v582准备下一物理/传感器幅度DR.05→.1命令，尚未启动。evaluate.py强制检查v580/combined_report.json同源31/31、>=32回合、无失败/生存退化才允许运行。spring2.155不变，延迟固定1周期概率.05/.02不变；实际质量/摩擦±2%、总惯量倍率[.9604,1.0404]、Dfb[.049,.051]/b[.0019,.0023]/C[.00095,.0014]、motor.2–3ms，非最终±20%/最高DR。正式门槛未过前不能启动或自动跳级，其他30条件/最终弹簧/站立/动态命令/sim2sim仍未完成。

## 2026-10-08 v579扩展成功31/61、旧28全保留；v580/v581正式复验运行

v579无偏置61点各4回合31通过，原28全部保留、无生存退化、runtime通过。新增.5/+.3 yaw.010509、1/+.2 yaw.015220、1/+.3 yaw.008958，stable均1，网络容量未增。1.5/+.2/+.3仍失败（yaw约.095/.096），静止及剩余30点未完成，不能称命令全域通过。

v580/v581同candidate SHA2a9ac3c7bcca93ad55b6c78187e0eb3d4d68cdab02c923b3c89fbac75aa75130，固定spring2.155/DR.05，将31项拆16/15×8副本（128/120环境），GPU0并行，各>=32回合/6500步/900秒，无训练。结束v580/combine.py联合核对runtime/hash/全部31点/实际参数；正式通过前不扩大DR。桌面v565不替换，最终spring10.775/最高DR/全域/sim2sim未完成。

## 2026-10-08 v577采样/v578拒绝/v579固定容量训练完成，闭环评估中

v577五正慢转缺口中4项有稳定通过偏置示范：.5/+.3、1/+.2、1/+.3、1.5/+.2；1.5/+.3仍无通过示范。1.5/+.2示范进一步因共享残差预算不可达被筛除，仅前三项用于训练，不改预算。v578现有positive reference网络600步+真实旧观测rehearsal，目标MAE.006536超过.005门槛，拒绝无候选。v579相同容量/600步，加反算tanh前目标辅助loss，逐层可达检查通过，目标最大MAE.004945/回放.001085，原门槛通过，保护域同输入输出保持。

候选SHA2a9ac3c7bcca93ad55b6c78187e0eb3d4d68cdab02c923b3c89fbac75aa75130，源v573。固定spring2.155/DR.05，原执行器与延迟，当前61点无偏置3200步/600秒评估，必须保留旧28点/无存活退化；未结果promoted=false。1.5正慢转仍缺口，不把未选训练条件声称修复，最终目标未完成。桌面候选不变。

## 2026-10-08 v574/v575同v573正式28/28各36回合，进入命令域扩展

两批exit0，合并28/28、每点36回合，无失败/生存退化，同hash与runtime/质量惯量motor读回通过。spring2.155/DR.05当前命令子集阶段可用于下一步适应，promoted=false仍表示非最终发布；单seed/静态/稀疏1周期延迟不等于最高DR或全域。报告v574/combined_report.json。MuJoCo0.5直行失败及其他跟踪差仍未解决，不继承Isaac结果为sim2sim证书。

按用户顺序固定当前弹簧/DR，v577采集.5/+.3、1/+.2/+.3、1.5/+.2/+.3五个缺口，差分轮偏移0/-.005/-.015/-.04/-.08，原consumer限幅，1600步/300秒、保存500帧，零训练。source v573 SHA39b4b8095a504085502dc35606ba69df4f2adac053ed9257349cb904baecc294。偏置只作可行数据，不部署；后续训练保护28项、全61点retention。命令/status在v577，最终spring10.775/全速度/最高DR/实机未完成。

## 2026-10-08 v573导出及MuJoCo回归完成，正式Isaac复验仍运行

v573导出6254输入JIT差0/ONNX3.576e-7通过，hash已存export/export_hashes.json。v576使用相同source、canonical rmuc_peak55.xml和spring2.155/Dfb.05/b.002/C.001/5.5峰值/PD60/2/2ms/8ms做9项12秒CPU MuJoCo诊断。8项duration_reached、.5直行于3.616s倾倒；不是行为8/9通过。新增1/-.2实测vx.95548/yaw-.29648、1/-.3实测vx.95129/yaw-.36659，均存活但过转；其他正大转弯速度/yaw仍有明显误差。完整命令/contract/hash/report在v576，不覆盖桌面v565包，不称sim2real高成功已证明。

v574/v575原GPU正式评估进程仍活（PID86636/86637，session20016/5731），尚无最终28点结论。继续原作业不重启，结束后v574/combine.py。模型容量保持493492，不增加专家；最终弹簧/全域/最高DR/站立/动态命令/正式MuJoCo仍未完成。

## 2026-10-08 v573无偏置28/61、旧25全保留；v574/v575正式复验启动

v573全61点各4回合28通过，无旧25丢失/生存退化，runtime通过。正爬行.25/+.1恢复yaw.03637/stable1；.25/+.3 yaw.01205/stable1；1/-.2 yaw.00795、1/-.3 yaw.04403和1.5/-.1 yaw.02871均stable1。固定容量局部恢复有效，但仍单seed快筛，promoted=false，静止未掌握、其余33条件不通过。

v574/v575同candidate SHA39b4b8095a504085502dc35606ba69df4f2adac053ed9257349cb904baecc294，将28通过点分两批14×9环境，GPU0并行，每点>=32回合，各6000步/900秒。只增加回合预算，无训练；spring2.155/DR.05/原执行器与延迟保持。结束运行v574/combine.py核对完整28覆盖/同hash/runtime/质量惯量motor读回与retention。无结果不升DR/弹簧，桌面候选不替换；正式通过后按用户顺序命令域再扩展/DR。

## 2026-10-08 v571保住新增保护域但爬行退化，v573局部恢复候选评估中

v571完整快筛27/61，1/-.2 yaw.00795、1/-.3 yaw.04403通过，1.5/-.1恢复yaw.02871/stable1，静止无生存退化但仍未掌握。旧.25/+.1退化yaw.06745/stable0，故retention失败/promoted=false。不能以总通过数宣称完成。

改用已有外层physics.network仅恢复正爬行输出，以v565动作重标v532实际爬行观测，不再改共享参考网络、不增加参数。v572600步后MAE.001504未达.001门槛，拒绝无候选；逐层可达性审计无越界。v573在同600步预算改用反算tanh前目标辅助loss，MAE最大.000862达到原门槛，保护域同输入逐值一致，candidate SHA39b4b8095a504085502dc35606ba69df4f2adac053ed9257349cb904baecc294。

v573固定spring2.155/DR.05，全61点无偏置3200步/600秒retention运行，原25点丢失或存活退化拒绝，桌面包不替换。参数规模不变、最终目标未完成。新训练不能因拟合通过当闭环认证；结果需report.py后接续。

## 2026-10-08 v570补实际保护轨迹，v571完成600步回放适应并复验中

v570源v565采1.5/-.1与静止，新增可选capture-observation-stride=3，以500帧覆盖至episode age1497；默认stride1保持旧行为，数据shape500×8×25。1.5/-.1通过（yaw.03375/stable.9992），静止仍0存活，后者只是防退化回放，不能当成功示范。报告v570/final_report.json。

v571从v565重做600CPU步、26.85秒，只更新已有inner frozen_reference网络，保留原真实观测rehearsal并加两保护域10倍loss约束，模型容量不增。目标MAE最大.002764、旧回放平均最大.000708、新增保护平均最大.000796；冻结审计通过。候选SHA03d213a632fe344ffecc139aaeb939160dfe7f0513bf8b56cdd0fc4503223fb3。固定spring2.155/DR.05，61点无偏置3200步/600秒评估已启动，旧25保留/无存活退化为门槛，未报告不晋级。桌面v565不替换，最终目标未完成。

## 2026-10-08 v569实际训练结果28/61，但retention失败不晋升

600CPU步固定容量训练后无偏置61点各>=4回合25→28。1/-.2 yaw.004404/stable1修复，1/-.3 yaw.044542/stable1新增；1.5/-.2和-.3分别yaw.02217/.02084、stable1新增；.5/-.1也通过但裕量较小。旧1.5/-.1退化（yaw.05243/stable.0953），静止存活相对v565下降。retention=false，promoted=false，不以通过数提升掩盖损失，不替换桌面v565包。

模型仍493492参数，训练只改已有inner frozen_reference网络。真实观测rehearsal MAE很小仍未覆盖1.5负小yaw和静止的关键闭环状态，说明仅保存样本拟合误差不能替代行为retention。下一步需补这两个保护域的当前源实际轨迹后同容量回放适应，不继续升DR/弹簧，不盲加训练轮数。完整v569/final_report.json、runtime_verified.json、retention_audit.json、training_report.json已落盘，GPU评估exit0。最终目标未完成。

## 2026-10-08 v566/v567正式24/25，不晋级；v569固定容量适应已训练

合并正式结果为24/25、各36回合、无生存退化，但1m/s/-.2rad/s yaw误差.052697/stable.206615失败。不能把此前25点快筛写成新弹簧正式完成。保持spring2.155/DR.05修该条件，桌面v565候选不覆盖，也不称最终认证。

v568小差分轨迹在1/-.2与1/-.3上可行，+.02/-.02时yaw.00660/.04737、stable1、全存活；单seed2回合偏置不认证。v569基于v565仅更新已有physics.frozen_source.frozen_reference_network，600CPU步约17.57秒，使用v542/v561/v564实际旧观测重标当前源动作rehearsal，无新增专家/参数；actor仍493492。目标MAE最大.000234，rehearsal平均最大.000059、单元素最大.001102，冻结审计通过。共享分支不保证域外逐值一致，必须看全61点retention。

新候选SHA0027c60104dddfe7f6a9a931bc040d86434c204c99723008d5a21758e25ffe2f，v569无偏置61点评估已启动（3200步/600秒）。固定spring2.155/DR.05/原执行器与延迟，检查旧25保留和相邻-.3扩展，未结果promoted=false。完整command/contract/training.log/hash/freeze/evaluation command齐全。目标仍active；CubeAI与MCU实时性未认证，网络扩张此次避免。

## 2026-10-08 用户要求交付候选ONNX及不倾倒MuJoCo桌面demo，已完成

对v537/v545/v565各自匹配弹簧阶段（1.61625/1.885625/2.155）、同canonical rmuc_peak55.xml、active_motor_v1 Dfb.05/b.002/C.001、5.5峰值/3持续、PD60/2、2ms/8ms，逐项10命令×12秒CPU MuJoCo。三候选均9/10完成且未触发tilt45deg或height.08m阈值，.5m/s直行均失败。无证据统计实机成功率；选择v565作为当前交付候选，因为新弹簧25点Isaac快筛且旧24保留，MuJoCo存活不劣，不能称最终认证。

桌面/home/kellen/桌面/RMUC_v565_20261008已生成：export/policy.onnx（SHA3deb2604b3f8d4c5d5efc7c00fbf1f08302a07f067250c275131c55253c0aff8）、JIT/checkpoint/完整controller契约、canonical MJCF及meshes、CPU回放源码、controller参考源码、README、metrics.csv和SHA256.json。9段未倒录制为.25直行及±.1、1直行及±.8/±1.4、1.5直行，每项完整12仿真秒；合辑RMUC_v565_demo.mp4约105.9播放秒（9段独立重置，非动态连续切换）。视频标明目标/实测后6秒速度yaw和名义诊断非硬件认证。失败.5直行JSON保留evidence，不进demo；9张中帧视觉检查通过，哈希/controller校验通过，拷贝包1秒真实回放成功。

首次录制因默认离屏framebuffer640小于960失败，保留日志；仅修改run_mujoco_probe.py的显示缓冲尺寸和可选跟随相机，未改动力，重录9项均成功。所有30探针与9录制命令/JSON/日志在artifacts/delivery_20261008。当前v566/v567正式回合仍运行；桌面README明确新候选只快筛、未继承旧证书，弹簧非最终10.775、硬件符号/1ms/最高DR/热/动态命令未认证。用户允许跟踪差但不接受倾倒视频，已遵守；不将视频选择当全域成功率。

## 2026-10-08 接续v565结果恢复：25/61、旧24全保留，v566/v567正式复验启动

中断后核查v565原PID不存在，status仍running但完整evaluation.json已落盘；运行report.py验证61点各>=4回合、源hash/runtime和retention。25/61，旧24全部恢复，无生存退化；.25/-.1 yaw.003663、speed error.05923、stable1/survival1。退出码未知，不虚构exit0；另存status_reconciliation.json说明，仅原wrapper状态未收尾，未重复评估。6254输入导出JIT0/ONNX3.576e-7通过，hash已存export/export_hashes.json。

固定v565 SHA8a46d0aec4b96fc31088d55724161ba96918a384167fe1e5a1f9e7c160bdd825/spring2.155/DR.05，v566/v567将25通过点拆12/13×9副本GPU0并行，各>=32回合/6000步/900秒，零训练。只改回合预算，不继承旧候选证书。结束后v566/combine.py联合核对hash/runtime/25条件/回合数/质量惯量motor读回，失败不接受新档。当前promoted=false，最终弹簧10.775/全域/最高DR/正式sim2sim未完成；通过后按用户顺序扩大命令域。

## 2026-10-07 v564找到爬行裕量，v565局部拟合后全域复验中

v564 .25/-.1小差分探针全部存活，+.02/-.02轮动作时yaw误差.003976/stable1，速度误差.06379；零偏移yaw.05322失败。2回合偏置诊断，不作为认证。v565复用reference隔离，仅放宽配置校验允许负yaw支持域[-.2,0]，无增加冻结层数；旧正yaw配置行为保持，12测试通过。

v565仅外层reference_network做400CPU步约5.27秒，示范原残差budget可行、最大MAE.000133、其他参数冻结/保护域同输入逐值保持，候选SHA8a46d0aec4b96fc31088d55724161ba96918a384167fe1e5a1f9e7c160bdd825恢复成功。固定spring2.155/DR.05，无偏置全61点3200步/600秒评估已启动，必须旧24全部恢复且无生存退化才接受候选；后续正式32回合仍必须。此时promoted=false，未完成最终弹簧/全域/最高DR/sim2sim。

## 2026-10-07 v563恢复直行，仍缺爬行；v564小轮差分诊断启动

v563无偏置61点每点4回合24通过，.5直行恢复：速度误差.14960、yaw.00361、stable1、存活1。v562恢复的两项转弯保留，无生存退化。原24点仍丢失.25/-.1，因另有新通过点总数回到24，不能将24/61当全部retention成功；spring2.155仍未接受。source candidate SHA1a1509cfeebee84df5dfca17b04bd16c588efd3cd2277c9072497d6c2e3f75a8。

v564固定该候选/新spring/DR.05，仅.25/-.1测试轮差分0/.0025/.005/.01/.02，腿动作不变，1600步/300秒、零训练，保存500步数据；偏置不进部署。查v564/status与真实PID接续，结束后report.py，若有可行示范必须检查原残差budget并无偏置全61点retention。最终全域、spring10.775、最高DR、MuJoCo仍未完成。

## 2026-10-07 v562真实闭环恢复两转弯，v563直行拟合后评估启动

spring2.155/DR.05下v562全61点由v558的21→23，+0.8恢复（yaw.03657/stable.75）、+1.4恢复（yaw.01248/stable1），无生存退化；旧24仍丢失.5直行和.25/-.1，不接受新弹簧档。全部为4回合快筛，runtime通过，报告v562/final_report.json。

GPU等待期间CPU拟合v563现有physics.frozen_source.scoped_network，仅窄直行域，400步约5.28秒；用v561的.5直行右active-.05可行轨迹，原动作/残差预算不变，其他参数冻结，保护域同输入逐值保持。最大MAE.000182，候选SHA1a1509cfeebee84df5dfca17b04bd16c588efd3cd2277c9072497d6c2e3f75a8。v563当前无偏置全61点3200步/600秒复验中；爬行缺口未处理，不因拟合小误差晋升。命令/contract/training.log/freeze审计已保存，promoted=false。

## 2026-10-07 v561负方向找到3项可行轨迹，v562两专家拟合完成/闭环评估中

v561全部桶存活；右active-.01/-.025/-.05在.5直行均通过，-.05速度误差.14773；1/+.8选择-.01，yaw.01203/stable1；1/+1.4选择-.025，yaw.01177/stable1。爬行.25/-.1仍yaw约.053不通过。各2回合/非配对DR，只是可行轨迹，非策略验收。

v562用v561原始观测/action重建误差3.725e-7，仅拟合原local专家5(+.8)、3(+1.4)，每项400CPU步共800、约10.68秒；第二副本验证动作MAE最大5.58e-5，其他参数冻结，保护命令同输入逐值一致，原预算保持。候选SHA2cc32384beb0867149e1dfcd6aa9ac653ade6fbf1721df276203cae8e7874e04。固定spring2.155/DR.05无偏置全61点3200步/600秒评估中。直行/爬行缺口没有被这两专家修复，必须全24恢复才接受新弹簧档，promoted=false。

## 2026-10-07 v561右active负向小步探针启动

v560正向无效后，固定v545/spring2.155/DR.05，四退化条件只测试右active动作0/-.01/-.025/-.05，经原consumer限幅，预算/模型不变。1600步/300秒、各桶>=2回合，保存500步轨迹和leg audit，无训练。正负方向不能由“助伸”静力直觉推导闭环效果，以真实可用轨迹为准；DR桶非配对，不夸大因果，偏置不用于部署。源SHA34e9afd05ea3c3b3eddc417d2e6b6126d8c0c53f36f7446e92d32c88221be2fe，命令/status在v561。未完成前无结论，promoted=false。

## 2026-10-07 v560结束：右active正偏移方向无效，拒绝用于拟合

固定spring2.155/DR.05，4命令×右active动作0/.025/.05/.1，各>=2回合；所有非零偏移均未通过。0.5直行速度误差由.20658随增量恶化至.23376/.25921/.30644；1/+.8 yaw由.04832恶化至.09945/.182/.310；1/+1.4的+.1倾倒（survival0），其余该项偏移yaw亦恶化；.25/-.1未修复。不同DR桶非配对，不能精确估计导数，但此单边正偏移无稳定可用示范，不拟合、不扩大正偏移，promoted=false。

双腿同向后置补偿改善不意味着只加右腿也改善，剩余动作裕量不代表该方向正确。下一步选择原动作范围内负向小步或腿轮联合适应，保留左腿clip约束，不能把不可达双腿示范强塞策略。报告v560/final_report.json、freeze_audit.json与500步数据，exit0无训练。新spring2.155仍未接受，旧阶段1.885625和v545不变。

## 2026-10-07 v560运行：原限幅前右active动作探针

固定v545 SHA34e9afd05ea3c3b3eddc417d2e6b6126d8c0c53f36f7446e92d32c88221be2fe/spring2.155/DR.05，四退化条件仅右active动作加0/.025/.05/.1，走原consumer限幅，不再后置腿目标偏移，不加动作预算。1600步/300秒，各桶>=2回合；保存500步观测/action和leg audit，若轨迹可用，拟合前仍须核对残差剩余预算与标签。DR桶未配对，偏置不进入部署，无训练、promoted=false。命令与status在v560；禁止把诊断通过当无偏置策略通过。其他变量和旧24点retention要求保持。

## 2026-10-07 v559结束：腿PD目标诊断改善3项，但低速左腿偏移不可直接由原动作实现

新spring2.155/DR.05下4命令×0/半/全deltaSpring/kp（全量.00448958rad）各2回合，全存活。全偏移时.5直行速度误差.19525/stable1、1/+.8 yaw.01439/stable1、.25/-.1 yaw.04896/stable.786；三项通过。1/+1.4零偏移桶通过、半/全偏移桶失败，且各桶DR抽样不配对，不作单一因果结论。v558本轮未配对延迟暴露问题仍存在。

审计joint_order为left_active,left_hip,right_active,right_hip，每环境1200稳态physics样本：所有腿电机torque_saturation_fraction=0；left_active action_saturation_fraction=1（此指标按±1统计，大转弯域有扩展上限，不能把大转弯也说成硬clip）。.5直行左目标.2044868超过原.2rad可达上限，.25/-.1左目标.1944887超过原.19rad可达上限，因此不能把诊断目标偏移直接拟合成受原clip的策略，也不能加部署目标偏移冒充学会。要用剩余可达腿/轮动作解决，预算不扩大。报告v559/final_report.json与freeze_audit.json，exit0、无训练、promoted=false。弹簧2.155仍未接受。

## 2026-10-07 v558拒绝新弹簧档，v559定位腿PD偏差

spring2.155/DR.05全61点24→21，丢失旧4点（.5/0、1/+.8、1/+1.4、.25/-.1），另新增1点；无生存退化，runtime通过。retention失败，不接受2.155，不升DR。延迟暴露17996/10497比v555增加，非完全配对扰动，不能将全部差异归为弹簧。旧已验收阶段仍spring1.885625。

未饱和腿PD的目标位移补偿为deltaSpring/kp=.269375/60=.00448958rad；受动作裁剪、电机响应等影响，不是保证可部署的等价变换。v559固定新档/同v545候选，在4退化命令比较active目标偏移0/半量/全量，1600步/300秒、零训练，附leg actuator audit；仅诊断，禁止写入部署或称技能通过。检查v559/evaluation_status.json及真实PID后接续。完整命令/contract已保存，下一步根据结果做策略内适应，仍需无偏置全61点retention。

## 2026-10-07 DR.05正式24点各40回合通过，v558弹簧迁移启动

v556/v557均exit0，联合24/24、各40回合，无失败/生存退化；新增质量/惯量张量实际读回及motor响应范围审计通过，报告v556/combined_report.json与各批joint_readback_audit.json。仍阶段spring1.885625/单seed/静态命令，其他37网格缺口不消失，延迟两通道未全域覆盖，不称最终鲁棒性完成。

按用户顺序启动v558：source v545 SHA34e9afd05ea3c3b3eddc417d2e6b6126d8c0c53f36f7446e92d32c88221be2fe，唯一变量spring1.885625→2.155Nm/腿；DR.05/2ms/peak5.5/continuous3/独立延迟概率.05/.02且1周期保持。全61点>=4回合、3200步/600秒，无训练，retention基线明确同DR的v555而非DR0；旧24丢失/存活退化拒绝新档。状态在v558/evaluation_status.json，未结果不接受新档。最终10.775/全域/最高DR/正式sim2sim未完成。

## 2026-10-07 v556/v557共享GPU等待，v558弹簧下一档已准备未启动

真实PID3341535/3341536均运行，约4分半日志1000步；GPU0还有其他工作区PID3339973，利用率99%，未干预其他作业。继续原进程与900秒预算，观察慢不算终止。v558已准备spring1.885625→2.155Nm/腿，其余同DR.05，evaluate.py显式检查v556/combined_report.json同源24/24、>=32回合、无失败/生存退化后才允许启动；当前未运行。source仍v545，禁止越过本轮验收；v558命令/contract已落盘供后续接续。

## 2026-10-07 v555 DR.05快筛24保留，v556/v557正式回合运行

v555全61点各4回合24通过，无旧点丢失/生存退化、runtime通过；观测/通信延迟实际10555/4503控制步。DR.05只完成快筛，不晋级。v556/v557同v545 SHA34e9afd05ea3c3b3eddc417d2e6b6126d8c0c53f36f7446e92d32c88221be2fe，对全部24通过点分两批各12×10环境、>=32回合，GPU0共享、各6000步/900秒。只改回合预算，spring1.885625/DR.05/固定1周期概率.05/.02/2ms/5.5峰值不变。

两批结束后运行v556/combine.py联合核对hash/runtime/24点与回合数；通过后按用户新顺序先弹簧小步迁移，再扩大命令域，再提高DR。此轮无训练，最新candidate仍v545；无报告不得宣称通过，全域/最终弹簧/最高DR/sim2sim未完成。

## 2026-10-07 用户确认递归推进顺序

当前DR.05同候选验收通过后，按“弹簧小步迁移→扩大命令域适应→提高DR→再次弹簧迁移”的顺序循环。每步固定其余变量，保留已通过全命令retention；失败固定该档针对性适应，不盲升档。延迟幅度独立从1周期逐级到2/3，不直接随机1–3。v555仍是4回合快筛，若通过先补当前同候选>=32回合再迁移弹簧；不把快筛当晋级，不用24点子集替代完整最终命令域目标。

## 2026-10-07 v555运行：物理DR.02→.05，独立延迟保持

依据v550/v551同候选24点各40回合DR.02通过及v552观测延迟5%快筛保留，v555固定v545 SHA34e9afd05ea3c3b3eddc417d2e6b6126d8c0c53f36f7446e92d32c88221be2fe/spring1.885625，仅物理及传感器幅度DR.02→.05，观测/通信概率仍.05/.02、固定1周期。质量/摩擦±1%、总惯量倍率[.9801,1.0201]，Dfb[.0495,.0505]/b[.00195,.00215]/C[.000975,.0012]，motor响应.1–1.5ms；leg/spring DR仍未加。5.5峰值/3持续/scale5.5/2ms保持，无训练。

61点各>=4回合快筛，3200步/600秒，24点丢失或生存退化不晋级；此为候选下一档诊断，不宣称DR.05已通过。完整command/contract与report.py在v555，结束后核对实际参数/exposure。真实弹簧方向/幅值映射未因照片或几何诊断认证，最终目标未完成。

## 2026-10-07 v552结束：观测延迟概率5%快筛24点保留

同v545、固定spring1.885625/jointDR.02，仅observation delay probability .02→.05，communication仍.02、命中固定1周期。全61点各>=4回合24通过，旧24无丢失/存活退化，独立概率配置及runtime核对通过；观测/通信实际暴露9004/4503控制步。无训练/无新checkpoint，promoted=false，不将稀疏命中结果当全命令固定延迟认证。报告continuation_v552/final_report.json，exit0。部署rmuc_low_level.h实查D_comp=.052不变；仅未饱和近似，非实机证书。

当前联合DR.02已有v550/v551的24点各40回合通过；延迟概率5%只有v552快筛。下一步继续小步课程，不直接随机1–3周期，不重复DR0验证。最终弹簧10.775/全域/最高DR/动态命令/正式MuJoCo等仍未完成；真实弹簧安装几何对应尚未认证，v554仅仿真虚功方向审计。

## 2026-10-07 v552启动：观测1周期延迟出现概率.02→.05

完成方向最小几何诊断后，保持spring1.885625和v545候选、jointDR.02，仅观测延迟概率.02→.05，通信仍.02；命中幅度固定1周期8ms。61点各>=4回合快筛、3200步/600秒，无训练；source SHA34e9afd05ea3c3b3eddc417d2e6b6126d8c0c53f36f7446e92d32c88221be2fe。旧24点及生存retention通过前不晋级，实际暴露另查，命中概率不是每步随机切delay。

用户照片显示偏心连接结构，但另一端归属/准确尺寸仍未确认；v554的212点只证明假设负恒力矩在MJCF中助伸，不能证明真实气弹簧等效正确，也未辨识幅值。保持模型、符号及MJCF不变，不依据照片直接代入恒力矩或径向力。v552状态/命令/contract已落盘，最终目标未完成。

## 2026-10-07 v554闭环几何方向检查：212/212负力矩助伸腿

CPU加载当前rmuc_peak55.xml（SHA49757c2fef4362af1eabee83fca96c62477e050a46f77e50237ea5118ec1c9b8），固定髋关节，求四被动关节闭环约束，在active±1e-4扰动下求dL/dq_active。两侧各45个active[-.5,.6]扫点+61命令平均关节对，共212姿态；全部闭环误差<1e-4m（实际max2.46e-12m），分支局部连续，dL/dq范围[-.14539,-.10131]m/rad，全负。当前负active力矩对该伸腿方向做正功，未发现反向助力；腿长覆盖.23118–.40152m，零位约.3427m，两侧一致。

这是同MJCF闭环几何下的虚功方向证据，比直接sin恒号更可靠，但仅固定髋的局部运动方向、平均命令姿态和几何扫描；不是全轨迹极值/碰撞接触/实机安装点或力行程验证，不证明1.885625或10.775恒值准确。无修改MJCF/弹簧符号，无训练。报告continuation_v554/kinematic_report.json与可复现脚本；可以保留当前负向等效模型继续渐进DR，物理幅值误差仍待覆盖。

## 2026-10-07 v550/v551 DR.02完成，v553气弹簧方向审计

两批联合24/24、各40回合，无失败/生存退化，runtime通过；报告v550/combined_report.json。并非各命令同时覆盖观测/通信延迟，不能称固定延迟全域通过。已将sensor profile可选observation_delay_probability/communication_delay_probability独立接入env，旧配置默认概率仍=strength、DR0仍0，18测试通过和env编译通过。v552准备仅观测延迟概率.02→.05、通信保持.02，尚未启动；因用户提出弹簧方向问题先审计，不热改动力。

v553核对当前Isaac/MuJoCo为两active_link关节负恒力矩+机身等反作用，未使用sin(front-back)。2048姿态四元数代数审计：轴投影恒-1.885625、总力矩反作用匹配。旧实机controller的sin项是电机补偿，左右原始电机符号相反，front/back带编码器方向/offset；不能等同被动弹簧力矩，更不能因sin恒号认证实机方向。

按当前helper的q*[1,1,1,-1]假设，对v542已存500步24诊断桶扫描未见超过±.01的双符号；q零位时sin=0。该角度映射尚非硬件几何零位认证，轨迹只覆盖.5/1/1.5m/s正慢转，不包含完整站蹲过程，不能称全范围恒号。证据v553/spring_direction_report.json。下一步需要从机构几何验证front/back零位并检查可达闭环伸缩路径的虚功/力矩方向；当前不反转弹簧、不新增建模或扩大DR。最终物理方向一致性仍未证明。

## 2026-10-07 v550/v551运行：联合DR.02同候选正式回合复验

v549快筛24点保留后，固定v545 SHA34e9afd05ea3c3b3eddc417d2e6b6126d8c0c53f36f7446e92d32c88221be2fe与同一joint profile，24点分两批各12点×10副本，GPU0并行，各>=32回合/6000步/900秒。唯一变化是回合预算，无训练；spring1.885625、Dfb中心.05/b中心.002、DR.02、固定1周期延迟幅度保持。不以随机低概率覆盖替代固定延迟验收，两批联合报告将逐命令记录观测/通信实际暴露。

状态在continuation_v550/v551/evaluation_status.json，结束后运行v550/combine.py核对两个exit0、源hash、runtime、回合数和全部24条件覆盖。旧点丢失或生存退化不晋级；其他37条件、最终弹簧/全DR/动态命令/MuJoCo仍未完成，promoted=false。

## 2026-10-07 v549完成：联合DR.02快筛24点保留，延迟覆盖不足

同v545候选、spring1.885625，DR.02显式profile，61点各>=4回合24通过，原24无丢失/生存退化，runtime读回通过；无训练，promoted=false，不按快筛升DR。观测/通信非零延迟累计3002/4503控制步，所有最大幅度<=1周期；仅触及4/5个命令条件，部分仅边界2步，不能称全域1周期适应。具体命令见continuation_v549/delay_coverage.json。下一步当前档正式回合与足量固定1周期暴露，再决定是否需要适应/升档；不能因未明显退化盲增训练，也不能跳过32回合门槛。

完整结果final_report.json、runtime_verified.json、retention_audit.json已生成，进程exit0。其他37条件、最终spring10.775、最高DR/多seed/连续命令/正式MuJoCo仍未完成。当前profile leg actuator和spring DR关闭，惯量为质量乘shape，均保留明确限制。

## 2026-10-07 v549运行：固定弹簧，首档联合DR.02/固定1周期延迟

source v545 SHA34e9afd05ea3c3b3eddc417d2e6b6126d8c0c53f36f7446e92d32c88221be2fe，已有24点各40回合DR0证据。只引入显式joint profile/.02强度，零训练，spring1.885625/peak5.5/continuous3/scale5.5/2ms保持。质量/摩擦±.4%，惯量shape±.4%与质量耦合，实际总倍率[.992016,1.008016]；Dfb[.0498,.0502]/b[.00198,.00206]/C[.00099,.00108]。电机时间常数40–600us；传感器noise/bias按声明最大值*.02；观测/通信延迟各2%命中，命中固定1policy周期8ms，非随机1–3。

61点各>=4回合快筛3200步/600秒，原24保留/无生存退化才进入下步，非32回合DR晋级。leg actuator DR及spring DR此档关闭，最高DR未完成；质量×shape会扩大总惯量范围，不能写成最终总惯量±20%。完整实际范围/command/contract在continuation_v549，结果必须报告实际延迟exposure，未命中条件不能声称延迟通过。状态查真实PID，无报告不晋级。

## 2026-10-07 v547/v548结束：同候选24点各40回合通过

两批exit0，联合报告continuation_v546/combined_report.json核对同SHA34e9afd05ea3c3b3eddc417d2e6b6126d8c0c53f36f7446e92d32c88221be2fe、24条件完整覆盖、各40回合及runtime，通过24/24，无失败。仍DR0、单seed、静态命令、spring1.885625；其他37个网格条件未通过，不能称全命令域或DR/sim2sim通过，promoted=false。下一步可固定当前动力进入第一小档DR及固定1周期延迟课程。

用户核对checkpoint时间：logs最新PPO为13:34:35的v533/model_7265.pt；之后主要是诊断、CPU监督拟合/隔离修正与GPU评估，不是持续PPO训练。新候选存artifacts：v53513:49、v53714:14、v53914:27、v54415:00（拒绝）、v54515:11:27（当前复验候选）。不得将后续评估耗时写成训练时长，也不能让logs旧model时间被误解为没有新候选。

## 2026-10-07 v545导出验证完成，v547/v548行为评估仍运行

verify_rmuc_export.py新增遍历各继承层reference_update_domain边界，6042输入JIT差0、ONNX最大2.980232e-7，导出SHA见v545/export/export_hashes.json。这是raw actor链路一致性，非MuJoCo/controller认证。24条件32回合两批评估仍运行；结束后执行continuation_v546/combine.py，会核对两批exit0、相同checkpoint、实际物理读回、24点完整覆盖和回合数后生成combined_report.json；单批或未结束不能合并宣称通过。

## 2026-10-07 v546预检拒绝后，v547/v548分批覆盖24点正式回合

v545同候选34e9afd05ea3c3b3eddc417d2e6b6126d8c0c53f36f7446e92d32c88221be2fe，对全部24个已通过条件每点>=32回合复验。v546原24×12=288环境触发评估器128环境上限，exit2、无仿真；保留日志，不放宽验证上限。拆为v547/v548各12条件×10副本=120环境，在用户授权GPU0共享下并行运行，各6000步/900秒。两批联合才覆盖24点，不以单批替代全域。

物理固定spring1.885625/D反馈.05/b.002/C.001/DR0/2ms/peak5.5/continuous3，无训练。完整command/contract已落盘，查两个evaluation_status.json及真实PID。无最终报告前不升DR；此次仍单seed、静态条件，最终弹簧/联合DR/动态命令/MuJoCo未完成。

## 2026-10-07 v545完成：隔离修正24/61，原22点全部保留

新候选34e9afd05ea3c3b3eddc417d2e6b6126d8c0c53f36f7446e92d32c88221be2fe，无偏置61点每点4回合，22→24；新增.5/+.1（速度误差.00291、yaw.01777、stable1）和1/+.1（速度.06347、yaw.01264、stable1），无旧点丢失、无生存退化，runtime通过。v544共享更新导致的7项损失已在本轮恢复。此证据支持命令域隔离有效，不证明全域/DR或部署完成。

仍固定spring1.885625/D反馈.05/b.002/C.001/DR0/2ms；新候选只有4回合快筛，不继承v539的36回合证书，promoted=false。下一步同候选八点/新慢转及爬行正式回合复验，随后固定弹簧引入小DR、固定1周期延迟；+.3慢转/静止/目标全速度大yaw/最终10.775/最高DR/正式sim2sim仍未完成。证据continuation_v545/final_report.json、retention_audit.json，进程exit0。代码改动仅physics_adaptation可选reference隔离及checkpoint恢复、相应测试，11项通过。

## 2026-10-07 v545：隔离参考分支更新，闭环复验中

新增可选reference_update_domain和冻结原reference副本，将v544学到的修正只应用于正小yaw/速度>.3区域，smooth过渡，原共享残差预算保持，不增加冻结迁移深度。源v539，donor v544；候选34e9afd05ea3c3b3eddc417d2e6b6126d8c0c53f36f7446e92d32c88221be2fe。旧checkpoint无新字段行为不变。修改rmuc_physics_adaptation.py、rmuc_command_adapter.py及test_rmuc_physics_adaptation.py；11测试通过。2080输入保护域输出差0、保存恢复差0，但同输入隔离不等于闭环retention。

固定spring1.885625/D反馈.05/b.002/C.001/DR0/2ms，v545全61点每点>=4回合评估中，预算3200步/600秒；原22点/生存保留为门槛。无新优化器更新、无外加部署偏置。检查continuation_v545/evaluation_status.json与PID接续，不重复启动；promoted=false。

## 2026-10-07 v544拒绝：共享参考分支慢转新增2点、损失7点

无偏置61点各>=4回合，runtime通过，无生存退化，但22→17：新增.5/+.1和1/+.1（yaw.01719/.01233、stable1）；损失1/-.2、1/-.1、1.5/-.1，以及.25的-.1/0/+.1/+.3共7点。retention=false，promoted=false，候选098aad...不接受，源v539保持当前候选。共享reference网络确实能影响目标慢转，但训练会干扰既有负向/爬行技能；22命令合成观测rehearsal不足以保证真实闭环retention。不能继续此共享更新方案盲增训练或将退化候选推进DR。后续先解决现有网络更新范围隔离，仍保持原动作预算；v543已证明不能简单再堆冻结层。报告continuation_v544/final_report.json。

## 2026-10-07 v543拒绝叠层，v544已有参考分支适应完成、retention评估中

v543新增层构造被现有最大冻结深度保护拒绝，0更新/无候选，不增网络层数。v544改为只更新physics.frozen_source.reference_network，旧22命令合成观测rehearsal；600CPU步约3.14秒，独立第二副本轮MAE.00761/.00806，其他actor参数冻结。1.5m/s两项示范不满足原共享残差预算，明确拒绝使用，未扩预算；训练只使用.5/1m/s四项改善轨迹，其中+.3示范自身仍未过门槛，不能宣称已学会。

候选098aad676022143cbce7f39d93ab62ee0dc629d8cf40323a7e88450be135b1ae正在同spring1.885625/D反馈.05/b.002/C.001/DR0/2ms下跑无偏置全61点retention，3200步/600秒。此参考分支更改不保证保护域输出逐值一致，必须看实际旧22点和生存门槛；无报告前promoted=false。首次CPU预检因1.5示范不可行停止，已保留fit_rejection.json；teacher_feasibility.json记录筛除原因与实际选用四项。

## 2026-10-07 v542完成：慢转可达性与渐进延迟课程

固定v539网络/spring1.885625/D反馈.05/b.002/C.001/DR0，六慢转条件×四差分偏置×两副本完成，runtime通过。右轮-.15/左轮+.15时，.5/1/1.5m/s的+.1均短测通过（yaw误差.00985/.00657/.01978，stable1），1.5/+.3通过（.04496，stable1）；.5/+.3与1/+.3仍失败（.06059/.06598），但比无偏置约.30明显改善。24条件全存活，每点仅2回合；这是诊断可达性，不是已训练策略或retention通过，promoted=false。paired轨迹保存在teacher_probe.startup.pt，拟合前必须核对残差预算与门控可训练性，不能把固定偏置写入部署。

用户新要求：延迟幅度也逐级迁移，禁止低DR时直接随机抽1–3周期。源代码env.py显示strength只控制命中概率，命中后从min/max抽幅度；因此明确准备无延迟→观测固定1→联合固定1→联合固定2→联合固定3→最终混合1–3的六份profile（policy8ms）。已通过validate_joint_dr_profile，未运行。见continuation_v542/delay_curriculum.json；原schema观测/通信共用幅度边界，尚无独立action-only课程，不宣称已实现。继承profile惯量DR=0，不能冒充最终±20%惯量覆盖，启动前仍须匹配当前动力/实际范围；弹簧固定后再引入小DR档。

## 2026-10-07 v542慢转覆盖查错：当前两层network在目标命令下更新门控为0

从v539实际checkpoint恢复actor后核对：vx=.5/1/1.5、yaw=+.1/+.3六条件，最外层physics.network及其frozen_source.network的合成更新门控均0；.25/+.1外层为1。不能继续在原训练入口加轮数来修这些慢转条件，也不能将此解读成全部网络无输出或物理不可达。冻结reference/scoped分支仍会提供动作。证据continuation_v542/gate_audit.json，完整代码audit_gates.py。

固定spring1.885625/D反馈.05/b.002/C.001/DR0/2ms，v542零训练轮差分0/-.05/-.1/-.15六命令探针正在检查可达性，1600步/最多300秒；偏置只诊断，不进入部署。源SHA474a44e8859fa5f01f50968a3c00123c4d69d08e236e2672c9d8e5d5bb5c530f。teacher_status.json查PID，结果前不默认拟合；新候选必须全61点retention。

## 2026-10-07 v541完成：spring1.885625八点8/8，各36回合

同v539候选474a44e8859fa5f01f50968a3c00123c4d69d08e236e2672c9d8e5d5bb5c530f，DR0/D反馈.05/b.002/C.001/2ms/5.5峰值下，八点8/8、.25m/s正向+.1/+.3两点2/2，每点36回合、全存活。runtime与episode gate通过；此前v540全61点22保留。+1.4 yaw误差.049647、stable.571628，仍接近门槛，不能称鲁棒充分，不直接加大弹簧或DR。正爬行+.1/.3 yaw误差.03543/.01472，stable均1。

下一步固定本档，补+1.4裕量与慢转命令域。大角速度position-authority门控在约.45以上激活，不能将八点通过推断为小yaw通过；静止仍失败，.5/1/1.5m/s正小yaw仍欠转。原目标全域、1–2.5/ .8–1.8范围、最终10.775、最高DR、多seed、动态切换及正式MuJoCo仍未完成，promoted=false。v540/v541契约清除了继承的trainable/PPO字段，真实均零优化器更新；没有改运行物理或结果。证据continuation_v541/final_report.json，评估进程exit0。

## 2026-10-07 v540完成 / v541同候选32回合复验

v539同checkpoint在spring1.885625下61点各>=4回合：22/61，旧22点全保留，无生存退化，runtime核对通过。只完成快筛，未接受为正式新档，promoted=false。下一步v541固定同SHA474a44e8859fa5f01f50968a3c00123c4d69d08e236e2672c9d8e5d5bb5c530f和物理条件，八点+两爬行各>=32回合；最多6000步/900秒，无训练。D反馈.05/b.002/C.001、5.5峰值/3持续、DR0、2ms保持，部署D_comp=.052。命令见continuation_v541/evaluation_command.json；最终10.775/DR/全域/正式sim2sim仍未完成。

## 2026-10-07 v540：同v539策略spring1.61625→1.885625迁移筛查

只改变等效弹簧，源474a44e8859fa5f01f50968a3c00123c4d69d08e236e2672c9d8e5d5bb5c530f，网络不更新；D反馈.05/b.002/C.001/DR0/2ms与5.5峰值保持。61点各>=4回合检验原22点retention，失去任何旧通过点或生存退化即拒绝新档。先定位再局部适应，不盲增PPO；同最终候选新档32回合仍必须补，不能继承旧候选证书。状态continuation_v540/evaluation_status.json。

## 2026-10-07 v539完成：+1.4裕量改善，原22/61保留

同spring1.61625下，仅现有center+1.4专家400CPU拟合步，示范按最小yaw误差选右-.006/左+.006（-.003也稳定，不宣称选择的是最小幅值），验证轮MAE约.003、腿约1e-5、保护域逐值一致。候选474a44e8859fa5f01f50968a3c00123c4d69d08e236e2672c9d8e5d5bb5c530f，无运行偏置61点至少4回合22全部保留、无存活退化，+1.4速度误差.00940、yaw.02952、stable1、存活1。

runtime/freeze通过；当前新候选仅快筛，不继承v53736回合证书。解析bank约14.94度仍大，不称完整压弯姿态认证。selected_for_next_stage=true但promoted=false；下一步可小步spring迁移筛查，最终接受下一物理档前必须同最终candidate>=32回合。D反馈.05/b.002/DR0/2ms保持，全域/静止/最终10.775/最高DR/MuJoCo仍未完成。所有进程结束，证据continuation_v539/final_report.json。

## 2026-10-07 v539：固定1.61625弹簧补+1.4裕量

v538新档32回合虽8/8，+1.4 signed yaw误差-.04301、绝对.04955、stable.5739接近门槛。v539固定同v537源3a5b024c40f1ef1edc3f6a820660303482fdec5c3cc2ddfa0a20315dcfc00b15，诊断右-左+差分0/.003/.006/.01，1600步两副本；物理D反馈.05/b.002/C.001/弹簧1.61625/DR0/2ms不变。若有stable>=.9的通过轨迹，只拟合已有+1.4局部专家，不增预算/外环，随后无偏置全61点retention。状态continuation_v539/teacher_status.json，未完成不晋升。

## 2026-10-07 v538结束：新弹簧1.61625八点8/8各36回合，但+1.4裕量不足

同v537 candidate3a5b024c40f1ef1edc3f6a820660303482fdec5c3cc2ddfa0a20315dcfc00b15，八点8/8、正爬行两点2/2，各36完整回合、全存活，runtime读回通过，无丢失快筛通过点。+1.2 yaw误差.01681/stable1；+1.4 yaw.049546/stable.573857，虽现门槛通过但接近边界，不称鲁棒裕量充足。

此阶段可作为下一步迁移候选selected_for_next_stage=true，promoted=false；先巩固+1.4再较大弹簧步长/DR，不盲升档。只有单seed/DR0/静态命令、阶段弹簧1.61625、2ms；解析bank未纳入验收，不能宣称全域内倾压弯或sim2real通过。已在continuation_v537生成绑定候选/导出/MJCF/hash与实际阶段参数的controller契约，D_comp=.052核对不变；生成契约非部署验收。所有作业结束，报告continuation_v538/final_report.json，最终目标仍未完成。

## 2026-10-07 v538新弹簧同candidate32回合评估启动

源v537 3a5b024c40f1ef1edc3f6a820660303482fdec5c3cc2ddfa0a20315dcfc00b15，无网络更新，固定spring1.61625/D反馈.05/b.002/C.001/DR0/2ms。历史八点+两爬行10条件×12副本，每点>=32回合，最多6000步900秒。部署D_comp=.052从当前头文件核对；这是未饱和总阻尼参考，非饱和等价证明。状态continuation_v538/evaluation_status.json，无报告前不升级弹簧/DR；不借旧1.346875结果拼接认证。

## 2026-10-07 v537结束：spring1.61625无偏置恢复22/61，+.1.2修复

同新弹簧阶段偏置示范右-.005/左+.005，拟合原local expert2，不加预算/外环；CPU400步约5.2秒，第二副本轮MAE.00077、其他参数冻结/保护域逐值一致。candidate SHA3a5b024c40f1ef1edc3f6a820660303482fdec5c3cc2ddfa0a20315dcfc00b15。无偏置61点>=4回合恢复22/61，v535旧22点全保留、无存活退化；1m/s+1.2速度误差.07129、yaw.01936、stable1、存活1。runtime/freeze通过，5450输入JIT差0/ONNX最大2.98e-7。

新弹簧1.61625只有快筛，没有32回合或多seed认证，promoted=false。旧源checkpoint继承infos不得当新物理契约，实际以本轮experiment/runtime明确读回为准；后续部署导出须从本轮配置构建。同一candidate需新档32回合后再推进，不借用旧1.346875证据。最终10.775/DR/全域/静止/正式sim2sim未完成，解析bank未纳入门槛。所有进程结束，证据continuation_v537/final_report.json。

## 2026-10-07 v537运行：固定新弹簧针对+1.2局部适应

v536仅+1.2欠转退化，当前使用v535同源4ed315d258ca9b6475d4a4e5a2c20d1bda50fe77022ceca2d5eb178d0dc91ab6，固定spring1.61625、D反馈.05/b.002/C.001/DR0/2ms。4条件右-左+偏置0/.005/.01/.02，1600步采集轨迹，若无稳定通过示范则不拟合；如有则只更新原center1.2局部专家，保留其他专家与预算，随后无偏置61点retention原22点。无需新增网络或任务外环。命令/状态见continuation_v537；无最终报告前不晋升。

## 2026-10-07 v536结束：弹簧1.61625迁移快筛21/61，仅+.1.2退化，不升档

同v535候选无网络更新，spring1.346875→1.61625，runtime读回通过。61点原22→21，仅1m/s、+1.2rad/s原通过点丢失：存活1、速度误差.05571、yaw.05305、stable.40970。无其他通过点丢失，无存活退化；retention=false、stage_accepted=false、promoted=false。未完成新弹簧适应，不能把全存活宣传成通过。

下一步固定1.61625新档，仅针对+.1.2现有局部专家开展有界适应并复验retention；保留v535在1.346875下八点36回合8/8基线。不需要无差别重训全部网络或跳DR。完整证据continuation_v536/final_report.json；当前所有作业结束，最终10.775/最高DR/全域/正式sim2sim未完成。

## 2026-10-07 v536弹簧小步迁移前测正在运行

v535已当前D.05/阶段弹簧1.346875/DR0八点各36回合8/8，下一步回到目标弹簧迁移主线。v536不训练，固定同candidate4ed315d258ca9b6475d4a4e5a2c20d1bda50fe77022ceca2d5eb178d0dc91ab6，仅弹簧1.346875→1.61625Nm/腿，61点原22点retention；D反馈.05/b.002/C.001/DR0/2ms保持。网络冻结不代表物理变化后的行为保留，完整评估决定是否需要适应，不能默认升档。至少4回合快筛，非32回合认证；状态见continuation_v536/evaluation_status.json，禁止重复启动。

## 2026-10-07 v535完成：同checkpoint八点各36回合8/8，61快筛22/61

候选4ed315d258ca9b6475d4a4e5a2c20d1bda50fe77022ceca2d5eb178d0dc91ab6，无执行偏置/外环。先61点>=4回合原21全保留新增+.8，再同candidate八点及两爬行各36完整回合全存活：历史8/8、爬行2/2，+.8速度误差.00866、yaw.00878、stable1。runtime及冻结审计通过，5450输入导出JIT差0/ONNX最大2.98e-7。

这是当前D反馈.05、机械b.002、阶段弹簧1.346875、DR0、2ms、单seed静态命令跟踪的阶段里程碑；不是旧DR.05或当前最高DR完成，不是最终弹簧10.775。解析bank未纳入门槛且大转bank误差仍较大，不称完整内倾压弯/部署通过。选择本候选作为下一阶段源selected_for_next_stage=true但promoted=false；静止/负爬行/完整命令域/多seed/最终DR/MuJoCo仍欠缺，禁止拼历史证书。完整证据continuation_v535/final_report.json、report32.json，所有进程结束。

## 2026-10-07 v535快筛22/61，历史八点8/8；同checkpoint32回合评估中

无偏置完整61点，原21点全部保留，无存活退化。+.8新增通过：速度误差.00873、yaw.00823、stable1、存活1，历史八点快筛8/8。现有expert5示范拟合有效，但每点4回合不够；正在同candidate4ed315d258ca9b6475d4a4e5a2c20d1bda50fe77022ceca2d5eb178d0dc91ab6跑八点+两爬行点，每点12副本、至少32回合，产物evaluation32.json，状态evaluation32_status.json。不得借用v53336回合结果拼证书。

依旧DR0、阶段弹簧1.346875、D反馈.05/b.002/2ms、无外环/偏置；bank未作为解析姿态门槛，promoted=false。静止/负爬行/其余全域/目标弹簧/最高DR/多seed/MuJoCo正式验收继续未完成。

## 2026-10-07 v535：现有+.8局部专家示范修复，无偏置评估中

v534确认八点+.8过转，当前已有local expert index5/center+.8，无需新增网络。v533源固定物理扫描右+/左-偏置0/.01/.025/.04，只有.01有效：yaw误差.07142→.00859、速度.00799、stable1、存活1；更大偏置欠转，不能泛化成统一固定补偿。

CPU仅拟合现有expert5，400步约5.8秒，保留启动/ramp、第二副本验证，时序重建<1e-5；轮MAE.000767、腿MAE<=.000017，其他tensor冻结、保护域逐值一致、原authority预算不变。候选4ed315d258ca9b6475d4a4e5a2c20d1bda50fe77022ceca2d5eb178d0dc91ab6正在撤全部偏置61点retention，源为v53321/61。D反馈.05/b.002/C.001/阶段弹簧1.346875/DR0/2ms保持，未有闭环结果不晋升。完整证据continuation_v535。

## 2026-10-07 v534完成：每点36回合，历史八点仍7/8，新增正爬行2/2

同v533源213516858f9429df455ca57e941e74ec655924e38f632ea1dc596e98bb2dc982，不训练；10条件各36完整回合、全存活，runtime读回通过。历史八点7/8，唯一1m/s+.8失败：速度误差.01807、yaw.07020、stable0；其余八点中7个跟踪通过。新增.25/+.1 yaw.03167、.25/+.3 yaw.01766，均stable1，原快筛通过点无丢失。

此为单seed静态命令、DR0、阶段弹簧1.346875/D反馈.05/b.002；12副本重复确定性回合不等于独立多seed。解析bank未作为门槛，大转弯bank误差约6–15度，不能称完整压弯姿态或sim2real完成。promoted=false。下一项应回历史八点+.8跟踪缺口（现爬行头在该域门控为0），不能继续只修.25后宣称全域完成。报告continuation_v534/final_report.json；所有进程结束，最终DR/弹簧/速度域/起停/正式sim2sim仍未完成。

## 2026-10-07 v534：当前候选历史八点与新增慢转32回合检查

v53321/61仅每点4回合快筛。v534不训练，同source213516858f9429df455ca57e941e74ec655924e38f632ea1dc596e98bb2dc982，历史1m/s±.8/1/1.2/1.4及.25m/s+.1/+.3，每点12副本、至少32完整回合，最多6000步/900秒。D反馈.05/b.002/C.001/阶段弹簧1.346875/DR0/2ms保持。报告必须区分回合增加与独立多seed，不能凭重复确定性轨迹声称鲁棒认证；本次10点也不替代完整61点/全命令域。先查continuation_v534/evaluation_status.json及真实PID，避免重复评估；无结果前不晋升。

## 2026-10-07 v533结果：同61点20→21，正慢转稳定裕量改善

40更新210.7秒，final SHA213516858f9429df455ca57e941e74ec655924e38f632ea1dc596e98bb2dc982。runtime/freeze/覆盖通过，82阶段实际env leg scale与固定eval/controller映射核对通过。61点各至少4回合，原20全部保留，无生存退化；新增.25/+.1速度误差.02246、yaw.03131、stable1，.25/+.3速度误差.00393、yaw.01920、stable1，均存活1。相较v532+.3 stable.533边缘通过，当前裕量明显改善。不能把改善全部因果归于scale修正，因为无同预算未修正对照。

负-.3 yaw.11867/stable0仍失败，静止仍失败；.25直行虽原门槛通过但速度误差.13735仍偏大。仅Isaac阶段候选21/61，promoted=false，不升DR或弹簧；下一步保持同一命令—执行器映射补负慢转/静止，再扩域。v532的MuJoCo失败不能继承为新候选通过，v533尚未跨引擎行为复验。全部作业结束，报告continuation_v533/final_report.json，完整目标未完成。

## 2026-10-07 v533：修正低速训练腿目标尺度与验收不一致

从现有代码与v531 actual_env查到：PPO静止/爬行阶段scale=.20，固定命令eval及由其生成的MuJoCo为.18/.19；两路径分别读取stage scale和manual插值，非同一映射。v532示范采集与无偏置评估已使用后者，所以其20/61结论仍成立；不把尺度差异视为全部失败的唯一原因。逐阶段审计只发现stage0/1/2/7不一致，其余一致；prepare.py将后续训练对齐既有eval/MuJoCo映射，不改历史产物或门槛。

v533源v532 f6b74bfb351a559bbc45a5caa079c60367848b2b9273ab86760bae2a7aab4a0f，40更新、同61点retention原20点；正向隔离保持，不能声称本轮会训练静止/负向。唯一配置变化低速腿scale，D反馈.05/b.002/C.001/弹簧1.346875/DR0/2ms保持；非最终动力认证。完整命令与leg_scale_audit在continuation_v533，运行状态以真实PID为准。

## 2026-10-07 v532结果：无偏置Isaac20/61，MuJoCo行为未通过

原管理进程被中断后保留原评估PID完成，未重跑；退出码不可回收记null，以61点完整JSON、每点>=4回合、checkpoint hash核验产物。候选f6b74bfb351a559bbc45a5caa079c60367848b2b9273ab86760bae2a7aab4a0f，无运行偏置，旧19点全部保留，新增.25/+.3通过：速度误差.02383、yaw.04971、stable.53344、存活1。+.1 yaw.05016/stable.42475仍失败，且+.3接近门槛，不能称鲁棒掌握。freeze/runtime通过，promoted=false。

补scripts/verify_rmuc_export.py正向低速门控边界覆盖；5450输入JIT完全一致，ONNX最大2.98e-7。原rmuc_peak55.xml、显式同D反馈.05/b.002/C.001/阶段弹簧1.346875/2ms契约下三个CPU MuJoCo探针：.25/+.1与+.3均12s但后半yaw误差.32388/.22633、速度误差.12629/.15776；.5直行触发tilt/height，未通过sim2sim。保存MuJoCo实际输入重放，JIT差0、ONNX5.96e-7、MuJoCo raw4.77e-7，排除此次导出数值丢分支，不能证明物理一致或硬件成功。

控制契约从候选hash、导出hash、MJCF hash和evaluation命令腿scale生成；stand scale.18来源为当前配置/源码（evaluation未记录该字段），出处另记leg_scale_provenance.json。旧wheel_authored_effort_limit配置残留3不作运行限幅证据，actual actuator与motor readback为5.5。评估实际b=.002固定，配置DR范围字段不等于实际采样。证据continuation_v532/final_report.json、same_input_outputs.npz、mujoco_summary.json。全部进程结束；保持Isaac-only候选，不升DR/弹簧，不将诊断视频当验收demo，完整目标未完成。

## 2026-10-07 v532：由可行轨迹初始化现有正向残差，正在无偏置验收

不再重复PPO/std追加。并行只读审计确认继承残差没有反向破坏、预算仍可用。v531源eeb2136f9c80bb64b2d0c0fc9c951f13bcda6d1de8bb95bdc989156031a999af，同物理8条件两副本1600步扫描89.4秒；.25/+ .1选右-.075左+.075，yaw误差.01238；.25/+.3选右-.125左+.125，误差.04954、stable.7107。仅示范诊断，不是策略通过或部署外环。

按environment_bucket_ids对齐obs/raw action/episode age，验证source+时序bias重建误差<1e-5；完整保留启动、ramp、转弯段，拟合六槽以约束腿，拒绝共享残差预算外标签，按第二副本验证。现有physics.network仅600个CPU监督步约1.7秒，轮MAE.00182/.00206，腿最大MAE.00029，冻结/保护域检查通过；无新增网络/动作权限/外环，optimizer已移除，待以后显式重置PPO。

候选SHA f6b74bfb351a559bbc45a5caa079c60367848b2b9273ab86760bae2a7aab4a0f，正在撤掉全部偏置的61点至少4回合评估。D反馈.05/b.002/C.001/阶段弹簧1.346875/DR0/2ms保持；拟合通过不等于闭环通过，原19点retention与实际runtime须通过，不自动晋升。状态见continuation_v532/evaluation_status.json。沙箱无法访问驱动，已通过授权外执行GPU0，未触碰参考副本。

## 2026-10-07 v531完成：探索增大仍19/61，停止同类短训追加

40更新209秒，final SHA eeb2136f9c80bb64b2d0c0fc9c951f13bcda6d1de8bb95bdc989156031a999af，freeze/runtime通过，轮std按配置.16。同61点19通过无丢失/生存退化，负向/直行仍保持；+.1 yaw误差.10234→.09605，+.3 .29733→.25951，均未通过，+.3速度误差.00464。提升有限，不把训练随机转动或小幅误差下降当掌握。promoted=false，不升级弹簧或DR，不继续只增std/短训次数。下一路线需考虑由已测可行低速轨迹初始化策略并保留旧命令，不能将诊断固定偏置作为部署外环。全部进程已结束，完整目标仍未完成。

## 2026-10-07 v530完成19/61无新增；v531只增训练探索

v53040更新207.6秒，最终SHA998cbc10fe3b9f1cc1099a587bb6718d5f52cef4e80ca9789a6d421243f5fe86，freeze/runtime通过。中断使原管理进程退出，评估PID仍活着，保留原任务完成；退出码无法回收，已记录null，凭完整61点各>=4回合JSON、checkpoint hash及无traceback验证产物，不伪造exit0。报告19/61、原19全保留，.25负向/直行关键指标与v527逐值一致；正+.3 yaw误差.29767→.29733，几乎无改善。隔离保护有效不等于训练成功。

v531从v530仅将轮采样std.08→.16，固定其余配置40更新，检查跨过弱响应区是否改善；不扩物理权限，不新增外环，部署不加采样噪声。同61点确定性retention，未完成前不晋升。物理仍D反馈.05/b.002/C.001/阶段弹簧1.346875/DR0/2ms，完整目标仍未达成。

## 2026-10-07 v530正向爬行更新隔离启动

基于v529共享梯度干扰证据，从v527派生原输出等价候选4683426c8d19612fe707b61f90cd57ffd428a367066dfda0f0139ab5412bd260。top physics.network复制为冻结reference，新checkpoint显式positive_straight_enabled=0及正yaw gate；旧checkpoint默认语义不变，导出/恢复/预检同步支持。26测试通过，100完整actor输入初始逐值一致，正+.3/负-.3 gate为1/0，测试非零更新保留负向与>=.45速度输出。原残差预算与电磁限幅不扩。

v530 GPU0/40更新，物理/采样/reward保持v527：D反馈.05/b.002/C.001/弹簧1.346875/DR0/2ms、dense yaw .5，自动61点retention，原19点不得退化。只验证隔离效果，不声称已修复行为；运行状态见continuation_v530。最终目标未缩小，不升级弹簧或DR。

## 2026-10-07 v529固定rollout：发现慢转统计缺口与共享梯度冲突证据

wheel_advantage_audit原来仅覆盖1m/s，遗漏.25m/s；补充四个爬行yaw并标记vx，9项测试通过。v527源固定策略8x64步、512env采集，actor与critic全部张量前后逐值一致，未训练。age>=250样本26460：正+.3组原始/去组均值score约-.0927/+.0237，信号弱；负-.3欠转约+.836，符合右+/左-修正方向。报告continuation_v529/final_report.json。

同snapshot进一步计算共享physics网络一阶surrogate梯度：+.3自身组对轮差分方向导数-.158（符合正转所需减小右-左），全混合组+.137（反向）；负向组对同一输出贡献约+.97/1.60。支持正负共享更新干扰的候选解释，尚非Adam真实更新因果证明；重放均值最大差.000672，原始梯度未含PPO后续clip/Adam，相关样本无独立置信区间。gradient_interference.json保留限制，不能写已锁定唯一根因。下一步应隔离正向爬行更新、原负向/直行输出保留，不加共同reward或盲增轮数。当前所有进程已结束，候选仍v527、19/61快筛、promoted=false，完整目标保持。

## 2026-10-07 v528慢转闭环响应：轮差分有效，不是模型完全转不动

v527同物理，.25m/s目标±.3，14条件1600步短诊断93.7秒。正+.3无偏置实际yaw约+.00033；右轮action-.15/左+.15后约+.22844，生存1，速度误差有符号-.02857。负-.3无偏置实际yaw-.17965；右+.05/左-.05后-.31217，生存1。响应方向可用且有明显正负不对称；不能据此宣称实机符号已验或策略通过。腿槽3分别-.15/-.3未解决正yaw（实际约.0031/.0520），不能把慢转失败仅归因腿clip。低域consumer确实仍clip±1，历史1m/s大yaw域才扩腿权限。

用户问快转会慢转不会：旧专用网络/动作域和采样覆盖不同，慢域之前没有训练与可训练gate，不能假定跨域泛化；最新控制扰动说明当前策略差动力矩不足/不合适，非轮控完全失去响应。下一步核对PPO差动力矩更新与探索/奖励，不把固定偏置当部署外环或直接晋升。报告continuation_v528/final_report.json，promoted=false；原19/61候选及完整目标保持。

## 2026-10-07 v527结果：61点18→19，速度改善但正向慢转仍停滞

40更新212秒；final SHA 0ceab5c6c70bd6f15afb3ca886c6a54f8b8eff3a3351ace99c041e2b9c188d75，freeze/runtime/覆盖通过。61点原18全部保留，无存活退化，新增.25m/s直行通过（速度误差.13735、yaw.00613、stable1）。注意.137m/s对.25目标仍是显著误差，按现有门槛通过不等同高精度低速。负向-.1保留且速度误差.03366；正向+.1/+.3速度误差降至.01862/.01962，但yaw误差仍.10303/.29767，未解决转向。静止6次完整回合仍存活0；不能只凭次数减少断言已稳。

这次没有同预算未改reward对照，不能将速度改善单独归因dense yaw；正向yaw持续停滞，不再盲加同一项权重/训练轮数。下一步定位低速动作差分到实际yaw响应。候选19/61仅快筛，promoted=false，完整目标仍未完成，全部运行已结束。报告continuation_v527/final_report.json。

## 2026-10-07 v526轨迹否定爬行预算截断假设，v527仅试dense yaw

v525源同契约600步六点诊断，保存前500步obs/action并离线重放完整actor（最大差1.91e-6）。age>=125样本：五个.25m/s命令新增physics修正截断率均0；静止右/左轮16.7%/2.9%。这仅是采集窗口结论，不能据训练残差max碰上限断言爬行长期无权限；不扩大预算。报告continuation_v526/final_report.json、residual_audit.json。

下一实验v527直接续v525源d98f05b1422ef8c61a6e97bcc2ca6f6325c8431ccf373765c452b113082bea0f，仅dense小转弯yaw代价weight0→.5；网络/命令/物理均保持，40更新后同61点retention，不宣称奖励一定有效。D.05/b.002/C.001/阶段弹簧1.346875/DR0/2ms、5.5峰值/3持续保持。10项相关测试通过，状态见continuation_v527，未有报告前不晋升。

## 2026-10-07 v525结果：旧17点保留，爬行5点全存活但仅1点快筛通过

60更新305秒，final SHA d98f05b1422ef8c61a6e97bcc2ca6f6325c8431ccf373765c452b113082bea0f；freeze/runtime/实际12类命令覆盖通过。61点评估18通过：原56共同网格17→17无丢失/存活退化；新增.25m/s五点全存活，仅yaw-.1通过（速度误差.0899、yaw误差.0460、stable1）。这5点缺少旧候选同条件基线，不能声称全部由本轮训练新学会。静止14次tilt，存活0；.25直行超速约.244m/s，正向+.1/+.3 yaw接近0，误差.103/.298。promoted=false；不能称起停/低速域完成。

必要查错发现：归一化yaw误差在相对误差1处截断，dense小转弯项为0；这是正转停滞的待验证因素，不是已证因果。训练残差最大值触及预算，也需核查余量；raw腿action>1在authority consumer下不等同实物限位。failure_diagnosis.json保存指标。停止重复同配置长训，不升弹簧/DR；下一步应在固定物理下单独检验低速yaw代价或控制余量，完整目标未完成。本轮所有训练评估终止。

## 2026-10-07 v525：修补真实采样/可训练域缺口，固定弹簧补静止爬行

v524只覆盖直行和历史大转弯，静止及.25m/s没有采样且positive gate阻断低速更新，不能通过延长同配置训练获得这部分能力。v525固定D反馈.05/b.002/C.001/弹簧1.346875/DR0/2ms，冻结v524完整actor，新增零初始化低域physics层，update_domain=[.25,.45,.35,.5]，共享原总残差预算。99输入初始完整actor逐值相同；非零更新情况下>=.45m/s输出仍逐值相同，所有新目标gate=1。部署D_comp=.052已从rmuc_low_level.h确认，未更改。

训练源SHA f0e0e90d771615c620d33c14fec07676adadb2569d84994aec0bf51f89b55a3e；GPU0/512env/60更新，静止、.25m/s直行、双向.1/.3rad/s与旧技能rehearsal，实际覆盖需runtime验证。完成后原56点加5点爬行评估，不假定新点已掌握，不替代完整命令域/正式32回合/DR/最终弹簧与sim2sim。状态查看continuation_v525，未有最终报告前只能记运行中。

## 2026-10-07 v524结果：阶段弹簧1.346875保留17/56，尚未扩域

实际60更新297.8秒，56点评估178.2秒；final_checkpoint.pt SHA a6628a92319ebc7a92d7ec2b207587e75b6a238388be75ebcb2fb16408857c3b。freeze通过仅physics.network六张量改变，runtime读回通过；同网格17→17，无新增通过、无丢失通过、无存活退化，quick retention通过。该结果只支持D反馈.05/b.002/C.001/弹簧1.346875/DR0下的小步迁移候选，不支持全命令域、最终弹簧或部署完成。promoted=false，32回合、多seed、DR、sim2sim仍未认证。静止0/0仍有倾倒，下一步固定本档物理，先补静止/低速慢转弯，暂不继续提高弹簧；不要用冻结网络假定机械条件改变后必然保留行为。完整报告continuation_v524/final_report.json。训练及评估均已结束。

## 2026-10-07 v524：缩小弹簧步长，先直行，禁止绕过零梯度预检

v523未通过retention，回到v512候选；v524阶段弹簧1.0775→1.346875Nm/腿，D反馈.05、机械b.002、库仑.001、DR0、2/2/8ms保持。prepared源SHA c04551159bf095cadfae09848786708709d6fe6ab86488ba0179480de5619ad9，初始120输入输出完全相同。此为阶段近似，不是最终气弹簧或sim2real认证。

两次启动在gym创建前因目标±.8的update_domain为零而停止，未训练；此前将此解释为预检误判不正确。真实原因是v512低域update_domain与新positive gate交集阻断目标。修复预检计算base/update/positive/scoped全部支持交集且不消耗RNG，26项相关测试通过；保持保护，改active为1.5m/s直行，.5–1.25直行与旧转弯rehearsal。仅physics.network可更新，冻结输出不等于物理变化后行为保留。60更新后自动56点至少4回合筛查，不替代正式32回合、多seed、DR与MuJoCo验收。第二次失败日志在preflight_attempt2；第一次日志被错误覆盖，不能声称首轮日志完整。当前运行状态以continuation_v524的status为准，未有最终报告前不晋升。

## 2026-10-07 v523弹簧首档已结束：同56点17→16，暂不升档

实际训练60更新303.9秒，旧图全部冻结，新physics六张量更新；最终SHA08b3dbe1add3d3d22d54053c14ce5a5d870d150cf72facc9873ba08fabea431c。D.05/阶段弹簧1.61625/DR0/2ms实际读回通过。56点每点至少4回合16通过，新增1m/s+.8（速度误差.02297、yaw.04059、stable1，bank9.47°仅诊断不作解析bank通过）；丢失1m/s-.2和1.5m/s-.1（yaw.06135/.07352、stable0），无存活进一步退化。暂不晋升或加大弹簧。冻结参数不等于物理变更后的行为保留。

修复本轮report基线计数：v512报告含64点，原56点通过17，额外8点通过5；不能用22比较56点。更正后共同网格17→16，retention仍失败。曾准备更换门控但未启动，已撤回不兼容改动，8项测试通过；不支持仅凭本轮把退化归因共享更新。后续优先在本档保留成功直行/转弯并修复两负转，必要时缩小弹簧增量，不再扩展sim2sim排查；起停/全域/10.775与DR目标继续保留。v523所有报告已完成、无运行作业，promoted=false。

## 2026-10-07 用户纠偏：停止扩展sim2sim诊断，v523开始弹簧首档适应

用户指出今天最后一天，要求从简单到难扩命令域并立即加入气弹簧迁移。接受纠偏，不继续以局部MuJoCo排查占用训练主线。v521八点12000帧consumer输出逐值一致；准备了正向隔离模块/8项测试/1024完整actor输入一致，但v522未启动已cancelled_before_launch，不能写成训练过。

v523从v512完整冻结图新增零初始化physics迁移层（允许至三层显式冻结迁移、共用预算，旧张量逐个核对、165输入完全一致）。源副本SHA17448e430c389f6ef751af7968db4e0440a762bfcfbd623feda5154c7aaee2cf，阶段弹簧1.0775→1.61625Nm/腿，D.05/b.002/C.001/DR0/2ms/5.5峰值保持。512env/60更新10critic+50actor，简单.5/.75/1/1.25/1.5直行与1m/s八点转弯rehearsal，既有16/17通过并不等于所有训练条件通过。GPU0训练已实际开始，status与PID见v523/training_status.json，主流水run.py训练后自动冻结审计与56点每点至少4回合。冻结/非有限/coverage失败或1800s停止；retention失败不晋升。静止/起停未掌握仍属目标，DR/最终10.775和全部能力不得声称完成。

## 2026-10-07 v520：最终1ms执行周期三点行为近似不变，速度取样点一致

读取当前IsaacLab base_articulation_data确认root_lin_vel_w别名root_com_lin_vel_w；MuJoCo BODY mj_objectVelocity以非零角速度实测等于mj_jacBodyCom速度、不等于body原点速度。因此当前两边都取机身刚体COM，不能为修复转弯擅自换原点。

CPU同v512/v518 imp999模型仅轮/physics2ms→1ms，legPD仍2ms、policy8ms，D.05/b.002/C.001/阶段弹簧1.0775保持。0.5直行及1m/s±1三点均12s：速度误差.191102/.201395/.046442，yaw误差.045974/.306167/.174170，和2ms结果近似；没有证据表明周期是当前系统性过跟踪主因，也不代表全域实机高频认证。v520 contract/命令/完整trace/final_report留证，promoted=false。无需重复该周期对比；下一继续针对真实动力与方向性策略鲁棒缺口，目标未完成。

## 2026-10-07 v519：闭链对齐候选八点均存活12s，yaw系统性过跟踪

固定v512和v518/imp999模型，CPU补±.8/±1.2/±1.4六点并复用完全同契约±1报告；无重复训练、无改物理参数。八点每点一次12s全部duration_reached，但全部yaw误差高于.05：负向-.8/-1/-1.2/-1.4分别.17061/.17383/.17111/.18206，正向+.8/+1/+1.2/+1.4分别.33922/.30641/.28405/.36375。负向速度误差.0264–.0752，正向.1597–.2456。说明剩余问题呈双向过跟踪且正向较重；不等同八点正式通过，更没有32回合/多seed/最高DR认证。

v519/command.json与六份完整trace、复用报告路径、final_report保存。原入口MJCF不变，promoted=false。下一应围绕方向不对称的动力/策略响应核对，不再重复证明“跑得完12s”，也禁止增加未训练任务外环来掩盖误差。全命令域、静止/起停、最终弹簧/1ms/DR仍未完成。

## 2026-10-07 v518：实测闭链软度差异，阻抗对齐后.5直行不再早期倒地

新增rmuc_closure_audit.py，从USD四个球铰局部锚点和Isaac运行时xyzw刚体位姿计算世界距离；恒等/90度变换核对通过。固定v512/D.05/弹簧1.0775/DR0/2ms采500步，与MuJoCo共同467样本：Isaac最大闭合距离.13857mm，原MuJoCo1.44514mm；首8ms差异已出现，支持实际连接软度不一致，而非仅静态几何差异。

只改诊断副本equality solimp，原MJCF/质量/几何/接触/D/策略不改：常数阻抗.99闭合误差.820mm、10.936s仍倒地；.999闭合误差.08108mm、.5直行跑满12s，后半程速度误差.19110、yaw误差.04597。相同.999副本1m/s±1均12s：速度误差.20108/.04624，yaw误差.30641/.17383；正转yaw比原.279略差，不是全面改进。仅证明物理软度对稳定/速度有实质作用，仍不通过正式yaw gate，未替换正式入口、promoted=false。

诊断候选artifacts/continuation_v518/imp999/model.xml，hash与controller契约、全命令、读取数据、行为报告、final_report均在v518。后续同契约sim2sim可围绕此候选开展，但不得当最终标定，也必须保留原XML对照、正式八点/多seed/最终1ms与DR门槛。无训练更新，无运行任务。

## 2026-10-07 v517：排除作者armature缺失，闭链刚度敏感但不是充分修复

USD实际作者属性右active/long/wheel均physxJoint:armature≈.01，不能因配置null误判为零，也不应擅自把MJCF非腿armature清零。固定v512/.5直行/2ms/D.05/弹簧1.0775，CPU三份只读源模型诊断副本：baseline3.728s失稳、闭链solref时间常数.008→.004后7.736s失稳、仅接触solref.008→.004后3.808s失稳。闭链最大偏差1.445mm→.812mm，但加硬后后半程平均速度-1.247m/s，仍失败，不将存活延长称行为改善或采纳刚度。

run_mujoco_probe新增闭链距离和最深接触距离记录；源rmuc_peak55.xml未改，策略未改，无训练、promoted=false。v517命令/各模型hash契约/三份行为报告及final_report齐全。下一需要Isaac动态闭链误差才能解释合理对齐范围，不能任意调软硬到通过。

## 2026-10-07 v516：用户旧MuJoCo视频对应6684，旧记录并非仅yaw失败

用户提供桌面mujoco/rmuc_sim2sim_dr005_yaw1_fixed_mirror.mp4。按同名找到v423/mujoco_dr005_yaw1_fixed_mirror.json，其ONNX SHA569cf147d1464668d684fdd686424bda08e87d7d5ecb5902bd2b72b3a7adfbfc与export_locomotion_6684/policy.onnx完全一致（视频文件本身无checkpoint元数据，不冒称二进制hash匹配）。旧命令1m/s,+1rad/s跑8秒，4–8秒平均速度.47943、yaw1.32341，速度误差.52057、yaw误差.32341；当前v512相同窗口速度误差.37207、yaw误差.28044。该点当前并未比旧视频指标更差；不能把旧视频流畅观感说成速度通过。旧6684保留对照，.5直行当前失稳仍需修复。两者checkpoint/轮动力契约不同，此比较不作因果归因。

本轮重新只读审核USD/MJCF：总质量19.22113937/19.22113998kg、body名单一致、初态四闭链闭合误差约1e-16m；root坐标系位置/COM最大差5.87e-7m、惯量分量差3.69e-7。仅作者参数库存一致，不代表runtime物理一致，剩余armature/接触/约束仍要对照。证据v516/model_audit.json/final_report.json，无模型写入、无训练晋升。

## 2026-10-07 v515：首16物理子步同状态电机力矩一致

evaluate_rmuc_grid在显式startup capture时记录首环境前16次sim.step之前的关节位置/速度、目标、actuator实际输出和执行动作，默认不启用；20控制步采样不作行为认证。固定v512/.5/D.05/弹簧1.0775/2ms，将Isaac真实输入送入MuJoCo/controller compute_effort重算，16次最大力矩差4.3335e-7Nm、腿目标差3.373e-9rad。PD60/2、限位和电磁输出在该窗口一致；不包含被动wrench、约束反力或接触求解，不能称动力整体一致。下一检查这些尚未覆盖的动力项，不以改reward掩盖consumer问题。完整数据、可复现audit.py和final_report在v515，无运行任务，promoted=false。

## 2026-10-07 v514：定位.5启动轨迹，首8ms关节运动已分歧

固定v512/D.05/阶段弹簧1.0775/DR0/2ms采500控制步、2环境，无训练。首命令min-episodes0被参数校验拒绝，重试min1；不足完整回合只作轨迹，不作行为通过。与v513 MuJoCo同一.5直行逐步比较：初始观测最大差1.192e-7，首8ms关节速度观测差.02691、腿位置观测差.004323；0.4s Isaac速度约+.0473，MuJoCo约-.0205。后续闭环放大，不能仅归因reward/D，也未证明唯一动力根因。速度误差观测存在裁剪，报告不把裁剪后的重构速度当实值。

artifacts/continuation_v514保存500步Isaac输入/动作、两仿真对齐npz、最终报告。代码静态施力审计未见弹簧/腿PD重复施加证据；下一优先对齐首动作下的物理子步力矩/约束/接触响应。当前无运行任务，完整目标未完成，不增加盲训。

## 2026-10-07 v513：v512导出一致，2ms MuJoCo行为失败，禁止晋升

修复consumer诊断周期错配：separated默认仍1ms，只有diagnostic_physics_dt_s=.002且certified=false才允许2ms；腿PD步数按周期计算（1ms两步、2ms一步）。10项controller测试通过。export保存v512 scoped残差域元数据并覆盖门控边界。原训练课程数组与评估默认数组不一致，build_contract拒绝混用；改用评估实际runtime腿尺度数组，明确为诊断来源。0/.3/.4/.5/.75/1/1.5m/s插值尺度与训练一致，未发现这些点的尺度错配。

同v512 SHA989306b78f2832fb89cad1fa0c3dfe0553e4736b9bfcb686ecbed51c61eca2f2、rmuc_peak55.xml不改结构、D.05/b.002/C.001/弹簧1.0775/2ms/DR0，CPU三个MuJoCo诊断：1m/s±1跑满12s但后半程速度误差.37152/.20551、yaw误差.27921/.17533；.5直行3.728s触发tilt_or_height停止，全部不认证。同输入3467帧JIT误差0、ONNX/MuJoCo raw误差4.768e-7；.5初始Isaac/MJ观测误差1.192e-7。因此不能把失败归因于导出漏模块或初始观测明显错位，下一定位动力轨迹差异。未生成“成功验收视频”。部署常量D_comp=.052已核对存在，非饱和/饱和差异仍须实机验证。v513最终报告/同输入npz/全部hash/行为报告齐全，promoted=false，无运行进程。

## 2026-10-06 v512：冻结局部残差隔离保住.5直行，原56点17通过

不重复训练：v509完整网络为主，仅前进.3→.45平滑进入/.55→.75平滑退出、|yaw|.02→.1退出的区域选择v511同结构physics网络；其余命令保持v509，旧actor/两网络冻结、共用原残差预算、不加任务外环。CPU完整actor保护域60输入逐值一致、3域内输入与v511一致；7项physics/隔离/trace测试通过。checkpoint候选SHA989306b78f2832fb89cad1fa0c3dfe0553e4736b9bfcb686ecbed51c61eca2f2，artifacts/continuation_v512/candidate.pt。此为推理候选，已删除旧优化器状态，后续续训须显式重建优化器。

同D.05/b.002/C.001/阶段弹簧1.0775/DR0/2ms，原56点每点4回合17通过，v509原16点全部保留，新增.5直行survival/stable1、速度误差.16279。另8边界点5通过：.45/.55/.6/.7直行以及.5/-.05；.3/.4直行倒地，.5/+.05失败。不是连续命令域通过，也无动态跨门控边界证书。静止、正转、全目标域、最高DR、1ms、最终弹簧、MuJoCo仍缺口；promoted=false。新增模块恢复支持与export边界采样已修改，实际ONNX与MuJoCo尚未验收该候选。下一优先补低速稳定/正转，禁止继续扩大局部选择来掩盖连续域缺口。

## 2026-10-06 v511：采样重分配40更新，15/56，retention失败拒绝晋升

从v509续训，512env/40更新10critic+30actor，保持reward/优化器配方/动力/残差权限，仅正转cohort .5→.65、静止和.5直行rehearsal权重1→3。真实训练347.6秒，14命令coverage和冻结审计通过，只有新physics六张量变化。最终SHA162f897892c2c2a044032e3e58bb408ed5f55c560b17b031fc9edf1915793360。

同56点每点至少4回合：15/56，新增.5直行通过（survival/stable1，速度误差.1626）；丢失1m/s下-.2/-.1（yaw误差.07824/.04926、stable0/.431），静止仍失败；无相对v509的存活进一步退化。retention=false，promoted=false，保留v509，不把采样调整称成功。不沿此配置盲目加更新；下一需隔离失败域更新对已掌握负转的干扰，静止单独处理。完整command/contract/checkpoint/hash/readback/runtime/evaluation/freeze/retention/failed_conditions/final_report齐全，进程结束。DR、最终弹簧和MuJoCo仍未完成。

## 2026-10-06 v510：正向小转存在可控响应，统一大偏置会损害低速稳定

固定v509完整actor和同一D.05/阶段弹簧1.0775/DR0/2ms，六命令(.5/1/1.5×±.3)分别扫描轮槽2/5反向偏置0/±.1/±.2，30条件各至少2回合。1m/s+.3，零偏置yaw误差.299；bias=-.2时实际yaw约+.358、误差.06115、速度误差.1573、survival1。1.5m/s同偏置实际yaw约+.397、survival1；.5m/s同偏置两方向均倒地。因此接口存在正向yaw响应，不支持“正向动力学完全不可控”；也不能把固定偏置部署或视为策略学会。30条件正式门槛均未通过。完整evaluation/final_report保留于v510，promoted=false，无训练更新。

下一训练应针对速度/状态相关的正转探索缺口，不再重复小±.02扫描或盲目只增reward；强偏置仅为动作响应证据。v509静止survival0、.5直行survival.5是相对旧存活状态的退化，即使原已通过11点保留，也必须单独处理；不能用pass-count隐藏失败条件的存活退化。

## 2026-10-06 v509最终：100更新完成，56点由11提升至16，旧通过点全部保留

GPU0共享授权已落实为512env、100更新（10critic+90actor）训练，实际约15分钟。最终model_6985已保存artifacts/continuation_v509/final_checkpoint.pt，SHA256 5dcece5322d6240df3bc5895f5071e0a2cf74ec90c575eda4f7fe09a75c48a33。新增small-turn dense yaw shaping与扩大探索/更新属于多变量配方，不能单独归因于reward。冻结审计确认只有新physics六张量及显式配置std变化，实际动力读回通过。

最终相同56点网格、2副本每点至少4完整回合：16/56；相对v504原11点全部保留。新增(速度,yaw)：(1,-.2)、(1,-.1)、(1.35,0)、(1.5,-.1)、(1.5,0)，五点survival/stable均1。1.5直行速度误差.0565，1.5/-.1速度误差.00749、yaw误差.02346。正向小yaw仍失败，静止命令出现倒地，未完成全域。中途model6930保留；最终较中途新增能力，不应盲回退或重复短训。

D=.05主动电磁限幅内、b=.002/C=.001机械项限幅外、阶段弹簧1.0775Nm、DR0、2/2/8ms、5.5峰值/3持续/scale5.5；不是最终弹簧10.775、不是D随机范围/最高DR认证。完整报告、failed_conditions、retention、runtime、freeze与命令在v509目录，promoted=false。训练/评估进程均结束。下一优先保留v509增益，针对正负小转学习不对称和静止失败设计接续；正式32回合、多seed、1ms与MuJoCo仍未完成。

## 2026-10-06 v509运行中：按用户要求转入reward shaping与高吞吐适应

用户指出两小时无完整适应并授权充分使用GPU0。取消未启动v508扫描（final_report注明cancelled_before_launch），不重复25轮中仅15actor的短训。新增默认关闭的small-turn pseudo-Huber yaw cost，weight2只作用|vx|<=1.5且0<|yaw|<=.4501，误差反向与大尾部仍有连续方向；不删除speed跟踪奖励、不加任务外环，大转弯奖励/修正保持。15项reward/physics/contract测试通过。

v509源v504 SHA f1763c2c9c06512466c70148fb13d3038badebfee2f7178ae43b351c1f5130fa，512env、100updates（10critic+90actor）、64steps、4epochs、actorLR3e-4、wheelstd.08，属于多变量训练配方，不作单变量因果结论。D.05/b.002/C.001/阶段弹簧1.0775/DR0、5.5峰值/3持续/scale5.5、2/2/8ms保持，仅新低速physics6张量可更新，旧模块冻结。首启动被历史更新幅度guard挡在仿真前，重试显式allow_aggressive_highspeed；旧日志保留，真实运行命令command_retry.json/status_retry.json，PID见状态。14训练命令coverage已通过，实际512/新reward2读回确认；预计14分钟，40分钟墙钟上限，中途checkpoint行为无改善/退化停止剩余预算。未有最终报告不能称训练完成或行为晋升。

## 2026-10-06 v507：补回小yaw腿目标裁剪量不能恢复转向

v502记录1m/s±.3左active原始action约1.074–1.091、执行裁到1，而±1大转弯允许执行1.044–1.055。为隔离该差异，v507同v501策略/2ms/D.05/阶段弹簧1.0775/DR0，仅左active目标0/±.02rad偏移，±.3两命令各2副本。六条件全存活，yaw误差仍.2980–.3017，未恢复转向；速度误差随偏移变化.069–.110。轮最小载荷占比.459–.488、纵向滑移.00066–.00249m/s，不能声称轮接触丢失或明显纵滑是本条件原因。此为post-policy目标干预，不代表策略训练结果，也不排除其他腿/接触作用。

不据此扩大部署腿限幅，停止重复该偏移试验。v507/evaluation与final_report保存完整接触诊断，promoted=false。下一有效诊断需核对更充分的差分轮力矩响应，不能只因小偏置无效就判定没有可控性或奖励问题。全域能力仍11/56快筛、未完成迁移。

## 2026-10-06 v505/v506：奖励/优化器正常接通，小幅差分力矩无足够yaw响应

v505 CPU/reset权限审计：小yaw新分支gate1，1m/s轮动作差分宽松可达约[-.576,.787]；实际v504 optimizer新physics lr1e-4、6参数各60Adam步，critic100步；日志reward上下clip比例均为0（日志精度范围）。实际配置yaw_tracking_start_level0，基础yaw奖惩与转弯惩罚启用，不支持“完全无yaw奖励/优化器未更新/全被裁剪”的说法。只是初态权限及配置日志，不证明有效探索。

v506固定v501，.5/1/1.5m/s×±.3，每命令0/±.02轮槽2与5反向偏置，2副本、每条件2完整回合、2ms/D.05/弹簧1.0775/DR0；18条件全存活，但实际平均yaw仅约[-.008,.001]，远未达目标。执行动作读回随偏置变化，电机未饱和，不是偏置被接口忽略；也不能据此宣称策略完全抵消。1m/s+.3下active-leg槽3均值1，提示需查腿姿态/接触约束，非因果证明。无训练、不晋升、不把外加偏置用于部署。v505/v506 final_report齐全，下一联合核对leg/contact/wheel响应，不再只调yaw reward或盲增LR。

## 2026-10-06 v504：低速全域分支接通，25更新无新增通过，拒绝晋升

新增PhysicsAdaptation可选update_domain，只约束新分支，旧两阶段physics保持冻结且总预算不变；低速|vx|<=1.5、|yaw|<=.45全激活，至1.7/.6平滑关闭。现checkpoint已含冻结源，显式支持至两层冻结迁移并拒绝更深层；strict重建/映射冻结和165组完整actor零初始化输出逐值一致，21项相关测试通过。target-gate preflight与训练审计已使用实际新分支域，避免把旧physics门控误当可训练覆盖。

v504从v501派生零增量源SHA333c5d79808390fd2a4c5fa4b398e72d32ff0a7e6abcf28eba1a941322b9f9bb，10critic+15physics更新，最终SHAf1763c2c9c06512466c70148fb13d3038badebfee2f7178ae43b351c1f5130fa。14实际命令含站立/.5直行/1及1.5直行双向小yaw，coverage通过。只有新physics6张量变化，旧actor/startup/旧physics与buffer冻结；D.05/b.002/C.001/阶段弹簧1.0775/DR0/2ms实际读回通过。

同v502完整56点各>=4回合仍11/56，无新增通过；部分速度误差下降约.003–.004m/s，小yaw误差基本不变。大转弯保护域指标逐值保持，所有原11通过保持。冻结和覆盖不是行为进展，promoted=false，不追加同配置轮数；保留v501作为当前基线，v504仅失败候选。final_report/retention/runtime/freeze/failed_conditions和完整evaluation均齐全。下一核对低速域有效动作余量与学习信号，而非继续只改课程标签；正式全域/DR/1ms/MuJoCo未完成。

## 2026-10-06 v503：小yaw域命令确实传入，但现有修正训练域不覆盖

CPU同实际reset物理状态、完整v501 actor，改变vx/yaw观测命令槽并计算轮差分对yaw导数。1m/s从yaw-.45至+.45，left-minus-right action仅变化约.009669；各点局部导数约.019，不是命令被完全丢弃。PositionAuthority在|yaw|<=.45门控为0，expert3亦不覆盖此域，startup模块仅+1.4附近生效。因此继续expert3-only不能解决v502全域筛查中小角度不转的问题。单初态敏感性不能证明全部闭环行为或旧外环依赖因果；旧反馈参数仅作待核查差异。

下一适应范围必须包括站立、.5–1.5m/s直行、双向小yaw，并保留当前大转弯通过点；要求真实采样coverage与可训练门控同时非零，不能只更改课程命令后仍训练+1.4专家。v503/command_sensitivity.py及final_report记录证据，本轮无GPU训练、不晋升，D=.05全域/DR/MuJoCo仍未完成。

## 2026-10-06 v501/v502：+1.4修复有进展，全旧命令域仍大幅缺失

v501从v494冻结全旧图，新增startup residual（预算legs.4/hips.15/wheels.6 action，显式扩大权限仅作用vx约1、yaw约+1.4、估计速度低于.75的启动域），使用v496已通过.25s进入采集1200相关观测动作，CPU拟合2000步；无部署外环/计时器，命令重标为直接+1.4。旧张量逐位保持、重建输出及预算检查通过，16项相关测试通过。候选SHA426fe8d628d39df7985903820e153205358b501d9a2012b6dde58bfed33ff8fe直接九点8/9：+1.4四回合存活1、yaw.019476/stable1，+.8仍失败；非正式认证。

用户纠正不能仅盯+1.4。v502固定同候选、D.05/b.002/C.001/阶段弹簧1.0775/DR0/2ms、任务外环关闭，补测56固定点，每点至少4完整回合：仅11/56。历史40点2/40通过、当前八点7/8、补充8覆盖点2/8。1m/s小yaw几乎不转；1.5欠速/小转失败；2–2.9旧技能严重失配；站立有约.65m/s误差。全命令域远未适应，不能将权重冻结或九点通过当全域retention。v181历史报告D=.1并记录非零yaw/speed/pitch反馈，当前E2E外环关闭，须核查旧成功的控制依赖，不能全部归因D退化。

下一优先全域低速/小yaw迁移与历史接口差异，不继续仅修+1.4；正式32、多seed、1ms、DR、连续/随机/起停反向及MuJoCo验收均未完成。v501/v502最终报告齐全，未晋升。docs/rmuc_skill_status.md顶部已更新真实当前矩阵。

## 2026-10-06 v500：现有局部轮预算无法复现成功进入的首步动作

CPU同状态容量检查，源v494、v493真实reset观测，仅将命令yaw改为成功.25s ramp首步0以得到teacher首动作。teacher右/左轮action=-.064187/.023596；保持直接+1.4观测、允许整个local correction任取共享预算极值，右轮可达[-.539074,-.419074]，左轮[.219035,.339035]。teacher在可达域之外，最小差.354887/.195439 action；hip预算0也不能复现teacher约.0573/.0525动作差。两个active-leg槽teacher在宽松可达域内。

结论仅为“现expert3无法复现这条已验证成功进入方式的首步”，不是证明有限预算内不存在其他稳定解。后续帧teacher为反事实状态，不得宣称成功轨迹。这个容量约束足以否决继续以原±.06轮预算做启动模仿的路线；下一若做启动策略适应，必须重新定义有界状态作用域/保留原actor并做retention，而非继续盲堆expert3 PPO。没有扩大生产预算或修改checkpoint、无GPU仿真。v500/capacity_audit.py及final_report.json留证，完整目标未完成。

## 2026-10-06 v499追加：TF32解释重放差异，参数梯度核验完成

train.py显式启用CUDA matmul TF32。使用相同源checkpoint及快照16384输入、原128batch在GPU0纯推理重放：TF32开时与记录均值逐值相等，最大误差0；关时最大.000955164，单样本推理也有批量路径差异。故前次CPU2e-5断言失败可由数值模式重现，不支持权重损坏/观测错位结论。没有运行新仿真或修改权重。

按TF32/128batch重算expert3一阶PPO surrogate参数梯度，各阶段重放误差0：early/middle余弦.51869，early/late为-.37891，middle/late为-.56796；样本加权合成方向与early仍.62624、middle .87913、late -.16514。说明单批中不同episode阶段目标存在冲突，但不能称早期信号被完全抵消，也不是实际Adam有限步或行为因果证明。相关状态单批样本，未据此自动改LR/reward或继续同配置PPO。precision_replay.json与parameter_gradient_training_precision.json留证，v499 final_report已更新为verified replay，旧失败日志保留；策略行为未改善/晋升。

## 2026-10-06 v499：局部expert实际优势快照已取得，精确梯度重放尚未通过

从v494源执行3批128步capture-only，actor/critic/optimizer不更新，第三批保存ppo_gradient_snapshot.pt；最终checkpoint与源actor/critic逐tensor完全一致。GPU采集约45秒，D.05/弹簧1.0775/DR0/2ms保持，实际16384样本，+1.4按episode age分0–.25s1405、.25–.75s2609、.75–1.5s1033样本。Gaussian score动作方向随阶段有变化，但相关单批数据不能断言有效物理方向。

CPU参数梯度重放的均值对照未达2e-5门槛：最大.00095528 action，未确认是否为GPU数值精度/其他路径差异；故不发布阶段梯度冲突的确认结论，也不据此改reward/LR。保留失败断言和日志，final_report明确parameter_gradient_verified=false。capture-only真实接线与冻结已验证，下一可直接用已存快照查重放差异，不再采重复仿真。未进行适应训练/晋升，全部行为缺口保持。

## 2026-10-06 v498：真实失败观测的CPU动作Jacobian审计

无新GPU实验。读取v493失败轨迹16个记录状态，对v494源与v497候选分别计算actor输出slots0/2/3/5对expert3全部参数的梯度。全部采样expert3 gate=1、四槽Jacobian范数非零；因此这些状态没有支持“门控关闭/完全断梯度”的解释，不能推广到所有rollout。v497同输入相对v494最大动作差[.00482431,0,.00057457,.00436050,0,.00055206]，确有小幅改动，但方向是否提升回报尚无证据。此审计不是PPO优势或因果证明。

发现现有RMUC_CAPTURE_PPO_ROLLOUT仅允许physics训练，局部expert训练未保存优势快照。已将显式可选捕获扩展至authority/local，默认仍关闭；capture-only路径保持跳过优化，不修改通常训练。15项既有authority/training-contract测试通过，尚未原生验证local capture。下一次必要诊断先采实际启动状态、episode age、actions/means/std/advantages，检查学习信号后才改学习率/奖励或增加训练。v498/audit.py、action_gradient_audit.json和final_report.json保存范围与结果，promoted=false。

## 2026-10-06 v496/v497：0.25秒进入通过，混合短训仍未修复突变启动

v496源v494，仅取消等待并将yaw ramp缩至.25s，+1.4四回合全存活、linear=.020685/yaw=.031472/stable1，2ms/D.05/阶段弹簧1.0775/DR0不变。停止进一步扫描时长。新增训练command_turn_entry_abrupt_fraction（默认0），固定环境分组让一半active目标直接进入、一半.25s进入，rehearsal/continuous不变；26项命令/采样测试通过。

v497从v494完成10critic warmup+15expert3更新，SHA8895e92fc91c1b5e5062ff88f7bd8ffd83a46a348ea26c9341c979b012c61145，实际动力与冻结审计通过。直接命令九点仍7/9，+.8跟踪失败、+1.4存活0%，七保护点通过；拒绝晋升，不继续延长混合进入局部PPO。证据final_report/failed_conditions/retention/runtime/freeze/evaluation齐全。保留v482基线及v494进入课程中间点。接下来必须检查实际启动状态下优势与动作梯度是否驱动有效修正，不能再仅改进入时长或盲加轮数。完整目标未完成。

## 2026-10-06 v495：进入时长缩半仍通过，跳过该档重复训练

v494 checkpoint保持不变，仅诊断进入时长从hold1s+ramp1s缩为hold.5s+ramp.5s；physics2ms、腿2ms、policy8ms、D.05/b.002/C.001/阶段弹簧1.0775/DR0。+1.4四副本4/4存活，linear=.0170529、yaw=.0394657、stable1，通过该单seed短诊断。默认不把它作为正式技能认证；v494直接进入仍失败、+.8仍失败。evaluator新增显式hold/ramp CLI与0..4s有限值校验，并在报告记录实际时长。下一步继续缩短已通过进入档，不在该档浪费训练更新；到达失败边界再决定必要适应。v495 final_report齐全，promoted=false。此结果没有与源v482同.5档比较，不能据此声称训练带来因果改善。

## 2026-10-06 v494：进入课程短训完成，仍非突变启动修复

新增训练command_turn_entry_hold_s/ramp_s，默认0；只作用active非rehearsal/非continuous命令，手动play不变；复用_acquisition_command_age使生成观测与下一physics/reward命令一致，有限范围检查在构造前执行。25项命令/采样/时序测试通过。v494源v482，2ms/D.05/b.002/C.001/阶段弹簧1.0775/DR0，hold1s+ramp1s，固定方向分组，10critic warmup+15expert3更新；实际readback与freeze通过，仅expert3的20张量改变。final SHA d087f0bbb032d625399e339e41a9ef69233f855234b8e2124cbe8c5e1df78742。

无斜坡九点仍7/9，七保护点通过，+.8跟踪失败、+1.4存活0。渐进入单点4/4存活、linear=.0148553、yaw=.0420536、stable=.778428，通过该诊断；相对源v492速度改善但yaw余量缩小，不能称总体改善。保留为进入课程中间候选，promoted=false；源v482也保留。下一步先测更短进入边界，不重复同hold/ramp继续堆更新，不按更新数自动晋级。final_report/retention/runtime/freeze、两个完整evaluation及命令齐全，正式32回合/1ms/全域/DR/MuJoCo行为均未完成。

## 2026-10-06 v493：同布局成对确认+1.4进入瞬态缺口

v493从v492命令仅去掉--diagnostic-turn-entry，源v482不改、seed4567、4副本、2ms/D=.05/弹簧1.0775/DR0保持。直接进入+1.4为0/4存活，无稳态样本；v492渐进入为4/4存活、速度误差.0578415、yaw.0162856、stable1。已逐项核对环境origins、root位置/四元数、checkpoint SHA、runtime_contract、实际physics readback、运行源码哈希完全一致，排除先前九点/单点布局差异。结论限于该单seed成对实验：进入方式足以改变结果，持续+1.4并非全失；尚未证明多seed或1ms。下一适应应利用这个可存活起点逐渐缩短进入过程并回到突变命令验收，禁止仅发布依赖固定进入斜坡的诊断方案。v493/final_report.json包含成对检查和两行完整指标，promoted=false，无额外训练。

## 2026-10-06 v492：+1.4持续能力与复位启动缺口得到区分

同v482、D=.05/b=.002/C=.001/阶段弹簧1.0775/DR0/physics2ms、legPD2ms、policy8ms，仅将命令进入改为1秒1m/s直行，随后1秒yaw平滑至+1.4。4/4完整回合全部存活，linear_error=.0578415、yaw_error=.0162856、stable=1，零termination。v491直接进入+1.4仅1/29存活；布局不同，仍需同布局成对复核，不能宣称单因已完全证明。证据支持优先处理进入瞬态而非继续将持续转弯能力一概判失。evaluator新增--diagnostic-turn-entry，改变命令在生成观测之前，不新增任务反馈、不直接改动作；诊断report总体passed=false。下一修复从已可存活的进入课程缩短过渡，最终仍需无辅助突变/动态切换/32回合及1ms验收，不把斜坡诊断充作完整技能。证据artifacts/continuation_v492/{evaluation_command,probe,final_report}.json。

## 2026-10-06 v488–v491：固定采样反证、真实2ms对照及部署D=.052

v488固定方向分组使+1.4 rollout占比恢复30.2%，25更新仅expert3变化，仍7/9，+.8 yaw失败、+1.4存活0；不晋升、不追加同配置PPO。v489用v482在同契约MuJoCo1m/s±1各12秒，正向速度/yaw误差.37177/.27869，负向.20587/.17521，均未通过；3000真实输入actor/JIT/ONNX/MuJoCo raw parity最大3.5763e-7，证明推理一致而非行为认证。

v490请求2ms但环境configure_wheel_torque_contract强制覆盖到1ms，已标invalid_timestep_comparison，不作为步长证据。修复为显式separated_wheel_physics_dt_s（默认1ms，可选诊断2ms），evaluator在构造后用实际sim.get_physics_dt及step_dt断言。v491实际读回physics2ms/policy8ms/legPD2ms，源仍v482/activeD.05/机械b.002/C.001/弹簧1.0775/DR0。九点4副本仍7/9，七保护点通过但数值非逐值相同；+.8 yaw=.0788216/stable0，+1.4仅1/29存活，稳态幸存样本stable1不能掩盖早倒。耗时89.37秒对先前实际1ms131.89秒仅为共享GPU参考，非严格性能基准。2ms可作为显式训练近似，最终候选必须回1ms验收，不声称已修复D迁移。v491 final_report/retention_audit/runtime_verified齐全。

按用户最新要求，部署C阶段目标RMUC_STAGE_D_COMP_NM_S_RAD=.052，来自训练activeD.05+被动b.002。它是假设实机b近零时的未饱和近似；若已知b_real应扣除，电磁限幅内补偿不能完全等同训练限幅外被动项，仍需制动/饱和验收。左右校准仍显式传入，不静默覆盖实机参数，不给部署端追加机械损耗项。54项wheel-passive/profile测试及C控制器边界/饱和/周期测试通过；docs/rmuc_deployment_control_contract.md和deploy README同步。D=.05全技能适应、全域/DR、正式MuJoCo和部署仍未完成。

## 2026-10-06 v487最终失败：局部PPO未改善目标，七点retention保留

v487实际完成10critic warmup+15局部更新，final SHA f64659b1db05e9c9b0c31022283e270a79b2e0f9f0a336fdc594d4ccd3a18a0a，仅expert3的20张量改变，freeze/runtime audit通过。当前同契约九点4副本仍7/9：+.8 yaw=.0707509/stable0；+1.4 survival0；七个原通过点survival/linear/yaw/stable与v482逐值相同。拒绝晋升，不追加同配置训练，保留v482。训练中+1.4存活差导致重置后重新抽方向，后期目标样本约8%，负向约57%；这是实际采样效率缺陷，不证明它是失稳唯一根因。后续局部适应若需要训练，先解决样本占比/启动闭环，不从v487继续加轮数。验收首次误指全域grid被128env上限启动检查拦截，未进入仿真；实际使用evaluation_command_retry.json九点grid，原失败日志保留。final_report/retention_audit/runtime_verified/freeze_audit/failed_conditions均已保存。

## 2026-10-06 v487：冻结physics映射的+1.4局部适应

历史v430成功训练采用10次critic warmup及expert3隔离更新。v487据此从v482（SHA099e9a4b9ceed9eaf027c6ee2a336048ca4acabcbc02d10ed641f3c3b53dc322）启动25更新预算，仅expert3可训；D=.05/弹簧1.0775/DR0不变，不加诊断腿偏移，冻结physics残差及其他actor。启动1因argparse nargs吞Hydra参数退出，启动2因只采正向与双向coverage冲突在第3次critic warmup终止，均无actor适应更新；原日志保留。实际启动使用command_retry2.json，双向采样，仍只更新expert3；日志2026-10-06_15-01-37_positive14_local_criticwarm10_d005_v487_balanced。前3次rollout覆盖检查及冻结检查通过。train.py将实际执行器readback扩展到全部position_authority训练，15项training-contract/authority测试通过；实际读回peak5.5、actuator damping0、activeD.05、b.002、C.001、1/2/8ms、弹簧1.0775、外环false。接续必须读status_retry2及最终报告，训练进行中不代表行为通过，禁止自动追加预算。

## 2026-10-06 v486结项：腿目标干预恢复存活，未恢复跟踪

复核v486完整probe：v482源、activeD=.05、阶段弹簧1.0775Nm、DR0，+1.4仅诊断性active-link目标+.0359167rad，4/4完整回合存活；linear_error=.1197845、yaw_error=.0751959、stable_fraction=0、bank_error=14.9564deg、pitch峰值19.4569deg。yaw/stable失败，promoted=false；不是策略自身能力，不得导出该干预作为部署补偿。原始结果及final_report保存在artifacts/continuation_v486。

本轮查PID无本项目训练/评估在跑，不启动重复训练。已知失败有两类：D变化引起轮力矩映射偏移，v482匹配已恢复7/9；+1.4早期pitch发散在当前高D也存在，不能把所有失败归给D。下一训练必须从实际成功闭环出发，只改一个迁移量；若D单因素退化得到成对证据，再细分D课程，失败回滚到上一通过档。当前不得无依据承诺D退火必解，也不从v481/v485继续加轮数。完整域、32回合、多seed、DR及MuJoCo行为仍未通过。

## 2026-10-06 v484/v485：腿目标诊断与策略输出不可直接等同

v484 +1.4短诊断400步：零偏移5次tilt、反向-.035917rad6次tilt，按2.155Nm/PD60计算的+.035917rad active-link目标偏移无tilt、pitch最大19.80度，但0完整回合，不能称存活或跟踪通过。说明对腿目标平衡位置敏感，不证明最终弹簧律正确或可直接部署。

v485将该偏移拟合到既有+1.4局部expert3，仅改四个子专家输出层8张量的active-link行，其他wheel/hip同输入输出不变，11组保护命令输出逐位相同，预算/门控不增；候选SHAe8cd3476641e2dc994f2e44f7c7d28f9099845444d44929b33284c29c555ff37仍+1.4 4/4倾倒，拒绝候选，不延长训练。真实失败前大部分时刻确实产生约.161–.164的目标action偏移，所以不是head没接上。源actor对previous-action输入移除假设偏移的输出响应最大.1142 action；这是同输入敏感性证据，不是单因证明。诊断post-policy腿目标偏移不会改previous-action观测，而策略输出修复会反馈到下一观测，两条闭环不同。此外末期yaw error clip使重构命令偏离、局部门控关闭，不能把指令门控恒为1当事实。保留v482 7/9诊断候选；下一次修复必须同时核对实际历史动作输入和局部激活，禁止将短时post-policy偏移直接转成部署补偿或盲延PPO。v484/v485最终报告和v485/previous_action_feedback_audit.json均保留。

## 2026-10-06 v483失效轨迹：高D也不是有效+1.4起点

固定+1.4/2副本400步，v482低D匹配候选首例139步倾倒，当前正确轴/1ms物理下冻结v430源恢复activeD=.080625仍174步倾倒。两者前71步pitch走势近似：21步4.42/4.28度、41步11.58/11.25度、71步20.55/19.97度。低D候选同输入actor回放误差3.3e-7，腿输出与源逐位相同，早期轮力矩匹配误差多数<.06Nm，倾倒后段扩大至1.223Nm。证据表明+1.4早期pitch失稳在高D参考也存在，不能把单纯D退火当已有可靠起点；仍不作单因归责。v483/{failure_analysis,final_report}.json留存。v484只做0与±(2*1.0775/60)=±.035917rad active-link目标偏移对照，验证旧新弹簧数值差对应PD姿态敏感性；固定D=.05、弹簧1.0775、b=.002/C=.001、5.5/3、1/2/8ms、DR0，6env400步，无训练/不部署该干预。

## 2026-10-06 v482结项7/9，禁止无效全局长训

v482 CPU轮力矩匹配候选在当前activeD=.05、阶段弹簧1.0775、b=.002/C=.001下九点4副本完成，7/9 raw通过：直行及-.8、±1、±1.2、-1.4；+.8 yaw误差.07075且stable0，+1.4存活0。±1的yaw误差.01376/.01842，±1.2为.04145/.03322，-1.4为.03349。比v480源1/9、v481直接PPO2/9显著恢复，但不晋升、不宣称D范围或全命令域、最终弹簧、MuJoCo/实机成功。候选SHA099e9a4b9ceed9eaf027c6ee2a336048ca4acabcbc02d10ed641f3c3b53dc322；final_report/retention_audit/quick_eval齐全，当前无本项目GPU任务。

用户要求停止无效训练，审计发现此前“五点存活→直接D=.05训练”的判断遗漏+1.4故障边界，应撤销此推断。v430成功世界与v476当前高D并非仅D数值相同：旧2ms→当前1ms物理步，旧IdealPD阻尼→active_motor阻尼+外置b/C，旧弹簧wxyz解释→正确xyzw，并有DR/seed差异。`v430_v476_runtime_diff.json`和`spring_convention_audit.json`保存对照；后者identity姿态下两种约定每腿相差2.155Nm，仅为纯函数证据，不宣称单独根因。此四元数问题在旧第3484行附近已有记录，不是本轮新发现。高D恢复不能据此视为历史闭环复现，不能回退错误坐标当最终物理。下一步只针对+.8与+1.4定位轨迹和隔离参数；必要D退火每个小档保留八点，禁止同配置共享残差全局长训。

## 2026-10-06 v481拒绝晋升，v482轮力矩匹配初始化

v481完成25更新，final SHAadc519374fcd5ca62a12cba37f6f677b6a20ef4bb70d14a1810a78fdbb6f6289，冻结与命令覆盖通过但直行+历史八点仅2/9通过（直行、-.8）；+.8存活.1111、+1/+1.2/+1.4存活0，负向-1/-1.2/-1.4虽存活但yaw失败。v480源补测±1.2/±1.4说明+1.4原本在D=.05就倾倒，其余三个短程存活；v481进一步破坏正向存活，不续此候选，不凭负向EMA改善晋升。

v482改为CPU力矩匹配warm-start：用25D内已有轮速，将旧activeD=.080625世界的电磁力矩目标拟合入新六张量physics残差，在实际activeD=.05保持旧actor/腿输出逐位不变、预算不扩、无部署外环。16384合成样本与3000真实MuJoCo观测混合，1600CPU优化步；真实输入平均/最大力矩差.00417/.04807Nm，heldout平均.01820但最大.95781Nm，宽域不完全等价，必须行为验证。候选SHA099e9a4b9ceed9eaf027c6ee2a336048ca4acabcbc02d10ed641f3c3b53dc322，D=.05/阶段弹簧1.0775/b=.002/C=.001/DR0/5.5与3热参考/1-2-8ms，正在GPU0做九点4副本2400步短筛。所有源码/命令/报告在artifacts/continuation_v482；未晋升，不是D范围、最终弹簧或全域成功。

## 2026-10-06 v480同契约MuJoCo基线

训练v481期间CPU并行完成冻结源在activeD=.05/弹簧1.0775Nm、1/2/8ms、5.5/3、PD60/2下MuJoCo双向1m/s±1rad/s各12秒基线：均跑满但不通过，正向后半速度/yaw误差.49671/.33384，负向.63714/.23001；无轮电磁饱和。3000个真实MuJoCo观测在恢复Isaac actor CPU/JIT/ONNX/记录raw action之间最大差2.384e-7，说明推理链一致，不证明物理行为一致。命令内嵌于报告invocation，MJCF与checkpoint/JIT/ONNX及contract hash均保留；证据v480/mujoco_positive1.json、mujoco_negative1.json、same_input_parity.json及same_input_outputs.npz。不因能跑12秒称sim2sim成功。

## 2026-10-06 v481已启动D=.05短适应

GPU0 trainer PID1674190，源v476/frozen_source_migration.pt SHA423570223dc22e478d1d8fd07e2e82807371a39d3c46f5bf58ce3815da486b35；保留v430阶段弹簧1.0775Nm/腿，唯一物理迁移activeD=.080625→.05，b=.002/C=.001、DR0、5.5/3、1/2/8ms、腿PD60/2±40与无外环保持。25更新含5critic warmup、128env×128steps、1200s上限；新physics六张量独训，旧图冻结、残差预算不扩。结束自动直行+历史八点4副本快筛，无自动晋升；命令、契约、状态读artifacts/continuation_v481。全域目标必须覆盖0至1.5m/s中间速度、小yaw、起停/连续切换及1m±1–2.5、1.5m±.8–1.8；已备69点domain_grid.json及命令，但未运行，不能把本轮九点当全域通过。D迁移后仍需向10.775Nm/腿迁移，MuJoCo同契约与正式多seed门槛未完成。

## 2026-10-06 历史八点门槛补正

v430历史正式八点是1m/s双向yaw .8/1/1.2/1.4，每点32完整回合，在阶段弹簧1.0775Nm/腿、高D=.080625及旧局部DR=.05下tracking8/8；v431/v433升至旧局部DR=.1时8/8存活但tracking仅4/8，四个正向点失败。不能把旧DR=.1当已通过，也不能将其泛化为新activeD=.05或联合DR。v432同checkpoint MuJoCo八点yaw均失败。v481快速retention网格现含直行+历史八点；其4副本短筛只用于续训决策，晋升前仍要每点至少32回合并补更宽命令域。

## 2026-10-06 v480主动D=.05固定源结论

冻结v430源SHA423570223dc22e478d1d8fd07e2e82807371a39d3c46f5bf58ce3815da486b35、1.0775Nm/腿阶段弹簧、activeD=.05、b=.002/C=.001、DR0、5.5/3、1/2/8ms五点4副本2400步快筛已完成：五点均短程存活，但仅1m/s直行通过；±.8 yaw误差正/负.30581/.09831，±1正/负.27403/.13340，四个转弯stable全0。与D=.080625的v476相比，降D明显损失双向跟踪，不能称D迁移完成。当前允许直接在D=.05做有界短训，因为未出现短程倾倒；每次必须检查双向retention，失败停配方。完整证据`artifacts/continuation_v480/source_at_d005.json`。

## 2026-10-06 五小时短期目标调整

用户确认暂保留v430的1.0775Nm/腿阶段等效弹簧律，优先主动D从.080625迁到.05；这是短期取舍，不是最终10.775Nm/腿实机契约已经满足。v479终值弹簧固定策略探针见约114步tilt后按新优先级中断，未生成完整JSON，状态为`interrupted_no_conclusion`，不可称结果。v480只将active D降至.05，固定原弹簧、b=.002/C=.001、DR0、5.5/3、1/2/8ms，五点短筛；若不保留双向低速技能则先插中间D，不盲训。最高DR、最终弹簧、MuJoCo正式行为及实机验证仍未完成。

## 2026-10-06 v477结论与v478同条件对照

v477完成50更新但五点快筛仅2/5，不能晋升：1m/s直行与+1.0通过，+.8稳定比例.0176未过；-.8/-1.0的yaw误差.1621/.1568、stable=0，虽然五点短程均存活。冻结审计通过只说明旧784张量未变，不说明旧技能保持；第4腿动作槽在五点均100%原始饱和。`artifacts/continuation_v477/final_report.json`为准，不续该checkpoint、不把训练reward上升当适应。

v478使用原冻结源SHA423570223dc22e478d1d8fd07e2e82807371a39d3c46f5bf58ce3815da486b35，在相同1.5Nm/腿、activeD=.080625、b=.002/C=.001、DR0、5.5/3、1/2/8ms、五点4副本2400步评估：直行、-.8、-1.0通过，+.8/+1.0未过，全部存活；负向yaw误差.01538/.02221、stable=1。与v477相比，负向退化由更新导致而非1.5Nm弹簧增量本身导致。当前仅为阶段物理快筛，非最终10.775Nm/D=.05或正式32回合认证；证据`artifacts/continuation_v478/source_at_spring15.json`。

## 2026-10-06 v476有效起点与v477小步迁移

v477首次启动在仿真前被高速保护误拦：按level>=9将level49的1m/s扩展转弯误判高速。修复为同时检查实际stage speed达到高速起始速度，保留真正高速保护，新增回归测试后7项contract测试通过；attempt_01日志保留。重启PID1600428，已进入PPO；runtime读回activeD=.080625、spring1.5、5.5峰值/1/2/8ms/外环关闭全部通过。非训练成功结论。

v476已完成：active_motor D=.080625/阶段弹簧1.0775下五点均存活，直行速度误差.13696；±.8 yaw误差.08078/.01376，±1.0为.13342/.01794，负向stable1、正向stable0。可学习转向仍存在，但非最终物理、不晋升。该结果支持从旧可转起点迁移，不再从零yaw失败分支重复加噪声。

v477源为v430全冻结/新physics头零初始化，SHA423570223dc22e478d1d8fd07e2e82807371a39d3c46f5bf58ce3815da486b35，阶段弹簧唯一物理增量1.0775→1.5Nm/腿，D=.080625不变、DR0、5.5/3、1/2/8ms、PD60/2、b=.002/C=.001，外环关闭。50更新含5critic warmup/1200s，主动±1.0，30%rehearsal直行与±.8；只训练新六张量，原784张量保留，预算不扩。源yaw observation error clip=2保持，不同时改6。结束自动同五点快筛；命令/状态见artifacts/continuation_v477，当前不把阶段弹簧和高D称交付条件。最终仍须到10.775近似及主动D=.05范围、联合DR和全域验收。

## 2026-10-06 v475结论与v476迁移起点

v475原v430在当前activeD=.05/10.775Nm近似下7点完成：1m直行误差.03944通过；±.3yaw误差.29992/.29975；+1.4存活0、-1.4存活1但yaw误差.21246；1.5m±.8仍无有效yaw。恢复旧权重本身不足解决迁移失效，见v475/final_report.json。

v476复用v467已做的高D/小弹簧起点，只将反馈D由postclip改为active_motor_v1，原v430 SHA5960533edf88c8a08a564ab86ca19e679faa49b44074f6cb3ba9873cfa1ad502、D=.080625/b=.002/C=.001、弹簧阶段1.0775Nm（不是最终10.775Nm）、5.5/3、1/2/8ms、DR0，五点4副本2400步600s；不能把该起点当交付契约。若存活和双向yaw仍可学，再从此小步迁移，不复用无yaw源加噪声配方。

CPU准备v476/frozen_source_migration.pt：SHA423570223dc22e478d1d8fd07e2e82807371a39d3c46f5bf58ce3815da486b35，原784张量逐位保留，新physics头零初始化；2156组actor输出精确相同，预算[.15,0,.35,.15,0,.35]未扩大。仅准备，未训练。任何后续必须声明阶段物理与最终物理的差距。

## 2026-10-06 v475旧网络审计与低速域目标

目标按docs/rmuc_long_term_goal.md当前条目：1m/s双向1–2.5rad/s，1.5m/s双向.8–1.8rad/s，并覆盖小yaw、起停/中间速度/连续切换，不能以±.3替代全域。delivery_target_grid.json只是验收规划，未训练/认证，.25/.5/.75/1.25采样点是工程选点，不冒称用户指定上包络。

CPU实读v430与v474：8个历史local authority专家各20张量在中间版本已有变化，另physics7张量和std不同，故近期freeze通过不等于v430全网未变。旧position authority仅speed(.8,1.2)且|yaw|>.4501支持；local experts同速度域+中心±.15窄窗。1m±.3与1.5m时这些分支门控0，1m±1.4为1；并非整个actor零输出。不能直接扩大门控把未训练域当会，需隔离扩展且retention。

v475固定v430/model6835 SHA5960533edf88c8a08a564ab86ca19e679faa49b44074f6cb3ba9873cfa1ad502在当前activeD=.05/b=.002/C=.001、5.5/3、10.775Nm近似、1/2/8ms、DR0、无外环下跑7点，唯一干预旧actor。28env×1800步/600s；命令与network_audit/status在artifacts/continuation_v475。结果出来前不启动新的长训、不称迁移修复完成。

## 2026-10-06 v474结项：扩大独立轮噪声未学会慢转

50更新及12点快筛完成，promoted=false。冻结/覆盖通过，候选SHA760b9a412e10399785141e03787614c992bad4695f9d0addbc383814857f71c7。固定主动D=.05诊断12点均存活但仅1m/s直行raw通过（速度误差.06540、yaw误差.00641、stable1）；1m/s±.3速度误差.01692/.03122，yaw误差.29516/.30511、stable0。零命令虽存活但速度误差.67436，不能称站立掌握；±4yaw误差3.99877/3.99456。训练中随机动作产生的少量yaw没有转化为确定性策略，不能继续凭加std延长同配方。未运行正式32、多seed、D范围验收、联合DR或MuJoCo行为。证据artifacts/continuation_v474/final_report.json、quick_eval.json。

## 2026-10-06 v473结果与v474探索修复试验

v473冻结model7050轮差探针完成，10组均4/4存活。零bias双向yaw误差.29786/.30215；+.15 bias在+.3目标产生平均yaw约+.65986，-.15在-.3目标约-.89431；±.06响应弱且不对称。说明执行链可产生双向yaw，不能说轮差被完全吃掉，也不能仅由该探针认定具体摩擦阈值或保证加大std必然学会。所有stable=0、无晋升；证据artifacts/continuation_v473/probe.json和final_report.json。

v474从v472/interrupted_model7050（SHA19e6c1ab146042b733d7c47e019c42d75bf32671dc61be4ac853ecc2b299040a）接50更新，唯一训练参数变化wheel std .06→.18，原70%±.3目标/30%stand/.5/1 rehearsal保持，5critic warmup、1200s预算。主动D[.04,.06]、b=.002/C=.001、5.5/3、10.775Nm近似及1/2/8ms保持，physics六张量独训、旧图冻结/残差不扩，strength1仅D范围，其余DR关闭。结项修复std启动例外误报，慢转advantage审计已启用；结束自动12点快筛，不自动晋升。命令/status/contract读取artifacts/continuation_v474，启动不等于收敛。

## 2026-10-06 v473：慢转审计补漏与轮差因果诊断

补充结项审计bug：v472的源对比漏掉已声明action_std_override，源std与启动std不同会被误报frozen mutation。保存std实测[.03,.03,.06,.03,.03,.06]与命令一致，逐更新std未变化；freeze_audit.json现区分该启动例外和六个physics学习张量。run.py修复仅豁免“与显式启动std逐值匹配”的distribution.std_param，其余变化仍拒绝；不重启已停止训练。这同样属于审计错误，不能冒称yaw根因。

v472纠正计数：实际停止前完成149个rollout update，最后保存model7050对源iteration差145，二者不可混用；只有训练遥测，没有确定性行为认证，尤其不能称站立已经平衡。未晋升候选仍可用于明确标注的故障诊断，此处替代下文“禁止评估”的过宽措辞。文档上次patch失败是匹配文本漏了前缀，并无证据证明并行写入。

查明wheel_advantage_audit及delivered_command_counts只枚举历史±.8至±1.4，遗漏当前±.3，导致空梯度诊断/other分类。已补±.3并增加含非零实测yaw的命令重构与surrogate符号测试，8项测试通过。此为审计修复，不是已证实的训练失败根因。model7050实际optimizer physics LR=1e-4，未被base2e-6覆盖；外环关闭时动作直接经过限幅返回，没有强制轮同步。旧wheel_noise_hold为欠速common-mode，不能冒充yaw探索。

v473冻结model7050，源SHA19e6c1ab146042b733d7c47e019c42d75bf32671dc61be4ac853ecc2b299040a，1m/s±.3两命令，在2s后1s ramp叠加轮差0/±.06/±.15，40env×1800step/600s预算。固定主动D=.05、b=.002/C=.001、5.5peak/scale、3持续热参考、10.775Nm/腿近似及1/2/8ms，无外环、DR0；唯一干预为轮差，零偏置同批对照。完整命令/契约在artifacts/continuation_v473，结果未出前不预判物理响应、不启动同配方长训。该探针不做策略晋升。

## 2026-10-06 v472：主动D窄范围长适应，尚未验收

**已提前终止，未晋升。** v472实际145/300更新后，±.3命令分别已有833557/848363 rollout样本，冻结参数/缓冲始终无变化，主动D采样及5.5/1/2/8ms读回正确；没有数值发散。然而最后正/负yaw误差EMA仍`.2973/.3028rad/s`，aligned实际yaw=`-.0005`、wheel differential action=`.0010`，等同零yaw响应。平衡和正向存活不能抵消该失败；继续余下155更新无证据价值。保存最后checkpoint `artifacts/continuation_v472/interrupted_model_7050.pt`，SHA256`19e6c1ab146042b733d7c47e019c42d75bf32671dc61be4ac853ecc2b299040a`，仅诊断，禁止评估、导出或晋升。最终报告明确未运行grid/32回合/ONNX/MuJoCo/多seed/联合DR/实机。

v471已结项未晋升：25更新后12点快筛仍仅1点通过，1m/s直行速度误差.05563，±.3转向yaw误差约.300；五项MuJoCo回放正常退出不构成行为通过。physics residual确实非零且经过actor，冻结审计仅六个允许的network张量变化；不能将短训无改善直接归咎于未接入。

用户授权必要长训练。v472从v458（SHA256 7565b5d972c3b3d807078d1ae86b0e4ff65f8f1d19ee1b6b164ddb59f05aa642）warm-start，300更新含5critic warmup，上限5400s；70%目标1m/s±.3、30%站立/.5/1直行rehearsal，wheel exploration std=.06；只训练既有physics network，旧图冻结、残差预算不增加、课程不自动升级。这是训练配方实验，不是单因素因果实验。

active_motor_v1主动D名义.05、独立分轮采样[.04,.06]；机械b固定.002、库仑固定.001/eps=.1，非饱和线性总D为[.042,.062]。strength=1只展开该窄范围，其余质量/惯量/摩擦/传感器/延迟/电机响应DR关闭，绝非最高DR通过。5.5峰值和action scale、3持续热参考、1ms轮/physics、2ms腿PD60/2±40、8mspolicy保持；气弹簧10.775Nm/腿近似、径向及firmware spring关闭、无任务外环。

部署C核心六项D=.05边界测试通过（含两轮安装符号换算、反转限幅和连续性），33项相关Python测试通过。训练命令/契约/验收命令位于artifacts/continuation_v472，结束自动12点快筛，32回合命令仅准备；实际状态以status.json为准，未有最终报告不能称适应完成。电流哨兵不是机械b辨识，热电流/温度数值未经硬件验证，不能作为部署证书。

## 2026-10-06 v471：修复训练通道与低级rehearsal后续训

v469的两项配置问题现已加CPU前置检查：`required_training_commands`计入rehearsal实际起始level，避免把低级未启用的replay算可达；新增active target门控检查，选中的+1.4局部专家在±.3目标门控全零时会在gym.make前拒绝，单方向局部修复仍允许另一方向冻结。13项相关测试通过。

v471保持v470物理：源v458 SHA`7565b5d972c3b3d807078d1ae86b0e4ff65f8f1d19ee1b6b164ddb59f05aa642`、active_motor_v1/D_fb=.048、b=.002/C=.001、5.5峰值与scale/3持续热参考、10.775Nm/腿近似、1/2/8ms、PD60/2/±40、外环/径向/firmware spring关闭、DR0固定值。只训练已有physics network六个张量，预算沿源[.681818,0,.35,.681818,0,.35]不扩大，源图其余参数与buffer冻结。试验独立覆盖站立、.5/1直行和1m/s±.3；60%低级rehearsal从level0启用，保持1m直行样本；turn yaw reward eligibility显式从level0启用。该训练配方包含采样/可训练通道/奖励适用域修正，不将效果归因某一单独因素。

25更新（5critic warmup+20 actor/critic），128env×128steps，训练上限1200s，结束自动v470同12点快筛；不自动进入DR或正式验收。实际启动后wheel参数/时序/助力矩检查通过，前三个rollout五条命令均有样本（4776/8738/9568/16054/10016），修复覆盖缺失，不等于学会。状态及最终证据读`artifacts/continuation_v471/status.json`和`final_report.json`；训练中不能提前称成功。

## 2026-10-06 v470：部署主动阻尼候选，七小时交付收缩

**本轮已结项、未晋升。** GPU短筛约129s正常结束，12点仅1m/s直行raw通过（4/4存活、速度误差.040229、yaw误差.001859、stable1）；0速站立46/46 tilt；±.3慢转虽4/4存活但yaw误差约.300，1.5m/s±.8误差约.799/.804，±4自转误差约3.999/3.996。DR0、短样本不能证明D范围适应。MuJoCo三条12s正常结束，直行后半速度/yaw误差.06693/.03503，1m/s+.3为.13320/.34252，自转+4实测yaw仅.07089、误差3.92911；行为未通过。72组MuJoCo观测在恢复Isaac actor CPU回放/JIT/ONNX之间最大误差2.384e-7；此为相同输入推理一致性，不是物理轨迹等价。见`artifacts/continuation_v470/final_report.json`、`same_input_parity.json`。

额外定位到v469的训练通道错误：源v430的局部专家index3中心为+1.4、半宽.15，±.3目标激活严格为0；同一通道对1.5m/s和原地自转也为0。已有PhysicsAdaptation通道在这些目标激活为1，后续应使用可达通道，保留受限动作/冻结与retention，不能只加local expert更新数。`trainable_domain_audit.json`保存源checkpoint实际centers和六个命令计算值。v469已被coverage guard中止，无最终候选；本次不启动未经门槛的新长训，未运行正式32回合或升DR。

用户最新要求覆盖此前继续下探D的计划：机械被动b仍为[.001,.005]先验，在部署端加入主动阻尼，将未饱和时总有效D目标收敛为[.04,.06]，不再以逼近纯机械低D为当前课程目标。气弹簧停止详细建模投入，只保留最小迁移及必要扰动验证；这不证明气弹簧影响可忽略。主要参数±20%、观测/通信各1–3周期等联合DR要求没有取消。

代码核查发现旧separated模式把反馈D与被动损耗一起施加在电磁clip之后，而C部署核心此前没有反馈D。若部署真正通过电机增加阻尼，反馈力矩必须占用±5.5Nm电磁额度。新增显式`wheel_feedback_damping_mode=active_motor_v1`：`tau_em=clip(5.5*sign*clip(a,±1)-D_fb*omega,±5.5)`；`tau_net=tau_em-b*omega-C*tanh(omega/eps)`，净轴力矩不再clip。Isaac、MuJoCo consumer及1ms C核心采用同一顺序；旧`postclip_loss_v1`保留复现，不改写历史成绩。反馈使用关节输出轴rad/s，电流换算及安装符号仍须实机核对；无需精确辨识b也不能取消这些接口核验。饱和及延迟时不能再把系统等同常数D相加。

v470先用v458源SHA`7565b5d972c3b3d807078d1ae86b0e4ff65f8f1d19ee1b6b164ddb59f05aa642`做固定checkpoint短筛，唯一新干预为主动D_fb=.048；b=.002、C=.001、DR0，未饱和名义总D=.05。沿用10.775Nm/腿恒等效近似、径向0、firmware_spring关闭，绝不称该近似已与旧实机弹簧辨识一致。peak/scale5.5、continuous3仅热参考；轮1ms/腿PD2ms/policy8ms、PD60/2/±40、任务外环关闭。未来计划b[.001,.005]与D_fb[.039,.055]独立采样，使未饱和总D支持域[.04,.06]；此范围尚未投入训练或验收。86项相关Python测试及C核心编译测试通过，属于代码验证。

证据目录`artifacts/continuation_v470`，`command.json`为实际短筛；`formal_evaluation_command.json`仅准备每点32回合的后续命令，尚未执行。站立、.5/1/1.5直行、1m/s±.3、1.5m/s±.8、原地±3/±4分别评估。所有结果以该目录status/最终报告为准，不因D接近历史值就写成适应成功。旧八点retention、多seed、联合DR、MuJoCo行为及实机replay仍须完成。

七小时是预算而非收敛承诺：0–2h完成新轮律接线与低速存活候选；2–5h在已过门槛上加入联合DR并扩展≤1.5及±4，冻结旧图/限幅残差；5–7h保留正式回合、retention、同输入JIT/ONNX/controller和MuJoCo验收。若2h仍无稳定低速候选，停止扩技能，交付范围退到真正通过的平衡/低速慢转；不能占用末两小时盲训或宣布全鲁棒性。

补录v468/v469：v468缩放.1固件弹簧探针在1200步内未完成完整episode，±.4几乎没有yaw响应，不能把未倾倒等同存活验收；该可选路径默认关闭并暂停投入。v469因实际rollout缺少要求的低速rehearsal命令而由coverage guard中止，无最终checkpoint和技能结论，不能视作完成25更新。本轮不绕过该guard续跑原配方。

## 2026-10-05 v465-v467：reward尾段与旧气弹簧/D同源对照

用户更新优先级：气弹簧以只读旧控制器`/home/kellen/桌面/combat-dm-balance-main/application/chassis/chassis.c`的实时逻辑为主。该源码`GetSpringTorqueByLegLength`按`clip((-83.121*L0+40.242)*0.8,16.5,27.0)`给幅值，再以`sin(front_angle-back_angle)`生成前后关节一正一负的**电机补偿命令**；左/右电机安装符号也不同。这不是已辨识的机械广义力矩，不能直接把16.5–27当成active-link恒力矩。当前每腿10.775Nm恒定辅助仅保留为先前等效近似和历史评估条件，不再作为新模型与旧固件已对齐的证据；实机弹簧力—行程、力臂/安装点及关节符号仍需映射。`gas_spring.py`现有纯函数复现固件补偿公式，尚未接入Isaac/MuJoCo动力；不以此宣称同契约sim2sim。下一物理迁移先验证L0、夹角、被动力与电机补偿的符号/虚功，再同时更新Isaac、MJCF consumer和controller，按小步课程重训。

v465直接给速度/yaw大误差加完整有界尾段，前期reward低端裁剪从v464约6.8%升至16–18%，在10/25更新人工停止；仅中间checkpoint，无策略结论。v466将尾段增益降为0.1，25/25更新及冻结审计通过，但DR0十二点快筛仍1/12：1m/s +.8为83/83倾倒、+1.0为85/85倾倒，负向-1.0也23/23倾倒；`promoted=false`。因此reward shaping可保留速度/yaw区分度，不能单独修复此物理闭环；开关默认关闭，不续同配方长训、不升DR。证据`artifacts/continuation_v465`、`continuation_v466`。

v467用同一v430 checkpoint、seed4567、五点网格、旧1.0775Nm阶段弹簧、分离轮动力比较唯一参数postclip反馈D：D=0时+.8 58/58、+1.0 69/69倾倒，直行速度误差.703m/s；D=.080625时五点均存活、3/5快筛通过，但正向+.8/+1.0 yaw误差.0808/.1328rad/s且stable=0，仍不合格。旧高D只是未标定迁移锚点，不能写成实机阻尼或最终策略。迁移顺序：先在旧技能附近的小范围闭环存活与双向速度/yaw跟踪上建立可学习起点，再逐步降未标定反馈D并接入固件弹簧参考模型，随后逼近实机被动b[.001,.005]及目标DR；每步保留直行/双向retention并做MuJoCo同模型对照，失败即回到上一级。来源`artifacts/continuation_v467/final_report.json`。

## 2026-10-05 v461-v464 当前接续：旧1rad根因与闭环修复

历史v430的1m/s,+1rad/s确实在32/32回合存活、速度误差.03785m/s、yaw误差.04515rad/s；但当时气弹簧仅1.0775Nm/腿、旧轮D约.080625、局部DR=.05且解析bank门槛关闭，实际roll约-5.70度、bank误差11.52度。v447同源策略在修正轴向后的新postclip轮动力、仍为1.0775Nm弹簧且D_fb=0时首次正向终止，故退化先发生于轮动力迁移，不是10.775Nm弹簧单独导致，也不能凭现象断言符号错。v462同v458 checkpoint/最终10.775Nm弹簧只把D_fb改回.080625，+.8仍51/51 tilt；高D不能单独救当前闭环，且不是实机辨识值。

审计发现separated模式将轮IdealPD damping清零，而旧`max_wheel_damping_fraction` DR只缩放这个零值；实际机械b、C、反馈D此前均固定，不能称现有DR覆盖未知D。v461把每环境双轮b/C/D采样接入实际postclip受力路径；DR1原生读回b各轮位于[.001,.005]，C=.001固定、D_fb=0固定，legacy actuator damping DR明确inactive。它只证明接线，最终主要质量/惯量/摩擦±20%、延迟1–3周期和行为均未认证。用户指定postclip公式保持，硬件D施加位置仍待固件实测确认。

旧`combat-dm-balance-main`轮速PID Kp/Ki/Kd全0、电流前馈有效，左右电流和里程计符号内部一致，现证据不足判符号错。但旧应用的±8192 counts限幅按其自承未校准的换算仅约2.45Nm等效命令，不能证明当前5.5Nm峰值可达；电机任务1ms、DM发送2ms而上层RobotTask 5ms，旧腿走VMC/LQR力矩而非PD60/2。新独立C参考控制核心拒绝未经校准符号、counts/Nm、限值和腿力矩接口，不加旧近零速任务反馈；这不是已接入实机。实机周期更快不能自动保证1/2/8ms保持、状态估计和限幅等效。

v463修复MuJoCo controller contract遗漏的命令腿尺度knots：1.0m/s=.22rad、1.5=.24、3.2=.286，已有JIT/ONNX和checkpoint hash继续校验；1.5m/s,+.8在修正后MuJoCo跑满6秒但后半速度/yaw误差.808/.700，仍不通过。此漏项不解释1.0m/s正向失败，但使之前1.5/动态命令跨引擎对照不合格。

本轮硬件profile审计已同步当前postclip公式并运行通过，输出`artifacts/continuation_v464/hardware_profile_audit.json`；检查通过只表示配置算术自洽，`measured_calibration=false`、`hardware_certified=false`。反馈D绝对值、安装符号、库仑平滑、电流换算与5.5Nm实机可达性仍待硬件证据。CPU相关128项测试和C控制核心测试通过，不是策略或实机验收。

v459/v460按episode提高正向比例仍严重缺控制步。v464新增可选固定方向环境cohort，源码CPU测试通过；首次训练6/25更新后无traceback退出且无checkpoint，仅保留日志、不能处理为完整训练。独立重跑25/25更新完成，候选SHA`e8145191f54d5eb589b4243edcf1abe488950caff519cfa12cd7e32702942e01`。固定方向下+.8/- .8控制步3078/8275，对比v460的3140/11929，正向绝对样本并未增加。DR0十二点快筛仍1/12，+.8 83/83 tilt、+1.0 87/87 tilt，负向-1.0亦退化为tilt；`promoted=false`。停止同配置长训和DR升级。阶段rehearsal重置抽样仍导致控制步不均衡，但采样比例不是唯一闭环缺口；下一步必须证明更大可训练动作通道/课程初态对早期稳定的因果作用，保护负向与直行。机器证据`artifacts/continuation_v461`至`continuation_v464`；短期交付仍未认证。

## 2026-10-05 v459/v460闭环结论（覆盖下文v458下一步）

按用户“两天半内优先实机/DR鲁棒性，可少技能”要求，v459、v460均只做名义动力下的正向转向修复，不升DR。v458源SHA`7565b5d972c3b3d807078d1ae86b0e4ff65f8f1d19ee1b6b164ddb59f05aa642`；v459仅训练已有physics residual network 25更新，候选SHA`a0894f8937474bb8e997b857329f12590926d6938f606b61ae9ee3291748cd9f`，12点快筛仍1/12，1m/s正向+.8/+1.0分别83/88次全部tilt。对照固定命令的轮共同力矩反事实：轮动作共同+0.2（对应减少前进轮力矩）能使+.8四回合存活，但速度误差.7566m/s、yaw误差.3950rad/s、stable0；共同-0.2更差。该偏置只定位俯仰/接触失稳，不是可部署控制器。原始`artifacts/continuation_v458/{turn_failure_probe,wheel_common_bias_probe,positive08_bias_full}.json`保存失败前状态、双轮载荷和轮动作。

v459虽按episode正负各50%分配，实际1m/s+.8仅1791控制样本，-.8达30165，因正向约68–72步即倾倒、负向长回合占据环境。v460尝试95%正向分配在前三轮遗漏三个负向保护命令，由coverage guard在critic warmup终止，无checkpoint/行为结论；原失败见`artifacts/continuation_v460/attempt_01_status.json`。独立80%重跑完成25更新，26命令覆盖、6个physics网络张量之外均冻结，候选SHA`87650cf56ea9cb7492293e4e2236d52f4d4d0cf716771041c493f2f5f1b95bbe`；+.8样本升到3140但-.8仍11929，12点快筛仍1/12，正向+.8/+1.0分别83/86次全tilt，负向-.8 yaw误差.2933。首次v460快筛因代理中断未写出JSON，保留`quick_eval.log`；同命令独立重跑`quick_eval_retry.json`才是结论。v459/v460均`promoted=false`、`certified=false`，不进行正式32回合、DR晋级或新候选sim2sim部署。

**下个必要修复**：不能再以episode正负比例或简单增加PPO更新数代替有效控制步。先在同checkpoint/相同硬件契约下检查短寿命正向轨迹对PPO优势、reward与实际轮/腿动作的贡献，建立控制步/存活时间平衡的命令分配或专门的早期稳定训练机制；只做一个可归因的改动并保护负向/直行，再做名义闭环快筛。只有名义存活和跟踪达标，才逐级加入机械b[.001,.005]、主要参数±20%、观测/通信各1–3周期等DR；同checkpoint MuJoCo还须闭环行为通过。`rmuc_peak55.xml`静态2ms，带hash的consumer运行时设1ms，腿PD2ms、policy8ms；USD/MJCF静态质量/惯量/轴审计通过，但接触求解器、实机安装符号和反馈D绝对值没有认证。不能声称高sim2real成功率；可继续缩小交付技能集，不能降低已声称技能的验收门槛。

## 2026-10-05 v458结项与两天半交付优先级

v458重跑从v457候选（SHA `c26de0ed177e05e8e4ed0d0ddcc875b110b34c72d31cdc50217adad9eccd860a`）完成20 updates，候选 `artifacts/continuation_v458/final_checkpoint.pt`，SHA `7565b5d972c3b3d807078d1ae86b0e4ff65f8f1d19ee1b6b164ddb59f05aa642`。前5次critic warmup、后15次局部专家训练；26个要求的有符号命令均实际进入rollout（最低983样本），证实v457采样缺失已修。791个actor张量中只变动预定局部专家的160个，无非有限值；保护域同输入输出完全相同，目标域最大动作差仅.003。首次尝试中断且无checkpoint，独立标为无结论；本段只指重跑。

行为**未修复、未晋升**：最终5.5Nm峰值、3Nm持续仅热参考、后置机械b=.002/Coulomb=.001、反馈D=0诊断、10.775Nm/腿、轮1ms/腿PD2ms/policy8ms、DR0及外环关闭下，12点4回合快速筛查仅1/12；1m/s正向+.8与+1.0仍全部倾倒，1.5转向速度/yaw跟踪失败，±4自转失败。唯一raw通过为1m/s直行，速度误差.319869m/s，使用旧宽.35门槛，不是交付证明。不得升DR、追加盲目local expert更新或宣称sim2real成功。正式32回合、多seed、闭环retention、±20%主要参数、1–3周期延迟、热/电流/实机符号及实机replay均未完成。完整原始证据见`artifacts/continuation_v458/final_report.json`。

同checkpoint JIT/ONNX逐输入最大误差`2.384e-7`，显式yaw observation clip=6；MuJoCo直行6秒后半速度误差.444451m/s，1m/s,+.8在.552秒倾倒/高度终止，sim2sim行为失败。当前USD/MJCF逐体质量最大差`5.67e-7kg`、主惯量最大差`2.90e-7kg·m²`，关节轴和闭环初态数值匹配；轮gear1/±5.5Nm、腿±40Nm匹配。`rmuc_peak55.xml`静态timestep=.002s，consumer按签名contract把编译模型设为.001s；腿PD保持.002s。接触模型/求解器、实机安装符号和反馈D仍未证明等价，不能把静态资产一致当作sim2real认证。MuJoCo consumer已拒绝与contract不符的摩擦/被动阻尼覆盖，并读回实际值。

交付顺序收窄：先用真实rollout定位正向转向的可达动作、腿位余量、轮差动及接触载荷，得到有因果证据的单变量修复；只对稳定存活且跟踪达标的最小技能集逐级加入机械b[.001,.005]、主要参数±20%、观测/通信延迟各1–3周期等DR，并做同checkpoint MuJoCo与实机回放。时间不足时缩减速度/自转范围，不降低已声称技能的鲁棒性门槛；recovery、jump及全速度暂缓。

## 最新接续快照（2026-10-04，v457结项）

本节覆盖下文旧“当前/下一步”状态。短期交付范围为≤1.5 m/s稳定转向与±4 rad/s原地自转；GPU0允许共享，recovery/jump暂缓。最终轮端peak/scale5.5 Nm，continuous3 Nm仅热参考；机械b先验.002、目标[.001,.005]、库仑.001 Nm，反馈D独立且未辨识；轮1ms/腿PD2ms/policy8ms、腿PD60/2/±40 Nm、等效弹簧10.775 Nm/腿、径向力0、任务外环关闭。旧D[.002,.04]及D=.080625不得再称机械阻尼实测。最终主要参数±20%、观测/通信延迟各1–3周期仍未认证，ONNX/MuJoCo链路可并行推进但不能称行为通过。

v456和v457均未晋升。v457完成25 updates和12点快速诊断，候选SHA `c26de0ed177e05e8e4ed0d0ddcc875b110b34c72d31cdc50217adad9eccd860a`；正向+.8/+1.0仍早倒、1.5转向/±4自转失败。raw 1/12仅为1.0直行在4回合、旧0.35m/s误差门槛下通过，实际误差0.319869m/s，不构成交付或相对v456的通过率提升。正式32回合、闭环retention、DR晋级、sim2sim认证和实机部署均未完成。证据：`artifacts/continuation_v457/final_report.json`、`sampling_audit.json`、`quick_eval.json`。

已确认的执行错误：v456/v457只改了rehearsal gate列表，未把对应低速命令加入采样优先列表；v457全部409600个rollout样本中1.0m/s±.8/±1/±1.2/±1.4计数均为0。下次先修采样覆盖并读回实际命令，不把训练轮数当该技能学习量。yaw error clip=2还使静止时±3与±4分别混叠，当前课程没有spin命令，不能继续原样做自转续训。v457原始stdout未留档，`training.log`明确标为TensorBoard事件重建，证据缺口已记录；后续必须从结构化命令执行并保存原始日志。

v458修复计划已落盘但尚未启动：源为v457候选，唯一训练通道改为已有`rmuc_position_authority.local_correction`专家0–7，physics/source actor冻结；priority IDs与weights同时覆盖48–60，新增启动前配置校验和前三次rollout实际25D命令coverage guard。为解除静止±3/±4输入混叠，显式迁移`command_yaw_observation_error_clip`从2到6，checkpoint、Isaac、JIT/ONNX metadata、MuJoCo和controller必须读同一观测契约。仍为5.5/3/10.775、后置b=.002/Coulomb=.001、D_fb=0诊断、DR0、外环关闭；缺样或契约不一致会在训练前/前三轮直接停止。证据目录`artifacts/continuation_v458/`。

2026-10-05 v458 attempt01：实际启动并完成5次critic warmup，26个要求的有符号命令均进入rollout（最少262样本），修复了v457的目标命令零样本；随后进程无Python traceback消失，未产生checkpoint，actor更新0，行为无结论。保留原日志和`artifacts/continuation_v458/attempt_01_status.json`，仅在相同源/配置下以新日志和独立PID重跑，不覆盖历史证据。当前最高优先级调整为少量真正通过的技能在目标DR与同契约sim2sim下可用；未拿到实机符号/反馈D标定及实机回放，不能声称sim2real成功率已保证。

## 历史执行口径（2026-10-01，后续更新以上节为准）

- 最新推进门槛（用户再次确认）：先在Isaac中证明最终动力契约下最高联合DR通过，再正式迁移到sim2sim验收。目标damping范围鲁棒只是其中一项；还需完整八点、多seed、连续命令及实际赛场DR覆盖。强度数值1不代表未实现因素已验收。此前MuJoCo探针仅为链路诊断，不是获准迁移或行为通过；当前damping及最高联合DR均未通过。
- 用户最新确认：轮端5.5Nm为瞬态峰值/执行器指令饱和能力，3Nm为持续热参考，不再把3Nm当所有时刻的硬限幅。目标profile已升级schema2，区分peak与continuous；旧3Nm实验保留为降额诊断，不改写成绩。
- 当前优先级：先做locomotion的damping鲁棒性，再推进ONNX/MuJoCo同契约链路及后续训练。热时程/RMS/超3Nm占比与连续时长后续补验，不以其阻塞当前damping适配，也不宣称热安全已验证。峰值随转速/电压衰减的包络仍需辨识，不能把高转速低扭矩当热安全证明。
- 气弹簧最终目标仍为每腿10.775Nm等效力矩、径向力0；1.0775Nm只是当前10%迁移阶段，不能冒充最终验收。
- 用户允许无意义训练提前终止：25更新分段检查真实行为，明显退化/接线错误立即停止，连续两段无改善不默认续满100。此规则覆盖旧“不得因短期停滞停止”的执行偏好；少量探针不代替正式八点验收。


## 轮电机峰值与持续指标：必须分开（用户2026-10-01确认）

实机轮端为 **RM3508 电机 + C620 电调**，以下力矩均为**每只轮电机减速器输出轴侧**，不得再次乘减速比。

| 数值 | 含义 | 当前用法 | 禁止混用 |
| --- | --- | --- | --- |
| **5.5 N·m** | 瞬态峰值、执行器指令饱和能力 | 当前目标 action torque scale 与 actuator effort limit 均为5.5；以实际输出读回复核 | 不是持续热额定值，也不代表所有转速下都能提供5.5 |
| **3 N·m** | 持续输出/热参考 | 后续检查力矩时程、平均绝对力矩、RMS、超3占比和最长连续时长 | 不是默认瞬时硬限幅；不得在ONNX或MuJoCo出口偷偷裁成3 |
| **10.775 N·m/腿** | 气弹簧最终等效助力矩 | 两个active-link关节及机身反作用，径向力模型关闭 | 不是轮电机力矩，也不是10.775 N径向力 |
| **1.0775 N·m/腿** | 气弹簧10%迁移阶段 | 当前v392明确采用此阶段参数 | 不是最终助力或最终硬件验收 |

权威目标：`configs/rmuc_hardware_target.yaml` schema2，显式区分 `max_peak_output_torque_nm=5.5` 与 `max_continuous_output_torque_nm=3.0`。这是用户确认的工程契约，不冒称温升/电流标定实测。<5%占比和<300ms连续超3时长仅为提出的筛查参考，未验证热安全；高转速下力矩减小也不能单独证明热安全。真实转速—力矩包络、电流映射及热统计后续补验，当前不阻塞damping训练与sim2sim。

历史3Nm硬限幅实验、旧XML、旧checkpoint与报告保留原含义。目标配置更新不等于所有旧运行默认值已自动更新；每次启动核对所选profile及实际执行器读回。文中旧“最终3Nm”“当前运行”描述属于当时记录，不能覆盖本节。

用途：按用户 2026-09-22 要求，每次发现瓶颈、修正判断或验证攻克方法后更新本文件，用于后续执行前反思。它记录证据与经验，不替代当前代码、配置、日志和验收报告。

## 记录约定

- 每条包含：现象/影响、证据、反思、采取的方法、验证结果与限制、下一步。
- 状态限定为：待验证假设、原因已验证、修复已验证、局部行为通过、全部验收通过。代码修复与技能通过分开记录。
- 未通过或被推翻的尝试也保留；后续更正追加说明，不能删失败样本、覆盖旧报告或追溯改写通过标准。
- 更新触发点：新瓶颈出现、一次有结论的实验结束、改变训练基线/奖励/控制契约、判断被证据推翻。每次续接任务先核对本文件未闭环项。
- 所有阶段沿用25-D/6-D、closed-loop USD、frozen skill graph与retention要求。用户授权的高度标准变更单独注明，不能当成学习能力提升。

## 2026-09-22：本轮补录

### 1. 训练长期卡关，反复短跑却没有解决整体阻塞

**状态：执行问题已确认；技能恢复未完成。**

- 现象：多轮复训仍停在Level26。只看当前turn gate，会漏掉旧技能retention失败。
- 证据：v157 iteration1544日志同时显示当前负向yaw error=0.0617、stable=0.4194，以及Level23 retention speed error=0.2746、stable=0；完整日志见 `train_critic_unclipped_calibration_to31_v157.log`。这些是该轮快照，不是永久当前值。
- 反思：此前重启过于频繁，重新积累完整episode；旧启动命令stop_at_command_level=27也无法连续推进到31。仅播报接近阈值、只列stage gate均不完整。不能承诺“一小时必然收尾”。
- 方法：续训停止点改为Level32 entry，保留完整日志和安全STOP checkpoint；独立评估逐速度/逐方向，不跳过retention。
- 结果：当前v157仍未跨过Level26，不能认定攻克。一次指标波动不是新退化的充分证据。
- 下一步：用正在进行的2.0/2.2 m/s双向完整回合复测定位失败方向，结合retention决定下一基线。

### 2. frozen base没有变，但组合策略仍削弱旧直行

**状态：局部原因已验证；2.9直行nominal与早期DR通过，未全局采用。**

- 证据：旧model600仅关闭局部2.9直行domain输出，speed error从0.168342改善至0.150587，stable从0.026129改善至0.382316，但仍未通过。见 `artifacts/continuation_v153/no_local_straight64_nominal.json` 和 `artifacts/continuation_v151/model600_straight64_nominal.json`。
- 进一步证据：v130在当前控制条件下2.9直行speed error=0.146924、stable=0.604515、survival=1；后续model900为0.200698、0、1。原base能力仍在，外接speed expert覆盖与domain叠加需要单独审计。
- 反思：只证明base参数冻结，不能证明最终动作和行为不遗忘。L2锚点若本来已弱化，也不等于保护正确能力。不能把所有残差一概判断成“爆炸”。
- 方法：构造诊断候选 `artifacts/continuation_v158/exact_base_2p9_candidate.pt`，禁用外部speed expert混合，局部抑制legacy straight correction并清零2.9straight局部输出层；保留原冻结权重。
- 验证：332个张量未变；2.9直行探针输出与base完全一致，另外44个抽查锚点不变。相邻速度/小yaw过渡区会变化，尚未覆盖。完整JIT/ONNX校验通过，另有7项路由兼容/局部性/非法值测试通过。
- 行为结果：seed43、64回合nominal：speed error=0.146924、stable=0.604515、survival=1、root_min=0.294759；seed44、DR2/19、64回合：0.146764、0.505539、1、0.291888。两次所有checks通过。证据为v158目录 `exact_base_straight64_nominal_height015.json`、`exact_base_straight64_dr_seed44.json`。
- 失败尝试：用两个浮点残差相减恢复base的前一候选出现2次失败，未采用。尚不能断言浮点抵消就是失败根因。
- 下一步：转向、过渡区、更多seed/fullDR及后续训练不再削弱该路由的验证；官方v157未切换候选。

### 3. 同状态同命令，rehearsal与active奖励不同

**状态：原因与奖励一致性修复已验证；不等于技能通过。**

- 现象：仍在学习的domain head面对不同cohort/global level的不同优化目标。
- 证据：真实物理状态下只切换cohort/global level，旧奖励最大差异1.933624/5.093191；新逻辑均为0。见 `artifacts/continuation_v153/reward_contract_physics_audit.json`。
- 反思：控制契约一致仍不够，奖励也必须与命令一致；不能忽略rehearsal中的可训练分支。
- 方法：启用command_reward_contract_by_command，速度资格按command决定，同命令bank系数一致。
- 结果：实际奖励函数测试及物理状态counterfactual审计通过；后续model900仍未通过全部行为验收，证明单项修复不足。

### 4. 提高bank奖励后，总奖励被cap压平

**状态：学习信号修复已验证。**

- 证据：bank scale=2时正项上界14.85，旧cap=10；真实姿态扫描yaw error=0.06时仍有31.25%状态到cap，使不合格与更精确动作丢失奖励差异。见 `artifacts/continuation_v154/reward_clip_actual_config_audit.json`。
- 反思：调整单项奖励未检查总量与截断，是执行遗漏；不能靠持续训练恢复已被截断的区分信号。
- 方法：cap改为20，不改变速度/yaw/bank等gate。
- 结果：同姿态同配置扫描high_clip_fraction全为0，恢复正确排序。见 `artifacts/continuation_v155/reward_clip20_physics_audit.json`。奖励修复不是Level通过证书。

### 5. critic误差很大，却沿用绝对单位0.1的value clipping

**状态：机制已验证；真实校准改善但未完全解决。**

- 证据：MC审计model900 critic偏差约+559；实际PPO分支中预测-1000向目标-1500移动到-1001，旧clipped分支梯度0，普通MSE梯度998。固定数据留出环境100轮校准：旧RMSE439.47，新159.11。见 `artifacts/continuation_v156/critic_clipping_comparison.json`。
- 反思：高EV不能掩盖均值偏差；不能把actor的clip比例直接当成适合critic量纲的限制。
- 方法：关闭critic value clipping，保留actor PPO clip=0.1；真实多任务50轮critic-only warmup。未采用单command离线拟合critic。
- 结果：真实MC复核RMSE467.04→374.97，bias429.45→328.56；warmup actor权重不变。见 `artifacts/continuation_v157/live_critic_calibration_check.json`。仍有明显误差，不能宣称已彻底收敛。

### 6. control表现与训练gate不一致

**状态：配置差异已确认；旧GUI问题根因尚未完整证明。**

- 证据：`control.py`仍默认加载9月18日旧Level18 entry；键盘max_forward_speed=3.0，不是2.0。当前play会恢复完整策略模块，不能说当前只加载base。
- 差异：control默认未启用command-based feedback和low-speed yaw clip=0.3；v157/grid启用。历史Level编号与当前课程不能直接类比。
- 反思：不核对checkpoint、obs/action/feedback、命令和实际速度就继续复训，可能训练与使用两套不同对象。
- 方法与边界：已审查实际代码，记录 `artifacts/continuation_v158/control_contract_audit.md`；不自动将未认证中间策略设为最终默认。
- 下一步：同checkpoint、同命令下进行控制配置A/B及连续命令验证，才能把差异归因于控制器或策略。

### 7. 高度门槛与用户允许低姿态的新要求冲突

**状态：授权标准变更已落实；触地边界未验证。**

- 用户允许root z约0.15 m，拒绝仅因低于0.20 m就判失败。
- 方法：行驶稳定性/grid最低root z改为>=0.15，行驶最低高度惩罚目标0.36→0.15；补齐旧stage evaluator的command平均高度门槛0.15。站立任务目标及其他姿态/速度/bank/生存检查保留，仍有软姿态奖励偏好。
- 验证：相关13项测试及stage脚本编译通过。0.15并非已确认的导轮触地高度，需姿态/接触实测。v157旧进程仍用启动时0.20配置。
- 反思：修改一套评估后必须搜索其他入口，避免训练/grid/stage标准不一致；历史报告不能按新门槛追溯宣称通过。此次2.9候选最低高度>0.29，成功不依赖门槛放宽。

### 8. 将1000步中途日志误认为复测结束

**状态：观察错误已纠正；终端日志损失已披露。**

- 经过：DR测试仍运行时，因JSON暂不存在和日志显示1000步而错误判断未完成；未确认旧进程终止就再次启动。
- 结果：原进程正常完成1500步/64回合且通过，重复启动被禁止覆盖JSON检查拒绝，JSON完整。但shell重定向先覆盖了原.log，留下重复启动拒绝信息。
- 反思与执行提醒：报告暂不存在、一次poll超时、进度日志均不代表进程结束。先poll原session/PID，再读结果；未确认终止不重启。重试采用新日志路径或排他创建，禁止用同名 `>` 覆盖证据。

## 尚未闭环的任务

- Level26当前方向和旧技能retention仍须共同通过；候选2.0/2.2双向复测正在运行，未出结果前不得预判。
- 2.9恢复候选仅通过nominal与DR2/19；缺全速度/双向/过渡区/fullDR/actuator/deployment证书。
- 随机化与动力能力可能矛盾：满强度DR的高damping/低gain组合可能使3.2 m/s命令不可达，需按当前力矩、阻尼和轮径做物理边界核对并验证，不能通过删掉难样本伪造鲁棒性。
- 用户要求实时检查base是否被残差削弱：继续记录冻结权重变化、分命令最终动作偏差和独立行为指标；只看参数或L2 loss均不足。

## 2026-09-22：续接后的证据更新

### 9. 当前训练的持续阻塞与随机化力矩余量

**状态：训练阻塞已用窗口统计确认；动力边界仅完成解析筛查，尚未物理验证。**

- 证据：`artifacts/continuation_v158/v157_recent_gate_snapshot.json`保存近50轮逐轮指标。负向2.2 turn yaw error均值约0.06014、stable0.47117；正向约0.02403、0.89809。Level23 retention speed error约0.29998、stable0。阶段指标均值仅用于诊断，不代替连续gate或独立episode验收。
- 反思：把“负向turn接近通过”当成整个阶段接近通过，忽略了retention的持续失败。修复直行候选与解决负向yaw必须共同推进；当前两次直行通过还不足以切换全部策略。
- 方法：继续原进程并执行候选双向2.0/2.2的64回合/bucket独立评估；新增路由兼容/局部性/非法buffer测试7项通过。补齐旧stage evaluator行驶平均高度门槛0.15；合并相关测试13项通过。
- DR筛查：`artifacts/continuation_v158/audit_actuator_envelope.py`读取当前配置，保存config hash与全部假设到`actuator_envelope.json`。fullDR低gain0.95、高damping0.11时最大命令力矩5.225 Nm；按r=0.06、稳态无滑移计算，3.2 m/s需要5.8667 Nm仅抵消阻尼，差额-0.6417 Nm；空载估算2.85 m/s。当前nominal 5.5 Nm/0.1也只剩很小高速余量。
- 限制：不是严格动态速度上限，不覆盖腿部推进、瞬时滑动、外力或实际轮胎有效半径；没有根据解析值伪造仿真失败。未改USD/物理/DR范围，也未以缩小目标处理矛盾。
- 下一步：完成正在运行的转向复测后决定候选采用；在fullDR/actuator桥接前做实际极端参数仿真与实测阻尼核对。不能为了“通过”删除难参数组合，也不能让物理冲突无限消耗训练时间。

### 10. 修复的直行head可能在下一轮被optimizer再次改坏

**状态：可选保护代码及optimizer测试通过，尚未在正式续训中启用。**

- 瓶颈：2.9直行base路由恢复后，speed-local domain head仍是可训练大tensor的一部分；仅冻结base不能维持恢复效果。
- 反思：仅把该切片梯度置零不够，已恢复的Adam动量或AdamW weight decay仍可能更新参数；也不能为保护一个速度冻结所有速度/方向的学习。
- 方法：新增`--protect_nominal_straight_anchor`与`scripts/rsl_rl/rmuc_anchor_guard.py`。仅在scheduled DR strength=0时，保护2.9直行局部head的4个参数切片，梯度屏蔽、对应optimizer状态清零、step后精确恢复受保护切片；DR开始后解除。必须显式使用已恢复base路由，否则拒绝启用。安装点位于critic warmup冻结之前，不提前解除warmup。
- 验证：`tests/test_rmuc_anchor_guard.py`构造已有动量的Adam/AdamW，验证保护切片逐位不变、其他切片更新、momentum/variance清零及解除后恢复更新。连同路由、warmup、residual safety共18项通过。运行中的v157不受代码修改影响。
- 限制：保护的是局部参数，不是随机探索或DR行为证书；3.0以上stage head与连续过渡区仍须独立验收。新L2参考若继续使用旧的弱化直行锚点，可能与修复目标冲突，采用候选时必须明确、可追溯地处理，不能静默漂移锚点。

### 11. 双向完整复测定位retention失败，进入v159定向续训

**状态：候选2.2双向局部通过；2.0双向未通过；下一轮已启动待运行验证。**

- 证据：`artifacts/continuation_v158/exact_base_turns_nominal_seed43.json`，seed43/12000steps，每bucket64完整回合，四桶survival均1。2.2正向：yaw0.03028/stable0.79849/bank2.50344°；负向0.04361/0.52341/0.85108°，两桶全部通过。2.0正向bank3.45673°超3°；负向yaw0.05172、stable0.42140，未通过。
- 反思：当前阶段与retention必须分别查；完整回合显示候选2.2已具备本次nominal能力，而旧2.0技能仍需修复，不能只继续集中更新2.2。候选来自1350，比正在运行的v157晚期checkpoint旧；选择它是基于明确复测证据，不声称最新checkpoint必然更差。
- 方法：v157用STOP正常保存`model_requested_stop_1643.pt`并退出。v159从已测候选恢复，nominal直行切片保护启用，rehearsal比例0.40→0.60，其余stage/domain LR不变。采用用户新高度0.15口径，50轮critic warmup适应奖励。显式将L2参考改为固定候选，保留旧271锚点及所有旧checkpoint；新参考不是全技能已认证声明。
- 验证：33项相关测试通过。启动计划与PID在`artifacts/continuation_v159/launch.json`；必须看到实际迭代及guard日志才视为训练运行。保留完整Level26→31与后续目标，不跳retention，也不改yaw/bank gate。

### 12. 用户质疑反复回退：阶段通过不能冒充持续掌握

**状态：验收遗漏已修复于磁盘，v159运行进程尚未加载；整体收尾未完成。**

- 质疑：从Level29回看28/26/23，再看2.0 m/s，是否之前稳定学会的说法不实，何时才能结束。
- 反思：历史通过只能证明当时checkpoint和当时gate覆盖的条件，不能证明后续组合残差仍保留能力；不同阶段编号也不能代替速度/方向明确标注。此前“稳定掌握”的表达超出了跨checkpoint和完整行为验证证据，应纠正，不能笼统宣称旧结果全是假或全部有效。
- 已确认的两类问题：旧2.9直行被组合图削弱，恢复base路由后nominal和DR2/19重新通过；retention未检查bank，可能放过速度/yaw正常但没有正确压弯的旧技能。后者不能追溯证明每个历史checkpoint都不会压弯，但证明旧gate覆盖不足。
- 方法：磁盘env.py已补retention双向bank EMA、3°判定、日志、进阶/actuator推进检查，并让方向优先采样考虑bank deficit。新增4项方向/直行豁免测试，全部37项相关测试通过。v159原进程尚未加载此改动，不假称live gate已修复；完整grid始终保留bank检查，作为最终证据。
- 当前真实进展：v159 iteration1392仍critic warmup，但恢复路由后的训练内Level23 speed error=0.1473、stable=0.6002，较v157约0.30/0确有改善。2.2双向候选64回合通过；2.0失败只是正向bank3.46°、负向yaw0.0517/stable42.1%，不是已全部不能行驶。训练内snapshot不替代独立证书。
- 收尾提醒：使用固定checkpoint/hash+实际速度/yaw+nominal/DR强度+明确gate报告管理通过项；后续改动影响该行为才安排针对性回归。保留完整全速度/双向/fullDR/3Nm与实测阻尼/部署目标；没有证据时不报虚假的完成百分比或小时承诺。停止把无限短跑和扩大复测清单当成推进。

### 13. retention bank修复已进入实际运行，保护通过真实checkpoint核验

**状态：修复已加载，整体技能仍待通过。** v159完成50轮critic校准，保存1409；2.9straight局部head的weight1/bias1/weight2/bias2切片与候选逐位相同，验证保护在真实actor更新中有效。v160从1409接续，不重做校准、不换L2锚点/奖励/LR/physics；新retention bank指标已实际输出。另补真实统计函数测试，验证只累计ready且匹配cohort/stage/direction的bank样本，排除非rehearsal和未ready污染，bank测试9项通过。初期一次2.2正向85.5°倾倒已保留日志；deterministic少量完整回合通过不保证随机策略绝无失败，不能用单次倾倒推翻全部证据或再盲目重启。证据：artifacts/continuation_v160/launch.json、train_retention_bank_complete_to31_v160.log。

### 14. 用户明确必须稳定学会；满强度DR实测未通过

**状态：稳定掌握尚未证明，fullDR缺口已量化。**

- 要求更新：单次阶段通过不能作为最终完成；固定同一策略，重复独立seed、双向、连续命令/较长持续行驶、fullDR、后续retention及actuator/deployment都需实际证据。
- 方法：保持v160连续训练不重启，对已通过nominal/DR2/19的固定候选进行seed45、fullDR、64环境、1500步验证。参数确实为mass±5%、friction±15%、damping±10%、gain±5%、噪声0.01及延迟上限1步。
- 结果：69已结束回合（早终止环境可重开），33次tilt，survival0.521739、linear_error0.335625、stable0.279171，coverage也未通过；root_min0.186895，因此并非高度门槛拒绝。证据`artifacts/continuation_v160/full_dr_straight64_seed45.json`。
- 反思：earlyDR通过不能冒充fullDR；这是从未完成的鲁棒性验收，不是又推翻同一条件下的nominal通过。不能把动力边界估算当成此次倾倒的确定根因：fullDR最差空载估算2.85仍高于2.9的0.15误差下界2.75，首16个失败记录也都未低于2.75。3.2上限矛盾需单独处理。
- 失败数据：报告保留前16条倾倒前轨迹及实际DR参数，筛查`full_dr_failure_screen.json`明确只截取先发生的失败，没有survivor参数配对，不能据此归因。后续需要物理DR与观测噪声/延迟的A/B，当前不取消随机化或删困难样本。
- 防遗忘验证：v160/model1425受保护2.9head的4个切片仍与固定候选逐位相同，证据`anchor1425_check.json`；参数保护不等于fullDR行为通过。
- 证据管理：按当前课程去重得到71个有符号速度/yaw点，固定候选的已有证据索引写到`candidate_evidence_matrix.json`（同checkpoint hash）；只索引该候选已有报告，不把历史其他checkpoint的通过拼成最终证书，也不把未测试点说成失败或重新遗忘。生成脚本`build_candidate_evidence_matrix.py`可重建。无虚构完成百分比。

### 15. 将fullDR失败拆成物理与观测随机化对照

**状态：诊断入口与防误认证测试通过，实验正在运行。**

- 瓶颈：fullDR同时改变质量、摩擦、阻尼、增益、噪声和延迟，单次倾倒不能指出是哪一类影响，也不能直接归因于动力上限。
- 方法：`scripts/evaluate_rmuc_grid.py`新增诊断参数`--dr-ablation physics_only|observation_only`。前者仅关闭随机观测噪声/延迟，后者仅将物理参数随机幅度置零；fullDR默认路径不变。所有诊断整体passed强制False，保留逐桶checks用于比较，并记录实际配置maxima。
- 验证：直接执行实际report passed表达式的测试证明两类诊断即使所有row通过也不能成为完整证书；连同layout/bank共15项通过。启动串行worker `artifacts/continuation_v160/run_dr_ablation.py`，同候选/seed45/64环境/1500步，排他创建日志避免再次覆盖；状态与子进程PID在`dr_ablation_status.json`，进程未终止不重启。
- 初步证据：physics_only已记录约580step倾倒，故观测噪声/延迟不是产生倾倒的必要条件；完整回合结果未到齐，不比较最终失败率。不同配置可能改变RNG消耗与后续reset分布，不把同seed当作所有episode严格配对的因果实验。
- 下一步：读取两份完整报告，对照combined fullDR，再决定后续鲁棒性训练重点。v160继续原基线与原课程，不因诊断直接放宽DR范围。

第15条阶段结果：physics_only已完成，实际数值以dr_physics_only_straight64_seed45.json为准；observation_only已由同一worker顺序启动。后续补全对照，诊断不算完整DR通过。

## 最终启用策略的硬条件（2026-09-22 用户明确）

最终control/controller及部署启用的必须是**同一份完整专家混合残差组合策略**，对应唯一checkpoint hash与导出制品；必须经过最高DR（strength=1.0）训练/验证并具备足够鲁棒性，同时覆盖全速度直行、双向压弯行进和转弯。不能只认证最高速度一个点、只验一个转向、只用nominal/早期DR结果，也不能拼接不同checkpoint的通过记录作为最终证书。

稳定掌握需在同一最终策略上重复独立seed、连续速度/yaw变化和持续行驶，逐点检查tracking、bank、survival、接触/姿态与用户认可的高度口径；完成后续actuator迁移后需在最终部署控制契约和最终动力参数下重新验证最高DR。全程保持25-D/6-D、closed-loop USD与frozen skill graph/rehearsal/retention。导出需完整组合图及数值一致性校验，control默认策略只能在上述证据齐全后切换。

若随机化范围与动力能力存在物理矛盾，必须明确诊断、解决并验证；不得静默降低DR范围、放宽性能gate或掩盖失败来交付。当前候选尚未满足这些条件，不能作为最终启用策略。

### 16. DR对照已完成：当前主要缺口在物理参数变化

同候选、seed45：combined fullDR生存率52.17%/33 tilt；physics_only 56.06%/29 tilt；observation_only 64回合全部存活、speed error0.147035、stable0.576087，逐桶checks全部通过（诊断整体passed仍强制False）。见artifacts/continuation_v160/dr_ablation_comparison.json及三份原始JSON。支持该候选在本次条件下对物理变化不稳，不代表所有速度都已对噪声/延迟鲁棒，也未区分质量/摩擦/阻尼/增益各自贡献。同seed并非逐episode严格配对。后续需按最高DR覆盖全部速度/双向完成训练与验证，不能以observations-only通过代替完整DR。

### 17. 连续训练的固定里程碑验收，避免再次碎片化重启

**状态：v160持续运行；自动评估worker已核实存活、正在等待真实checkpoint。**

- 瓶颈：1498–1528的31轮窗口中，2.0正向bank均值3.446°、负向yaw0.06510仍不合格；2.2负向yaw0.05880仍需收敛。2.9直行误差0.14704、stable0.54734保持平均达标，但窗口均值不是最终证书。
- 反思：频繁读取接近阈值的单轮数值、立即调参或重启，既无法证明持续学习效果，也重复消耗统计窗口。改为在明确学习量/真实进阶时验证一次完整策略。
- 方法：新增`artifacts/continuation_v160/validate_milestone.py`，选择首次Level27 entry或model1700完整checkpoint，核对actor与sha256后，对2.0/2.2、yaw±0.30、每bucket32环境/3000steps/至少64完整回合进行nominal独立验证。不会控制或重启trainer；排他创建日志，进程身份按/proc命令行核实；trainer退出且无里程碑时明确终止等待，不伪造报告。
- 状态：`milestone_validation_status.json`，worker PID225172、trainer PID221425当前均存活，waiting_checkpoint；不是评估完成。脚本编译通过；本轮不变奖励、LR、DR和策略基线。
- 限制：与旧8 replicas×12000steps的报告并非严格配对轨迹，不能将数值差直接归因于单一改动；按同一验收标准判定新checkpoint行为。检查点尚未产生，不能预判通过。
- 通信延迟说明补录：现有DR为reset采样固定观测延迟，最大1控制步=8ms，fullDR概率35%；没有动作下发延迟、逐步抖动/丢包/乱序模型。先前observation_only通过只证明该既有观测扰动组合的一次2.9直行诊断，不是完整通信鲁棒性证书。

### 18. 部署能力不能直接由USD数值与策略导出推断

**状态：当前配置已核对；实机数据待补，不阻塞当前nominal训练。**

- 证据：configs/rmuc_wheel_leg.yaml当前sim_training，控制8ms，wheel action scale/effort limit=5.5Nm，explicit damping=0.1。wheel_authored_effort_limit_nm=3.0的注释明确是USD元数据，不是电机实测。real_hardware limit_ratio_real=0.8；若bridge最终3Nm且保留该比例，命令限幅实际为2.4Nm（条件计算，不是实测）。
- 反思：把“迁移到3Nm”直接说成“实机拥有3Nm有效命令能力”，或把导出文件存在当成部署通过，会漏掉限幅、反馈、观测与实际通信契约。
- 方法：生成artifacts/continuation_v160/deployment_contract_inventory.json，记录config hash、已核对值与未实测项；已向用户请求可用的电机持续/峰值扭矩、阻尼/摩擦、控制周期/延迟数据。未获得的数据不自行编造、不修改USD或真实参数。
- 下一步：当前v160与milestone worker继续；到actuator/deployment环节，用明确的最终有效参数重新验证同一策略的最高DR与全速度双向能力。若尚无实机数据，明确区分仿真验收与实机验收，不擅自降低最终要求。

### 19. 核实局部head确实更新，区分空跑与尚未学好

**状态：真实checkpoint更新已确认，不代表技能收敛。**

- 方法：比较v160源checkpoint1409与model1550，逐速度/方向检查speed-local domain bank的weight1/bias1/weight2/bias2，另读checkpoint实际optimizer param groups。
- 结果：2.0正向weight1/weight2最大变化0.01337/0.01291，负向0.01066/0.01107；2.2正负向也非零。domain LR=1e-4，stage LR=2e-5；2.9straight的四个受保护切片变化全部0。证据artifacts/continuation_v160/learning_update_audit.json。
- 反思：排除了“整个待修复head被误冻结”的解释，但参数改变不等于有效学习，也不能仅凭这些数值提高LR或宣称收敛。继续原训练至既定进阶/1700里程碑，以完整行为验收判断；不新增restart。

### 20. 硬件扭矩口径已由用户澄清

用户明确轮电机RM3508、3 Nm为持续扭矩。先前第18条“3 Nm只来自USD，持续能力未知”的状态已被此新信息更新；来源是用户确认，不是本代理实测或官方规格独立核验。最终目标使用3 Nm持续能力，不依赖更高瞬时峰值。已更新deployment_contract_inventory.json与交接文档。既有real_hardware 0.8限幅仍需显式处理，未偷偷改动。当前训练不因此中断，最终需完成从5.5 Nm仿真到3 Nm硬件契约的迁移和最高DR验证。官方搜索工具本次返回HTTP404，未获得可引用规格，不据此编造厂家结论。

第20条口径更正：用户进一步明确3 Nm为“持续最大扭矩”。部署记录字段改为max_continuous_torque_nm=3.0；不能将其解释为固定输出扭矩或擅自假设更高峰值。当前仿真参数未改动，最终迁移与验收须遵守该持续上限。

### 21. 参考XML发现轮端力矩重复放大；气弹簧不在XML内

**状态：轴端由用户明确，力矩转换已实算验证；气弹簧尚未接入当前RMUC。**

- 用户提供wheel_leg_mjrl-lqr/wheeled_infantry.xml，确认实机有气弹簧；随后明确RM3508的3Nm持续最大扭矩是减速器之后。
- 实际MuJoCo加载：XML gear15.7647058824、ctrlrange/forcerange±3，输入3时actuator_force3、wheel qfrc_actuator47.294Nm，两侧一致。说明参考参数化把轮端规格重复乘传动比，不能因“参考工程”就直接相信。
- 方法：原文件只读，内存模型保留gear、将输入/actuator force上限折算±0.190298507462Nm，mj_forward验证两侧正负轮端均为±3Nm。证据reference_motor_spring_audit.json、wheel_output_torque_conversion.json；均位于artifacts/continuation_v160。
- 气弹簧：XML spatial tendon仅测腿长、关节stiffness=0；lqr_deploy.py:1621对左右active_link_pitch施加qfrc_applied=-10.775Nm。warp_flat_controller还有与电机合力矩裕量裁剪的不同语义。它是恒定广义力矩近似，不是测得的力—行程关系，不能误认成XML已含气弹簧。
- 当前差异：RMUC环境_spring_idx为空、无对应气弹簧辅助力调用；USD中电机drive stiffness不是独立被动气弹簧。轮轴阻尼参考0.01，当前0.1。这些动力差异影响最终部署，现有nominal通过不能直接证明实机稳定。
- 下一步：在保留USD闭链与25-D/6-D条件下，验证气弹簧力矩方向/虚功和与motor effort的独立性，再纳入渐进迁移和最高DR验收；最终轮端持续上限3Nm，不使用47.294Nm虚假能力。当前训练不因参考读取而偷偷改变物理。

### 22. 气弹簧独立机械辅助已接通，默认关闭，尚未完成迁移

**状态：实现与短时原生PhysX验证通过；不是稳定性/实机证书。**

- 方法：新增gas_spring.py，对当前USD左右active_link_pitch（parent-local X，localRot0 identity）生成连杆各-10.775Nm与机身+21.55Nm的世界坐标力矩对；启用前检查USD关节父子关系/轴，拒绝不匹配资产。经instantaneous_wrench_composer逐physics子步累加，保留其他外力；不写入电机命令，不消耗轮电机3Nm额度，不增加25-D/6-D维度。被动弹簧不会因电机estop自动消失。
- 配置：gas_spring_assist_torque_nm默认0，允许参考近似0..10.775；v160运行进程仍未启用。独立evaluator新增--gas-spring-torque，仅用于诊断，整体passed强制False。
- 验证：45项相关测试通过，覆盖随机姿态下净力矩守恒、虚功、零幅度无力、当前USD轴/方向，以及诊断不可冒充证书。8环境/500steps/零命令原生PhysX无终止；另400物理子步直接读取composer真实输出，world torque误差max1.91e-5Nm，净力矩误差max2.19e-5Nm，见gas_spring_wrench_probe.json。
- 反思：不能用“没崩溃”替代力学验证，更不能用短时探针证明气弹簧改善行驶。参考幅度与零幅度对照的root均值0.325070/0.325226、最低0.302988/0.303984，差别很小，不能凭此断言伸腿增益或稳定性提升；仍需关节位形、motor负载和闭链约束残差下的迁移评估。
- 证据：artifacts/continuation_v160/gas_spring_reference_probe.json、gas_spring_zero_probe.json、gas_spring_wrench_probe.json。默认资产/训练动力均未改；真实气弹簧力—行程不是该常力矩近似已被验证。
- 下一步：1700里程碑评估已自动启动，先接收该同checkpoint双向结果；气弹簧与轮端3Nm/参考阻尼的渐进迁移另需验收，不把当前无气弹簧nominal通过当最终实机能力。

### 23. 气弹簧应为径向力，不能把力矩通路验证当成机械等效

用户指出应等效为小腿径向力。第22条恒定active-link torque只是参考控制代码的近似；即便满足净力矩守恒/虚功并写入PhysX，也不能证明等效真实气弹簧。正确起点是作用点、作用轴线与F(l)，然后以tau=J^T F映射闭链广义力，或在两端直接施加成对作用力；不能凭10.775Nm猜推力。XML小腿为刚体，膝到轮心固定距离，对同一刚体两端施加共线内力不会自动形成可变腿长助力，必须核实所指径向坐标与实际跨接刚体。已请求等效腿长方向/实际安装轴线、推力和安装位置。为防误用，非play训练拒绝非零恒定力矩近似；现有v161默认0不受影响。保留原探针作为接口证据，不作为气弹簧模型通过证书。

### 24. v160里程碑失败点与v161唯一奖励资格修正

完整model1700验证完成：2.2正负方向yaw0.02574/0.03258、stable0.8940/0.7455，均通过；2.0正向bank3.44663°失败，负向yaw0.05926/stable0.31637失败，四桶各64回合survival均1。与旧候选环境布局不同，不声称严格配对因果，但独立证书明确缺口。旧配置强化yaw罚项及bank scale2只从2.2生效，2.0使用较弱信号。实际奖励代码固定状态审计：起点26→18使2.0 yaw误差0与0.06的奖励间隔0.8953→1.8953、bank0与3.46°间隔2.5660→5.1145；抽查1.5/2.2/2.5/2.9逐位不变，证据retention_reward_eligibility.json。不能据此宣布唯一根因已定。

v160停止于1753（最后snapshot2.0负向yaw0.0335已改善，但正向bank仍3.4536）。v161原checkpoint接续，仅资格阈值移至2.0、10critic warmup适应；其余训练/物理不变，固定L2与nominal直行保护保留。20–30分钟是观察有效性的窗口，不是保证攻克时间；首次Level27 entry或2050轮自动64回合/bucket复测已安排，worker PID234115，训练PID233017。

### 25. 恒定主动连杆扭矩不是纯径向力：闭链虚功核对

用户允许用DR覆盖气弹簧近似误差。已用参考XML默认姿态、真实MuJoCo equality Jacobian消去被动关节速度，投影到“髋到轮心腿长+腿摆角”坐标：每侧-10.775Nm主动连杆力矩等效约40.93N径向力，同时还有-10.775Nm腿摆角力矩。因此不能把固定扭矩直接当成纯径向气弹簧。equality残差1.11e-16、线性闭链速度残差5.14e-17，重构主动力矩正确；证据artifacts/continuation_v161/reference_radial_projection.py/.json。这只是默认姿态局部投影，不是实测推力/力—行程，不能只凭一点就确定全姿态DR范围。下一步采用明确标注的虚拟腿径向力近似，按其作用点施加内力对，并在DR中覆盖不确定参数；不把新范围未经验证就宣称覆盖实机。

### 26. 纯径向力几何与虚功测试完成，未冒称动力迁移完成

**状态：基础力学函数已验证，尚未接入运行模型或确定实机覆盖范围。**

按用户允许DR覆盖近似的方向，gas_spring.py新增radial_spring_force_pair：输入髋/轮心世界坐标与每侧非负伸长力，输出沿腿轴等大反向作用力及各自作用点。明确必须在这些点施力，不能改到body COM而丢失正确力矩；不在函数内猜测推力或力—行程曲线。虚拟腿假设仍需在迁移试验中验证。

tests/test_rmuc_gas_spring.py共4项通过：随机姿态/位置下净力为0、任意原点净力矩为0，功率严格等于F*dL/dt，对固定腿长的切向摆动功率为0；零推力无力，拒绝零长度/非有限/负推力。恒力矩诊断路径仍默认关闭，现有v161物理不变。不能把纯函数测试称成已启用径向弹簧或已覆盖实机。

训练iteration1823仍Level26，冻结权重变化0，2.9retention速度误差0.1458；2.0正向bank3.4567仍缺口。保持2050/首次27entry检查窗口，不依据这个单次快照重启；milestone worker仍等待真实checkpoint。

### 27. 径向助力已接入独立诊断，旧策略联合fullDR探针失败

**状态：机械通路与随机参数采样已运行；稳定性未通过，未用于正式训练。**

- 新增gas_spring_radial_force_n及gas_spring_radial_dr_fraction，默认0。USD读取髋轴在chassis的localPos0和轮轴在wheel的localPos1，不硬改资产。逐子步将两端点变换到world，施加轴向等大反向力，保留作用点；禁止与旧恒关节力矩同时启用。reset时左右独立采样，幅度乘现有DR strength；关闭时不额外消耗RNG。
- evaluator增加径向力/随机幅度诊断参数，记录每个env实际推力与配置，诊断整体passed强制False。源代码、资产几何、径向虚功和防误认证相关14项测试通过。
- 8环境/500steps/零命令/fullDR物理探针：名义40.93N来自参考单姿态投影，±50%仅为探索假设、不是实机覆盖证据；最终样本27.384–60.956N，出现7次tilt，root_min0.19433。报告artifacts/continuation_v161/radial_spring_dr_probe.json。无足够完整回合，不是稳定性认证；发生倾倒足以拒绝直接采用。
- 反思：DR能覆盖近似不确定性，但不能在未适应该新被动力的旧policy上直接开满DR就期待稳定。这次同时改变径向助力与fullDR，不能据此判断气弹簧本身、幅度范围或其他DR的单独责任。下一步需渐进迁移并作分项对照；v161仍无气弹簧、原有训练不变。

### 28. 对径向助力联合fullDR失败做分项对照

v161仍连续到iteration1870、Level26；2.0负向yaw snapshot0.0306，正向bank3.4729，不能据单轮宣称通过。为区分第27条联合探针的原因，启动串行radial_ablation_worker（PID235985）：同一候选、8环境、500步、seed43、左右±50% force DR，先`no_spring_full_dr`再`spring_nominal`。前者检查原策略在fullDR下的基础失败，后者检查仅径向助力对旧策略的影响；两者均诊断整体passed=False，未作为策略证书。进程/报告由radial_ablation_status.json记录，未完成前不重启。\nEOF
ps -p 233017,234115,235985 -o pid,etime
cat artifacts/continuation_v161/radial_ablation_status.json
### 29. 径向助力分项对照完成：旧策略fullDR先天失败，助力也不能直接叠加

同候选、8环境、500步、seed43：`no_spring_full_dr`仅完成1回合即tilt，survival0；`spring_nominal`仅径向40.93N±50%但DR0完成8回合，8次tilt、survival0。证据`artifacts/continuation_v161/radial_ablation_status.json`及两份JSON。结论范围严格限定：原候选fullDR本身有严重失败，径向助力未经适应直接叠加也会破坏旧策略；不能说明合理训练后的径向助力必然不可用，也不能把短探针当完整回合证书。后续必须以新的机械基线渐进训练/验证；当前v161仍保持无气弹簧。\nEOF
ps -p 233017,234115 -o pid,etime
### 30. v161在1897轮显示定向奖励有效但正向bank仍未过门

v161当前训练到iteration1897，2.0负向retention yaw约0.0306、stable0.8077，已较model1700明显改善；2.0正向bank约3.4151°仍超过3°。active yaw0.0499、stable0.4776接近本轮门槛。残差安全日志iteration1899显示immutable/frozen weight delta均0、finite_actions=true，anchor_action_abs_max_delta约0.01997；已确认不是冻结权重被误改。保留model1800/1825/1850/1875和level26 best candidate，不在2050前重复评估或重启，避免破坏连续统计。里程碑worker仍等待model2050或Level27 entry。\nEOF
ps -p 233017,234115 -o pid,etime
### 31. v161继续收窄到2.0正向bank缺口

当前iteration1907/2754，v161仍Level26、无DR。2.0正向retention bank约3.4429°持续略高于3°，2.0负向yaw约0.0306/stable0.8077已达标；2.9直行速度误差0.1479，active正负turn yaw/stable约0.0195/0.9292与0.0461/0.5138。`command_gate_blocked_retention_bank=1`，显示新bank gate确实在阻止进阶，而不是只在日志中记录。继续到model2050或Level27 entry做独立复测，再决定是否仅对正向bank增加rehearsal focus。\nEOF
ps -p 233017,234115 -o pid,etime
### 32. 2026-09-23续接：v161已完整跑完，不能继续用等待解释bank停滞

核实v161和milestone worker均已退出，训练到2753/2754并保存model_2753.pt，无新Level27 entry。2050独立64回合/bucket复测完成：2.2双向通过；2.0正向bank3.43778°失败、负向yaw0.05809/stable0.375失败。故此前单轮“负向已稳定达标”不能当独立稳定掌握证据，应纠正。最终训练日志2.0正向已1077回合、负向264回合，正向bank3.4467°；采样不足不能解释持续缺口，不应盲目进一步提高正向rehearsal比例。

已启动最终2753同条件独立复测，记录artifacts/continuation_v162/final_eval_launch.json，进程307942；没有原样重启trainer。capacity_sampling_audit.json中合成2.0正向输入未见domain0.04上界饱和，但这不是实际rollout状态，不据此排除真实容量/梯度问题。下一步必须依据最终行为及真实动作证据决定修复，不能靠无限追加相同训练。

### 33. 从真实动作区分bank缺口与执行裁剪

2050复测2.0正向平均roll=-0.06219°、bank error3.43778°；原候选roll=-0.04324°、bank3.45673°。两份报告该桶全部6维raw动作饱和比例为0，腿部executed mean与raw一致；因此该稳态分桶没有证据支持“动作裁剪吞掉腿部修正”。实际姿态几乎直立，不能把3.44°描述为已经掌握压弯只是小误差。

已启动final2753的单command诊断：zero对照+4腿通道各±0.04偏置，9组×8replicas、1500步、seed43；记录artifacts/continuation_v162/leg_sensitivity_launch.json（PID308629），最终报告leg_sensitivity_2p0_positive.json。仅测试当前残差量级能否产生bank响应及是否损害yaw/survival，不采用偏置、不开更大动作范围、不计作8回合正式证书。最终2753四桶64回合复测进程仍活跃，未重复启动。\nEOF
ps -p 307942,308629 -o pid,etime
### 34. 最终2753行为与小幅扫描确认bank平坦区

最终checkpoint2753四桶各64回合完成：2.0正向bank3.43749°，负向yaw0.05299/stable0.43050均未过；2.2双向全部通过，四桶survival1。相比2050正向bank几乎无改善，不能再声称多跑相同配置必然解决。证据final2753_turns.json。

小幅动作扫描（单2.0正向，8replicas每case，1500steps）：四腿各±0.04和无偏置共9组全部survival1，但bank3.4285–3.4761°，全部失败，报告leg_sensitivity_2p0_positive.json。这支持当前姿态在局部修正范围内bank响应弱，不证明任何状态相关残差都不可能成功，也不能用不同env布局的极小差值当严格梯度。

已启动更大幅度诊断：四腿各±0.24，左右active连杆差分±0.12/±0.24及对照，共13组×8环境；不改变训练head输出上限，不采用偏置、不放宽gate。记录larger_leg_launch.json，PID309468。目的为确认可达bank姿态及yaw/生存代价，再决定需要新增局部修复head还是其他约束处理；绝不能仅靠扩大残差就宣布解决。所有文件位于artifacts/continuation_v162。

### 35. 更大局部动作扫描找到压弯可行方向，仍须64回合确认

13组×8env诊断完成：active差分[-0.12,0,0,+0.12,0,0]的bank1.7912°、linear0.07749、yaw0.04618、stable0.7320、survival1，各桶checks通过；左hip(action4)+0.24同样bank2.3976、linear0.07239、yaw0.03309、stable0.85535、survival1。对照bank3.45865失败。过大active差分0.24虽bank0.7853但yaw0.0657/stable0失败；rightactive-0.24全倾倒。因此不能简单全局放大残差，动作方向与范围均重要。报告larger_leg_sensitivity.json整体仍diagnostic passed=False。

±0.04范围响应弱，较大且方向合适的修正可离开近直立区域；这为定向局部repair head提供证据，不证明梯度训练必然成功。已启动seed44、64replicas、1500steps的差分0.12确认，文件differential_confirm64.json、launch PID310091。保持原checkpoint冻结、无DR/无气弹簧，不采用小样本偏置为最终策略，不改变gate。若确认有效，再实现速度/yaw局部平滑路由、保护其他已认证锚点并验证完整组合图。所有证据在artifacts/continuation_v162。

### 36. 64回合确认局部差分有效，构造完整组合候选

seed44/64回合差分[-.12,0,0,+.12,0,0]诊断完成：survival1、bank2.03958°、yaw0.04903、linear0.07639、stable0.49101，逐桶checks通过，但diagnostic整体仍False；yaw/stable余量小，不宣称稳健解决。证据differential_confirm64.json。

候选positive2p0_bank_seed_candidate.pt仅新增domain bank内的固定positive_2p0_bank_seed buffer，默认0保持旧checkpoint；支撑区speed(1.75,2.2)，positive yaw>0.25，2.0/.3全量，速度两侧smooth fade。所有named parameters逐位不变，62个其他抽查锚点输出逐位不变；相邻速度/大yaw过渡区明确改变。可训练domain仍为0.04*tanh，不全局放大；固定修正不是新已认证技能，需后续完整复测。证据bank_seed_provenance.json及可复现build_bank_seed_candidate.py。

新增旧checkpoint缺buffer兼容、局部支持、参数越界测试通过；相关17项测试通过，随后含新case的anchor routing8项通过。完整JIT/ONNX导出校验完成，报告bank_seed_export/export_parity.json。无action-bias参数的完整候选64回合物理复测已启动PID310782，bank_seed_positive64.json未完成前不采用、不启训练。所有文件在artifacts/continuation_v162。

### 37. 完整局部候选64回合通过，并检查支撑区与路由保护

不使用action-bias的完整candidate正向2.0/0.3、seed44、64回合复测通过：bank2.01961°、yaw0.04861、linear0.07704、stable0.53595、survival1。报告bank_seed_positive64.json。此前diagnostic偏置确认不是唯一证据，现在buffer修正已包含于完整可导出图；仍不代表2.0负向/fullDR/全目标通过。

新增支撑区网格：1.9/2.0/2.1 × yaw0.275/0.3，seed45、layout2026、每桶16env×6000steps，至少64回合，PID311449；报告bank_seed_transition_grid.json。固定命令网格不能替代连续命令切换，结果未到前不预判。

防遗忘审计以前仅比较named_parameters，固定buffer可能改变最终动作却不被发现。rmuc_residual_safety.py现把rmuc_*路由buffer纳入immutable检查，记录immutable_routing_buffer_abs_max_delta；篡改会保存并停止，新增测试覆盖。启动时取明确基线，期间禁止静默改路由buffer；checkpoint候选构造与新基线仍须独立追溯。测试结果见本轮实际pytest输出。

### 38. 2.0负向yaw对小幅差动轮动作有明确响应

同一候选seed44、7组×16env、1500步诊断完成：对照yaw0.05721/stable0.40029失败；rightwheel+0.01/leftwheel-0.01为yaw0.02882/stable0.71760，±0.02为yaw0.02270/stable0.85326，全部survival1，linear/bank也达标；反向差动恶化。证据negative_yaw_sensitivity.json。支持存在小幅可达修正，不代表固定偏置全速度/DR通用。

已启动±0.02同方向seed45/64env完整回合确认，negative_confirm_launch.json；仍diagnostic，不写正式策略。正向候选相邻速度/yaw固定网格PID311449继续运行，未当作连续命令证书。候选证据与尚缺项记录candidate_status.json。

### 39. 用户要求1–3周期观测/通信延迟与分传感器噪声

最高DR下observation与action下行延迟分别reset采样整数1..3步，当前dt0.008对应8–24ms；低强度以strength概率启用该范围，非启用环境0步。两段独立，不把它误说成闭环总滞后最多3步；当前每回合固定，没有实现逐包抖动/丢包/乱序。action FIFO位于外环合成动作之后，实际执行动作用于previous-action观测；原PPO采样动作/logprob不重写，增加了环境隐藏状态。reset清队列，启动未到达指令用零归一化动作（并非关闭全部电机PD）。

IMU初值（工程假设，未实测）：gyro bias±0.02rad/s，bounded随机游走0.0005rad/s/sqrt(s)，white std0.005rad/s；accel bias±0.1m/s²/white0.05m/s²只等效到gravity估计并归一化，因25D无原始加速度通道，不宣称完整IMU滤波模拟。编码器角度std0.001rad、速度0.02rad/s，forward velocity估计std0.01m/s。保留当前command和实际执行动作槽，yaw error与同一含噪gyro重算，延迟速度误差按当前command重基准；历史clip意味着速度估计仍有删失，非精确重建。外环仍读仿真真值状态，最终部署反馈噪声覆盖仍需核对，不能据此宣称全链路传感器鲁棒性。

代码：configs/rmuc_wheel_leg.yaml sensor_model，env_cfg/env.py，observation_noise.py，evaluate_rmuc_grid.py报告新参数。physics_only诊断显式禁用新传感器路径，避免A/B口径错误。6项传感器/实际action FIFO/奖励测试通过；原生16env/100steps fullDR smoke完成，无证书，报告sensor_cycles_smoke.json实际obs delay=[2,2,1,3,...]、communication=[1,2,2,2,...]均在1..3。测试脚本test_rmuc_sensor_model.py。旧5–25ms插值方案已撤换，不再使用。

另：negative_confirm64和bank_seed_transition_grid已完成，结果来自新契约修改前启动的进程，必须按旧噪声/通信语义解释，不能作为本轮新DR证明。下一步读取完整结果并继续策略修复/新DR训练。

### 40. 两个修正作为学习初始化，启动v163而非宣称全通过

negative_confirm64：seed45/64回合survival1、linear0.03894、yaw0.03738、bank0.74668、stable0.57792全部checks通过。正向邻域6桶通过5桶，2.0/.3在seed45仍yaw0.05268/stable0.36664失败，说明余量不足，不能把前一个seed通过包装成稳定学会。

新增negative_2p0_yaw_seed buffer，旧checkpoint默认0；与positive bank seed同速度局部支撑、独立负yaw门，原模型参数逐位不变。CPU输出差与已测wheel差分吻合。22项相关测试通过，新增负向局部/旧checkpoint兼容case后anchor routing9项通过。artifacts/continuation_v163/provenance.json记录hash及初始化限定。

v163从seeded2753继续actor/critic/optimizer，固定L2参考显式改为新初始化，10critic-only校准，保留所有门槛和nominal直行保护、参数/路由buffer审计。最新sensor model配置已加载，将在原课程DR阶段生效；原DR结果不冒充新通信/传感器契约通过。PID314868已出现真实迭代与审计，非仅启动文件。气弹簧默认关闭、wheel5.5Nm/0.1仍未完成硬件迁移。后续须多seed复测并验证新DR，不再以无限小样本偏置搜索替代正式学习。

### 41. v163修正已在训练内生效，等待完整里程碑证据

10critic warmup在2763结束；早期snapshot2.0正向bank1.8855°、负向yaw0.0432、2.9straight speed error0.1405，说明动作修正已进入环境，但完整回合计数不足前不能宣布进阶。2768安全审计immutable weights/buffers均0变化、finite_actions true。训练PID314868真实存活。

已安排首次Level27 entry或model3000独立双向验证，worker PID315552，artifacts/continuation_v163/validate_milestone.py与milestone_validation_status.json；当前waiting_checkpoint，不是完成。完整双向seed策略导出一致性验证通过：JIT最大误差0、ONNX1.79e-7，见export/export_parity.json；只认证raw normalized actor数值，不代表外环/延迟/actuator链路部署通过。保持连续学习，不提前根据单轮snapshot重启或改参数。

### 42. 新DR的日志必须与实际通路区分，不能读取遗留字段误判

审查发现旧domain_observation_noise_std/domain_observation_delay_steps仍是旧模型buffer，新传感器通路实际使用_sensor_delay_steps/_communication_delay_steps与物理单位bias。现新增训练指标sensor_model_enabled、sensor_observation_delay_steps/ms、communication_action_delay_steps/ms、imu_gyro_bias_abs_max_rad_s、imu_accel_bias_abs_max_mps2；独立evaluator失败trace新增sensor_model_state真实延迟/bias。编译通过；不重启v163仅为加载日志。当前进程仍无新增日志字段，读取其params/env.yaml和新评估report的sensor_model字段核对实际契约，不能声称live已输出新指标。

2780训练snapshot：2.0正向bank1.8187、yaw0.0513，负向yaw0.0281，旧2.9直行误差0.1448；该桶正负完整episode计数约10/8，未达到64。持续积累和学习，不据早期均值宣称进阶。原训练/worker进程均存活。

第42条补充验证：test_rmuc_sensor_model新增直接执行env实际reset采样代码的测试，1024环境fullDR只采到1/2/3步，观测/指令独立；bias在配置界内，DR0清偏差与延迟；仅reset指定env的action FIFO，未reset环境保持不变。4项传感器测试通过。v163 iteration2793审计仍immutable weights/buffers=0，milestone worker实际存活且waiting_checkpoint。该测试不修改训练参数、不证明行为鲁棒性。

### 43. 早期gate审查：不要把剩余阻塞只说成回合数不足

v163完整日志块2800–2816显示，retention readiness不足之外，active yaw/stable、retention yaw/stable/survival也随时间阻塞。2816的2.0正负episode约37.5/36，仍未满64，不能宣称“只等计数”。日志writer可能正在输出当前block，读取时仅使用含Iteration time的完整块，避免把尚未写出的指标当0或不存在。结构化证据artifacts/continuation_v163/early_gate_audit.json保存最近20完整块。进程/worker持续存活，未重启/改配置，等待既定3000/首次27entry独立评估。

### 44. 独立评估加入启动配置与代码/资产溯源

evaluate_rmuc_grid.py现在rollout前保存关键10个源文件/配置/USD的SHA256，并把实际生效的command/domain/wheel/gas-spring/observation等primitive配置写入runtime_contract，防止后续代码修改把旧结果错标成新契约。实际1env/10step接入测试生成provenance_smoke.json，含10hash与205运行字段，新sensor delay_max_steps=3，passed=False（无完整回合）；不把这种接入测试算技能验证。现有旧报告不补写假造启动hash。

v163保持原参数连续运行，2833完整日志2.0正负各48回合，正向bank2.0649/yaw0.0472；计数未够，未宣称进阶。训练及milestone worker已核实存活，新增溯源不影响训练，不重启。

### 45. Level26瓶颈取得真实进阶证据，独立验证尚在进行

v163在2866保存model_level_27_entry_2866.pt并进入2.4m/s、±0.15yaw。active双向yaw max0.035、stable min0.708、bank max1.936°、survival min0.971；retention speed max0.144、yaw max0.046、stable min0.595、survival min0.985。未放宽gate。突破方式为物理偏置扫描确认可达方向→64回合诊断→局部平滑固定修正初始化→保留冻结技能并继续domain/stage学习，而非单纯加训练时间或增大全局残差。独立entry checkpoint2.0/2.2双向复测PID318756已启动，不能将进阶等同最终稳健掌握。

32个关键base/expert/修正buffer张量与初始化逐位一致，level27_entry_integrity.json；其余冻结head由实时全参数审计检查。进阶后样本计数归零为新阶段正常行为，必须先查command_level与entry checkpoint再解释。训练继续不停止，仍保留全速度最高DR、3Nm持续轮端与径向气弹簧迁移、最终部署目标。

### 46. 同一entry checkpoint四桶独立通过，继续核对小余量

Level27 entry2866的seed43 nominal独立复测四桶(2.0/2.2×±0.3)各64回合全部达标、survival1。此前长期卡住的2.0正向bank已2.0033°，但yaw0.04924/stable0.47053接近阈值；2.0负向yaw0.04305/stable0.53595。不能因为进阶+一次独立通过就称全速度/充分鲁棒掌握。已在同checkpoint上启动seed45/layout2027重复完整验证，PID320379，level27_seed45_launch.json。训练保持Level27连续，不为复测停止。通过结果、源hash/实际配置在milestone_turns_nominal_seed43.json，后续只有受影响行为或小余量理由才重复检查，避免无目的复测。

### 47. 同checkpoint第二seed通过，后续验证覆盖新技能

entry2866在seed45/layout2027独立四桶各64回合全部checks通过：2.0正向yaw0.04692/bank1.88437/stable0.61821/survival1；负向yaw0.03185/stable0.71551/survival1；2.2正向yaw0.02694/stable0.85796/survival1；负向yaw0.04463/stable0.54077/survival0.96875。两份报告checkpoint hash一致，汇总level26_repeated_nominal_evidence.json，范围仅nominal固定命令，不把两个seed说成最高DR/部署认证。

为避免只测旧技能而漏新技能，新增validate_level27_skill.py：首次Level28 entry或model3200时独立2.4×±0.15、64replicas/1500steps、每桶64回合。worker PID321130已核实等待；2919训练仍Level27，当前blocked hold/retention readiness及部分retention yaw/stable/survival，不宣称仅等待时间即可进阶。保持连续训练不重启。

### 48. Level27生存/retention仍有阻塞，持续观察而非放宽

v163从2938持续到2964，训练与新技能验证worker(PID321130)均经ps核实存活，无重启。阶段hold阻塞在后续日志消失，但2964仍有active turn survival、retention ready/yaw/stable/survival阻塞；Level23 positive记录的survival EMA0.9413低于0.95。不能因之前nominal固定policy通过就忽略训练随机探索失败，也不能在冻结权重不变时断言行为永不退化。完整snapshot保存artifacts/continuation_v163/level27_wait_snapshot.json。当前继续原课程到entry/3200检查点，不根据一轮EMA改噪声/生存gate或跳关。

### 49. 直行生存波动未见均值策略或探索幅度恶化证据

比较v163初始化、entry2866、model2950的Gaussian std：腿固定0.004，轮从约0.006155/0.006073到0.006106/0.006000，未上升。当前checkpoint在1024个合成观测(target2.9/yaw0)上动作均值与初始化及base逐位一致，max delta0；证据exploration_snapshot.json、straight_mean_integrity.json。合成探针不是行为证书，不能据此断言随机策略不会跌倒；但不支持“残差篡改直行或增大std导致生存下降”的解释，未据此改LR/降std/放宽gate。

继续实测等待至2999，Level27未进阶；active gate无阻塞的时段已出现，但retention readiness和yaw/stable/survival仍波动，不能宣称只有计数。训练/worker持续存活，不重启，待entry28/3200复测。

### 50. Level27进阶及新技能独立验收通过，继续Level28

3051真实进阶Level28并保存entry；原Level27(2.4/±0.15)双向独立64回合通过，survival均1、speed error0.02159/0.02394、yaw0.02185/0.01991、bank2.12960/0.47836°、stable0.87202/0.85708。证据level27_skill_2p4_small_turns.json与train日志，不只依赖课程编号。原定3200检查点未到就因entry28触发评估，未重复启动。

当前2p4_large训练head开放，冻结技能/路由审计3064全0变化。验证Level28大转弯的worker325742等待entry29或3400；正常连续训练，不重启、不降低gate。新技能nominal通过不代表Level29DR、最高DR或3Nm/gas-spring部署通过，最终仍需同策略完整验收。

### 51. 新1–3周期通信/观测噪声契约最高DR已做实测，未通过

用户强调最高DR是硬验收条件。本轮对nominal已通过的同entry3051、同seed43、同2.4/±0.15做strength1对照，未改训练或调低DR。正向74结束回合survival0.56757、yaw0.07007、stable0.18004；负向83回合survival0.43373、yaw0.07493、stable0.03878；coverage亦失败。linear/bank/height通过不能补偿survival/yaw失败，不以完成episode数多于64误说存活回合充足。

报告level27_skill_full_latest_dr.json明确观测/指令延迟实际值集合各为{1,2,3}，启用新IMU偏置/噪声和编码器噪声，包含运行参数与代码/USD hash。最高DR已经测试不等于通过，更不等于已在该强度训练完成；当前训练仍Level28 DR0，后续按课程适应并必须重验。外环真值状态、5.5Nm/0.1与未启气弹簧仍非最终部署条件，本次不声称全链路硬件契约验收。

### 52. 用户认可低DR缓慢推到高DR，核对已有执行机制

实际运行params/env.yaml确认start_level29、full_level46、override=-1。代码strength=(level-start+1+transition_fraction)/19：29为1/19→2/19，30为2/19→3/19，以此至46为18/19→1。质量/摩擦/阻尼/增益/传感器噪声幅度按strength缩放；延迟启用概率按strength上升，启用时仍随机1–3周期，fullDR全部启用。物理参数只在reset采样，不在回合中突变。

反思：阶段内transition按时间推进，不应误称已经实现“每次加一点都先独立验收”的自适应DR控制；跨阶段进阶才受全部gates约束。当前已有低起点，无需为用户认可此方向重启训练。若后续持续失败，基于具体强度证据巩固，不跳过retention、不偷偷下调最终DR，最终同策略全速度最高DR条件不变。

### 53. 最终目标明确包括原地0–3rad/s双向旋转，旧课程覆盖不足

用户要求原地转弯速度0–3rad/s可快可慢，并适应大部分行驶速度下的转弯。按左右双向[-3,3]解释。核对发现当前课程无vx0原地转弯，最大yaw仅0.45（高速多为0.30）；keyboard max0.30。obs yaw error clip2会在静止请求3时丢失目标，noise/delay helper也硬编码clip2；奖励excessive_yaw从1rad/s起惩罚，与新目标矛盾。不能只抬命令上限，不得把原课程全过说成完成新目标。

记录spin_requirement_audit.json；保持25D/6D和冻结旧技能，需要兼容的yaw专用编码/命令相对超速处罚、追加原地渐进双向课程和漂移/启停换向指标，并测试从原地到行驶转弯的连续切换。行进v与yaw组合受附着/倾角/3Nm限制，应给实测能力包络而非默认3.2m/s同时3rad/s可行；不静默缩减已承诺全速度原有任务。当前训练配置未改，先继续Level28，后续追加这一明确任务。

### 54. 原地±3rad/s的观测/奖励冲突修复为兼容选项

新增command_yaw_observation_error_clip（默认沿用2，不改变旧checkpoint观测），只扩展yaw槽，不改速度误差±2或25D/6D；原地±3与反向切换可显式设置6。command-consistent噪声、物理sensor noise及延迟重构均传入独立yaw clip，避免只改env一处仍被helper截断。

新增command_yaw_overspeed_relative_to_target，默认False保持旧奖励；启用后超速起点max(1,abs(target_yaw))，合法±3不再受旧1rad/s阈值处罚，低yaw原有规则不变。实际奖励AST测试证明旧±3处罚2、新处罚0，而低yaw超速项一致；噪声/延迟测试证明反向±3命令可重构且速度clip仍2。9项sensor/reward测试与代码编译通过。运行v163未启用这两个新选项，不热改当前训练。

这是接口/奖励基础修复，不是原地旋转能力完成；还需零速漂移指标、新旋转课程、局部专家/残差支持、连续启停换向与最高DR最终验收。v163本轮3158仍Level28，继续既定训练/entry29或3400独立复测，不为这些未来spin选项重启。

### 55. Level28当前缺口是大转弯生存和跟踪稳定性，未进入DR

保持v163连续观察3167–3189，仍Level28，无Level29 entry。3167双向survival0.9050/0.9467；3178约0.9050/0.9167，yaw max0.0659、stable min0.385，不能以早期bank/速度合格宣布阶段掌握。3189安全审计immutable weight、frozen head、routing buffer delta均0；故没有base/head被误解冻的证据，但组合策略行为仍需要巩固。按既定首次entry29/model3400做独立2.4大turn复测，worker325742真实存活且waiting_checkpoint。未因两次snapshot修改LR、奖励、DR或gate，也没有启动重复fullDR评估。

### 56. 后期自转扩至10rad/s，超速0.5宽限与强欠速奖励落实

用户明确先完成全速度行走压弯，后期小陀螺提升至10rad/s。目标按双向±10，原±3作为中间能力。新增默认关闭spin_asymmetric_reward_enabled：仅零平移/非零yaw命令启用，按实际yaw*sign(target)计算；超速[0,0.5]的yaw误差与excessive_yaw均为0，避免主yaw tracking又扣回宽限带；spin stillness只看平移和roll/pitch角速度，不用yaw抵消目标。欠速额外有界强罚，默认weight10、scale0.5rad/s；目标0沿用停止要求。姿态/能耗/扭矩等约束未取消，不能解释成整份奖励都不受真实动力影响。

实际_get_rewards测试覆盖±3/±10、+0/+0.25/+0.5奖励相同、+0.6更差、欠速0.5至少额外10分、反转更差、开启此选项不改变行进转弯奖励。旧局部AST测试因新增spin_mask依赖先报NameError，补齐关闭spin的fixture后11项sensor/reward测试全部通过，代码编译通过。不是训练/行为已通过。当前v163未启该选项，保留先行走后自转顺序。后续spin独立gate也需支持非对称误差，不能拿当前严格yaw±0.05直接当自转规则；全速度行驶gate暂不放宽。

### 57. 按先行走后自转继续Level28，生存率尚未通过

v163连续运行3222–3246，不启用新spin奖励，也未重启。3222双向survival0.9065/0.9151，3235回升0.9188/0.9283但均低于0.95；yaw/stable仍波动。最近50完整日志块已保存level28_survival_trend.json，不能把一轮回升当突破。3246仍有survival、completed yaw及部分retention yaw/stable阻塞；base/frozen heads/routing buffers审计全0变化。Level28独立worker325742等待entry29或3400实际checkpoint。遵守先全速度行走压弯再拓展自转10rad/s，最高DR硬验收不变。

### 58. v163最终独立复测生存未改善，停止原样追加训练

中断后核对trainer和worker均已终止，v163跑完至3753、仍Level28。3400报告正向13tilt、survival0.81944，负向2tilt/0.96875；最终3753独立报告正向0.80556、负向0.89063，两侧coverage/survival失败。不能把steady tracking/bank通过说成稳定学会。

3400捕获15条失败轨迹，13正向；部分正向在389/402/412step即tilt，末态pitch>85°（近奇异姿态不能据此认定坐标/符号错误）。20°之前采样中轮动作已接近或达到1，说明姿态恢复余量值得检查，尚不能确定因果。turn_failure_analysis.json保存真实前终止值和采样限制。

已针对3753构造no_level28_head_candidate.pt，只清零index0(2p4_large)的共享/正/负输出层，共6个tensor，其他状态保持原样；同seed43/各64replicas/1500step复测已启动PID526935。原checkpoint保留，候选仅用于因果隔离，不采用为训练/部署，不改物理或gate。依据对照决定后续处理，不再无证据追加相同1000轮。证据位于artifacts/continuation_v164。

### 59. 关闭Level28 head不能恢复生存，保留原图并检查恢复余量

同3753基线、seed43、2.4±0.3、每桶64replicas/1500steps对照：关闭index0 output后正向survival0.74026（原0.80556），负向0.9375（原0.890625），两侧coverage/survival仍失败。不同终止导致后续reset抽样不同，不能把比例差当精确配对因果效应；足以否定“仅删除head即可解决”的候选。报告no_level28_head_large_turns.json，未采用该候选。

已启动原3753完整图的轮共同动作±0.02/0.04/0.06及零对照，两个yaw方向各8replicas、1500steps，用于检查倾倒前高轮动作/速度-pitch耦合对生存影响。整体诊断不算通过证书，不改外环反馈符号、USD或gate，不仅凭8回合结果部署。证据wheel_common_launch.json。仍无正式trainer，避免原样追加无效训练。

### 60. 共同轮动作扫描出现可行恢复方向，等待独立64回合

3753基线2.4±0.3、每case8replicas诊断：双轮+0.02/+0.04在两侧该批小样本均survival1且各checks通过。+0.02正向linear0.07551/yaw0.02342/stable0.81062，负向0.06175/0.02291/0.74833；-0.06正向survival0.2222，+0.06双向速度/稳定失败。前终止轮动作饱和不能推出“减少共同驱动力就会更安全”，控制耦合要以A/B证明。不同布局与reset样本不能作为严格配对效应估计。

已选择较小+0.02进行seed44、双向各64replicas/1500steps/min64episodes确认，common002_confirm_launch.json PID529512。所有证据artifacts/continuation_v164。仍为外部偏置诊断、整体passed不能当证书；不改变原checkpoint、不增加电机额定力矩、不解除冻结图、不提前恢复训练。等待结果再决定局部初始化或状态相关修复。


### 61. +0.02共同轮偏置大样本未解决正向倾倒，不采用为正式修复

common002_confirm64.json已完成：seed44、2.4m/s、±0.3rad/s、双向各64replicas、1500steps。正向68个completed episodes、7次tilt、survival0.89706，coverage及survival失败；负向64回合、2次tilt、survival0.96875，各行checks通过。整体仍为诊断且passed=false。小样本两向通过未在独立较大样本复现，不将偏置写入正式策略，不称瓶颈攻克；不同seed和reset抽样不能直接当严格配对改善幅度。

重核用户自转要求与当前代码：spin_asymmetric_reward_enabled默认关闭，后期专用阶段再启用；同向超速0–0.5rad/s无yaw奖励扣分，欠速另加weight10、scale0.5的有界强罚，零目标仍停转。目标顺序为先全速度行走/压弯，再双向±3逐渐扩至±10rad/s；奖励单元测试不是旋转训练完成，最高DR仍须同最终策略实测通过。


### 62. 对剩余+0.04候选做同seed/规模确认，限制继续盲扫

+0.02大样本9条tilt轨迹仍显示pitch约16–20°前双轮动作接近/达到1；只说明恢复余量相关，不证明外环符号或推力方向错误。启动common004_confirm_launch.json（PID531917），保持3753、seed44、2.4/±0.3、64replicas/1500steps及相同物理配置，只改外部共同偏置为+0.04。若失败，不把常数偏置当已验证修复；需转向状态相关控制/学习证据。启动时确认无旧评估或正式trainer在运行，未重复启动。inclusion顶部同步已完成的head消融和+0.02失败，避免后继沿用过时“等待”状态。


### 63. +0.04大样本同样失败，常数共同轮偏置路线未获支持

common004_confirm64.json已落盘且PID531917结束。正向survival0.88235、8次tilt、coverage失败；负向survival1且checks通过。正向steady linear0.12048、yaw0.02298、bank2.59232°、stable0.85518，进一步说明只看幸存窗口平均会高估安全性。与+0.02相同seed/规模但仍受分叉reset影响，不宣称严格配对因果幅度；不采用两者，不追加更大常数偏置。common004_decision.json保存决定及失败episode step。

检查_reset_idx：nominal使用默认root/joint pose与velocity，reset_speed_bootstrap_enabled默认false；不能把早期倾倒武断归因为随机起姿或bootstrap。下一步需要观察从起步到失稳的完整状态与组合动作贡献，选择状态相关修复或训练干预；目前没有已验证的新修复，正式trainer仍未恢复。未改奖励、gate、物理资产或策略权重。


### 64. 记录动作输入并精确重放，发现部分失败态base已超出最终动作幅度

评估器failure trace新增产生该步动作的25D输入观测、speed/propulsion/pitch/yaw外环输出；物理状态为执行后的时刻，观测为执行前，明确此时序，不混作同时状态。baseline44_attribution.json使用原3753无偏置、seed44、双向各64replicas完成实测。attribute_failures.py对12个失败env共36个采样状态恢复完整actor，各加性贡献之和与actor输出、GPU记录raw action逐项比对通过，保存failure_action_attribution.json。

部分pitch约10°失败状态中base轮输出达1.25–1.45，而各可学习轮修正通常仅百分之几，说明仅冻结权重并不能保证当前状态下base安全，且原始输出超过最终clip可能吞没小修正。只是动作归因，尚非根因或因果证明；不能将所有倾倒归咎于残差篡改，也不能用此证据全局改base。

新增默认关闭--base-wheel-clip诊断：只在评估进程对base MLP输出轮槽限幅，腿槽/参数不改，最终passed强制false。base_wheel_clip090_launch.json启动同seed同网格对照，限幅0.9以检验残差/转向余量假设，未改正式checkpoint或训练配置。该开关不是部署修复，若诊断失败须保留证据。


### 65. base轮输出预限幅0.9恶化，拒绝把饱和关联当直接修复依据

base_wheel_clip090.json已完成（PID535129结束），正向survival0.52128、45次tilt；负向0.875、8次tilt；均coverage/survival失败。原图同seed44正向0.80556/负向0.890625，预限幅没有提供有效恢复。不能为了给yaw留动作余量直接牺牲平衡推力，也不能将饱和现象等同于过大驱动力根因。此诊断不采用、不写入正式策略；下一步需考察倾倒前腿姿/起步过渡及训练的恢复状态覆盖，不能继续单靠常数轮修正。

本轮代码改动仅scripts/evaluate_rmuc_grid.py的观测/外环轨迹记录与默认关闭诊断开关，py_compile及真实128env两轮实测完成；离线36个动作重放最大误差4.768e-7。正式训练未恢复，全目标未完成。seed43/44 nominal基线各项终止计数一致，nominal默认reset不随机，不能把只换seed当独立随机物理验证。


### 66. 转向腿姿验证，并发现旧2.2修复head在2.4正向持续叠加hip动作

leg_posture_positive_scan.json：原3753、seed44、2.4/+0.3，7case各16replicas/1500steps。active共同±0.04和差分right−0.08/left+0.08该批行checks通过；差分−0.08 bank1.01355°、linear0.04481、yaw0.02412、survival1。反向差分bank3.90708失败，共同±0.08生存失败。不同case布局不同，未采用任何小样本通过候选；不能又重复“8/16样本通过即已修复”的错误。

完整动作归因中，正向失败态stage15（2p2_large_repair）给双hip约+0.21/+0.21，负向约−0.02/+0.02；其speed gate在2.2以上一直为1，非局部速度岛。这是需要隔离的旧head外推关联，不是权重被改或已确认根因。新增默认关闭--disable-stage-head诊断，forward hook仅清零选中head输出、保留权重，整体passed强制false；启动disable_stage15_launch.json（PID537694），原3753/seed44/2.4±0.3/各64replicas。先验证此关联再决定局部路由或腿姿候选，保持正式checkpoint不变。


### 67. 旧stage15消融改善生存但牺牲bank，不能直接删除

disable_stage15.json已完成：正向survival0.96923（2tilt），但coverage失败、bank4.17185°失败；负向survival0.96875（2tilt），coverage及yaw0.05163失败。较原图倾倒显著减少，却没有同时保留压弯/跟踪，因此不是完整修复。说明旧head可能提供压弯同时带来恢复代价，应做分量或局部路由验证，不以生存单指标删除已冻结技能。

启动disable15_active_difference_launch.json：关闭stage15诊断基础上叠加右active−0.08/左+0.08，仅2.4/+0.3、64replicas、seed44/1500steps，检验独立腿差分能否恢复bank且保留生存。仍是组合诊断，不改变正式策略，不能证明2.2 retention或负向已修复。


### 68. 删除stage15再加active差分达到存活，但未恢复压弯

disable15_active_difference_positive64.json完成：64replicas正向survival1、0tilt，coverage通过，linear0.09933、yaw0.02988、stable0.87233；bank3.88314°仍失败。因此该组合不能采用或称掌握。此前保留stage15的差分小样本bank1.01355°不能直接外推到删除head后的图，腿姿响应存在组合依赖且布局也不同。下一步优先考虑保留部分旧head贡献的局部权衡，而非全删或用bank gate豁免；仍须完整候选和retention实测。所有checkpoint参数仍未修改。


### 69. 保留半量stage15恢复压弯，但生存仍未达标

half15_active_difference_positive64.json完成：stage15乘0.5，active差分−0.08/+0.08，2.4/+0.3各64replicas。survival0.92424、5tilt、coverage失败，bank2.35339°通过；linear0.06772/yaw0.02821/stable0.83738通过。保留部分贡献存在bank/survival权衡，但未确认有同时通过区间。评估器新增--stage-head-output-scale（有界[0,1]，仅指定head时有效，默认0保留已有全消融语义），整体diagnostic仍不能认证。

基于0与0.5两端证据启动scale030_head15_launch.json，只将head比例改0.3，其余一致；不是全局新路由，也未修改正式checkpoint。若候选可行，仍需局部支撑保留2.2技能、完整组合图和双向/retention实际验收后才可采用。


### 70. 30%旧head加腿差分正向诊断通过，生成局部完整候选

scale030_head15_positive64.json：2.4/+0.3、64replicas，survival1/0tilt、coverage通过，linear0.07915/yaw0.02793/stable0.85875，bank2.95088°仅0.04912°余量，全部行checks通过但仍diagnostic。不能将小余量单网格视为稳定掌握。

local_2p4_positive_candidate.pt从3753生成，所有named parameters逐位不变；仅stage15的positive_2p4_suppression=0.7及domain的positive_2p4_bank_seed=[−.08,0,0,+.08,0,0]。平滑支撑speed(2.2,2.6)、正yaw(.15,.3)，2.4/.3全量；2.2/2.6、负yaw、小yaw锚点不受影响，过渡区明确改变。旧checkpoint新buffers默认0并严格校验边界，冻结buffer审计仍覆盖。54个其他命令锚点合成probe逐位一致，诊断中心动作差匹配；这是输出验证而非retention物理证书。

21项routing/safety/anchor-guard测试通过，完整JIT/ONNX导出最大差1.788e−7。无诊断参数的完整候选双向64replicas实测已启动local_candidate_launch.json（PID543221），负向按原图未修复，不能期待整体通过。待结果选择续训或补充修复，不自动部署、不覆盖正式checkpoint。相关源码scripts/rsl_rl/rmuc_command_adapter.py、tests/test_rmuc_anchor_routing.py；provenance与构建脚本保存在continuation_v164。


### 71. 完整候选跨环境规模未复现通过，转入有边界的学习试验v165

local_candidate_large_turns.json（128总环境，正负各64）：正向survival0.84286/11tilt、coverage失败，bank2.98140°；负向0.890625/7tilt，与原图负向一致。先前诊断为64总环境/单方向，布局不同，不能把未复现直接归因为候选构建错误；512个合成中心观测动作匹配及完整导出已通过，但均不能证明闭环在另一布局稳定。正向候选仍非验收策略。

不再无限追加常数扫描。v165以此候选作为明确标注未通过的实验初始化，固定L2参考到该初始化，stage LR从2e−5升至1e−4让当前head适应新的局部姿态；其余沿用v163，128env、128rollout、1epoch、domain/critic1e−4、10critic warmup、rehearsal0.60、腿std冻结0.004、轮floor0.006、全gates及nominal2.9保护不变。仅200updates，Level28起、32entry停止；实验分支不替代control默认策略。launch位于artifacts/continuation_v165/launch.json。

PID544451已实际运行至3760（critic warmup阶段），residual_safety中immutable weights、routing buffers、frozen-step delta全0，actions有限；不是已改善生存的证据。run_dir=logs/rsl_rl/rmuc_wheel_leg_course/2026-09-23_19-23-45_local_head15_posture_learning_v165。下一步warmup结束确认actor恢复更新，再在实际新checkpoint/进阶时做同128env双向评估，不能只凭训练return宣布突破。

补充实时验证：v165在3763结束critic warmup，3770已恢复actor更新，anchor动作最大差0.002198、L2约2.315e−7；immutable/frozen weights与routing buffer变化仍全0，actions有限。训练PID544451持续存活，不以预热期间零更新误认为学习正常。


### 72. 用户要求连续随机命令泛化，接入独立训练cohort并完成真实smoke

核对旧实现：课程点/rehearsal点离散，只有阶段过渡插值，不等于随机覆盖速度域。新增command_continuous_sampling_fraction默认0、上限0.25；reset按总体配额从非rehearsal环境分配，回合内固定命令。sample_opened_commands在当前已开放阶段的速度min/max间连续均匀采样，yaw在相邻速度各自最大已开放yaw的较小界内双向均匀采样；重复速度取最大已开放yaw，未开放自转±10不混入高速行驶。该保守包络可能在某些区间只采直行，不是全部物理可行转弯包络；已开放不等于已认证。

随机cohort与rehearsal不重叠，不计入active rolling/episode tracking/completed survival及双向gate，不代替固定锚点证书，也不享受bootstrap。要求command-consistent reward/feedback，腿尺度按实际采样speed插值，25D/6D不变；episode内不随机跳命令。默认0路径不采样随机数，现有v165运行未启用，不热改。

33项采样/真实gate mask/奖励/bank/sensor/route测试通过，py_compile通过。16env、2updates真实训练smoke成功，日志continuous_sampling_smoke.log，实际2/16=0.125配额（目标0.15按整数舍入），出现2.191/−0.155、1.277/0等非课程点；finite actions，冻结weight/routing delta0，保存model3755。证据continuous_sampling_smoke_evidence.json。不声称已泛化或正式最高DR通过，后续正式续训可启0.15，仍需随机网格/命令过渡验证。代码env.py/env_cfg.py、新command_sampling.py与test_rmuc_command_sampling.py。

原v165仍持续训练，3775参数diff只涉及当前stage0/domain speedheads/std，base逐位不变。3800完整checkpoint双向64replicas独立评估已启动model3800_eval_launch.json，未重启训练。


### 73. v165中途3800独立评估仍失败，固定最终评估防止无界续训

model3800_large_turns.json完成，2.4/±0.3、各64replicas、128总环境、seed44、DR0：正向survival0.81429、负向0.921875，两侧coverage/survival失败；正bank2.97330°、yaw0.02828、linear0.08306，负bank0.81463°、yaw0.04132、linear0.05757。steady指标合格不能补偿完整回合失败。与初始化正向0.84286/负向0.890625相比，没有双向突破证据，不据此宣布局部修复已成功。model3800_decision.json保存结论。

训练仍Level28，3825审计immutable weight/routing变化0、anchor最大动作差0.004799。此前3775真实checkpoint差分只发生在stage0/domain speedheads/std，见checkpoint3775_integrity.json。允许原定200updates完成，不因中途一轮失败马上重启。新增evaluate_finished_run.py与worker PID551989，绑定trainer PID544451及Linux start ticks身份；只在该进程确实结束且model3953存在后评估最终双向网格，缺checkpoint/缺报告则明确失败、不重训、不覆盖。final_evaluation_status.json为原子更新状态；worker及trainer已ps核实存活。

连续随机采样已完成代码/smoke，但v165仍默认0，计划在最终checkpoint证据明确后启下一段训练；不把未启用的功能说成已完成泛化。全速度最高DR/硬件迁移/部署仍待完成。

### 74. 修正最终评估worker的checkpoint编号假设

训练命令每25轮保存一次，v165末轮3953不保证存在同名文件；原worker若只等待model_3953.pt会在训练结束后错误报告缺checkpoint。已改evaluate_finished_run.py：绑定的trainer身份校验不变，进程结束后按实际model_*.pt数字选择最高编号（预期通常3950），只评估该真实最终保存点；无任何checkpoint才报告失败。修复已py_compile验证，未重启训练或worker。


### 75. 撤回第74条未核实的末轮保存假设，恢复精确最终checkpoint要求

直接inspect当前安装的rsl_rl.runners.OnPolicyRunner.learn确认：循环结束后另有无条件最终self.save(model_{current_learning_iteration}.pt)，不受save_interval25限制。因此正常跑完v165确实应保存3953，前条“通常3950”的推断错误。已撤回最高编号fallback并恢复原精确3953要求，避免把异常提前退出留下的旧checkpoint冒充最终结果。当前运行worker551989本来已加载原代码，磁盘修改不会热更新它；本次恢复后运行/磁盘一致，无需重启。训练PID544451仍真实存活，继续原计划。此反思明确记录，不删除错误历史。


### 76. v165最终负向通过、正向未过，按用户要求开始连续命令v166

v165正常完成200updates并保存model3953，最终审计immutable weights/routing delta全0，anchor动作最大差0.008216。绑定worker实际完成final3953_large_turns.json：负向survival0.953125、coverage及所有checks通过；正向survival0.88235、coverage失败，bank3.03654°失败，linear0.08529/yaw0.03104/stable0.81918通过。相比3800正向0.81429、负向0.921875有改善，但仍未达双向完整要求。尤其训练3927的双向survival EMA超过0.95不能替代实际固定模型验收。

下一段v166从3953继续，启用command_continuous_sampling_fraction0.15，128env预计19个连续cohort、rehearsal0.60、其余固定active；维持stage/domain/critic LR、同L2固定参考（v164初始化）、10critic warmup、所有gates与冻结保护不变，不重置已有学习。最多300updates，仍28起/32entry停止；当前physical5.5Nm/gas spring off及DR课程边界不变，非最终硬件/最高DR验收。启动记录artifacts/continuation_v166/launch.json；启用的是实际训练选项，不再仅停在smoke。启动后须检查真实采样cohort和安全审计。

v166已实跑至3961，日志continuous_sampling_fraction=0.1484（19/128，符合0.15整数配额），immutable weight/routing delta0，动作有限；当前处于critic warmup，不将actor未更新视为失败。run_dir=/home/kellen/桌面/11 (1)/11/wheeled-legged_RL/logs/rsl_rl/rmuc_wheel_leg_course/2026-09-23_19-48-38_continuous_commands_local_turn_v166。


### 77. v166连续采样实际覆盖和预热后更新已核实，绑定4000/4253独立评估

v166在3963结束critic warmup，3966 actor对固定anchor最大动作差0.010004；immutable weights/routing delta0、actions有限。前13轮审计合并得到59个命令pair，含非课程点；审计为512观测分层子样本且命令四舍五入，混有active/rehearsal，不能据此声称全域覆盖或已泛化。early_sampling_audit.json保存准确限制。当前速度采样下界来自已开放课程最小值（此阶段约1m/s），不是已覆盖0速自转；最高DR及后期自转仍需独立课程。

新增evaluate_milestones.py，worker562628绑定trainer561199/start ticks；仅对实际可加载的4000和训练终止后的4253 checkpoint评估双向64replicas，拒绝覆盖报告和重复评估，缺checkpoint/评估失败则明确记录。状态milestone_evaluation_status.json，worker/trainer均已ps确认存活。未再次重启或调整训练参数；固定2.4命令评估不能证明随机邻域泛化，后续还需要留出网格及命令切换证据。


### 78. 连续命令覆盖审计发现直行节点误清零邻域转弯，已修采样器

使用v166真实params/env.yaml在Level28离线抽100000点：旧采样器取全部速度节点相邻yaw界最小值，使1.0–2.0等已有转弯锚点之间被直行专属课程节点清零，straight fraction0.79974；2.2–2.4且|yaw|>0.25仅0.01644。sample_envelope不是物理可行域证书，直行课程yaw0也不应解释为该速度禁转。这会削弱用户要求的连续转弯泛化。

修正command_sampling.py：速度域仍来自全部已开放课程，转弯包络单独由已开放非零yaw速度节点构造；相邻转弯节点取较小yaw界、范围外仍0，不外推到未开放高速转弯或自转。重复速度仍取已开放最大yaw。5项采样/实际reset/gate测试通过，含直行中间节点不能清零转弯、无转弯/单节点不外推回归。相同离线抽样修复后straight fraction0.29899，2m/s以下非零转弯占总样本0.50075，超过2.4的非零turn计数0。

证据sampling_envelope_audit.json与corrected_sampling_envelope_audit.json；修正仅磁盘代码，已运行v166仍加载旧函数，不伪称已热更新，不为此中断正在进行的300update实验。后续续训启用修复；当前4000/4253评估仍独立固定命令，不受采样器改动影响。


### 79. v166/model4000固定薄弱点未改善，冻结完整性不能替代行为防遗忘

独立报告model4000_large_turns.json已完成：2.4/±0.3、128env/每向64replicas、seed44、DR0。正向survival0.80282、coverage失败、bank3.05224°失败；负向survival0.9375且coverage/survival失败。v165final3953对应0.88235/0.953125，当前固定点表现下降，不能把采样多样性增加说成泛化或收敛改善。随机命令也分走了固定active样本，贡献与学习阶段/轨迹变化尚未隔离，不能单凭此归因为随机化有害。

直接比较3953和4000的actor_state_dict：13张量改变，只涉及distribution.std_param、stage0、domain speedheads；base逐位相等，无非授权改变。checkpoint4000_integrity.json、model4000_decision.json保存证据。保持既定最多300updates，trainer561199持续运行，worker562628已转为等待4253；不提前采用4000、不放宽gate、不因单个中间checkpoint马上重启。运行v166仍旧采样包络，上一条采样修正尚未在该进程生效。


### 80. v166/model4000四个非课程中间点通过，等待前置模型对照

model4000_intermediate_grid.json：seed46/layout2028、2.1/2.3×±0.225，128env/每点32replicas/3000steps，每桶64完整回合，四桶所有checks通过、survival均1。正向bank分别2.72759/2.97147°，后者余量仅0.02853°；负向0.42262/0.26876°。该证据范围仅nominal固定四点，不代表2.4大turn失败已修复、连续切换或最高DR通过。

这些具体命令未作为固定课程锚点，但处于训练采样分布内，不称为未见分布泛化。为防止把已有能力归功于新随机采样，已启动同seed/layout/grid的3953前置模型对照reference_intermediate_launch.json；结果未出前不作随机采样因果改善断言。v166训练与4253worker继续，不重复重训。训练4024两向完整回合计数84/78，负向生存及bank/retention仍阻塞，不能解释为单纯等样本。


### 81. 中间点基线也全部通过，撤销将其视为随机采样新增突破的可能解释

reference3953_intermediate_grid.json完成：同seed46/layout2028、2.1/2.3×±0.225、每点64回合，3953四桶同样全部checks通过且survival1。2.3正向bank2.97559°，4000为2.97147°，微小差值不能解释成有意义的鲁棒突破。intermediate_grid_comparison.json保存同命令对应比较；当前只能证明4000保留了这四个中间点，不能证明随机采样令策略新学会它们。2.4/±0.3仍以既有失败报告为准。

同时inspect当前安装PPO.update确认actor与critic分别clip_grad_norm(max_grad_norm1.0)，没有共同norm把actor随critic一起缩小的代码证据；未据此修改optimizer/LR。trainer561199持续真实运行，等待既定4253最终评估，不为微小中间点差值提前进阶或改gate。


### 82. 等待v166既定最终评估期间排查恢复学习率，没有发现配置回退

直接读取model4075 optimizer_state_dict：base/std group2e−6、stage group1e−4、speed-expert group2e−6、domain group1e−4、critic独立group1e−4；核对train.py在runner.load后重新应用stage/domain overrides。故不支持“resume把stage LR悄悄覆盖回2e−5”假设，未修改优化器。证据optimizer4075_audit.json，组名以代码重建逻辑为依据，checkpoint只有critic有rmuc_group标签。

4077训练仍Level28，正向survival EMA0.9681、负向0.9370，bank2.9924/0.8146，immutable weights/routing delta0、动作有限。不能用训练正向EMA过线推翻4000固定模型失败。trainer561199与绑定worker562628均已核实存活，仍执行原300updates及4253最终评估，不追加中间点重复测试，不热改采样器。


### 83. 用户质疑复训效率：不能再用旧课程编号或局部通过暗示全任务接近完成

本次实时核实v166到4106仍Level28，iteration time6.46s，rehearsal0.6016、continuous0.1484，固定active约0.25（32/128，方向再分）。慢既包括当前困难点的采样被分摊，也包括长期小改动/偏置扫描未形成可靠恢复能力；不能确定归因于单个LR或冻结策略。阶段进度不等于全任务完成百分比，未定义验收点集合和权重时不得再虚报总百分比。

证据明确区分：3953在2.1/2.3×±0.225四桶64回合通过；2.4/±0.3仅负向通过；4000两向生存又均失败。旧3m/s直行gate不能证明同策略全速度双向压弯已掌握。最高DR实测entry3051连2.4/±0.15生存也仅0.56757/0.43373；最终3Nm径向气弹簧与部署、自转±10尚未验收。不是所有既有技能虚假，但“完整高速双向压弯已稳定学会、只需很快复训收尾”的前提不成立。

反思：连续许多微小偏置诊断和中间点评估占用时间，验证小样本再被更完整条件推翻，不能一直当作项目接近收尾。当前v166按已有有界实验完成；最终报告必须用于决定实质性训练/控制处理，而不是再原样无限追加。保持实际通过范围、失败和证据清楚，不通过百分比或过度乐观ETA掩盖缺口。


### 84. replay比例不等于浪费：已有足够retention回合仍出现真实稳定性缺口

核对4115完整训练日志，当前强制retention stages8/15/18/23/26各方向回合计数均≥64（最低26正向71），不是只等旧技能样本。23正向survival EMA0.9470、26正向0.9440低于0.95，26负向0.9733，其余多数1。日志计数/EMA是episode聚合量，不当作固定策略独立验收。retention4115_audit.json保存精确指标。

60%replay确实减少当前点样本，但现有证据不支持“直接削减即可安全提速”；旧技能仍有训练条件下的稳定性缺口，降低replay也可能损害用户防遗忘目标。此轮未更改比例/LR/gate，继续v166既定实验及4253最终评估。后续若调配额，必须在行为证据支持下实施，不能单靠参数冻结或已有旧证书豁免retention。


### 85. v166最终完成仍未解决正向生存，确认停止原样追加

恢复中断后以进程和报告核实：trainer561199、worker562628均已结束，model4253_large_turns.json完成。2.4/+0.3生存0.82857、12次tilt、coverage失败，bank2.96624°；负向生存0.96875、2次tilt，各checks通过。300updates未产生双向通过，不把负向或幸存窗口通过扩张为全任务完成。launch.json更新completed，正式控制默认策略未替换。运行该段仍为旧连续采样包络，磁盘修正版未热更新。

### 86. 速度外环消融严重恶化，拒绝凭前终止关联删除外环

4253失败轨迹显示若干正向加速期间pitch超过10°时speed反馈仍为−0.08，这只提供冲突假设、不证明根因。新增evaluate_rmuc_grid.py默认关闭--disable-high-speed-feedback诊断：只设高速速度反馈gain0，其他控制/物理不变，报告标记且整体passed强制false。以4253同seed44、128env、2.4±0.3、每向64replicas/1500steps实测。

no_speed_feedback_large_turns.json结果：正向survival0、191次tilt（重复失败回合），无steady样本；负向survival0.921875、5tilt，低于基线0.96875。明确拒绝删除该反馈，原控制参数未改，checkpoint权重未改。feedback_ablation_decision.json保存对照与决定；py_compile和真实执行证明新诊断路径工作。该反证不证明所有gain最优，更不是完整根因。下一步需针对保留现有控制时的状态恢复能力，不再原样无限追加或以单项稳态指标宣称攻克。


### 87. 用户询问全局门控归一化/残差限幅：确认当前仅部分保护，不能宣称完整实现

直接核对rmuc_command_adapter.py：base+g*(turn_expert−base)属于权重(1−g,g)的凸混合；speed expert另做局部凸混合。stage head内部shared+单方向输出，外部各stage逐个相加，虽各smooth gate有界[0,1]，没有跨head总和归一化，也没有stage/turn adapter输出tanh或硬限幅。Domain学习部分0.04*tanh有界，但固定2.0/2.4 seed在tanh外加入，seed加载仅校验各元素≤0.24；不能说整个domain或整个residual总量≤0.04。L2参考正则是软约束，冻结权重是完整性约束，最终action±1 clip不是残差预算。

多head重叠会叠加，旧stage15在2.4仍生效的已测现象与之相符；但缺少全局归一化不是自动根因，也不能直接给所有旧heads套softmax破坏checkpoint已学叠加语义。后续优先核查各输出通道的叠加预算、重叠和饱和，并用兼容候选验证；不宣布已修复。

本轮此前准备的早期pitch奖励选项已写入env.py/env_cfg.py，默认沿用旧阈值，不改变现有checkpoint评估；新增command_large_turn_pitch_tolerance_deg/scale_deg，在speed≥2且|yaw|≥0.2时可独立调整。实际奖励测试验证候选4°/8°在pitch10°时额外罚0.6、6°正常lean和直行/小yaw不变，含采样/bank共21项测试通过。尚未启用训练，因本架构问题优先而暂缓该奖励实验，避免同时变更多个机制。


### 88. 全残差预算实测与用户追问MoE结构：保护缺口不能继续含糊描述

v168/residual_budget_audit.json离线重放4253失败环境的完整采样历史（去除重复env），最大误差5.960e−7。pitch<20°正向267样本：stage有效gate和1.3，仅stage0/15非零；总残差逐通道abs最大[.08178,.10174,.02637,.08554,.10105,.03543]，负向stage gate和2。残差确有新增动作饱和，但轮base本身也在部分状态饱和，不能把所有倾倒归于残差失控；样本仅失败环境无幸存对照。

全残差预算诊断（包含固定seed及所有adapter，参考冻结expert凸混合）腿±.08/轮±.04：正向survival0.88060，bank3.13977°且coverage失败；负向survival0.953125但coverage失败。报告total_residual_budget_large_turns.json。预算未被证实能解决当前问题，未纳入正式策略；新增evaluate_rmuc_grid.py默认关闭diagnostic选项，仅允许vx2.4以避免speed expert混合参考含糊，整体passed强制false。

用户询问状态router、负载均衡loss、base LoRA。当前实际RMUC路径均未实现：router只按重构command目标vx/yaw的手工门控，虽然obs2用于重构yaw，它不是姿态/失稳状态router；专家MLP本身吃25D状态不等于gate吃状态。CLI确有moe_load_balancing_coef，但属于其他policy配置入口，RMUC当前adapter和PPO路径无对应可学习router/aux loss，不能以参数名存在说已启用。base无低秩A/B注入，只有冻结MLP；L2 anchor也不是LoRA。

责任反思：此前知道残差可削弱base行为，却以冻结参数、局部seed、L2和action clip代替完整组合约束，没有及时明确架构只属命令路由的加性残差专家，不能称与成熟可学习MoE等价。负载均衡是路由训练问题，LoRA是参数适配方式，二者均不能单独保证保留行为；后续设计必须同时覆盖状态路由、组合预算和行为retention验证，不盲目全局softmax强制异质技能等量参与。


### 89. 全局动作anchor正则与参数隔离：用户提出的梯度冲突必须测量

当前regularize_mean对整个minibatch的六维组合动作施加lambda*mean((mu−anchor)^2)，lambda1，未按旧技能/新技能mask区分；不是参数weight decay。当前学习点也被拉回固定参考，因此可能阻碍必要行为修改，但此前loss小不能证明不存在冲突，也不能据此认定唯一根因。

现有base/旧stage参数已冻结，当前stage有正/负独立网，但domain相邻速度插值共享可训练slice，当前head作用区也会覆盖其他命令；参数隔离并未完整覆盖行为支撑区。可采用旧图冻结+新技能独立可训练模块+受控路由作为主要防遗忘机制；共享router或LoRA仍需边界保护，不能简单删正则就声称不遗忘。

为避免继续猜测，rmuc_residual_safety.py增加每minibatch动作均值空间的PPO/anchor梯度norm、ratio、cosine记录，原梯度求和完全不变；零PPO梯度用null避免Inf。这不是参数空间冲突证明（还经过Jacobian/clip/Adam），但能直接观察输出正则是否抵消PPO。新增反向梯度及零梯度测试，保留显式Gaussian+L2等价和冻结审计测试。没有新训练运行数据，不能编造实测冲突比例。状态router模块尚未实施、LoRA尚未实施，全局正则尚未删除；后续架构改动需要兼容性和实际行为验证。


### 90. 全局anchor正则真实梯度测量：当前输出层未呈现主导压制

v169从4253启动独立8updates（4254–4261），无critic warmup以测真实actor更新，保持lambda1与v164固定参考、LR/奖励/物理及gates不变；加载已修正连续采样包络，此差异明确记录。不是正式能力续训或部署候选。进程1058528正常结束，8条安全审计、32个minibatch。

gradient_summary.json结果：anchor/PPO动作均值梯度范数比min4.925e−6、median7.026e−6、max7.811e−6，即0.0004925%–0.0007811%；cosine min−0.01697/median0.002297/max0.02598，37.5%为负但整体近正交。immutable weight/routing变化0。该证据不支持当前输出层正则主导压住PPO，不能以“全局”二字直接认定瓶颈。仍仅8轮全命令混合统计，经过参数Jacobian/梯度裁剪/Adam后可能不同，不能排除局部参数冲突。

保留原正则系数与正式checkpoint，不因未经证实的猜测删正则。状态router/参数隔离仍需针对共享domain和路由支撑重叠设计，而不是声称删L2即可修复。下一步若扩展诊断应分命令/可训练参数组观察，不重复相同全局loss数值报告。相关run与启动命令在artifacts/continuation_v169，当前无运行trainer。


### 91. 先结构后reward：完成独立状态router原型，尚未接入正式PPO

新增scripts/rsl_rl/rmuc_state_router.py：使用现有25D测量观测，无新增privileged接触/状态位；左右独立参数，每方向4个小专家，状态MLP softmax归一化选择，专家tanh输出后凸组合并乘逐通道预算。初始化专家末层0，修正严格为0；修正参考应是冻结的完整旧技能图，不能宣称限制了旧图已有总残差。预算腿.08/轮.04只是实验初值，之前同数值硬裁旧残差失败，不能据此称最优或稳定。

局部支撑速度(2.2,2.6)、|yaw|>.15，2.4/.3全量，其他锚点输出不变；此模块不是最终全速度router。软importance balance只对当前eligible样本按方向统计，鼓励batch使用多专家，不强迫每状态均匀；全部专家dense求值，无token dropping，不冒称稀疏MoE容量机制。balance_loss接口尚未接PPO。

参数隔离新增isolate_optimizer_step：backward后、step前对无eligible样本方向设grad=None，避免已有Adam动量/AdamW decay在零梯度时仍改权重。测试覆盖状态影响router、权重和1、极大专家权重下修正预算、零初始化、旧锚点、正负分离、负载失衡梯度、空样本、已有动量/decay下不变与非法预算。

模块尚未安装至现有actor/train/restore/export，不能宣称已优化完成或已用于训练；现有checkpoint与物理不变，LoRA仍未加入。用户认可结构问题处理完再针对性调reward，故早期pitch新奖励选项继续默认旧行为、不启实验。后续必须完整冻结旧图、接独立optimizer及aux loss、恢复/导出、真实训练smoke与retention行为验证，再做同条件reward对照。


### 92. 状态router接完整推理/恢复/导出，尚未接PPO

rmuc_command_adapter.py新增显式install_rmuc_state_router，完整forward在旧图之后加受预算约束的新修正；verify_rmuc_export.restore_actor和play.py检测checkpoint前缀自动重建模块并strict load。旧checkpoint不安装模块、原路径保持。play.py已编译验证，实际GUI play尚未单独运行。

v170/build_router_checkpoint.py从4253构造zero_router_candidate.pt：原图全部state tensors逐位不变，2048观测探针零初始化及保存/恢复输出逐位相同。新增参数不在旧optimizer内，候选明确移除optimizer_state_dict并标记必须新建optimizer，不能直接旧optimizer resume。nonzero_router_test_only.pt人为给专家输出bias，仅用于验证非零分支不会在恢复或导出时遗漏，绝不用于训练/部署。

非零完整JIT/ONNX验证通过，JIT误差0，ONNX最大2.384e−7，支持batch1/7/247；19项router+anchor routing测试通过。zero_router_inference_smoke.json真实8env/100steps运行完成、无nonfinite/tilt终止，短程不足episode门槛，整体passed=false是预期，不能称技能验收通过。所有证据在artifacts/continuation_v170。

尚未完成PPO安装、新optimizer参数集合、aux balance loss反传/分支梯度隔离与冻结旧图的端到端审计；未启动正式新训练。旧图总残差仍不受新预算限制，新模块只约束相对完整旧图的新增适配，不能改写成全局旧残差已经限幅。LoRA仍无，reward选项仍未启。


### 93. 状态router接入真实PPO、辅助loss与优化器隔离，恢复smoke通过

新增rmuc_state_router_training.py与train.py显式--isolated_state_router入口：安装checkpoint模块后只将router40张量和critic8张量加入两组Adam；旧图（含stage/domain/分布std）全冻结，不调用旧stage自动解冻scheduler。要求freeze_action_std，拒绝旧stage/domain LR覆盖与未验证teacher hooks。首次零router候选使用reset_optimizer；新格式checkpoint可恢复匹配两组optimizer，不静默丢失moments。

aux balance loss系数0.01，通过autograd.grad取router参数梯度再由参数hook加到PPO梯度，发生在原PPO clip之前；计算aux时禁止注入旧缓存防止递归污染。optimizer prehook将本minibatch无eligible方向grad=None，避免Adam动量/decay越界更新；每轮检查完整旧图参数和buffers逐位不变，记录实际router权重、aux及新增动作预算。单测与显式(main+aux)loss+clip+Adam连续3次更新对比通过，含原router/safety共21项通过。

v171真实32env/3updates（4254–4256）成功，checkpoint仅40个router张量改变。首次日志flatten stride漏采环境且iteration取runner旧值，修为rollout_probe分层采样与本地更新计数；不将早期日志零correction误认为没学。随后完整恢复model4256再跑2updates4257–4258成功，Adam steps12→20，旧actor所有张量仍与初始化逐位相同，router最大修正约0.000902。证据smoke_evidence.json、resume_evidence.json。

这是链路/隔离验证，不是稳定驾驶通过。新预算限制相对冻结完整旧图的新增修正，不能声称旧残差全局预算已修复；LoRA无，当前router仅2.2–2.6大转弯邻域。为先隔离结构贡献，v172从原零router候选开始128env、200updates、原reward、aux0.01、router/critic1e−4、全图冻结、L2=0（保留审计），corrected连续采样0.15/rehearsal0.60。stop29防止超出当前router能力范围后无效续训。先验证2.4双向与retention再调reward；launch在artifacts/continuation_v172。


### 94. v172真实checkpoint参数隔离成立，安排新技能与旧技能行为验证

运行到4275，对zero_router初始化与model4275逐张量比较：只有rmuc_state_router前缀40个参数变化，旧actor全部state逐位相同，证据latest_checkpoint_integrity.json。运行审计同样旧graph/buffer变化0，新增修正max约0.002464，尚未达到预算。正/负分支平均专家概率约0.245–0.253，说明未发生明显负载偏斜，不证明有用的专家分工或稳定驾驶。

绑定trainer1067090进程start ticks的worker1068788已启动，evaluate_milestones.py只做读已保存checkpoint的评估；4300和正常最终4453分别2.4±0.3各64replicas，若按stop29提前结束则使用实际model_level_29_entry保存点；最终再2.0/2.2×±0.3各64回合retention（32replicas×3000steps）。异常结束无期望checkpoint则报失败，不把旧中途文件冒充最终。不覆盖报告，不自动重训。状态evaluation_status.json，两个PID均ps核实存活。

旧图参数不变不等于旧行为自动认证，尤其新router支撑区边界/命令过渡仍需以后实际验证；此worker覆盖已知旧锚点和当前新点，不能外推至全部速度、最高DR或部署。reward继续原值，无LoRA，不按负载均匀图样宣称架构优化已完成。


### 95. v172/model4300隔离成立但双向实测未过，结构正确不等于恢复能力已学会

checkpoint4300_output_isolation.json：800个合成状态探针中761个有效gate严格为0，输出与zero_router参考逐位相同；其余新增修正均在预算内。探针中命令重构浮点误差可使边界gate极小非零，按实际gate分类，不凭命令标签把它写成严格0；物理retention仍需独立实测。

model4300_large_turns.json完成，2.4/±0.3、各64replicas/1500step：正向survival0.88235，bank2.99494°、yaw0.02815；负向survival0.90625，bank0.66871°、yaw0.03407。两侧coverage/survival失败，其他steady checks通过。冻结参考4253为0.82857/0.96875，当前一侧上升一侧下降，不能宣称双向突破；轨迹分叉也不构成精确配对因果量。model4300_decision.json保存范围与决定。

训练1067090继续原定200updates/stop29，绑定worker等待最终4453或entry29再做新技能及2.0/2.2 retention。未更改reward、预算、LR，不以单侧改善给新架构认证。后续reward应基于最终残存失败，不在本段中途同时调整混淆结构对照。


### 96. 基于4300真实失败轨迹定位pitch tracking奖励迟滞，候选暂不启用

新增v172/audit_pitch_signal.py，以报告runtime_contract里的原reward参数计算已采样物理状态的单项pitch tracking penalty；desired pitch按实际heading vx/目标vx及欠速项重建，原容差8°/scale20°/weight2。10个去重失败env首次记录|pitch|≥10°的状态中9个该项处罚0，余者仅0.13446；距离终止时间中位0.684s，范围0.64–3.216s。pitch_signal_audit.json保存逐例状态/单项值。

此前准备的large-turn容差4°/scale8°在这些同状态下处罚约0.2824–1.3362，能提前提供信号；它是候选配置，不是已经验证能防摔。轨迹只包含失败环境且每10步采样，无幸存对照，不能说所有10°姿态危险、总奖励没有处罚或该项是最终根因。其他姿态/速率/能耗/终止奖励仍存在。真实_get_rewards单位测试先前已覆盖正常6°lean、直行/小yaw不变及cohort一致性。

v172保持原reward继续到最终，以避免把结构与reward变化混在一起；当前无新增仿真作业、不重启、不放宽门槛。后续需在同checkpoint/预算下实测候选，再评价加速速度是否因更严pitch奖励被压慢。


### 97. v172完整结构实验收尾：旧四锚点通过，新2.4大turn未过，进入针对性reward实验

v172实际完成200updates至4453，旧actor全部state逐位相同，40个router张量变化，200条冻结审计全无违例，新修正全程观测最大腿0.006181/轮0.001577，未触及预算（不是全状态严格最大）。final_integrity.json保存。final_large_turns.json：正survival0.86957、bank3.02542°失败且coverage不足；负survival0.9375失败，coverage通过。没有双向新技能突破。

同最终4453的final_retention_turns.json：2.0/2.2×±0.3各64完整回合全部checks通过、survival全1，正bank1.98417/2.08083°，负bank0.82882/0.44184°。这是同策略、nominal四个旧锚点行为保留证据，非全速度/最高DR/部署证书。worker与trainer正常结束，未重复启动。参数隔离与这些旧技能行为通过，允许下一步针对仍存失稳的reward进行试验。

v173从4453带Adam续训，启用早已测试的command_large_turn_pitch_tolerance_deg4、scale8（原8/20），仅speed≥2且|yaw|≥0.2的pitch tracking项提前处罚，不改正常目标lean、直行/小yaw该项、不调高上限weight2，不改控制器/预算/物理/gates。10critic warmup适应新reward后仅router+critic训练；128env200updates、stop29，corrected连续采样0.15/rehearsal0.60不变。初始L2仍0、旧图冻结，resume保留新optimizer两组moments。

这是顺序训练试验，不是同算力严格因果A/B；需检查速度获取是否变差及真实双向生存，不以奖励值改变当能力提升。启动记录artifacts/continuation_v173/launch.json，待实际启动/预热审计。LoRA未加、当前router仍局部；其他最终目标未完成。


### 98. v173早期pitch配置实际生效、预热后router恢复学习，固定独立评估已安排

读取run params/env.yaml确认command_large_turn_pitch_tolerance_deg4/scale8，普通pitch tolerance8/scale20保持；日志4463 critic warmup complete，4469 aux4minibatches与新增修正证明router更新已恢复，旧graph/buffers无变化。runtime_reward_audit.json保存运行契约和最新审计，不把配置已生效当能力已改善。

worker1085971已绑定trainer1084482及start ticks，4500与最终4653/实际29entry独立评估2.4±0.3各64replicas；最终再2.0/2.2双向retention。沿用原固定物理/gates，不以reward改变评估定义；评估不依赖新reward值打分，只使用实际轨迹指标。异常终止明确报错，拒绝覆盖报告，不自动追加训练。状态artifacts/continuation_v173/evaluation_status.json，保持当前训练参数不再叠加改动。


### 99. v173/model4500中途评估：负向生存达线但coverage失败，正向未过

同seed44/128env/2.4±0.3/每向64replicas，model4500_large_turns.json完成：正向survival0.84286、coverage失败、bank3.00554°仍按原3°门槛失败；linear0.09740/yaw0.02715通过。负向survival0.96875、linear0.05983/yaw0.04633/bank0.90119°通过，但coverage失败，因此不能将负向称整体通过。参考v172final4453 survival0.86957/0.9375，一侧下降另一侧上升，不能确定新reward已改善总体恢复。

两侧steady速度误差未超门槛，但这些指标只代表steady采样，不能证明所有起步或失败轨迹都没有减速代价。保留失败证据model4500_decision.json，不将bank3.00554四舍五入成通过，不改变coverage要求，不按短期结果再次重启。trainer1084482继续原200updates，worker1085971已等最终4653/entry29与retention，reward等参数不再修改。


### 100. 核对实际PPO时间尺度，密集恢复奖励与终止信号不能混为一谈

从v173运行params读取gamma0.99/lam0.95、policy dt0.008s；inspect当前PPO.compute_returns确认GAE递推为delta+not_terminal*gamma*lam*advantage。首次尝试RolloutStorage.compute_returns出现AttributeError，仅说明当前版本该方法在PPO类，已定位实际实现，不停止审计或据此报运行错误。

(gamma*lambda)单个未来TD残差回传权重的e-fold约0.1304s；80步0.64s权重0.007391，86步0.688s权重0.005115，gamma本身对应权重分别0.4475/0.4213。temporal_credit_audit.json记录公式/参数/实际源文件。与前倾10°到倒地约0.684s的轨迹证据结合，支持测试更早密集pitch信号；但它不是总advantage或参数梯度，critic bootstrap可携带长期影响，不能宣称所有终止学习信号只剩0.5%或已找到唯一根因。

当前v173继续既定实验，不同时改变gamma/lam，不在中途扩大reward权重；后续若最终仍失败，再以独立对照验证时间尺度，避免无限同时调参。


### 101. v173最终未改善双向，保留失败证据并转同起点GAE对照

中断后核实trainer1084482及worker1085971均已结束，final_large_turns.json和final_retention_turns.json齐全。4653正向survival0.869565（与4453相同）、bank3.05668°失败；负向0.890625（4453为0.9375），两向coverage均失败。2.0/2.2×±0.3旧四点全部checks通过、生存1。直接checkpoint核对旧actor所有非router张量逐位相同。final_decision.json保存结果，不能把提前pitch处罚当已验证改善，也不原样加罚续训。

v174回到v172/model4453和原reward，做同初始化/optimizer、seed47、每组100updates（10critic warmup）、128env的串行对照：baseline lambda0.95 vs candidate0.99，gamma仍0.99，其余LR/冻结图/预算/命令采样/gates一致。不同lambda会改变价值目标和后续轨迹，单seed对照不能保证因果或泛化；提高lambda可能增加方差，须以固定双向/retention报告判断，不能以公式权重变大宣布解决。候选在0.688s处单TD权重约0.177，相比原0.0051，但不等于完整advantage。

run_credit_ab.py仅执行两个有界训练与每组2.4双向、2.0/2.2retention评估，拒绝覆盖，异常明确停止，不部署/无限续训。若课程提前进29则评估实际entry，需按实际轮数解释而非声称两组等轮数。worker251795及baseline trainer251796已核实运行至4461，baseline实际lambda0.95/原pitch8/20已核对，旧图审计无变化；candidate尚未启动。记录artifacts/continuation_v174/status.json、baseline_runtime_audit.json。此轮没有修改算法源码或物理资产。


### 102. v174基线组完成，当前薄弱点未过，候选尚不可比较

baseline095_training正常完成100updates至4553。baseline095_large_turns.json正向survival0.842857、bank3.04367°、coverage失败；负向survival0.90625且coverage失败。两侧steady速度仍过门槛。保留baseline_interim_result.json，不能仅凭基线失败断言lambda0.99会改善。

worker251795已顺序进入baseline095_retention（PID259873核实存活），完成后才启动同4453/seed47的candidate099。未覆盖基线、未改变A/B预算和reward、未重复训练。最终需两组全部报告与runtime契约核对，单seed结果仍不扩张为最高DR/全速度证明。


### 103. v174基线retention通过、lambda0.99候选实际启动并核对契约

baseline095_retention.json四桶2.0/2.2×±0.3各64回合全部checks通过、survival1，基线的新2.4双向仍失败。worker251795顺序启动candidate099 trainer261310，实际已运行至4461，审计旧图/buffers无变化；没有重复启动或从基线训练后的权重继续，候选仍从共同4453/Adam开始。

runtime_ab_comparison.json比较两组真实params：算法差异只有lambda0.95→0.99；另外run_name及action/observation/state Gym space内部_np_random序列化值不同。任务env与train入口未找到space.sample调用，动作由torch policy产生，但保留差异而不宣称同seed就有逐位相同仿真轨迹。候选参数已生效不等于效果更好；等待同条件新技能/retention报告，仍未最高DR验收。


### 104. 用户提出单侧容量/倾角引导/降低重心：检查发现压弯不足及残留站立高度偏好

candidate099最终2.4/±0.3报告：目标bank约∓4.19767°，正向稳态平均roll−1.17108°、bank error3.02659°；负向roll+4.00013°、error0.75018°。yaw tracking两侧合格，不能把问题直接称偏航跟踪子网不会转向。当前正负router各5084参数/20张量、4专家、hidden32，结构容量相等；不排除任务难度不同/有效容量或梯度覆盖问题，但暂无证据靠单侧加宽即可解决。

root平均正0.30718/负0.30433m差约2.85mm，不能据此认定高重心是单侧根因；root_z不是全机质量加权COM，报告均值也不证明失稳前有主动下蹲。代码确认command姿态reward仍用stand_metrics.height_score，中心stand_root_height_target_m0.40、tolerance0.08，经posture0.4*shape0.5*0.5=0.1乘入；最低高度0.15只是gate/不足罚阈值，没有消除站立高度偏好。腿长奖励也仍在姿态项，需要一起检查。没有命令相关降低高度目标。

此残留偏好是具体奖励矛盾，但权重小、其他项存在，不能断定它主导倾倒。已存directional_posture_audit.json。优先下一步测原height/leg-length shaping与压弯奖励的同状态贡献，再独立测试转弯允许较低姿态；不盲目全局追求最低rootz，也不放宽导轮接地/姿态/速度门槛。直接增大bank权重需检验是否损害pitch恢复；此前终止主要pitch发散，roll tracking不足与其关系仍须闭环证据。

v174候选retention现已四点全通过，GAE候选比baseline两向生存改善但正向仍失败，尚非交付策略。


### 105. 用户撤销0.4m高度吸引、重申随机命令泛化：已落实行驶reward并实际训练

env.py行驶分支移除posture_score中的stand_metrics.height_score，保留原0.5*leg_length_score（未翻倍），因此不再在行驶reward中追求0.4m rootz。站立课程自己的高度奖励未改；腿长偏好仍存在，不能称所有姿态偏好消失。最低高度、接触/姿态/终止、速度/压弯gates未放宽，不设置新较低目标高度。

实际_get_rewards测试证明固定其余状态时height_score0→1不改变行驶reward，允许rootz0.15/0.2/0.3/0.4下无额外高度吸引。初始测试错误地假设低于floor总reward必然更低，但其他safe masks也变化，已改为同一低高度状态有/无height penalty对照，精确确认floor项保留。17项reward/bank测试通过。

v175从v174 candidate099/model4553带Adam继续，lambda0.99、原pitch reward8/20、同旧图冻结/router预算，唯一新增reward变更为移除height_score；128env100updates、10critic warmup、进29停。连续随机命令0.15显式保留，真实日志fraction0.1484=19/128，运行params同为0.15；不是只写计划。PID273763已到4560，冻结审计无变化；launch在artifacts/continuation_v175。

泛化仍未全覆盖：当前采样按已开放阶段，约1–3m/s速度，转弯支撑至2.4，其外部分仅直行；0速、自转/低于1与未开放高速转弯还需后续阶段。episode内固定随机目标尚非动态命令切换。必须最终同策略覆盖完整域并通过最高DR，不把采样代码存在当泛化能力完成。v174 A/B整体已完成，两组retention均过、lambda0.99双向生存较好但正向仍失败；采用为后续实验起点而非部署认证。


### 106. v175预热后更新正常，绑定新技能与retention独立验收

v175在4563结束critic warmup，4574 router实际aux4 minibatches、新修正有限，旧graph/buffers变化0。随机命令0.15持续启用；不能以移除height_score推断已主动下蹲或COM降低，腿长奖励仍在且动作/动力学决定真实姿态。

worker275512已绑定trainer273763/start ticks，等待4600及最终4653/实际29entry独立2.4±0.3评估，最终另做2.0/2.2四旧点retention；原速度/生存/coverage/bank/高度gates不改。evaluate_milestones.py已编译，状态evaluation_status.json；两进程核实存活，不重复启动。后续比较应同时看root轨迹/roll/pitch与存活，不把更低root平均值当作降低真实COM的充分证据。


### 107. v175/model4600取消高度吸引后尚未出现可观测降姿，正向仍失败

model4600_large_turns.json同128env/seed44/2.4±0.3实测：正向survival0.85507、bank3.00713°且coverage失败；负向survival0.953125但coverage失败。正roll−1.19054°、root mean0.307181m；负roll4.08137°、root0.304415m。与初始v174candidate099的root0.307182/0.304334相比，差约−0.000001/+0.000081m，没有明显降低机身证据，不能声称取消高度项就已主动降重心。root mean还不是COM或失败前瞬态高度。

用户取消0.4m行驶高度吸引已落实，不恢复该项；但移除一项偏好不等于新增低姿态控制，也可能仍受腿长姿态项、冻结旧图与有限新修正影响。保留model4600_posture_comparison.json，等待最终与retention再决定后续，不在中途同时改腿长reward/预算/网络容量。随机命令训练继续，全速度/最高DR目标未完成。


### 108. v175最终取消高度吸引实验未解决2.4转弯，旧四点保留

v175正常完成100updates至4653，trainer273763及worker275512结束。final_large_turns.json正向survival0.869565、bank3.04857°、coverage失败；负向survival0.9375及coverage失败。root mean0.307094/0.304539m，未出现可观测的大幅降姿。final_retention_turns.json的2.0/2.2×±0.3各64回合全部checks通过、survival1。final_decision.json保留结论，不能把旧技能保留作为新能力完成。

用户取消0.4m行驶高度吸引的要求仍有效，不因本次不改善而恢复该项。单纯删除高度奖励并未主动引导新姿态，腿长奖励及旧图输出仍可能约束姿态，但需另做因果验证；不直接以这些候选原因认定根因，不原样追加相同训练。当前无trainer，下一步保留随机命令与冻结/预算契约，针对剩余姿态因素验证。


### 109. 剩余0.34m腿长偏好已隔离为可回退选项，v176实际启动

代码核实leg_length_score=exp(−0.5*(mean|root到轮轴矢状面距离−0.34|/0.05)^2)，不是纯接触或安全项。其行驶reward最高贡献0.5*0.5*0.4=0.1；权重小，不能认定它主导单侧失败。新增command_leg_length_preference_scale默认1保持兼容，仅乘行驶的leg_length_score。v176设0，不恢复0.4m高度吸引，不给新低高度目标，站立课程不改。

23项reward/bank/采样测试通过；实际_get_rewards验证仅减少预期腿长奖励，leg_length_score变动不再影响该行驶项，contact与attitude仍起作用，replay与active一致。随机命令0.15、完整旧图冻结、router预算、lambda0.99、控制器和所有gates保持，最多100updates、10critic预热、stop29。是从v175final4653带Adam的顺序实验，不是同预算严格A/B，也不能把代码删除偏好当姿态能力已学会。

trainer742541实跑至4659，runtime config腿长scale0/随机命令0.15；worker743504绑定其start ticks，等待4700/最终4753或实际29entry，新技能2.4±0.3及最终旧2.0/2.2四点各64回合验收。启动/状态artifacts/continuation_v176，未重复启动。后续必须同时看root/roll/pitch及生存，未通过不原样无限延长。


### 110. 用户确认0.34m腿长偏好也不要，行驶默认永久关闭

按用户要求，command_leg_length_preference_scale在env_cfg fallback与configs/rmuc_wheel_leg.yaml均设0，不再依赖启动时显式覆盖。行驶无0.4m高度吸引、无0.34m腿长吸引，最低运行高度/height penalty阈值仍0.15m，姿态/接触/速度/生存等验收不变。仅高度项允许在阈值以上选择姿态，不把高度合格当全部验收合格，也不等同导轮接地边界已校准。

v176本来以scale0启动，不需要重启；站立阶段专用目标保留，不再进入行驶对应偏好项。奖励测试显式启旧scale1只用于legacy对照，同时断言新默认0，避免默认变更使测试失真。


### 111. v176/model4700未显示释放站立偏好后的姿态改善

model4700_large_turns.json：正向survival0.83099、bank3.05369°且coverage失败；负向survival0.953125但coverage失败。正roll−1.14397°、root mean0.307188m，负roll3.98601°、root0.304538m；相对v175最终root基本不变。model4700_decision.json保存同条件指标，不能将移除姿态偏好当主动降姿已实现。

用户明确取消0.4m/0.34m行驶偏好，继续保持关闭，不因本轮未改善恢复。奖励不再阻止某姿态不等于策略探索到了有效姿态；尚不能由此认定容量不足或预算过小（已有修正远未触顶）。trainer742541继续原100updates，worker743504等待最终4753/entry29与旧四点retention，不在中途同时改网络/reward/LR。


### 112. v176最终腿长偏好关闭实验完成：旧四点保留，新技能仍未达标

正常完成100updates至4753，final_large_turns.json正向survival0.869565、bank3.03524°且coverage失败；负向survival0.96875但coverage失败，不能只凭survival达线称通过。root mean0.307110/0.304349m，无明显降姿。final_retention_turns.json四桶2.0/2.2×±0.3各64回合全部checks通过、生存1。checkpoint与v175参考逐张量对比，所有非router旧actor状态相等，final_decision.json保存证据。

取消0.4m高度与0.34m腿长偏好保持默认有效，不恢复用户取消的目标。两个连续消融未产生显著姿态或生存突破，不能继续把站立偏好称为已经确认的主要瓶颈，也不能用少量新修正和未触预算推断网络容量已够或不足。当前训练已结束，无原样自动续训。下一步应结合失败状态下的动作敏感性/学习覆盖做有依据的干预，保持独立旧技能与全速度最高DR验收范围。


### 113. 回查之前双向突破：是按失败类型分方向局部修复，不是单纯加长训练

重新读取v163 milestone_turns_nominal_seed43.json与level27_entry_turns_seed45.json，确证同model_level_27_entry_2866在2.0/2.2×±0.3两轮独立网格全部通过，最低survival0.96875。当时2.0正向是bank3.44°近直立但survival1，负向是yaw/stable不达标；正向active连杆差分−0.12/+0.12、负向轮差分+0.02/−0.02经独立诊断确认后仅在1.75–2.2速度区间平滑生效，再继续学习至2866，旧参数冻结，非全局改所有速度。2.4/±0.15随后3051也通过，但不等于2.4/±0.3已通过。

当前2.4正向持续弱是实际问题，必须追查，不能用正常随机波动敷衍；但与当时不同，当前主要增加了pitch倾倒和动作饱和，旧局部seed在2.4为0，不可直接复制幅值扩大范围。此前stage15消融改善生存却破坏bank，表明姿态组合有权衡。后续应复用“分方向诊断→局部可达动作验证→完整候选→双向/旧技能验收”的方法，不把旧成功归因于无证据的容量扩张或统一奖励增大。

补记：上一轮写inclusion时误用Path.write导致AttributeError，之前final_decision与第112条已落盘，本轮改用write_text补齐最新交接；未影响任何训练/模型文件。


### 114. 用户要求高速倾角引导：核对已有roll目标、侧滑与接触力缺口

当前不是只有yaw奖励。实际_get_rewards有desired_bank=−atan(v_cmd*yaw_cmd/9.81)，roll=atan2(−gravity_y,−gravity_z)，高斯roll tracking（tolerance1.5°、weight1.5并乘bank scale2）及绝对bank误差罚（weight2、scale6°并乘bank scale2）。2.4/0.3对应约−4.20°。gravity_y同时仍被向零的roll_error²处罚，turn scale0.25只是减弱；upright/tilt与stand attitude也保留，存在目标间竞争，但不能不量化就说它们主导失稳。

侧向速度当前只有body-frame |vy|趋零处罚（weight0.35，截断2m/s），没有任意侧向指令tracking；command_y本就0。转弯需要侧向加速度，不等于要追踪非零侧滑速度。接触项使用轮net force三维范数的接触判定与左右幅值平衡，没有侧向力方向/摩擦利用率/质量加权支撑合力目标。不能把该范数称作已实现侧向接触力约束。主动降低COM尚未实现；取消高度/腿长偏好只允许变化，rootz不是全机COM。后续如引导必须用质量加权COM/姿态和接触证据，不能奖励无条件蹲低或把转弯必需侧向力罚到0。

方向/位置混淆对照已有新发现：v176同checkpoint只打乱bucket placement（layout2029）正/负survival变0.93939/0.921875，原0.869565/0.96875的差距反转；正bank仍2.98238°、roll−1.21529，说明压弯偏差与生存位置敏感需分别排查。完整交换位置对照正在独立运行，结果落盘前不判布局根因，不直接把10个百分点全归因于正向子网。


### 115. 倾角奖励同状态量化与位置交换完成：不再把竞争项当未经验证的主因

v177同v176/model4753、seed44、DR0交换两组命令顺序后，负向survival0.90625、bank1.36068°，正向0.96923、bank2.91243°；正向仍coverage失败、负向survival/yaw/stable失败，整体不通过。连同shuffle0.93939/0.921875与原布局0.869565/0.96875，生存方向差异随位置/分配反转；布局、初始化和数值因素是需排查的混淆，未证明具体根因。正向bank不足在三个布局仍一致，不等于所有方向问题都由布局造成。证据layout_comparison.json，三份checkpoint SHA一致。

新增artifacts/continuation_v177/audit_bank_reward.py，执行真实_get_rewards AST，合成相同2.4m/s、±0.3rad/s、零pitch/rates、相同接触/actions状态，对比直立、正确±4.197668°、反向倾角；使用当前训练bank scale2/start18，真实stand attitude公式。正确倾角相对直立bank项提高5.738662，总reward提高5.728996，upright/tilt、gravity_y归零和attitude合计只抵消0.009665；双向严格对称，reward clipping后排序保留。此同状态证据不支持“没有倾角引导”或“小角度直立奖励压过bank奖励”，但不证明动态可达、梯度信用分配或高速大倾角同样无冲突。结果bank_reward_audit.json。脚本首次漏math命名空间，补齐后实际成功；未修改生产reward。

用户要求的物理链条仍有缺口：已有roll目标及body-frame侧滑惩罚，gravity_y已参与roll估计，不需无依据再叠加同义loss；接触力只有范数/接触平衡，没有朝向或摩擦利用率目标。主动降低真实COM未实现，不能以移除0.4m/0.34m偏好替代。若引入侧向力/降低COM，先采质量加权COM、heading-frame轮接触合力、载荷分配和失败前轨迹，且区分稳态倾角目标与转弯瞬态；向心力不应罚零，下降rootz不应无条件获奖。当前实际动作修正局部、动态失稳仍未解决，不能仅凭reward审计宣称攻克或启动相同无效训练。


### 116. 位置敏感性的地面表示对照已启动，不再重复同配置训练

上一轮reward与布局审计属于实际进展。当前读取reset源码：root默认pose+env_origins、统一asset quaternion，未发现按yaw正负指定不同root初姿；仍不能排除动力学/初始化差异。地面是128m单tile生成mesh，现有尺寸检查覆盖grid+安全移动半径，不能直接断言地面缺失。IsaacLab interactive_scene的GPU clone启用enable_env_ids过滤，不能因env.py只在CPU显式filter就认定GPU机器人相撞。

evaluate_rmuc_grid.py新增--diagnostic-plane-terrain，仅显式诊断时切换解析plane，默认地面与训练不变；该标志强制report.passed=False。报告新增实际env_origins、reset root position/quaternion与25D observation用于追溯。编译通过。v178 PID772907实际进入cuda:0环境初始化，沿用4753/seed44/128env/2.4±0.3/1500steps、原命令分组，仅换地面表示；记录artifacts/continuation_v178/launch.json。结果未出，不得预判有效或当成能力通过，不重复启动。


### 117. v178未进入rollout的初始化故障已定位，禁止误计为策略失败

772907实际退出且无JSON，日志停于默认plane远程USD资产stat；未取得策略结果。改用诊断专用TerrainImporter子类本地UsdGeom.Plane+CollisionAPI，使用原physics_material，不改机器人USD或训练地面。775037仍初始化退出；发现Kit close可能遮蔽异常，evaluate_rmuc_grid.py增加close前traceback打印。775702重试明确ValueError: Environment spacing must be specified for configuring grid-like origins；设置terrain.env_spacing=scene.env_spacing修复。当前776109为修复后同实验，启动local_plane_spaced_launch.json、日志local_plane_spaced.log，前序失败日志全部保留。所有重试前均确认前PID消失，无并行重复。结果未出，不称策略通过；该诊断默认仍强制passed=False。


### 118. 解析平面出现实质改善，原网格复核已启动

v178/local_plane_direction_grid.json实际1500steps结束：4753同checkpoint、seed44、128env、2.4±0.3、DR0，每向64completed episodes；两向survival=1、各项checks全部True，bank误差2.617338/0.126525°、pitch绝对峰11.1633/9.6339°、无任何终止原因。report.passed=False是解析plane诊断强制标记，不是单项gate失败。初姿quaternion全部一致，env_origins首[16.5,-15,0]末[-16.5,3,0]，root均值0.30756/0.30509。相比原mesh的pitch倾倒和布局敏感出现重大差异，不能继续默认策略缺少容量/奖励是唯一主因。

尚不能把此单次替代地面结果当交付通过：未验证最高DR、硬件3Nm或全速度；需要核对材质/碰撞表示和初始状态一致性。原mesh使用当前同一评估脚本与新增初态记录复核，PID777950，mesh_recheck_launch.json及mesh_recheck_direction_grid.log；未改训练环境或策略。此轮先找动力学差异而非继续无效加训，不能以换成易通过地面缩小最终鲁棒性目标。


### 119. 同源码同初态A/B复现：地面表示是当前失败的重要因果因素

原mesh PID777950已结束，正/负survival精确复现0.869565/0.96875、bank3.035240/0.816546，全部与v176原报告一致。与local_plane报告比较：checkpoint SHA相同、所有记录源码SHA相同，128个env_origins、root positions、root quaternions与25D初始observations逐项精确相等；同seed/命令/steps/actor，不是换了策略或初态。plane双向survival1和全部单项通过、mesh失败的差异有了成对证据，plane_mesh_comparison.json保存完整对照。

这一结果改变后续优先级：先拆解三角网格/解析plane接触求解、法线/边界和材质绑定，避免把模拟接触表示的敏感性误判成网络容量或reward不足，再作训练环境修正及跨地面/最高DR复验。尚未定位接触内部具体机制，不宣称USD机器人有错；不能以plane单次通过替代最终3Nm、随机命令全速度和最高DR交付。当前两评估进程均已结束，无需重复启动，默认训练地面尚未改变。


### 120. 最终场景限定flat，必须包含多摩擦平地鲁棒性

用户明确最终只要求flat地面，不扩展台阶/坡地/rough任务；不同摩擦平地仍必须由同一最终policy适应，不能因解析plane单次通过就移除摩擦DR。源码已核实现有progressive DR逐reset将机器人碰撞shape静/动摩擦乘随机scale，full强度范围0.85–1.15倍、动摩擦不超过静摩擦，通过root_view.set_material_properties实际写入。它并非直接随机全局地面，轮地有效值还取决于双方材质和combine mode；当前±15%不能宣称覆盖多种地面范围。需以实际材质读回和有效接触范围决定后续摩擦课程、固定低/中/高摩擦独立验证，最终联合其他最高DR，不能只随机传感器或仅测nominal。

v179 PID780690正在单变量mesh细分对照，五次subdivide不移动原平面、保留原TerrainImporter材质路径；新增terrain_prim_audit记录碰撞schema、物理属性、材质binding和face数量。默认生产地面未改，诊断强制overall passed=False。评估脚本编译通过且实际进入首动作，结果待定。


### 121. 不换材质仅细分网格也双向通过，锁定大三角面表示敏感性

v179已结束，同4753/seed44/128env/2.4±0.3/DR0、原TerrainImporter材质，仅五次三角细分。每向64回合，survival均1、bank2.598988/0.154192°，单项checks全部通过；诊断overall仍False。与v178原mesh的checkpoint及全部initial_state_audit精确一致。实际USD审计地面Mesh10240faces，physics static/dynamic friction均0.5、restitution0，绑定/碰撞schema已记录在subdivided_mesh_grid.json。

解析plane与原材质细分mesh都成功，而原大面mesh可重复失败，说明大三角面接触表示/数值敏感是当前关键因素，不能再直接归因为容量不足。尚未证明PhysX内部具体算法问题，也没有最高DR/最终actuator验收。下一步选择可复现、有限细分的flat网格作为候选修正，做旧技能和多摩擦联合DR对照；不因nominal过关删摩擦要求。当前无运行评估或trainer，结果均已落盘。


### 122. flat网格修正正式接入，开始同checkpoint新旧技能与最高DR验证

将v179有效的五次mesh.subdivide逻辑移入flat_terrain.py，RmucFlatTerrainImporter沿用原TerrainImporter的材质创建和碰撞绑定，env_cfg的flat工厂与YAML默认flat_mesh_subdivisions=5；设0可回退旧大三角面。机器人closed-loop USD、25D/6D、冻结策略图、reward/gates与rough工厂均未改。新增几何测试验证边界、面积、Z高度、法线、原输入不变与非法配置拒绝；连同reward/layout共16项通过。evaluate报告加入生产subdivision参数与新文件source hash，诊断mesh/plane标志仍不计正式验收。

v180 worker784180串行验证同4753：2.4双向nominal各64回合，2.0/2.2双向retention各64回合，然后2.4双向最高DR。脚本evaluate_candidate.py、状态status.json存当前子PID和command，每项缺JSON则按基础设施故障停止，不能用退出码或意图算完成。最高DR用于测新地面下实际缺口，不把目前小摩擦范围当作用户多摩擦要求已经覆盖，也不替代3Nm硬件验收。此时训练尚未重新启动，避免在未知retention/DR结果前盲训。


### 123. 生产地面修正复现2.4双向通过，摩擦仍需运行时读回

v180/nominal_large_turns.json正式默认配置passed=True，2.4±0.3每向64回合survival1，bank2.598988/0.154192°，各项gates全部通过，与v179诊断数值一致。不是修改checkpoint或放宽门槛得来。worker784180已转入nominal_retention，实核子PID785739存活，后续队列full_dr_large_turns仍待执行，不能称最高DR已过。

直接用pxr遍历closed-loop USD，未发现显式friction/restitution/combine mode物理属性或physics材质绑定，usd_friction_audit.json保存空rows及限定范围。空结果表示依赖运行时/引擎默认或后续材质写入，不能据此宣称无摩擦。用户要求flat多摩擦，仍需运行时get_material_properties读回和地面combine检查后确定实际有效覆盖，而非仅依据max_friction_fraction0.15。


### 124. 用户指出低速老技能漏测：扩展为全部已开放课程命令复核

此前retention仅2.0/2.2四点，不代表1.0/1.5等完整旧技能已保留，冻结参数也不能替代地面修正后的行为验收。由实际YAML Level0–28展开转向正负、去重得41个命令点；v181/manifest.json和11个grid文件精确记录，优先1.0/1.5直行及双向0.1/0.2/0.3，1.0另含±0.45，再覆盖各过渡直行与2.0以上已开放转弯。

evaluate_rmuc_grid新增--command-grid有限数字[vx,yaw]精确列表，避免笛卡尔积误测未开放组合；保留原CLI默认行为和原gates。每批最多4点×32replicas，每点至少64完整回合，最多4500steps，nominal先检行为保留，不算最高DR验收。worker788585/evaluate_all_retention.py已启动，实核v180父进程存活则等待，待其完成退出后才串行评估，避免同时占GPU；缺JSON则停止并记录基础设施失败。结果未出前不得称1.0/1.5现在通过，也不得宣称遗忘。当前生产地面与同4753checkpoint均保持。


### 125. v180最高DR失败与retention缺口明确，不能只报生存率

v180全部结束。nominal2.4双向通过；旧四点生存均1，但2.0/+0.3 yaw误差0.090612rad/s超过0.05，yaw/stable两项失败，其余三点通过。此前“旧技能参数冻结”不能替代地面修正后的行为验收，需针对该缺口恢复，不得把survival1当四点通过。fullDR2.4正/负survival0.830769/0.444444，正linear/yaw/stable/bank亦失败、负yaw/stable失败，coverage均失败。终止均tilt（正11、负45），记录前16例pitch绝对值85.04–87.22°；full_dr_failure_summary.json注明样本截断，不能从这16例认定延迟或摩擦是唯一根因。

v181全41点复测实际进入grid00，子PID788985实核存活。需等其覆盖划清低速缺口，不原样扩大训练。评估脚本新增初始/最终get_material_properties读回，记录静/动摩擦范围、DR buffer写入误差和采样scale；这是shape系数，不冒称combine后的地面有效摩擦。编译通过，新增读回从之后启动的评估生效，旧v180报告不补造数据。

用户最新要求最终连续随机命令泛化全速度，保留此要求：训练应逐步覆盖完整目标速度/角速度域，并最终验证未见随机命令与命令切换，固定anchor仍是retention gate。当前训练采样只覆盖已开放范围、episode内固定；不能称已实现全域动态切换或最终泛化。当前阶段限flat，多摩擦及其他最高DR联合验收不变。


### 126. 1.0首批老技能已过，现有router训练支撑域不足以恢复其它速度

v181/grid_00_report.json通过：1.0m/s×{0,+0.1,−0.1,+0.2}每点至少64completed、生存1、所有checks为True。worker788585已转grid01，实核子PID797803运行；未测点不得推定通过。当前没有新trainer。

直接调用实际IsolatedTurnStateRouter.eligibility，对41个旧课程命令构造合法clean25D输入：仅2.4±0.3两点gate非零（阈值1e−6），其他39点gate0。router_command_support.json保存全部结果。此模块连续支撑在2.2–2.6速度、|yaw|>0.15，并非全域适配器；完整旧图冻结情况下原样继续router训练不可能修复2.0正转。它的局部保护原本是防遗忘设计，不能误当全速度学习已经可训练。后续按实测缺口扩展新的隔离修正支撑/训练分组，保留旧graph、预算、retention，不盲目扩大共享梯度或直接解除旧图冻结。此处尚未修改网络或checkpoint。


### 127. 2.0正转隔离恢复分支已实现并验证零扰动，不再向零gate区域无效训练

IsolatedTurnStateRouter新增显式可选recovery_positive_2p0，仅速度(1.8,2.2)、正yaw>0.15平滑生效、2.0/+0.3权重1；与旧router(2.2,2.6)支撑不重叠，沿用每动作预算和独立4专家softmax，初始输出严格0。旧checkpoint不含新key则原结构/forward保持，installer按checkpoint key恢复；训练optimizer inactive分支grad=None、balance/load日志同步包含新分支，避免Adam动量跨技能污染。未改base/旧skill参数、USD或25D/6D。

12项router/training测试通过，新增测试覆盖非零原router+零新分支精确同输出、负向/区间外不变、总预算，以及有历史AdamW动量时只更新恢复分支。v182/zero_recovery_candidate.pt从4753初始化，399个原actor tensor逐项精确相等、新20tensor，2048输入全actor输出差0，initialization_audit.json保存。optimizer state已移除，后续必须--reset_optimizer，尚未训练。另人工非零分支nonzero_recovery_export_test_only.pt仅用于导出验证：247命令点JIT差0、ONNX最大2.384e−7，通过；严禁将测试候选当训练/部署策略。

v181 grid01也通过：1.0m/s×{−0.2,+0.3,−0.3,+0.45}生存1，全部checksTrue；目前前8点已过，余点继续。该报告摩擦读回shape[128,40,3]，静/动范围均[0.5,0.5]，未出现动摩擦大于静摩擦；是nominal shape系数而非最高DR有效接触范围。当前子PID799717/grid02实核存活，未抢占评估启动训练。


### 128. 恢复实验已排队，明确预算、旧图等值审计和独立验收

v181 grid02通过，1.0/−0.45及1.5×{0,±0.1}生存1且所有checksTrue，因此1.0的全部9个课程锚点已过。其余仍以每批实际report为准，不预判。新增aux梯度集成测试包含原router与恢复分支混合batch，实际parameter hooks、裁剪、Adam更新与显式loss对照，13项router/training测试通过。

v183 worker804225已启动但等待v181真实进程退出，尚未开始训练。training_command.json基于已验证训练CLI：从v182零恢复候选开始、reset_optimizer、Level18、stop19、最多100updates、10critic warmup；lambda0.99、0.15连续采样、0.60rehearsal、旧图冻结/预算/原gates保持，nominal恢复优先。以较早level定位已知2.0正转缺口，不代表宣布高速度技能丢失或重做完整基础训练。

recover_after_retention.py限定一次训练，需19entry或4853正式checkpoint才算结束；随后逐张量校验除新增recovery_positive_2p0外的所有actor状态（含原2.4 router）不变，违规停止。通过审计后独立评估2.0/2.2/2.4×±0.3各至少64回合；GPU串行，不抢正在跑的41点评估。nominal恢复不替代最高DR与全速度泛化，队列意图不算训练完成，所有状态写status.json。

本条末次复核：grid03也已通过，1.5×{±0.2,±0.3}生存1且全部checksTrue。因此1.0全部9点、1.5全部7点共16个低速课程命令均已通过当前nominal复测；正在grid04，子PID804334。此结论不外推最高DR。


### 129. 继续真实队列并补齐评估溯源，不把排队当训练已运行

本轮实核v181 worker与当前评估子进程存活，v183 worker仍等待完整复测结束，无训练子进程；不重复启动。evaluate_rmuc_grid的source_sha256_at_start新增实际rmuc_state_router.py、rmuc_eval_layout.py和--command-grid JSON，避免后续恢复分支或命令清单变更未被原hash集合记录；不回填旧报告。脚本编译通过，此变更只补证据、不改物理/策略/gates。
grid04已完成：[(1.2, 0.0, 1.0, []), (1.35, 0.0, 1.0, []), (1.6, 0.0, 1.0, []), (1.75, 0.0, 1.0, [])]；以grid_04_report.json为准。


### 130. 2.0较小转角复测通过，已知缺口不能泛化为全速度遗忘

本轮先核实806813存活并运行到2000steps/每点至少32回合，属于真实等待而非停滞；未重复启动。随后grid05正式完成：2.0×{0,+0.15,−0.15,+0.25}每点64回合、生存1、全部checksTrue；yaw误差分别0.003631/0.019243/0.014555/0.022359，+0.25 bank误差2.887325°仍在3°门槛内。当前前24个命令全通过，转入grid06子PID809001并实核存活。

这进一步限制恢复范围：已确认失败的是之前2.0/+0.3，不是2.0所有转弯，更不是全部旧技能遗忘。新recovery分支在+0.25也非零，后续验收必须特别保护该已通过点，不能只看+0.3修复而忽略邻近角速度退化；最终完整anchor与随机命令验收仍必要。v183仍等待v181，未启动训练。


### 131. 2.0正大转角失配重复确认，恢复后邻近角度保护已落实队列

v181 grid06完成：2.0/−0.25、2.0/−0.3、2.2/+0.15全部通过；2.0/+0.3生存1、bank1.75705°合格但yaw误差0.0902496，yaw/stable失败，复现v180的0.090612。因此当前完成28点中27点通过，不能称所有旧点通过。当前grid07子PID810946实核运行。

v183新增recovery_retention_commands.json，将训练后独立验收从原六点补为八点，额外包含2.0±0.25，仍每点至少64回合/原gates。修改了worker的实际evaluation command；因旧worker已载入旧脚本，先确认804225处于waiting_for_v181且无child、cmdline匹配，再只终止此等待进程并启动810909（expanded_retention_launch.json），没有中断v181仿真。新worker实核等待，尚未开始训练。

评估报告新增steady signed yaw/speed error（measured−command，steady样本均值），用于区分跟不上/超速而非仅看绝对误差；无样本设None。原gate继续使用原绝对窗口误差，未更改门槛。编译通过，旧报告不回填未采集的signed数据。


### 132. 41点全量结果为39通过，恢复训练严格anchor加载错误已修复并实际更新

v181全部结束，final_summary.json统计41点中39点通过：2.0/+0.3 yaw/stable失败；3.0直行64episodes全tilt、survival0且无steady samples。其余旧点包括1.0/1.5、各过渡直行、2.2/2.4转弯及2.5–2.9直行均通过nominal。3.0失效不能隐去，也不将39/41作为全项目完成度。

v183自动首次启动失败于configure_residual_safety strict load：旧v170 anchor缺新recovery20个tensor，未产生训练更新；Kit关闭exit0不代表成功。worker原按run_name匹配目录也使错误归类不够准确，改为独占训练期间目录集合差分，并显式agent.run_name覆盖。保留training_failed_anchor.log、status_failed_anchor.json；将residual_anchor_checkpoint改为已证明零扰动的v182初始化候选（lambda仍0），没有放宽strict load或关闭审计。旧进程确实消失后重启worker910584，trainer910585。

实测已到4765，critic预热后recovery分支参与4minibatches、probe132samples，旧positive/negative无样本；frozen graph/buffers changes均空，新动作修正约2.8e−4以内，证明已实际学习而非仅排队。最大100updates/entry19停止与后续八点验收保持，恢复成功尚待证据。

3p0_action_attribution.json保存一个失败episode的实际pre-action观测离线组件分解：早期step161/241主要轮输出来自base、domain贡献约0.001–0.003、router0，之后发生pitch倾倒。这只是截断轨迹归因，不是闭环干预，不支持把此次失败直接归为残差失控压base；3.0尚无可训练router支撑，后续需要独立适配。


### 133. 真实训练checkpoint隔离验证通过，仍等待闭环效果

当前v183 trainer910585实核存活。独立读取已保存model4775与v182初始化候选逐tensor对比，仅recovery_positive_2p0的20tensor变化，所有旧actor状态（含原2.4router）相等。对41个课程命令各32个随机状态共1312输入进行完整actor前向对照，恢复支撑域之外差严格0，域内最大差0.000448406，证明实际训练后的参数与作用范围隔离而非仅初始化测试。证据model4775_isolation_audit.json；这是离线审计，不是2.0恢复通过，不替代最终八点验收。未额外启动GPU任务或改reward。


### 134. 3.0直行隔离支撑准备完成，等待前一轮实测后选择正式起点

实际3.0直行64次全tilt且现有router无支撑，故增加可选recovery_straight_3p0：speed≤2.9严格0、2.9→3.0平滑开启，|yaw|≤0.02全开、≥0.06关闭；独立4专家softmax、零初始化、同动作预算。与现有2.0/2.4支撑不重叠，inactive AdamW动量隔离和balance/load日志沿用branches接口。旧checkpoint无此key不安装，当前运行训练不变，不改25D/6D或旧skill graph。它只为高速近直行提供训练能力，不能冒称高速压弯或3.0能力完成。

15项router/training测试通过，新增覆盖2.9锚点保护、转弯关闭、组合预算、有动量的转弯分支在直行更新时不变。v184使用尚未验收的model4800仅做初始化/导出试验，原419个actor状态张量相等，新增20tensor，2048输入差0；非零直行分支247命令点JIT误差0、ONNX最大1.788e−7。两个checkpoint均明确test_only，不能作为部署或续训起点；待v183最终八点评估后再从选定策略零初始化正式候选。证据artifacts/continuation_v184。

v183训练实到4829，trainer910585仍在运行，冻结审计累计0违规；原100updates/entry19上限及八点独立评估未变，不提前认为2.0恢复成功。


### 135. v183完成100updates至4853，最终旧actor等值审计通过并开始八点评估

恢复训练正常达到既定上限model4853，未触发entry19提前通过，不追加同配置训练。state_router_training.jsonl记录100次update，冻结graph/buffers违规0；worker逐tensor检查最终checkpoint除新增recovery_positive_2p0外全部actor状态与初始化相等，old_actor_changed=[]。training_audit_summary.json保存更新区间和修正幅度。

自动八点评估已经实际启动，PID921344实核存活、环境初始化及首动作成功，包含2.0±0.25/±0.3、2.2±0.3、2.4±0.3，每点至少64回合，不改gates。尚无最终JSON，不能称2.0已恢复。3.0直行与最高DR仍为待解决缺口；v184高速直行模块仅准备，不是能力证据。


### 136. v183恢复八点结果：邻近技能保持，2.0正大转角仍为欠速缺口

model4853八点nominal评估已完成并写入recovery_retention_grid.json：2.0±0.25、−0.3、2.2±0.3、2.4±0.3共7点全部checksTrue、生存1；2.0/+0.3生存1、bank1.63935°，但yaw_error0.0848457>0.05、stable失败。相比训练前0.09025仅小幅改善，signed yaw error−0.101522rad/s确认实际转速低于命令，是欠速而非超速。新恢复分支未破坏+0.25（bank2.88660°仍接近3°但通过）或负向。

这次结果证明参数隔离和训练流程有效，但100update局部容量不足以完全修复+0.3；不能重复原样延长。已启动v185诊断PID925773：同model4853、2.0/+0.3、交错layout，5个轮差分offset 0/±0.01/±0.02，24replicas×3000steps，带failure traces；仅用于识别动作方向/可达性，整体强制不作验收。结果待定，未启动新训练。


### 137. 轮差分种子把2.0/+0.3 nominal欠速修复到通过，开始最高DR单点验证

v186固定恢复区种子wheel_right−0.01、wheel_left+0.01的8点nominal候选：2.0/+0.3 yaw0.029746、signed−0.028252、bank1.11863，所有行为checks通过；2.0±0.25/−0.3、2.2±0.3、2.4±0.3数值保持，只有episodes标记失败，因为16replicas×3000steps不足64回合，不能算行为失败。诊断日志/report留存，候选仍test_only。

种子由recovery branch显式限幅，残差预算扣除seed绝对值，15项router/training测试通过；向旧v183 checkpoint读入时installer自动补缺失buffer，保持旧checkpoint兼容。已启动v187 PID1047515，64replicas、2.0/+0.3、DR1、3000steps/64episodes单点，验证此可达补偿能否抵抗最高DR；结果未出，不能称DR鲁棒。


### 138. 轮差分种子最高DR单点仍失败：nominal可达但鲁棒训练不足

v187 64replicas、2.0/+0.3、flat DR1实测结束。固定−0.01/+0.01种子生存0.984375，bank1.65265°，但coverage/linear/yaw/stable失败，yaw0.06384；1次tilt。日志确认full flat DR实际启用：mass±5%、friction±15%、wheel damping±10%、wheel gain±5%，传感器延迟/噪声也打开。该结果改变下一步：方向和幅值是nominal可达，但固定seed没有抵抗联合DR，需要把seed作为隔离可训练分支的初始化/先验，在低→高DR训练并做同一checkpoint验证。不可把nominal诊断candidate当部署。


### 139. 延迟 DR 覆盖审计及旧恢复 checkpoint 加载修正（2026-09-25）

用户再次明确最终 DR 必须覆盖观测、通信及执行器延迟。现有 RMUC env.py 已有独立的观测历史与动作 FIFO：sensor_model.delay_min_steps=1、delay_max_steps=3，action_delay_enabled=true；每个控制周期为 0.002×4=0.008 s，因此两条路径分别为 8/16/24 ms。每个 episode reset 独立采样，低 DR 通过启用概率逐渐引入，DR1 每条路径必采 1–3 周期。动作 FIFO 位于控制残差合成之后、六路动作执行之前，覆盖命令送达执行端的纯延迟；两条路径闭环合计可达 16–48 ms，不能把总延迟也写成 8–24 ms。当前并非逐帧抖动模型。

缺口必须明确：独立于通信 FIFO 的电机响应滞后/带宽模型尚未实现及标定；已有 damping/gain DR 不等价于它。最终验收保留该要求，不能把一条动作 FIFO 重复宣称为通信和独立电机动力学两项均完成，也不能无依据再叠加一组 1–3 周期。待执行器响应范围有依据后补入，并以联合 DR 的实际轨迹和同一 checkpoint 验证。最高 DR 的最近实测 v187 仍失败，不能因延迟实现存在而宣称鲁棒性通过。

核查同时发现第137条所谓兼容加载有行为回归：旧 v183 恢复分支缺 wheel_seed buffer 时，installer 补入非零 −0.01/+0.01，会改变动作且缩小学习残差预算，即使旧权重逐 tensor 相等也不等价。现改为新模块默认零种子、旧 checkpoint 缺失只补零；显式保存的非零种子保持不变。撤回先前基于自动注入种子加载对照的等价性结论，v186 候选仍仅为实验，不能作为 frozen graph 保持证明。新增回归测试以非零已学习分支比较整个 router 输出，分别验证旧缺失种子和显式种子。

本次验证：sensor_model、state_router、state_router_training、anchor_routing、final_turn_envelope 五个文件共34测试通过，包含真实 env.py FIFO/reset 代码路径、独立 1–3 周期采样、部分 reset 隔离、旧加载输出保持。测试不是仿真闭环验收。本轮未启动新训练或评估。更大角速度扩展仍在准备：command 下已追加目标课程与目标范围，修正测试读取错误及半径舍入比较；这不代表连续命令采样、观测截断、奖励冲突和新专家支撑已经全部接通，也不代表已学会。


### 140. 建立长期最终目标，从正式4853保留权重构造显式种子候选

用户要求接续剩余任务并设立长期目标。已建立active goal并写docs/rmuc_long_term_goal.md，保留全速度/双向大转速/连续命令、先行走后±10自转、最高联合DR、轮端3Nm与有效damping/径向气弹簧、完整图导出和部署验收。开始时ps无匹配trainer/evaluator，RTX4080可用；未按历史PID重启。

v188从正式v183/model4853构造explicit_seed_preserved4853.pt，所有原actor张量保持，仅显式新增恢复支撑内轮差分−0.01/+0.01，未像v186重置已学恢复分支。旧loader默认零seed与实际已学分支的pre-seed forward逐位相等；2112观测完整actor探针域外差0，域内改变与seed减预算公式吻合。候选明确是新的行为初始化，不是无损动作等价或部署证书；optimizer显式移除并要求后续reset_optimizer，critic/已学权重保留。完整JIT误差0、ONNX最大2.384e−7。39项既有sensor/router/training/anchor/envelope/sampling测试通过。

已在4080启动八点nominal实测，PID1114805，launch.json记录身份及完整命令；每点16replicas、7500steps、至少64完整回合，首动作及3000steps实际运行已核实。输出artifacts/continuation_v188/seed_nominal_retention.json，未完成前不宣称通过。优先确认正式权重保留版本有足量邻域证据，再决定低DR学习；不从test-only模型直接续训、不原样无限延长v183。

### 141. 目标大转速包络接入连续命令采样，保留旧阶段分布

原因：原final_turn_envelope只被配置测试读取，运行采样仍取相邻旧转弯节点最小值，2.2/2.4/2.6等旧±0.3节点会阻断新增锚点之间的目标覆盖。新增command_extended_sampling_start_level=47与env_cfg读取，实际env reset传入采样器；只在已开放扩展锚点之间线性构造训练包络，以最终目标包络限制新增范围，区间外保留旧包络，不提前开放未来速度。yaw_min不是采样下限，直行、小修正和双向继续保留。此插值是探索范围，不是物理可行性/已认证能力承诺，未在评估端裁剪用户命令。

12项sampling/envelope测试通过，含实际env reset调用路径、Level18/28/46同种子采样逐位相等、未开放节点不泄漏、完整域中间速度双向大yaw覆盖与范围校验。运行中的v188评估已经加载旧源码；此次改动不热更新其进程，也不把离线采样称作已训练能力。后续仍须解决新增阶段过陡、fullDR46继承、yaw clip/超速reward与专家支撑；本条不声称扩展任务已接通全部链路。


### 142. 正式4853保留权重种子八点足量nominal全过，启动低DR有界学习

v188/seed_nominal_retention.json已完成：八点每点64完整回合、survival全1、全部checks通过。此前2.0/+0.3 yaw从v183的0.0848457降至0.0289990，bank1.11250°；邻近+0.25 bank2.88754°通过但仍小余量。使用同正式4853已学权重与显式seed，补足v186小样本不足且恢复分支重置的证据缺口。范围仅这八个固定nominal点，不扩展至全域/最高DR/3Nm。摘要nominal_acceptance_summary.json。

通过后worker1119770才启动trainer1119771，实际已输出迭代。最多200updates，目标最终5053或实际Level19 entry；Level18定位局部恢复，DR固定0.05以低强度开始，10critic warmup、128env/128rollout、rehearsal0.60/continuous0.15、lambda0.99与所有gate保持。旧actor/路由buffer不许改，最终worker逐tensor进一步要求只有recovery_positive_2p0学习张量变化。显式新optimizer、critic和已学权重继承；不修改wheel5.5Nm/damping0.1中心或气弹簧。

队列train_after_acceptance.py只运行一次训练，随后同最终checkpoint做八点nominal及DR0.05独立seed45各至少64回合；缺checkpoint、非零退出或冻结违规停止，不自动重训/升DR/部署。身份与完整命令见training_status.json；当前训练运行不等于低DR已通过。当前采样器修复已在这次新进程加载，但Level18分布与旧版相同。35项sampling/envelope/sensor/reward/bank测试通过及三个生产文件编译通过；一次使用不存在测试文件名的命令未跑测试，已按实际文件名重跑成功。

第142条运行补充：run_dir=logs/rsl_rl/rmuc_wheel_leg_course/2026-09-25_17-45-51_preserved_seed_low_dr_v188。4864审计aux_minibatches=4、recovery probe150样本、旧positive/negative支撑样本0、frozen graph/buffer变化均空；不把早期审计当最终行为证书。


### 143. 扩展大转速改为渐进阶梯，独立DR区间避免新技能直接满强度

本轮接续先核实v188 worker1119770/trainer1119771存活；未重启。读取真实model4875：仅recovery_positive_2p0的20个学习张量改变，所有其他actor张量含seed和旧router保持，live_checkpoint_integrity.json保存。训练已越过4890，最终行为尚待既定验收。

新增扩展课程原15阶段跳跃过大：1.0旧yaw0.45直接1.0/1.8/2.5。现保留旧0–46的完整stage字典逐项相同，仅将未训练扩展改为35阶段、Level47–81；每速度从旧最大yaw起步，增量≤0.25rad/s，仍达到最终全部目标上限和3.0/1.0扩展边界。1.0起点0.6，1.5/2.0起点0.45，2.5/3.0起点0.4。artifacts/continuation_v189/curriculum_change.json记录原47阶段hash及完整计划。

新增独立extended_start_command_level47/full81的DR调度。旧0–46公式逐位保持，显式strength override优先；进入新域从1/36≈2.78%开始，随stage/transition增长，最后81阶段transition完成时达到1.0。跨stage仍受实际gates限制，阶段内部仍按transition时间，不冒称全性能自适应。新阶段暂降DR是新技能渐进训练安排，不是取消最终最高DR，最终必须全域同策略重验；评估strength1也不被新调度降级。

17项采样/包络/实际env DR函数测试通过：旧47阶段在三个transition时刻保持原调度、扩展单调至1、play/disabled关闭、显式DR1/0.05优先、每速度yaw阶梯≤0.25、未开放目标不泄漏；生产文件编译及配置diff检查通过。首个生成命令f-string花括号报SyntaxError，未写入文件，修正后成功；没有把失败命令当配置生效。当前v188已经启动的进程不热更新课程，Level18/DR0.05与新磁盘行为一致。新专家支撑和yaw observation/reward配置仍未完成，不能直接开47宣称可训练全部目标。


### 144. 新旧router分支可显式冻结，避免扩展域采样更新已掌握分支

新增--state_router_train_branches明确选择可训练分支，其余actor包括旧router参数与全部buffer进入冻结审计；optimizer只加入选定参数，inactive梯度隔离保留。默认未选择时保持原路径，运行v188不受热修改影响。optimizer保存分支顺序，显式选择恢复时要求checkpoint同分支provenance，否则必须reset_optimizer，防止同shape不同分支Adam状态错配。

19项router/training测试通过（后补optimizer身份检查另测），包括旧正负命令同样eligible且已有AdamW动量/weight decay时只有目标恢复分支改变、篡改冻结router被运行审计捕获。编译通过。这是后续扩展隔离基础，不是新增大角速度专家已经实现；yaw观测与奖励选项仍需统一训练/评估/部署契约。

### 145. 用户要求分清已掌握与复训：复用证据并纠正低DR起点遗漏

用户明确避免搞错技能状态浪费GPU。新增docs/rmuc_skill_status.md及可重建CPU审计artifacts/continuation_v189/audit_skill_status.py、skill_status.json。原4753至最新4913，所有原有actor张量逐位相等；逐旧命令检查新恢复gate与完整actor输出。当前8点已有4913直接nominal证书，31点nominal旧证据+支撑/输出保持，2.0/+0.15为浮点边界但probe输出差0，3.0直行原失败且输出未改变。有限probe不是全状态物理证书，不能拼成伪造同hash报告；足以避免把没有退化证据的全部旧技能从头复训。

v188实际60次更新至4913自动进阶19并停止，未跑满200。最终nominal八点各64回合、survival全1、全部checks通过，2.0/+0.3 yaw0.0275693。当前worker顺序进行DR0.05独立验收PID1132078，无正式trainer、不追加GPU任务；本轮结果仍未覆盖最高DR/全部目标。

必须纠正：开始v188前没有独立验证显式seed起点在DR0.05是否已达标，故不能证明此低强度训练全部必要，也不能把最终低DR通过因果归功于训练。后续每段DR训练先使用同起点/同强度现有证据，缺证据先测试，不把未测试当失败或按惯例复训。当前2.0 nominal不再是训练目标；3.0直行明确待恢复，2.4转弯最高DR明确有缺口；新大yaw/动态命令为待获得新能力，低速旧nominal保留而不重复全量GPU复测。

第144条补充：新增optimizer分支身份检查后，training专项7项测试通过，包含同shape不同分支恢复拒绝与显式reset许可；尚未在新分支选择CLI上启动GPU训练。


### 146. 4913同策略八点nominal与DR0.05均通过，不再复训这一强度

v188/final_low_dr.json完成，checkpoint SHA35a183504aa7ae1a2d85f019530539051a1b8022407d9cb0e07a82f20da30c6e与final_nominal一致。2.0±0.25/±0.3、2.2±0.3、2.4±0.3每点64完整回合、survival全1、全部checks通过，2.0/+0.3 yaw0.028366。worker正常complete，accepted_low_dr=true。只证明八点DR0.05，不代表fullDR/所有旧技能在DR0.05都认证。更新技能清单和CPU证据，后续不重复这一强度训练；第145条起点未测的因果限制仍保留。

### 147. 从已验收4913准备3.0直行隔离恢复，并实际启动v190

3.0直行已有v181的64/64 tilt失败，4753至4913原有actor张量保持，该命令完整actor探针输出相同。本轮不重复同失败GPU基线，也不重训已过的低速nominal。v190/prepare_straight.py从正式4913新增零初始化recovery_straight_3p0；所有原张量保留，2948输入安装/恢复完整输出逐位相同。初始候选zero_straight_from4913.pt用于此次真实实验，非沿用v184/model4800 test-only；JIT差0、ONNX最大1.788e−7。

仅指定recovery_straight_3p0可训练，旧2.0恢复及2.4router显式冻结并入运行审计。用Level17仅选择已有3.0直行命令，DR0、200updates上限，进18停止；将该stage线速度gate收紧至0.15、survival到1200，原其余47阶段默认逐项与v188实际env.yaml核对一致，不把未训练的更高转弯stage列为已掌握retention。最终只评估受影响的2.95/3.0直行（各至少64回合）；CPU确认其他actor状态保持，不重复全部旧低速网格。

首次启动因nargs+吞掉Hydra overrides，在prepare_router_optimizer分支校验时退出，无Learning iteration、无训练更新。所有失败日志/状态/命令改为*_failed_cli保留。修正CLI为单个逗号分隔值，并新增实际parser AST测试确认Hydra参数保持在remaining；training专项8测试通过。另在重试前将stage gate数组构造改为执行实际env_cfg赋值，避免简化默认值公式误放宽高速度旧gate；核对仅17变严格。

确认旧进程退出后重试：worker1140476/trainer1140477，状态artifacts/continuation_v190/status.json；实际已首动作、iteration4914。run_dir=logs/rsl_rl/rmuc_wheel_leg_course/2026-09-25_18-07-03_isolated_3p0_straight_v190。新参数选择链路已进入真实训练，仍需预热后checkpoint隔离与最终行为结果，不能称3.0已恢复。没有变更轮力矩/damping中心或USD。

第147条实际学习验证：4923预热结束，model4925与零初始化候选逐tensor对比，仅recovery_straight_3p0的20参数变化，其他actor及全部buffer保持；optimizer仅记录该分支，证据checkpoint4925_isolation.json。4927日志aux_minibatches4，旧2.0恢复分支仍有68个eligible probe但保持冻结，证明显式隔离在真实混合采样下有效。3.0技能尚未通过，继续既定有界实验。


### 148. v190早期真实窗口仍失败，不把参数更新当恢复

本轮复核worker1140476/trainer1140477存活，未重复启动GPU任务。至4942完整窗口，最近10轮active实际vx均值−0.4788m/s、active survival EMA0，仍没有3.0直行改善证据；retention min survival EMA为1，但retention blocked标志仍存在，不能只凭该均值称全部训练gate通过。learning_window.json保存完整窗口而非截断日志。此时仅约19次预热后学习更新，不据早期窗口更改reward/LR或无限追加预算，继续既定≤200updates并做最终受影响区间验收。

后续大yaw clip及相对目标overspeed已有可选代码，但尚未统一进入新训练/评估/部署配置；本轮未改变运行v190观测契约。实测有效damping、电机响应时间/带宽及气弹簧推力/安装位置仍待硬件数据，已向用户询问可用数值，当前仿真任务继续，不伪造标定或用通信FIFO代替电机响应。


### 149. 用户给定工程动力参数并允许DR覆盖误差，更新最终迁移依据

用户明确暂无实测，允许按经典/工程区间并用DR弥补差距：RM3508减速器输出轴持续最大3Nm；damping估计0.005–0.08（本工程按N·m·s/rad记录）；气弹簧为每条腿各10.775N径向力，已专门确认是每腿而非双腿合计。原参考10.775Nm恒关节力矩与单姿态40.93N投影不得作为新的气弹簧名义中心。用户授权估计+DR后，不以缺完整实测为由停止仿真迁移；仍须保持力学方向/作用点和轮端硬限幅正确，并区分仿真鲁棒性与实机验证。

新增独立configs/rmuc_hardware_target.yaml，不自动热改当前trainer：damping中心拟取算术中点0.0425、对称fraction0.8823529411764706恰覆盖[0.005,0.08]；每腿10.775N，初始推力DR提案±20%（8.62–12.93N），明确是工程假设而非用户实测容差。电机响应时间常数仍需显式选定工程边界与独立验证，不用通信FIFO冒充；已有real_hardware0.8软件限幅单独列为待确定。

CPU脚本scripts/audit_rmuc_hardware_target.py验证数值和单位契约，报告v190/hardware_target_screen.json。按现有轮径0.06、无滑移稳态、damping当作未补偿粘性损耗且不计腿部推进，0.08在3.2m/s耗4.2667Nm，超3Nm；空载轮驱动估算2.25m/s。这是具体动力边界筛查，不是完整机器人不可达证明。后续必须区分explicit actuator D、passive joint损耗与torque补偿语义，实测/仿真极端参数确认；不能以DR或延长训练掩盖物理矛盾，也不擅自删除0.08端点。当前v190原动力保持。

此前本轮离线重放v181的16个失败env共800状态，原actor动作复现最大误差9.54e−7；失稳前|pitch|<20°的3.0新分支gate全1，仅9个后期状态gate0。4950新修正最大约0.001405，不支持“早期失败态门控关闭所以空跑”的解释；仍不证明恢复已有效。证据failed_state_support_audit.json，无新增GPU任务。


### 150. 执行器迁移中的阻尼补偿被前置限幅裁掉，改为直接插值explicit D

审查实际env._apply_action与本机IdealPDActuator.compute确认：旧bridge先加(d_base−d_target)*qdot，随后requested effort被限幅，再由IdealPD减d_base*qdot；只有未触限幅时才等效目标阻尼。例：轮速50rad/s、3Nm请求、base0.1→target0.0425，旧补偿经前置3Nmclip后得到−2Nm，而真实目标D应得+0.875Nm。因此原_last_effective_wheel_damping记录的是意图，不保证饱和时实际等效。

现利用已有DR每reset直接写入的同一个explicit actuator.damping tensor，在bridge启用时每动作按anneal fraction和domain scale直接插值D，不再添加会被裁掉的bridge补偿；保留原torque scale/effort limit迁移与全部clip、互斥选项、real_hardware禁用sim-only迁移。未启bridge路径不改，v190原进程未热更新且target−1，不受本次修复影响。

11项CPU测试通过，实际执行env bridge AST及本机IdealPD compute/clip源码，覆盖DR端点、中间fraction、partial reset后重写、关闭路径、正反速度/请求/制动和3Nm施加上限。报告artifacts/continuation_v191/damping_bridge_audit.json，编译通过。不是原生PhysX完整迁移或行为验收；后续单次启用前仍需运行时D/effort读回与独立行为确认。用户允许DR覆盖模型差距不意味着可用DR掩盖该确定的执行器实现错误；[0.005,0.08]上端力矩边界筛查仍成立。


### 151. 独立电机力矩响应模型可选接入，尚未原生仿真/能力验收

新增motor_response.py，在现有explicit PD compute之后对施加力矩按physics dt进行一阶响应tau_next=tau_prev+(1−exp(−dt/T))*(tau_desired−tau_prev)，再按当前effort_limit裁剪并反馈到actuator.applied_effort。覆盖leg_motors四路和wheel_motors两路，不影响passive/guide wheel，也不新增通信FIFO。reset清指定环境状态，每电机独立采样时间常数；DR0严格bypass不耗RNG，DR strength缩放响应时间。当前previous-action槽仍为送达动作命令，不冒称它直接反馈实际电机力矩。

默认motor_response_enabled=False，当前v190不启用、不热修改。工程探索范围2–30ms，不是RM3508厂家/实测规格，用户允许工程近似+DR后先明确记录假设，仍需敏感性和最终联合DR验证。更新独立hardware_target配置，记录implementation_available与尚未启用/原生验证状态。

evaluate_rmuc_grid新增诊断范围参数，实际配置/每环境每电机T与源码hash写报告；显式诊断不能整体passed，observation_only消融禁用电机模型避免污染传感器对照。20项motor_response/damping_bridge/sensor测试通过：精确阶跃曲线、DR0逐位一致、独立采样与partial reset、模拟effort_limit5.5→3时不泄漏旧高力矩、实际wrapper施加路径与诊断不可认证。编译通过，artifacts/continuation_v192/implementation_status.json说明未原生仿真，不将CPU测试当成执行器建模及训练验收全部完成。

v190当前5021轮仍实际运行，保持原有界直行实验，不新增并行GPU任务。之后在GPU空闲时做短原生接口探针，再决定响应DR适应；不把未启用模型写成当前训练已经覆盖。


### 152. 3.0学习超过100次仍无改善，提前保存停止，避免机械跑满预算

用户要求不浪费GPU。至5034，预热后已超过100次更新，最近20完整窗口survival EMA全0、active vx均值约−0.49；没有改善趋势。将原200上限视为上限而非必须跑满，记录early_stop_decision.json和原status后，只终止等待中的旧worker1140476，向训练run写STOP。trainer1140477按更新边界正常保存model_requested_stop_5035.pt退出；共122updates（其中10critic warmup），较上限少78，不是中途损坏/基础设施失败。

新finish_early_stop.py worker1164452严格绑定原trainer身份，等实际退出、读取唯一STOP checkpoint并逐tensor验证仅新直行20参数变化，再评估2.95/3.0各32replicas3500steps，未重复全41点。报告final_straight.json：2.95的64完整回合survival1，但linear error0.221539>0.15、stable失败；3.0完成128回合且survival0，全tilt、无steady。5035不采用为正式能力基线，不原样续训；4913原nominal及DR0.05八点证据保留。新分支隔离和更新正确不代表当前预算/初始化/命令路径能学会3.0。

后续应区分起步获取与稳态可达性，必要时验证渐进命令进入3.0以及新的动力契约；不能再同配置无限追加或放宽速度gate。官方资料搜索接口本次返回HTTP404（工具endpoint不可用），没有据此添加未经核实的RM3508厂家速度/响应参数。

### 153. 独立电机响应已通过原生链路短探针，非能力认证

finish worker在前一评估真实结束后串行执行v192/native_motor_response_probe.json，正式4913、8env、128controlsteps、DR1、2.0/+0.3、诊断T范围2–30ms。实际leg_motors为8×4，T约2.549–29.891ms；wheel_motors为8×2，T约3.160–29.835ms。两组response.effort与actuator.applied_effort严格一致、均未越过当前effort_limit；末态与无响应的clipped PD desired effort最大差腿5.31657Nm/轮0.79411Nm，证明实际执行路径非零响应，而非只保存配置。

报告passed=False是诊断且不足完整回合，不能作为联合DR、最终3Nm或实机证书。当前仍5.5Nm/0.1动力；目标配置仅将native_interface_verified改true，robust_policy_validated仍false。worker正常complete，目前无该训练/评估进程，不自动开启新训练。


### 154. 相同5035与物理初态，渐进命令消除3.0倾倒但仍欠速

针对冷启动获取与稳态可达性的区分，新增显式诊断--diagnostic-speed-ramp-seconds，逐episode从1.0m/s在2秒内升到原2.95/3.0目标，之后保持原目标；观测生成前同步command，reset仅重启对应env斜坡，观测历史仍每步只推进一次。稳态统计在原1200step预热之后，此时斜坡250step及300step窗口已结束；生存仍统计整个episode，不隐藏起步失败。诊断整体passed强制False。14项斜坡/电机相关测试通过。

v193/ramped_straight.json：同5035、同seed46、同64env布局、机器人初态及env_origins逐项一致。2.95和3.0均64完整回合、survival1；最终目标linear error分别0.163892/0.185375，仍linear/stable失败。对照v190固定命令3.0的128回合全tilt，说明这一次命令获取路径改变产生实质差异；不能把只改最终目标或幸存者采样当解释，因为目标保持、全部环境都有稳态样本且无倾倒。比较保存comparison.json。

这不是3.0技能最终通过或取消从静止要求：物理仍从静止开始，只是显式改变命令轨迹；固定3.0阶跃失败记录保留。下一步应在可存活的命令进入路径上解决剩余稳态欠速，并验证命令切换/完整范围，而不是原样冷启动复训。观测slot7从3.0改2.9的单失败env45状态CPU敏感性约0.001–0.004动作，仅作为输入诊断，不证明大容量或动作方向根因。

### 155. 用户目标动力配置接入显式诊断，不再只是参数说明文件

新增scripts/rmuc_hardware_profile.py与evaluate_rmuc_grid --diagnostic-hardware-profile：检查轮端3Nm、不再乘gear、damping中心/比例恰复现[0.005,0.08]、左右各10.775N径向力，原恒力矩关闭，六路电机响应模型选定；将中心0.0425与effort_limit3实际写入robot_cfg wheel actuator，torque scale3，补偿/bridge关闭避免重复计D。仍保留未明确变更的real_hardware0.8软件比例，诊断使用sim比例1以单独检验用户3Nm上限。

报告增加目标profile/源码hash和真实actuator damping/effort limit读回。该显式profile诊断不能整体认证。20项hardware profile、ramp、motor测试通过，含非法gear/力矩/不对称力/DR范围拒绝且不部分修改cfg。当前尚未运行目标动力原生评估，也未热改任何策略或物理；下一阶段可用它量化真实迁移差距，不把旧5.5Nm结果代替3Nm证据。


### 156. 目标动力原生探针确认明显迁移差距，不能直接替换旧动力

v194/hardware_target_nominal_probe.json：5035、24env（1.0/2.0/3.0直行各8）、2秒命令斜坡、1500steps、DR0，目标profile实际wheel actuator damping读回0.04250000045、effort_limit读回3.0。三点分别23/32/40个结束回合，survival均0、无steady，拒绝直接采用。样本不足64但大量早期tilt足以证明当前旧策略未适应；同时改变torque/D/径向助力，不单独归因某一项。DR0电机T为0，不能说这次检验了2–30ms响应鲁棒性。

径向力配置为每腿10.775N，但final_radial_spring_force_per_env_n为None：评估器旧逻辑只在独立--radial-spring-force参数时记录，漏掉profile启用路径；本轮读取时的NoneType报错是报告提取问题，不是仿真失败。已改为按实际cfg径向力>0记录，旧报告不回填假数据。transfer_gap_summary.json明确缺失读回，不宣称已测每腿真实施加力。此次仅迁移差距/接口探针，不是3Nm能力证书。

### 157. 起步斜坡接入训练且不污染最终跟踪gate，启动v195

新增默认0的command_reset_acquisition_ramp_s，仅fixed active、最终speed≥2.9且yaw0时，在每次真实零速reset后由1.0命令渐进至原目标；rehearsal和continuous cohort保持各自命令。没有给机器人初速度或外加推动。观测命令在后续physics/control/reward期间保持同一episode age（pre_physics计数+1单独抵消），避免actor看到t命令而外环使用t+1。

跟踪统计要求斜坡结束后再积累完整300step窗口，原survival1200/完整回合失败仍照常计数；早期倾倒不会因斜坡而排除。active gate分母仍保留全部active env，未用缩小分母伪造ready。32项training acquisition/command sampling/reward/bank测试通过，覆盖默认关闭、cohort隔离、观测与执行命令时序、斜坡窗口排除和早期失败仍入completed mask。生产文件编译通过。

v195从已实际斜坡诊断全存活但欠速的5035继续，保留匹配的Adam与全部已学权重，仅recovery_straight_3p0训练、10critic warmup、最多100updates至5135/实际18entry。动力仍5.5/0.1，不把未适应的新硬件同时混入本次单因素实验。worker1186105/trainer1186106已实际输出迭代，artifacts/continuation_v195/status.json。最终同2.95/3.0斜坡各64回合验收；只有全部行checks通过才额外测3.0固定阶跃，不重复无关低速全量网格或原样追加训练。正式已验收参考仍4913。

### 158. 用用户报告4m/s约束damping工程先验，支持分数迁移

用户补充原车最高速度可达4m/s。现配置wheel_radius_m0.06，轮速66.6667rad/s；每轮输出持续上限3Nm，在持续无滑动、无额外腿推进、其他损失非负且D为未补偿黏性损失的假设下，b≤3×0.06/4=0.045Nm·s/rad。最高速度不是阻尼标定，未确定是否持续水平带载或该转速可用力矩；不得把上界当真值。原0.08在4m/s需5.333Nm。独立hardware_target迁移DR先验暂收窄为0.005–0.04，中心0.0225、对称比例7/9；上端在4m/s留约0.333Nm余量，余量是工程选择并非测量。保留original_user_estimated_range及完整条件，旧报告不改，当前训练/USD未热改，4m/s只作物理依据不擅自扩大验收目标。

CPU审计artifacts/continuation_v197/speed_conditioned_damping_audit.json已执行；profile相关11项测试通过。现主配置wheel velocity_limit63.3734rad/s对应3.802404m/s，小于新硬件速度锚点，需核对执行器限速实际语义/实车轮径，未擅自调USD或宣称已完成4m/s验证。

此前已增加--diagnostic-hardware-fraction与apply_hardware_profile migration_fraction：0保持cfg不变，0..1插值实际基础力矩、完整D DR上下界、径向力及响应时间，1到达独立目标；避免旧策略直接跳目标动力。仅CPU检查，尚无部分迁移原生验收。

v195实际complete：5135斜坡2.95/3.0各64回合survival1，但linear error0.162740/0.183243均超0.15、stable0，未触发固定阶跃额外评估，不追加同配置训练或晋升。正式参考仍4913。v196串行新转弯baseline已运行（只评估无训练），以真实status/报告为准。

### 159. 大转弯基线四点确认跟踪失败，增加独立双向容量并启动v198

v196已真实complete，正式4913在1.0m/s、yaw+0.6/−0.6/+0.8/−0.8各64结束回合均survival1；linear error0.198945/0.245806/0.301939/0.293326，linear与stable全部失败，+0.8另yaw超限。失败根因尚不能唯一归因，但明确不是已掌握或单纯倾倒。因此先训练±0.6，不重复旧小yaw技能，亦不原样追加v195欠速实验。

IsolatedTurnStateRouter新增可选extended_turn_1p0，两个独立StateResidualExperts只在speed0.8..1.2且对应yaw超过0.4501生效（0.0001是命令重构数值保护），在1.0/±0.6达到完整门控，保持旧6维budget；与原router支持域不重叠。默认不安装，按checkpoint键重建，原图不动。35项router/training/loader测试通过，包括极端权重预算、旧±0.45精确保留、AdamW反向动量隔离。

v198从正式4913增加零输出容量，2948完整图输入输出逐位一致，原所有tensor相等；restore后同样一致。export检查命令网格增加新门控边界和大yaw，529输入/动态batch的JIT误差0、ONNX最大2.98e−7。初始化证据及脚本位于artifacts/continuation_v198，零初始化导出不代替学习后导出，worker训练后会重新执行。

worker1224256/trainer1224257已实际迭代，run2026-09-25_19-32-09_extended_1p0_turn_v198；10critic warmup、最多100updates，仅两个新分支，DR0、旧5.5Nm/0.1动力以隔离变量。stage47只选择1m/s±0.6，不表示之前课程均已通过；rehearsal/retention仍在，未放宽门槛。前4update审计frozen_graph_changed与frozen_buffers_changed均空。结束验证实际修改仅新分支、完整导出、1m/s±0.45/±0.6/±0.8各至少64回合，自动晋升关闭。当前不能称新技能已掌握。

补充第158条限速疑问：查当前安装IsaacLab IdealPD.compute与ActuatorBase._clip_effort，力矩裁剪只使用effort_limit，不用velocity_limit；项目仅设置velocity_limit而未设置velocity_limit_sim，后者默认从USD读取，原生启动轮关节限速表显示1.7e+04rad/s。因此63.3734×0.06=3.8024只是配置数值换算，不是该IdealPD实际速度硬上限，无需为4m/s盲改它。证据v197/velocity_limit_semantics_audit.json；真实转矩转速曲线仍未测量。

### 160. 为全存活但欠速增加稳态证据，已学新分支隔离/导出核验

v196四点全部存活，原failure_traces为空：终止轨迹无法诊断持续欠速。evaluate_rmuc_grid在原--failure-traces开启时新增有界steady_tracking_probes，仅采集原ready成立且linear/yaw跟踪超限的非终止状态，每10步、每env最多20帧，输出每command bucket最多2个replicas。保存实际产生动作的25D观测、raw/executed动作、物理状态和反馈，附当时滚动linear/yaw误差。GPU暂存，不改变控制、reset、统计分母/窗口或passed；采样数据不能替代完整网格统计。v198既定最终评估将直接使用，不另起GPU诊断任务。

实际采集代码块在CPU执行：上限20、空选择、未ready与跟踪通过排除、动作观测帧完整、源输入不修改均通过，文件编译通过。证据steady_probe_cpu_audit.json；原生输出尚待v198最终评估，不能提前说已经原生通过。

v198当前trainer1224257存活，至4936仍欠速EMA约+方向0.2011/−方向0.2458，旧图与buffers审计仍无变化，保持原100updates上限，尚不追加。对真实model_4925单独CPU审计：确有非零学习且只新双向分支参数变；957网格输入中909个保护区域输出最大差0，新支持域最大变化0.00019348。该已学checkpoint完整导出JIT误差0、ONNX最大2.98e−7，证据export_4925/export_parity.json与learned_4925_isolation_audit.json。此为隔离/导出证据，不是行为通过；最终仍验±0.45保留点和±0.6/±0.8。

### 161. 欠速诊断发现持续腿动作饱和，尚未断言因果或改预算

保持v198原有训练，trainer1224257在4952真实存活，正负linear EMA0.1997/0.2451、stable均0，旧参数与buffer不变。检查v196基线原报告，发现四点raw action index3（左腿控制通道）steady饱和比例全部1，执行均值1；raw均值依次1.083230/1.088765/1.065716/1.087566。轮raw饱和比例和施加力矩饱和均0。因此不能笼统说动作无饱和，轮扭矩未饱和也不代表腿通道有控制余量。

当前新分支腿残差budget0.08，在同一组基线状态上即使全额负修正，±0.6的平均raw仍约1.00323/1.00876。提示该腿控制通道的局部适应余量可能不足；均值不能证明每一状态都无法解除饱和，也不证明它就是欠速根因，学习后状态可改变。证据artifacts/continuation_v198/baseline_action_headroom_audit.json。待既定最终稳态快照检查逐点旧图输出与新修正；若仍失败，先有界动作偏置敏感性再决定局部预算或初始化，不未经证据提高全部预算或继续相同训练。此次未新开GPU实验。

### 162. v198停滞后正常停止4984，修复超128环境的评估编排

停滞monitor1236989绑定真实trainer1224257及startticks；至少60次非warmup学习后检查最近20轮，正负linear较原baseline平均仅改善0.001195/0.001901m/s，stable均0，触发STOP。trainer在4984正常保存model_requested_stop_4984.pt；总71updates，其中61实际学习（监控触发到边界多1），节省原100预算余下29次。旧参数/buffer无非法改变；完整checkpoint导出JIT误差0/ONNX2.98e−7，hash cf0ab1795e13270c522ed733f36026910402452f20440f31a6b39be6690dad9c。并非训练崩溃，也不把微小训练EMA变化当验收。

原worker随后评估6点×32replicas=192，被evaluate_rmuc_grid已验证1..128场景上限在仿真启动前拒绝，exit2。此为编排错误，原失败status/log保留，不放宽上限、不重训。新增resume_evaluation.py串行4新点128env及2保留点64env，各32replicas/3500steps/至少64回合，same4984/seed51/DR0；新点顺序保持v196，便于对照。worker1322265、child1322266已实际启动并输出first action，接续状态看resume_status.json（旧status记录原失败不可当现任务状态）。

前一中断前新增analyze_steady_probes.py会核对checkpoint hash、冻结图与记录动作回放，再计算局部delta和逐点clip余量。合成CPU对照回放误差2.98e−8，错误动作对及hash均被拒绝；不冒称原生结果。恢复worker将在新点评估后自动做CPU分析，再做保留点评估；无自动策略晋升。

### 163. 4984新转弯仍欠速、旧点保留；左腿解除clip收益分方向

v198恢复评估真实complete：同4984新±0.6/±0.8各64回合生存1，但linear error依次0.195227/0.243747/0.298019/0.291601，四点未通过。±0.45保留点各64回合通过（沿原对应命令门槛），不能将保留点较宽历史门槛挪给新点。正式参考仍4913，不晋升4984。原生steady probes已生成，160帧回放与真实动作最大差2.384e−7；±0.6多数采样状态旧left-leg index3超过1.08，负0.08容量无法在这些固定状态解除clip。仅采样证据，不代表全部状态或因果。

v199在相同4984、DR0上做双向±0.6、left-leg index3偏置0/−0.08/−0.12/−0.16，16replicas/1500steps，所有行生存1。正方向linear0.196809/0.194435/0.230925/0.269400，负方向0.244086/0.243720/0.215424/0.179548。较大负修正确实解除clip但一侧改善另一侧恶化，且没有通过；禁止据clip线索统一扩大全腿预算或声称已经找到欠速单因。偏置从reset施加、环境位置不同，属于小样本敏感性而非严格同初态因果试验或能力认证。

v200串行轮共同模式偏置0/±0.04/±0.08、双向±0.6、12replicas共120env、1500steps，worker1330854/child1330855已原生启动；以区分轮共同驱动力响应，尚无结果。没有新增训练、自动晋升或更改硬件目标；v199/v200均为显式diagnostic_action_bias，overall passed强制false。详见各artifacts/continuation_v199、v200/status.json及sensitivity.json。

### 164. 更正速度误差方向：正转偏慢、反转偏快；轮共同模式响应明确

检查原报告steady_signed_speed_error_mps后更正此前“双方欠速”措辞：v196正±0.6对应signed为−0.20035/+0.24580，v198为−0.19668/+0.24371，即正yaw偏慢、负yaw偏快。绝对linear error不能给出方向，历史raw报告没错，不覆盖；后续初始化必须按符号，而非统一推进。

v200已complete，12replicas诊断：正yaw共同−0.04将linear0.19884降到0.06777，survival1，但yaw0.05030略超0.05、stable0.037仍失败；共同−0.08已轻微超速，正共同+0.04/+0.08显著退化且倾倒。负yaw共同+0.04/+0.08将linear0.24374降到0.11939/0.003319，survival1、stable1，行内checks通过但小样本显式偏置overall强制false，不构成正式掌握或新增policy。负共同偏置使反向更快，符合signed读数。原0.04轮budget对负方向已有局部有效范围，不能仅凭腿clip判断必须扩大腿预算。

v201串行正yaw+0.6，以共同−0.04固定，加差动0/0.005/0.01/0.02（右轮更负、左轮更正），16replicas共64env/1500steps，worker1332944启动；保持yaw gate0.05，诊断是否补足剩余偏差。不自动训练、不晋升；验证结果待status。

### 165. 正向小差动敏感性通过后，构建局部门控seed候选v202

v201在正yaw+0.6共同−0.04基础上差动0/0.005/0.01/0.02，16回合全生存，速度error0.06366/0.06697/0.06555/0.06850，yaw error0.050295/0.049016/0.047607/0.044994；非零差动行stable1，diagnostic整体仍false。选择中间0.01作为后续候选，而非扩大门槛。负方向选v200已经行内通过的较小共同+0.04，不直接使用+0.08。

IsolatedTurnStateRouter新增显式checkpoint可选seed buffer：正轮[-0.05,-0.03]、负轮[+0.04,+0.04]，同旧新转弯gate，仅speed0.8..1.2/yaw超过0.4501。原4984所有已学参数与旧budget完全保留。新增wheel-only seed上限0.06（运行clamp），局部新router总轮上限为原0.04+0.06=0.10，腿仍0.08；不是无限残差或扩大所有旧分支。训练审计改用command-local correction_bound，仍检查冻结buffers不变。旧checkpoint没有seed键则不装，不能补非零默认值。36项router/loader/training测试通过，含零seed旧输出一致、保护域不变、极端seed/权重不越局部预算。

v202/prepare_seeded_candidate.py从4984生成seeded_turn_from4984.pt，原全部tensor一致，909保护输入差0，预期局部加法最大差5.96e−8。hash a6cd8c0e7d0ba53613312a9615055c0b49f24e32c2b20f55eb388f4f6489e788。删除旧optimizer并注明若后续训练需reset，当前没有任何新增训练。完整图恢复后先export，再以同seed51/DR0串行四新点与两保留点，每点至少64回合，worker1336754启动，artifacts/continuation_v202/status.json。这是候选假设，需正式策略评估验证从全程偏置到局部门控的改变，不把诊断小样本当正式通过，不自动晋升。

### 166. v202同完整策略±0.6足量nominal通过，±0.8仍需修复

v202/new_turns.json真实完成，同候选hash a6cd8c0e7d0ba53613312a9615055c0b49f24e32c2b20f55eb388f4f6489e788、seed51、DR0，每点64完整回合。+0.6 linear0.065851/yaw0.047600，−0.6 linear0.119113/yaw0.030524，两点生存1/stable1且全部行检查通过。不是外部action-bias诊断：修正在完整checkpoint图内部、gate与export已验证，无新训练更新。

+0.8 linear0.174338/yaw0.054336，−0.8 linear0.163496/yaw0.036959，均生存1但stable0、未通过，不能从±0.6外推。证据nominal_progress.json限定旧5.5Nm/0.1、单seed nominal，不是目标3Nm/最高DR或全任务完成。正式旧参考4913保持，候选后续应保护±0.6已过行为，只针对±0.8缺口，不整体复训。旧±0.45保留复验已串行启动child1338884，最终状态需核实v202/status.json；本条写入时尚未完成保留复验。

第166条完成补充：v202/status.json已complete，±0.45保留点各64回合全通过（linear0.136668/0.212558，沿原门槛）；与4984保留报告一致。候选±0.6已过，±0.8仍失败，整体passed=False正确保留。当前本组无运行中的训练/评估，下一步保护±0.6继续处理±0.8，不重训已过点。

### 167. 用户澄清新技能可跑满原上限，停止沿用短平台期早停

用户明确：避免浪费GPU是不要重复训练百分百稳定掌握技能，新技能该试仍需试，可跑满原先上限。此前v198因60次实际学习后改善不足0.005m/s而提前STOP，是对用户意图的过严解释。已将最新规则置于技能文档与长期目标顶部，优先于历史停滞建议；不删除旧实验记录、不倒写预算。后续新技能保留原max_iterations，不单靠短期无改善提前停；允许正常探索，仍保护旧图与必要retention。旧plateau monitor已结束，不重启。

v203对v202的±0.8共同偏置0/±0.02已complete：正yaw共同−0.02速度error0.10520、yaw0.05497未过，负yaw共同+0.02速度0.10304/yaw0.03756、stable1行内过，均仅12回合诊断。接续v204正yaw+0.8、共同−0.02加差动0/0.02/0.03/0.04、16replicas共64env/1500steps，脚本probe_positive_08.py已启动。此为前轮未完成诊断，不是替代足额新技能训练或重复已过±0.6。先取得结果再确定局部候选及训练起点。

### 168. v204差动诊断通过后构建0.8局部可学习分支v205

v204正1m/s/+0.8、v202完整策略、共同−0.02加差动0/0.02/0.03/0.04，各16回合生存1；yaw error0.055006/0.049644/0.046981/0.044277，非零差动行内通过，其中0.03速度error0.109937/stable1。v203负方向共同+0.02行内通过。均是小样本显式诊断，不作能力认证。选正增量轮[-0.05,+0.01]、负增量[+0.02,+0.02]为候选初始化，未改变v202已学参数。

新增SeededStateResidualExperts与两方向extended_*_1p0_0p8：gate在原1m/s局部速度域、yaw从0.6001到0.8平滑开启，保护此前≤0.6命令。seed+学习量共享新层cap [0.08,0.08,0.06,0.08,0.08,0.06]，学习量仅使用cap−abs(seed)剩余空间，不是在cap外叠加seed。旧graph、budget、buffer完全保留，完整新router最大局部轮界为旧0.10+新0.06=0.16，腿界0.08+0.08=0.16；审计correction_bound明确按command叠加各已声明上界，不声称总界仍0.04。后续若训练仅打开这两个新分支，旧±0.6参数冻结，遵守用户最新完整预算规则。

38项router/loader/training测试通过，含极端学习量限幅、零初始化精确一致、≤0.6保护与单方向训练不动旧参数。v205候选从v202生成，所有旧tensor一致，1067保护输入差0，seed预期加法误差5.96e−8；候选hash428ae6ca78eaa88b6fc8b1ad4c976ad0447ba5f38fcd2451ffea14225dedcc0e。完整导出JIT误差0/ONNX最大2.384e−7。worker1348690/child1348805已实际原生评估四点±0.6/±0.8，各32replicas、3500steps、至少64回合、seed51/DR0。未重复旧±0.45全量，已有域外逐位保留检查；最终完整验收仍须覆盖旧点。当前未新增训练，候选行为尚待报告，不自动晋升。

### 169. v205四点通过，v206确认仅正1.0需训练；v208实际启动

v205完整候选在seed51/DR0/旧5.5Nm和D0.1下，±0.6/±0.8四点各64回合全部通过，生存/稳定均1；+0.8 linear0.109201/yaw0.046888，−0.8 linear0.103945/yaw0.037631。单seed nominal不是最终硬件或最高DR证书。

v206同v205检查±1.0/±1.2各64回合全存活：+1.0 linear0.229930/yaw0.053898未过，−1.0 linear0.144645/yaw0.044762/stable0.952132通过；+1.2 linear0.353654/yaw0.060201、−1.2 linear0.169150/yaw0.051960均失败。方向明确正向偏慢负向偏快，不把−1.0列入同条件专项复训。

v207增加零初始化extended_positive_1p0_1p0，速度0.8..1.2局部域、正yaw0.8001..1.0开启，所有负yaw和正≤0.8受保护。新层六维cap[0.08,0.08,0.06,0.08,0.08,0.06]，已有界上再明确加该层：最大局部腿0.24/轮0.22，并非全图无限输出。所有旧参数/buffer一致，1218输入初始输出逐位相同；40相关测试通过。完整JIT误差0、ONNX667输入最大2.384e−7，v207/export_initial。

2026-09-26 v208编排首次启动因写入换行语法、候选路径错误失败，尚无训练。原日志保留，失败claim/status归档startup_failed_*。一次rm -f清理请求被自动审查拒绝（不允许rm -f），已改为保留证据重命名；未删除失败记录。随后使用issaclab解释器纠正启动。worker1916755/trainer1916756已实际完成4985首轮，run2026-09-26_10-15-14_positive_1p0_turn_v208；日志确认仅extended_positive_1p0_1p0可训练，旧图/buffer无变化。

原定100updates含10critic warmup，到5084或按原全部课程完成门槛进入50时停止；不启动plateau monitor，也不以短期无改善早停。去掉旧highspeed_regression_guard的行为早停，仍保留逐更新冻结图/有限值/残差限幅检查、rehearsal与retention。阶段49只是选择1m/s/±1命令，负方向仅作保留采样而不开放参数，不能声明此前全部课程通过。结束自动核验改变仅新分支、最终导出及±0.8/±1.0各64回合。当前训练未验收，查看artifacts/continuation_v208/status.json与真实进程。

### 170. v208开始实际新分支学习，5000已学checkpoint隔离及导出验证通过

中断后重新核实worker1916755/trainer1916756均存活，训练至5003，共19次更新、10critic warmup后9次实际学习；不是status中初始training_updates=0代表未训练，权威进度来自run的state_router_training.jsonl。正1.0当前训练EMA linear0.2283/yaw0.0539/stable0，尚未达标；负方向linear0.1456/yaw0.0448，当前正常保留。继续原100次预算，不据短期平台停止。

新增artifacts/continuation_v208/audit_learning_snapshot.py，CPU读取实际保存model_5000.pt：仅新extended_positive_1p0_1p0的20个参数tensor改变，全部旧参数及buffer不变。1131个状态命令探针中1091保护输入输出逐位相同；新作用域输出最大变化0.00023715，确认新分支有非零学习且声明残差上界未被突破。优化器实际state_router与critic LR均1e−4，state_router组仅新分支。证据learning_snapshot_5000.json，checkpoint hash e5cbea2aacffdca1019f3a16ac050a3565b870fbc737f7d6a6fc15a36385875a。

同一已学5000完整图导出验证667输入/动态batch，JIT误差0、ONNX最大2.384e−7；证据export_snapshot_5000/export_parity.json。此次只CPU核验，GPU原训练继续，未开并行仿真或额外训练。此为隔离和导出正确性，不作行为技能证书；结束后仍按既定方案独立验收±0.8/±1.0。

### 171. 为v208补充只读最终证据核对，不干预训练预算

新增artifacts/continuation_v208/final_evidence_audit.py，等待原训练/评估worker1916755真实退出后核对：状态complete、实际更新序列4985起连续且不超100、少于100必须是原课程50entry完成checkpoint、每次只开放正1.0新分支且旧参数/buffer未变，最终checkpoint/评估/export hash相同，完整导出通过，±0.8/±1.0每点至少64回合并保持原DR0/seed58且无外部action bias。继续调用CPU snapshot审计验证学习后冻结图、输出保护及残差上界。

汇总正负1.0相对v206与保留±0.8相对v205的误差和通过状态；明确±0.8基线seed51与本次seed58不同，不把这种比较写成严格同初态因果。无稳态指标保留None而不写0。CPU检查已验证未完成run被拒绝、相同报告改善为0、缺失稳态不伪造通过，final_audit_cpu_checks.json。只读审计worker1923559绑定真实训练worker等待，既不STOP也不重启、不晋升；最终报告尚未产生。

当前GPU仍仅原v208训练，预算100保持。最终证据脚本通过不代表最终任务通过；只有实际最终报告到位才判断新技能是否掌握。

### 172. v208完整100次预算已完成，正1.0仍失败；保留三点通过

实际完整训练100updates（10critic warmup+90学习）正常保存model_5084.pt，未平台早停。最终checkpoint hash bbd01c99405a8bfd338dc45aff39b413e28ed036afb620c2ed3757b623afa7d2。final_evidence_audit.json已生成：更新序列/预算、冻结图/buffer、完整导出及最终评估同hash均核对通过；训练确实改变仅新正1.0分支，并非未更新或导出漏分支。

同5084/seed58/DR0的独立四点各64回合全存活。±0.8分别linear0.109036/0.103905、yaw0.046886/0.037591通过；−1.0 linear0.143566/yaw0.044745通过；+1.0 linear0.223805/yaw0.053888未过。相较v206+1.0绝对速度误差0.229930仅小幅改善，不能把完成预算当学会；原已过点保持，不晋升为全部新技能通过。

在完整预算探索与足量验收后，v209仅诊断+1.0轮动作局部偏置：对5084比较未改、[-0.04,0]、[-0.05,-0.01]、[-0.06,0]，腿零偏置，16replicas×4共64env/1500steps，seed60/DR0。范围依据前阶段共同推进和差动响应，仍是新1.0的未验证假设。worker1939387已启动，诊断整体不可认证，无新增训练或自动晋升；结果用于下一次局部初始化依据，不改旧±0.8/负方向分支。当前状态看v209/status.json。

### 173. 用户要求反思奖励与物理/观测并对照原V14：确认bank物理方向错误

本次接续用户三项要求：复盘怠速/yaw及roll/侧滑/高度引导，进一步检查物理和观测，查原仓库V14。已完成源码继承链、真实reward/observation AST、独立世界几何与原生仿真诊断，汇总docs/rmuc_v14_reward_observation_review.md。原V14源码未改；本地HEAD b8ff79f及相关工作区文本（归一化换行）一致，溯源v210/v14_source_provenance.json，不假称网上最新版或V14实际训练证书。

关键纠正：RMUC canonical[-source_z,source_x,source_y]是左手排列，现reported roll与−atan(v_cmd*yaw_cmd/g)组合鼓励物理外倾。实际_state及安装IsaacLab quat函数核验：正yaw真内倾5.8204°被算11.6409°bank误差，外倾反而误差0；负方向对称。原v177reward审计生成状态直接复用了同一负atan假设，只证奖励自洽，未证物理正确。v211实际初态quat确认+90°X；同5084原生8replicas/1500steps测得按转弯方向定义的physical_inward_lean均负（+1方向−6.1805°、−1方向−3.3069°），确认实际外倾。旧bank gate/reward/evaluator三处一致地错，历史“通过”不可继续称物理内倾压弯掌握；原速度/yaw/生存测量仍保留。下一步先修世界几何契约并独立验证，不继续盲目提高原bank奖励。

V14由V13继承wheelbipe25_v3的观测/奖励：原始command+height_cmd+gyro/gravity+关节/action及可选扩展；critic额外读root_lin_vel_b/obs_height，并有历史/估计器变体；当前RMUC actor/critic同25D缺这些特权状态。V14默认轮速控制，RMUC直接轮力矩+外环，奖励每项V14乘step_dt而RMUC每步直接累计。V14姿态项主要向直立约束且随速度调权，没有这里的解析bank目标，因此不能说V14原有正确压弯目标已原样移植，也不能直接复制权重或改维度破坏25D旧图。

实际reward AST按v208 runtime config核验：1m/s额外fine-yaw资格为0、bank scale1（不是配置写2就全部生效）；2.5m/s对应1/2。合法yaw2.5仍被旧绝对1rad/s超速阈值罚0.5625/step。root0.32→0.20但其余不变奖励差0，仅安全下限无主动COM规划。存在速度改善的奖励梯度，不能说全无速度引导。当前wheel std约0.00614且冻结，有效轮修正0.05–0.06约8–10std，信用分配/持续探索不足需检查；value loss约4941不足以单独判critic根因。原V14特权critic及dt尺度差异值得优先对齐。

观测AST证实相同其余状态vy0/0.6时25D完全相同，零gyro时yaw2/2.5/3/10全部clip/解码为2；但当前1rad初始命令未clip，不能用它解释本轮失败。gyro_z是body投影，非世界heading角速度；当前DR0无延迟不能将nominal缺口归咎随机延迟。证据v210/observation_aliasing.json与reward_landscape.json。

新增只读--audit-turn-dynamics、scripts/rmuc_turn_diagnostics.py和3项CPU测试；eval不改控制、reward/gate。原生v210同5084四点16replicas确认：+1m/s yaw1 root steady0.3170m、质量加权COM0.4034m；负向root0.3116/COM0.3982。旧root轮速差指标+1为0.2597，按各轮局部纵向速度计算仅0.0487；负向0.2274→0.0190，正常内外轮差不可当侧滑惩罚。heading侧速约0.015/0.012m/s。2.5m/s负yaw1弱轮normal载荷比例约0.156，提示COM/卸载裕量重要，但不证明动力模型或腿结构本身错误。

ContactSensor.net_forces_w在安装版本只含NORMAL contact force，不能拿其水平分量当摩擦利用率。v210首报告contact_tangential_normal_ratio命名已发现不准确：原报告保留，contact_force_semantics_correction.json明确勘误，新代码改名normal_force_horizontal_vertical_ratio；没有宣称已测摩擦饱和。真实摩擦需独立friction接口，准静态支撑proxy也不是严格ZMP。

本轮未改变USD、马达参数、25D观测内容、奖励或已学权重；v209偏置小样本部分行虽过旧gate，在bank方向纠正前不晋升为物理压弯通过。完成的是问题定位与诊断，下一步优先修坐标/资格/观测契约，再安排足额学习与物理内倾的新验收。

### 174. bank方向修复落地，统一奖励/课程/评估并完成同策略足量对照

新增bank_geometry.py的bank_reference与契约world_heading_inward_v2。保留25D projected gravity及gyro原值，基于旧forward/right/up重排推导世界heading-left物理倾角：先归一化g，observed=atan2(−g_y,−g_z*sqrt(g_y²+g_z²))，target=+atan2(v_cmd*yaw_cmd,9.81)。水平forward投影项处理pitch，不只是盲翻符号。奖励、课程活动/retention bank统计、独立evaluate均调用同函数，原−atan路径已替换；观察、动作与USD未改。env_cfg与训练launch metadata记录contract，评估provenance_version2、bank_contract标签和helper hash区别于旧证据。

25项相关测试通过：1000随机SO(3)姿态以独立world body-up/heading-left几何为oracle；真实reward AST偏好双向真实内倾；真实课程bank误差接受内倾拒绝外倾；evaluator实际语句与共享参考一致，原reward/gate/诊断测试兼容。修正了旧reward测试合成姿态的错误符号，但不覆盖历史v177报告。

v212同5084 hash bbd01c99…、seed58、DR0，±0.8/±1.0各64回合完整对照。initial_state_audit逐项相同，四点linear/yaw/survival差值严格0，说明没有改变策略物理动作。bank误差旧0.3834/2.5895/0.3464/2.5133→新9.7111/6.7351/11.9925/9.1278，四点bank均失败；这纠正的是评价，不能写成旧技能突然遗忘或新技能完成。证据v212/corrected_bank_baseline.json与legacy_comparison.json，代码修复不等于策略已学内倾。

v213接续修复后的物理内倾可达性诊断：同5084，1m/s±0.8，right/left active leg动作差动0/±0.12/±0.24（轮不加偏置），12replicas×10桶=120env，1500steps，seed62，原gate不放宽。worker1963206/child1963207已原生启动，记录COM与物理内倾角；不是训练、不自动晋升、未扩大生产策略预算。先辨识正确姿态动作方向和速度代价，再为实际缺口准备局部适应。用户新技能可跑满预算的规则保持。

### 175. 轮速环迁移原型及原生等效探针完成，未切换默认策略

用户讨论V14轮速环与力矩方案的总部署时间，明确应区分控制增益和机械损耗。V14本地默认wheel IdealPD stiffness0/D0.2/effort5是速度比例控制，不是已标定RM3508 PI或机械阻尼；用户轮端持续3Nm仍是目标约束，不能照搬5Nm或将4m/s推算的被动损耗范围当Kp。此前官方搜索接口返回404，没有补写厂家响应规格或电调实现假设。

新增wheel_velocity_control.py纯Torch离散PI：显式physics dt、参考轮速限幅、输出轴effort限幅、测量一阶滤波、条件积分anti-windup、反转、partial reset、停用清积分；Kp可按环境/轮传入，但DR范围尚未标定，不默认采样。独立配置configs/rmuc_wheel_velocity_migration.yaml是提案、不被生产trainer自动加载：初始工程P0.2/I0、2ms周期、目标3Nm/80rad/s，实机0.8软件比例保留时仅2.4Nm必须明确；机械损耗单列，不把controller D和被动摩擦重复计入。

明确等效诊断将旧已限幅effort_target转换为omega_ref=effort_target/old_D，Kp=old_D，Ki=0，在每physics步得到同样clip(effort_target−D*omega)。不是把policy两轮动作直接当rad/s、不是速度环收益或硬件迁移已完成。evaluate新增显式--diagnostic-wheel-velocity-equivalence，仅DR0/无hardware profile/无电机响应wrapper可用，拒绝不兼容旧速度目标；passed强制false，记录controller计算次数及逐步力矩误差。默认运行未安装速度环，旧25D/6D、previous-action、USD与策略参数保持。

15项控制器/电机测试先通过（含执行已安装IdealPD真实compute对照、3Nm限幅、饱和1000步无windup、反转、滤波/局部reset/缩小限幅）；新增诊断不可认证检查后controller共7项通过。v214原生8env/128controlsteps、5084、DR0、旧5.5Nm动力短探针完成：513次physics compute，最大等效力矩误差1.192e−7Nm，最大速度reference9.1396rad/s，dt0.002。velocity_equivalence.json与implementation_summary.json保存证据；目标3Nm原生行为和实机控制器辨识均未验证，不自动训练或切换默认方案。

补记v213诊断已完整结束：所有腿差动组生存1，但物理内倾/综合门槛均未通过。正yaw+0.8加right/left差动+0.24/−0.24使外倾约−5.046→−2.170°，仍未真正内倾；负yaw用−0.24/+0.24约−2.073→−0.008°、bank误差4.670°仍超3°。这证明腿控制方向有效但单一常数差动还不足，不能当物理姿态已修复，更不能因此恢复错误bank目标。后续须在最终控制契约下做有足够探索的姿态学习。

### 176. 用户推荐只迁移轮子：显式3Nm直切失败，转为轮执行器分数迁移

用户提出部分迁移、只针对轮子。执行范围明确为轮执行器速度环与后续轮输出适应；腿继续位置控制、腿执行器参数/旧策略权重、USD、气弹簧与25D/6D不动。迁移不等于禁止后续为已确认物理内倾缺口单独训练腿姿态，不能指望轮速环替代压弯腿控制。

新增install_wheel_velocity_controller显式按已送达_policy_actions两轮分量×55rad/s×原符号生成参考，physics2ms计算PI并以实际effort_limit限幅；旧IdealPD stiffness/D置0避免叠加，原5.5动作尺度仅保留为旧外环/归一化历史契约，不再当实际马达输出。profile只改wheel_motors，preserve原motor_response_enabled（不顺手关掉腿响应）；有现有wheel响应wrapper则安装器拒绝，避免重复或改变腿链路。目标控制元件Kp0.2/I0/limit3是工程起点，非实机标定。

v215原生24env、3命令（1/2m/s直行和1m/s+0.8），计划1500steps但500steps时三组各8早期tilt满足评估结束计数。三组survival0、无steady；不能直接切换。2001次physics compute真实max effort3.0Nm、参考max55rad/s、饱和joint-step比例0.050766；读回旧D[0,0]、effort[3,3]，接口限幅确实生效。失败不是训练失败或速度环不可行，只说明冻结旧策略直切Kp和limit不适应。证据v215/{target_velocity_probe,transfer_summary,status}.json。

增加--diagnostic-wheel-velocity-fraction及apply_velocity_profile(...,migration_fraction)：f0恢复等效Kp0.1/limit5.5，f1到Kp0.2/limit3，reference scale55保持；I/滤波按f从0渐入。f0.05为0.105/5.375Nm，f0.1为0.11/5.25Nm，过渡限幅超过3必须标仿真阶段，不能实机直接采用。profile校验完整后才修改，腿/gas/旧action尺度及motor_response标志不改变。控制器测试累计15项通过（含5个fraction、leg参数与气弹簧保持、单位/timing拒绝、真实3Nm执行、诊断不可认证），相关motor9项此前同时通过。

v216 worker1982500已串行启动f0/f0.05/f0.1同5084/seed64、上述三命令各8replicas，先判断保留表现与迁移余量；无新训练、自动晋升或默认控制模式切换。结果尚待完成，最新看artifacts/continuation_v216/status.json。用户允许新技能跑满原预算的规则保持，评估结束规则按事先指定完整回合数，不是训练早停。

### 177. 用户暂缓轮速迁移，确认RL直接轮力矩、LQR仅对照/备用

用户说明此前实车LQR由状态反馈输出力矩，没有现成可复用闭环轮速接口，要求先不急迁移。进一步明确选择“RL直接输出轮力矩，LQR作对照或备用接管（延续当前训练接口）”。后续部署以直接轮力矩为准，不继续按速度环作为主线，也不将LQR静默接在RL下面。configs/rmuc_hardware_target.yaml增加deployment_contract，docs/rmuc_deployment_control_contract.md记录完整执行链、单位及未完成项；速度原型只保留实验，默认控制一直未切换。

收到暂缓时重新核实v216已complete，无需终止，无后续GPU迁移或训练继续运行。user_deferral.json记录不自动恢复。f0/0.05/0.1三组均survival0，尤其f0也未复现旧策略长时行为，所以v214的513步逐compute公式等效不能被外推为v216完整闭环等效。原型仍有配置/执行闭环差距待以后排查，不将这批失败写成“轮速控制不可行”，也不继续为其消耗GPU。

用户选择直接轮力矩仍不等于裸ONNX动作×3即可部署：当前仿真在actor后有速度/pitch/yaw外环、通信延迟和力矩映射；外环有真值状态来源。这些环节需要明确复现或经训练迁出并重验，不能遗漏，也不等同用户LQR内环。现有LQR可用于力矩口径/状态估计/响应对照，接管机制尚未实机验证。辨识以有效力矩增益、死区、反转和响应为先，不要求用户先精确建模每个摩擦柔度项；DR须基于合理范围且不能弥补错误bank符号或错误动力上限。主线回到真实内倾姿态修复和直接力矩3Nm/最高DR部署契约。

### 178. 先克服压弯瓶颈：修复旧外倾附近bank惩罚平台

用户要求优先反思并克服简单压弯，不推进后续部署优化。暂停拟议privileged critic扩展，先审计实际奖励和腿动作可达性。v208完整100更新是在旧错误bank目标下，不能当修正目标后的训练失败；旧正向1rad单局部分支也不能覆盖双向全部内倾修复。

v217用v208运行标量执行当前真实reward AST，发现abs(bank_error)/6被clamp1，且Gaussian容差1.5°：旧外倾约9–12°误差区域几乎无bank改善信号。12→11.9°bank收益float32为0。将惩罚改为x≤1时x、x>1时2-exp(1-x)，保留6°内原式、连接点一阶连续且有界；weight2时远端惩罚上界2→4，几何和门槛不改。新收益0.0123656，45状态扫描中25个近目标状态逐值不变、远端改善均正、无总reward裁剪。14项相关测试通过，源码py_compile通过。env_cfg/train metadata标记linear_near_exponential_tail_v1。

原生80env诊断同5084，1m/s±0.8、腿差动0/±0.24/±0.30各8replicas1500steps，检查执行限幅下姿态方向；不是训练、不晋升。初拟0.48被既有CLI限值拒绝，在启动仿真前失败，已改为0.30且保留拒绝日志，未放宽限制。状态/结果见artifacts/continuation_v217/{status,reachability}.json。完整分析docs/rmuc_bank_bottleneck_review.md；旧动作饱和、小std冻结、fine-yaw速度资格和COM配合仍需逐项验证，不能宣称奖励改动已解决技能。


原生结果补记：v217全部10组各8回合生存1，综合门槛全部失败。正yaw正确腿差动0→+0.30，bank误差9.7016→5.7415°，线速度误差0.1105→0.2791m/s。负yaw差动−0.24→−0.30，bank4.6790→4.6646°几乎不变，速度误差0.1421→0.2119；left动作原始1.2135→1.2370而执行始终1.0，明确出现方向性执行余量限制。不能把这个结果外推为全关节物理不可达：只扫描了一个两关节差动方向。下一步优先在失败命令域让左右腿共同姿态与轮补偿协同、核验实际关节目标余量和探索范围，而非继续堆常数差动。reachability_summary.json留存紧凑证据；worker已正常结束，未启动训练，无新策略权重。


### 179. 用户明确最终E2E：剥离任务级外环，不将teacher复现当部署

上一目标轮完成v217奖励平台修复与80回合原生诊断，属有效进展；最新用户要求明确最终E2E。更新hardware/deployment/long-term契约：完整组合actor输出腿位置与轮力矩，速度/pitch/yaw/推进外环只能过渡teacher，最终必须移除；允许状态估计、单位换算、底层执行、限幅急停，LQR独立备用。

新增policy_outer_feedback_enabled（历史默认True）；False时生产_compute_policy_action仅清洗/clip策略动作，并清空全部外环诊断，不读取_state/command/robot。训练metadata记录开关；同时禁止基于真值速度的轮阻尼补偿路径。独立诊断CLI --diagnostic-disable-outer-feedback关闭全部任务外环，报告实际标志且passed强制False，不能误认证E2E。15项动作/几何/奖励测试通过，实际AST在根本不存在状态/机器人对象时验证精确动作恒等与诊断归零。

v218同5084，1m/s±0.8各4env/128控制步原生短探针完成，外环标志False，原配置damping compensation ratio0。短探针只验证无外环执行链可运行；不满足完整episode/稳态验收，未训练、未取得E2E压弯能力。后续仍优先压弯：针对动作饱和与腿轮协同准备局部学习，并沿外环移除目标验收，不以旧外环通过替代。证据artifacts/continuation_v218/e2e_probe.json。新增阻尼补偿守卫在探针后补齐；探针当时该ratio已为0，没有额外补偿生效。


### 180. 完整E2E对照证明旧策略转向主要依赖任务外环

v219同5084、同seed62、同命令1m/s±0.8与场景，每方向16完整回合/1500步；带外环与无外环initial_state_audit存在且逐项相等。两组双向生存均1。带外环yaw误差0.04694/0.03765，关闭全部任务外环后0.80139/0.79789，实际body yaw约−0.00147/−0.00220rad/s，几乎没有转向。带外环bank误差9.7107/6.7250°仍失败，无外环4.6685/4.6539°也未通过。不能以姿态更直立冒充压弯改善：此时转向已消失。

因此后续不只做“腿姿态小修”：首先需在失败命令域学习轮差动力矩/转向，再联合正确bank与腿轮协同，冻结旧参数同时明确新适应分支的动作余量和探索。现有单正向1rad的小分支及冻结std不等于具备这项适应条件。最终仍必须无任务外环、完整组合图和最高DR验收；本轮只是旧动力nominal因果对照，不是3Nm最终认证。

用户问1rad/1.2rad百分比：v208 +1rad旧目标训练100updates已全部执行（90策略学习），但技能未过；±1.2仅旧baseline评估未完成适应训练。按正确bank+E2E最终标准，±1/±1.2四目标点均未认证，0/4不等于整个项目工作量完成率0。不得提供未经定义的“七八成”学习进度。当前无后台trainer；v219两组评估均已结束。证据artifacts/continuation_v219/{e2e_baseline,teacher_baseline,comparison}.json，docs/rmuc_skill_status.md已标明E2E缺口。

### 181. E2E轮转向学习落地：配对采集、冻结旧图的双向初始化与完整图导出

v219已完成完整因果对照，下一步不再只加腿偏置。新增evaluate --collect-wheel-teacher，仅DR0/有外环/无动作偏置；每8控制步保存actor实际25D输入、raw action与reset前已执行的两轮归一化动作，记录episode步数与done。只提供轮目标，不模仿旧外倾腿目标；采集作为诊断不认证。v220同5084、seed65，1m/s±0.8/±1/±1.2，各8replicas1500步，获得9024条配对样本。CPU重放actor与raw action最大误差3.58e−7，证明观测—actor动作对齐；通信delay实际全0。

新增可选e2e_positive_1p0/e2e_negative_1p0两独立分支，只有含对应key的checkpoint才重建，旧checkpoint默认不启用。复用速度0.8..1.2、yaw幅值0.4501..0.6平滑支持；±0.45及域外输出保持。新增每动作0.3局部预算明确单列，不修改旧预算或执行clip。两分支六动作保留给后续腿轮联合PPO，本次只以两轮监督loss更新，腿输出同观测逐值未变。

CPU500次监督更新（不是PPO），每命令最后2/8replicas留出，共6768训练/2256验证；轮执行动作RMSE0.18226→0.01632。旧参数与buffer逐项一致，66受保护固定探针输出逐值一致；新delta未超0.3、保存后完整重建逐值一致。该留出集不是独立seed行为泛化证明，不将拟合改善当技能通过。候选artifacts/continuation_v220/wheel_initialized.pt，sha927a7082…，丢弃旧optimizer且标明需reset；原5084不改。

22项router/E2E动作测试通过；完整图JIT/ONNX导出校验见v220/export/export_parity.json。原生无外环六点各8replicas/1500步已启动，完成结果见e2e_initialized.json；不能因为离线拟合好而跳过闭环检查。最终3Nm/真实damping/最高DR与多seed验收尚未完成。


v220原生无外环评估已完整结束，六点各8回合全部survival1；yaw误差按+0.8/−0.8/+1/−1/+1.2/−1.2为0.0280/0.0789/0.0293/0.0399/0.0616/0.1595rad/s。说明轮初始化已在闭环中产生转向，不能再把v219“旧5084不转向”当新候选现状。与v219不同seed/场景，不作严格配对收益估计。所有bank仍失败（约7.46–14.01°），+1/+1.2速度误差0.2357/0.3497。48回合只证实nominal初步转向恢复，不认证E2E压弯或晋升。完整图JIT误差0，ONNX最大2.38e−7。behavior_summary.json保存结果。下一步从此候选启动无外环的正确bank腿轮联合适应；本轮监督训练及两项原生任务均结束，没有后台PPO。

### 182. 无外环腿轮联合PPO已启动：100更新预算，双向1.2主任务与0.8/1.0复训

接续v220已具备初步自主转向但全部bank失败的候选，不再依赖外环改善指标。v221从wheel_initialized.pt（sha927a7082…）继续，明确policy_outer_feedback_enabled=False、正确物理bank与dense tail奖励、DR0旧动力作为当前姿态学习阶段。128env、每更新128步、max_iterations100，其中10 critic warmup，其余策略学习；不设置平台早停。主任务stage50实际为1m/s双向1.2rad，35%rehearsal优先stage48/49（双向0.8/1.0，priority fraction0.8），15%连续命令采样保留；课程与retention门槛不放宽，达到51才可依预定完成条件结束。

只开放e2e_positive_1p0/e2e_negative_1p0两分支，旧确定性策略参数与buffer冻结；各分支0.3新增预算保持。为姿态探索显式将固定腿std从0.004调整0.03，轮std设0.012；这是探索分布的明确变更，不伪称旧std逐值保持。初次全动作0.03被既有high-speed guard拒绝、尚未训练，日志已归档startup_std_guard_*；保留守卫，改为允许的轮0.012后重启，没有偷偷放宽保护。初次nohup未留下活进程/claim，经ps核实后改用持久exec会话启动，没有并行重复训练。

实际worker2012827、trainer2012829，run logs/rsl_rl/rmuc_wheel_leg_course/2026-09-26_12-01-22_e2e_joint_bank_v221。已读回运行env.yaml确认外环False、stage50、DR0、rehearsal/continuous和奖励版本；最初2次更新记录存在，两方向新分支都有采样、旧参数/buffer无变化；当时仍处critic warmup，不能称策略已优化。startup_verified.json留存快照，后续以进程和实际jsonl为准。

worker在训练结束后检查旧图隔离（只允许新分支及明确std变更），导出完整图并自动做六点无外环每点32回合评估，不自动晋升；status.json/训练日志记录失败或完成。任务目前未验收通过，最高DR/3Nm/实测damping仍未完成。

启动后补查：已越过10次critic warmup，观察到14次总更新、其中4条存在策略梯度minibatch记录；旧参数/buffer仍无变化。trainer2012829当时仍在运行，详情live_learning_check.json。此为学习执行证据，不是行为通过。

中途CPU快照核验：model_5100.pt（sha a98065fe…）两E2E分支各20参数张量已变化，旧确定性参数/全部buffer逐项不变；明确std分布变化为腿0.03/轮0.012。1131固定探针中1061受保护探针输出差0；两方向均存在执行clip后的有效动作变化，仍不等于真实姿态改善。新增audit_snapshot.py可复用，报告snapshot_5100.json。随后进程复核仍存活，已观察32次更新，继续100预算，不重复启动、不提前晋升。

### 183. v221运行中断后按完整checkpoint恢复，不重头训练

重新核验原worker2012827与trainer2012829均不存在，原status仍写training已纠正为interrupted_process_missing；日志到59次更新（5143），无异常栈，不能推断正常完成/OOM。最近完整checkpoint5125（保存41次更新），其后18次仅有日志、权重未保存。以checkpoint5125为源保留actor/critic/optimizer，执行剩余59次至原100次有效权重预算，critic warmup0，不重复预热；不可将丢失更新算作保留进度，实际GPU计算需额外计入18次中断损失。

恢复脚本/证据位于artifacts/continuation_v221_resume，使用独立session子进程启动，避免依赖工具会话生命周期。初次恢复脚本的全局run_name替换误改了源checkpoint路径，在任何训练前被hash读取检查拒绝；已恢复精确源路径并存档startup_path_error_*，无重复训练。旧完整图不改，原无外环/双向分支/探索/奖励/命令预算条件保持；后续仍自动导出和评估六命令点。

恢复已确认实际进入学习：worker2033509/trainer2033510，run2026-09-26_12-11-06_e2e_joint_bank_v221_resume，从5126开始更新、首条即有策略minibatch（没有重复warmup），外环False，旧参数/buffer检查通过。启动核验见startup_verified.json；后续进度需实时读，不按此快照推断完成。

### 184. E2E联合训练完成完整预算但六点失败：速度改善伴随正向yaw退化，不晋升

v221/v221_resume最终model_5184.pt，sha b0505769…；final_evidence_audit.py核对保留更新5085..5184连续100次、策略90次，实际执行118次含18次未保存损失。旧确定性参数/全部buffer保持、两新分支均更新，1061受保护固定探针输出差0，完整图JIT差0/ONNX最大2.38e−7。不可将CPU隔离、导出通过等同技能成功。

最终无任务外环、DR0旧动力，seed67、六点各32完整回合合计192，全survival1，但bank全部失败。按yaw +0.8/−0.8/+1/−1/+1.2/−1.2，线速度误差0.1986/0.0582/0.1445/0.0771/0.0073/0.0989，yaw误差0.5143/0.1250/0.5286/0.0128/0.4380/0.1189，bank误差5.0977/7.1551/8.2465/8.9097/11.5747/10.5046°。最终候选正向转向退化，不能将速度/倾角误差下降称为正确压弯。v220初始与本次seed/样本数不同，不作精确配对改善估计；新候选自身门槛已足以拒绝晋升。

重要后续：1m/s命令的额外fine-yaw项此前核查因速度资格未启用；本轮保持原资格做bank/腿轮学习，结果揭示速度与转向的取舍仍未约束好。先审核这条命令局部资格和真实reward取舍，再决定下一轮，不能照旧配置堆训练。保留v220轮初始化作为可回退起点，5184仅失败实验checkpoint，不覆盖已认证基线。worker与评估进程均正常结束，无后台trainer。最高DR、3Nm与实机契约未完成，完整长期目标继续。

### 185. 纠正低速大转弯奖励取舍，并从v220启动新对照预算

v222用v221实际运行标量执行生产_get_rewards的180个合成状态。固定外倾姿态、1m/s目标yaw1.2时，实际速度0.8→1.0且yaw误差0.05→0.45，旧总reward反而+5.7785；额外fine-yaw资格在1m/s为0。不能只按“bank误差下降”判断训练好坏，也不能仅开启旧相对yaw项就假定足够。

新增显式command_low_speed_turn_yaw_tracking_enabled，默认False保证旧实验可复现；开启时仅0.8<|vx_cmd|<1.2且|yaw_cmd|>0.4501的失败扩展域启用原相对fine-yaw项，并加绝对门槛项1-exp(-max(|yaw_error|-yaw_gate,0)/0.1)。使用既有fine-yaw weight（运行5），相对+绝对两项总分上界2、惩罚上界10；原bank尺度不改变，速度/安全奖励不删除。新相对项保留大误差方向性，绝对项对齐0.05rad/s门槛，不放宽验收。开启后的同一速度换yaw取舍reward−0.7965。180状态扫描无clip，16项奖励/几何测试通过，含双yaw方向、受保护速度/±0.45逐值不变、active/rehearsal同命令一致。不是物理成功证据。

v223从v220 wheel_initialized.pt（sha927a7082…）重新对照，不从退化5184继续。原100更新/10critic warmup、双向新分支、stage50、35%rehearsal和15%continuous、DR0、无外环、腿std0.03/轮0.012均保留，仅启用上述奖励开关；checkpoint保存间隔25→10以减少再次中断损失，不改变学习预算。worker以独立session启动，结束自动检查隔离、导出并评估六点各32回合，不因平台早停、不自动晋升。新训练metadata记录开关，状态见artifacts/continuation_v223/status.json。当前仍没有完整E2E压弯通过证据。

v223继续核验：trainer2046672实际存活，已观察25次更新、其中15次策略更新，旧参数/buffer检查通过。准备audit_and_compare.py在完整训练/导出/六点评估结束后核对100/90更新和相同评估seed/初态；脚本尚未产生最终报告，不把准备工作当行为验收。状态见live_learning_check.json。

### 186. 新yaw约束完成100更新与192回合：转向被守住，但正确内倾仍全部失败

v223完整连续100次更新（90策略学习），最终5184 sha8c5ed34a…；实际训练开关True、无任务外环，旧确定性参数/buffer未变，完整图导出通过。audit_and_compare.py已执行，final_evidence_audit.json核对同checkpoint、全部六点每点32回合、与v221相同seed67及逐项相等initial_state_audit。两次最终行为可比较，但v221训练曾中断，不能称完全一致的无中断训练消融。

新结果按yaw +0.8/−0.8/+1/−1/+1.2/−1.2：线速度误差0.0094/0.0832/0.1297/0.1069/0.2423/0.1265；yaw误差0.0567/0.0997/0.0290/0.0184/0.0522/0.1402；bank误差9.5080/7.3476/11.9536/9.0474/13.8874/10.7052°。192回合全部survival1，但六点整体都失败。+1yaw从上一轮0.5286降到0.0290，+1.2从0.4380降到0.0522，证明策略不再主要通过丢转向换取速度；同时bank仍外倾且+1.2仍欠速。不能将“修复奖励取舍”写成“压弯完成”。

该候选不晋升，无后台trainer/评估继续运行。下一步优先检查四腿动作组合、执行饱和与姿态可达性：此前只测一个两关节差动方向，不能据此断言整个动作空间做不到；也不能继续无依据叠加yaw/bank权重。保留25D/6D和USD，E2E无外环路线不变。最高DR/3Nm/多技能部署目标仍未完成。

### 187. 四腿动作基底与起稳后平滑对照：不能把正倾角或瞬态失败当可达性结论

v224同v223最终5184，1m/s±1，9组（基线、第一对/第二对/两对同向/两对反向，各±0.3），每组每方向6replicas共108env，1500steps，无外环/DR0，不放宽现有clip或USD。完整结果leg_modes.json/summary.json。没有组合通过；+yaw两对共同正差动使world bank从约−6.116°到−0.037°且yaw误差0.019，但速度误差0.270，仍未达到目标+5.82°。第二对正差动等多组reset即tilt，不能据此证明全动作空间不可能内倾。

新增显式诊断偏置时序：--action-bias-start-seconds与--action-bias-ramp-seconds，默认0保持旧路径；每episode按年龄先等待再smoothstep，reset正确归零。仅有action-bias-sweep可用、范围0..5s，诊断仍不能认证。v225以完全相同seed/布局和偏置，先原策略2秒，再2秒平滑施加，初始state audit相等，基线统计逐值相同。第二对正差动/+yaw生存0→5/6，说明突变确实影响部分失败；其他多组仍tilt或破坏速度/yaw。所有组合整体仍未通过，偏置响应包含旧策略反馈反应，不是纯逆运动学工作空间证明。

v225该组虽报告world bank均值+5.589°，但实际heading vx均值−0.762m/s（命令+1），bank绝对误差仍6.229°。此前physical_inward_lean_deg只乘sign(yaw)，倒退时语义错误；禁止据此宣称已实现正向内倾。已改只读诊断为world_bank*sign(actual_heading_vx*world_yaw)，加入contract标签signed_actual_heading_speed_times_world_yaw_v2；它仍是v*yaw稳态proxy，不是精确实测横向加速度。奖励、门槛和策略不改。原始v224/v225报告保留，inward_semantics_correction.json勘误，不用平均值乘符号假称重算了逐样本统计。

14项诊断/时序/几何测试通过，包含倒退、零速、延迟起点、partial reset及旧时序保持。两个原生worker均结束，未训练新checkpoint、不晋升。下一步需区分腿姿态改变时的轮力矩配合、执行限幅和旧策略反向响应，不能继续仅凭常数腿差动或单一倾角均值下结论。

### 188. 用户确认腿电机DM-J8009P-2EC/9:1：真实限速修复、armature核实与腿DR落地

用户指出腿velocity60/V14为17并怀疑armature。核查实际asset factory/安装IdealPD.compute发现velocity_limit没有形成速度硬约束，原生solver读回17453.293rad/s；仅改60文本无效。另一方面RMUC腿armature从USD继承0.010kg·m²，并非0。V14_2显式DM8009_ARMATURE=1.95e−4×9²=0.015795是代码参考，具体P-2EC型号的惯量/极限转速尚未实测；外部检索工具404未获得厂家确认，不能冒称厂家参数。型号与9:1为用户明确确认。

v226同v223策略1m/s±1，seed71每点4replicas：baseline、仅solver17、17+参考armature各1500steps。每physics步6001次compute记录基线峰值2.7784rad/s、steady峰值0.8952、超17次数0。仅限速17后真实读回17，初态及速度/yaw/bank/动作/关节位置统计逐值相同。参考armature实际读回0.015795，bank误差12.0734/9.1397°，基线11.9485/9.0482°，均存活但没有解决外倾。故修限速必要，但这组压弯失败不能归因“腿跑到60”或“没有armature”。

默认rmuc_wheel_leg.yaml现腿velocity_limit17、velocity_limit_sim17、armature显式0.010，保留原PD60/2及40Nm，USD/轮不改。asset factory支持真实solver limit和armature键。新增leg_actuator_dr模块，每reset按当前DR strength将nominal插值至速度[12,17]、armature[0.008,0.024]的独立四关节抽样，并写入实际solver；默认随既有DR启用，strength0精确保留nominal且不消耗随机数。范围明确工程先验待辨识，不是已标定硬件范围。独立rmuc_leg_actuator_target.yaml记录型号/减速比及可选V14惯量profile，不自动替换默认0.010。硬件target文档同步腿电机信息。

原生DR1/128step探针读回速度12.0291–16.9346、armature0.008412–0.023568，证明真实写入，不仅配置字符串变化；短探针没有完成回合，不用survival0宣称摔倒或策略通过。3项采样/零强度/插值/配置保持测试通过。新增诊断CLI、physics速度审计、profile helper与docs/rmuc_leg_actuator_audit.md；原始证据v226，不以此次诊断替代新联合DR训练认证。

默认配置最终原生短验证完成：513次physics compute，四腿真实solver限速17、armature0.010读回一致；implementation_summary.json记录。腿DR/奖励/E2E动作相关15项测试通过，diff检查通过。此次全部诊断结束，无新训练启动。

### 189. 分离旧腿反馈诊断：不能简单固定或移除腿闭环，转查位置目标余量

v227在当前真实腿solver17/armature0.010下做同seed68/相同初态的live与held配对，每组9偏置×双yaw×6replicas=108env，1500steps。held在2秒截取当时clip后的四腿动作，之后保持该参考再叠2秒smoothstep偏置；轮动作仍由完整actor实时输出。新增HeldLegReference只供显式诊断，逐episode reset释放参考、轮槽不改，必须关闭任务外环且整体不能认证。6项时序/锁存测试通过；默认评估/训练路径不启用。

两组均完整结束、无新训练。live腿物理峰值最多12.0819rad/s，低于17；基线bank正/负约11.9361/9.0466°。held在部分此前失稳的负向组合中恢复生存，但+yaw无偏置基线从生存1降0，说明原腿反馈承担平衡，不能简单删掉它或声称它只会抵消内倾。held稳定组合仍无正确压弯通过，部分姿态均值改善伴随yaw/速度退化。所有原始结果和comparison.json保留，初态逐项相等；本实验改变控制策略，并非纯机械工作空间证明。

下一步停止扩展相同常数偏置扫描，核对四腿目标角、实际角、PD计算/实际力矩及位置目标边界：当前1m/s腿动作scale0.22，已有左腿clip饱和与关节负载偏差，需要区分命令余量不足、静态柔顺与学习协调。尚不能凭角度差断言电机力矩不足。旧参数、USD和生产控制链不改，当前无本任务后台进程。

### 190. 腿目标余量与关节速度一致性实测：发现TGS速度偏差的重要来源

v228同v223 checkpoint、无外环/DR0、当前17限速、1m/s±1每点4回合1500steps。物理步记录确认left_active目标始终0.22rad、action饱和占比1；实际均值约0.511/0.374rad，腿computed/applied effort一致且40Nm饱和占比0。限制更像位置命令余量而非力矩已满，未擅自扩大动作范围或提高PD。

进一步同条件记录目标速度/前馈均0，+yaw左主动关节实际速度平均0.717rad/s，而位置差分平均0.000295；−yaw对应0.336/−0.000589。这是明显速度—位置不一致，既影响PD阻尼也可能进入actor关节速度观测。审计读数均在同一compute输入时刻，目标在IdealPD原地清空前保存，逐关节按执行器名字对齐；位置差分排除reset跳变。4项审计/腿DR测试此前通过。原v226对称限速/armature读回不受原展示顺序差异影响，新报告显式记录policy slots。

v229只诊断性开启当前TGS的enable_external_forces_every_iteration。安装PhysxCfg源码明确说明该选项有助速度更新精度；默认False，原日志有noisy velocities警告。初次physics config尚为None导致仿真前失败，已存档startup_none_physics.log，显式构造默认PhysxCfg重试，其余求解参数未改。同checkpoint/seed/初态对照后，+yaw左主动速度差分绝对误差从约0.717降到约0.02rad/s，其他关节也减小；详细逐方向均值见v229/comparison.json。速度一致性改善，但旧策略正/负bank误差仍13.0097/8.7774°，8回合生存1，不认证技能。

新增显式诊断CLI，结果强制不认证、记录实际solver开关；默认生产solver暂不切换，USD/actor/奖励均不改。全部评估已结束，无新训练。后续先在其他命令与闭链误差/速度层继续验证物理一致性，再迁移训练；不能继续把旧solver异常当实机电机阻尼来辨识或用DR掩盖。

### 191. TGS速度一致性扩展验证：另一seed复现，非转弯E2E缺口明确

v230同v223最终checkpoint，seed73六命令[0,0]/[1,0]/[2.5,0]/[1,+1]/[1,−1]/[−1,0]，每点4replicas、1500steps，无外环、DR0、solver17/armature0.010；仅对比external_forces_every_iteration False/True，初态逐项一致。两个worker已完整结束，未训练或改默认solver。

双向转弯各4完整回合生存1：+yaw四关节速度一致性绝对误差旧[0.7153,0.0904,0.1979,0.0629]→新[0.0168,0.0242,0.0045,0.0040]rad/s；−yaw旧[0.3372,0.0614,0.5855,0.0823]→新[0.0047,0.0037,0.0093,0.0079]。倒退命令组也从主动腿约0.476降至0.0043/0.0021，但实际速度跟踪严重失败（新线速度误差1.1697），故这是“倒退命令条件下”的一致性证据，不是成功倒退运动证据。

零速旧组虽生存1但漂移速度误差0.598，新组tilt而无稳态样本；1/2.5直行两组均无稳态样本，不能用零样本量宣称一致性改善。compare.py按实际physics稳态样本数加权，无样本输出null。该候选无外环的非转弯技能明显未建立，旧带外环证据不能替代E2E验收。

结论：修正TGS开关对已取得有效稳态的命令和第二seed确有一致性收益，但不足以宣布整个闭链模型精度或全技能通过。暂不全局切换并沿用旧认证；下一步需修正物理下的训练/适应并补足非转弯E2E能力，不能用DR掩盖系统偏差。原始与比较证据artifacts/continuation_v230/{baseline,tgs_force,comparison}.json。长期目标保持，无候选晋升。

### 192. 修正TGS物理下的无外环适应训练已启动

v230证明多个有效稳态条件下关节速度一致性显著改善，同时旧策略非转弯E2E缺口仍在。本轮不再仅诊断，显式启动v231物理适应：从v223最终5184（sha8c5ed34a…）继续，仅两E2E双向局部分支可训练，旧确定性图冻结；TGS external_forces_every_iteration=True，任务外环False，保留新低速yaw约束、正确bank目标及dense tail，当前腿solver17/armature0.010、DR0。

新增trainer显式--rmuc_tgs_external_forces_every_iteration，环境建立前配置PhysxCfg并记录training metadata；默认不带flag的历史路径不改。运行params/env.yaml已实际读回True/无外环False/新yaw约束True；GPU原生worker2447478/trainer2447479已启动，run2026-09-26_21-43-44_e2e_tgs_bank_v231。100次更新预算、10critic warmup、90策略更新，重置optimizer用于新物理价值校准，checkpoint间隔10，不因平台早停。

主任务1m/s±1.2，rehearsal优先±0.8/±1，连续采样保持；该局部分支不能自动解决非转弯E2E缺口，不将本轮成功（若有）替代全任务。训练结束自动完整图隔离/导出，再用同样TGS True六点各32回合无外环评估并记录腿速度一致性，禁止在旧solver下评估新训练。脚本与证据artifacts/continuation_v231，当前仅启动验证，尚未取得行为改进。

v231后续进程核验：观察25次更新，其中15次策略更新，旧参数/buffer检查无变化，trainer2447479真实存活。新增audit_final.py核对源checkpoint增量100/90、完整导出、训练与评估solver一致以及六点完整回合；尚未执行最终审计，不能据此声称通过。

v231中途model_5210只读审计完成（sha961c7e9c…）：相对v223源两分支各20张量变化，旧确定性参数、全部buffer、std均保持；1061受保护探针输出差0，局部动作变化经clip仍非零。报告snapshot_5210.json。后续观察36次更新，trainer2447479仍真实存活；内倾训练EMA仍超门槛，未宣称物理通过。

### 193. 修正TGS下100更新/192回合完成：速度一致性保留，位置动作饱和仍阻碍内倾

v231最终model_5284.pt sha a0e50e48…；audit_final.py实际执行确认100连续更新（90策略学习）、相对v223源只有双向新分支变化、完整图导出通过、训练与评估均TGS external forces=True/无外环。六点每点32回合合计192，全部survival1，正确bank全部失败，不晋升。

按yaw+0.8/−0.8/+1/−1/+1.2/−1.2，速度误差0.0547/0.0238/0.0776/0.0429/0.1888/0.0667；yaw误差0.0541/0.1072/0.0245/0.0119/0.0484/0.1317；bank误差9.9808/6.9149/12.4584/8.6132/14.3668/10.3644°。不能用±1跟踪较好替代整体压弯认证，−1.2yaw仍失败。

最终physics审计各命令按真实稳态样本数加权，速度—位置差分平均绝对误差约0.0037–0.0455rad/s，较旧solver的大偏差明显缩小。但六点left_active_link_pitch action saturation均1.0、四腿40Nm torque saturation均0。说明已修复的速度一致性问题并未解除位置命令边界；下一步针对局部位置目标余量设计可逆适应，而非继续相同配置堆训练。leg_physics_summary.json/最终审计完整保留。

worker、trainer和评估均已正常结束，无本任务后台进程。默认历史solver未全局切换，v231使用显式迁移标记；后续物理适应必须继续相同修正设置，不能倒回旧solver伪造改善。非转弯E2E、最终3Nm/全DR与完整部署仍未完成。

### 194. 位置目标余量原生对照：改善内倾但须腿轮协同，不能额外偏移冒充E2E

v232同v231最终5284、修正TGS True、无任务外环、solver17/armature0.010，1m/s±1，基线及左右active关节目标差动±0.05/±0.10/±0.15rad，每点8replicas、112env、1500steps。起步2秒后2秒smoothstep偏移。新增显式--diagnostic-leg-target-offsets，JSON四腿目标偏移在策略动作clip后、物理关节限位前生效，最大0.15rad；禁止混入raw action bias，必须无外环，结果不能认证。生产路径未开启、USD/马达/旧权重未改。7项时序/目标边界测试通过。

负yaw正确差动−0.15使left_active目标从0.22到约0.37rad，bank误差8.6318→3.1879°，survival1且腿力矩饱和0；但速度误差0.2633、yaw误差0.3356，未通过。正yaw相反差动+0.15使bank12.4324→4.6227°，速度0.2023/yaw0.1236仍失败；+0.10时速度误差0.0059但bank5.8362、yaw0.0626仍失败。负yaw错误方向+0.10/+0.15发生tilt。所有组合综合门槛均未通过，不能宣称动作空间已充分可达或技能掌握。

实测支持位置目标余量是有作用的控制自由度，尤其负方向被clip上界限制；也证明不能只增大范围而忽视腿轮协调。下一步设计局部、方向独立且可学习的目标范围迁移，保留25D/6D与旧图冻结，完整部署必须显式包含最终动作映射；不将诊断偏移当额外部署控制。原始leg_modes.json、加权summary.json及worker证据在v232，全部进程已结束，无训练、不晋升。

### 195. 可学习位置余量原型与零初始化执行等价证明（尚未接入生产）

直接放大leg scale会重解释旧动作/历史动作，直接放宽clip会使旧超限raw突然执行。新增未自动安装的rmuc_position_authority.py原型：适应域先保留旧clip执行值，再加双向独立可学习修正；active槽0/3新增0.15/0.22单位预算、轮槽2/5为0.06、hip不增加，active最大1.6818对应0.37rad，其他槽保持1。域外raw保持且consumer仍旧clip；不改USD/旧参数/25D/6D个数。并非已完成部署映射。

v233在真实v231源5284上1131固定探针证明零分支执行动作逐值一致，1061域外raw逐值一致；head JIT batch1/7/1131一致。3项测试通过，包含饱和处新分支非零梯度、方向隔离与轮限幅。prototype_head.pt仅原型，不是完整策略导出。

未接入生产actor/trainer，不启动PPO。需先完成checkpoint重建/完整图导出、wrapper与env逐槽限幅、previous-action一致性、原生确定性对照和实际Gaussian logprob语义。局部mean clip会改变随机分布，不能从确定性等价推断训练采样等价。docs/rmuc_position_authority_contract.md明确待办，artifacts/continuation_v233/audit.json保存证据。当前无后台训练，完整压弯仍未通过。

### 196. 位置余量head接入完整图重建/导出，运行consumer仍待接通

在完整actor的旧router合成后、Gaussian distribution更新前调用可选rmuc_position_authority；无head的旧checkpoint路径不变。新增显式安装器，restore_actor按checkpoint key重建并strict load，未静默修改旧图。v234从v231最终5284生成zero_authority.pt（sha6c8f8505…），保留旧573状态张量逐值一致、丢弃optimizer并标明需重新初始化与运行consumer未验证。

1131真实策略固定状态验证零head下旧执行动作逐值一致、1061域外raw逐值一致，保存重建结果逐值一致。完整组合图零head及人为非零双向head各做JIT/ONNX batch1/7/667校验，JIT差0、ONNX最大2.38e−7；非零诊断checkpoint明确不是学成策略。24项position authority/router测试通过。每次导出增加action_contract.json声明局部active限幅1.6818、轮/hip限幅1、legacy单位、尚未原生验收；不是只导出head。

尚未安装至trainer/env运行consumer。evaluate当前uniform clip1不兼容，已在发现新head时显式拒绝而非悄悄吞掉新增余量。后续需统一wrapper/env限幅、历史动作、PPO实际Gaussian样本/logprob，再原生零初始化对照及完整预算训练。确定性等价不意味着随机采样等价。当前无后台训练，无新技能证书；docs/rmuc_position_authority_contract.md及v234证据记录待办。

### 197. 位置余量consumer接通环境，原生零初始化确定性对照通过

新增env.position_authority_enabled默认False与显式eval --position-authority-consumer。新consumer关闭RSL wrapper统一clip1，环境按缓存的实际actor输入观测确定局部扩展限幅；不读取新命令/真值来替代延迟观测，要求无任务外环且legacy clip1。active腿局部limit1.6818，wheel/hip保持1，域外旧limit1；nan/inf处理保持保守原值。原_apply_action映射、关节margin和previous-action执行值来源不改，保持25D/6D。checkpoint有head与consumer开关必须成对。

5项位置余量/真实env动作AST测试通过，包含随机动作跨各域逐值符合prototype消费者与旧E2E路径。v235同当前修正TGS、DR0、同seed73，旧v2315284与v234零head候选分别1m/s±1各4完整回合1500steps，初态逐项相等。生存/速度/yaw/bank/执行动作/关节位置/转弯物理诊断逐值相等，全部steady physics腿目标/实际角/力矩均值也相等。comparison.json记录证据，不只比较raw网络输出。

该结论仅证明确定性零head运行兼容，不代表随机策略等价或新技能完成。尚待非零head原生余量读回、trainer wrapper配置/新head优化器隔离、PPO真实Gaussian样本logprob语义核验。未启动PPO、无后台任务，不将人工非零测试checkpoint作正式候选。新增消费者不改变默认旧策略路径。

### 198. 非零位置余量物理读回与实际PPO采样语义验证完成

v236使用明确diagnostic_nonzero_head人工参数（非训练策略），8env/128step、修正TGS/无外环/position consumer原生探针。phys compute目标峰值按left_active/left_hip/right_active/right_hip为0.364604/0.034310/0.252816/0.159517rad，active确实越过旧0.22且未超0.37。证明额外余量送达执行器，不能当技能成功或完整回合认证。增加max_abs_target_position_rad只读审计，相关6测试通过。

CPU直接执行当前安装RSL-RL PPO.act与仓库_configure_env_action_credit（AST提取实际函数），使用完整非零head策略。第一次旧小std样本未触限，断言拒绝无效覆盖；改为仅CPU临时std0.5且不保存，再验证原Gaussian样本与logprob未被执行动作替换、prepare调用一次、未更新策略ratio精确1。限幅前后最大动作差0.63134，若错用执行动作logprob会偏差3.43549；本次实际存储raw logprob逐值正确。该测试不等于已验证原生trainer optimizer或随机策略与旧策略等价。

所有原生/CPU探针结束，无后台训练。下一步仍须接新head optimizer/冻结旧图、trainer consumer与恢复metadata，先原生少量更新检查，再启动100预算。证据artifacts/continuation_v236/{nonzero_native,ppo_sampling,summary}.json。未改变旧权重或训练noise配置。

### 199. 位置余量PPO训练链原生通过并启动完整预算

新增rmuc_position_authority_training.py与trainer --train_position_authority：显式开启env consumer，wrapper clip=None，仅head两方向参数与critic入optimizer，旧整个组合actor（含原router）冻结；不启用旧残差解冻schedule。恢复optimizer须有position_authority/critic组来源；采样沿用已验证raw Gaussian action/logprob、env缓存执行路径。每更新检查旧参数与buffer逐值不变、局部修正有限且不超声明预算；对无样本方向清空grad避免Adam动量漂移。

v237原生3更新（1critic预热、2策略更新）已完成而非失败：日志虽有启动插件告警，实际5285..5287更新与model_5287存在。核对40个新head参数张量变化，其余旧状态不变；env authority=True、外环False、TGS True、wrapper clip=None，optimizer仅新head/critic。完整图导出通过。5项head/optimizer测试通过（包括双向实际梯度更新与旧参数冻结）；v237/audit.json保存证据。

v238从v234零head候选重新开始正式100更新（前10critic），v237作为额外3次接线成本单独记录，不混算技能进度。主任务1m/s±1.2、rehearsal优先±0.8/±1、DR0、腿17/0.010、修正TGS、新yaw约束保持；保存间隔10，不按平台早停。worker2928772/trainer2928773已原生启动，读回运行配置与隔离检查通过。结束自动核对只有head变化、完整导出，再以相同consumer/TGS无外环六点各32回合评估，不自动晋升。

训练脚本与状态artifacts/continuation_v238；新head增加的是原动作单位内的可学习位置余量，最终部署必须包含action_contract，不能恢复统一clip1。当前只是训练链完成并开始学习，压弯/全DR/3Nm/非转弯E2E尚未认证。

v238进程续查：trainer2928773实际存活，已观察30次更新，两方向均有采样且head修正非零，旧参数/buffer逐更新检查通过。新增audit_final.py，待完整训练/导出/六点评估后核验100更新、旧图/预算不变、consumer/TGS一致及每点32回合；目前仅脚本编译通过，未冒称最终审计已执行。live_check.json保存进度。

### 200. v238完整结束、验收语义复核：解析bank不是已验证的必要条件

补记中断前已实际完成的v238结果：100更新，最终5384 sha f1423145…，旧图及预算不变、新head40张量更新、完整导出与训练/评估consumer和TGS一致；六点各32回合共192生存1，但历史综合gate全部失败。负向left_active目标均值仅约0.2237rad，额外位置余量使用较少，力矩未饱和。证据final_evidence_audit.json与authority_use.json真实存在，worker已结束，不再写“运行中”。

用户询问内倾及gate依据后，必须纠正前述过强解释：bank符号修复证明的是坐标几何正确，不证明atan(v_cmd*yaw_cmd/g)±3°是有轮距/独立腿的RMUC必须满足的机体姿态。机体roll、真实COM与法向载荷不等价，原解析角硬gate缺乏充分机构及任务依据。不得继续以“bank没过”等同“不会稳定转弯”，也不直接删除gate宣称整个任务完成。已通过异步问题请求用户明确“稳定跟踪姿态自主”或“机体明确内倾独立目标”；本条记录时答复待定。

v239只读审计原始checks与hash，不改奖励/课程/历史报告：v231±1只有bank失败；v238±1与+1.2只有bank失败，其余速度/yaw/生存/stable/height/回合覆盖均true。v238±0.8、−1.2仍有真实yaw/stable等失败，不可抹去。部分接触/侧滑指标仅稳态均值，准静态支撑proxy不等于ZMP/动态裕量，不据此认定安全或全DR通过。所有证据仍旧轮动力nominal，未覆盖最终3Nm/多seed/动态切换/完整非转弯E2E。

新增docs/rmuc_bank_gate_review.md与v239审计脚本/JSON，明确不自动晋升；没有启动v239训练，没有将尚未验证的内倾gate直接撤销。下一步依据用户目标确定姿态验收定义，并补最差载荷/侧滑/动态恢复证据，避免继续消耗GPU追逐欠依据的解析姿态。

### 201. 用户明确操作手控制、稳定跟踪为主：同步移除解析倾角硬目标并补载荷极值证据

用户明确最终部署RMUC轮腿步兵上赛场，由人控制；稳定跟踪转弯为主，姿态由策略选择，检查载荷、侧滑和稳定余量。这是用户目标澄清，不是代理自行降低成功标准。长目标的所有速度/转弯范围、起停倒退/连续动态命令/自转/最高DR/3Nm/完整E2E部署保持，明确加入操作手命令链验证；不改为自主导航，不假设键盘工具已等价实车遥控。

env新增command_require_analytic_bank=False默认。同步关闭bank跟踪奖惩、active与retention硬门槛、方向focus中的bank欠额；仍记录角度和真实物理指标，保留倾倒保护及一般姿态正则。历史True可复现原解析目标。evaluate默认operator_tracking_v1，--require-analytic-bank显式历史严格模式，报告记录契约，不重写历史passed。训练metadata记录开关。32项奖励/几何/课程gate/载荷极值测试通过，包含关闭后任意bank权重不影响reward、历史模式仍有效、active gate新旧模式，以及诊断单步失联/episode reset/无样本处理。

新增rmuc_turn_extrema.py，在控制步记录全程/稳态极值、低载荷/双轮接触不足/滑移/负支撑proxy事件比例及最长时长。阈值2N/0.1载荷/0.15mps仅诊断分类，不是事后验收门槛；8ms采样可能漏子步事件，support proxy不是ZMP。v240旧严格语义下同5384 seed75六点各8回合：部分点单步8ms低载荷、占比0.04%–0.29%；局部纵向滑移最大约0.078mps、heading侧速最大0.053mps、proxy最小约0.142m。不能由均值或极值正值直接认证动态安全。

v241新语义原生同初态重复同5384/seed75/六点各8回合，速度/yaw/生存/bank数值/执行动作/全部极值诊断逐值一致，证明只改验收含义未改冻结策略行为。新row通过−1/+1.2；+1在另一seed仍yaw/stable失败，±0.8及−1.2同样未过；没有宣称百分百掌握或整体晋升。最终报告仍因诊断物理/consumer证据范围不作为部署认证。全部任务已结束，无新训练。docs/rmuc_operator_tracking_contract.md与硬件target/长期目标/部署/技能状态/交接均同步用户决定，v241/contract_comparison.json保存同状态证据。

### 202. 按用户操作手跟踪目标启动v242，不再追逐解析内倾

从v238最终5384（sha f1423145…）继续新契约适应，不从零重建策略。显式command_require_analytic_bank=False，任务外环False、位置余量consumer True、修正TGS True；动作范围/leg17与0.010/DR0/网络预算/旧图冻结均保持。仅新authority head可训练，100更新预算（10critic预热、90策略学习），新奖励需重新校准critic并reset optimizer，不因平台早停。主stage50双向1.2，rehearsal优先±0.8/±1，原连续采样配置保持；本轮仍为局部跟踪适应，不能覆盖非转弯E2E或替代真实遥控动态验收。

worker2951897/trainer2951898实际启动，run2026-09-27_11-37-00_operator_turn_v242。启动params读回解析bank False、consumer True、outer False、TGS True；已观察3条预热更新记录、旧参数/buffer检查通过。结束自动完整图导出、同consumer/TGS六点各32回合operator_tracking_v1评估并保存载荷/侧滑/proxy极值。audit_final.py已适配新源checkpoint与契约，尚未执行最终验收。状态/命令/启动核验均在artifacts/continuation_v242。

用户关于赛场人工控制的要求已保持在operator_tracking_contract；本轮没有接入真实遥控，不声称失联接管/场地测试或最终3Nm/最高DR已完成。后续优先动态指令及多seed验证，不恢复解析bank作为必要成功条件。

v242运行续查：params再次确认解析bank=False/outer=False，已观察17条更新记录，日志越过10次critic warmup后actor_update_l2非零，两方向有采样，旧参数/buffer检查通过。trainer2951898真实存活。新head本来非零，不能用残差非零单独证明本轮学习；live_check.json保存事实，最终仍以六点/载荷诊断结果为准。

v242后续已观察81次更新且worker/trainer仍存活，逐更新旧参数及buffer检查均通过。补强audit_final.py：核对评估checkpoint hash与导出一致、完整六点命令覆盖、全程与稳态载荷/侧滑/支撑proxy极值有限且有样本、事件比例与持续时间有效；最终汇总保留失败checks及完整极值，不把诊断数据完整等同安全gate通过。脚本编译通过，最终审计仍等待训练/导出/评估结束；live_check.json为最近实际进程快照。

### 203. v242完成100更新与完整导出，六点行为评估运行中

已实际完成5385..5484共100更新，逐更新冻结参数/buffer检查全通过，worker检查仅authority head参数变化。最终checkpoint model_5484 sha256=8e5f7bac7f5a6c0c859988a9566c9c910757f858c8328d90e935137732327990；完整图JIT最大误差0、ONNX最大误差2.384185791015625e-7，export_parity.json passed=True。这仅证明导出数值一致，不证明行为达标。

同一worker2951897自动启动评估child2965963，seed67六点(1m/s,±0.8/±1/±1.2)，16replicas、3500steps、每点至少32完整回合。解析bank不作gate，无任务外环，authority consumer与修正TGS开启，载荷极值和腿执行器审计开启。实查评估进程存活，日志到1000steps；最终结果及audit_final尚待完成，不得从100%预算进度推导技能完成。live_check.json已更新训练结束/导出通过/评估进程证据。没有重复启动任务。

### 204. v242验收结束：局部改善伴随退化，发现rehearsal配置静默截断

训练、导出及评估均真实结束，final_evidence_audit.json已执行通过100更新/冻结图/完整导出/consumer与物理一致/载荷极值有效性审计。六点各32回合共192，生存全1；按operator_tracking_v1仅+1.2通过。yaw误差按+0.8,-0.8,+1,-1,+1.2,-1.2分别0.21194,0.18593,0.07664,0.07263,0.04300,0.05607。相比v238同seed/步数/初态，负1.2由0.11008改善，但±1及±0.8退化，不能晋升此最终checkpoint，也不原样续训。

稳态局部纵向滑移最大0.10871m/s，heading侧速最大0.05449m/s，准静态support proxy最小0.14029m；弱轮最小载荷比例出现0，双轮>2N不足事件约0.063%–0.167%，最长8ms；弱载荷比例<0.1最长16ms。仍是控制步诊断，不能认证ZMP/摩擦裕量/实机安全。完整极值见final_turns.json和最终审计。

检查实存params发现priority ids覆盖为[48,49]，weights却保留旧13项。env_cfg类体检查早于Hydra override，运行时zip静默取前两项[0.02,0.12]，实际优先阶段概率1/7、6/7；retention gate列表也缺48/49。该配置缺陷已证实，但尚不能单独归因全部退化；共享双向head在多个yaw命令上更新也可干扰。retention是晋级门槛，不是梯度回滚，仅补阶段不能保证不遗忘。

env.py新增最终override后的优先阶段/权重等长、唯一、正且有限校验，在super初始化仿真前执行，避免再静默消耗GPU；配套tests/test_rmuc_rehearsal_runtime_config.py及bank gate共20测试通过。后续启动必须显式同时覆盖weights并纳入48/49 retention；旧实验命令原样保存，不篡改历史。tracking_and_rehearsal_audit.json保存同初态比较与配置证据。当前无本任务后台进程，未新开训练；下一步处理局部retention/更新约束再开展有区分力的训练对照，仍保持自主姿态和完整最终任务。

### 205. v243修正rehearsal配置的对照实际启动

从同一v2385384起点而非已退化v242出发，保持v242 seed66/奖励/100更新含10critic/optimizer/head预算/DR0/主±1.2/35%rehearsal/15%连续采样。显式weights=[1,1]配ids=[48,49]，保留旧retention列表并加入48/49；自主姿态、无外环、authority consumer与修正TGS保持。此轮检验正确配置的采样效果，不将retention晋级门槛误当参数回滚，不预先保证遗忘消除。

新增每轮完整rollout的delivered_command_counts，按25D观测重建速度与yaw指令，统计±0.8/±1/±1.2及other，明确它是交付观测中的命令计数，不是内部stage标签；当前DR0，未来噪声会影响近似匹配。22项optimizer/计数/runtime config/bank gate测试通过。worker2972759/trainer2972760实际运行，run2026-09-27_12-07-47_rehearsal_fix_v243，已观察4条预热更新：16384样本完整计数、六命令都有覆盖，旧参数/buffer不变。params读回成对权重、retention覆盖、bank False/outer False/consumer True/TGS True均通过，startup_verified.json保存证据。

artifacts/continuation_v243包含实验契约、运行脚本、自动导出/六点32回合评估流程及最终audit脚本。最终审计待结束后执行，尚未策略行为验收。当前只是critic预热，不把继承head残差非零当作新策略学习证据，不因平台提前结束预算。

v243续查已越过10次critic预热，至少15次总更新，actor_update_l2出现0.0172/0.0134/0.0133，确认策略实际开始更新；进程2972760存活且所有已观察冻结检查通过，live_check.json记录完整rollout累计命令计数。技能状态表同步5384双seed及5484单seed的逐方向跟踪结果，明确+1跨seed不一致、-1/+1.2仅局部通过、±0.8/-1.2仍缺口，未新增稳定掌握认证。历史解析内倾要求不再作为当前执行依据。

v243后续已看到level48/49双向retention样本和yaw误差日志，证明新增阶段参与监测，并非仅改配置文件；仍有跟踪缺口。audit_final.py补入v238/v242只读逐方向对照、评估初态/seed/动力等上下文差异列表与训练累计命令计数；不重写历史passed，不把单seed配对对照当完整因果或鲁棒性认证。脚本编译通过，最终运行仍待本轮结束；live_check.json持续保留已执行的进程/冻结检查。

### 206. v243完整训练与导出结束，固定六点评估已启动

100更新全部完成，所有逐更新旧参数/buffer检查通过；worker只发现authority head允许范围内参数变化。最终run2026-09-27_12-07-47_rehearsal_fix_v243/model_5484.pt，sha256=d310f4dd4d5bf006d7eb0338e01671ee70b9adb4b9f184f6fcef145d1bb3012e，不能与v242同编号checkpoint混淆。完整导出JIT误差0、ONNX最大2.384185791015625e-7，export_parity passed=True。

worker2972759已自动转入evaluating，child2986826实际存活。仍按原定seed67/六点/每点16replicas/至少32完整回合、同consumer/TGS无外环测试，尚未生成最终验收结论。下一步等待final_turns.json后执行audit_final.py，核对采样修正是否改善±0.8且保住±1/+1.2，不仅看总reward或主stage。live_check.json已保存训练结束与评估进程证据，未重启任务。

### 207. v243六点验收结束：三点通过但仍互相干扰；v244第二seed验证启动

v243全部结束，audit_final.py实际执行成功，100更新、旧图/声明预算、导出与六点载荷诊断完整。评估192回合全部生存，+0.8/+1/-1.2通过；-0.8/-1 yaw/stable失败，+1.2 linear/stable失败。按+0.8,-0.8,+1,-1,+1.2,-1.2排序，速度误差0.09326,0.05570,0.04323,0.07803,0.16132,0.09783；yaw误差0.03864,0.20070,0.02083,0.08454,0.01908,0.04196。与v238/v242对照的seed/初态/步数/replicas/所列动力字段完全一致。配置修正使正0.8/正1跟踪改善，但负0.8约多转0.202rad/s、负1多转0.084rad/s，正1.2速度偏低约0.160m/s；不能把全部问题归咎于旧权重截断，也不能再原样共享head续训而忽视命令间干扰。

稳态局部纵向滑移最大0.09032m/s，heading侧速最大0.05497m/s，支撑proxy最小0.13955m；正1.2本次弱轮最小比例0.2257且无接触不足，其余命令有单步8ms低载荷事件。这些是诊断，尚非动态/实机安全认证。累计1638400样本记账完整，六点和other均有明确计数；主任务负1.2采样明显更多，不能从优先weights1:1推断所有指令均匀采样。完整对照及物理极值在final_evidence_audit.json。

先以冻结v243最终5484（sha d310f4dd…）启动v244第二seed75六点各8replicas/1500steps/至少8回合，同consumer/TGS/无外环，覆盖新通过点及剩余缺口，不训练也不自动晋升。worker2990342，脚本与命令在artifacts/continuation_v244。第二seed结果仍待生成；后续应依据跨seed结果保留改善、隔离修复失败命令，并继续动态操作手/最终动力DR任务。

### 208. v244第二seed与局部失败一致；命令局部修正CPU原型通过

v244实际结束，seed75六点各8回合全生存，同一d310f4dd…checkpoint仍仅+0.8/+1/-1.2通过。按+0.8,-0.8,+1,-1,+1.2,-1.2顺序，yaw误差0.04007,0.20133,0.01871,0.08275,0.01930,0.04006；+1.2速度误差0.15833。cross_seed_audit.json核对checkpoint hash并保留两seed逐点结果；192+48回合不是全DR/动态命令或部署认证。当前无后台训练或评估。

新增rmuc_local_authority.py独立CPU原型：准备冻结整个v243图，对-0.8/-1/+1.2各建零初始化局部专家，yaw支持半宽0.15并沿用原0.8–1.2m/s速度支持，使用交付25D观测重建命令。源修正加新修正后仍受原authority总budget限制，不增腿/轮动作总范围；+0.8/+1/-1.2处新gate精确为零，模块返回源修正。4项CPU测试通过零初始化一致、非零专家在保护命令/其他速度无效、总预算有界及中心梯度隔离。

原型尚未接actor恢复/optimizer/export/native；不能声称原生冻结行为或新训练已经完成。下一步按docs/rmuc_local_authority_contract.md接可选模块、只训练新专家、无样本清梯度，再做原生/导出接线验证。现有训练代码和checkpoint未改接该模块，保护点仍需后续native回归，不把CPU性质当技能掌握。

### 209. 命令局部模块恢复/导出/原生对照通过，独立optimizer仍待接线

PositionAuthorityHead新增可选local_correction与实际correction计算接口；installer仅在明确安装或checkpoint包含local centers时恢复，旧checkpoint不自动新增模块。局部centers与half_width作为buffer保存。完整导出action_contract记录局部模块及支持范围，动作consumer/总budget不变。共享authority optimizer拒绝带局部模块的候选，防止误解冻源head；相关11项CPU测试先后通过（含新增拒绝错误optimizer路径）。

v245从v2435484生成zero_local.pt（sha01088f29f94a3cf4a6fbaee6682e8103332d107cf7f89bcddf0049aa5ce0576d），保留源615张量并移除旧optimizer。150点完整actor零输出与严格恢复逐值一致；人工非零候选132个支持域外样本保持源输出，总budget不扩大。零和非零完整JIT/ONNX导出均通过。人工非零checkpoint只作诊断，禁止作为训练初始候选。

worker2994903顺序执行两次原生六点seed75各8回合，均真实结束：零候选初态和所有评估行与v244源策略逐值一致；人工非零候选只改变-0.8/-1/+1.2，保护点+0.8/+1/-1.2的完整评估行（含物理极值）仍逐值一致。native_comparison.json保存全部逐点结果；这是隔离执行证据，不是新增技能达标。当前无后台任务，无新训练。下一步独立optimizer及无样本分支动量隔离、真实合计残差审计、原生少量更新验证，然后再开始失败命令训练。

### 210. 局部专家独立PPO原生接线通过，v247正式100更新启动

trainer新增--train_local_authority，必须同时走authority consumer路径，显式local_only optimizer组local_authority/critic，仅新三个专家可训，源615张量及原head全部冻结；恢复optimizer须组名匹配。audit改为检查head.correction真实合计预算，局部无样本分支清grad防Adam动量漂移，记录local_samples；旧shared模式保持。12相关测试通过，包含先产生动量后切换命令时旧分支仍不动及源图冻结。

v246原生3更新（1critic+2策略）实际结束，模型5487 sha80fa8ba4…，只改变三个局部专家60张量，旧图/buffer不变，optimizer组核对通过。192个保护命令观测完整actor输出与零候选逐值一致，更新后完整导出passed=True。证据artifacts/continuation_v246/{audit,export/export_parity}.json。这3次属于额外接线成本，不计正式100预算，不采用人工非零诊断候选训练。

v247从v245 zero_local重新开始100更新（10critic/90策略），网络源为冻结v243；原动作预算、DR0、主±1.2、优先48/49成对1:1及retention覆盖保持。解析bank关闭，无外环，consumer/TGS开启。worker3001300/trainer3001301实际启动，run2026-09-27_12-49-25_local_turn_v247；已观察5条预热更新，三个局部gate均有样本，冻结检查通过。startup_verified.json保存实际配置与进程。结束自动导出及六点各32回合评估；audit_final.py预备核对仅local专家变化并对照v238/v242/v243，尚未执行。没有新增技能认证，仍需完整行为/动态命令/最终动力DR验收。

v247续查已越过10critic预热进入局部策略学习，源参数/buffer检查通过，三个专家均有实际样本。最终audit增列+0.8/+1/-1.2保护点相对v243完整评估行的逐值一致性及变化字段（含物理极值），并标注上下文是否一致；编译通过，尚待最终评估后执行。live_check.json记录实时进程与更新数，不把保护点CPU性质代替最后行为回归。

### 211. v247训练中发现方向quota历史重算可令局部专家暂时无样本

观察到更新5555/5556附近+1.2交付命令样本为0，主任务8192样本全部落在冻结的-1.2，第三局部专家probe=0；不能沿用每轮“三个专家都有样本”的表述。源图冻结检查正常，无样本专家会清grad，当前100预算照常运行，未按平台停止，也未中途改变实验采样。

只读核查_rebalance_turn_assignments发现target_positive=round((全部历史分配数+当前batch数)*最新focus比例)，然后减去历史正向数。focus变化会重解释过去配额，导致当前连续batch只分配一个方向，即使配置fraction在0.2–0.8。AST执行实际方法CPU复现：历史[800正,200负]、新目标正向20%、新batch128，实际0正128负。sampling_focus_review.json保存源码hash、复现与本次零probe更新；该机制已证明，尚未捕获真实运行的assignment counters，不能断言它是全部性能问题的唯一原因。

下一步保持本轮可比性，结束验收后修quota为逐batch累计期望正向credit，不用最新比例重算历史，并测试大幅focus反转与单环境reset；不能只用配置0.8宣称当前批次有20%反向覆盖。当前源代码未改此分配逻辑，无新任务启动。

### 212. v247完整100更新及导出结束，六点验收实际运行

100更新完整结束（5485..5584），逐更新源参数/buffer冻结检查全通过，worker确认仅local experts允许范围内变化。最终2026-09-27_12-49-25_local_turn_v247/model_5584.pt，sha256=f76fd74dfb82d02c08a6372422f6cc95b87610e0643e0560e2577c4acb9a3b6d。完整图导出passed=True，JIT最大误差0、ONNX最大2.086162567138672e-7。

worker3001300已自动转入evaluating，child3020179实查存活。按原定seed67、六点各至少32完整回合，并记录载荷/侧滑极值；当前最终结果未出。结束后执行audit_final.py，特别核对三保护点相对v243完整行以及三个失败点是否改善。采样quota问题没有在中途修改，待此原定评估结束再修复，避免混淆本轮证据。live_check.json已更新为训练结束/导出通过/评估运行状态。

### 213. v247五点通过且保护行完全不变；结束后修复方向quota并启动v248

v247原定验收实际结束，final_evidence_audit.json已执行通过。192回合全生存，+0.8/+1/-1/+1.2/-1.2通过，只有-0.8 yaw/stable仍失败。按+0.8,-0.8,+1,-1,+1.2,-1.2顺序，速度误差0.093257,0.035263,0.043226,0.055709,0.069283,0.097831；yaw误差0.038640,0.170674,0.020827,0.020793,0.029970,0.041963。三个保护点完整行（含物理极值）相对v243逐值相同，context一致，说明本轮局部隔离保留了这些已验证行为；五点通过仅限本seed/nominal，不是全部任务认证。

验收结束后才修改env方向quota：新增每stage float64舍入credit，每次只增加当前batch*当前focus比例，分配后扣除实际正向数；历史assignment count继续独立统计，不再乘以最新比例。measurement reset同时清credit。CPU执行真实分配/重置方法，修前5失败1通过，修后连同command sampling/bank/runtime config共36测试通过，覆盖80%→20%/20%→80%/80%→50%反转、单env异步reset、stage隔离和状态清零。修改没有混入v247训练或原评估。

v248冻结同一v2475584（sha f76fd74d…）启动第二seed75六点各8回合评估，在修后代码下同时验证原生初始化/reset兼容，不新增训练。worker3034988；命令/脚本/修复证据在artifacts/continuation_v248。结果尚待生成；下一步确认跨seed后，只针对剩余-0.8缺口安排独立训练，保留其他局部专家及整个源图，仍需动态指令/最终动力DR完整验收。

v247稳态物理诊断：局部纵向滑移最大0.10870m/s、heading侧速最大0.05755m/s、支撑proxy最小0.13943m；-0.8双轮>2N不足事件最长16ms，其余最长8ms。这些极值不替代动态/实机安全门槛；源策略参数冻结与局部改善也不等于载荷风险已完整认证。

### 214. v248确认五点跨seed通过；仅-0.8专家独立续训v251已启动

v248实际结束，seed75六点各8回合全生存，仍只有-0.8失败（yaw0.17078），其余5点通过；同一f76fd74d…checkpoint hash核验并记录cross_seed_audit.json。修quota后原生初始化/reset正常，不能把固定grid验收当作动态方向focus训练证明。当前仍有明确剩余缺口，不认证完整技能。

新增--local_authority_train_indices选择局部专家，optimizer保存rmuc_local_indices，恢复时检查相同索引及顺序，不匹配必须reset；省略仍为全部局部专家。测试覆盖所有三个gate有样本但只选0时仅专家0更新、非法索引拒绝。15项optimizer/local/quota测试通过。选择0对应-0.8，其余源图及专家1/2冻结。

v249首次短验证配置rehearsal0.8+continuous0.15，被现有至少10%fixed active约束拒绝，未发生训练更新。Isaac进程exit0但日志实际异常，worker因无run拒绝完成，失败证据保留。修为rehearsal0.7/continuous0.15/fixed0.15，priority48/49 weights9:1、priority_fraction0.9。v250三次原生更新（1critic+2策略）成功，只变专家0的20张量，其他源状态不变，optimizer索引[0]，200个五保护命令观测输出逐值相同，完整导出通过。目标probe样本134/138/137，证据v250/audit.json。

v251从v247正式5584而非短验证模型启动100更新（10critic/90策略），只训局部专家0；动作预算/25D/6D/闭环USD不变、无外环、bank关闭、consumer/TGS开启、DR0。采样配比采用已验证0.7/0.15，重点48但保留其他命令回归。worker3045520/trainer3045521实际运行，run2026-09-27_13-34-12_local_single_v251，启动核查已11更新、源冻结通过，实际配置权重9:1与rehearsal0.7读回匹配。结束自动完整导出/六点各32回合，final audit预备检查仅expert0变化及其余五点完整行对照v247。当前未产生新行为认证，原全任务继续保留。

### 215. v251完整100更新与导出通过，六点验收运行中

5585..5684共100更新完整结束，逐更新冻结检查全通过，worker核对仅专家0允许范围内变化。最终2026-09-27_13-34-12_local_single_v251/model_5684.pt sha256=3333631e9ff6c8e60cd076c49178aa067f82089d6efaf1729ec587f48a5bdb6b。完整图导出passed=True，JIT误差0、ONNX最大2.086162567138672e-7。

worker3045520已转入evaluating，child3057080实查存活；同seed67六点各至少32回合测试正在运行，尚未生成最终结果。下一步执行audit_final.py确认-0.8缺口与其余五点保护行为，仍不能把预算100%或导出通过当技能通过。live_check.json保存最新评估进程证据，未重复启动训练。

### 216. v251六点全部通过且五保护行一致；转入独立验证与更高yaw缺口评估

v251实际验收结束，audit_final.py已执行，100更新只变专家0、完整导出/配置/诊断审计通过。seed67六点各32回合共192全生存，全部现有跟踪checks通过。-0.8速度误差0.011643、yaw误差0.011328，相比源v247的yaw0.170674明显改善；其他五点完整行（含物理极值）与v247逐值相同，context一致。该结论仅覆盖1m/s、±0.8/±1/±1.2、旧轮动力nominal，不是最终全任务或实机安全认证。

不再重复训练这六个点。v252冻结同一5684（sha3333631e…），先seed75重复六点各至少32回合；随后同checkpoint评估1m/s双向1.4/1.6/1.8/2.0/2.5，各8回合初筛，识别原目标范围的真实缺口再安排新技能训练。两步都记录载荷/侧滑与执行器诊断，不放回解析内倾gate。worker3063263已实际启动；脚本/命令/status在artifacts/continuation_v252。首个启动调用因工作目录漏前导斜杠未创建进程，修正后只启动这一worker，无重复评估。

后续仍需连续/动态命令、1.5/2/2.5速度区间、自转、最高联合DR与最终3Nm/径向弹簧/有效damping，以及同一checkpoint部署链验收，不以六点局部nominal达标缩小最终范围。

### 217. v252验收结束：重复seed未增加扰动覆盖，更高yaw有明确缺口

v252全部结束，冻结同一5684 hash3333631e…：seed75六点各32回合均通过，但与seed67相比物理初态/材质/延迟及所有完整评估行逐值一致。DR0下更换seed未产生独立条件，故只能算nominal重复性，不能加总成更强鲁棒性或不同初态证据；不再重复类似无变化seed浪费GPU。后续须用真实初态/物理/命令扰动扩大验证覆盖。

1m/s更高yaw初筛十点各8回合全生存，但全部跟踪未过。按+1.4,-1.4,+1.6,-1.6,+1.8,-1.8,+2,-2,+2.5,-2.5顺序，速度误差约0.2726,0.1120,0.4282,0.1275,0.5426,0.1412,0.5952,0.1548,0.9207,0.1893；yaw误差0.0231,0.1681,0.0649,0.2989,0.0975,0.4342,0.1824,0.5739,0.1774,0.9400。+1.4主要速度不足，-1.4主要转向不足，不能用同一个“不会压弯”标签代替具体误差。

assessment.json保存同checkpoint核验、重复性边界及每点完整物理极值；所有后台任务已结束，无新训练。下一步在保留六固定命令的完整图基础上，为±1.4增加可隔离的局部适应能力，先严格恢复/导出/原生接线再新技能训练，随后推进1m/s至2.5rad/s及原定其他速度范围。现有三个local中心[-0.8,-1,+1.2]不能直接覆盖±1.4，不应解冻已通过专家来硬训新点。最终动力/DR/动态操纵部署范围不变。

### 218. ±1.4局部专家扩展恢复/导出/原生隔离通过，尚未训练

LocalAuthorityCorrection允许显式追加中心并保留原专家对象/中心顺序，checkpoint恢复按保存中心数量构造，不再硬编码三个专家。新增+1.4/-1.4索引3/4，原0/1/2保持。候选源为v2515684，原677张量仅centers buffer追加两值（原前缀不变），其余原参数/buffer不变，half_width及总budget不扩大。生成v253/zero_extended.pt sha01c6cc36d6bf002e1108b45fd8c8d24381eb1e0489bb56721be7cb9c85699747，移除旧optimizer；人工非零扩展只用于诊断，不能作为训练起点。

15项相关CPU测试通过，覆盖追加零输出一致、非零扩展不影响六保护命令、非法追加原子拒绝、动态数量严格恢复。完整actor150点零输出一致，人工非零144个新支持域外观测保持源输出。导出校验新增实际中心及支持边界，最大batch1403；零/非零JIT误差0、ONNX最大2.384185791015625e-7，均passed。

v253顺序原生source/zero/人工nonzero三组，每组八命令各4回合，全部实际结束。source与zero初态及全部完整行逐值相同；人工nonzero只改变±1.4，两方向±0.8/±1/±1.2六保护行仍逐值相同（含物理极值）。native_comparison.json保存证据。这是接线验证，不是±1.4已学会。当前无后台任务，无新训练；下一步从zero_extended仅选择索引3/4做短原生optimizer验证，再按新技能完整预算学习±1.4，保留六点回归及全部最终目标。

### 219. v254短程已核验、v255实际运行与验收收尾补齐

v254三次原生更新完成，真实最终为model_5687（先前临时核查误用了5587文件名，未改变训练）。补存audit.json：仅局部专家3/4的40张量变化，源图/buffer不变，optimizer索引[3,4]正确。v255实际成功run为2026-09-28_12-06-38_higher_yaw_v255，worker3853959/trainer3853960，预算100含10critic，命令stage51=1m/s±1.4，源v253零扩展；中断等待后实查同进程持续运行、冻结审计正常，不重复启动。

前次正式启动经历参数nargs吞入Hydra覆盖项，以及claim/training.log独占创建冲突，均发生在训练更新前；实际成功运行只有上述run。更正前述“两次失败”的简化表述：还包括独占日志检查重试，不能将worker启动次数混入学习轮数。旧失败部分日志已在前次重试中移除，现存worker日志与command.json为可查证据，不声称全部失败日志完整。

检查当前run_train_final.py发现它只完成训练，不自动导出/评估；因此新增finish_evaluation.py，必须status=complete后显式执行。脚本核对100连续更新/仅专家3与4变化，再完整导出并八点±0.8/±1/±1.2/±1.4各至少32回合评估，保留载荷与侧滑极值和逐点checks。编译通过，尚未执行。八点128环境与旧六点96环境布局不同，不据此承诺完整行逐值一致；原六点需行为回归。当前长期目标未完成，未新认证±1.4。

### 220. v255完成100更新及完整导出，八点验收已实际启动

训练5685..5784完整结束，原worker退出，所有逐更新源/旧专家/buffer冻结检查通过。显式启动finish_evaluation.py，其checkpoint审计确认仅专家3/4变化。最终model_5784.pt sha256=5f3bf71699f99221b1e43df9d1b5fd4bd68be7800afd549b0a44d6ed1cc6ee7f；完整导出passed=True，ONNX最大误差2.384185791015625e-7。

验收worker3869764/child3869908实查运行，当前evaluation_status.json=evaluating；seed67八命令各至少32完整回合，包含六原命令及±1.4，尚未出行为结果。status.json=complete仅指训练，必须继续查evaluation_status.json和final_evidence_audit.json，不能按训练完成停止任务或晋升。live_check.json已更新为导出通过/验收运行。未重复启动训练。

### 221. v255八点验收结束，原六点通过，±1.4尚有跟踪缺口

最终5784 sha5f3bf716…核验一致，八点各32回合共256全部生存，原六点仍通过。+1.4速度/yaw误差0.19808/0.05889（signed速度-0.20537，yaw-0.05255），-1.4为0.09795/0.14673（signed yaw+0.14646，转向幅度不足）。没有新增±1.4认证，也不因全生存认定稳定跟踪已完成。

两新增点稳态局部纵向滑移最大约0.10037/0.09561m/s，侧速最大0.02468/0.02283m/s，support proxy最小0.13528/0.13321m；双轮>2N不足最长8/16ms，力矩饱和统计0。仅为当前nominal控制步诊断，不是实机动态裕量认证。完整数据见final_turns.json与final_evidence_audit.json。

此前源策略v252高yaw初筛与本次环境数/布局不同，不能直接据误差变化断定训练收益。v256冻结源v2515684，使用v255同八点、16replicas、seed67、至少32回合进行匹配对照；worker3873139已启动，artifacts/continuation_v256保存命令与状态。当前无新训练，待源对照核实初态和完整保护行、量化±1.4改善后再决定训练配置。全部最终目标不变。

### 222. 同场景确认六保护行完全不变，v257仅±1.4专家继续学习

v256已结束，八点初态/材质/seed/步数/replicas/执行契约与v255一致，六个保护命令完整行逐值相同。+1.4速度误差0.27024→0.19808，yaw0.02413→0.05889；-1.4速度0.11253→0.09795，yaw0.16867→0.14673。确认有局部改善且正向存在速度/yaw权衡，不称已达标。comparison.json保留配对证据。无须重新训练六个已通过固定命令，鲁棒性/动态场景仍未认证。

v257从v255最终5784继续额外100更新，仅indices3/4可训，保留optimizer/critic状态并核验来源，critic warmup0；奖励、门槛、预算、采样及physics不改。新技能允许完整预算，不因短期平台停止。worker3877041/trainer3877042实际存活，已观察6更新（5785起），双新专家probe109/147，冻结检查通过。run及实际进程证据见continuation_v257/live_check.json。

run_continuation.py在训练后检查100更新及冻结记录，自动调用finish_evaluation.py做完整导出、八点各32回合验收；训练status complete不等于最终evaluation_status complete。当前没有新的行为结论，下一步继续核查正向速度/yaw权衡与负向yaw改善，完整最终目标保持。

### 223. v257续训验收：原六点保持，±1.4仍未达标且出现权衡

v257已完成100额外更新并自动导出/八点验收；final model5784? 实际checkpoint与审计文件为当前run，完整导出通过、仅专家3/4变化，八点各32回合全生存。原六点±0.8/±1/±1.2全部逐项通过并保持v255行；+1.4速度误差由0.19808降至0.09981，但yaw误差升至0.33609并失败；-1.4速度0.09460、yaw0.09517，较v255 yaw0.14673改善但仍失败。两专家共享同一PPO目标出现速度/yaw权衡，不能自动继续解冻或认定新技能完成。

证据artifacts/continuation_v257/{status,final_evidence_audit,final_turns,live_check}.json；训练100和导出一致性审计通过，行为all_behavior_rows_passed=False。下一步应拆分+1.4与-1.4的局部专家训练目标或采用命令条件的独立损失，先保护六通过点；继续前须设计可验证的速度/yaw权衡约束，不重复原共享更新。全DR、动态指令、1.5/2/2.5速度、自转、3Nm/弹簧/damping和部署仍未交付。

### 224. v258正向专家独立诊断启动

更正第223条的因果表述：专家3/4参数与支持域本来独立，v257结果只显示正向+1.4自身存在速度/yaw权衡，不能证明两个方向互相干扰。v258冻结v257 model_5884，仅训练专家3、其余专家/源图不变，用于判断正向权衡是否能单独改善；不是直接认定共享损失为根因。

v258一次参数解析失败发生在训练前，nargs索引吞入Hydra覆盖项；已保留artifacts/continuation_v258/failed_parse_attempt并加preflight解析校验。修正后真实run2026-09-28_13-12-12_positive_yaw_v258，worker3898086/trainer3898087，已观察8次更新，专家3 probe105、专家4仍诊断采样151，冻结检查通过。结束后需单独评估+1.4、保护六点及负1.4，不能自动晋升；全任务目标保持。

### 225. 补齐v258验收记录，修正v259评估入口并真实启动

v258已完成100更新与八点各32回合验收，model5984 sha b9888826…仅专家3的20张量变化、导出通过。+1.4速度/yaw误差0.03813/0.13437，仍yaw/stable失败；其余七点完整行相对v257不变，其中-1.4仍yaw0.09517失败，原六点通过。final_evidence_audit.json已按原结果补存；先前“负向+1.4”是笔误，应为-1.4。仍无新增技能认证。

v259从上述5984继续100更新，实际5985..6084，仅专家3的20张量变化，最终6084 sha ed1769fab2e4a26957e019a7ed3fd6fa420fb72ad224792db5e49b07643d200e，完整导出通过。它没有新加yaw专项奖励，属于原奖励下的单专家续训，不能因实验名turn推断目标函数变化。

先前验收误取训练command.json，实际调用train.py并被Hydra拒绝评估参数，未产生行为结果。当前确认该进程已退出，失败命令/日志/PID移入failed_evaluation_entrypoint保留；从v258/evaluation_command.json构造正确evaluate_rmuc_grid.py入口，核对hash、八点、16replicas及32回合后启动run_evaluation.py。worker158750；最新child见evaluation_status.json。脚本输出最终八点评估及与v258同初态七保护点逐值比较，当前仍在评估，不能沿用训练status complete称任务通过。

### 226. v259完整验收后继续正向小缺口，v260保持optimizer状态

v259正确评估已结束，八点各32回合共256全生存，六旧点通过。+1.4速度误差0.020596、yaw误差0.064681（v258为0.038128/0.134370），仍yaw/stable失败；-1.4保持0.095168 yaw未过。七保护点完整行与v258逐值一致，context_mismatches为空，final_evidence_audit.json保存配对结果。未降低0.05 yaw门槛，不把接近达标算通过。

v260从v2596084继续额外100更新，仅索引3可训，保留optimizer/critic、warmup0；奖励/physics/门槛不改。启动前用实际train.py argparse定义校验命令，核验checkpoint hash及optimizer索引，避免重复nargs错误。worker164930/trainer164931实际运行，6085起已观察11更新，冻结检查通过，live_check.json保存实际run/进程/采样。run_continuation.py结束后自动调用finish_evaluation.py导出并八点各32回合验收。此轮只追踪尚未达标的新技能，不能扩展为全DR/动态/部署认证；负1.4与更高目标仍待完成。

### 227. v260正1.4通过，v261只训练负1.4专家

恢复后实查v260训练与自动验收均已结束，无原进程。最终model6184 sha02aae0ae8594b33776e8a33731f05393f07e4187a5f6cc6b41a58e6c8265fef3；八点各32回合全部生存，7点通过。+1.4速度/yaw误差0.015351/0.042934通过现有门槛，-1.4保持yaw0.095168失败。与v259相同初态，其他七行逐值相同，protected_comparison.json保存核验。正向仅nominal固定命令通过，不是独立扰动/动态或最终动力认证。

v261从同一6184仅选择专家4（-1.4）；冻结专家3及其余旧图。由于optimizer参数集合变更，reset optimizer，保留critic、warmup0；奖励、门槛、100预算、采样和physics保持。CPU在真实源actor上进行临时更新预检：只改专家4，七保护命令输出逐值不变，未保存临时权重；实际argparse也已校验。结束自动完整导出/八点32回合并保存相对v260七保护行比较。

worker710166/trainer710167实查运行，run2026-09-29_13-03-23_negative_yaw_v261，6185起已9更新，负专家probe148，所有已观察冻结检查通过；params读回无外环/bank关闭/consumer开启。证据continuation_v261/{optimizer_preflight,experiment_contract,live_check}.json。继续此实验不重复训练新通过的正向专家，最终更高yaw/其他速度/动态命令/最高DR/3Nm等全范围仍待完成。

### 228. v261八点通过；恢复goal并开始25D部署接口适配

v261训练100更新及最终验收已完成，model6284，完整导出通过。八个1m/s命令±0.8/±1/±1.2/±1.4各32回合全部生存、通过；-1.4速度误差0.081163438、yaw误差0.008046533，七保护行完整结果与v260相同。完整证据continuation_v261/final_evidence_audit.json；不外推全DR/动态命令/最终硬件。

用户已恢复goal并确认近期行走＋独立复位双ONNX、后续越障第三ONNX方案；保留全部原长期能力范围，近期优先十月八日前MuJoCo与复位交付。sim2real_RL/sim2sim为部署入口，上游部署仓库固定dd367bf78c7e393d811595edeb4affe253156c95；normal-only35D和轮速度主线不能直接替换RMUC25D/轮力矩策略。此轮未重训已通过nominal点。

新增rmuc_policy_interface.py（右先六关节，tracking_error_with_target，2.1/0.5/1命令缩放，8ms策略周期，有界authority consumer）及verify_interface.py。验证直接执行训练env.py AST的nominal观测构造前缀，对1024随机状态及命令门控边界做对照；观测与consumer误差0，ONNX Runtime/完整TorchScript最大误差7.1526e-7。报告sim2real_RL/sim2sim/interface_verification.json含训练源码和ONNX hash。训练issaclab环境无onnxruntime，使用已有robot环境运行通过（Torch2.6.0/ORT1.18.1/MuJoCo3.11.0），未修改依赖。

下一步审计既有/home/kellen/wheel_leg_sim2sim/assets/wheeled_infantry.xml与当前USD的闭链、质量惯量、姿态和执行器；发现其旧policy接口虽同25D却是左先、5帧历史和100Hz，不能照搬。新接口输入仍需部署状态估计和实际坐标映射，物理关节限幅/动作延迟/PD与力矩映射尚未接入，不宣称已完成sim2sim。训练未启动，goal继续。

### 229. MuJoCo候选模型初审，查明轮力矩gear换算

新增sim2real_RL/sim2sim/audit_model.py及model_audit.json。robot环境实际编译/home/kellen/wheel_leg_sim2sim/assets/wheeled_infantry.xml，issaclab子进程读取当前训练USD；39刚体名称全部一致，总质量分别19.2211399797/19.2211393714kg。每刚体质量最大误差5.67e-7kg，主惯量特征值最大误差2.91e-7kg m²；四闭链初始site连接误差<1.2e-16m。旧文档18.970kg不代表当前文件。尚未证明COM/惯量方向、关节轴、碰撞和运行时执行器匹配，不把静态编译成功当物理闭环通过。

候选轮执行器gear15.7647058824；直接ctrl=3会形成约47.294Nm。六执行器分别注入经过gear换算的控制，mj_forward后qfrc_actuator与要求的关节输出力矩一致（腿1Nm/轮3Nm，误差<=1e-8）；轮3Nm的ctrl约0.1902985。未积分、未改动原MJCF或USD，后续适配必须处理输出轴单位，旧5.5Nm和最终3Nm配置分别验收。

发现上一轮consumer校验参考为平滑门控helper，而实际env._compute_policy_action使用直接支持域条件，极邻近边界可能有float舍入差。部署consumer改成实际环境条件，verify_interface.py直接提取真实执行分支AST并覆盖相邻float32边界。1024状态观测及consumer误差0，完整ONNX/JIT误差7.15e-7，更新interface_verification.json。训练图和权重未改；接续坐标/执行器映射及MuJoCo基础闭环，无新训练。

### 230. 同一v261 ONNX首次MuJoCo十秒闭环，存活但跟踪未达标

新增sim2real_RL/sim2sim/run_mujoco_probe.py。实际加载完整v261 ONNX、候选闭链MJCF，右先关节映射、canonical=(-source_z,source_x,source_y)、水平heading速度、8ms策略周期。腿位置scale0.22+物理限位margin0.02、PD60/2与40Nm裁剪；轮-action*5.5减0.1*qdot再裁剪5.5Nm，按gear换算ctrl。六主动DOF阻尼在模型内存清零避免重复PD，腿armature0.01，原XML/USD不写入。维持候选XML1ms物理步，不能当PhysX2ms完全等价。

初次2秒诊断无倾角/高度停止，末速度0.413/yaw-1.384，随后10秒检验持续性；修正mj_step后派生位姿滞后一子步的问题，每策略步先mj_forward再构造观测，并重跑10秒。最终probes/v261_turn_10s.json记录真实轨迹和profile/hash：运行10秒，最大倾角5.04781°、腿速度2.55709rad/s，后5秒625样本平均速度0.842378、yaw-1.502968，速度/yaw平均绝对误差0.157622/0.102968。未通过原yaw门槛0.05，不因存活即晋升；certified=false。

此为实际策略物理闭环进展，尚非ROS控制器接通/最高DR/最终3Nm部署认证。被动阻尼、接触材质、惯量与坐标、17rad/s PhysX速度限制和初态未完全对齐；下一步用成对诊断分离建模差异，暂不依据一次跨引擎偏差重训或加任务反馈外环。长期全任务保持。

### 231. 步长与被动阻尼对照，排除当前主要跟踪偏差的两个候选来源

run_mujoco_probe.py新增显式--physics-dt/--passive-damping及运行源码hash和被动阻尼原值记录；资产不改。按同v261 ONNX、同模型初态、1m/s/-1.4、10秒各运行基线、dt0.002、被动八关节阻尼0.01、组合四组，全部达到时长。后半程速度/yaw平均绝对误差分别0.157622/0.102968、0.156987/0.103340、0.157502/0.102955、0.156872/0.103327。不能把微小变化解释为主要原因，也未达到yaw0.05。证据sim2real_RL/sim2sim/probes/dynamics_ablation.json和四个独立完整trace；run_dynamics_ablation.py可复现。

进一步读取本机IsaacLab base_articulation_data.py：root_lin_vel_w是root_com_lin_vel_w别名。给候选MuJoCo模型根自由度非零平移/角速度，比较mj_objectVelocity(BODY)与mj_jacBodyCom*qvel及mj_jacBody*qvel：前者误差<6e-17，原点速度有约0.093差别。因此当前probe读取的是与训练相同类别的COM速度，未发现这一项接口错误；没有据此宣称整个坐标/状态估计对齐。后续继续接触、关节坐标/惯量方向与匹配初态诊断，不调整奖励或复训。

### 232. 根坐标系质量几何与惯量对照；训练摩擦0.5迁入后仍失败

扩展audit_model.py，USD XformCache与MuJoCo初始世界变换都转到root frame，比较39刚体位置、COM以及主轴旋转后的完整惯量张量，而非只比特征值。最大分量误差分别5.8688e-7m、5.8688e-7m、3.6855e-7kg m²；从USD joint localRot0/body0推导转轴，与MuJoCo xaxis转root比较最大2.22e-16。锚点逐关节误差也写入model_audit.json。没有发现这组初态资产中的明显质量/姿态/轴符号错配；不等于PhysX运行时惯量、接触或完整闭链动力已认证。

查v261/final_turns.json initial/final_material_readback，机器人40形状静/动摩擦均0.5；训练params中地面静/动摩擦亦0.5。候选XML轮与地面1.1。probe新增--sliding-friction（geom与显式pair滑动分量）及每控制步contact_sliding_friction读回。仅将摩擦改0.5，其他条件同10秒基线，真实运行contact friction全为0.5；后5秒平均速度0.787736、yaw-1.552709，绝对误差0.212264/0.152709，跑满但比1.1更差。证据probes/v261_friction05.json。该结果证明应区分物理对齐与调参过单点，不能因为1.1成绩好就宣称它代表训练值。继续初态/接触与运行时驱动对照，不改奖励或策略。

### 233. 实际reset obs对齐证据与3Nm执行口径澄清

用户再次强调obs与策略对齐，确认两台RM3508、四台DM8009；此前完整腿型号DM-J8009P-2EC、减速比9:1保留，未据简称擅改型号参数。实读USD轮关节drive maxForce3Nm、四腿40Nm；Isaac v261 actual_wheel_actuator_effort_limit_range实际5.5Nm，当前MuJoCo probe为旧训练契约5.5Nm诊断，不代表最终每轮持续3Nm。候选MJCF ctrlrange±3还乘gear15.7647，必须在关节输出侧执行3Nm限制并换算ctrl；四腿40Nm仅当前模型限幅，不能冒称用户确认持续扭矩。

新增compare_initial_observation.py，读取v261/final_turns.json初始记录并选择同1m/s/-1.4命令的16环境112–127，与probes/v261_friction05.json首帧逐槽比较。25D最大差1.1920929e-7，唯一非零槽为投影重力；同完整ONNX输出最大差3.5762787e-7，比较通过。报告probes/initial_observation_comparison.json。比人工随机输入测试增加真实两引擎初态证据，但不宣称所有隐含物理状态/完整轨迹/传感器估计器一致。未改变策略或训练参数。

### 234. 轮端3Nm限幅不改变该点轨迹，直接缩动作尺度会失败

run_mujoco_probe.py新增--wheel-action-scale/--wheel-effort-limit，按训练顺序先裁剪feedforward target，再减显式0.1阻尼，再裁剪输出；逐子步读qfrc_actuator核验实际轮端力矩不超过声明上限，记录实际峰值与饱和步数。保留gear换算，不改MJCF源。

同v261/1m/s/-1.4/摩擦0.5，三组实际运行：旧scale5.5/limit5.5与scale5.5/limit3完整10秒trace逐值一致，实际两轮峰值1.342332/1.198442Nm，饱和0；scale3/limit3在1.984秒倾角45.8503°停止，尚未达到后半程统计窗口。证据sim2real_RL/sim2sim/probes/torque_migration_comparison.json含完整命令与三份trace；原yaw偏差0.152709仍未通过。

此证据只证明该点在当前MuJoCo诊断配置中未依赖>3Nm以及直接更改动作含义破坏控制，不能外推全技能有足够力矩裕量。最终要求是输出轴持续上限3Nm，并不数学上强制旧归一化action乘数也必须3；优先验证保留旧尺度加3Nm硬限幅，避免无必要地改变已学映射。后续仍需Isaac对应验证、最终damping与径向气弹簧和最高DR，现未修改正式训练配置、权重或认证状态。

### 235. 按用户要求将MJCF与依赖网格放入sim2sim目录

用户强调MJCF必须与USD对齐，并要求本地可见文件。将此前实际使用的wheeled_infantry.xml及其21个引用网格复制至sim2real_RL/sim2sim/wheeled_infantry.xml和meshes_mj/，共22文件50,956,968字节；不使用符号链接，逐文件SHA-256校验与原来源一致，model_source_manifest.json记录来源。未覆盖原工程或改资产物理参数。

audit_model.py、run_mujoco_probe.py、run_dynamics_ablation.py默认使用新本地路径；实跑本地模型完整质量/坐标/闭链审计以及0.08秒ONNX闭环smoke。对齐状态继续区分authored USD与训练runtime覆盖：已核查质量/COM/惯量/关节轴，但接触和最终执行器仍未完成。复制成功不能代替物理对齐，也未把XML旧ctrlrange±3误当轮端3Nm。README新增模型入口与用户对齐约束。

### 236. 显式轮端Nm配置生成，腿轮拓扑保持

新增build_mjcf_profiles.py，从当前USD drive maxForce与v261 params生成rmuc_training_v261.xml（轮5.5Nm）及rmuc_usd_limits.xml（轮3Nm，腿40Nm），gear统一1，ctrl直接输出轴Nm；步长2ms、滑动摩擦0.5、被动阻尼0.01、主动显式PD避免双重阻尼。原wheeled_infantry.xml不改。编译后质量、惯量、body/joint/geom几何数组与源逐值相同，六执行器正负100输入饱和探针实际qfrc_actuator符合各自限制，mjcf_profiles.json保存证据。

USD限制版以旧5.5 action尺度加3Nm输出上限实际运行10秒，最大轮端力矩1.347584/1.198885Nm，后半程速度/yaw误差0.211308/0.151626仍未通过；probes/v261_usd_limits_local.json。只完成执行器单位/限制和部分nominal参数对齐，不代表径向气弹簧/实测damping/最高DR已完成。

用户进一步要求不改腿轮拓扑。实时比较本地原模型与来源逐字节一致；原副本及两生成配置的body父子层级与位姿、joint连接/轴/范围、全部equality闭链约束逐项相同，topology_verification.json记录。生成器加入相同结构不变断言，后续不为调通策略改变拓扑。

### 237. MuJoCo径向气弹簧接入与力学验证

新增sim2real_RL/sim2sim/radial_spring.py和verify_radial_spring.py；runner --radial-force表示每腿伸长力N，默认0保留旧nominal诊断。沿hip关节锚点到wheel关节轴心施加等大反向力，分别作用在robot与wheel真实位置点，通过mj_applyFT累计qfrc_applied，每子步更新；不是10.775Nm恒关节力矩，不修改拓扑或资产。

512随机几何对真实训练gas_spring.radial_spring_force_pair最大误差3.55e-15；模型内每腿实际10.775N，根自由度净广义力<4.45e-16；施力功率-0.06923349555与腿长有限差分-0.06923349468一致，radial_spring_verification.json记录。此为恒力虚拟径向作用模型，不冒充实测力—行程/摩擦曲线。

同v261/1m/s/-1.4、rmuc_usd_limits.xml、3Nm限幅/旧5.5尺度，开启每腿10.775N实际跑满10秒，后半程速度/yaw误差0.258780/0.111332，轮力矩峰值1.357524/1.198636Nm、最大腿速度2.3591rad/s。无弹簧对照0.211308/0.151626，速度变差yaw改善，仍未通过；证据probes/v261_radial10775.json。不据此改奖励或重训已有nominal；下一步把明确的最终动力条件与跨引擎缺口纳入适应验证，完整任务及独立复位目标仍未完成。

### 238. 用户确认按USD执行器口径处理gear，默认入口切3Nm配置

用户对gear减速比提出疑问后，已实测源gear15.7647通过ctrl=输出Nm/gear，与新gear1/ctrl=输出Nm在[-3,3]十三采样及两轮上qfrc_actuator完全一致，gear_unit_verification.json记录。USD+IdealPD直接输出关节力矩，不等于实物没有减速器；未把源15.7647或用户约19:1当已辨识硬件比值。

用户随后明确无需再解释数字，只要与USD执行器对齐。默认probe/audit模型切到本地rmuc_usd_limits.xml，轮端上限3Nm、腿40Nm、gear1，旧ONNX动作尺度5.5保留，输出仍硬限3Nm。增加声明限幅不得超过加载MJCF能力的检查；默认模型请求5.5Nm实跑确认被拒绝，默认0.08秒闭环smoke和完整模型静态审计通过。历史ablation显式选原模型和5.5以维持旧证据，不混入默认路径。拓扑未改，实物减速箱等效惯量/摩擦仍是需要正确建模的部分，不能把gear1当辨识完成。接触、17rad/s约束、最终有效damping和完整DR仍待完成。

### 239. 有效damping敏感性实测；解除Isaac迁移动作尺度与力矩上限耦合

MuJoCo runner新增--wheel-damping，保持3Nm/旧scale5.5/每腿10.775N/默认USD限制配置，原用户宽估计0.005–0.08的低端、中点、高端实跑：0.005在4.344秒触发停止，0.0425在6秒停止，0.08跑满10秒但后半程平均前进速度0.020614、速度误差0.979386、yaw误差0.063378。实际力矩均未超3Nm，probes/damping_sensitivity.json附命令/统计与三trace。证明旧策略对这类有效损耗变化不鲁棒，不把某端存活当技能通过。原正式候选profile已有基于4m/s的条件先验0.005–0.04，本次较宽敏感性扫描不是擅改最终DR范围或实测辨识。

发现scripts/rmuc_hardware_profile.py此前强制target_action_torque_scale_nm==3，与已确认保留旧动作单位加3Nm限制的路线冲突。现接受正有限动作尺度独立插值，物理effort_limit目标仍3Nm，torque_reference与禁止重复gear校验保持。旧scale3配置完全兼容，默认configs/rmuc_hardware_target.yaml不变；未来试验需显式选择5.5尺度。tests/test_rmuc_hardware_profile.py新增半程/全程尺度保留不放宽实际限幅和非法尺度无部分写入等测试，总18项通过。未启动训练或改变策略/拓扑。

### 240. 用户要求腿PD律实机一致；共享参考与真实源码对照完成

用户强调RMUC腿部PD在训练与实机部署完全一致。实读Isaac IdealPD.compute：tau=Kp*(q_des-q)+Kd*(qd_des-qd)+ff后按effort_limit裁剪，当前实际配置Kp60/Kd2、40Nm；v261 leg_actuator_audit记录目标速度与前馈均0。新增leg_pd.py作为关节输出侧共享参考，MuJoCo runner改为调用它；不改现有律、增益或拓扑。

verify_leg_pd.py直接提取本机Isaac IdealPD.compute和ActuatorBase._clip_effort源码AST，1024组四关节状态，含1349正饱和、1345负饱和及非零目标速度/前馈对照，最大误差0，报告leg_pd_verification.json。默认模型0.08秒共享PD smoke通过。LEG_PD_CONTRACT.md明确目标限位顺序、右先映射、输出侧单位、不重复乘9:1、默认2ms PD/8ms策略、禁止叠加隐藏环、气弹簧独立于电机PD；目前未连接固件，hardware_controller_verified=false，实机周期/滤波/协议及持续扭矩仍待确认。

本轮原计划进行目标动力Isaac八点评估，但尚未启动；运行检查发现GPU有用户另一个fudan训练，未终止或改动该任务。继续按真实进程与资源情况安排评估，不能把尚未执行计划当结果。

### 241. 目标动力Isaac八点诊断启动，已出现真实物理适应缺口

资源实查4080余量约27GB、另有fudan训练832110；不终止用户进程，在CUDA_VISIBLE_DEVICES=0启动小规模128env评估。artifacts/continuation_v262保存hardware_profile.yaml、command.json、experiment_contract.json及worker.py；worker852280/eval852281真实存活，初始化与首动作成功。来源固定v261/model6284 sha240ddf1aa366d446625c361d3313cafb3a4523a5fccd4900a7890bf41b2437fb，仅八点评估、16replicas/点、每点至少32回合/3500控制步上限，不训练、不自动晋升。

profile显式采用5.5动作尺度/3Nm执行器上限、每腿径向10.775N、damping nominal0.0225（现有候选范围0.005–0.04）、既有独立电机响应先验0.002–0.030s；DR0，无任务外环、authority consumer及TGS修复保持。profile来源和代码hash记录；范围未实测，不当作最高DR或最终硬件认证。此次同时启用多项目标条件，不能用单点失败归因于某一个参数。

末次核查实际eval852281运行约1分54秒，日志steps1000/min_completed0；-1.4在898episode步触发tilt85.08°，全部速度/pitch/yaw外环反馈0。这是在Isaac内也出现的目标动力适应缺口，不只MuJoCo问题；完整八点最终结果仍待生成。继续查status/真实PID、完成后读target_turns.json与summary.json，勿重复启动。

### 242. v262最终八点0通过，确定目标动力适应为下一阶段

v262 worker852280/eval852281已结束且PID不存在，status complete/exit0/passed_rows0。七个yaw点±0.8/±1/±1.2/+1.4各32回合生存1，但速度误差0.659373–0.886871、yaw0.178441–0.381407，均linear/yaw/stable失败；-1.4共44回合全部tilt，coverage不足、跟踪误差null，不作零误差解读。summary.json保存简表，完整target_turns.json保留所有指标。

source_target_comparison.json证明与v2618/8 nominal通过使用同checkpoint和完全相同记录初态，实际读回effort_limit3Nm、damping0.0225000009、径向每腿10.7749996N。电机响应模块虽启用，DR0 sample使腿轮time_constant实际全0；此前“启用响应先验”只描述安装，不意味着本次有真实滞后，不能算响应/最高DR验证。同时修改多项物理，不把全部差异因果归咎一个参数。

补做MuJoCo同damping中心0.0225、每腿10.775N与3Nm名义条件，probes/v261_target_nominal0225.json约5秒触发诊断停止（两引擎停止门槛不同，不比较失败时间等价）。目标动力不足已在Isaac内证实，后续应渐进适应而不是继续无限排查obs或重复旧nominal训练。旧技能证据限定原动力仍有效；下一步冻结源图、保持动作单位和有界适应，配合保留与目标验收，再逐步联合DR。没有新训练/晋升。

### 243. v263迁移25%评估启动；单一有界物理分支原型通过CPU预检

为定位可学习的迁移起点，v263沿v262目标profile取fraction0.25，同v261完整checkpoint、同八点128env与32回合门槛，DR0。名义轮effort4.875Nm、damping0.080625、每腿径向2.69375N，动作尺度保持5.5。这是允许的渐进中间动力，不是宣称最终3Nm。worker857699/eval857700实际存活，首动作与原基线相同，末次日志steps1000；最终结果待查continuation_v263/status/target_turns/summary。无新训练。

检查现有local专家只支持部分命令且总轮修正预算0.06，不能直接当覆盖全低速物理适应的结构。新增rmuc_physics_adaptation.py为一个共享25→64→64→6零输出MLP原型，abs(vx)<=1.5及abs(yaw)<=4范围内生效，1.5–1.7/4–4.2平滑退出；动作单位修正预算[.15,0,.35,.15,0,.35]，不扩大既有consumer/最终关节目标或3Nm执行限制。轮预算依据damping变化量0.0775、20rad/s示例所需action约0.282设置0.35，是设计预算，不是新的硬件参数。旧图全冻结，目标是只增加一个物理适应模块而非逐命令不断扩展。

tests/test_rmuc_physics_adaptation.py三项通过：零初始化原始动作逐值相同、非零修正有界且域外精确不变、梯度只更新新模块。进一步restore真实v261完整actor做256状态CPU合成一步优化，旧actor所有state逐值保持、零组合一致，新模块可更新且幅度有界；physics_head_cpu_preflight.json记录。不保存合成训练权重，不宣称PPO/完整导出/native已完成；后续需安装/恢复/导出接线、独立优化审计与真实训练smoke。原策略与拓扑未改。

### 244. 25%迁移存在七点缺口；物理分支完整图恢复/导出接通

v263已结束：八点各32回合全生存，仅-1.2通过（速度/yaw0.132439/0.019300）；-1.4为0.151785/0.052593仍未过，其余点也有跟踪不足。readback_summary.json实际轮effort4.875Nm、damping0.080625、每腿弹簧2.69375N，不能当最终3Nm。此中间物理已需要适应训练，原旧nominal无需重训。

rmuc_command_adapter.py新增显式install_rmuc_physics_adaptation，完整forward在position_authority之后、Gaussian之前调用新head；按checkpoint键恢复，不给旧策略隐式加模块。rmuc_policy_export.py新增分支存在标记、预算与域说明；verify_rmuc_export.py扩充±1.5/±1.7及±4/±4.2边界探针。没有改训练PD、consumer物理限制或资产拓扑。

v264零分支checkpoint zero_physics.pt sha43287819149bef7d90d677bb316b47f03a61f5c611c0fbf8ac7b831fd0f9a09e，源v261旧state逐值保持，512CPU输入完整输出零组合一致，严格restore一致。人工bias0.2 nonzero_diagnostic.pt只验证非零导出，不作为候选训练起点。两者完整JIT/ONNX在2409命令/边界探针通过，最大ONNX误差2.384e-7；证据initialization.json及zero_physics_export/nonzero_diagnostic_export。相关7项pytest通过。当前尚未接入独立PPO优化器与native训练smoke；旧checkpoint带local optimizer不能直接续用，新参数集合需要reset及全旧图冻结审计。下一步完成训练接线后从零head做25%动力适应。

### 245. 物理分支独立PPO与冻结审计接通，正式25%动力适应启动

train.py新增--train_physics_adaptation并要求authority consumer、排除local专家训练，optimizer来源组为physics_adaptation，恢复参数集合变化需reset。prepare_authority_optimizer只解冻新物理MLP及critic；configure_authority_audit继续核查所有旧参数/buffer，并记录物理分支预算/采样数、无域内样本清grad避免Adam动量漂移。新增真实optimizer/audit测试，27项相关测试通过。

v265从v264零分支执行3次原生更新（1critic预热），run2026-09-29_16-45-28_physics_smoke_v265/model6287；仅rmuc_physics_adaptation.network的6张量改变，旧actor全部state不变，预算审计通过，训练后完整导出通过。artifact freeze_audit/status/export保存，不把短smoke当行为能力。

v266从同一零分支43287819149bef7d90d677bb316b47f03a61f5c611c0fbf8ac7b831fd0f9a09e启动正式100更新/10critic预热（不从合成非零或smoke权重起步），128env，25%动力effort4.875Nm/damping0.080625/径向每腿2.69375N，动作尺度5.5、DR0。实际smoke env参数已读回这些值后复用；PD60/2、源图冻结与rehearsal保持。worker867784/trainer867839已存活并打印首更新14.19秒，完整预算100不因短期平台提前停。结束自动冻结审计、完整导出、按v263目标profile八点至少32回合；无自动晋升。motor response训练关闭、评估模块启用但DR0旁路，后续非零DR需明确统一，当前不算响应鲁棒性认证。

### 246. v266持续训练且冻结正常；并行完成复位初态CPU物理检查

v266 trainer867839实际存活，position_authority_training.jsonl末次28/100更新（含10critic预热），所有旧参数/buffer审计均无变化，新物理分支已非零学习；继续原预算与自动八点验收，不提前宣称完成。

训练期间不追加GPU，新增sim2sim/audit_recovery_starts.py：保持原腿轮拓扑、零关节闭链姿态，将root旋转为前倒/后倒/左轮组件侧倒/右轮组件侧倒，依据实际可接触ground的geom最小距离抬高到3mm，四闭链初始误差接近1e-16，qpos/qvel与最近地面geom保存recovery_start_audit.json。并核查base_collision确实接受ground，不因visual mesh无碰撞就误判机身穿地。

settle_recovery_starts.py每个姿态以电机零力矩，在0与每腿10.775N径向力下各运行2秒，共8例。全部状态finite，峰值闭链误差<=0.001502m，主动四腿峰值速度<=12.747rad/s（单独统计，不能把全部关节包含轮的26.6rad/s误当腿超速）。有弹簧左右侧案例达到本次根速度+全关节速度静止诊断阈值，其他案例部分根已静止但轮/导向轮仍转动，报告保留真实末态速度，不清零伪造静止。

这些只为恢复课程构造物理初态，不证明起身可达，更不是已有recovery策略。原行走大倾角终止仍需在独立复位任务内调整，不能解除其正常保护；下一步分别建立静止倒地与动态跌倒分布及独立复位动作/交还契约。用户已再次确认后续包括平地倒地自主复位，按双ONNX计划保持。

### 247. 独立平地复位环境建立并通过原生128步检查

新增rmuc_wheel_leg/recovery_env.py与Gym任务Robotics-RMUC-WheelLeg-Recovery-v0。复用原USD及25D/6D观测/执行器链，独立reset为四种倒地姿态；不使用行走tilt终止，保留非有限/关节限位/离地过高/越界/过大速度等保护；奖励为起直、高度、稳定保持、动作变化与腿力矩代价，尚属原型非调优完成。起稳判据要求12度内、height>.28、平移<.15、角速<.3、双轮>2N连续0.5秒，之后仍保持不立即结束，后续须加策略交还验收。

腿部IdealPD仍60/2与40Nm，但复位action单独映射物理关节范围内（margin0.02），不是行走的小幅0.22目标；必须分别导出动作契约。轮端3Nm/暂定damping0.0225/径向每腿10.775N、DR0，不改结构或行走配置。复位动作可达性与范围风险尚需训练检查，不宣称实机可直接使用。

scripts/probe_rmuc_recovery.py四env/128控制步真实Isaac smoke完成：初态tilt约90度，四个initial_done均False，所有观测和reward有限，termination_counts全0；实际PD60/2与轮effort3Nm读回确认。首次physics配置None导致启动失败，修复为显式PhysxCfg与TGS外力每迭代开启，失败日志保留；最终recovery_native_smoke.json确证成功，不以进程exit0单独认定。该smoke没有起身策略、更未证明复位成功，v266正式行走动力适应未受改动。

### 248. 独立复位新网络PPO接线通过，动作/交还契约落实

v268从全新actor开始，不加载行走checkpoint，128env/64rollout/3次PPO更新，run logs/rsl_rl/rmuc_recovery/2026-09-29_17-05-40_recovery_smoke_v268/model_2.pt已保存。worker状态complete且训练PID结束，update_audit.json对首/末checkpoint确认actor参数改变且所有张量有限；不是起身成功证据，未启动长预算复位训练。

runtime_contract_readback.json读真实run params：PD60/2、轮effort3Nm/damping0.0225、每腿径向10.775N、DR0、policy_outer_feedback_enabled=false、观测tracking_error_with_target与2.1/.5/1缩放。新增docs/rmuc_recovery_contract.md明确复位零命令但速度误差槽仍随真实状态变化；腿action映射完整可用关节范围而非行走0.22尺度，25D/6D维度相同不等于consumer可混用。交还必须按实际目标/力矩契约映射历史，并训练/验收，不能靠同维度生搬。

行走v266继续原100更新预算，期间实际观测到75更新、全部冻结审计通过；无更换权重/停止/重启。后续先查v266最终结果，复位另需独立deterministic全episode起身评估与完整导出，不将本次3更新或reward视为掌握。

### 249. 完整回合复位评估接通；独立倒地自起立正式训练启动

新增scripts/evaluate_rmuc_recovery.py，读取独立checkpoint确定性actor，对四种姿态首完整回合在_get_dones自动reset之前捕获up/height/ready/终止原因，独立累计连续起稳时长及ever_established，要求终端仍established且无unsafe才算候选hold通过。报告明示handover/最高DR/certified=false，不使用reset后的站姿误计成功。

v268三更新模型实际四env完整约8秒，四方向均超时且无起稳，最长ready0，末倾角约180°、高度0.07m；证据continuation_v268/recovery_evaluation.json。训练链路可用而技能未获得符合预期，不能把此前短smoke当已学。

正式v269从新网络开启初期复位探索，128env/128rollout/1000更新预算，四固定90度倒地姿态、DR0，轮3Nm/damping0.0225、每腿径向10.775N及腿PD60/2/40Nm，拓扑不变；不加载/改行走checkpoint。worker891687/trainer891715实际运行，run2026-09-29_17-11-32_recovery_initial_v269，初始化和首动作成功。artifacts/continuation_v269保存command/experiment_contract/worker/status；无短期平台提前停，完成自动四姿态每种4replicas首回合验收，不自动晋升。当前固定姿态只是初期课程，后续必须扩展姿态/关节/跌倒动能随机化、最高DR、完整图导出与行走交还。v266行走动力训练同时继续原预算，不重启、不缩目标。

### 250. v266完成100更新进入验收；独立复位导出和动作契约验证

v266 status实查已evaluating，最终model6384，actual_updates100，仅新physical network6张量变化，旧图冻结与完整导出通过。eval892858实际存活，末日志steps2000/min_completed16，最终八点结果未出，不能因训练完成称通过。v269独立倒地自起立继续1000更新预算，实际日志到34更新附近，无重启。

新增scripts/export_rmuc_recovery.py，检查实际run params标识recovery_prototype/无任务外环/正确观测模式，拒绝含rmuc行走残差的模型。由USD关节范围+margin和实际PD/轮配置生成独立action_contract，记录checkpoint/params/USD hash、policy_kind=rmuc_recovery_v1、左右关节顺序、leg normalized→物理范围映射、PD周期与策略周期，声明不能混用locomotion consumer；仅导出raw归一化动作，不把执行器映射藏成另一个任务外环。

v268已知未起身的model2用于导出接线验证，完整JIT/ONNX verifier通过，四个实际倒地首帧又用ONNX Runtime对TorchScript通过（fallen_observation_parity.json）。这些只证明导出数值一致，runtime_validation_complete/hardware/handover均false。v269当前worker原流程只有训练和评估，结束后仍需执行新导出入口，不能因代码存在宣称最终模型已导出。文档rmuc_recovery_contract与交接记录更新。

### 251. v266验收仍1/8但明确改善，保留optimizer续训v270

v266最终八点全部32回合生存1，仅-1.2通过。相对v263，+0.8速度/yaw0.283852/0.270058→0.124338/0.142444，+1.4为0.398406/0.322931→0.173991/0.113741；-0.8 yaw有退化0.029849→0.066571，故不做全部改善的笼统结论。behavior_comparison.json记录各点与相同初态校验。当前25%动力未完成，不推进更难物理，也不新增专家。

v270从v266/model6384保留physics_adaptation/critic optimizer组、critic与iteration，warmup0继续100更新。只physical head训练、旧图冻结、相同奖励/物理/rehearsal/DR0，完成自动导出与八点验收，无自动晋升。启动worker904518，状态位于artifacts/continuation_v270/status.json。复位v269独立继续1000预算。用户询问训练记录/导出位置，已核实v266/export有policy.onnx/policy.pt/action_contract/export_parity，v268/recovery_export为未学会起身的短训练接线文件，不称正式复位成果。

### 252. 独立复位ONNX→MuJoCo完整回合诊断链实际运行

新增sim2sim/run_recovery_onnx.py，读取独立recovery动作契约并校验policy_kind、ONNX hash、25D/6D、关节顺序、速度误差观测模式、PD60/2/40Nm/2ms、策略8ms及腿目标范围；拒绝行走consumer契约。export_rmuc_recovery.py新增ONNX文件hash绑定，v268重新导出。runner使用共享leg_pd和radial_spring、关节输出侧Nm及3Nm轮限制，无拓扑变更，无额外任务反馈。

v268三更新未学会模型四倒地方向各8秒MuJoCo实跑完成，longest_ready均0，完整obs/action/接触/力矩轨迹保存在artifacts/continuation_v268/mujoco_recovery_diagnostic.json。与Isaac四点均未起身事实一致，但不能宣称轨迹等价或sim2sim技能通过；本诊断contact判据及终止保护与Isaac尚不完全一致，报告certified=false/handover=false明确保留。

训练期间CPU验证没有更改正在运行的训练配置；v269实际日志79/1000，v270同25%动力100更新续训实际进行。后续正式recovery checkpoint仍需本入口以及严格独立恢复/交还/最高DR验收，不能将短训练模型导出当交付。

### 253. 复位MuJoCo按导出契约使用同一阈值与监控关节

export_rmuc_recovery.py新增recovery_criteria/termination_limits/episode_length_s，读取真实run配置和源USD十四关节范围，连续轮无界记录null。run_recovery_onnx.py从契约读取起稳/终止条件，增加关节超界、root过高和XY逃逸检查，避免此前仅高度低/速度大的保护不完整。轮接触由各点法向力绝对值求和改为世界法向力向量合成后取模，并包含wheel实际接触而非仅ground，接近Isaac net_forces_w语义。

v268未学会模型重新导出通过，MuJoCo四初态各8秒真实重跑，最长ready均0、duration完成，mujoco_recovery_contract_guards.json记录。此为判据语义对齐进展，接触求解器/PhysX速度限制及terminal采样时序仍未证实完全一致，不产生技能证书。训练中recovery_env.py未改，v269仍原1000预算运行。

v270实际jsonl14/100，全部旧参数/buffer冻结检查通过；v269日志87附近继续。保持双任务预算，不因短期曲线修改奖励/物理或提前晋升。

### 254. 复位model100曲线提升但四姿态真实起身仍为0

v269 model100完整四姿态确定性评估已结束：front/back/left/right均8秒timeout，longest_ready_seconds=0，最终root高度约0.07m、倾角约180°、recovery_hold全false；recovery_behavior_summary.json。TensorBoard training_curve_snapshot.json对应平均episode length从56升至999、reward从约72升至442，说明策略学会避免早终止/保持有限状态，但没有形成起身行为，不能晋升或扩大DR。

下一步不盲目把1000预算继续烧完：先检查动作输出/腿目标是否饱和、轮腿分工及起身奖励项与ready判据是否匹配，再在保留四初态和同PD/3Nm契约下进行针对性课程修正。行走v270和复位v269仍为独立任务，未混用checkpoint。

### 255. 复位model100动作/探索审计，独立新技能PPO配置准备

v272完整导出model100后MuJoCo四姿态8秒实跑，ready仍0。实际归一化动作绝对值峰值约0.269，四组所有action_clip_fraction=0，四腿峰值力矩约7.8–36.7Nm均低于40，故本诊断不支持动作/力矩硬饱和导致起身失败。报告action_exploration_audit.json明确为MuJoCo，不能代替原生Isaac全轨迹。

核对v269实际params/TensorBoard，fresh recovery继承公共yaml中的行走精调std0.004、lr2e-5、epochs1。学习信号弱与极低探索相关是合理假设，未做配对新训练不能当因果。新增独立RmucRecoveryPPORunnerCfg，Recovery注册只指向该cfg：std0.15、lr3e-4、epochs4、entropy0.005，128rollout/1000预算，网络尺寸不改。这些为后续对照默认，当前v269已加载旧配置，不修改运行中的探索/奖励/动力或重启；保留原1000预算，遵循用户“新技能可跑满上限”，第254条关于先决定是否继续的措辞不得成为短期失败提前停止依据。

代码语法检查通过，新配置尚未原生启动，不宣称已改善。v269与v270实查仍存活，后续可在完成本预算后用相同四姿态比较独立探索配置，其他泛化/DR/交还目标不变。

### 256. 新复位探索配置原生smoke完成，纠正reward增长归因

v273从新网络3更新验证独立RmucRecoveryPPORunnerCfg，run2026-09-29_17-34-41_recovery_exploration_smoke_v273/model2。实际params确认init_noise_std0.15、lr0.0003、epochs4、entropy0.005，末actor std约0.1474–0.1547，update_audit与runtime_config_verified.json记录权重更新/有限性。未把smoke权重当正式起身成果，未改变v269已运行参数或停止其1000预算。

直接提取真实recovery_env._get_rewards AST，分别给定静态倒置/侧倒/起直已保持/起直未稳定并令effort与动作变化0：每步reward依次0.35/2.425/10/5。倒置999步未折扣累计约349.65，reward_counterfactual.json记录。这说明正height项仍给倒置静止奖励，但起直保持奖励更高，不能直接说奖励数学上偏好倒置或证明其为唯一原因。

同时训练日志明确randomize initial episode length=True，初始episode length56→999不必来自策略减少危险终止。收回第254条“说明策略学会避免早终止”的因果表述；只保留reward/长度曲线原始值及model100四姿态未起身实测。下一次对照需区分探索与reward改动，当前任务仍未获得复位技能。v269与v270继续，goal保持active。

### 257. 复位跨引擎采样时序与终端动作记账修正

发现MuJoCo复位runner先在t0将ready加一个控制周期，且原循环在最后积分前记录状态、没有terminal采样；原8秒长度也与Isaac recovery _get_dones的max_episode_length-1不一致。export_rmuc_recovery.py新增first_episode_control_steps=ceil(episode_length/policy_dt)-1，当前999步/7.992秒。runner改为包含最终控制边界，t0不计保持、整数hold计数，每个实际间隔末更新，终止时不生成未执行action。日志由含糊action分为previous_executed_action及next_action，terminal next_action=null。

model100重新导出并MuJoCo四姿态各999步实跑，四方向ready仍0；model100_terminal_aligned.json保存，terminal_timing_audit.json检查每例1000观测采样/999动作间隔，最后时间7.992，历史action等于上一帧已执行动作，全部通过。只校验采样/记账，不外推仿真接触/速度限制完全等价或行为通过。

v269复位trainer891715和v270行走trainer904554末次实查均仍运行（约26分24秒/17分20秒），未重启或改训练PD/拓扑/奖励。继续预算完成后独立验收，goal保持。

### 258. 两任务全预算完成：行走6/8，复位0/16；新对照与续训启动

重新实查v270和v269 status均complete、原trainer结束。v270/model6484八点各32回合全存活，6点通过（-0.8/±1/-1.2/±1.4），剩余+0.8速度/yaw0.025475/0.053168、+1.2为0.085296/0.062187未过。behavior_comparison.json配对v266同初态，不能放宽门槛或把接近算完成；仍25%中间动力，无最终3Nm/DR认证。

v269原1000预算确已跑完，model999四种固定姿态各4replicas共16回合仍longest_ready0，全部约180度倒置超时，非起身成功。按独立入口补导出最终recovery_export，完整数值校验通过但behavior仍失败，不当可部署模型。

v274从全新网络same seed268启动复位1000更新，使用已原生验证的独立PPO组std0.15/lr3e-4/epochs4/entropy0.005，奖励/PD/物理/reset保持；实际params读回正确，env配置仅空间对象序列化字段不同，runtime_verified.json记录。整组PPO变化是实验因素，不能归因某单一超参。worker1180367/trainer1180423已运行、初始化/首动作/iteration0正常。

v275从v2706484保留physics_adaptation+critic优化器、critic、iteration，warmup0继续100更新，同25%物理、奖励、rehearsal、DR0；不增加专家、不解冻旧图、不提高强度。worker1180368/trainer1180424实查运行，首动作成功，结束按原流程freeze/完整导出/八点验收。两个新任务状态分别continuation_v274/v275，勿按旧PID重复启动。goal继续。

### 259. 按用户要求建立独立复位奖励模块与显式版本切换

用户要求参考华南虎为自复位等课程单独reward。新增recovery_rewards.py，独立项upright、upright_height、upright_progress、established、action_rate、leg_effort、time_cost及可配置权重。height乘max(up,0)，水平/倒置不再累计该项；progress按上一步up差计算，reset首帧不产生伪进展。旧legacy_v0原式保留，当前v274正在进行的PPO对照不改，后续显式选择env.recovery_reward_profile=recovery_v1；没有偷偷放宽起稳门槛、改PD或拓扑。

tests/test_rmuc_recovery_rewards.py四项通过，覆盖倒置高度不套利/保持优先、reset进展与反向惩罚、action/effort仅代价、未知key拒绝。scripts/probe_rmuc_recovery.py增加reward-profile并记录加权项；v276原生四env/128步新reward_v1检查通过，四90°初态无即时终止、有限观测/reward、termination0，报告reward_smoke.json。尚无长训练效果或起身证书。

evaluate_rmuc_recovery.py改为从checkpoint的params恢复reward profile和权重，老文件无字段仍使用legacy，避免新代码改变正在跑的任务自动评估口径。当前v274/v275实际进程1180423/1180424均存活。同期prepare_recovery_curriculum.py生成30/45/60/75/90度各四方向共20个CPU几何候选，地面间隙3mm/闭链误差<1.8e-16，旧90度初态逐值一致；仅候选备选，未接入本轮训练。

### 260. 独立recovery_v1奖励实际PPO与导出验证完成

v277沿用v273新探索PPO设置及相同seed268，只显式选择env.recovery_reward_profile=recovery_v1，原生128env/64rollout/3更新实际完成。run2026-09-29_21-49-56_recovery_reward_smoke_v277/model2，update_audit确认参数更新/finite；runtime_verified.json读真实params reward profile/weights、std0.15、lr3e-4正确，完整导出验证通过。export_rmuc_recovery.py新增training_reward_profile/weights记录，不把旧奖励和新奖励产物混称。

此为新奖励参与真实PPO更新且可导出的证据，尚无起身效果证据，不将短smoke自动升级为长训练或宣称解决失败。当前v274 legacy奖励的探索对照仍按1000预算运行，末日志75；v275行走同阶段续训35/100，旧参数/buffer冻结全部通过。训练中的reward、PD、拓扑不变，后续依据完整对照结果单独启动新reward正式实验。

### 261. v275完成100更新与完整导出，进入八点行为验收

v275已完成100次更新，最终model6584，freeze_audit.json确认仅rmuc_physics_adaptation.network的6张量变化，全部逐更新旧参数/buffer冻结检查通过。export/export_parity.json完整导出通过。实际worker1180368转入evaluating，eval1213250存活；按原八点、每点16replicas/min32回合、3500步执行，同25%中间动力/DR0，无外环。结果尚未产生，不能把导出成功当行为通过，也未晋升物理强度。

v274独立复位仍legacy奖励的新PPO对照，trainer1180423持续运行，末查218/1000附近；不更改本轮奖励、PD、拓扑或预算。新recovery_v1正式训练仍待该对照完整结果。

### 262. v275中间动力7/8通过，保留优化器续训v278

v275/model6584八点各32回合全部存活，7/8通过，v270原六通过点保留，+1.2新增通过（速度0.058671、yaw0.048330、stable_fraction0.544314）。唯一+0.8仍失败（速度0.006027、yaw0.052974、stable_fraction0.206940），不放宽门槛；部分负向误差有所增加，不能称全面改善。behavior_comparison.json与v270逐点对比并确认initial_state_audit完全一致。仍25%中间动力/DR0，非最终硬件和最高DR。

v278从6584保留optimizer/critic/iteration、warmup0继续100更新，仅physical head，旧图冻结、相同rehearsal与奖励/动力；不新增专家或提高物理强度。worker1220980/trainer1221080实际存活，首更新完成；完成后完整导出及八点验收，无自动晋升。v274复位继续原1000预算。

### 263. 补齐自复位 reward/状态日志通道

核查发现 recovery_env 原先将 recovery 指标写入自定义 `extras['recovery']`，而当前 RSL-RL logger 只消费 `extras['episode']` 或 `extras['log']`，因此训练曲线不会记录 ready/established 及 reward 各项。已在 `recovery_env.py` 增加每步 `episode` 日志：ready、established、up、root height，以及 recovery_v1 的七个加权 reward term；每步创建新字典，避免 logger 保留引用后被下一步覆盖。奖励计算、PD、观测、拓扑和正在运行的 v274/v278 条件未改变。

`artifacts/recovery_logging_audit/source_method_check.json` 用 legacy_v0/recovery_v1 两种 profile 执行实际 `_get_rewards` 方法检查，确认键集合、fresh dictionary 和有限 reward；`py_compile` 与 `tests/test_rmuc_recovery_rewards.py`（4 passed）通过。该修复只影响后续启动的 recovery 训练日志，不追溯修改当前已加载进程。

### 264. v274复位未通过、v278停留7/8；启动独立奖励正式对照v279

v274完整1000更新及16固定倒地首回合评估结束，0/16 recovery_hold通过、longest_ready均0。部分回合best_tilt接近直立，但终态不稳定且部分joint_limit终止；不能宣称学会起身。v278完成100更新，冻结/导出通过，八点仍7/8；+0.8 yaw0.052961、stable_fraction0.207149，与v275几乎停滞，暂不盲目再续相同配置或晋升动力，下一步诊断该命令的学习信号/执行动作。其余七点保持通过。

v279从fresh seed268启动独立recovery_v1正式1000更新对照；与v274相同PPO/四90度初态/物理/PD，只切reward profile并记录新增日志。worker1702547/trainer1702567实际运行，首更新完成；训练日志已实际出现RecoveryReward七分项，补齐第263条仅源码检查的运行证据。未加载行走权重，仍3Nm/每腿10.775N/DR0，自动最终16回合验收，不自动晋升。完整目标保持。

### 265. +0.8停滞的样本覆盖与限幅审计

v270/v275/v278三轮日志和评估配对审计：+0.8样本占比约5.57%/5.57%/5.32%，而±1.4合计约50%；+0.8 yaw误差0.053168→0.052974→0.052961。评估力矩饱和0、平均raw/executed动作相同，physical residual采样峰值未到预算。奖励源码在0.053附近仍保留yaw误差梯度。不能直接归因为力矩饱和或奖励无梯度；采样偏重已过±1.4是可检验假设，下一步核实stage映射后做聚焦采样对照并保留rehearsal/八点验收。证据artifacts/continuation_v278/plateau_audit.json。v279实际进程继续，未改运行中的奖励/预算。

### 266. v280仅调整复习采样权重，聚焦±0.8对照

核实configs/rmuc_wheel_leg.yaml的stage48为1m/s、0.8rad/s幅值（双向采样）。v280从v278/model6684保留optimizer/critic/iteration、warmup0继续100更新，仅优先rehearsal权重[1,1,1]→[8,1,1]，对应stage[48,49,50]；其余奖励、动力、课程主阶段、连续命令、冻结边界及八点验收不变。此为采样对照，不是已证明根因或晋升。

worker1705362/trainer1705416实际运行，前三更新冻结检查全通过；第3次实际+0.8占2068/16384≈12.62%（原整轮约5.32%），-0.8为1800，采样调整已生效，但短窗不代表整轮分布。证据artifacts/continuation_v280/experiment_contract.json与startup_audit.json。v279新奖励复位trainer1702567同时存活，继续1000预算。

### 267. 数值超限与恢复动作剧烈程度诊断

v281复现v274终止：6次joint_limit中4次仅超2.384e-7rad（float32量级），另外2次right_hip_pitch超0.005596/0.006618rad。recovery_env新增默认0的数值容差配置（允许0至1e-5rad），当前训练默认和已加载进程不变；评估入口支持显式诊断1e-6并记录报告。v282诊断启动，不将其当原口径验收。

用户要求控制自复位动作剧烈程度。evaluate_rmuc_recovery.py新增每回合腿目标单步变化（含首动作跳变）、实际腿速度、施加力矩及最大body净接触力审计；仅控制边界采样，不冒充substep冲击峰值。v283用同v274checkpoint/1e-6容差启动动作审计，launch.json记录PID1713953。语法检查通过，数值容差CPU检查确认屏蔽2.38e-7误差但保留真实0.0066超限。未来若采用目标变化率限制必须同步Isaac/导出/MuJoCo/实机，不能部署单独滤波。当前v279奖励/PD/动作接口不变，待诊断结果决定调整，不以加大惩罚代替物理检查。

### 268. v274动作强度实测及数值容差对照结果

v282与v283完成，v274checkpoint加1e-6rad诊断容差仍0/16起稳；v283剩2例真实joint_limit。排除float32误终止不足以解决恢复。动作审计显示各姿态腿目标最大单步变化1.85–3.08rad（含首动作），实际腿速峰值约17rad/s、16例均触及40Nm，最大单body净接触力约17029N。指标仅控制边界采样，不是substep峰值，也未定位冲击部位/时刻；不能直接拿该峰值设阈值。证据v283/motion_evaluation.json、motion_summary.json。

这与先前v272 model100低动作/未饱和结论不矛盾：当前是v274/model999另一策略，不能沿用早期动作温和判断。下一步准备恢复专用腿目标变化率对照，并同步训练/导出/MuJoCo契约，保持PD60/2/40与轮3Nm不变；当前v279完整reward对照和v280采样对照不中途改变。未宣称根因唯一或恢复技能通过。

### 269. 大接触力峰值定位于倒置机身

v284原生16首回合接触峰值定位完成。back_down四例最大净接触力15.07–17.03kN均在root body robot，倾角约178–179度、root高度约0.071–0.072m，腿速度约13.5–14.5rad/s且腿力矩40Nm；发生于0.376s或5.48/6.82s。该现象对应倒置机身接触，不是正常直立轮地支撑，但尚未用接触点/穿透深度分离真实冲击与求解器效应。其他姿态峰值分布在轮、平行连杆、下腿和导向轮，不能统一把所有非轮接触都当错误支撑。证据artifacts/continuation_v284/contact_evaluation.json。

评估脚本已新增contact_peak_event（body/时刻/力/姿态/高度/腿速/力矩），原生运行通过，未改变训练奖励。下一步目标变化率对照优先，接触惩罚按部位及峰值分布设计，不以单个17kN作为直接阈值。当前v279/v280继续，完整目标未完成。

### 270. v280采样对照训练与导出完成，进入验收

v280完成100更新，model6784，全部冻结检查通过，仅physical network6张量变化，完整导出parity通过。整轮+0.8样本占15.57%（源v278为5.32%），sampling_audit.json保存计数。当前eval1730024实查存活，按既定八点各至少32回合执行，未产生通过结论；不能以采样增加代替效果证据。v279新奖励自复位仍运行，预算不变。

### 271. v280采样对照未解决+0.8并退化-1.2，不晋升

v280/model6784八点评估完成，6/8通过；+0.8 yaw0.052765、stable0.225962仍失败，-1.2由源v278的yaw0.034578→0.054937、stable1→0退化。提高±0.8复习覆盖改善-0.8但没有解决正向瓶颈，不能将样本不足视为已证实根因。behavior_comparison.json保存同initial_state_audit配对（逐值一致）。拒绝晋升v280，保留v278/model6684的7/8为后续基线，不继续盲目扩大同采样权重；下一步需要检查正向目标处共享残差与冻结源图反馈的局部敏感性，再决定修改。没有提高动力强度或放宽门槛。v279新奖励复位继续原1000预算，末查300附近。

### 272. 启动+0.8轮差分局部敏感性诊断

v285使用v278/model6684固定1m/s、+0.8，五个轮差分动作偏置0/±0.01/±0.02，每种4replicas、至少8回合、3500步；2秒后1秒渐入，腿动作不偏置，同25%动力/DR0/无任务外环。偏置只用于局部闭环敏感性诊断，不写入checkpoint，不作策略通过证据。命令与偏置文件在artifacts/continuation_v285，实际进程1735720初始化及首动作完成，结果待出。v279恢复训练1702567仍存活。

### 273. 小轮差分可改善+0.8局部误差，但不作为部署补偿

v285五种偏置各8回合全存活。本诊断布局零偏置速度/yaw误差0.010970/0.057223；+0.01轮差分为0.041986/0.027760、stable1；-0.01 yaw0.125883，+0.02 yaw0.110619，-0.02 yaw0.193487。说明局部小幅轮差分有修正余量，大偏置可过调，不能将误差完全归于物理不可控。每case独立bucket、4replicas且与原八点评估布局不同，零偏置基线也不同；不声称多seed或同轨迹因果认证。未将常量补偿加入策略或任务外环。证据v285/sensitivity.json与summary.json。下一步检查策略在该命令处的梯度/优势及奖励折中，利用这一方向作诊断而非直接硬编码。v279自复位仍运行。

### 274. 奖励审计恢复低速yaw开关，启动真实配置扫描

检查evaluate_rmuc_grid的audit_training_env_config发现只恢复bank scale/start level/reward clip，漏读训练显式启用的low_speed_turn_yaw_tracking_enabled，默认False会造成奖励审计口径错误。现仅在显式audit配置入口增加恢复该开关、reward_contract_by_command和require_analytic_bank，并记录audit_reward_overrides；普通八点行为评估不改变。语法检查通过。

v286沿用v285偏置诊断，加audit_reward_contract和v278真实params/env.yaml，末态扫描yaw误差与raw/clipped reward。实际进程1741129完成初始化/首动作；结果尚未出。不是优势梯度因果证明或策略通过证书，当前v279训练不改。

### 275. yaw奖励扫描有信号，当前复习口径一致

v286真实训练reward开关扫描完成。当前consistent模式主课程/复习、level26/31固定同态reward差均0。零偏置bucket固定末态yaw误差0/0.025/0.05/0.06/0.075/0.1对应raw=clipped reward约-12.0338/-12.2521/-12.4910/-13.0645/-13.8406/-14.9405，high_clip_fraction均0。说明该样本处降低yaw误差有奖励信号，不支持此处奖励被裁剪为平台的解释。仅末态改yaw传感值反事实，不能代替真实训练轨迹优势估计或排除速度/姿态等权衡。summary.json保存；未改变正式策略或晋升。下一步应检查on-policy优势/共享梯度或安排隔离修正试验，避免继续无根据增大reward。v279继续原预算。

### 276. 启动±0.8随机策略critic诊断

v287使用v278/model6684与实际params奖励配置，1m/s±0.8各8replicas/至少16回合/3500步，随机策略及checkpoint critic，gamma0.99与训练一致。不加动作偏置，使用现有returns/tail bounds诊断比较预测价值与采样回报；仅诊断，不能当确定性技能认证。进程1746270实查存活并完成首动作，command/launch/evaluation.log位于artifacts/continuation_v287。v279自复位仍运行。

### 277. critic存在方向相关偏差，尚非停滞因果证明

v287随机策略critic诊断完成，±0.8各14096个满足tail bound<=5的相关状态样本。+0.8预测596.18/回报665.77、bias-69.59、RMSE145.78、EV0.846；-0.8预测623.76/回报566.99、bias+56.77、RMSE101.81、EV0.967。整体bias-6.41掩盖方向差异。按obs yaw误差分层，正向<0.05bias-83.49、0.05–0.1bias-34.29；负向对应+33.27/+56.05。保存critic_stratified.json和原始critic_samples.pt。

该证据提示估值校准偏差，不能证明导致actor停滞：样本相关、条件状态分布不同、常量baseline偏差不必改变期望策略梯度，且尚未检查实际PPO优势及梯度。后续宜做冻结actor的critic校准对照或直接记录训练优势，不能凭整体平均误差正常排除问题，也不据此解冻旧技能图。v279训练继续，无技能晋升。

### 278. v288冻结actor进行critic校准对照

v288从v278/model6684启动50更新critic-only校准，warmup50覆盖全预算，actor/噪声全部冻结，保留优化器、原[1,1,1]复习权重和相同物理/奖励。不是续训v280退化策略。worker1751633/trainer1751669实查运行，首批更新actor_trainable_fraction=0。结束worker逐值比较完整actor_state_dict并要求critic发生更新，再用同v287命令/seed/副本数重新诊断；不执行无意义的确定性技能重测或宣称技能提高。完整actor冻结最终检查尚待结束。v279恢复训练继续。

### 279. critic校准完成，完整actor逐值不变

v288已完成50次critic-only更新，model6734。freeze_audit.json逐值确认完整actor_state_dict与v278一致，critic8张量变化，动作策略未改变。worker转入同v287随机策略估值诊断，eval1762010已启动；估值改善待结果，不新增技能认证。v279新奖励复位仍运行。

### 280. critic校准关键正向RMSE退化，不晋升

v288复验完成，完整actor不变，v287/v288的observations、returns及tail bounds逐值一致，可同样本比较critic。+0.8 bias-69.59→-60.19但RMSE145.78→162.85、EV0.846→0.785；-0.8 RMSE101.81→94.80。整体校准没有改善关键正向质量，不采用v288作为后续优选基线，保留v278/model6684。不把平均bias缩小称校准成功，不继续盲目加critic-only轮数。comparison.json保存，下一步检查实际PPO优势/梯度信号或隔离共享修正影响；v279继续原预算。

### 281. 增加PPO轮差分优势信号审计

rmuc_position_authority_training.py增加wheel_advantage_audit，在物理分支PPO update前读取原始rollout obs/actions/Gaussian mean,std/advantages，按八命令记录优势均值、轮差分(+right,-left)score×advantage均值及命令内中心化版本。Gaussian params顺序已核实当前rsl_rl源码为(mean,std)。仅局部旧策略surrogate上升导数，不是实际网络梯度、optimizer方向或物理因果；样本相关不输出虚假独立置信度。

CPU合成数据正/负优势方向检查通过；尚未在实际训练短rollout验证，后续需独立短诊断，不自动加载当前恢复进程。未修改actor/critic损失、门槛或执行器，v279继续原预算。

### 282. PPO优势审计原生通过，启动样本方向与稳态修正不同

v289从v278源checkpoint原采样配置完成3次实际PPO更新，逐更新冻结检查全通过，wheel_advantage_audit真实字段读取正常。+0.8的差分surrogate上升导数三次-2.056/-4.773/-5.857，命令内中心化后-1.637/-6.469/-6.020，方向与v285稳态正偏置有效相反。仅三个启动rollout，可能包含欠转加速与稳态过转混合；不能断言梯度错误或critic唯一根因。需要按yaw误差符号/阶段细分并延长诊断覆盖稳态，未晋升该短训练模型。summary.json保存，v279继续原预算。

### 283. PPO优势按欠转/过转分组，启动20更新诊断

优势审计增加按obs8*sign(command yaw)分under_turn/near_target/over_turn，阈值±0.05rad/s；只区分误差，不把near_target自动称稳态。v290从同v278源模型启动20更新诊断，原物理/奖励/采样；进程1770102实查运行，前4更新冻结通过，各组计数和等于命令总数。第4轮+0.8过转组151样本，surrogate+7.267/中心化+7.519，整体+0.921，说明前三轮负方向不应外推为恒定错误梯度。需等待完整20更新统计，避免选单轮作因果结论。当前v279恢复训练继续，未晋升短诊断模型。

### 284. 分组优势汇总完成；新奖励自复位全预算仍未通过

v290完整20更新结束，逐更新冻结通过，最终model6704相对v278仅physical network6张量变化。+0.8欠转1346样本差分surrogate均值-6.520、近目标9259样本-0.973、过转6320样本+1.900；组内中心化后分别-1.103/-0.338/+1.490。过转方向与v285局部有效修正一致，不能把混合负信号称策略学反。误差分组不等于时序稳态、surrogate导数不等于共享网络梯度干扰。完整统计v290/summary.json，未晋升诊断checkpoint。

重新实查v279已跑满1000更新且自动16回合评估complete，recovery_v1仍0/16起稳，所有longest_ready0；原trainer已结束，勿当运行中或重复启动。behavior_summary.json按四初态汇总。该结果不支持仅靠本次奖励改动和预算即可解决恢复；下一步应针对已实测目标跳变/饱和设计一致的动作变化率实验及渐进倒地课程，不重复相同配置盲训。尚无可部署恢复策略，完整目标继续。

### 285. 恢复限速诊断实现纠错，v291不作为有效对照

v291试运行暴露限速实现错误：physics子步使用step_dt导致速率超设定，首动作直接初始化到请求目标而绕过限幅。已改为cfg.sim.dt并从实测关节位置初始化；评估请求目标跳变字段改名max_requested_leg_target_step_rad，不再冒充PD执行目标。v291已标注invalid_slew_diagnostic.json，不可当8rad/s效果证据；它用v279而旧动作审计用v274，不能比较接触峰值后归因限速。修正后尚待原生重验，正式训练未启动，部署契约同步仍待完成。默认限速关闭，旧策略条件保留。

### 286. 恢复8rad/s实际PD目标子步验证通过，对照待完成

v292同v279/model999、同4replicas每初态及1e-6数值容差进行串行限速/不限速对照。evaluate_rmuc_recovery新增最终set_joint_position_target写入截获，每physics子步验证实际腿目标增量，包含首动作且只统计首回合。限速组已输出16回合，最大增量0.016000032rad/2ms，8rad/s断言全部通过，起稳仍0。不限速基线待完成，不依据旧checkpoint结果宣称减冲击有效。正式训练未启动，export/MuJoCo契约同步仍需完成；当前限速默认关闭。

### 287. 新增高度达标与连续起稳奖励v2并原生检查

用户建议高度达标和稳定时长奖励，已新增显式recovery_v2，默认仍legacy。v2替换v1高度项为双轮支撑且直立阈值内的高度进度（0.28m封顶、权重2）和高度达标（权重2），增加ready连续保持0→0.5s线性封顶奖励（权重3），失稳hold清零，原established保持奖励保留。姿态/进展/动作/effort代价不改，不增加观测维度。当前直立门槛12度会使高度信号仅在接近直立时生效，尚需学习效果验证，不称解决探索问题。评估读取standing weights、导出记录训练权重。

6项reward单测通过（腾空/倒置不获站立高度分、高度封顶、保持增长与失稳屏蔽）。v293原生4env/128步smoke完成，reward_v2有限且termination均0，站立新项在倒地时为0，artifacts/continuation_v293/smoke.json。未启动长训练，未声称起身成功。v292同checkpoint限速对照两组0/16，前/右倒冲击恶化、后/左倒改善，不能作为旧策略直接部署补丁；comparison.json保留。限速部署同步仍待完成，下一正式实验需明确隔离因素。

### 288. 高度/保持奖励正式对照v294启动，防止实验接口误导出

v294从fresh seed268启动recovery_v2正式1000更新对照，相同128env/128rollout、新技能PPO、四90度初态与物理，仅切换高度/稳定保持reward版本。不限速、joint tolerance0，避免同时改执行链与奖励。worker1853262/trainer1853303实查存活，首更新完成；params读取profile/standing weights及禁用限速、0容差，runtime_verified.json保存。新三项奖励已在原生日志出现，尚为0，不能称学习成功。完成自动16首回合验收，无自动晋升。

评估入口补读checkpoint保存的slew rate并通过cfg初始化验证（兼容yaml .inf）；导出入口暂拒绝有限slew或非零joint tolerance，避免尚未同步的MuJoCo consumer静默使用旧控制/保护契约。该检查不影响旧默认模型导出；语法检查通过。部署一致性工作仍未完成，不将guard当同步实现。

### 289. v295优化epochs对照启动

v295从v278/model6684保留optimizer/critic、warmup0启动100更新，唯一学习超参改动PPO epochs1→4，采样仍原[1,1,1]、同物理/奖励/限幅/旧图冻结，优势日志只读。目标为检验更充分优化是否改善7/8瓶颈，非已证明方案。首次启动被仓库high-speed update-size guard拒绝（无run生成），保留first_attempt日志；独立对照显式加--allow_aggressive_highspeed，不改变freeze或验收保护。worker1872239/trainer1872276实际运行，保存params确认为4epochs，前2更新冻结检查全通过，runtime_verified.json。完成自动完整导出/八点验收，无自动晋升。v294恢复奖励对照继续，不修改其预算。

### 290. 赛场弹击、云台耦合、气弹簧误差纳入最终DR要求

用户新增最终赛场DR要求：弹丸约40–400N、云台世界系瞄准保持/主动转动带来的有效惯量和反力矩、气弹簧精度误差。已写docs/rmuc_competition_disturbance_contract.md，纳入联合最高DR验收而非仅nominal。峰值力不等于冲量，持续时间/作用点/方向待明确；云台静态惯量随机化不替代动态反力矩；气弹簧每腿10.775N，±20%仅既有先验。现有root惯量缩放/弹簧幅值DR/速度push不等于已实现全部要求，保持USD拓扑、25D观测、3Nm及统一PD。后续分项建模/回读再联合训练与MuJoCo验证，不中途改变当前两条对照。

### 291. 装甲板短脉冲扰动范围补充；v295评估进行中

用户补充弹击位置在装甲板、方向随机、持续时间很短（乒乓球弹墙类比），已更新competition disturbance contract。随机点限制装甲有效表面、随机方向限定物理可达入射方向，r×F作用力矩与脉冲冲量同时记录；类比不作为已知毫秒时长。具体装甲几何与冲量分布仍需定位/辨识，尚未接入训练，不冒充DR完成。

v295已100更新、旧图冻结及完整导出通过，八点评估1911259仍存活；v294恢复训练1853303运行。等待结果，不因评估耗时重复启动。

### 292. v295四epochs仍6/8，不晋升

v295四epochs对照100更新及八点评估完成，八点各32回合全部存活但仅6/8。+0.8 yaw由v278的0.052961恶化至0.063946、stable0；-1.2由0.034578恶化至0.052598、stable0.0953，其他六点通过。initial_state_audit逐值一致，behavior_comparison.json保存。不晋升v295，保留v278/model6684的7/8基线；增加epochs与提高±0.8采样均未解决瓶颈，不再重复同方向盲训。共享物理分支存在任务权衡的迹象，但尚未直接测梯度冲突；下一步应测命令间参数梯度/隔离训练，而非宣称已证实残差爆炸。v294恢复训练仍实查运行。

### 293. gate与退化复核；梯度快照复现未过需继续排查

按用户要求细查退化/gate：评估stable本身含窗口yaw误差<0.05，因此yaw和stable失败不是两个独立原因；还要求姿态/侧速/高度/轮接触。v278/280/295的+0.8及-1.2具体checks与signed yaw已保存v296/gate_audit.json。冻结参数不保证共享物理分支作用域内旧行为保持，这解释为什么权重冻结检查通过仍可行为退化，但具体梯度干扰尚未证实。

v296完成单更新rollout采集，新增可选RMUC_CAPTURE_PPO_ROLLOUT=1仅首更新保存obs/actions/advantages/分布参数和actor快照。离线全actor均值复现最大差0.000619322（CPU/GPU及128batch均存在），严格一致性检查未过，故gradient_audit.py拒绝输出参数梯度冲突结论；须先排查采样状态/模型非state_dict配置差异，不能掩盖误差作因果推断。v294恢复训练不改。

### 294. TF32与批形状解释复现误差；发现局部命令梯度冲突

v296离线均值偏差定位：train.py启用CUDA TF32，离线默认未启用；仅改TF32仍受batch形状影响，同时匹配TF32和128环境forward batch后rollout均值逐值复现（max error0）。全16384batch仍约0.000670差异，不能混用精度设置/批形状作严格rollout重放。gradient_audit.py使用128分块前向保留计算图，余弦在CPU float64计算，避免TF32点积误差。

首批真实rollout的完整共享physical network参数梯度显示+0.8与-1.2余弦-0.94038，与+1.2为-0.82713、与+1.4为+0.84612。首次有直接局部参数梯度冲突证据，与多命令退化相符；仍单批启动数据、旧策略surrogate非Adam最终步/稳态因果，不可称唯一根因。gradient_audit.json保存，修正第293条悬而未决的复现问题。下一步应进行有retention的冲突处理/隔离对照，禁止再次仅增epochs或放宽gate。v294恢复训练保持。

### 295. 离线梯度投影不保证全部命令retention

检查当前scripts/rsl_rl与source/agent_rl未发现可直接复用PCGrad入口（历史日志命名不代表当前路径已启用）。使用v296已严格均值复现的八命令参数梯度，在CPU float64做不改actor的投影可行性分析。按样本数加权的原方向对-1.2一阶内积-0.57164；一轮确定顺序冲突投影后-0.01615仍负，且+1.4从+0.67488变-0.04363，说明简单投影不保证全任务不退化。projection_feasibility.json与command_gradients.pt保存。仅首批局部surrogate，不含其他命令/Adam/物理闭环，不作为实际策略退化量或PCGrad最终效果证明。下一实现应检查约束可行性和实际更新方向，保留完整行为retention；当前未改变训练优化器。

### 296. 共同改善方向一阶可行，有限步长仍需retention检查

使用v296同批八命令梯度解CPU float64约束投影：寻找最接近+0.8归一化梯度且所有八组一阶内积>=0的方向。SLSQP收敛，方向norm0.25713，+0.8内积+0.29276、+1.4+0.13382，其余非负（多项在约束边界）。有限步长TF32/128batch原rollout复验：step0精确一致，1e-5/1e-4/1e-3时+0.8 surrogate改善但部分保护命令出现1e-6量级负变化；不能把一阶可行当实际retention保证，也不能把量化/曲率微差直接当显著行为退化。脚本恢复全部参数且未写checkpoint。constrained_direction.json、direction_line_search.json保存。下一步若接入训练必须检查实际参数步与surrogate回退，且仍需完整行为复验，不能直接对投影后的梯度用Adam便宣称保证。

### 297. 恢复未触达站立奖励的反思及再次跌倒惩罚要求

用户指出站不稳再次摔倒应额外惩罚。末查v294974/1000，ready/established/新高度奖励均显示0，root均值约0.1302m，尚不能把现状归为已站稳再摔倒。反思v2高度奖励同时要求双轮接触和12度直立，信号入口过近成功条件；侧倒姿态常数正回报仍存在，继续仅调成功奖权重未必解决起身探索。下一版应分开起身过程塑形与严格成功验收，验收阈值不放宽。

新增再跌倒惩罚的设计要求：只有先到达有支撑的站立姿态/高度并短时保持后才armed，真正低高度或大倾角持续越界触发一次；reset初始倒地不罚、瞬时接触丢失不等同跌倒，重站后可重新armed。惩罚参数和滞回时长需显式记录；目前尚未实现/启用，不声称已经修复。应与渐进初态课程和动作平滑一同规划但分对照验证，不能再以奖励上涨代替技能成功。当前v294保留完整预算和原口径最终评估。

### 298. 侧倒静止零姿态收益、恢复progress接线，下一训练必须渐进

用户明确要求侧倒无收益、加入0–1 progress，并再次要求课程渐进且动作不能过猛。新增显式recovery_v3：静态姿态项在up<=0.5（倾角>=60度）为0，真正翻正的变化仍可获得进展信号；保留站立高度/保持和代价。恢复专用可选progress观测占用原恒零目标vx槽index7，仍25D/6D但语义不同：姿态恢复0–0.5、直立撑高0.5–0.8、满足ready连续保持0.8–1；状态倒退progress可退，不按时钟自动前进。行走/旧恢复默认未变，旧恢复checkpoint不得混用新语义，导出暂拒绝progress模式，部署consumer同步待完成。

上轮再跌倒功能已接入默认关闭：ready连续0.1秒armed，大倾角>60度或root<0.18m持续0.08秒触发一次惩罚，之后重站才rearm，reset清零。v297 penalty5原生4env smoke初始倒地refall_cost0。8项单测通过；v298新v3+progress+refall5原生4env/128零动作检查完成，25D有限、初始progress约0.25276、termination全0，初始倒地再跌倒惩罚0。这些仅接线检查，不是起身或温和动作认证。

v294已正式1000更新+评估完成0/16。下一正式恢复实验必须采用渐进初态和训练内目标变化率约束，先低倾角且验证真实PD目标增量，再提高难度；不能把当前未限速的90度smoke当下一训练方案。8rad/s只为诊断候选，既有对照有方向性退化，不称最优值。渐进课程和部署限速/progress同步尚未完成，本轮不启动全难度长训练。

### 299. 渐进倾角课程接入，30度温和配置原生初始化通过

已把几何审核的30/45/60/75/90度初态接入恢复cfg，recovery_initial_tilt_degrees显式选择，旧90度高度保持原数值。评估默认读取checkpoint保存角度，可显式diagnostic override90做最终完整倒地复验，报告实际角度，防止把低倾角通过冒充90度恢复。没有自动晋级，后续须独立验收及已过阶段retention。

v299用recovery_v3/progress/refall5、30度、4rad/s目标变化率做原生4env/128零动作smoke完成，初始倾角四例约30度，termination全0、refall cost0。4rad/s是温和起点候选而非实测硬件最优，零动作检查不能证明随机策略的实际目标限幅或学会扶正；需非零动作子步检查及训练smoke。未启动长训练，deployment progress/限速consumer同步仍待完成。证据v299/smoke.json，USD拓扑/腿PD/轮3Nm不改。

### 300. 非零动作限速验证通过，反思落实清单明确剩余项

用户要求落实反思。docs/rmuc_recovery_contract.md新增已修正与未完成清单：侧倒收益、稀疏高度入口、90度直接探索、动作时间尺度错误、progress部署同步、冻结参数不等于行为retention。明确高度项仍有12度/双接触门槛，不能称过程奖励问题全部解决。

v300非零正反目标切换检查完成：4env、128控制步、512physics子步，每个最终PD目标增量<=0.008000002rad（4rad/s×2ms），包含首动作和第64步显式reset，termination全0。脚本probe_rmuc_recovery.py新增exercise-slew模式及实际writer审计。仅目标slew验证，不证明实际速度/冲击或扶正成功。新正式训练尚未启动，仍需课程过程信号和部署consumer一致性。

### 301. 连续撑高信号与30度温和PPO短验证完成

recovery_v3的supported_height过程项改为supported×clamp((up-0.5)/0.5)×clamp(height/0.28)，倾角60度以内有连续撑高信号，侧倒/倒置/无双轮支撑仍0；严格height_reached/ready/hold成功阈值不变。9项reward测试通过。此为过程塑形候选，不代表所有倒地状态均有密集奖励或已消除全部局部最优。

v301使用30度、4rad/s目标slew、progress index7、refall penalty5、v3完成3次原生128env/128rollout PPO更新；model2有限且actor相对model0有变化，实际params回读正确。supported_height日志0.0964/0.0171/0.0093，证明在该课程可触达，但下降不能解释为效果改善；没有技能验收。runtime_verified.json保存。未启动长训练，下一步仍须同步progress/slew到导出与MuJoCo并验证实际序列，再正式渐进训练。

### 302. 恢复progress/slew导出与MuJoCo consumer实际接通

恢复导出支持明确的新policy_kind=rmuc_recovery_v2（区别reward版本），记录progress index7/公式版本、4rad/s目标slew、reset实测q、训练初态角度。旧v1不变；非零数值joint tolerance仍拒绝导出避免未同步保护。MuJoCo consumer验证新契约，按相同姿态/高度/ready保持更新progress，逐physics子步目标slew并从实测q初始化，按记录的30度初态运行，非90度认证。

v301/model2实际完整JIT/ONNX导出通过，v302/export保存。Numpy/Torch phase公式2048状态一致、1000随机slew步边界通过；MuJoCo四30度初态各7.992秒完整执行，实际目标增量<=0.008rad/2ms，均未起稳。verification.json及mujoco.json保存。这是接口/公式/运行证据，不证明接触动力一致、实机接入、恢复成功或DR；实际硬件controller仍需独立实现/验收。下一步可启动30度温和正式课程，严禁直接晋升90度。

### 303. 30度温和恢复正式训练启动

v303正式30度温和恢复课程启动：fresh seed268、128env/128rollout/1000更新，recovery_v3连续撑高/侧倒静止零姿态收益、progress index7、4rad/s目标slew、再跌倒penalty5，腿PD60/2/40、轮3Nm、每腿径向10.775N保持。worker1986859/trainer1986933实际存活，日志已9/1000，params逐项回读runtime_verified.json。新过程奖励可触达，不把非零reward称起稳。

完成自动4方向各4replicas首回合评估，默认读取30度训练角度，禁止将其通过冒充完整倒地恢复，不自动提高倾角或DR；随后需阶段retention/更大倾角和最终90度随机跌倒验证。初态四固定几何尚非多seed泛化。导出/额外MuJoCo最终模型复验须在完成后执行，当前已有v301短模型接线证据，不代替v303最终产物。

### 304. 实际PPO步retention回退原生接通，三次均拒绝

新增rmuc_physics_retention.py并通过RMUC_PHYSICS_RETENTION=1显式接入physical-only训练。每rollout PPO前保存物理参数/对应Adam状态及观测动作优势；PPO实际步后按1/.5/.25/.125/.0625步长检查同rollout逐八命令及other聚合surrogate，各组不得降低且+0.8须改善。无可接受步则回退物理参数；缩步/回退恢复该组Adam状态，critic更新保留。使用相同forward batch计算前后基准，非实际行为保证；other聚合不能替代全技能保留。旧图冻结/限幅审核保持。

v304从v278基线3次原生更新完成，三次均拒绝/回退，完整最终actor_state_dict与源逐值一致，冻结检查通过。verification.json保存。说明实际回退链可运行，但未解决优化可行性、没有策略改善，不安排无意义八点重测或把全拒绝长训浪费GPU。下一步需要构造约束候选方向而不仅缩普通PPO步，再配合同rollout回退及独立八点验收。v303恢复训练继续，不改。

### 305. 当前反思：过程奖励可达不等于自复位成功，梯度冲突尚未解决

实时核查 v303 `recovery_gentle30_v303` 到 238/1000 更新：`up_mean=0.3406`、`root_height_mean=0.2076m`、`supported_height=0.4629`，但 `ready/established/height_reached/stable_hold=0`。因此当前主要问题仍是从支撑动作进入稳定站立的过渡和保持，不是缺少一个更大的静态姿态奖励；不能用 reward 上升或 supported_height 非零宣称学会起身。30 度课程也只验证温和初态，尚未覆盖真正侧倒/90 度泛化。

旧技能侧，v278 的八点 gate 只有 +0.8 未通过（yaw error 0.05296>0.05，stable fraction 0.207<0.45）；v296 同批局部物理分支梯度显示 +0.8 与 -1.2 余弦约 -0.94038。v304 三次实际 retention PPO 步全部回退，证明“冻结旧图”与“旧行为不退化”不是同一件事，也证明普通缩步不能解决共享物理分支的方向冲突。后续必须先构造满足多命令约束的候选更新方向，再做同 rollout 回退和独立八点验收，不能放宽 gate 或反复消耗 GPU。

### 306. v303中段出现支撑姿态奖励与站立高度脱钩

实时日志约到492/1000更新时，`up_mean=0.6271`、`supported_height=0.6801`，但`root_height_mean_m=0.2148`，`height_reached/ready/established/stable_hold`仍全为0。该证据说明策略可能在轮接触和抬姿态的局部奖励上形成了 reward-hacking，而没有把机体撑到严格站立高度；不能因过程项上升而提前结束或晋级。当前v303仍按原预算运行，下一轮需把支撑质量、机体高度和稳定保持拆开审计，并保留高度门槛。

### 307. v303提前停止及渐进续训方案

用户确认可以在末期无进展时停止。v303于约839/1000更新停止并保留`model_800.pt`：更新300–839期间`root_height`范围约0.2083–0.2574m，严格门槛为0.28m，`ready/height_reached/established/stable_hold`全程为0；这是过程奖励平台，不是训练崩溃。状态和证据写入`artifacts/continuation_v303/stop_reason.json`，没有把缺失的`model_999.pt`误报为完成。

相对初期仍有真实改善：早期高度最低约0.1358m，姿态`up`曾为负；中后段高度进入0.24m左右、`up`约0.45–0.66，但尚未跨越站立门槛。下一轮从`model_800.pt`接续，保持严格成功阈值0.28m，同时将v3过程高度塑形目标单独提高到0.36m，使0.24–0.26m平台仍有上升梯度；该改动已接入配置、评估读取和导出contract，9项reward测试及py_compile通过。续训前先做短预检，再决定是否投入完整预算。

恢复 residual 合入组合策略后必须重新执行 locomotion 八点保护 gate；恢复专用checkpoint不得冒充行走八点结果。八点证据沿用`artifacts/continuation_v296/gate_audit.json`中的v278基线，检查episodes、coverage、survival、linear、yaw、stable、height、bank，任何一项失败都不能晋级。

### 308. 续训预检与稳定优先奖励

从v303 `model_800.pt`接续做了3次受控预检，启用独立过程高度目标0.36m而严格成功门槛仍为0.28m。801次观测到瞬时`root_height=0.3383m`、`up=0.8154`，但802/803分别回落到0.2090/0.2529m，三次`ready/stable_hold`均为0，故不把瞬时高度当成进展证据；结果保存于`artifacts/continuation_v303/rise_target36_smoke_v307.json`。

根据用户要求把稳定性放在最高高度之前：v3新增`stable_pose`连续塑形项，联合双轮支撑、直立程度、平面速度和角速度；严格`ready`与连续`stable_hold`判据保持不变。旧三项standing权重在评估时向新默认权重兼容合并。10项reward测试通过，下一次GPU续训需用短窗口检查stable_pose、ready和stable_hold是否同步上升，再决定是否完整投入。

八点保护不变：恢复分支只有在合入 locomotion residual 后，才执行v278基线对应的episodes、coverage、survival、linear、yaw、stable、height、bank八项验收。

### 309. 稳定目标高度改为显式课程档位

用户强调最高高度不是核心，稳定站立优先。恢复配置现允许将严格稳定门槛`recovery_min_height`按课程设为0.25或0.28m，过程塑形目标`recovery_rise_shaping_height`独立支持0.25/0.28/0.36m；`ready`仍同时要求直立、低平面速度、低角速度和双轮支撑，`stable_hold`仍要求连续保持。评估读取和导出contract均记录这两个高度，避免不同档位混用。v308稳定塑形预检未出现ready/stable_hold证据，未启动长训；下一次先用0.25m严格稳定档位做短窗口，再逐级提升到0.28m。

### 310. 0.25m档位验收揭示主因是冲击与限位，不是高度门槛

对`recovery_stable025_smoke_v309/model_requested_stop_806.pt`做四方向×4 replicas完整评估，16个回合`recovery_hold_passed=0`、`longest_ready_seconds=0`。前后倒方向超时且最终高度约0.174–0.185m；侧倒方向多次因膝/主动连杆越限退出，记录最大机体接触力约4.8–7.1kN、腿部达到40Nm限幅。目标slew子步边界仍为0.008rad，说明仅降低严格高度不能解决动作冲击和稳定保持。

下一版恢复reward新增关节限位屏障和接触冲击软惩罚，分别在硬终止前提供代价；`stable_pose`继续联合双轮支撑、姿态、平面速度和角速度。11项reward测试通过。续训须把目标slew降到更温和档位并先做短窗口，不能直接投入长GPU。

### 311. 1.5rad/s直接续训预检失败，需渐进slew迁移

从v303 model_800直接以严格0.25m、过程0.28m和目标slew 1.5rad/s续训3次：801为`root_height=0.3094m/up=0.3056`，802/803迅速跌至0.1410/0.1302m、`up=-0.627/-0.664`，`ready=0`，stable_pose由0.00984降至0.00132。说明v303策略已适应4rad/s目标变化率，直接改成1.5rad/s会破坏动作时序；下一步应在训练内从4逐步退火到中间速率，再验证稳定，不直接把1.5用于长训或部署。

### 312. 明确slew退火而非跳档

用户指出当前没有理由直接跳到1.5rad/s。已将1.5定义为退火终点，不作为下一步起始档：恢复环境新增`recovery_slew_start_rad_s`、`recovery_slew_end_rad_s`和`recovery_slew_anneal_steps`，按`_global_step_counter`在线线性退火；默认退火关闭以保持旧契约。实验计划为4.0→3.5→3.0→2.5→2.0→1.5，每一级先检查stable_pose、ready、stable_hold、限位和冲击，再决定继续。评估固定使用退火终点并在contract记录完整退火参数，避免部署依赖训练时序。

### 313. slew退火公式单测锁定

将恢复目标slew的线性退火抽为`linear_slew_schedule`，验证4.0→2.75→1.5及超出终点的有界行为，避免训练内实际速率与contract记录不一致。恢复reward/安全项测试现为12项全通过；尚未启动退火长训。


### 316. slew对照与计数语义纠正

v315产生model_853；6400退火计数是聚合环境步，128env下仅50控制步（0.4秒），非50 PPO更新。此前“渐进迁移有效”未被证实。修复evaluate_rmuc_recovery显式slew override被退火终点覆盖，以及报告实际rate不一致；开始同checkpoint四方向×4实例的4/3.5固定策略配对评估，证据在artifacts/continuation_v316。八点locomotion是八个命令点，不是八项指标；仍未完成的+0.8/物理分支冲突不能因恢复任务而遗漏。

v316配对评估完成：同model_853，4和3.5均0/16通过、最长ready为0；unsafe分别8/16与11/16。该单seed固定策略结果不支持继续降速。暂保留4作为对照，下一步定位ready各子条件及支撑接触；不以本结果宣称最佳slew或训练因果。comparison.json保存同checkpoint hash与实际执行速率。


### 317. 固定4rad/s的ready条件拆解

新增evaluate_rmuc_recovery逐条件及联合条件累计，只统计reset前首完整回合；每步断言诊断合取等于环境ready。model_853、30度、严格0.25m、4rad/s、四方向各4实例实际执行完成，0/16，结果在artifacts/continuation_v317。前倒upright约93.94%、双轮99.12%、低角速94.94%，但height仅1.90%、低平面速度3.08%，且支撑+直立+高度同时成立为0；后倒upright1.00%、height2.30%、低平面速度0.30%，双轮95.37%。左右侧倒在终止前upright和双轮支撑均0，高度单项却分别100%/79.54%。因此前倒应重点诊断撑高与制动，侧倒应重点诊断单轮支撑到双轮支撑及限位；不把总体均值或高度峰值归因为同一原因。不同方向回合时长不同，比例不可解释为同等时长成功率。没有新训练或认证；八点转向和共享物理分支冲突仍待处理。


### 318. 恢复安全惩罚覆盖缺口修复及原生短验证

原reward仅主动四腿关节限位、两主轮接触；终止与诊断却涵盖被动膝关节和导向轮/机体，存在覆盖缺口。新增默认关闭recovery_full_safety_scope，启用后使用与硬限位终止同一_dof_ids及全部contact sensor bodies。评估读取保存的scope/权重，旧缺省安全项恢复为0，导出记录training_safety_reward，防止历史reward被新默认值改写。

v318从model_853续训3次（854–856），固定4rad/s、退火关闭、严格0.25m，全覆盖显式启用。model_856有限，实际params核验通过；ready三次仍0，不能据短验证宣称学会或无学习潜力。12项现有reward测试及编译通过。冲击项按body均值且权重0.002，四位小数日志仍约0，后续需用精确原始量衡量信号强度；覆盖修复不是冲击已解决。证据artifacts/continuation_v318/verification.json。


### 319. 完整安全覆盖固定4rad/s的100更新续训

v318 model_856的接线短验证之后，从该checkpoint保留actor/critic/optimizer接续新增100更新（857–956），运行recovery_full_scope_v319。保持30度、0.25m严格高度、0.28m过程目标、4rad/s固定slew和退火关闭；实际params全部recovery字段与源一致，worker2091170/trainer2091206真实存活并已输出860更新。结束自动对model_956进行四方向各4实例完整恢复评估，不自动晋升；artifacts/continuation_v319保存command、源hash、worker状态、训练日志和runtime_verified。

精确TensorBoard读数补充：源v318冲击加权项约-3.3e-6至-4.2e-6，stable_pose约9.1e-6至1.4e-4，而upright约1.85至2.18；四位小数显示0不能当作没有接触/没有信号。该差异提示塑形信号可能太弱，但均值不足以独立证明归因。本轮先固定配置观察学习，避免每3更新变更控制接口。八点locomotion待办（v278中+0.8及物理分支冲突）继续保留，不用恢复评估代替。


### 320. 八命令梯度增加严格正向余量的离线可行性

恢复v319继续运行时，CPU读取v296已核验command_gradients.pt，对八组归一化梯度求最接近+0.8方向的约束投影，约束从>=0改为>=margin（1e-4、1e-3、1e-2）。三档均收敛并独立检查可行；margin0.01时单位方向对八组最小内积约0.03871，对+0.8约0.20322。此前多项恰在零边界，新方向提供正向余量，值得进行同rollout有限步长复验。仅单批八命令一阶证据，不含other、未来状态或Adam，未修改actor/optimizer，不宣称梯度冲突或八点gate已解决。脚本、方向张量和JSON在artifacts/continuation_v320。


### 321. 八命令有限步长surrogate出现全非退化候选

v320/finite_step.py在相同v296 rollout、TF32、128 forward batch下验证三档正向余量及0/1e-5/1e-4/1e-3步长。全部step0动作逐值复现，finally恢复源physical参数并断言一致；没有checkpoint写入。margin0.01、step0.001时所有已表示组surrogate非负，最小改善约2.30488e-5，+0.8改善0.000908221；其他非零候选至少一个组负变化。finite_step.json保存各命令结果。

这首次为后续受约束实际更新提供了有限步长候选，但仍仅同一批rollout，不能声称八点行为gate通过或未来状态保持。下一步应将方向构造与实际surrogate回退结合，并经过独立八点评估；不把该方向送入Adam后默认仍有同样保证。恢复v319继续原预算。


### 322. 约束方向候选完整八点评估仍7/8，待当前脚本基线配对

v322从v278/model6684构造margin0.01、step0.001的独立候选，确认v296快照actor逐值等于源，仅physical network六张量变化；移除不再匹配的Adam状态，artifact仅用于评估，原基线未覆盖。八点各32回合、全部生存，仍7/8；+0.8 yaw0.0542896、stable0.111204未过，其余七点通过。此前同rollout surrogate全改善没有转化为gate通过。旧基线yaw0.0529605/stable0.207149，但evaluate_rmuc_grid.py hash发生变化，不能据跨脚本记录作严格因果结论。v323已以当前脚本重新评估原v278，命令、初态、动力、seed保持；不是重训已掌握技能。v322不晋升。

v319完成100更新到model956并自动评估，0/16恢复通过、8/16不安全终止，没有稳定保持认证。保留全部产物，不续跑相同配置长训；后续仍需针对前后低高度行驶和侧向支撑转换设计学习信号。


### 323. 当前脚本的v278配对基线与恢复终态条件汇总

v323原v278八点复测已实查eval2119229存活，日志steps1000/min_completed0，不能视为已完成或重启。已准备compare_candidate.py，只有源码hash、初态、材质、执行契约、seed、布局与门槛一致且八点各至少32回合才输出paired_comparison.json。

v319完整恢复条件汇总保存condition_summary.json：前倒upright约91.59%、低平面速度3.75%，支撑+直立+高度同时成立为0；后倒upright0；左右侧倒双轮支撑0，各4实例全部unsafe。安全项覆盖修复没有带来稳定通过，不能推断增加安全项已解决学习瓶颈。后续应保留4rad/s并分离恢复过程奖励与稳定保持入口，避免继续仅调高度/限速或重复同配置长训。


### 324. stable_pose乘积门控存在过程学习盲区

CPU反事实直接调用当前stable_pose_shaping：严格height=.25时，height<.21或both_wheels=False使该奖励及对速度输入的导数恒为0；高.25、up=.99、angular=.1时，planar速度.1/.3/.6对应奖励0.28114/0.00803/4.93e-8。此为奖励函数输入导数，不是PPO梯度，也非因果唯一证明。结合v319前倒末态约.17m、侧倒双轮支撑从未出现，可知该稳定项在这些状态不能直接提供减速引导。不能靠继续放大同一乘积项解决零信号。

后续设计应分离低姿态下的减速过程信号、单轮到双轮转换信号与严格ready连续保持；侧倒静止仍不得获益，翻正过程不应被全面抑制。保留当前checkpoint和4rad/s控制，不在本次审计中改变训练reward。artifacts/continuation_v324保存可运行审计脚本和24个反事实。v323仍真实运行，已2000步、每点至少16完成回合，等待32回合覆盖。


### 325. 配对拒绝转向候选；低高度制动过程代价接通

v323基线与v322候选源码hash、初态、材质、执行契约、seed全部匹配，各点32回合。候选仍7/8，+0.8 yaw0.0529605→0.0542896、stable0.207149→0.111204，明确不晋升，保留v278。同批surrogate全正不能代替闭环验收，后续需重视稳态采样与目标一致性，而非直接重复单批投影。paired_comparison.json保存。

针对v324盲区，新增默认0的recovery_braking_penalty：至少一轮接触、up>.5时按speed平方/(1+speed平方)给有界负代价，不依赖高度；侧倒/腾空/零速不会获得正收益，不抑制全局翻转角速。评估/导出同步读取权重。13项测试和编译通过。v325从v319/model956、4rad/s、严格.25m、weight=.5完成3更新957–959，制动项-0.1677/-0.1838/-0.1934可触达，ready仍0；model959有限。此仅新信号接线，不证明减速或起稳改善。证据artifacts/continuation_v325。


### 326. 固定4rad/s、独立制动代价的100更新学习窗口

从v325/model959保留actor/critic/optimizer接续100更新960–1059，braking_penalty=.5，30度、严格高度.25、过程高度.28、4rad/s、full_safety_scope=True、退火关闭；源与新run全部recovery参数逐项相同。worker2125445/trainer2125488实查存活、首动作执行，命令和hash/runtime_verified在artifacts/continuation_v326。结束自动四方向各4实例恢复评估，不自动升角度/DR、不覆盖locomotion v278。对制动的判断以低平面速度、ready连续保持和各方向unsafe变化为准，不能以代价变小或reward上升代替行为。八点候选v322已由v323同脚本配对拒绝，下一转向实验需要新的稳态证据。


### 327. 八点梯度快照与稳态评估的时间范围不匹配

实读v296快照observations形状[128,128,25]，捕获在第一次PPO更新前，只有128控制步物理历史；v323八点评估steady_warmup_control_steps=1200，另含300步窗口。随机episode_length不能变成真实经历的物理历史。因此首批surrogate方向不能代表稳态，v322失败不能简单解释为投影求解错误或仅学习率问题。sampling_audit.json保存，仍不是唯一因果证明。

rmuc_position_authority_training新增RMUC_CAPTURE_PPO_ROLLOUT_UPDATE（从1计数，默认1保持旧行为），可在后续rollout捕获；快照记录窗口长度、之前控制步数，并显式steady_samples_certified=False，因为异步reset可能混入初态。编译通过。未启动新转向训练，后续捕获必须核对实际回合年龄/命令连续性，不能只把采集轮数延后就宣称稳态。恢复v326继续原100更新。


### 328. PPO快照真实回合年龄与命令连续性原生验证

仅显式RMUC_CAPTURE_PPO_ROLLOUT启用时包装alg.act，在env.step前记录_curriculum_step_counter与观测还原命令的连续年龄；reset或命令变更清零，跨rollout保留连续记录。捕获时断言逐帧观测与storage逐值一致，避免reset后错位。v328实际从v278跑3次诊断更新，在第3次捕获，标签形状[128, 128]、物理年龄范围[0, 383]、命令年龄范围[0, 383]；>=1200步样本为0，正确拒绝称稳态。编译与原生运行通过，verification.json保存；诊断checkpoint不晋升。恢复v326继续运行。


### 329–330. 固定策略后期梯度与恢复制动实测更新

v329 capture-only收集12个rollout，最终actor、critic、optimizer均与v278逐值一致。第12批真实物理年龄0–1498，筛选episode age和command age均>=1200得到3352样本；八命令各234/368/177/190/128/94/807/1049，不能视作均衡独立样本。v330同TF32和128batch均值复现误差0；+0.8对-1.2余弦+0.8963（启动批-0.9404），对+1.0/-1.0/+1.2为-0.7156/-0.5833/-0.7143。冲突依赖状态与优势分布，旧首批结论不能泛化；未改actor或晋升。证据v329/verification、v330/gradient_audit。

v326完成100更新及四方向16实例评估，仍0/16、8 unsafe、最长ready全0，但前倒低平面速度达标从v319的3.75%提高到93.59%，双轮支撑98.20%；后倒低速仍0.30%，两侧双轮支撑仍0。前倒有单项改善，不称站稳或完整恢复；comparison.json保存。候选应保留，后续针对前倒高度、后倒和侧向支撑继续，而不是丢弃已出现的制动能力。


### 331–332. 后期候选仍有早期代价；恢复保留制动后尝试撑高

v331使用v330后期梯度生成三档正向余量方向，并在同批全部命令、late与early子组做有限步长复验；非零候选未通过全组保护。margin.01/step.001时+0.8 late改善8.84e-5，early下降1.616e-4；不生成checkpoint，不浪费八点GPU。说明只用后期梯度也不能默认保留起步/切换，下一方向需联合约束相应状态子组。

v332从v326/model1059接续新增100更新1060–1159，唯一恢复配置变化为supported_height权重2→4，制动.5、slew4、严格高度.25、过程高度.28不变；首动作和进程实际验证，params差异核验通过。目标是已有前倒制动改善基础上探索撑高，不追求最高高度；必须同时保持制动、ready/hold并检查四方向unsafe。结束自动评估，不晋升。artifacts/continuation_v332保存命令、源hash、配置差异和日志。


### 333. 早期/后期联合约束暴露other命令保护缺口

从v329固定v278 rollout分别计算八命令early/late共16组梯度，TF32/128batch均值复现误差0。联合正向余量投影后有限步长检查：margin.001或.01、step.001的八命令早期/后期组均非负，但other总体约-5.4274e-5、late_other约-1.6031e-4、early_other约-3.9069e-5，仍拒绝；没有checkpoint写入，源参数恢复验证通过。下一候选必须把other状态子组也纳入约束，不能只保证固定八命令。other本身还是聚合，不等于全连续域保证。artifacts/continuation_v333保存梯度/投影/有限步长证据。

恢复v332真实运行到约1084更新，未改变当前学习配置，待100更新结束自动评估。


### 334–335. 全分组约束候选通过有限步长检查，启动八点行为验收

v334在八命令early/late外追加other early/late两组，18组归一化梯度联合投影。小步1e-5至1e-3仍有组负变化；扩展.002/.003/.005后各档余量均通过同批全部/early/late分组surrogate检查，零步均值复现0、源参数恢复。选margin.001、step.002最小扩展步长，最小组改善4.1966e-6，+0.8整体改善6.0591e-5。other仍聚合、单批可能过拟合，不称全域行为保证。

v335独立候选只修改physical network六张量，源v278与v329快照actor逐值相同，删除失配Adam状态，仅评估用途。按同八命令各至少32回合/seed67/25%中间动力开始完整行为评估，原v278保留。证据artifacts/continuation_v334与v335；结果未出，不晋升。恢复v332保持原预算，最新约1120更新。


### 336. v335全约束候选最终拒绝，v332撑高权重对照未改善恢复

v335八点评估完成，仍7/8；+0.8 yaw=0.0530826、stable=0.223035，未越过yaw .05且stable .45门槛，其余7点通过，候选不晋升，v278继续作为基线。

v332 supported_height权重2→4的100更新完成，恢复仍0/16、unsafe8、最长ready0。相对v326前倒低速保持改善（约93.6%），但ready仍0且高度约.175m；后倒upright、侧倒双轮支撑仍未解决，权重加倍没有形成完整恢复。paired_recovery_summary.json保存。当前不继续相同奖励长训，下一步需专门重构后倒/侧倒动作阶段和接触转换，不能再把高度权重当通用解。

### 337. 四方向standing reward bank接线

v332证实全局supported_height权重翻倍不能解决后倒和侧倒。恢复配置新增front/back/left/right四套`recovery_case_standing_weights`，由已有倒地case索引选择；默认四套与旧全局权重完全相同，因此25D/6D、观测、动作、PD和旧checkpoint行为不变。评估读取与导出contract同步记录case bank，13项reward测试及py_compile通过。后续可只提高后倒翻正或侧倒支撑转换的对应bank，不再牵动已改善的前倒制动；新bank实验仍须四方向完整评估。


### 337. 四方向reward bank索引修复并通过原生smoke

首次bank smoke发现多环境权重索引产生[env,env]形状，已修复为按case与env逐项取权重。修复后recovery_v3/30度/4rad/s/progress/refall配置原生4环境初始化通过，实际初始倾角30度、final_ready全false（仅初始化/合成动作），腿PD读回60/2、轮限幅3Nm、25D progress观测保持；13项测试、py_compile和`artifacts/continuation_v337_smoke.json`核验通过。该smoke仅验证接线和reset，不是自复位行为认证，case bank默认值与旧全局权重一致，未启动新训练。

### 338–339. 侧倒双轮支撑事件奖励接线与短预检

新增`double_contact_acquired`一次性事件：仅从上一控制步非双轮到当前双轮时触发，静态侧倒不产生持续收益；默认bank权重0，旧模型兼容。修复多环境索引后，13项测试、编译和v338原生smoke通过。v339从v332/model1159用左右侧倒bank权重2做3次短预检，30度课程中`Recovery/double_contact_event_fraction=0`、ready全0；这说明该短课程没有触发事件，不能据此判断侧倒长课程价值，也未启动长训。后续若启用必须用侧倒初态专门课程和完整四方向验收。

### 340. 气弹簧口径勘误：10.775为每腿等效力矩

用户修正先前参数：气弹簧名义不是10.775N径向力，而是每条腿`10.775 N·m`等效辅助力矩。当前RMUC硬件目标配置已改为`equivalent_assist_joint_torque`，径向力默认关闭；recovery cfg未来默认读取等效力矩10.775Nm。历史10.775N径向力实验、探针和结果不改写，只保留为旧假设证据，不能再作为最终执行器契约。径向力模型仍可作为独立辨识/DR分支，但必须有新的作用点、力行程和Jacobian依据。硬件审计脚本已改为检查10.775Nm等效力矩，审计通过；未启动新训练，旧checkpoint行为不重新解释。

### 341. 最终赛场DR与实机gap是同一条迁移验收主线

用户再次确认：3Nm执行器限制、damping不确定性以及仿真到实机的动力/传感/时延差异，都必须通过版本化的domain randomization覆盖后再验收。这里的3Nm是最终轮端输出硬限幅，不是可以由DR放宽的随机上限；最终gate必须在每轮持续3Nm限制下运行。damping当前以硬件目标配置的工程先验`0.005–0.04 N·m·s/rad`、中心`0.0225`作为迁移范围，旧`0.005–0.08`仅保留为历史敏感性诊断，不能把任一区间称为实测标定。

最高联合DR必须覆盖并记录：质量/惯量、接触摩擦、轮端damping、轮端 torque gain、气弹簧`10.775 N·m`等效力矩误差、电机响应时间常数、独立observation delay、communication/action delay、传感器噪声与bias，以及赛场40–400N短时装甲板冲击和云台惯量/反作用力矩。当前配置中已经有部分质量、摩擦、damping、torque gain、sensor noise/delay和motor-response入口；装甲板冲击、云台动态反力矩等仍需逐项实现并做作用点/持续时间审计，不能仅凭配置字段存在宣称最高DR已覆盖。

最终验收必须使用同一个checkpoint、同一25D/6D与执行器contract，在最高联合DR下完成多seed、连续命令、起停/反向、动态切换、全速度压弯、自转、recovery四方向和策略交接；固定nominal、单项DR、单seed、曲线变好或接口通过都不能替代该证书。DR可以覆盖未辨识实机参数的合理不确定性，但不等于实机控制器、周期、限幅和真实动力已经测量验证；发布前仍需controller/hardware replay与实机验收。

### 342. 当前接续目标：先恢复课程，后最终动力与最高DR

用户确认继续推进。当前唯一训练权威为`/home/kellen/桌面/11 (1)/11/wheeled-legged_RL`；`/home/kellen/wheeled-legged_RL`只读。当前没有该权威仓库的运行中训练进程。v278仍是locomotion保留基线，v335不晋升；v332/model1159和v339短预检属于旧径向力/新接线过渡证据，不能直接当最终等效力矩checkpoint。

**短期 goal（恢复与动力迁移，完成后才进入部署FSM）：**

1. 以v3 recovery结构为基础，建立训练内渐进初态课程`30°→45°→60°→75°→90°`，四方向都覆盖；每一级必须保存初态分布、课程门槛、ready/stable_hold、unsafe和接触转换统计。
2. 将腿目标slew作为训练内物理约束退火，先验证4 rad/s边界，再逐步降低到已定义的终端速率；slew必须在Isaac、export contract和MuJoCo使用同一physics-dt与measured-position reset语义。
3. 在最终等效气弹簧`10.775 N·m/腿`、轮端硬限幅3 Nm、damping工程范围和电机响应模型下，从明确记录的旧checkpoint初始化新适应实验；旧checkpoint只作初始化来源，不能继承旧动力的通过结论。
4. 每个课程阶段结束后执行四方向独立评估；前倒、后倒、左右侧倒分别检查翻正、撑高、单轮到双轮支撑、ready连续保持和再跌倒，不能用reward或阶段编号晋级。
5. 通过nominal最终动力后，再逐步打开质量、摩擦、damping、气弹簧误差、电机响应、sensor/communication delay等DR；每次只改变明确定义的变量，并保留四方向和旧能力retention。

短期 goal 的通过条件是：同一recovery checkpoint四方向均有完整ready/stable_hold证据，最终动力与指定DR下不出现未解释的unsafe，export/MuJoCo contract逐值一致。未达到前不实现或宣称recovery→locomotion生产交接。

**长期 goal（完整RMUC E2E交付）：**

在同一最终checkpoint体系中完成locomotion的3.2 m/s直行、全速度双向压弯、`±3→±10 rad/s`自转、起停/反向/连续动态命令，以及赛场40–400 N冲击和云台惯量/反作用力矩最高联合DR。随后完成locomotion/recovery/obstacle独立ONNX contract、状态机滞回与fallback、MuJoCo闭环、controller接入、多seed和实机replay；最终发布证据必须同时包含checkpoint/ONNX hash、25D/6D与slot7语义、PD/3 Nm/10.775 N·m参数、DR范围、失败项和实机验证状态。


### 343. v342新动力课程接线、短PPO与真实四方向评估

新增可选按环境30/45/60/75/90度初态课程；只在完整物理回合timeout且终态连续ready/hold、无unsafe时计成功，手动reset、随机初始回合年龄、曾经成功后再倒不计。默认独立评估授权上限为stage0，未取得四方向独立评估与retention不允许向更高阶段开放。课程采样状态不是技能证书，目前尚未接入checkpoint课程状态持久化，正式跨阶段断点续训前必须补齐。slew现有退火计数是全局transition数（每步增加num_envs），不是PPO更新数或单环境物理步。

原生课程smoke曾出现多环境高度索引广播错误和episode日志初始化顺序错误，已修复；之后四环境128控制步、512子步、反向目标及中途reset探针通过，课程仍stage0。相关测试38项通过（含连续成功与授权上限测试）；这不证明课程晋级行为或恢复成功。

v342从旧径向力v332/model1159恢复actor/critic，重置optimizer，128env/128rollout、seed268完成3次预检更新1160–1162。显式采用每侧-10.775Nm及机身反作用、径向力0、轮3Nm、PD60/2；严格高度.28、过程高度.36、progress开启、slew4→3.5在49152个transition内退火，课程封顶30度。多项变化仅用于集成预检，不解释为单变量收益。source hash、命令和状态保存artifacts/continuation_v342，locomotion v278未改动。

同候选sha256 c6f0d0840b0dbab042823ea71675bbf1335e68898ce19c74f41bb687d1e1892e独立四方向各4实例：30度0/16通过、8unsafe；90度0/16通过、16unsafe；最长ready全0。30度前倒平均终高约.1744m、后倒.1851m，两侧全部unsafe。评估实际回读等效力矩10.775Nm、径向力0、固定终端slew3.5。该结果不能证明3.5优于4，也不能说明气弹簧迁移收益；不晋升，不按此配置盲目长训。下一步需要检查侧倒支撑事件与full_safety_scope的接触索引、后倒过程可达性、关节限位原因；最终最高联合DR和部署目标保持未完成。


### 344. full_safety_scope误把双轮事件扩展为全身接触，已修正

实际源码确认：recovery全身冲击惩罚启用后，contact_forces被替换为全部刚体，后续both_contact错误要求所有刚体接触。现独立wheel_forces始终只取左右轮，全身范围仅用于impact penalty；未修改ready、终止阈值、PD、动力或观测。新增test_rmuc_recovery_contact_scope直接执行真实_get_rewards接触代码块，对非连续轮索引、机身600N冲击、双轮新接触/持续接触/单轮/无轮接触验证；full scope开关均通过，连同reward测试17项通过。

90度四方向原生零action/128步探针在每侧10.775Nm、轮3Nm、slew3.5、full safety scope开启条件下运行完成，termination均0、ready均false、double_contact事件0。只证明接线/初始化，不是起身成功，也未证明侧倒事件自然可达。保存artifacts/continuation_v343。v342的全部unsafe记录为joint_limit，既含2.38e-7rad浮点级超界也含数e-5rad超界；不因数值很小直接放宽门槛。v342事件权重为0，此修复不能被解释为v342失败的唯一原因；v339使用非零侧倒事件权重的历史零事件结论需保留该实现缺陷限制。

当前无长训练；下一步仍须取得倒地过程可触达信号与关节限位轨迹，再决定侧倒/后倒学习变量，不能把函数修复当90度课程通过。最高DR、多seed、sim2sim与部署均未认证。


### 345. v344支撑转换轨迹与单变量学习窗口

新增evaluate_rmuc_recovery --trace-stride，保存自动reset前的关节位置/限位余量、实际PD目标、腿力矩、两轮载荷、高度和姿态轨迹。v342确定性30度单实例四方向轨迹显示：前后倒低姿态约.174/.185m停滞；侧倒先单轮支撑并升到约.52m，然后丢失双轮支撑、翻至160度以上，最后才关节限位终止。故不能把限制放宽或增加高度权重当通用解。证据artifacts/continuation_v344/source_trace30.json。

新增默认0的recovery_wheel_gap_progress_weight：仅左右侧case、连续前后帧至少一轮支撑、机体up>0时，对左右轮高度差缩小给有符号进展，静止/初帧/离地转换/倒置/前后case不获该收益。支持条件改变时会门控，故闭合循环的零和测试仅限持续支持；仍须实际行为审计，不能宣称无所有reward hacking风险。真实接触范围测试与reward逻辑共18项通过，export与eval回读新权重。

v344从v342/model1162、sha c6f0d0840b0dbab042823ea71675bbf1335e68898ce19c74f41bb687d1e1892e保留actor/critic/optimizer，启动100更新1163–1262预算。唯一新增学习项是侧倒gap进展权重0→2；slew保持源退火终点3.5（不重播退火），严格.28、每腿10.775Nm等效力矩、轮3Nm、PD60/2、DR0、30度课程上限不变。参数runtime_verified已核实，实际更新日志出现非零gap项，ready仍0；结束自动30与90度四方向各4实例评估。仅故障/非有限值提前停，无平台期STOP，训练状态须查status.json和PID。首次启动命令误替换checkpoint路径，未发生更新，已保留first_attempt日志并修正路径后启动；不能仅凭Isaac退出码0认定成功。

同时完成源v342 recovery_v2 ONNX/JIT导出及新气弹簧contract的真实MuJoCo四方向30度诊断：四例均到时，longest_ready全0，非行为通过。该source_export包含旧模型的真实hash及新物理params，未篡改原checkpoint/导出。后续继续完整四方向和最高联合DR目标，不把此次100更新当目标完成。


### 346. 用户纠正优先级：物理复核与近站姿保持先于继续调高度奖励

四方向reward bank共享actor，不是参数隔离；仅保持前倒权重不能保证前倒制动不退化。后续每个候选必须四方向retention。历史配置实查：v326全局supported_height2、v332全局4且均无case bank；v339/v342存在显式bank。新配置bank改为空的稀疏覆盖，运行时从全局继承；评估恢复缺bank则继承存档全局，显式bank原值保留，不重写旧产物。对应真实初始化代码测试验证全局4与显式2覆盖；historical_bank_audit.json记录源hash。

接触上升沿不等于每回合一次。新增debounced_once模式：正up且双主轮连续接触.04s才记账，每env每回合一次，丢接触不清paid，reset清零；旧checkpoint缺mode时按legacy_edge解释并明确限制。full safety scope只扩展冲击惩罚，不扩展双轮接触条件。接触抖动、反复离地重接、局部reset及bank继承测试共20项通过。不能将事件函数通过写成恢复行为成功。

新增0/5/15度初态几何候选；使用原MJCF逐碰撞体测距确定3mm离地余量、原闭链、原关节角，不改USD/MJCF结构。0度root z=.402993m；5/15度高度按四方向各自计算并保存gentle_initial_geometry.json。新课程序列支持0→5→15→30→45→60→75→90，保留显式旧序列兼容，阶段上限仍由独立评估控制。0.25m可以作为明确声明的低站姿目标，但本轮没有改变严格.28m门槛；保持双轮支撑、姿态、低速及连续hold。

0度零action四实例1000控制步：无终止，最长ready约.512/.488/.600/.552秒，末态ready均false。读取4000个Isaac实际composer子步力矩，world误差<=2.29e-5Nm、净反作用误差<=2.77e-5Nm；结合v341的64组真实MuJoCo广义力/虚功对照，证明当前恒定-10.775Nm/active_link与机身反作用实现，不能证明真实气弹簧力矩-行程已辨识。当前恒定模型是用户明确接受的近似，DR覆盖合理误差仍待实现和最高强度验收。

v342同新动力策略从0度初态四实例均未ready，最终高度约.087m并unsafe。0度四个标签仅为相同站姿复制，不冒称四种倒地方向。零action出现过连续ready、旧策略从近站姿仍失败，支持先获得稳站轨迹、再扩展轻微倾斜的路线；不是全姿态机械可达证明。低姿态完整接触类型与关节余量需继续核查，evaluator trace已增加主轮高度及机体/导向轮等接触名单。本轮新默认不热改已加载的v344实验，其自动评估读取存档bank与旧event模式。


v344现已完成100更新及两组四方向评估：30度仍0/16，unsafe从源8减到6；90度仍0/16、unsafe16，不晋升。comparison.json只作描述性对照，不当严格同源码因果配对。当前训练已结束，后续优先0度稳站和轻微倾斜课程，保持已确认动力/固定slew，不再用短预检或侧倒事件未触发决定跳90度训练。


### 347. v346近站姿保持学习启动：只改变初态分布，保留终端控制契约

先以v344/model1262做0度固定初态16实例独立基线：0/16、unsafe16、最长ready约.008秒。低高度轨迹主要出现robot、active_link及其他连杆接触，不能把低矮停留当双轮支撑站稳。0度的四case标签仅为同姿态副本，不是四方向倒地成功证据。

v346从该同checkpoint保留actor/critic/optimizer，300更新1263–1562预算，128env/128rollout、seed268；唯一学习条件变化是30→0度初态。严格.28m、过程.36m、slew3.5固定、PD60/2/40、轮3Nm、每腿-10.775Nm等效辅助、DR0、reward及各case有效权重与源逐项一致。事件模式改debounced_once仅为记账修复，源及新实验的double_contact权重全0，无额外reward变化。参数与checkpoint hash核验见artifacts/continuation_v346/runtime_verified.json。

首次启动因configclass不能向空dict新增嵌套bank字段而拒绝，未发生PPO更新，日志完整保留。确认所有源bank等于全局权重后移除冗余bank注入，使用等价继承，已实际启动首更新。不擅自跳级，上限保持0；结束自动0/5/15/30/90各16实例独立评估，30/90用于旧能力诊断与retention，不表示训练已扩展到倒地阶段。预算内不因短期平台STOP，故障/非有限值需保存证据；v278不修改。最高DR、实机差异覆盖和部署目标仍未完成。


### 348. v346运行期间只读最高DR覆盖审计

真实trainer存活并继续原300更新，未热改物理/奖励。实际执行env._progressive_domain_randomization_strength源码AST：recovery继承curriculum_stage=0，即使enabled=True、override=1仍返回0；stage1对照返回1。当前v346明确DR0不受影响，但后续recovery不能仅改override冒称最高DR。还需区分恢复DR入口与行走command课程门控，单独修复并原生读回。

v346参数wheel D中心.0225、max fraction .1，假设强度1也仅对应[.02025,.02475]，不是目标先验[.005,.04]。等效气弹簧当前是固定cfg标量，reset随机化仅接旧径向力；名义上限10.775还会拒绝±20%先验上端12.93。最高DR阶段必须单独实现并验证左右独立/共同力矩误差，不把旧径向DR复用说明当执行证据。装甲板脉冲/云台动态反作用也仍缺失。

修正docs/rmuc_competition_disturbance_contract.md残留的10.775N径向口径，明确当前每侧-10.775Nm等效力矩与未完成的误差DR。证据artifacts/continuation_v347/coverage_audit.json；此次只读审计和文档修正，不改变运行中的v346，也未完成最高DR验收。


### 349. recovery显式DR门控修复，默认DR0保持

恢复类单独实现_progressive_domain_randomization_strength，不再依赖locomotion curriculum_stage=1。enabled且显式override在[0,1]时生效，-1解释关闭；play仍遵守apply_in_play，非法强度拒绝。没有用行走课程编号自动提升恢复DR，运行中的v346已加载旧代码并维持DR0，后续自动评估仍默认DR0。

probe新增独立dr-strength和target-damping-dr参数，后者从硬件target读中心.0225及fraction覆盖[.005,.04]。真实4env/128步DR1读回强度全1、damping约.01306–.02476、质量scale约.95755–1.00025、独立obs delay [2,1,1,2]与communication delay [1,3,1,3]，termination均0。DR0对照读回强度0、D=.0225、两种delay全0。实际配置强度及部分样本正确，不表示端点/全部分布已覆盖，不能称最高DR技能通过。

真实门控源码测试及reward/contact回归29项通过，artifacts/continuation_v348保存两原生探针和verification。等效气弹簧力矩DR、装甲板冲击、云台反作用仍未实现；motor_response未在此次probe开启。本次不升级训练DR，不改变最终3Nm上限。


### 350. v349等效气弹簧共同/独立偏差DR接入及跨引擎验证

新增默认关闭的gas_spring_torque_common_dr_fraction/independent_dr_fraction；reset按显式DR strength采样每env两腿力矩，零强度或零幅度不消耗RNG，名义无DR保留原标量施力。共同和独立相对误差采用加法，总半幅为两fraction之和，hardware profile把±20%工程先验明确拆成共同±10%和各腿独立±10%，范围8.62–12.93Nm，不是两个±20%相乘。名义校验仍<=10.775，实际随机采样可高于名义；底盘反作用为左右力矩总和反向，不再假设两腿相同。旧径向DR字段不再承载等效力矩误差。

独立4env、128控制步/512子步DR1 probe实际两腿不同幅度约8.839–12.204Nm，施力读回与采样目标最大误差2.67e-5Nm、净反作用误差3.24e-5Nm，termination均0；不认为4个样本覆盖了全部区间。标量及非对称MuJoCo共用consumer支持独立两腿幅度，64组随机姿态广义力/虚功对照误差<=2.14e-14、错误轴拒绝。gas spring/DR gate 15项及hardware profile18项测试通过。export记录训练误差分布，部署名义consumer不随机施力；MuJoCo联合DR runner仍需显式接入采样而非仅有力矩函数。

产物artifacts/continuation_v349/spring_dr1.json、asymmetric_parity.json。v346进程仍按加载时旧代码的DR0/300更新运行，不热改；这次接线没有给v346增加DR认证。赛场装甲板/云台动态扰动仍待实现，最高联合DR策略证据仍缺。


### 351. v346中间model1375：短暂连续ready出现，仍未持续站稳

主trainer存活、保持原300更新和配置；取已完整保存的model1375独立0度16实例评估，不覆盖模型、不提前停止。checkpoint有限、actor相对源更新L2约1.12437、源hash未变。该次16/16曾达到连续hold门槛、最长ready .824s、unsafe0，但终态通过0/16；最终平均高度.18255m、倾角11.277度，双轮支持比例.9844、低平面速度.8567、高度达标仅.1510。

这限定为近站姿初态下短暂保持及生存改善，不能当倒地翻正成功或持续站稳。源v344的0度基线为unsafe16/16、通过0/16，本次也非同源码配对因果证书；未改部署/动作映射的相关修复已有接口验证，行为判断保留上述范围。持续撑高仍是瓶颈，不降低.28门槛、不凭最高高度/中途hold晋级；继续原定预算后看最终独立五角度结果。证据artifacts/continuation_v350/{checkpoint_audit,standing0,summary}.json。


### 352. model1375低姿态由导向轮分担支撑，零action干预能抬高但失稳

保留同model1375的0度四实例完整控制边界轨迹，新增身体接触名单后可见：由约.39m降至.182m，低姿态除两主轮外guide_wheel_lower_right_03持续承担约55–90N；代表实例最后四个主动关节仍有明显限位余量（全部受监控有限关节最小余量约1.39rad），腿电机约4–14Nm，未到40Nm。不能把该低姿态直接归因于关节硬卡死或电机饱和；其他姿态的机械可达性不能据此外推。

独立诊断在2秒后强制6D normalized action为0，仍经原physics-step slew3.5和PD执行，没有修改USD、力矩限幅或成功门槛。四例低姿态最高升至.2656–.2671m，随后倒置约180度、末高.07m；无joint/nonfinite保护终止不等于安全或成功（recovery本就允许倒置初态）。最长ready约.8s均发生在干预前，不能归功于零action。轮action也清零，因此此探针只说明原低姿态可移动、零action不是稳定撑高方案，不单独分离腿控制和轮平衡因果。

新增--diagnostic-zero-action-after显式标记policy_behavior_evaluation=false并强制整体all_recovery_hold_passed=false；每行物理hold统计保留但不得称策略通过。artifacts/continuation_v351保存原轨迹、命令、干预与汇总。v346仍按原300更新运行，未据诊断热改奖励或策略。


### 353. recovery评估补充实际轮电机3Nm审计

评估器在每次physics动作前与控制边界读取applied_torque主轮通道，对非有限或>3Nm+1e-5立即拒绝；记录每个实例两个轮峰值、样本数及明确采样范围。该值是电机施力快照，排除闭链约束和被动气弹簧，不以配置effort_limit代替读回，也不冒称所有求解器内部接触冲击峰值。

同model1375、0度四实例完整回合探针通过3Nm断言，artifacts/continuation_v352保存实际峰值。仍是旧中间checkpoint接口/动力读回，不是当前最终策略证书。v346接近1497更新时训练ready约99.7%、established约93%，仅是rollout步比例；原300预算继续，最终0/5/15/30/90独立评估尚未完成，不提前晋级或重写旧0/16。


### 354. v346完成：0/5度近站姿保持通过，15度后倒仍失败

原预算300更新已完整完成到model1562，sha256 cc6067bc15dad29b1860350a2e83d43fbb3f33827146e88fa0a7c57be954e4db。同checkpoint五角度各16实例独立评估：0度16/16、unsafe0；5度16/16、unsafe0；15度12/16、unsafe3；30度4/16、unsafe5；90度0/16、unsafe0。0度全部实例连续ready约7.976s，5度最短连续ready约7.368s，终态保持成功；这是真实近站姿/轻倾固定条件行为证据，不是reward或曾经hold统计。90度不触发保护但未恢复，不能用unsafe0称成功。

观察轮端峰值分别约1.009/2.829/3/3/3Nm，所有报告同hash、严格.28m、PD60/2、轮3Nm、等效气弹簧10.775Nm、固定slew3.5。结论限定当前nominal固定初态与默认eval seed，不是多seed/随机初态/最高DR/MuJoCo策略证书。0度四case仅同姿态复制。保留model1562作为近站姿基线，不覆盖locomotion v278，不宣称90度自主恢复完成。

新增artifacts/continuation_v346/summarize.py，只接受worker complete及同hash无动作干预的全部评估，保存final_summary.json并分别标注能力边界。15度失败集中在back_down（0/4，unsafe3），另外三方向各4/4；下一步定向处理后倒，训练内保留0/5度rehearsal、阶段末独立retention，再考虑更大倾角。先学站稳路线已获得局部证据；完整最高DR和部署goal仍保持进行中。


### 355. v353后倒15度rehearsal启动与中断续接v354

增加显式(angle,case)初态cohorts，不增加观测或策略维度，固定配比不自动晋级。128env分配：15度back_down72；0度四case共16；5度四case共16；15度另三个方向各8。共享actor需依赖真实retention而非声称参数隔离。混合初态原生4env测试实测15/0/5/15度及case1/0/2/3，128步无终止；reward/contact/cohort逻辑21项通过。v353从v346/model1562保留actor/critic/optimizer，原300更新1563–1862，唯一学习变化是初态采样分布。动力、reward、slew3.5、严格.28和DR0保持，结束自动0/5/15/30/90评估。

后续实查原worker2925476/trainer2925534均不存在，工具句柄也消失；status仍training，日志停1602、最新完整checkpoint1600。未找到退出原因，不据此归因为OOM或策略失败；另一目录command_v6训练仍运行，未触碰。保留status_before_interrupt及全部产物，原status改interrupted。model1600含actor/critic/optimizer且有限，sha f0c45b74316e20def38b6971063cfe721b10a941cd0d05cb0fd61609e2e93abc。

v354从1600续接剩余262更新1601–1862，计入已保存38次后仍以300次持久化更新为目标；未保存更新/轨迹不冒称恢复，模拟episode重新reset，不是逐帧连续重放。worker独立session启动，实际trainer2953570已进入首动作。逐项比较recovery/gas_spring配置与两类actuator参数完全一致，初态配比回读一致；见artifacts/continuation_v354/runtime_verified.json。保持原预算，无平台期提前停。原近站姿基线和locomotion v278保留，完整恢复/最高DR仍未认证。


### 356. v346近站姿基线在MuJoCo 0/5度保持，v354继续原预算

评估器显式清空训练cohorts、关闭自动晋级，固定角度四方向等量replicas；从checkpoint参数恢复hold秒数、upright、线/角速度门槛和episode时长，并在报告记录实际门槛、params hash、eval seed及训练cohorts。源训练覆盖与独立验收覆盖不能混用。该改动只补齐恢复与证据，未改变运行中的v354。

v355独立导出v346/model1562（sha cc6067bc…），JIT和ONNX ReferenceEvaluator数值通过；以实际MuJoCo轨迹7992组obs重做ONNX Runtime/JIT对照，最大差2.54e-7，裁剪后输出与记录执行动作差0。保持25D/6D、recovery slot7 progress、PD60/2、轮3Nm、每腿10.775Nm恒定等效力矩、slew3.5，原模型与旧export均未覆盖。

MuJoCo0度四相同姿态实例均跑满、连续ready7.968s、终态established=true；5度四方向分别连续ready7.424/7.488/7.792/7.816s且终态保持，均到时。0度不是四方向倒地，5度每方向仅一例；当前只有固定nominal近站姿跨引擎局部证据，consumer仍certified=false，接触求解/PhysX速度限制、随机初态、多seed、最高DR和controller交接未认证。新增--diagnostic-initial-tilt显式记录覆盖角度，未修改导出contract的原训练初态。

证据artifacts/continuation_v355/{export,mujoco_standing0.json,mujoco_tilt5.json,runtime_parity.json}。当前v354已实际运行到约1625，继续剩余预算和阶段末五角度retention；不以源模型sim2sim结果晋升当前训练候选。


### 357. 独立retention汇总按角度/方向保护，不以总通过数掩盖遗忘

新增scripts/compare_rmuc_recovery.py，读取complete worker的0/5/15/30/90全部报告，要求每方向4replicas及同候选hash、匹配等效气弹簧/径向力/slew/关节容差。拒绝动作干预、诊断容差和终态成功标记与unsafe矛盾的报告。对reference每个4/4通过的角度-方向组建立保护；候选15度back_down成功也不能抵消任一保护组退化。eligible_for_nominal_review只表示进入复核，promoted/certified/maximum_dr_verified始终false，完整门槛/物理来源、多seed和随机鲁棒性另审。

6项测试覆盖“后倒改善但0度退化”、缺角度、混hash、契约不匹配、动作干预和伪通过。用真实v346对自身做读取/汇总自检：retention=true、back15_all=false，与已知12/16一致；此为工具自检，不是新增行为实验。产物artifacts/continuation_v356/checker_selftest.json。v354训练仍活跃至约1648，不修改配置或提前结束；完成后用v346对照新候选五角度报告。


### 358. v346稳站基线在已实现的联合DR1下失败，保留nominal与鲁棒性边界

独立recovery evaluator增加seed、diagnostic-dr-strength、diagnostic-hardware-profile入口；开启诊断时显式记录profile hash、overrides和首次reset的真实DR强度、D/3Nm限幅、等效力矩、质量/摩擦、独立obs/communication delay。默认不改变现有nominal评估。nominal retention汇总拒绝含DR或hardware诊断的报告，8项相关测试通过。

v357用v346/model1562、0度初态、seed357、每case1实例进行已实现联合DR1预检：0/4通过、unsafe2、最长ready<=.08s。真实强度全1，轮D约.01761–.03406、两腿力矩各自采样，轮限幅均3Nm，sensor delay[1,3,3,3]与communication delay[2,1,3,1]。多个DR同时变化，不能从此定位唯一原因；0度case标签为同姿态复制，不是四类倒地分布。这个结果证明nominal稳站通过不能外推随机动力/传感鲁棒性，不能据四个样本定成功率或放弃DR路线。

源进程启动后文字scope标签才被修正，因此原standing_dr1.json中nominal scope和参数来源说明文字过时；保留原报告不重写，verification.json明确以diagnostic overrides和初始读回为准。后续脚本已修正文案。装甲板脉冲和云台反作用不在此试验内，maximum_dr_verified仍false。主v354继续原DR0后倒适应，不因该预检热改课程或追加新训练。产物artifacts/continuation_v357。


### 359. 用户确认低强度渐进DR，先保留当前nominal适应与retention

DR暂定0→.05→.1→.2→.4→.6→.8→1，按行为验收逐级提高，困难时细分；最终最高联合DR不降级。当前v354实查worker2953543/trainer2953570存活并进入1753更新，维持原300次持久化更新预算、DR0和全部控制/奖励参数。上轮DR1仅诊断，不据此直接开始高强度训练。

长期目标文档补充渐进DR门槛，并修正当前初态路线从0/5/15度开始。源码检查确认电机时间常数随强度缩放，独立观测/通信延迟按强度控制命中概率，命中后保留1–3周期范围。硬件profile名义值变化需与DR强度区分，下一项诊断用v346保留基线在相同目标profile下对照DR0与DR.05；不是当前v354候选的通过证书。


### 360. 低强度DR.05近站姿基线：14/16终态通过，仍需适应

artifacts/continuation_v358对v346/model1562做相同硬件profile、seed358、0度初态各16实例独立评估。DR0为16/16、unsafe0、最短连续ready7.976s；DR.05为14/16、unsafe0。两组checkpoint/源参数/profile hash、成功门槛与slew一致，实际强度及3Nm限幅读回通过，实际轮电机采样峰值未越限，源checkpoint hash未变。完整报告、命令和核验见该目录verification.json。

低强度已经暴露终态保持缺口，支持从小幅扰动开始适应，不支持直接跳高DR。两组强度改变会改变随机数消费，不是逐实例相同物理扰动配对；失败与延迟等字段的共现不能作为单因素根因。0度是同姿态复制、单seed，不是四方向倒地认证。当前v354实查仍活跃至1778，保持原预算；须先完成该候选的nominal五角度retention，再以实际选定起点评估低DR。


### 361. 训练内slew退火原生physics-step验证，跨进程续训计数仍待补齐

scripts/probe_rmuc_recovery.py增加显式退火样本数、起止rate和实际目标增量审计。4env、128控制步、512physics substep的合成动作验证4→3.5rad/s、384个累计环境控制样本退火；每个substep目标增量均在当时rate*2ms内，中点3.75对应实际.0075000003rad，末段3.5对应.0070000002rad。64控制步处主动reset前后累计样本均256，不重启episode间退火；四env均未终止。同步实际施加气弹簧及机体反作用审计最大误差约2.77e-5Nm。证据artifacts/continuation_v359/{command,slew_anneal,verification}.json。

该课程单位是num_envs累计控制步，不是physics步或PPO更新；128env、128步/更新时每更新16384样本。起始第一个动作前已累计num_envs，所以首动作rate略低于start。这里只验证合成动作执行链路，不证明退火对策略有益。当前_global_step_counter从新进程0初始化，runner resume不恢复其值；后续开启真正退火训练前需明确续接状态，不能用恢复PPO迭代号冒称完整课程连续性。v354保持原固定3.5、anneal_steps=0，不受此限制影响，未热改训练配置。


### 362. recovery slew checkpoint保存累计样本，续训不再从0退火

新增scripts/rsl_rl/rmuc_recovery_checkpoint.py，在启用退火时通过runner.save的infos保存独立slew累计样本数和起止rate/总样本契约。训练入口在load之后、learn之前安装；recovery环境仅对slew计算增加恢复offset，不改其他环境全局课程时钟。固定slew不包裹save、不读额外checkpoint，所以当前v354不受影响。旧固定速率checkpoint可明确开始新退火；旧活动退火缺计数或新旧退火契约不同则拒绝，禁止按PPO迭代号猜测已消耗样本。

tests/test_rmuc_recovery_checkpoint.py五项通过：跨保存续接384→512样本、保存原infos与actor、旧固定速率源新课程、拒绝旧活动退火缺元数据/不同课程、固定速率零影响；相关源码编译通过。该证据为保存/加载接口测试，尚非实际PPO中断续训行为认证。前条512substep原生检查验证未恢复时的执行链路，二者不能拼成已训练退火策略的行为证书。


### 363. slew计数通过真实PPO跨进程checkpoint续接

artifacts/continuation_v360从保留v346/model1562单独做两个进程、各一次PPO更新的接线验证。4env×32控制步，退火总样本256，4→3.5rad/s；首进程model1563保存elapsed_samples=128，第二进程日志明确从128恢复，model1564保存256。实际checkpoint均含actor/critic/optimizer与新增infos元数据，actor有限，源checkpoint hash不变。命令、日志及verification.json独立保存，不覆盖任何基线/候选。

这些短程更新只证明真实runner保存/加载计数链路；不依据两次更新判断训练成效，不将该诊断checkpoint晋升。v354仍固定slew3.5、DR0，实查进入1825更新；本次不改变其学习配置、预算或自动五角度评估。


### 364. v354完成：15度四方向通过，但5度与30度retention失败，不晋升

v353已保存38次加v354续接262次，共300次持久化更新完成，model1862 sha256 db934ccba08edd84a02d700cb5e9dcb52d34f4b291bc6ae1d11edff8ed355be5。五角度各16实例：0度16/16 unsafe0；5度14/16 unsafe0；15度16/16 unsafe0；30度0/16 unsafe10；90度0/16 unsafe0。15度四方向各4/4，最短连续ready3.712s；基线后倒0/4得到局部改善。实际轮电机采样均<=3Nm。

retention_comparison.json明确拒绝晋升：5度back_down、left_wheel_side_down各一例终态连续hold不足；虽然曾ready6.76/5.12s、结束时瞬时ready=true，但末段established=false，仍判失败。原30度front_down4/4也丢失。奖励共享/actor共享的遗忘由行为证据确认，不能用15度总通过率抵消。artifacts/continuation_v354/{final_summary,retention_comparison}.json保存；v346与locomotion v278均保留，完整倒地恢复仍未完成。

### 365. v361以初态rehearsal配比修复遗忘，保持DR0和控制契约

从未晋升候选v354/model1862创建独立训练分支，保留actor/critic/optimizer；预算300更新1863–2162，不做平台期提前停止。唯一学习因素变化是初态采样：128env中0度四方向共16、5度共32、15度共48、30度front_down32。30度前倒是已通过后丢失的retention项，不是借机扩到其他30度方向或90度。DR0、PD60/2、腿40Nm、轮3Nm、气弹簧10.775Nm、固定slew3.5及全部奖励参数保持。

worker2998729/trainer2998750已实查首动作和1863更新执行；源hash不变，保存运行配置逐项比对recovery/gas_spring及所有actuator一致，cohort数量验证见artifacts/continuation_v361/runtime_verified.json。结束自动五角度评估，同时对照v346原有保护组和v354新增15度能力，任一退化不自动晋升。先修复retention再低DR；最高DR/八点locomotion/全速度与部署目标不变。


### 366. v354遗忘分解：5度末段连续性与30度翻倒不可混为撑高不足

复核既有逐控制步gate统计及stride25轨迹，未新增仿真或修改门槛。5度失败back_down的姿态/高度/线速/角速达标比例全1，仅双轮接触.991992；left_wheel_side_down姿态/高度全1，线速.994995、角速.990991、双轮.975976。两例曾连续ready6.76/5.12s但终态未达.5s连续hold；因此不是单纯低姿态撑高失败。stride25轨迹不足以区分真实离地与求解器/接触传感瞬态，不据此放松支撑验收或断言物理根因。

30度front_down四例upright达标比例均0，双轮接触仅约.009–.060，末态倾角约129–134度，明显翻倒；即使保护中含浮点量级关节越限，也不能据数值容差消除真实行为失败。保留严格判定，现有v361按既定初态配比修复，不追加奖励或动力变量。证据artifacts/continuation_v354/retention_failure_detail.json；v361进程实查活跃至1877更新。


### 367. retention对照拒绝成功门槛、seed或reset分布不一致

scripts/compare_rmuc_recovery.py在现有动力/控制对照外，新增recovery_criteria、evaluation_seed、evaluation_reset_mode一致性检查。两份报告都有元数据却不同则拒绝；历史报告缺字段时显式列出evaluation_provenance_gaps，不补写或冒称契约一致。仍仅descriptive nominal review，不晋升或认证。

13项测试通过，含降低高度、缩短hold、换seed、换reset分布和历史字段缺失。重新读取v346/v354现有报告写入retention_provenance_review.json，三组retention失败结论不变；旧v346报告确有部分元数据缺失，保持证据范围。未重跑已掌握技能、未改正在训练的v361；其阶段末新报告对v354可作完整字段对照，对v346保留显式历史来源缺口。


### 368. 用户排除“脸朝天”高难度起身，调整后续recovery范围

用户明确不用训练“脸朝天”那种高难度起身，已同步长期目标，作为此前笼统全倒地目标的最新限制。具体姿态边界待确认，不把后倒15度与完全翻覆混为一谈；确认前不开展新的大角度后倒/翻覆训练。当前v361已完成300更新到model2162，正在原定独立评估；0/5度目前各16/16 unsafe0，15/30/90结果未齐。既有90度评估仅诊断，不作为自动要求高难度训练的依据。


### 369. v361完成：0/5/15度均通过，30度前倒仅3/4，仍不全量晋升

300更新完成到model2162，sha256 f928e6014d307ddba076190c6d1ba0d6ed69f86cb384f5feb42e8b434c5d164d。五角度各16实例：0/5/15度均16/16 unsafe0；30度3/16 unsafe8，三个通过均为front_down；90度0/16 unsafe0。双基线对照：对v354全部保护组保留；对v346仍缺30度front_down一例（3/4而非4/4），因此不作全量晋升。0/5保持修复且15能力保留，是单seed固定初态下的局部进步；未完成随机初态、低DR及最高DR认证。

完整报告、final_summary.json与两份retention对照在artifacts/continuation_v361。用户排除“脸朝天”高难度起身优先；90度失败保留为诊断，不自动转化为训练任务，具体排除姿态仍待用户澄清。30度前倒的缺口不等于被排除的仰躺姿态，不据范围缩减抹去现有retention结果。


### 370. v362接续未通过的30度前倒终态保持，原课程不变

v361已终结且worker/trainer均不存在，完整验收缺口为30度front_down一例：末态height.3423、tilt3.24度、final_ready=true，但final_established=false；曾连续ready6.608s不能替代末段hold。用户排除脸朝天不影响这个前倒轻角度缺口；未开启大角度后倒或翻覆训练。

从model2162独立续接v362，actor/critic/optimizer保留，预算300更新2163–2462，所有recovery/gas_spring/actuator/DR参数与源逐项相同，初态仍0度16、5度32、15度48、30度前倒32；不热改奖励、不提高DR、不启用slew新变量。结束自动原有五角度诊断与v346/v361双基线对照；90度仅历史诊断，不作为强制训练目标。worker3041485/trainer3041513已执行首动作，实际配置核验保存artifacts/continuation_v362/runtime_verified.json。该轮用于未掌握的终态连续性适应，已掌握近站姿仅rehearsal；不因短期平台提前停，不自动晋升。


### 371. 用户确认选项1：只排除完全倾覆、底盘朝上

用户明确保留普通前倒、后倒和侧倒恢复，仅排除完全倾覆、底盘朝上姿态的高难度起身。这覆盖第368–370条中待确认的边界；普通90度倾倒不能一概排除，按实际身体朝向解释。排除仅针对训练初态及必须恢复的起点，保留初态恢复过程中再次翻覆仍记录失败，不能因此豁免retention/安全要求。长期目标文档已同步，最高DR与八点locomotion要求保持。

当前v362只含0/5/15度及30度前倒初态，均在保留范围内，继续原预算及固定参数；无需因边界澄清中断或重启。


### 372. 用户新增recovery独立坡度DR，确认±10度目标

独立坡度课程0→2→4→6→8→10度，覆盖不同坡向、保留平地retention，最终与动力DR联合验收；坡度与倒地角分开，完全倾覆起点仍排除。新增configs/rmuc_recovery_slope_dr.yaml显式记录目标且默认关闭、原生接入未认证。

新增recovery_slope.py提供坡面normal/wxyz旋转、局部地面高程及法向净高计算。五项几何测试通过，覆盖平地等价、四坡向正负号、四元数与平面法向一致、净高及非法倾角拒绝。当前尚未接入Isaac碰撞地形/reset/reward/evaluator，不能声称已实现坡度DR训练；后续需原生碰撞与初态验证，再独立低坡度适应。未改正在运行v362的环境行为或动力。


### 373. 用户询问云台/发射机构碰撞：当前只确认机体简化几何，未确认独立覆盖

直接遍历当前训练rmuc_robot_closed_loop.usd的CollisionAPI：机体robot仅sphere(radius.045)和box(size1、父scale .32/.14/.14)，另有腿连杆、主轮与导轮碰撞体；未见独立云台/发射机构碰撞体。robot外观为visual_merged，外观存在不能证明凸出部件具备准确碰撞。不能排除机体盒体近似覆盖部分装配，但完整云台/发射机构外形包络未核实。清单及资产hash见artifacts/continuation_v363/collision_inventory.json；未改机器人资产。


### 374. 独立固定坡面原生接入预检：2度几何初态及施力通过，尚非坡度DR策略

新增RmucSlopeTerrainImporter，沿指定坡向旋转实际地面碰撞网格；recovery默认坡度0不改变平地路径。非零坡度在clone后计算每env原点的坡面高程，将已审计平地初态整体旋转到坡面，使用坡面法向净高用于reward/终止和两主轮高度差。重力和电机契约不变；当前是每run一个固定坡面，尚不是每env/每episode随机坡度，坡向/角度独立于动力DR。

首次原生probe发现178度而非2度，定位为几何helper wxyz与Isaac Lab3 xyzw契约不同；在调用原生quat_mul前显式转换后，四env实际初态均1.99978度，128控制步无终止，512substep气弹簧反作用误差<=2.29e-5Nm。失败报告保留，成功报告artifacts/continuation_v363/slope2_fixed.json。仅合成零动作预检，不是站稳策略、全部碰撞无穿透或±10度覆盖证书；evaluator/导出契约与随机坡度接入仍待完成，不开启坡度训练。


### 375. v362末checkpoint仍退化，停止盲目追加训练并检查已有checkpoint

v362/model2462完成五角度：0度14/16、5度16/16、15度16/16、30度4/16（前倒4/4，unsafe8）、90度0/16（unsafe8）。0度两例不满足末段连续hold；对v346和v361均有0度保护组退化，不晋升。不能因30度补齐而宣称整体改善。此前共享actor仅依赖初态rehearsal且总取末checkpoint，防退化不足；当前没有追加新训练。

新增逐控制步ready条件、hold、角速度和轮世界速度记录。原checkpoint/seed/16env/0度重测精确复现14/16，失败例其他ready条件全程通过，双轮载荷短暂归零使hold清零。例0在7.736s左/第一主轮由约94N降0、轮心高度.0600055→.0602202m、竖直速度约-.093m/s，下个8ms控制步载荷恢复128N；还有轮运动变化，不能凭零载荷认定纯传感错误。例10有多次短中断。没有下调2N支撑阈值、.5s连续hold或高度门槛。证据artifacts/continuation_v364/{standing_full_trace,contact_interruptions}.json。

下一步利用已训练的model2450/2425做有限checkpoint选择验证，先0度/30度前倒，再5/15度和90诊断；任何保护组失败即拒绝该候选。此为选择集，不冒充独立多seed最终验收；无候选同时通过则需改变保护方法后再训练，不能继续原样追加300次。评估命令和结果保留artifacts/continuation_v365。

### 376. 坡面评估与导出记录独立地形，2度近站姿局部通过

recovery evaluator恢复源坡角/坡向并支持显式诊断覆盖，报告新增实际地形与高度参考；export记录training_ground且不认证坡度。flat retention拒绝坡面或坡度诊断报告，相关与几何测试共21项通过。v361在2度、坡向0、0度相对坡面初态四实例中4/4终态通过、unsafe0；单坡向小样本近站姿，不能称四方向倒地或±10度DR通过。当前仍固定坡面每run，不是随机坡度DR；产物artifacts/continuation_v363/policy_slope2.json。报告启动时scope文字仍写nominal dynamics，这指动力参数；地形由recovery_ground与diagnostic_slope字段明确标注，后续源码已补充坡面scope条件。


### 377. 有限checkpoint筛选也未找到全保护通过者，拒绝拼接与继续盲训

v365用相同门槛与16replicas依次检查已有model2450、2425。2450的0度16/16、30度前倒4/4，但5度14/16，拒绝；2425的0度16/16、30度前倒4/4、5度16/16，但15度15/16，拒绝。均未通过全保护项，因此未继续该候选90度诊断，不改原训练/报告，不拼接不同checkpoint的通过结果。selection_progress.json是选择集记录，不是最终独立证书。

处理原则修正：共享actor的初态rehearsal只能提供训练样本，不保证旧动作/支撑行为受约束；单看末checkpoint或总通过数均不足。有限筛选失败后，停止相同配置无限追加；下一训练前需建立基于保留基线的旧状态动作/策略更新保护，并用同一候选全保护项验收，再独立seed/随机初态验证。现有精确支撑与hold门槛不降低，接触短中断仍按原规则失败；高频接触根因未完全分离，不能把全部变化归因于奖励、学习率或传感器。


### 378. recovery rehearsal固定基线动作约束接入并通过原生PPO验证

新增scripts/rsl_rl/rmuc_recovery_anchor.py和显式CLI，固定source actor指导<=15度rehearsal cohort，30度目标cohort不加该项。附加损失为保护样本上6D原始mean action与reference的均方差，通过mean梯度hook在PPO裁剪前加入；不改奖励/action/logprob、不增加25D输入。mask临时附加到storage observations随minibatch索引打乱，只用于梯度选择，update后移除。该项是软约束，不宣称参数隔离或绝对不遗忘。

4项测试通过：参数梯度与显式辅助loss反传一致、输出不变、空mask/零系数无影响、mask与shuffle一致。原生3更新（32env、32步、24保护env、4epochs）每更新3072保护样本进入16个minibatch，loss正且actor有限、源hash不变。首次日志目录未建立报错已修复；原生日志iteration读取runner尚未更新导致标签滞后，后续改独立更新计数，不篡改旧日志。证据artifacts/continuation_v366/verification.json；三次更新仅接线证明。

### 379. 用户要求加快节奏，启动100更新短程v367

用户明确直接短程续训、尽快坡度/最高DR/sim2sim。已从v361/model2162启动v367预算100更新2163–2262，新增的唯一学习因素为上述anchor coefficient1；固定reference同v361。128env中96个0/5/15度rehearsal受约束、32个30度前倒目标不加此项，其他reward/PD/slew3.5/DR0/初态配比不变，源配置逐项回读一致。worker3090725/trainer3090762已实际运行，前3次更新保护样本49152/update、16minibatches，loss为正，详见artifacts/continuation_v367/runtime_verified.json及run内recovery_anchor.jsonl。

预计本轮训练加五角度评估约10–15分钟（不是完成最高DR的承诺）；阶段末对v346/v361双基线验收后再低坡度/低DR，仍不晋升退化候选。新增长训练不得原样重复失败配置；短程评估通过后及时进入下一未掌握条件。普通大角度恢复、±10坡面、最高联合扰动、完整sim2sim依旧待完成。


### 380. v367短程100更新通过现有全部nominal保护项，进入低坡度/低DR基线

同model2262 sha256 58c74c5c33ae374a8c46c020814e35bc1034f9f42fe6292dd969a82780362c47，0/5/15度各16/16 unsafe0；30度4/16且前倒4/4、其他方向0/4、总unsafe8；90度0/16 unsafe0。v346与v361双基线retention均通过无退化，第一次同候选覆盖两者保护项并集。原动力/严格height.28/hold.5/slew3.5/3Nm不变，实际电机采样限幅通过。只说明固定初态、当前seed下的局部能力；不是普通全方向倒地、随机初态、多seed、最高DR或sim2sim认证。

加入anchor的本轮得到有效局部结果，但未做等预算无anchor配对，不能把全部改善严格因果归于辅助loss。旧基线保留，完整goal不完成。artifacts/continuation_v367/{final_summary,continuation_v346_retention,continuation_v361_retention}.json保存。

v368已启动独立新条件基线，不继续相同nominal长训：同v367模型，2度坡面/相对坡面15度倒地初态，坡向0/90/180/270分开测；另在平地用硬件profile、DR.05测0/15度。各16实例、seed368，坡度和动力DR不混合。结果用于决定短程适应缺口，不是最高联合DR证书；worker3104413已运行，artifacts/continuation_v368保存各命令、报告与状态。


### 381. v368低坡度与DR基线完成，未同时加大扰动

同v367/model2262、seed368、每条件16实例。2度坡面/15度相对坡面初态：坡向0/90/270各16/16 unsafe0；180度11/16 unsafe0（前4、后3、左0、右4）。说明坡向确有差异，不把三个坡向通过外推全坡面。硬件目标profile下DR.05：0度14/16 unsafe0；15度14/16 unsafe1，均为前倒/左侧各一例未过。单seed低强度诊断，不是最高联合DR；具体报告与读回在artifacts/continuation_v368。

### 382. v369开始100更新DR.05适应，坡度保持0并独立验收

从v367/model2262接续100更新2263–2362；初态配比、奖励、固定slew3.5、PD与3Nm、anchor reference v361/系数1保持。启用硬件profile中的damping随机范围、气弹簧共同/独立各.1、电机响应模型，并以DR强度.05实际缩放，与v368基线对应；名义轮damping.0225、名义气弹簧10.775Nm不变，坡面为0。该轮是组合动力低强度适应，不归因于单个物理因素。不会把低坡度和动力DR同时升高。

worker3112452/trainer3112472已实际运行至2265，运行配置逐项核验并保存artifacts/continuation_v369/runtime_verified.json。阶段末自动五角度nominal retention对照v367，再在同seed/profile/DR.05复测0/15度；无自动晋升。evaluator补齐从源参数恢复等效力矩DR分量与电机响应字段，默认nominal strength仍0，不将训练随机化强度混入平地retention。坡面180度的薄弱方向仍列为后续独立适应项，±10度目标不变。


### 383. v369低DR短程未通过整体验收，保留v367不升强度

v369完成100更新到model2362。nominal：0度15/16、5/15度各16/16、30度前倒4/4（总4/16 unsafe8）、90度0/16 unsafe0。相同seed368/profile/DR.05下，0/15度均14/16 unsafe0；相比v368，15度unsafe1→0但通过数未提高，且nominal0度退化一例。该例高度/姿态/线角速度全程合格，仅双轮接触短中断导致终态hold不足，最长ready7.672s。仍不晋升、不提高DR、不原样追加训练。

现有anchor仅约束新rollout的保护cohort状态；新动力/噪声改变了访问状态分布，不能视为旧nominal状态的直接保护。这是方法的覆盖缺口，不声称已经严格定位唯一退化原因。下一步增加固定旧状态reference replay，保留原严格门槛与物理契约。

### 384. v370采集并校验固定nominal成功状态—动作参考数据

评估器新增可选--save-reference-rollout，每4个控制步记录实际送入actor的25D观测、raw6D动作、env/case/control_step，拒绝动作干预混入。只记录首个完整episode内尚未结束的实例，不重写训练动作或奖励；现有未启用路径不采集数据。

v367/model2262的0/5/15度各16实例重新采集，三组均16/16终态通过；每角度4000样本，共12000。CPU恢复同一actor复算存储观测，最大动作误差分别4.03e-7/5.29e-7/5.97e-7，checkpoint和数据hash记录artifacts/continuation_v370/verification.json。数据包含开始到稳站过程，不是仅末段均值；将用于旧状态更新保护，不能再当独立测试集或多seed鲁棒性证书。当前没有新增长训练。


### 385. 固定旧状态replay接入，梯度在PPO裁剪前合并

新增scripts/rsl_rl/rmuc_recovery_replay.py与训练CLI，读取v370带hash的成功reference数据，拒绝缺角度、非同checkpoint、失败或非平地数据。按0/5/15度×四方向×起步/后期共24组抽样，每组16条、384条/minibatch，独立Generator不改变主PPO随机流。固定旧观测上的raw mean action MSE梯度通过参数hook加入原PPO backward，保留原gradient clipping；不改rollout动作/reward/logprob，不增加actor输入。软约束仍不能替代行为验收。

梯度与显式联合loss在裁剪前后相等、清理后不残留、原有anchor测试共6项通过。v371真实32env/32步/3更新中每更新16minibatches/6144固定参考样本进入，原online anchor同步记录；actor有限、源hash不变。原生证据artifacts/continuation_v371/verification.json，不用短测判断能力。

### 386. v372固定回放短程对照启动，100更新DR.05不升强度

使用与v369相同v367/model2262起点、128env/128步、seed268、DR.05硬件profile、同初态/奖励/PD/slew与online anchor，仅新增固定nominal replay coefficient1。预算100更新2263–2362，阶段末五角度nominal retention对v367及相同seed368的DR.05在0/15度对照。运行配置逐项与v369比较一致，新增replay契约单独记录，不换物理门槛或晋升失败候选。

worker3136276/trainer3136297已实际执行，前3更新固定replay loss正且采样数6144/update；见artifacts/continuation_v372/runtime_verified.json及run内recovery_replay.jsonl。相同训练条件便于比较，但是否改善保留与DR行为仍待阶段末验收，不将辅助loss非零视为防退化成功。


### 387. v372固定回放保住全部nominal保护项，低DR通过数未提高

100更新到model2362完成。nominal0/5/15各16/16 unsafe0，30度前倒4/4（总4/16 unsafe8），90度0/16 unsafe0；对v367的全部保护组retention通过。相比v369 nominal0度15/16，本轮旧状态固定回放得到局部保留改善，仍非绝对不遗忘证明。

相同seed368/profile/DR.05在0/15度仍各14/16 unsafe0，失败均为index4/10，初始obs delay0、communication delay3（24ms）。0度失败末高约.352m、倾角5–6度，最长ready仅.08s；15度前倒例最终倒置，左侧例高.352m但保持不足。低DR仅降低延迟命中概率，不降低命中长度；nominal保住不等于延迟鲁棒性通过，不升全局DR强度。

### 388. v373同抽样通信延迟消融支持独立延迟课程

新增显式diagnostic-communication-delay-steps评估入口，仅在首次reset采样后覆盖动作通信延迟；记录原抽样值与实际覆盖值，其他物理DR采样不变。flat retention拒绝此诊断，17项测试通过；不修改训练默认延迟或成功门槛。

同v372checkpoint、seed368、profile、DR.05，0/15度各16例配对。强制通信delay0后0度14→15通过、15度14→16通过，unsafe均0。核验两组初始mass/friction/damping/spring/obs delay/3Nm等所有其他读回逐项相同；三个失败例变通过，0度index10仍因接触短中断失败。可以支持动作通信延迟对这几个样本有因果影响，不能推广为所有DR唯一根因。

证据artifacts/continuation_v373/paired_verification.json。下一步将延迟长度独立渐进，再回到原1–3周期分布及最终最高联合DR；强制关闭延迟只是诊断，不作为目标DR验收。没有原样追加v369/v372训练。


### 389. 独立通信延迟长度课程，保留最终原始分布

新增recovery_communication_delay_max_steps，默认-1不限制，课程可0/1/2/3控制周期；在reset原抽样后截断，只改已抽中延迟的长度，不改变零延迟、其他DR或RNG。保存截断前与实际延迟。单测覆盖默认等价、逐级上限、输入不变/RNG不变及非法值，与retention测试共22项通过。评估默认不继承训练cap，明确用原分布；只有显式diagnostic cap才做课程评估，并被nominal retention拒绝。导出额外记录训练cap，不冒称最终DR通过。

v374同v372/seed368/profile/DR.05，仅cap1评估0/15度仍各14/16。两例原3周期变1，其他初始DR读回逐项相同，说明8ms也需适应，不能跳到3周期或宣称只差极端延迟。

### 390. v375启动100更新1周期课程，三类评估保持分开

从v372/model2362接续100更新2363–2462，唯一训练条件变化为communication cap1；DR.05、坡度0、初态配比、reward/PD/slew及online anchor与固定reference replay保持，源参数逐项核验。worker3158586/trainer3158641已实际执行至2367，证据artifacts/continuation_v375/runtime_verified.json。

阶段末自动五角度nominal retention对v367/v372、原始未限制DR.05的0/15度、cap1课程的0/15度三类评估。课程改善不能冒充原始1–3周期或最高联合DR通过；普通四方向恢复、±10度坡面、八点locomotion/最高DR/sim2sim目标保持，完全倾覆起点仍排除。


### 391. v375通过1周期课程且保留nominal，原始延迟分布仍未过

100更新完成model2462（sha cfdc7f7de8c0a09a942a2c5c5c74ff2f15edc9e4f6d413883e5337a8ee0d8931）。nominal0/5/15各16/16、30度前倒4/4，对v367/v372全部保护组通过。原始DR.05/1–3周期分布：0度14/16 unsafe0、15度14/16 unsafe1，不认证该分布。

同seed368/profile/DR.05、communication cap1课程：0/15度各16/16 unsafe0，较v374的各14/16改善。只在所测固定初态/seed下通过短延迟课程，不等于全范围恢复或最高DR。保留所有未限制分布失败记录；不把cap1成绩覆盖原始分布。

v376已启动同model/seed的cap2（16ms）0/15度基线评估，之后按缺口决定短程适应，不直接提高全局DR或跳过2周期。证据artifacts/continuation_v375与v376，完整目标仍进行中。


### 392. v376确认2周期缺口，v377单独将延迟上限1→2适应

同v375/model2462、seed368、profile/DR.05，cap2基线0/15度均14/16 unsafe0；失败均集中于前倒/左侧各一例。1周期通过不能外推16ms，未上调全局DR。

从v375独立续接v377预算100更新2463–2562，唯一课程变化为communication cap1→2。DR.05、坡度0、初态配比、奖励、PD/3Nm/10.775Nm、slew3.5、online anchor和固定replay均保持。worker3182807/trainer3182834已执行至2464；实际保存配置逐项与源比较通过，见artifacts/continuation_v377/runtime_verified.json。

阶段末自动nominal五角度对v367/v375 retention，外加同seed下原始延迟分布、cap1、cap2各0/15度评估。必须保留1周期能力，cap2通过也不冒称3周期或最高DR完成。v376基线和全部命令/报告保留，不重新解释旧分布成绩。


### 393. v377未保住1周期能力，2周期仍未通过，不晋升

v377完成100更新。nominal0/5/15各16/16、30度前倒4/4，保护项保留；但cap1下0度15/16、15度14/16，低于v375各16/16。cap2下0度14/16、15度12/16；原始1–3周期分布也为14/16和12/16，unsafe均0。2周期尚未掌握，1周期有遗忘，不能将nominal保持视为全部retention通过。

现有固定reference仅覆盖nominal，未直接保护已学延迟条件；这是覆盖缺口，不等于已确定唯一根因。v375仍为1周期基线，v377不替换它，不追加同配置训练。

### 394. v378补齐同一teacher的nominal与cap1回放并按延迟分层

从v375/model2462同一checkpoint采集nominal0/5/15与DR.05 cap1的0/15，五组各16/16通过、unsafe0。共20000条实际送入actor的观测—动作，CPU复算最大误差4.77e-7。记录每env初始DR读回、实际通信延迟、cap及profile hash；训练参考数据不得当独立验收。

replay loader扩展为condition/角度/方向/起步-后期/实际延迟分层，稀少延迟样本独立成组，避免被零延迟样本稀释；新增DR强度/cap/强制干预一致性检查，保留旧manifest兼容。真实数据校验48组、20000条，相关梯度测试6项通过；artifacts/continuation_v378/verification.json记录文件hash与复算误差。中断后确认全部采集已结束，无重复启动。

### 395. v379启动100更新条件回放对照

与v377使用相同v375起点、seed268、128env/128步、DR.05、cap2及奖励/PD/3Nm/10.775Nm/slew；只替换固定回放数据为同teacher v375的nominal+cap1数据并分层抽样，系数仍1，online anchor不变。预算100更新2463–2562，阶段末nominal/原始延迟/cap1/cap2独立评估，保留v367/v375基线，不自动晋升。

worker3241218/trainer3241238已实际运行至2468；保存配置与v377逐项一致，48组×16样本×16minibatch=12288参考样本/update。记录artifacts/continuation_v379/runtime_verified.json和run内recovery_replay.jsonl。固定回放依旧是软约束，是否同时保留旧延迟能力并学会2周期待完整验收。


### 396. 用户调整优先级：先完成locomotion八点与sim2sim链路

2026-10-01用户要求先把locomotion做好；八点是ONNX/MuJoCo sim2sim链路的一环，不是独立终点。recovery暂不追加训练，已有课程、斜坡±10度、最高联合DR等目标仍保留。

v381以保留v278/model6684、seed67、128env、每点至少32完整回合测试最终名义动力（轮限幅/动作scale均3Nm、damping .0225、每腿等效气弹簧10.775Nm、径向力0、DR0），结果0/8且无稳态样本。旧v323的7/8使用轮限幅4.875Nm、动作scale5.5、damping .080625、等效力矩0及径向力2.69375N。两者不是同条件回归比较；不能说旧训练成果全部失效，也不能将旧7/8宣称为最终硬件成绩。此前直接切换全部参数缺少旧条件复测，需要补齐归因。

v382保持最终轮参数只关闭气弹簧仍0/8；恢复10.775Nm重复v381各点结果，不能将失败全部归因于气弹簧。原生3200物理子步实际世界力矩误差最大2.10e-5Nm、反作用合力矩误差2.48e-5Nm。此证据仅核查施加，不证明真实气弹簧曲线已辨识。

完整actor ONNX/JIT复核通过（2409条合成边界观测，ONNX最大误差2.38e-7）；基础MLP单独导出相差最多.446856，交付必须保留完整模块。MuJoCo locomotion诊断探针新增显式等效气弹簧参数及与历史径向力互斥检查；八条命令在3Nm/.0225/10.775Nm下均于.544–.784秒触发探针45度/高度限制。该探针每点单次、阈值和reset尚未统一，不是MuJoCo正式八点验收，不可直接比较两个引擎成功率。64组广义力/反作用/虚功验证通过。证据保存在artifacts/continuation_v382。

v383已启动同checkpoint、seed、八命令和回合数的旧动力条件复现，以当前代码确认旧7/8是否仍成立；历史参数仅用于诊断，不修改最终硬件目标、不恢复为部署契约。需完成后再决定渐进动力适配，不直接宣布代码无回归。

v382追加真实轨迹推理核查：160条Isaac记录观测经ONNXRuntime和MuJoCo共用consume_action复算，raw及executed动作最大误差均8.35e-7，见trace_parity.json。覆盖这些失败状态上的推理与动作处理，不覆盖MuJoCo状态构造、reset或动力学等价。


### 397. v383旧条件准确复现7/8，v384只检验3Nm限幅

v383当前代码下复现历史v323的八点：每点32完整回合、survival均1，仍仅+0.8未过。八点completed_episodes/survival/linear_error/yaw_error/stable_fraction/passed逐项与历史报告完全一致，见artifacts/continuation_v383/historical_comparison.json。这支持旧技能在该固定seed和八命令下未丢失；v381的0/8不能解释为同条件代码或训练退化。

下一步v384同checkpoint/seed/命令，仅将实际轮输出限幅4.875→3Nm；动作scale仍5.5、damping .080625、等效气弹簧0和历史径向力2.69375N均保留。该设置只用于分离限幅影响，明确不是最终硬件契约，不能替代10.775Nm等效力矩迁移。评估已启动，完成后根据证据决定后续单因素迁移，不改变验收门槛。


### 398. v384单独3Nm限幅为6/8，开始正确气弹簧模型的渐进迁移

v384每点至少32完整回合，六点通过；+0.8仍yaw误差.0529605、stable .207149。新增失败+1.4：252回合survival0、无稳态样本。其余七点survival1。因此3Nm硬限幅有局部影响，但不是最终参数0/8的充分解释；均值电机力矩低不能替代瞬态限幅检验。

遵照用户要求，让策略逐渐覆盖不同动力学条件。v385先评估正确等效助力模型的10%迁移起点：3Nm硬限幅、动作scale5.5、damping .080625、径向力0、每腿等效力矩1.0775Nm、DR0。此处改变的是迁移阶段的名义助力，不是把最终10.775Nm重新定义为1.0775Nm，也不是最高DR。该起点评估完成后再制定100更新短程适配；先前历史径向力对照保留但不作为最终训练分布。


### 399. 正确气弹簧迁移起点7/8，v386正式100更新适配已启动

v385无新增训练即获得7/8：3Nm限幅、scale5.5、damping .080625、径向力0、每腿1.0775Nm等效力矩、DR0。+0.8已通过（yaw误差.021953、stable1），其余六个通过点保持；仅+1.4在271回合中survival0且无稳态。不是最终10.775Nm或最终damping成绩。

v386从原v278/model6684续训100更新6685–6784，保持v385动力参数，训练既有physics-adaptation六个网络张量，冻结其余actor；主课程51对应±1.4，rehearsal保留48/49/50及历史anchors。源checkpoint实际保存的physical optimizer lr为1e-4（critic同为1e-4），不要将通用algorithm lr2e-6误记为该组实际步长；此轮保持已保存optimizer。固定动作std、1个PPO epoch，未改reward来追分。

worker3360058/trainer3360096已进入实际更新，检查时audit至6686；保存配置核验3Nm、PD60/2/40Nm、2ms/8ms、1.0775Nm、径向0和DR0一致，见runtime_verified.json。训练结束自动检查actor非physical状态不变，并评估迁移八点对v385七点retention及完整最终硬件八点，不自动晋升。该轮先建立正确模型下名义适配，之后再扩展动力DR；不能称当前已经覆盖最高DR。


### 400. v386完成100更新但retention失败，不晋升

v386/model6784迁移条件6/8：+1.4仍survival0，-1.2由通过退化为yaw误差.058034、stable .008152；其余六点通过。最终3Nm/scale3/damping .0225/10.775Nm八点仍0/8。原checkpoint与七点迁移基线继续保留，不能直接追加相同训练或升DR。完整候选hash、参数冻结校验和行为对照见artifacts/continuation_v386。

100更新累计实际+1.4样本214001，-1.4样本603592；说明回合方向配额不等于转移样本均衡，但失败方向并非完全没有数据，不能据此认定唯一根因。只有physics网络六张量改变不意味着行为隔离，-1.2实际退化再次证实必须做逐项retention。


### 401. 按用户要求查错并单独训练+1.4，停止共享physics更新

复盘v386及旧教训：冻结旧模块不等于行为隔离，共享physics-adaptation仍对正负转向共同生效；reward bank、回放或梯度投影都不能代替闭环retention。v386只改变六张量仍丢失-1.2，属于真实失败，不晋升。回合方向配额也不等于转移样本均衡（+1.4累计214001，-1.4累计603592），但不能将此认定唯一根因。实际optimizer组lr1e-4与通用algorithm lr2e-6必须分开记录。

复用已有local correction expert3（center+1.4，signed yaw紧支撑1.25–1.55），不再新增共享模块；从未退化的v278/model6684重开，只训练expert3并冻结完整physics-adaptation及其他actor，重置optimizer避免加载原共享组动量。保持v385的3Nm、scale5.5、damping .080625、1.0775Nm等效助力、径向0、DR0和原reward/成功门槛。

新增默认关闭的command_turn_positive_fraction_override用于显式方向采样，运行时拒绝越界/非有限值；本轮设为1，关闭rehearsal及连续命令采样，使训练只暴露+1.4。默认-1保留旧分配逻辑，独立评估不启用强制方向。异步reset定向测试及既有local/quota测试共18通过。

完整actor隔离检查在2048探针上对expert3执行三次测试optimizer更新，其他七命令1792观测的输出逐位不变，target输出确有改变，且仅expert3参数改变。此证据限定于可正确重构命令的观测：yaw-error截断状态和邻近连续命令仍须行为评估，不宣称全域绝对隔离。artifacts/continuation_v387/isolation_audit.json保存证据，未将测试更新写成训练checkpoint。

v387正式100更新已启动，worker3400116/trainer3400161，检查至6686。最新rollout16384样本中16383被重构为+1.4，其他1（可能受观测误差截断影响，不据此改标签），七个保护命令及负1.4均0；local_samples仅index3非零，冻结校验无违规。runtime配置已核验，结束后自动同条件八点retention及最终动力八点，不自动晋升。短接线与隔离测试不证明起步已学会。


### 402. v387完成单分支100更新；发现yaw误差截断的路由歧义

v387/model6784已保存，sha256=12250b058e9054c7c287f53dcc72cf4983a998d5b99f58dadf23174c77884e45。完整checkpoint审计仅local_correction.experts.3的20张量改变，其余actor状态和原checkpoint不变。当前进入迁移八点与最终动力八点评估，尚未晋升。末段训练平均回合约115步，不能从冻结校验通过推断起步成功。

查当前saved cfg及观测构造：command_yaw_observation_error_clip=2。完整actor恢复后可复现：真实command+1.4、实际yaw=-1时，delivered error=clip(2.4)=2，重构command=-1+2=1，expert3 gate变0；实际yaw4时也会重构成2并关门。样本标记出现other/少量其他命令不能直接解释为sampler混入命令，因为它按同一截断观测反推。训练后期局部分支门控不再仅index3非零，但非目标分支仍冻结。

command_reconstruction_audit.json记录反例，仅证明路由歧义可发生，不证明真实失败频率和因果主次。不能为了单独训练在部署中偷偷使用训练器真值命令，或直接改变25D/6D契约而跳过ONNX/MuJoCo验证。下一步结合原生失败轨迹定位歧义与跌倒的先后关系，再决定显式观测契约修正；当前运行评估的观测和门槛保持不变。


### 403. v387七点逐项保持但+1.4仍失败，开始原生轨迹定位

单独训练100更新后迁移条件仍7/8。七个保护点completed_episodes/survival/linear_error/yaw_error/stable_fraction/passed与v385逐项完全相同，闭环retention通过；+1.4累计411回合survival0，无稳态样本，仍未解决起步。最终动力八点0/8。候选不晋升，不追加同配置训练。

v388从原source与v387候选各采集同seed、同迁移配置、仅+1.4的16env失败轨迹，每组至少32完整回合。evaluate_rmuc_grid仅在failure-traces诊断中追加6关节角、角速度及电机applied torque，明确是policy步末物理子步读回，不包括被动气弹簧/约束反力或中间峰值。用于检查姿态、门控和限幅的先后关系；采样周期仍10控制步，不能据稀疏轨迹宣称唯一因果。物理配置、动作和验收门槛均未改。


### 404. 用户授权提前结束无意义训练；先定位早期俯仰失稳

v388原生对照已完成：原基线16条失败轨迹中无命令重构误差，终止step中位168.5；v387候选16条均出现误差，终止step中位114，但误差均在pitch首次超过20度以后。轨迹每10控制步采样，因此不能排除未采样瞬间；现有证据不支持把门控误路由当原始起步失败主因，禁止据此直接修改观测契约宣称修复。

用户明确允许不达标时提前结束无意义训练。后续短程实验按25更新一段保存并检查真实目标行为；明显退化或接线错误立即停止，连续两段无行为改善不默认追加至100，更不能以reward上涨继续。早停是资源决策，不等于证明任务学不会。候选晋升仍需完整八点retention；即时少量样本仅作停止/继续依据，不能代替正式验收。

v389正在相同正确等效气弹簧1.0775Nm、径向0、damping .080625、scale5.5下对比3Nm与4.875Nm限幅；高限幅仅诊断，不作目标回退。failure trace追加wheel请求和预PD裁剪后target，以区分请求裁剪、实际电机输出限幅及被动损耗。原基线早期pitch持续增大而命令重构正确，需检查这个阶段的驱动力/腿姿态，不继续沿失败的单分支配置盲目续训。


v389完成：3Nm下32回合survival0；4.875Nm诊断下40回合survival.30，yaw误差.032595、stable .890468仅来自存活进入稳态的样本，整体仍未通过。两组实际读回气弹簧1.0775Nm、径向0、damping .080625、scale5.5一致。3Nm源轨迹第11步右轮请求3.422106Nm、预PD target3Nm、实际步末电机输出2.140293Nm，发生在pitch仅.106度时。限幅影响早于明显俯仰失稳，但高限幅仍有70%失败，不能当唯一根因或部署解法。保留3Nm最终硬限幅，下一步从起步期驱动力/腿姿态协调和渐进动力适配继续查证。未追加训练。


### 405. damping单因素未救起步，转查转向难度课程

v390保持3Nm、scale5.5、1.0775Nm等效助力、径向0、原checkpoint，仅将wheel damping分别降到.060625与.04；+1.4各32完整回合survival0、无稳态。不能把降低damping当起步修复，不继续无依据参数扫。

v391回到已证实七点通过的迁移动力（damping .080625），仅评估+1.25/+1.30/+1.35三个中间命令，每点16env、至少32回合；寻找局部单分支训练的可学习难度起点，再按25更新分段推进到+1.4。门控center+1.4、halfwidth .15意味着+1.25处目标expert门控为0，故该点只作边界对照，不可选作新分支有效学习点。其余七点与最终动力/最高DR/同契约sim2sim要求不变。


### 406. 用户纠正峰值/持续契约，优先damping鲁棒与sim2sim

此前将3Nm持续指标当作执行器瞬态硬限幅，偏离用户现明确的5.5Nm峰值契约。configs/rmuc_hardware_target.yaml升级schema2：peak5.5、continuous3、action scale5.5、effort limit5.5；hardware helper与审计同步，明确支持归档schema1降额诊断但不重解释旧结果。20项profile测试通过。3Nm训练候选不晋升；尚运行的v391已按用户新口径终止，状态标注cancelled而非伪装评估失败。

用户进一步要求不执着于3，先damping鲁棒再sim2sim并赶进度。v392从原model6684以5.5Nm峰值、scale5.5、1.0775Nm阶段助力、径向0，分别评估damping .080625和.04的完整八点（每点至少32回合）；不同时改变reward或训练新权重。目的为选择damping短程课程起点，随后25更新分段、单独分支优先、全八点retention，尽早同契约MuJoCo复测。最高联合DR/最终10.775Nm/热筛查仍未完成。用户提出的<5%和<300ms仅记为工程筛查参考，不标记为厂家认证阈值。


### 407. 实机RM3508+C620确认，MuJoCo峰值链路完成八条12秒诊断

用户确认轮端硬件为RM3508+C620，已写入目标配置。旧rmuc_usd_limits.xml轮actuator ctrlrange/forcerange仍为±3，不能用CLI宣称5.5而绕过模型能力检查。v392生成独立rmuc_peak55.xml，将两只轮电机范围改为±5.5（gear1输出侧Nm），原模型不修改；编译后逐数组核对body/joint/geom/mesh/质量惯量/闭环约束/armature/damping及传动关系未变，证据peak_mjcf_audit.json。

同源model6684完整ONNX、2ms物理/8ms策略、PD60/2、scale5.5、峰值5.5、damping .080625、1.0775Nm阶段助力，在MuJoCo八命令各单回合12秒均跑完，未触发探针45度/高度停止。但后半程速度误差.147–.389m/s、yaw误差.180–.368rad/s，不能当跟踪通过，更不是同门槛/多回合sim2sim正式验收。它证明新峰值配置的执行链路实际可运行，仍存在跨引擎行为gap。commands/summary及逐步轨迹保存在artifacts/continuation_v392/mujoco_12s_*。当前Isaac峰值下damping对照仍在运行，不追加训练或热约束改动。


### 408. damping鲁棒尚未成立，按5.5峰值启动25更新低强度DR

v392完成：damping .080625为7/8，+1.4存活30/44；damping .04为0/8，七点存活但yaw误差.058–.319超标，+1.4存活0。均为5.5Nm峰值/scale5.5、阶段助力1.0775Nm、径向0。不能据恢复峰值就称已鲁棒，不能向正式sim2sim迁移。用户再次确认正式迁移前需最高联合DR验收，已更新所有顶部交接口径。

v393保留原model6684主策略，补齐已有command-local bank缺少的+.8/+1.0/-1.2三个零初始化分支，使八个命令均有独立参数。旧state及新增前后完整actor输出、严格恢复一致，见initialization.json；不使用失败v386/v387权重。只训练local experts，冻结共享physics及原策略其余部分，fresh optimizer，主stage51及75% rehearsal覆盖其余三档、连续采样关闭。

预算仅25更新6685–6709。DR strength .1，仅damping与轮力矩增益随机化；damping中心.080625、当前采样区间[.07659375,.08465625]，尚未覆盖目标[.005,.04]。峰值/scale均5.5；持续3Nm仅热参考，气弹簧固定1.0775Nm；mass/friction/spring/response/sensor等DR此阶段关闭，以分离初始学习因素。新增damping_torque_only评估分支用于与训练分布匹配，不能当最高DR。worker结束自动nominal与低DR八点各至少32回合，检查七个名义保护点，不自动追加预算或晋升。


### 409. MJCF与峰值执行器契约对齐，修正默认入口

当前MuJoCo执行器入口：`sim2real_RL/sim2sim/rmuc_peak55.xml`（轮gear1、ctrl/force均±5.5Nm；腿±40Nm；2ms物理步）。`run_mujoco_probe.py`默认已切到此模型及5.5峰值，policy/damping/gas-spring-torque必须显式指定。旧`rmuc_usd_limits.xml`保留为历史3Nm诊断，不作当前默认。原生12项正反输出饱和检查通过，见`sim2real_RL/sim2sim/peak_mjcf_audit.json`；几何/惯量/关节/闭环约束未变。PD60/2在runtime执行，不重复叠加。此为执行器契约检查，不代表最高DR或sim2sim行为验收。


### 410. Recovery峰值契约入口修正，并区分阻尼/速度参数与实际包络

用户要求所有技能统一执行器契约。recovery新建环境默认scale/effort/authored peak改为5.5；评估器此前未恢复存档轮参数，现先从checkpoint params恢复轮scale/authored limit及轮腿PD/effort，再应用显式诊断profile，避免把历史3Nm策略静默改为5.5。轮输出审计与报告不再硬编码3，按当前actual actuator limit检查。

恢复export继续记录存档真实peak/scale，新增RM3508+C620、持续3参考及是否匹配当前5.5标记；MuJoCo recovery consumer去除乘3/裁3硬编码，读取contract选择足够容量的XML并检查，旧3保持旧语义。编译通过，新增7项真实源码片段回归测试覆盖旧/新参数恢复、3与5.5动作映射、按动态peak拒绝超限；尚未运行新的5.5 recovery行为验收，不启动恢复训练来打断locomotion。

轮端阻尼/速度补充：直接力矩控制、Kp0，显式输出为clip(clip(sign*scale*action,±peak)−D*轮速,±peak)。当前课程D=.080625，v393范围[.07659375,.08465625]；目标先验[.005,.04]中心.0225，尚未覆盖/验收。63.3734rad/s是既有actuator参数，当前IdealPD未实现速度—力矩衰减且无显式63.3734的solver硬限速；不能把它当已验证无载速度或速度闭环。MuJoCo受控关节被动D清零、runtime只扣一次同阶段D。详见docs/rmuc_deployment_control_contract.md“轮端damping与速度语义”。


### 411. v393首段25更新：名义7/8保持、低DR5/8，补齐训练前同分布对照

v393/model6709名义七个保护点全部通过，+1.4存活率30/42=.7143仍不足；低DR只通过5/8，缺+.8（yaw误差.054045）、+1.2（.059581）、+1.4（存活30/34=.88235、yaw误差.070775、stable .39922）。不扩大DR、不晋升、不向正式sim2sim迁移。

实际读回damping[.0766363,.0846419]、轮effort limit[5.5,5.5]；报告确认mass/friction/observation noise/delay等DR关闭，motor response关闭，见result_audit.json。当前局部范围仍不覆盖目标[.005,.04]，更不等于最高联合DR。

补齐v394训练前同DR分布基线。采用v393/zero_extended.pt而非直接原五中心checkpoint：它经完整actor逐位等价验证，但具备与候选相同八中心结构。evaluate先manual_seed再restore_actor、后env.reset，若架构不同会消耗不同初始化RNG，影响DR初态配对；同一seed不自动代表同一DR抽样。待v394报告再做同条件因果比较，不能把nominal到DR差异冒称学习收益。


### 412. 同分布前后均5/8但通过项交换，按独立分支回退后再验

v394训练前低DR通过5/8，缺+.8/+1.0/+1.4；v393训练后通过5/8，缺+.8/+1.2/+1.4。+1.0由yaw误差.059519→.049588转为通过；+1.2由.049176→.059581退化；+1.4存活.911765→.882353。双方runtime contract、seed、DR模式、回合要求、验收门槛和initial_state_audit逐项一致，见paired_comparison.json。故不能按总通过数称无变化或整体晋升，也不能把+1.4的nominal/DR差异冒称训练收益。

v395以v393候选为底，仅将local experts index2(+1.2)、index3(+1.4)的40个张量恢复至训练前zero_extended.pt，其他六分支保留已学更新。4096条完整actor探针上，回退两点与训练前逐位相同、其余六点与训练后逐位相同；未改变控制契约、reward或物理参数。去除混合后不再匹配的optimizer状态，后续如训练必须reset optimizer。选支使用seed67成绩，仍需完整行为和独立seed验证，不能直接算6/8或鲁棒通过。

v395已启动nominal与低DR各完整八点，每点至少32回合；当前不追加训练、不扩大DR、不正式迁移。候选hash及混合来源见composition_audit.json。


### 413. 分支回退低DR仍5/8；分离跨环境reset造成的DR随机流耦合

v395 nominal7/8；低DR仍5/8，缺+.8/+1.0/+1.4。+1.2恢复，但未修改权重的+1.0从v393通过变为失败（yaw误差.049588→.060282）。4096探针只证明给定观测的参数/输出归属，不保证每个策略获得相同随机化时程；此前不能仅据same seed及首次initial_state相同就推断整个episode配对成立。

源码_apply_progressive_domain_randomization在reset批次使用全局torch RNG抽样。其他分支改变reset时序会改变后续环境的随机参数，这是机制上确认的耦合；v395实际成绩变化的贡献大小尚待原生配对，不声称唯一原因。

新增仅用于damping_torque_only评估的--independent-wheel-dr-streams：固定seed、环境数量和reset预算下，预生成每env每reset的D/gain样本，按各自reset计数索引；其他物理DR仍关闭。两项测试验证其他env额外reset/不同batch顺序不改变本env序列且不消耗全局RNG。报告增加初始D/gain读回、采样协议和每env reset计数。默认旧协议保持，旧报告不改写。

v396在相同新协议下比较v393训练后与v395回退候选，各完整八点至少32回合；不是新的训练，也不是最高DR验收。不用新旧随机协议之间的分数差声称策略改善。待比较未改分支的真实行为后再安排缺项短训。


### 414. 用户要求加快渐进DR训练，取消非必要长诊断，降至.02开始短段

复盘：近期诊断/双套长评估占时过多，未及时推进低到高课程。按用户新要求取消仍运行的v396长配对诊断，保留日志并标记cancelled，不当作失败成绩、不重启。独立env/reset随机流功能与测试保留，用于后续廉价配对筛查。

v397从v395保留候选开始25更新6710–6734，峰值/scale5.5保持，DR strength由.1降到.02；本段名义D=.080625、采样[.07981875,.08143125]（±1%），轮力矩增益[.999,1.001]。气弹簧仍1.0775Nm阶段助力、径向0，其余DR关闭。保持各命令独立local参数、共享主图冻结、同reward和PPO设置；从混合候选恢复必须reset optimizer。

训练结束仅做source/candidate配对快速筛查：同独立随机协议、八命令每点4env/至少4回合、最多1600控制步。其用途是停止/继续决策，不替代每点32回合正式验收。阶段推进/候选晋升前再做完整八点；最终仍必须最终动力最高联合DR通过后才能正式sim2sim迁移。计划低到高strength .02/.05/.1/.2/.4/.7/1，实际按行为推进，不自动跳级；扩大随机范围与移动D中心分别进行，目标D[.005,.04]不变，当前尚未覆盖。

### 415. v397快速筛查8/8，立即进入DR 0.05

v397完成25更新，5.5 Nm峰值保持，DR strength .02。阶段末快速筛查（每点4环境、至少4回合）从基线7/8提升到候选8/8，八点无保护点丢失；这只是快速筛查，不替代32回合正式验收。按用户要求不再重复长对照，直接推进下一档。

v398已启动25更新，源model6734，DR strength .05，damping中心仍.080625、相对范围由strength缩放，其他DR保持关闭；峰值5.5 Nm、持续3 Nm仅热参考、气弹簧1.0775 Nm迁移阶段不变。阶段结束只做快速筛查，若无退化继续DR .1；最高联合DR和sim2sim仍需最终完整验收。


### 416. v398 DR=.05完成25更新，快速筛查未形成有效报告

v398实际训练完成25/25（iteration6759），5.5Nm峰值/DR课程保持，冻结审计0违规。worker原先按model6760判断导致假失败，已更正为model6759；候选hash和160个local expert张量审计保存在artifacts/continuation_v398。

阶段快速筛查的Isaac进程达到steps1000后在原生关闭阶段未返回，未写出有效candidate_quick报告；不将部分日志算作通过，也不据此直接进入DR=.1。v397的.02快速筛查8/8仍是最近有效阶段证据。后续可用更短单次原生检查或正式完整八点验收收口，避免再次重复长配对。

### 417. v398 DR=.05完成，单回合快速筛查7/8，不升DR=.1

v398实际25/25更新完成（iteration6759），worker因预期model6760的文件名错误假失败，已修正记录；冻结审计0违规，实际候选model6759。800步筛查不足一回合，结果作废；补做每命令1环境/1完整回合的3500步快速筛查，独立env/reset DR流正常结束：7/8，通过+.8、-.8、-1.0、±1.2、±1.4，缺+1.0（未跌倒但yaw误差.12864、stable0）。实际D范围[.0769896,.0840269]，轮峰值5.5Nm。按“完全鲁棒后再升DR”原则暂不进入.1；下一步只针对+1.0补强并快速复测，最终仍需32回合、最高联合DR和sim2sim。EOF
python3 - <<'PY'
from pathlib import Path
p=Path('inclusion.md');s=p.read_text();s=s.replace('### 下一步执行顺序','### 更新：v398 DR=.05快速筛查7/8\n\nv398已完成25更新。有效1回合快速筛查7/8，仅+1.0 yaw误差不达标，暂不升DR=.1；下一步针对+1.0补强，仍按5.5Nm峰值和逐级DR课程。800步短筛查不计结果。\n\n### 下一步执行顺序',1);p.write_text(s)
PY
### 418. v399在50轮第一停止点完成+1.0单分支适应

v399原计划100更新，按用户新规则在第一停止点处理；实际读取时已完成53更新，停止并保留最近model6810.pt。只改变+1.0 local expert index6的20个张量，其他actor逐项不变，冻结审计0违规。候选在5.5Nm峰值、DR=.05下做每命令1环境/1回合快速筛查：结果待记录于candidate6810_screen.json；这只是快速决策证据，不替代32回合验收或最高DR。

### 419. +1.0单分支53更新后完整32回合仍未通过，停止盲目延长

v399只训练+1.0 local expert6，实际53更新后冻结审计通过。针对+1.0的独立env/reset DR流完整32回合：survival1、yaw误差.065242、stable_fraction.41158、bank_error11.53°、steady wheel slip.275m/s，未达验收；不是单回合随机波动。raw action中一个腿槽饱和，说明更可能是bank/腿位/转向协调或现有控制路由问题，而不是expert样本数量不足。停止继续把该分支延长至100，不晋升、不提高DR；保留model6810和positive10_full32.json，下一步查控制/奖励路由。

### 420. 更正：bank公式oracle正确，+1.0是策略外倾/动作可达性问题

candidate6810的+1.0完整32回合报告：survival1、yaw误差.065242、stable .41158、bank_error11.53°；trace摘要roll_mean约-5.71°。当前bank_reference对command yaw=+1.0、vx=1.0的目标为atan2(1,9.81)=+5.82°，因此误差约11.53°，与报告精确对应。负向转弯通过而正向缺项，支持这是正向内倾符号/asset frame契约问题，不是单个local expert样本量问题。

静态测试test_rmuc_bank_geometry.py已独立证明公式与world/body-up heading几何一致，并证明正向内倾状态bank误差为零、外倾误差超过11°。因此不能修改bank_geometry符号；应查正向腿差动、动作槽/关节轴映射、active leg位置余量和执行饱和，必要时做正向内倾可达性消融，再同步训练和retention。v399继续加轮数无意义，已停止。

### 421. 正向内倾可达性核查：差动方向正确但单差动余量不足

静态bank oracle通过后，回看continuation_v217正向+0.8可达性诊断。固定腿差动从-.30到+.30时，roll由-8.17°逐步到-1.08°，方向正确但仍未达到目标约+4.68°；+.24时-2.16°。负差动明显外倾，故不是左右符号接反。+.30诊断仍有速度误差.279且原策略active-leg槽有饱和；单纯继续增差动会进一步破坏速度/腿余量。需要共同腿姿态偏置、差动和轮补偿联合可达性，不能继续只训练一个local yaw expert，也不能改bank公式。

### 422. 镜像注入与轮端 damping 扫描：先证实结构性不对称，再决定微调

按 nominal 25D interface 对 model_6684 做4096条 synthetic mirror probe。观测镜像变换满足二次镜像恒等；物理轴向量采用 `omega_x/omega_z` 变号、横向 gravity 与 yaw command 变号、左右关节/速度/history 互换。输出按四种关节轴符号假设分别比较：swap-only mean action error .31095，加入 active-link sign flip 后降至 .24877，active+hip sign flip 后 .22751；因此当前结果能确认策略输出不是简单左右镜像，但不能在物理关节符号oracle完成前把误差直接归因于策略网络。报告见 `artifacts/continuation_v421/mirror_report.json`，脚本保留所有符号假设。

固定 `vx=1.0,yaw=+1.0`、5.5Nm peak、1.0775Nm阶段气弹簧、关闭 transitional outer feedback，使用独立D profile做四点短扫描：D=.005 与 .020 四环境均在4回合内 tilt 终止；D=.050 四回合存活但 linear/yaw error=.312/.279、stable=0；D=.080 四回合通过，linear/yaw error=.0414/.0379、stable=.7667。D从.08降到.05明显退化，继续降到.02/.005直接失稳，当前有限点呈单调恶化趋势，支持“正yaw依赖较大阻尼余量”的主嫌疑，但这是4环境诊断而非最高DR验收。D=.080仍有约10.69° bank error，故 damping不是正向外倾的唯一问题。

扫描原始报告与汇总：`artifacts/continuation_v422/D_0.005.json`、`D_0.020.json`、`D_0.050.json`、`D_0.080.json`、`d_scan_summary.json`。下一步按用户方案采用D=.08→.05渐进课程和正yaw定向微调，同时保持bank公式与5.5/3执行器契约不变；不直接把D=.05称为鲁棒，也不迁移sim2sim。

### 423. 2026-10-02 当前推进修正：damping 下界降至0.002并准备并行sim2sim


## 2026-10-02 当前推进修正：damping 下界与 sim2sim 并行

用户将轮端 damping DR 目标下界更新为 **0.002 N·m·s/rad**，上界保持 0.04；权威硬件目标配置已同步为 range `[0.002, 0.04]`、center `0.021`、对称 full fraction `0.9047619048`。5.5 N·m 仍是瞬态峰值，3 N·m 仍是持续热参考；10.775 N·m/腿等效气弹簧仍为最终目标，不能因本次 damping 修改改变执行器契约。

除 damping 外，本阶段 DR 强度按用户要求先锁在当前工况的 20%（训练/评估参数中的 strength=0.20）；不把 20% 强度宣称为最高联合 DR。以 DR=.05 作为 sim2sim 链路并行打通的中间检查点，必须保留 Isaac 八点、多 seed、连续命令及最高联合 DR 门槛。

本轮已把 recovery 训练 CLI 接入渐进课程、目标 slew 退火和坡度参数：`scripts/rsl_rl/train.py` 新增 `--recovery_curriculum`、`--recovery_slew_*`、`--recovery_slope_*`。`probe_rmuc_recovery.py` 已实际验证 0°课程初态、4→3.5 rad/s 目标 slew 退火、10.775 N·m等效弹簧及 +10°坡度初始化；这些是接线/物理审计，不是 recovery 行为通过。

下一步按当前优先级：在不打断用户 GUI 的前提下用 DR=.05 运行同契约 ONNX→MuJoCo 可视化检查并保存桌面视频；随后从已保留 locomotion checkpoint 接续更高线速度/角速度训练，先中期读取一次、结束读取一次，失败或退化即停止。


### 424. DR=.05 gate 仍失败，静态契约通过但行为修复未完成

现有有效证据仍为 v398 7/8、v399 正向1.0专家追加53更新后未通过；不能把 `.05` 宣称为全通过。obs 25D、policy JIT/ONNX、8ms policy/2ms physics、腿PD和5.5/3/10.775执行器契约均完成静态审计。

镜像测试已改为直接调用训练源码的 mirror contract，并修正 wheel velocity/history 左右槽交换；旧 synthetic probe 的错误映射不再作为根因证据。一次误用“DR strength=.05”与“物理D=.05”概念的退火训练已立即停止，未产生新模型。下一步只在原失败条件下定位正yaw闭环问题，通过完整八点 `.05` 验收后再继续 DR 和 sim2sim。

2026-10-04 v454/v455 后置被动项复核：v454按后置被动项语义运行的10.775Nm弹簧短诊断仍未通过，条件为反馈D=.080625诊断值，不能归因于spring轴；v455修正xyzw spring audit后，500-step `max_world_torque_error_nm=1.7166e-5`、`max_net_world_torque_nm=2.0981e-5`，原生弹簧力矩方向与反作用链路正确。下一唯一训练源为v450 `final_checkpoint.pt`，只训练physics residual迁移到10.775Nm最终弹簧，机械b=.002、库仑=.001、反馈D=0诊断、5.5峰值/3持续参考、轮1ms/腿PD2ms/policy8ms；随后先验收1.0/1.5m/s双向转向与±4rad/s自转，再做MuJoCo/ONNX/controller对照。当前DR=.05正式八点、最高联合DR、sim2sim行为和部署仍未完成。

2026-10-04 v456最终弹簧physics residual续训：源 `artifacts/continuation_v450/final_checkpoint.pt`，SHA `a9feac926c66f657ba6e1df5d4138d706f972096d01858096a60e7d210082dd4`；唯一变量为10.775Nm等效气弹簧迁移和修正后的后置被动项。GPU0完成25 updates（10 critic warmup +15 residual），候选 `artifacts/continuation_v456/final_checkpoint.pt`，SHA `de127c2902306daab90f6fb24421fdda63e8e266fe46f40c5e09427c8360d822`。freeze audit显示源actor仅physics adaptation 6个tensor变化，其余冻结；实际读回峰值/scale5.5Nm、连续3Nm热参考、轮actuator damping0、机械b=.002、库仑=.001、反馈D=0诊断、气弹簧10.775Nm/腿、径向0、腿PD60/2/40、physics1ms、腿PD2ms、policy8ms、外环关闭。

v456快速Isaac交付网格（DR0，4 replicas，至少8回合门槛）12点全部 `passed=false`：1.0直行生存但欠速；1.0正向±0.8/±1.0有早倒；1.5直行及±0.8双向跟踪失败；±3/±4原地自转未达到目标。完整结果见 `artifacts/continuation_v456/quick_eval.json`，因此不进入正式32回合、不升DR、不晋升候选。JIT/ONNX同输入parity通过，最大ONNX误差2.384e-7，仅证明导出链路；MuJoCo同一 `rmuc_peak55.xml`、D=0后置被动契约直行6s后半速度0.5289m/s（目标1.0，误差0.4711），+0.8转向为 `tilt_or_height_diagnostic_limit`，`certified=false`。当前结论是contract/接口正确但行为未恢复；v450仍为历史迁移源，v456不作发布策略。

v457接续实验（启动前）：不再重复v456同预算训练。由v450派生 `artifacts/continuation_v457/migration_source.pt`，源SHA `a9feac926c66f657ba6e1df5d4138d706f972096d01858096a60e7d210082dd4`，派生SHA `c8bfb90e8fcfc304fd79699e0e18268ca16fddfc488c60414bd98885d3e43812`；唯一变量是physics residual腿预算 `[0.15,0,0.35,0.15,0,0.35]` 改为 `[0.681818,0,0.35,0.681818,0,0.35]`，轮预算不变，冻结源和执行器契约不变。`0.681818=0.15/0.22`，用于对齐已有position-authority腿限幅；该迁移会改变residual输出，按新派生源做retention，不能把它当作v450行为等价。训练stop condition为非有限值/冻结审计/仿真故障立即停止，或完成25 updates后只做快速12点网格；若无正向+1.0/+1.5转弯或±4自转改善，直接标记未晋升，不升DR、不做无关续训。


## 2026-10-02 v427：真实DR=.05完整八点评估，+1.0通过但+1.4生存未过

已完成当前目录接续核查、45项recovery/profile/peak测试及4项controller测试；硬件profile测试仅纠正旧D断言(.005/.0225→.002/.021)。目标硬件配置与rmuc_peak55.xml编译/实际饱和审计通过，不代表行为认证。v424仅日志6760–6763与model6760，状态complete不证明25更新完成；直接stage49使退火fraction从第一轮即1，实际D=.05。旧“无checkpoint”和“完成25更新”都与磁盘不符。

本轮唯一GPU实验为v399/model6810（SHA a9a929093b568252dc53122277f785e38cde0d38deef603f3ce29ab79e932bf0）的评估，无训练。5.5峰值/scale5.5、3持续参考、D中心.080625、仅damping/gain DR=.05，理论D[.078609375,.082640625]、gain[.9975,1.0025]、气弹簧阶段1.0775Nm/腿、径向0、任务外环关闭。128env、seed67、3000控制步，八点每点至少32完整回合。v398/v399旧引用报告实际为DR=.1，不追溯重写旧JSON。

结果7/8：+1.0 yaw error=.0451507、stable=.650188、survival1，当前tracking gate通过；+0.8及其余五点通过。+1.4共有33回合、3次tilt，survival30/33=.909091，仅survival gate失败。因此promoted=false，DR=.05整体未完成，不升DR，不基于旧.1失败证据盲目续训+1.0。+1.0 bank=11.5192°且analytic_bank_required=False，最终物理内倾尚未认证。active-leg局部动作限幅1.6818，旧abs(action)>=1饱和统计不证明真实执行饱和；需区分位置、动作及PD力矩余量。

新增带SHA/显式执行器契约的ONNX replay controller并接入MuJoCo可选consumer：同输入1512样本原Isaac actor（CPU恢复）/JIT/ONNX/consumer对照通过，ONNX误差2.384e-7；独立1024样本训练AST观测/consumer对照误差0。两秒MuJoCo CPU集成运行成功仅为接线冒烟，不是正式八点、多seed或实机replay。当前链路沿用阶段1.0775Nm，最终10.775Nm与目标D迁移仍未完成。未产出成功验收demo视频。

证据：artifacts/continuation_v427/current_status.md、attempt_02/{experiment_contract,runtime_verified,evaluation,retention_audit,freeze_audit,failed_conditions,final_report}.json、controller_contract.json、same_input_outputs.npz、controller_replay.json。首次256env启动被128env上限拒绝，原日志保留；重试在独立attempt_02中完成，未覆盖历史。v278/model6684与v379/model2562保持原SHA，恢复0/5/15局部成绩不扩展为普通倒地恢复完成。


## 2026-10-02 v428/v429：+1.4局部训练退化，镜像记录与评估条件已核对

v428从v399/model6810、同5.5峰值/scale5.5、D=.080625、damping/gain DR=.05、阶段气弹簧1.0775Nm开始，完成25更新，仅expert3的20个张量改变；35%旧正向锚点rehearsal，主图冻结。最终model6835 SHA c86560284e90485e376ebbb0213b032c3f2c322218e4416236790be627a49d2b。完整八点各>=32回合仍7/8，七保护点指标与v427逐值一致，但+1.4生存30/33→29/46（17次tilt），coverage也失败。因此确认局部训练退化，候选拒绝、promoted=false，保留v399/model6810，不追加同配置轮数。

用户质疑“原+1.0失败/+1.4存活为何互换”已核对原JSON：v398/v399八点筛查实际DR=.1且+1.4仅1回合；v399单点32回合+1.0在.1失败，而v427同hash在.05通过，是评估条件改变，不能宣传为新学习收益。更早v278/v335的+1.4确有32回合全存活，但契约是4.875峰值、DR0、径向2.69375N、等效助力矩0，不与当前混用。当前+1.0没有再次退化，v428真正退化仅+1.4。

v425修复的是镜像诊断probe漏交换wheel velocity/history槽位，复用train.py镜像函数，未生成修复策略checkpoint。v399与v427共有22个源码/配置/资产SHA完全一致，无证据把行为变化归因于镜像修复。v421历史错误probe不再作为物理根因依据，不重跑导入train.py会启动sim的mirror_by_yaw.py。

v429两组450步、相同128env/seed/独立DR协议，仅更换source/candidate。初态、D/gain与材质在pair内一致；source无失败、candidate一例env96在199步tilt。每10步保存的轨迹中，41步pitch已>10°，111步轮请求才>5.4Nm；目标yaw重构最大误差8.35e-8，支持起步俯仰协调退化，不支持镜像/命令误路由是该条轨迹先发原因。未分离具体动作通道因果，不能当唯一根因已证明。

注意WheelDRStreams依赖reset_budget=steps+2，450步pair并非3300步完整评估的同一DR序列，禁止跨预算强称同抽样或以短时无倒认定通过。完整证据：artifacts/continuation_v428/{final_report,decision,freeze_audit,retention_audit}.json；artifacts/continuation_v429/{comparison_and_mirror_review.md,analysis.json,source.json,candidate.json}。当前DR=.05整体未完成，最终10.775Nm与D目标范围/最高DR/部署均未完成。


## 2026-10-02 v430：critic warmup方案取得阶段DR=.05八点tracking通过

从保留model6810开始，总25更新，唯一训练协议变化critic warmup0→10（10次critic-only、15次expert3 actor更新），expert实际LR1e-4而非日志baseLR2e-6。第10次checkpoint与source的整个actor逐值一致，最终仅expert3的20个张量变化。候选logs/rsl_rl/rmuc_wheel_leg_course/2026-10-02_13-46-33_positive14_criticwarm10_dr005_v430/model_6835.pt，SHA5960533edf88c8a08a564ab86ca19e679faa49b44074f6cb3ba9873cfa1ad502。

完整八点每点32回合，8/8 row checks通过，全部survival1；+1.4 yaw=.04041836、stable=.75533027。七保护点指标与v427完全相同，初态/D随机化审计相同。说明这个训练协议产生了有效候选，不足以唯一证明critic失配是全部根因（actor更新数也从25变15）。

保留原evaluation.json整体passed=false：evaluator对diagnostic profile、damping_torque_only DR、关闭外环诊断标志及TGS诊断入口刻意不给总体认证。只认定阶段1.0775Nm气弹簧、D=.080625、5.5峰值/scale5.5、damping/gain DR=.05、seed67的八点tracking通过，选择为下一课程起点，不宣称10.775Nm最终动力/物理内倾/最高DR/多seed或部署通过。中断后发现worker退出但完整evaluation.json已写出，已核验SHA/全部回合并补全最终报告，未重跑训练。证据artifacts/continuation_v430/{completion.md,final_report.json,warmup_audit.json,runtime_verified.json,retention_audit.json}。

新增可选--wheel-dr-reset-budget，让短诊断显式匹配长验收的DR预分配；默认行为不变，本次3302与v427原默认一致。两项DR流测试通过。并行CPU新增locomotion命令时间表模块与MuJoCo接入，13项schedule/controller测试通过，但尚未新增完整动态物理验收；不把代码接入当行为成功。用户最新要求暂不推进recovery。后续训练/评估期间由脚本监控异常，代理等待进程结束再做结果分析，不逐轮重复读取日志。


## 2026-10-02 v431–v433：DR=.1适应未通过，保留.05；后续尽早转低强度联合DR

v431固定v430 checkpoint评估DR=.1：八点各32回合全存活，4/8 tracking通过，四个正向点yaw失败。v433从同source进行10critic-only+15actor更新，仅4个正向expert共80张量变化、其他图冻结。DR=.1仍4/8，正向yaw误差约.05983/.06177/.05583/.05937；+.8和+1.4的稳定指标变化不同，不宣称全部改善。DR=.05 retention八点各32回合全存活、8/8通过。promoted=false，不升.2；保留v430通过基线和v433未晋升候选。

v432同一冻结v430完整导出CPU检查通过，但MuJoCo固定八点12秒诊断全部yaw跟踪失败，不能向动态切换/成功视频或部署证书推进。全部命令、SHA和指标在../continuation_v432/。


用户指出仅damping/gain跑满后再加入传感器会重复适应，确认后续不再延长孤立课程到1。以目标damping迁移为主，尽早联合低强度IMU/encoder误差、独立观测/通信延迟和电机响应；10.775Nm主要动力迁移也不拖至旧动力满DR以后。实际噪声按strength缩放，延迟是按概率触发1–3周期，并不是时长乘strength；须记录分因素实际范围。详细当前结果与新顺序见artifacts/continuation_v433/completion_and_next_course.md。


## 2026-10-02 v434/v435：旧技能MuJoCo扩展复核与独立惯量DR开关

用户要求质量/惯量等联合DR，并将旧技能加入MuJoCo命令复核。当前质量DR原先仅root mass与inertia同比缩放，新增max_base_inertia_fraction（默认0）作为独立scalar shape乘子，保持旧RNG与默认物理数值；8项CPU测试通过，尚无原生GPU开启/行为认证。评估新增实际mass/inertia读回。COM/主轴比例/载荷空间分布/云台反作用仍缺，速度push不等于40–400N力脉冲。

多速度MuJoCo先修复动作尺度：从source params构造与Isaac一致的speed/leg_scale knots，每policy tick按绝对命令速度float32插值；1601速度对照真实训练函数误差0，15项controller测试通过。未继续使用所有速度固定.22的旧consumer。

v434使用同v430/model6835（SHA5960533e…）与v432 ONNX，在5.5峰值/3持续参考/D.080625/阶段spring1.0775Nm、外环关闭下，实际CPU跑42条旧命令（39个v181旧passed点+3补充点）：16条完整12秒、26条提前停止，失败全保留，不宣称sim2sim通过。历史外环缺字段记unknown，历史bank符号/动力与当前不同，不把旧passed当当前E2E掌握。详细原命令、门槛、来源、逐点指标与hash：artifacts/continuation_v434/{historical_commands.json,commands.json,summary.json,report.md,SHA256SUMS}。没有生成成功demo视频。


## 2026-10-03 用户确认：最终物理范围与全速度联合学习

最终“20%”指质量、惯量、摩擦等主要参数相对标称±20%，不是现有DR范围的strength=.2。D仍覆盖[.002,.04]；观测与通信延迟分别1–3个8ms周期。轮5.5峰值/3持续/scale5.5、腿PD60/2/40、10.775Nm等效弹簧与径向关闭保持。当前独立惯量shape系数乘质量系数，若两者各±20%，总惯量会到.64–1.44，不能误称±20%；最终总惯量应相对标称直接约束。COM与云台等尚缺因素仍要单列。

续训全速度压弯直接带联合DR学习，不先全域nominal学完再重做DR；新命令域从可学习强度渐进扩大，旧技能rehearsal/retention保留，最终同checkpoint完成全域多seed/联合DR/sim2sim验收。用户明确当前不推进recovery。训练等待期间由脚本监控、结束后再分析，不能拿中间小强度代替最终证书。明确口径保存在artifacts/continuation_v438/confirmed_final_requirements.json。


## 2026-10-03 最新用户确认：轮腿独立执行周期与被动损耗

本节覆盖先前将有效D范围当机械阻尼、以及轮腿统一2ms的目标描述，历史实验数值保持原样。轮端执行周期1ms，腿PD仍2ms（kp60/kd2，±40Nm），policy8ms。若physics1ms，腿PD每两步更新、期间保持电机力矩；只保持位置目标而每步重新计算PD不等价。

机械被动黏性先验b=.002Nm·s/rad，目标不确定范围[.001,.005]；库仑力矩先验.001Nm，均非实测辨识。反馈D独立、绝对值未知；不从不对称力矩差值推定。电机安装符号尚需实机核对。无反馈诊断可显式选D_fb=0，但不得声明硬件反馈已确认为0。

电机目标5.5*clip(wheel_action,±1)，电磁输出限幅±5.5；持续3Nm只作热/长期负载参考。机械损耗−b*omega−tau_c*tanh(omega/omega_eps)在电机限幅及响应之后作用于动力学，合力矩无外层电机clip。omega_eps=.1rad/s目前仅数值平滑诊断值，不是辨识参数。被动反力矩同时作用于轮与父连杆，不能只给轮加外部力矩而漏反作用。最终等效气弹簧10.775Nm/腿，径向力默认关闭；历史1.0775Nm阶段仍是未完成迁移。

旧checkpoint保留作迁移起点/retention基线，不作废；旧动力下的通过不能迁移为新契约认证。新模型需要原生接口读回、同输入跨引擎对照和重新行为验收。当前不推进recovery。

用户追加授权：允许共用GPU。其他工作区训练保持原样，本项目新进程先核对显存/负载并限定预算；不可继续把GPU被占用本身当成阻塞条件。先完成轮1ms/腿PD2ms/policy8ms的原生接口短验证，再决定迁移训练。


## 2026-10-03 v443/v446：轴向错误使新动力结果失效

v443完成25更新（10critic-only+15新残差），源784tensor映射冻结、初始化输出相同及warmup审计成立。候选model6860 SHA2ac906d0ea5db23940a8146fffec8ffebf5da14e98c89e9577f55fa820297f07。旧DR=.05 retention八点各32回合/8通过/全存活；但不得晋升为新契约通过。v444同候选JIT误差0、ONNX最大2.98e-7，MuJoCo两命令12s存活但跟踪失败。

独立只读核查发现本机IsaacLab3 body_quat_w采用xyzw，而新增wheel passive轴转换和历史gas_spring helper按wxyz解释。实际USD关节父局部+X经过错误转换变到错误世界轴；MuJoCo xmat路径正确，不能修改其wxyz约定来迎合错误。原v440 native测试只验证提交与buffer自洽，不能证明正确关节轴，其物理有效性结论撤销。v443新动力训练/评估标invalid_physical_axis_not_certified，已停止仅本项目GPU0新评估，保留日志/候选/旧retention证据。

修复策略：新separated模式轮与气弹簧明确使用Isaac xyzw；历史legacy spring保留wxyz路径并明示为旧数值复现，绝非正确物理。新检查必须用安装Isaac API+实际USD轴作为独立oracle，并核对真实刚体相对角速度功率。修复后重新做短native验证和有界迁移，不重写历史报告为通过。最高DR和正式sim2sim仍未完成。


## 2026-10-03 v448 修正轴后完整结果：未通过，不升级DR

v447原生独立API/USD轴与真实功率检查通过后，v448从原v430的等价冻结迁移源46010fbe38f73a883297905526f4c6853b67c6842cf31deb1a69e8440ac19517完成25更新（10critic-only+15新残差）。最终model6860 SHA6b1f0526549938928a81825a869bd1c65c4c682aececb6cb7a5cb9dd83ff0608；仅新增物理残差6个tensor改变，旧actor冻结及warmup审计通过。

新separated契约（xyzw轮/气弹簧，轮1ms/腿PD2ms/policy8ms，b=.002/Fc=.001/反馈D=0诊断，5.5峰值/3持续/scale5.5，阶段弹簧1.0775Nm，DR0）完整八点至少32回合：0/8。正向+.8生存约.0044，+1/+1.2/+1.4生存0；负向-.8/-1/-1.2虽全部存活但yaw误差约.2094/.2169/.2943，-1.4也失败。旧legacy DR=.05 retention八点各32回合均存活、8/8通过，仅证明旧条件保留。promoted=false。

同candidate CPU导出parity通过；MuJoCo直行12s后半均速-.03893m/s，yaw+1均速.32843m/s/yaw1.35619rad/s，跟踪失败。不能把能跑完写为sim2sim合格。训练、双GPU评估和CPU流程均已完成，无需重训本轮；完整证据artifacts/continuation_v448/{final_report.json,evaluation.json,retention_legacy_dr005.json,cpu/summary.json}。

历史DR=.1（v431/v433）为八点各32回合全存活、tracking4/8，四个正yaw失败；通过8/8的是旧DR=.05 retention，并非.1。最高DR不能跳级。v445已准备非对称机械b与独立总惯量DR纯函数/16项CPU测试，但尚未接env，不能宣传目标DR覆盖。后续优先用匹配物理条件定位移除旧feedback D及修正spring轴后的迁移缺口，不盲增同配置更新或升级DR。


## 2026-10-11 命令域扩展：修复 dense yaw reward 覆盖缺口

上一批 positive_bank_ppo/model7950 快筛40/56：旧33/33、新7/23，全部存活，但1.25/+0.6 yaw误差0.6011、stable=0；promoted=false。差分审计512条早期真实obs显示轮残差未饱和且差分梯度非零，不能归因为轮残差clip完全阻断。未做成功声明。

代码核对发现：low_speed_yaw_cost硬编码|yaw|<=.4501，对新目标1.25/.6为零；另一项强化yaw项限于.8<|vx|<1.2，同样未覆盖。普通yaw奖励仍存在，这只是已验证的覆盖缺口，尚非唯一根因。已新增command_small_turn_dense_yaw_max_rad_s，默认.4501保持旧行为；对照实验仅改.6001，固定原seed、60更新、512环境、spring4.31/DR.1及所有物理参数。20项相关测试通过，GPU0启动pipeline PID1544370，训练结束自动56点快筛；尚未有本实验行为结论。

证据目录：artifacts/teacher_course_20261009/low_speed_domain_20261011/positive_bank_dense06_ppo。source SHA fd036ec02c4ca6c2ab70a55e6ed76bb36de3bc7f5ab8676c169d4db4f6cff3c4。预算3600秒，契约/冻结/非有限失败停止；不自动升DR、弹簧，不触碰vy/导航，不开展学生或sim2sim。最终全域、严格速度/阶跃/圆弧、多seed及最高DR仍未完成。

启动更正：首次进程1544370在参数预检退出，无训练/无checkpoint；新增Hydra参数误置于nargs physics_train_modules后。失败证据完整保留startup_failure_01，修正参数位置后重启pipeline PID1545063，未改变实验变量。

重启实证：PID1545065已进入PPO更新；runtime_verified=true，实际env.yaml读回dense yaw weight=.5、max=.6001（reward_config_verified.json）。训练尚在进行，未宣称行为改善。

2026-10-11 dense06对照结束：60更新610秒，checkpoint SHA d1d96ddf0cd1a457492f87f0b15bed2ac07f37fb80ecebae85c395a2a6ca9a82；56点快筛40/56、旧33/33新7/23、全存活，1.25/+.6 yaw误差.6011456 stable0，promoted=false。单独奖励范围修复没有行为改善，不继续盲加轮数。验收首次因旧输出路径被保护拦截，失败保存evaluation_startup_failure_01；修正路径后178.97秒exit0完成，本批checkpoint hash对齐。启动positive_turn_differential_probe，冻结seed，三速度七档轮差分含零对照，84环境3200步最多600秒，仅动作响应诊断，不是策略晋升。

轮差分响应诊断完成exit0，21条件均存活；1.25/.6的diff=-.15将yaw误差从零对照.6007降至.1898，但速度误差.3127；1.5/.6对应yaw.1279、speed.4077，均stable0。说明能产生转向但共同速度/姿态未满足，不能作为策略通过；不同case DR抽样不配对。已验证实际物理契约，准备common+diff小网格，先验证可达动作，不盲目继续PPO。

common正偏置诊断完成：17条件均无stable通过，4个高偏置条件survival0，已排除。其余存活组合的共同正偏置使欠速加重（同diff=-.15，common0/.03/.06的speed MAE=.3205/.3839/.4549），故“加正共同量补速”推测被否定，不可用作训练示范。准备负共同偏置-.06/-.09/-.12与diff-.14/-.16/-.18，仅40env可达性验证，零对照保留，不更新策略。

负共同量诊断完成：common=-.09/diff=-.16速度MAE .0701、yaw .1869、survival1但stable0/bank7.2203deg，仍无合格组合；-.12/-.14存在倾倒。停止常偏置网格。CPU authority_transfer_audit（512真实早期obs）确认1.25/.6的position authority gate为0，consumer也不开放扩展腿幅；仅将该head的vx输入置1时gate1，但这不证明单一根因或行为成功。下一步回到已掌握yaw±1/±1.4，先以vx1.025/1.05/1.075定位相邻可迁移边界，不再直接重复1.25/.6的60更新。汇总expansion_diagnosis_summary.json，当前所有新候选promoted=false。

2026-10-11相邻速度边界快筛16点完成：1mps四个控制点全过；1.025mps ±1通过、±1.4未过；1.05与1.075的±1/±1.4全部未过，全部存活，主要yaw/stable缺口。启动adjacent_v105_positive1_ppo，源fd036ec与7890等价且旧域隔离，仅修复1.05/+1而非重试1.25/.6。512env60updates，物理spring4.31/DR.1固定，训练腿映射.225与原插值验收一致；旧33+新12自动快筛。冻结审计与evaluation零更新审计分文件保存，避免报告脚本覆盖训练freeze证据。PID1577320，尚无新训练结论。

adjacent_v105_positive1_ppo第11更新真实rollout审计：目标1.05/+1样本14080条，取2048条计算该隔离network的实际policy-gradient norm=2.8278、finite=true；冻结参数无变化。证明目标采到且有梯度，不代表技能通过。rollout_gradient_audit.json已保存。

2026-10-11 adjacent_v105_positive1_ppo训练60更新665秒完成，SHA3deadd613050f6fd9ab5f771511c0c755f50dc3797b39ab3cdb46b689c15aa53；45点快筛35/45，旧32/33、新3/12、全存活。新1.05/+1通过：linear .07367/yaw .02913/stable1；1.025/+1也通过。旧1/+1.4在90env布局yaw .06074/stable .38963未过，不能忽略，也不能跨布局判遗忘。promoted=false，准备原7890同45点90env对照，仅checkpoint不同。

用户要求停止低收益逐点磨速度并对照历史扩域：同45点90env原7890与新7950的旧33指标逐项完全相等，均32/33，1/+1.4的yaw=.06073960014、stable=.389632107为原有条件缺口，不是本次遗忘；仍不晋升。historical_expansion_root_cause_review.json核对v188的2.4±.3、DR.05八点通过，旧D=.1/无弹簧/保留速度pitchyaw反馈，与当前E2E不可直接等同。历史v178/v179突破包含mesh接触细分修复；当前仍10240faces/.5材质，未发现该修复丢失。v144/v149的3.2网格失败，不把课程上限冒称掌握。当前1.25/1.5 authority和consumer扩展腿幅关门，1.25左active持续clip；dense reward范围漏项修复单独无效。暂停单点0.025速度续训，先处理新域可训练路由/动作空间的一致性；不恢复任务外环、不改vy接口。

2026-10-12 authority_support15_probe完成：仅加speed_plateau_end=1.5 buffer，source7890旧张量全等、33命令各1024状态raw/executed parity0；29项测试通过。actor/local authority/Isaac consumer读取同checkpoint参数，旧默认1.0；扩展候选训练与导出暂时拒绝。闭环51点旧32/33、新0/18、全部存活，故单扩authority被证伪为充分修复，promoted=false。layer_audit.py/json在512条同状态命令替换定位遗漏上游：rmuc_state_router在vx1/.6轮差分约-.4428，vx1.25/1.5严格0；原turn expert约+.039，physics后最终差分约+.022/.029。state_router.py的extended_turn_eligibility仍硬编码(.8,1,1.2)，本次未改，说明先前只归因最后authority不完整。下一步针对实际承载转向的state-router分支作隔离复制/连续域训练，保留旧域；不再重复单独authority扩门或常偏置网格。

2026-10-12上游router支撑诊断突破：router_authority_support15_probe在authority_support15基础仅增加state-router turn_speed_plateau_end=1.5，所有旧权重不变，candidate SHA77d7b9d65a56fb9b6eedda9244c1db03f5442eddd32bfaa5fe33e2984b4fb4b3。22项测试通过，旧33各1024输入raw/executed完全一致；扩展候选禁止训练/部署导出直到隔离分支实现。51点同布局快筛新18从0→6：1.05的±1/±1.4、1.25的-1/-1.4全通过；1.25/-1 linear.1075/yaw.0322/stable1，1.25/-1.4 .0531/.0062/stable1。旧32/33同原近门槛缺口；1.25/+1.4 survival.5、1.5/+1.4 .75，必须明确失败，promoted=false。物理读回通过，matched_comparison保存，不能当完整32回合或多seed认证。这证实主要转向来自上游局域router，过去只训练physics残差/扩authority遗漏关键支撑。后续隔离复制转弯router并按方向/速度成片适应，不再逐0.025磨点；spring4.31 DR.1固定、vy/导航仍撤销。

2026-10-12隔离速度扩域副本准备：rmuc_state_router增加speed_extensions ModuleDict，仅复制7个既有转弯分支，原参数冻结，vx<=1输出保持旧分支、1→1.05平滑混合，在原动作预算内凸替换。真实完整checkpoint8192输入与上一诊断source输出精确一致且保存恢复一致，所有旧tensor相等；新seed SHA9f081b87557eaf6752452aea10f83e2d6f98abeb67ff7470befc3211ebe7cb84，仍继承正向大yaw倾倒，promoted=false。1.25/+1的正向副本梯度norm270.236非零，不能替代行为验收。新增optimizer API只选正/负副本，旧参数与反向冻结；27项测试通过（含实际Adam更新后旧域/反向输出不变）。尚未接CLI/目标门控和连续采样，训练禁用保护保持，不启动半接通实验。记录router_isolated_speed_seed/preparation_audit.json及implementation_status.json。

2026-10-12 router_positive_domain_ppo已真实启动PID2306921：CLI方向选择、纯CPU目标门控、冻结审计、inactive Adam momentum保护和signed连续域采样已接通，30测试通过。source9f081b87557eaf6752452aea10f83e2d6f98abeb67ff7470befc3211ebe7cb84，仅正向扩域副本/critic可训练；512env120updates/10warmup/2400秒，50%旧域复习+25%vx[1.05,1.30]/yaw[+.6,+1.4]连续域，固定锚点1.25/+1，spring4.31 DR.1和原动力固定。已进入PPO更新、runtime_verified=true，配置读回sampling_config_readback.json；结果尚未知，promoted=false。结束自动51点快筛，正式旧域完整>=32回合后才可晋升。当前已撤销vy/导航目标继续不启用。

router_positive_domain_ppo第11更新rollout实证：65536样本中固定1.25/+1为7552条，连续目标框非anchor16384条（25%）；vx×yaw 5×4共20格均有样本，选定正向副本实际PPO梯度norm4.2225、finite=true。此为早期覆盖与梯度证据，不是稳定验收；rollout_audit.py/json已保存，冻结审计当前无违规。

router_positive_domain_ppo中段model7910 CPU验收：仅80个正向speed_extensions参数tensor变化，unexpected0；旧33+新速度负向9共42命令、每条1024真实状态重置命令，完整actor输出差严格0。midpoint_integrity.py和mid7910_integrity.json已保存；这仅证明隔离，不替代闭环行为或最终retention。原训练PID2306921继续固定120更新预算。

2026-10-12 router_positive_domain_ppo最终120更新1240秒exit0，model8010 SHAb3efee8025c3ee17408cecfd87f58144649d7483c9f2b6747b0c89be2765cd09；51点快筛39/51，旧32/33、新7/18，promoted=false。新增1.25/+.6 linear.0688 yaw.036 stable.9055、1.25/+1 .0977/.0242/1、1.5/+1 .1352/.0351/.8269，均存活。新域回退1.05/+1和+1.4必须计入（虽旧33未丢）；1.25/+1.4 survival.75、1.5/+1.4 .3333。已修正final_report原模板空lost_conditions/survival_regressions为实际来源对照，expansion_retention_audit明确失败。新成功锚点未加入固定rehearsal，连续均匀采样对边界1.05不足；下一轮先补新域锚点复习/参考输出约束并分开处理启动倾倒，禁止同配置无脑续120或升DR/弹簧。

2026-10-12 router_anchor_handover_probe：只增加checkpoint blend_bounds=[1.05,1.15]保护旧扩门1.05锚点，同时保留训练后>=1.15输出；25测试及每命令1024同状态输出对照通过，source权重未变，candidate SHA9ee29f40f0ceab58d84d9160e43809ed972bb648c060d61855a277f1120285e6。63点126env快筛旧30/33、新16/30，1.05/+1,+1.4恢复，1.10/+1、1.15/+.6,+1及1.25/+.6,+1、1.5/+1通过；但1.10/+1.4 survival.5、1.25/+1.4 .3333、1.5/+1.4 .0952，1.15/+1.4跟踪未过。不同布局不能推断旧策略被破坏；不能忽略旧30/33或倾倒，promoted=false。后续冻结速度上限，收集高yaw失败前状态/动作，并加入已过新域固定复习，不继续扩速度/DR/弹簧。

2026-10-12 yaw1.4 failure-trace诊断完成（48env，1.05/1.10/1.25各16rep）：1.05通过survival1；1.10 survival.7778、1.25 .3913。采集首16事件，去重9env首次倾倒在1.04–1.688秒；pitch首次>=10deg中位.328秒、此时vx约.2但yaw约1.2，说明主要表现为起步pitch失稳，不能按幸存者稳态yaw通过。真实动作同观测回放最大误差1.79e-7；示例早期腿/轮不在限幅，不断言唯一根因。early_pitch_reward_audit仅量化三项处罚：这9状态pitch_excess15deg及通用pitch-rate1.5门槛均未触发，turn pitch-rate预警因start level对应速度2.4而不启用，三项全0；不等于全部reward无信号（pitch target项等未计）。下一步可单独开启低速大转弯早期预警并加入新域固定锚点复习，保持D/弹簧/外环不变；尚未新训练。

2026-10-12 router_pitch_rehearsal_ppo启动PID2347105，已进入PPO且runtime_verified=true。source9ee29f40f0ceab58d84d9160e43809ed972bb648c060d61855a277f1120285e6；组合干预而非纯reward消融：目标1.25/+1.4、turn_pitch_rate_start_level0、连续队列半数替换8个新域成功锚点（全环境12.5%锚点+12.5%连续，50%旧rehearsal）。新增纯采样helper参数与环境配置/接线，26测试通过；actual env.yaml参数读回intervention_readback.json。仅正向副本/critic训练，120updates/10warmup/512env/2400秒，spring4.31 DR.1及全部动力不变。训练后自动63点，同布局对照显式lost_conditions/survival_regressions，promoted=false；未宣称起步问题解决。

2026-10-12真实rollout发现锚点取样bug：router_pitch_rehearsal_ppo第11更新8锚点虽全部出现但共仅约1104/65536样本，1.5/+1仅3条。每次reset round(count*.5)对singleton恒0导致锚点逐渐消失。已停止pipeline自动链并SIGINT trainer2347105，250秒exit0（约23更新），保留checkpoint/冻结审计，final_report标stopped_invalid_anchor_sampling而非训练完成。helper改为floor+小数Bernoulli无偏随机取整，新增2000次singleton比例测试，27测试通过。修正版router_pitch_rehearsal_fixed_ppo从相同9ee29源启动，只有该采样修复，120更新/2400秒预算和63点验收不变，物理spring4.31 DR.1固定；优先读真实覆盖，不凭单测声称实际比例达标。

2026-10-12锚点sampling修复实证：router_pitch_rehearsal_fixed_ppo第11更新65536样本内7859锚点（11.992%，目标12.5%），8个全部出现；此前有bug批1104条，1.5/+1从3→1280。冻结审计无违规。证据anchor_rollout_audit.json；实际时长/观测误差影响占比，不宣称精确人口配额或行为通过。原PID2352967持续运行，未重复启动。

2026-10-12修正锚点批中途model8050再次审计：仅80个正向speed_extensions张量变化，unexpected0；旧33+负向扩域9共42命令各1024状态完整actor输出精确不变。mid8050_integrity.json留证，仅证明隔离，不能代替行为验收。PID2352967继续原120更新预算，目标stable训练EMA仍波动，尚无最终结果。

2026-10-12修正版120更新1260秒exit0，model8130 SHAa9661a0d1a73bd7fe699bafa20fa5c30633c1eb734a95a9c9ab563960860f9de，63点快筛49/63、旧31/33、新18/30、全部存活。新增1.10/+1.4、1.25/+1.4、1.5/+.6通过；1.05/+1.4未过，1.15/+1.4和1.5/+1.4仍失败。最终CPU审计只80正向副本tensor变化，44受保护命令（含1.05/+1.4）各1024状态精确同输出，因此不能把其失败直接说成权重遗忘；终止/重置影响抽样，仍是必须记录的行为缺口。候选promoted=false。启动formal_positive8，固定8130，8正向关键点各>=32回合/128env/4800步/600秒，检验起步无倾倒与跟踪，不代替旧33正式retention，暂不新训练或升DR/弹簧。

2026-10-12 model8130关键8点正式回合验收完成，实际3000步每点32回合（4800上限提前满足），同SHAa9661a0d1a73bd7fe699bafa20fa5c30633c1eb734a95a9c9ab563960860f9de及物理读回通过。旧stage6/8通过，但无观察失败+速度<10%仅5/8：1.05/+1.4、1.25/+.6,+1,+1.4、1.5/+1；1.25/+1.4 speed.02993 yaw.02958 stable.9521、32/32存活。1.10/+1.4为31/32、tilt1，stagepassed也必须记失败；1.15/+1.4 yaw.06947 stable.0929、1.5/+1.4 yaw.19793 stable0，均32/32存活但tracking失败。formal_positive8/final_report.json采用显式no_observed_failure，promoted=false，旧33正式retention/多seed/最终动力DR仍未完成。不盲增120更新，下一步修过渡区瞬态及1.15跟踪并保留新锚点，再处理1.5高yaw。


## 2026-10-12 用户确认课程顺序

先稳定掌握1.5m/s及以下原vx+yaw全目标命令域（含边界、双向压弯、中间速度、起停/动态切换和retention），随后继续渐进弹簧迁移至10.775Nm/腿，并逐档复验同一checkpoint技能保留。当前spring4.31/DR.1固定，门槛未通过；不以少数点通过、平均reward或快筛无倾倒代替全域验收。用户已授权全域通过后继续迁移，无需再次询问。vy/导航/75格要求仍撤销。

2026-10-12按用户确认先低速全域后弹簧推进：router_transition_rehearsal_ppo从8130（SHAa9661a0d1a73bd7fe699bafa20fa5c30633c1eb734a95a9c9ab563960860f9de）继续，仅课程采样重点转1.15/+1.4、映射.235与既有插值一致；新固定锚点加入1.10/+1.4和已32回合通过1.25/+1.4，原pitch shaping不变。512env80updates/10warmup/1800秒，正向副本隔离，旧50%+新锚点12.5%+连续12.5%，结束63点对照8130，spring4.31 DR.1固定。已实际启动，尚无行为结论；完整证据在实验目录，promoted=false。

router_transition_rehearsal_ppo第11更新真实rollout：固定1.15/+1.4样本8688，10锚点合计7500/65536（11.44%），冻结无违规；但1.25/+.6该快照样本0，不能将总比例达标当作每锚点覆盖。anchor_rollout_audit.json保留零项，最终63点必须检查该技能，暂无晋升。

2026-10-12 router_transition_rehearsal_ppo80更新850.1秒exit0，model8210 SHAc4c36cbb33ee02695d0d93b02fe3129e8a9720a4b6c200a1d61cbcff4f80e8fd；63点快筛50/63、旧31/33、新19/30、全存活、无已过条件丢失或survival回退。新增1.15/+1.4 linear.0091/yaw.0128/stable1；1.10/+1.4 .0229/.0154/1，1.25/+1.4 .0386/.0125/1；1.5/+.6和+1保留。1.5/+1.4 yaw.1364/stable.1162仍失败，1.05/+.6及负向-.6、负向1.5仍缺口，promoted=false。不得把19/30或当前63稀疏网格视为完整最终域；下一步独立负向副本适应并冻结正向进展，随后完整32回合/旧域/动态等验收。按用户最新顺序spring4.31继续冻结，<=1.5全目标域稳定验收后再迁移弹簧。

2026-10-12负向域适应启动router_negative_domain_ppo：source8210 SHAc4c36cbb33ee02695d0d93b02fe3129e8a9720a4b6c200a1d61cbcff4f80e8fd。报告显示-.6为overspin约-.27至-.39，1.5同时欠速，并非正向缺乏转向的同方向问题。仅负向speed_extensions/critic训练，目标1.25/-.6，门控[positive0,negative1]实测通过，连续vx[1.10,1.5]/yaw[-1.4,-.6]，旧50%+负向已过8锚点12.5%+连续12.5%；1.05冻结边界仍不能被此分支改动，作为缺口保留。512env120updates/10warmup/2400秒，当前已进入PPO且runtime通过，物理spring4.31 DR.1不变，结束63点比较8210并核验正向保留。尚未行为验收/promoted=false。

负向router_negative_domain_ppo中途model8250隔离审计通过：仅60个负向speed_extensions张量变化、unexpected0，旧33+正向扩域9共42命令各1024状态输出精确不变。mid8250_integrity.json保存；中断后PID2408588实核仍运行，未重复启动。训练目标yaw EMA约.097仍未通过，不能只凭改善晋升。

2026-10-12负向router_negative_domain_ppo120更新1225秒exit0，model8330 SHA89890c30a1306883cdacc291197006a6ed169df8c1454e77df25c4099e63fe17；63点快筛52/63，旧31/33、新21/30，全存活，无已过条件丢失或survival回退。新增1.15/-.6 speed.0085/yaw.0246/stable1及1.25/-.6 .0523/.0112/1；负向-1/-1.4旧锚点保持。1.05/-.6门控边界、1.10/-.6部分混合仍失败；1.5三个负向欠速.201–.255/yaw.073–.304仍失败。promoted=false，不把4回合52/63视为全域。下一步先核对1.5动作/分层输出和速度误差，保留正向与新负向，不能原样续120。spring4.31 DR.1固定，用户要求全<=1.5稳定后再迁移弹簧。

2026-10-12负向1.5审计：8330稳态轮/腿applied torque saturation0，active raw>1的值被expanded consumer同值执行，不能将raw saturation字段误读为执行限幅。1.5主要欠速.20–.25并overspin；此前1.5仅连续边界、无固定锚点。启动router_negative_v15_ppo，从8330 SHA89890c30a1306883cdacc291197006a6ed169df8c1454e77df25c4099e63fe17，固定1.5/-1，新增1.5/-.6,-1.4和已学1.15/-.6、1.25/-.6复习，仅负向副本/critic；奖励、D/DR/spring4.31不变。512env120updates/10warmup/2400秒，结束63点比较8330，尚未行为结论/promoted=false，完整预报与参数在实验目录。

router_negative_v15_ppo第11更新真实覆盖：1.5/-1目标8138条，12个锚点均出现、合计7414/65536（11.31%），冻结审计无违规，anchor_rollout_audit.json已保存。仍在原120更新预算内，不能以覆盖充分宣称1.5适应完成。

router_negative_v15_ppo真实rollout梯度复核：固定目标8138样本，20/20命令框格子均有覆盖（包含固定复习锚点，不能当纯随机覆盖），选定负向副本PPO梯度norm2.67045且finite。rollout_audit.py/json保存，训练仍在原预算内，尚无行为通过。

router_negative_v15_ppo中途model8350审计：60个负向speed_extensions tensor变化、unexpected0，旧33+正向扩域9各1024状态完整输出精确一致，mid8350_integrity.json保存。PID2433083仍运行至约46/120更新，目标speed EMA.221/yaw.0554/stable0，欠速仍是瓶颈；按原预算继续，不能据yaw改善升档。

2026-10-12 router_negative_v15_ppo120更新1240秒exit0，model8450 SHAc09c29fd09d35ea05d475d8b1fe9fb21d731e9de283a39ba74ed6a3c0c723b83；63点快筛53/63、旧31/33、新22/30、全存活、无已过条件回退。新增1.5/-.6 speed.13663/yaw.00610/stable.75；1.5/-1 yaw.02301通过但speed.17382/stable0，-1.4 speed.17849/yaw.22198失败。promoted=false，不升弹簧。CPU residual_capacity_audit同早期真实目标状态显示三个负向副本混合输出未近预算上限，不能据此扩大残差预算；speed_reward_range_audit按平均误差核对速度误差项仍有斜率，不能称缺速度reward（并非全reward因果证明）。下一步基于优势/动作变化分析speed-yaw权衡，不原样自动续120。

2026-10-12 negative_v15_action_response完成24条件、全部存活、物理读回通过。冻结8450的小幅动作诊断：1.5/-1零偏置speed.1657/yaw.0122/stable0，common-.02 diff0 speed.1230/yaw.0319/stable1，common-.04 diff0 .0773/.0361/.8604，两条件诊断checks通过；不能算策略自身通过或正式认证。-1.4所有组合未通过，diff-.02可明显减yaw但未过线。优势审计早期共同量方向负、方差大；本次物理响应支持局部-.02作为最小可达教师目标，不加全局外环、不把动作偏置直接当交付。不同offset抽样不配对，下一步以同契约局部拟合/冻结retention获取该修正，-1.4另处理；spring4.31/DR.1固定。

2026-10-12 negative_v15_teacher_capture补采16回合全存活：common-.02 speed.12312/yaw.02478/stable1；零对照欠速.16741。CPU negative_v15_local_fit用真实偏置轨迹拟合deepest physics network，平滑scope vx[1.35,1.45,1.55,1.65]/yaw[-1.2,-1.1,-.9,-.8]、network_anchor冻结，其余source8450全冻结。600步26.65秒，留出轮MAE约.00015、域外输出差0、恢复一致，candidate SHAd8c61f262025387d299f83c62038c752e77a1ba2f2507ef2c5768de8dcf98ba7，promoted=false。已启动63点无诊断偏置闭环验收，尚未行为通过；spring4.31 DR.1不变。

2026-10-12 negative_v15_local_fit无诊断偏置闭环63点完成：54/63、旧31/33、新23/30、全存活、无已过点回退；新增1.5/-1 speed.130315/yaw.029406/stable1。证明局部拟合的修正进入网络并在快筛生效，而非外部偏置通过；candidate d8c61f262025387d299f83c62038c752e77a1ba2f2507ef2c5768de8dcf98ba7。runtime/hash/freeze/retention证据完整，promoted=false，4回合不替代32/multiseed/全命令域；-1.4及低速过渡/旧域缺口仍待修复，spring4.31 DR.1继续固定。

2026-10-12 1.5/-1.4可达性细化：negative_v15_yaw14_fine_capture7条件及bracket_capture4条件全部存活但无组合通过。零offset overspin约-.23；diff-.025后变underspin+.08，夹逼common-.01/diff-.0175,-.02,-.0225的signed yaw约-.052/-.029/+.031而MAE仍.083/.074/.072，stable不足，常量bias不能可靠消除跨状态/环境误差。两批完整观测动作已保存，失败不作为成功教师。停止常偏置网格，保留已54/63的d8c61候选；下一步依据yaw_error_decomposition检查误差分布并训练状态相关修正，不改物理/预算/外环。

2026-10-12 router_negative_v15_yaw14_ppo从54/63候选d8c61f262025387d299f83c62038c752e77a1ba2f2507ef2c5768de8dcf98ba7启动，固定1.5/-1.4并新增已通过1.5/-1复习，负向副本/critic训练、物理残差（含-.02局部修复）及正向冻结，目标门控[0,1]通过。未使用失败常偏置示范，无额外控制外环；reward/动力/spring4.31 DR.1不变，512env100updates/10warmup/2100秒，结束63点比较d8c61源。当前已启动，promoted=false；完整命令、预算、retention、哈希均在实验目录。

2026-10-12 router_negative_v15_yaw14_ppo100更新1040秒exit0，model8550 SHA7e566ed7ea0d416efedb950ae425c7566225b9fcde8d22b03a1eafd34b56745f；63点快筛55/63，旧31/33、新24/30，全存活、无已过点回退。新增1.5/-1.4 speed.12017/yaw.01071/stable1；1.5/-1 .10938/.01848/1、-.6 .12684/.01513/1保持。证明负向高yaw经状态相关训练有快筛改善，但promoted=false，未32回合/多seed/全域。下一步正向1.5/+1.4及低速门控缺口/旧域缺口，最终1mps至2.5rad/s和1.5至1.8rad/s包络仍未覆盖；不得以63网格作为完整目标。spring4.31继续固定，用户确认全<=1.5目标域通过后再迁移。

2026-10-12正向1.5/+1.4适应启动router_positive_v15_yaw14_ppo，source8550 SHA7e566ed7ea0d416efedb950ae425c7566225b9fcde8d22b03a1eafd34b56745f；原目标速度误差.09393<10%、yaw .13638/signed+.12930主要overspin，survival1/稳态力矩饱和0。仅正向副本/critic，目标1.5/+1.4，连续[1.1,1.5]/[+.6,+1.4]并加入1.15/+1.4、1.5/+.6锚点，负向/physics及原图冻结，门控[1,0]通过。512env100updates/10warmup/2100秒，物理spring4.31 DR.1不变，结束63点比较8550保留双向进展；promoted=false，未宣称完整域或开始弹簧迁移。

router_positive_v15_yaw14_ppo第11更新覆盖核对：目标1.5/+1.4有8246条，12锚点全部出现、合计7414/65536（11.31%），冻结违规0；anchor_rollout_audit.json保存。PID2544786仍正常运行，训练刚过warmup，目标yaw EMA约.128/stable近0，无行为通过结论，不重复启动。

router_positive_v15_yaw14_ppo中途model8570隔离审计通过：仅80个正向speed_extensions tensor改变，unexpected0，旧33+负向扩域9共42命令各1024状态精确同输出（含负向1.5新技能），mid8570_integrity.json保存。约31/100更新目标speed EMA.0302/yaw.0450/stable.7061，仍未独立行为通过；保持原预算及spring4.31 DR.1。

2026-10-12 router_positive_v15_yaw14_ppo100更新1045秒exit0，model8650 SHA2a536d0ba6b49217c753642d83cbfcaf4bd554c145f688d934e99066e6f8e79a；63点快筛57/63，旧31/33、新26/30，全存活、无已过点回退。新增1.5/+1.4 speed.01426/yaw.01702/stable1及1.10/+.6；当前网格1.5双向±.6/±1/±1.4快筛都过，但1.5目标至±1.8、1mps至±2.5、动态/多seed/正式retention未完成，promoted=false。低速门控边界与旧2点仍缺口，next_decision列实际失败；spring4.31 DR.1继续固定，全<=1.5目标域验收后才恢复弹簧迁移。

2026-10-12 router_low_yaw_handover_probe准备并启动：source8650仅新增low_yaw_speed_blend=[1,1.05,.6001,.8]，候选SHA928dc36fe8d159597b163642c9ea7a24ab13a8d808795db0f68da3176f549ac9，权重不变。17测试通过；完整actor49受保护命令各1024状态精确同输出、恢复一致。目标验证1.05/1.10的±.6接管缺口，高yaw维持原规则；63点126env3200步600秒，物理spring4.31 DR.1不变，尚无行为结果/promoted=false。


## 2026-10-13 小MLP v8正式实机复盘（未启动新训练）

用户确认实测为v8，轮端阻尼补偿尚未写进部署端；日志/固件后续提供。按用户影响度记录：切换附近单腿突摆突然失稳 > 前进快指令弱响应/顿挫 > 前进腿摆角约8°不对称 > 零命令后溜失稳 > 组合转弯反向压弯/自转顿挫擦地。视频证实可较长时间平衡平移，也同时有垫上和瓷砖翻倒；不能把地面、符号、action撕裂或电池直接定为唯一根因。

复盘文件：sim2real_RL/deploy/reflection.md；五视频SHA/元数据、概览/失稳密帧及代码报告hash保存在artifacts/sim2real_reflection_20261013/。核查发现v8正式31项Isaac验收analytic_bank_required=false：1m/s、yaw+1.4平均roll−7.0366°、bank误差15.1585°仍passed，反向也有反向倾斜；所以压弯反确有策略端先例，不能只怪实机符号。0.5直行Isaac速度误差.15247m/s仍过旧stage（不满足10%）；旧MuJoCo 0.5/0于8.36s、0.5/0.3于6.184s触发倾角/高度停止。31项不含零命令站立、负vx、原地spin或动态切换；DR=.1窄范围和1周期小概率延迟不能视为最高DR。

记录D补偿应同源未缩放轮速、限幅内单施加点；.052近似补偿+真实b不一定在v8窄DR内，须读回/对照，不保证修复全部症状。保持25D/6D及原vx+yaw，禁止盲改q0、照搬四腿同号测试或落地冻结action分流。现有C参考CPU测试编译/运行exit0，非板端并发/时序认证。本次只更新复盘证据和记录，未改网络/训练/物理/部署代码、未启动GPU；原全<=1.5稳定后再迁移弹簧顺序保留，但不得以旧宽松bank通过代替正确压弯验收。


2026-10-13返回训练主线：上一批router_low_yaw_handover_probe已真实完成，63点60/63、全存活，新增1.05/±.6和1.10/-.6旧stage通过，剩1/+1.4、.25/+.1、1.05/+1.4；不把此结果当完整目标域认证。最新教师同样反向压弯：1.5/+1.4 roll−7.52776°、bank误差19.61056°，因此先修明确姿态缺口再扩域。
启动router_positive_bank_alignment_20261013，从候选928dc36fe8d159597b163642c9ea7a24ab13a8d808795db0f68da3176f549ac9继续，只启用现有analytic bank奖励，固定原1.5/+1.4目标/正向采样/旧rehearsal，仅正向speed_extensions及critic更新。spring4.31、DR.1、Dfb/b/C、5.5峰值3持续、2/2/8ms均固定；负向/physics/旧图冻结，无外环。512env100updates/10warmup/3000秒，结束自动63点快筛及显式严格速度/bank、接触诊断；不自动晋升，不改已部署v8。5项bank几何测试及目标门控[1,0]通过，尚无新训练行为结论。命令、source hash、contract和停止条件见实验目录。

router_positive_bank_alignment_20261013实际启动核查：trainer PID3230386存活，runtime全部通过，analytic bank启用及world_heading_inward_v2读回确认；首批冻结/缓冲违规0，当前审计iteration8652，尚未独立行为验收。日志logs/rsl_rl/rmuc_wheel_leg_course/2026-10-13_11-49-20_router_positive_bank_alignment_20261013；100更新后自动63点带turn-dynamics诊断，不承诺仅靠奖励开启即可修好外倾。

2026-10-13压弯奖励批被3000秒预算停止：训练94更新审计到8744，94条冻结/缓冲违规0；最后完整保存model8740。原批保留incomplete，不重复训练。仅80正向副本张量变化，SHAaca69e64ff5e5ec5c6afc74a7a919a331625185d034d30d56ae850a138b3c932。单独bank_alignment_budget_eval_20261013已启动63点+turn-dynamics验收（GPU0、126env、4回合、900秒），尚无行为结论、不晋升；动力spring4.31 DR.1保持。

2026-10-13 bank_alignment_budget_eval_20261013完成exit0、runtime通过：model8740 SHAaca69e64ff5e5ec5c6afc74a7a919a331625185d034d30d56ae850a138b3c932，63点仍60/63、旧31/33新29/30、全存活、无已过点回退。1.5/+1.4 speed .017344、yaw .010961，roll从−7.52776到−7.19280°、bank误差19.27560°，未解决向内压弯；world-space接触诊断也确认外倾。严格速度+bank3°+原tracking+无倾倒仅8/63，不能冒充最终全域。
CPU reward_capacity_audit：2062条早期真实目标状态上可训练腿残差未近tanh限幅；当前19.2756°处bank指数奖励约2.08e−36、惩罚局部斜率.03647 reward/deg（非零），不足以证明唯一根因。停止原样延长训练，next_decision记录先做冻结腿动作小幅物理响应/零对照，确认可达倾向及speed/yaw权衡再改shaping。无新部署/学生/弹簧档提升，promoted=false。
