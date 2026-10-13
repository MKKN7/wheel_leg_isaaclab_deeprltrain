# RMUC 轮腿机器人 RL 长对话交接文档

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

## 2026-10-08 当前v587：新spring2.424375的+.8修复探针

v586已结束30/61，旧31仅1/+.8失败（yaw.06550/stable.25167），无存活退化/runtime通过，新弹簧不接受。v587同v579源/新spring/DR.1，仅右active动作0/-.005/-.01/-.02诊断，1600步/300秒、500帧，无训练。查evaluation_status.json和真实PID接续禁止重复；完成report.py后决定可行拟合，必须原预算/已有专家及全61无偏置retention。正式基线仍spring2.155/DR.1/31点32回合，目标active。

## 2026-10-08 当前v586 spring2.424375迁移筛查运行

v584/v585正式DR.1同源31/31各32回合通过，无退化且runtime/质量惯量motor读回通过；v584/combined_report.json已生成，两个进程结束。稀疏1周期延迟不等于全域延迟验收，其他30条件未完成。

v586已按门槛启动，source v579 SHA2a9ac3c7bcca93ad55b6c78187e0eb3d4d68cdab02c923b3c89fbac75aa75130，唯一弹簧2.155→2.424375，DR.1与执行器/延迟不变；61点4回合/3200步/600秒，无训练。查status及真实PID接续，禁止重复；完成后report.py检查旧31retention及存活，未报告不晋级。目标active，最终交付尚未完成。

## 2026-10-08 接续v584/v585原进程，v586未启动

原PID121868/121869、session64213/2554真实存活，约2分钟/1000步。继续本轮正式31点，结束v584/combine.py；观察慢不重启。v586准备下一弹簧2.424375（从2.155），DR.1/延迟/执行器不变，启动器明确31/31正式门槛，当前未运行。若本轮失败不启动v586，留当前档适应。最终目标仍active。

## 2026-10-08 当前v584/v585 DR.1正式复验运行中

v582已exit0：31/61快筛、旧31保留/无存活退化、runtime和mass/inertia/motor审计通过，非32回合晋级。v584/v585同v579源，spring2.155/DR.1，31点分16/15×8环境，每点>=32回合、6500步/900秒，GPU0并行已启动。查各status/真实PID后接续，禁止重复；结束v584/combine.py。无新checkpoint，正式门槛未过不升DR/弹簧，目标active。

## 2026-10-08 当前v582 DR.1筛查运行中

v580/v581正式31/31各32回合全部通过，无退化，runtime及质量惯量motor审计通过，v580/combined_report.json已生成。下一项v582按门槛启动，同v579 SHA2a9ac3c7bcca93ad55b6c78187e0eb3d4d68cdab02c923b3c89fbac75aa75130，spring2.155、延迟概率5%/2%固定1周期不变，只提高物理/传感器DR到.1；实际质量/摩擦±2%、总惯量约±4%、Dfb[.049,.051]，不是最终±20%/最高DR。

61点4回合/3200步/600秒、零训练。查v582/evaluation_status.json与真实PID接续禁止重复；结果前不认定DR.1通过，旧31损失或存活退化即停在此档适应。桌面v565不替换，最终目标active。

## 2026-10-08 v583 MuJoCo回归结束，v580/v581仍待完成

v579的6项同契约nominal探针5项12秒未倒，.5/+.3在5.032s倾倒。1/+.2/+.3 yaw误差.03345/.01846但速度仅.7786/.765，非跟踪全部通过；三保护条件未倒。完整证据v583/final_report.json，不更新桌面ONNX，不称最高DR或sim2sim认证。

GPU正式31点评估v580/v581原PID102452/102451、session94482/48975仍运行，接续先核对真实进程，不能重启；结束运行v580/combine.py。v582 DR.1准备未运行，只有正式31/31门槛通过才可启动。最终目标未完成。

## 2026-10-08 接续位置：v580/v581正式评估仍运行，v582未启动

原PID102452/102451（session94482/48975）真实存活，最新1000步。保持同v579源与当前物理，等待正式31点结果后运行v580/combine.py，不重复启动。v579 export6254输入一致性通过，JIT0/ONNX3.576e-7及hash落盘；桌面v565不改。

v582仅准备DR.1筛查，启动器已加同源31/31正式>=32回合/无退化门槛。实际质量/摩擦±2%、总惯量约±4%，延迟固定1周期概率5%/2%和spring2.155保持；非最高DR、当前尚无结果。若正式失败应先修复，禁止运行v582越级。目标active。

## 2026-10-08 当前v580/v581正式31点复验

v579快筛31/61，旧28保留、无存活退化，新增.5/+.3、1/+.2、1/+.3均stable1，yaw.01051/.01522/.00896；1.5正慢转仍缺，静止和其他30点未完成。source SHA2a9ac3c7bcca93ad55b6c78187e0eb3d4d68cdab02c923b3c89fbac75aa75130。固定spring2.155/DR.05，31项16/15×8副本GPU0并行，各6500步/900秒、每点>=32回合已启动。

查v580/v581的status与真实PID后接续禁止重复，结束运行v580/combine.py。无新训练/容量增加、未正式报告不升DR；桌面v565保持，最终目标active。

## 2026-10-08 当前v579正慢转适应后全61点评估

v577有4项通过示范，但1.5/+.2超原残差预算被排除，1.5/+.3无通过；实际训练仅.5/+.3、1/+.2、1/+.3。v578600步未达拟合门槛无候选；v579600步用反算tanh目标辅助达门槛（MAE.004945），只更新原positive reference网络，无增参数，保护域保持。

source v573，候选SHA2a9ac3c7bcca93ad55b6c78187e0eb3d4d68cdab02c923b3c89fbac75aa75130。spring2.155/DR.05保持，61点无偏置3200步/600秒评估运行，查v579/evaluation_status.json和真实PID接续，禁止重复；原28保留/无存活退化前不晋升。桌面v565不改，最终目标active。

## 2026-10-08 当前v577命令扩展采样；v573正式28点已通过

v574/v575完成，同v573 SHA39b4b8095a504085502dc35606ba69df4f2adac053ed9257349cb904baecc294在spring2.155/DR.05下28/28、各36回合，无退化、runtime和质量惯量motor读回通过。报告v574/combined_report.json。不是全域/最高DR/多seed/动态命令/MuJoCo或实机完成，桌面v565不替换。

按循环顺序启动v577，固定物理，仅五正慢转缺口(.5,+.3),(1,+.2/.3),(1.5,+.2/.3)差分轮0/-.005/-.015/-.04/-.08诊断采样，1600步/300秒，无训练，数据500帧。查v577/evaluation_status.json和真实PID接续禁止重复，结束report.py后核对可行预算再做固定容量训练，保护28项并全61无偏置retention。目标active。

## 2026-10-08 v576 CPU MuJoCo结束，v574/v575仍运行

当前v573导出6254输入通过（JIT0/ONNX3.576e-7），hash已存。v576同契约canonical MJCF的9项12秒探针8项未倒，.5直行3.616s倾倒，不能称8/9行为认证。1/-.2与1/-.3存活但过转（实际yaw-.29648/-.36659），其他跟踪缺口仍在，桌面v565不替换。证据v576/final_report.json、controller_contract、command.json，全部CPU诊断完成。

GPU v574/v575还在正式28点复验，原PID86636/86637与session20016/5731；重新核对进程后继续，不重复启动。结束v574/combine.py联合审计，未结果不晋升。目标active，网络参数量不变。

## 2026-10-08 当前v574/v575正式28点复验

v573无偏置快筛28/61、旧25全部保留、无生存退化，正爬行和负慢转同时保住。新source SHA39b4b8095a504085502dc35606ba69df4f2adac053ed9257349cb904baecc294。固定spring2.155/DR.05，28条件两批各14×9环境、>=32回合，GPU0并行已启动，各6000步/900秒。查两个status和真实PID后接续，禁止重复。结束v574/combine.py联合核对，未完成不得晋级；无新增训练/模型规模变化，桌面v565保持。

v573仍快筛非最终认证，静止和其他33条件/最终弹簧/最高DR/MuJoCo/硬件未完成。正式通过后按弹簧→命令域→DR顺序推进，目标active。

## 2026-10-08 当前v573局部恢复评估中

v57127/61、负向慢转改善与1.5/-.1保护成功、无生存退化，但.25/+.1丢失，拒绝晋升。v572只恢复正爬行外层network的600步拟合未达.001门槛，无候选。v573同预算加入反算tanh前目标辅助，最大MAE.000862、保护输出不变，参数量不增；candidate SHA39b4b8095a504085502dc35606ba69df4f2adac053ed9257349cb904baecc294。

新候选spring2.155/DR.05下全61点无偏置retention已启动，3200步/600秒。查v573/evaluation_status.json与PID禁止重复。旧25及生存门槛未过不接受、桌面v565不改。最终弹簧/全域/最高DR/正式部署仍未完成，目标active。

## 2026-10-08 v571训练后评估运行中

v570补采真实1.5/-.1、静止状态，stride3/500样本跨至1497周期；1.5慢转通过，静止失败不可冒充成功。v571从v565固定容量600步训练，新增这两域强保护回放，目标MAE.002764/保护MAE.000796，冻结通过，新候选SHA03d213a632fe344ffecc139aaeb939160dfe7f0513bf8b56cdd0fc4503223fb3。

当前spring2.155/DR.05全61点无偏置评估已启动，3200步/600秒，查v571/evaluation_status.json与PID接续禁止重复。源仍v565，不从被拒绝v569继续；若旧25损失或生存退化则拒绝，未报告promoted=false。桌面包不变、目标active。

## 2026-10-08 v569结束：28/61但拒绝晋升，补rehearsal覆盖

v569固定容量600步训练后修复1/-.2（yaw.004404/stable1），扩展1/-.3、1.5/-.2/-.3和.5/-.1，通过总数25→28。但旧1.5/-.1失败、静止存活退化，retention=false/promoted=false，不更新桌面ONNX。完整report/runtime/retention在v569，评估exit0。模型参数仍493492。

下一步先采当前基底v565在1.5/-.1和静止的真实轨迹补足保护域回放，再做同容量适应，不能拿合成或未覆盖的rehearsal误差保证闭环保留。当前新spring2.155正式门槛仍未通过（v566/567的1/-.2缺口）；不升弹簧/DR。用户要求继续训练主线，H7型号/工具实测尚缺，不因此阻塞仿真训练，也不声称CubeAI实时性。

## 2026-10-08 当前v569实际训练后无偏置评估中

v566/v567正式结果24/25、各36回合，1/-.2 yaw.052697/stable.206615失败、全存活；新spring2.155尚不能晋级。v568采得1/-.2/-.3小轮差分可行轨迹，v569已完成600CPU监督训练+真实旧观测rehearsal，只更新已有inner frozen_reference网络，actor参数仍493492，不新增专家。源v565，新candidate SHA0027c60104dddfe7f6a9a931bc040d86434c204c99723008d5a21758e25ffe2f；目标MAE最大.000234，回放平均最大.000059，冻结审计通过。

v569当前全61点无偏置retention运行，3200步/600秒，spring2.155/DR.05保持，旧25丢失/存活退化拒绝。检查evaluation_status.json及真实PID接续禁止重复；结束后report.py，拟合不等于学会，promoted=false。桌面v565包不覆盖，正式失败必须诚实保留；总体目标未完成。

## 2026-10-08 桌面候选包与demo交付已生成，继续保留验收边界

用户要求最适合ONNX及不倒MuJoCo视频。实测v537/v545/v565各10命令12秒，均9未倒、.5直行倾倒；选择当前v565（spring2.155/25点Isaac快筛、旧24保留），不是实机成功率保证。桌面包/home/kellen/桌面/RMUC_v565_20261008，ONNX export/policy.onnx SHA3deb2604b3f8d4c5d5efc7c00fbf1f08302a07f067250c275131c55253c0aff8。RMUC_v565_demo.mp4为9段独立重置12仿真秒拼接，含.25直行±.1、1直行±.8/±1.4、1.5直行，跟踪误差如实标注；.5倾倒视频不包含，失败报告仍保存。另有分段视频、metrics、契约、原MJCF/mesh、回放和controller参考源码、README及hash清单。包校验及1秒实际回放通过，9帧视觉检查完成。

视频/候选对比证据在artifacts/delivery_20261008；首次framebuffer失败保留，修复仅显示尺寸/跟随相机，不改物理。v566/v567仍需核对真实进程并合并正式回合结果；新候选尚不宣称最终部署。README明确spring2.155而非10.775、wheel2ms仿真而非实机1ms、DR/硬件符号/热/动态命令未认证；0.5直行不可当可用技能。目标仍active。

## 2026-10-08 当前v566/v567新弹簧25点正式复验

v565评估JSON完整恢复：25/61、旧24全保留、无生存退化，爬行.25/-.1 yaw.003663/stable1。原PID已不存在、status遗留running，退出码未知；不改写exit0，说明见status_reconciliation.json。导出6254输入通过，JIT0/ONNX3.576e-7，hash已落盘。

source v565 SHA8a46d0aec4b96fc31088d55724161ba96918a384167fe1e5a1f9e7c160bdd825，spring2.155/DR.05不变，25条件分两批12/13×9副本，每点>=32回合，各6000步/900秒、GPU0共享已启动。查v566/v567真实PID禁止重启；结束运行v566/combine.py。无训练/无新checkpoint，未结果不接受弹簧新档；通过后命令域扩展，之后再DR。最终目标仍active。

## 2026-10-07 当前v565爬行修复候选评估中

v564轮差分+.02/-.02使.25/-.1短测yaw.003976/stable1，原预算内可行。v565只外层reference_network拟合400CPU步，局部门控负yaw[-.2,0]/爬行速度，12测试、保护输出冻结、最大MAE.000133、checkpoint恢复通过。SHA8a46d0aec4b96fc31088d55724161ba96918a384167fe1e5a1f9e7c160bdd825。spring2.155/DR.05/其他契约保持，全61点无偏置评估运行，3200步/600秒，查v565/status和PID禁止重复。无结果不认定爬行修好；旧24全恢复/无生存退化后仍需同候选>=32回合，新弹簧未认证，目标active。

## 2026-10-07 v563结束24/61但旧爬行仍缺，v564运行中

v563恢复.5直行（speed error.14960/stable1），两大转弯保持，无生存退化；原旧24仍丢失.25/-.1，总数24含另一个新通过点，不能接受spring2.155。证据v563/final_report.json。新候选SHA1a1509cfeebee84df5dfca17b04bd16c588efd3cd2277c9072497d6c2e3f75a8。

v564同源/同spring2.155/DR.05，仅爬行.25/-.1小轮差分0/.0025/.005/.01/.02诊断，1600步/300秒，无训练，腿固定，数据采500步。查v564/evaluation_status.json与PID禁止重复；结束report.py后决定可行拟合，偏置不进部署，仍需全61点retention。目标active。

## 2026-10-07 v562结束23/61，v563直行修复候选评估中

v562新spring2.155/DR.05下恢复1/+.8与1/+1.4，无生存退化；23/61，旧24仍缺.5直行、.25/-.1，promoted=false。v563在v562基础仅拟合原窄直行scoped_network，400CPU步，最大MAE.000182，其他参数冻结/保护域输出逐值一致，候选SHA1a1509cfeebee84df5dfca17b04bd16c588efd3cd2277c9072497d6c2e3f75a8。当前全61点无偏置3200步/600秒评估运行，查v563/evaluation_status.json/PID接续禁止重复。爬行缺口仍未修，尚不接受spring2.155；旧正式基线v545/spring1.885625/DR.05仍保留。

## 2026-10-07 当前v562评估中：两项大转弯专家已做CPU适应

v561负右active探针找到.5直行、1/+.8、1/+1.4稳定通过示范，.25/-.1仍失败。v562仅原专家5/3拟合两大转弯，各400步；其他冻结/保护域输入输出相同，标签重建3.725e-7、MAE最大5.58e-5。新candidate SHA2cc32384beb0867149e1dfcd6aa9ac653ade6fbf1721df276203cae8e7874e04，spring2.155/DR.05保持。无偏置61点retention已启动，3200步/600秒，查v562/evaluation_status.json和PID禁止重复。拟合不等于闭环通过，直行和爬行缺口仍待解决；全旧24未恢复前新弹簧不接受，promoted=false。

## 2026-10-07 v561运行：右active负偏移0/-.01/-.025/-.05

同v545候选、spring2.155/DR.05，四退化条件原动作范围内负向诊断，1600步/300秒、500步数据和leg audit。无训练，不生成candidate，不接受偏置为策略能力。先查v561/status和真实PID，结束后report.py；若有可用轨迹先检查残差budget再拟合，否则不盲训。已验收基线仍spring1.885625，最终目标未完成。

## 2026-10-07 v560结束：拒绝单边右active正偏移

四退化条件所有非零右active偏移.025/.05/.1均不通过；多数速度/yaw恶化，1/+1.4在+.1倾倒。无可用改善示范，不做拟合，不继续扩大正偏移。v560/final_report.json/数据已保存，进程exit0、零训练、promoted=false。双腿补偿与单边控制不可等同；下一步可测试负向小步或腿轮联合适应，原动作与残差预算不变。当前已验收仍v545/spring1.885625/DR.05/24点40回合，spring2.155尚未接受。

## 2026-10-07 v560右active动作诊断运行中

四退化命令、右active归一化动作0/.025/.05/.1在原consumer之前加入，仍原动作clip；不采用v559不可达左目标偏移。固定v545/spring2.155/DR.05，1600步/300秒，500步数据与leg audit。核对v560/evaluation_status.json与PID后接续，禁止重复启动。无训练/无候选；完成后report.py整理，再按原残差budget/标签可行性决定拟合。诊断不认证，旧已验收阶段仍1.885625，最终目标未完成。

## 2026-10-07 v559结束：不采用统一腿目标补偿

4条件目标偏移诊断完成、每桶2回合全存活。全偏移使.5直行、1/+.8、.25/-.1短测通过，1/+1.4不一致；DR桶不配对，不能认定单一因果。leg audit稳态各1200样本，腿torque饱和0。低速两条件左active目标.20449/.19449rad超过原.2/.19可达范围，故示范不可直接蒸馏到原动作clip，禁止部署追加偏移或放宽预算。大转弯域扩展限幅不能按raw>1误判硬饱和。证据v559/final_report.json，exit0，无新checkpoint，promoted=false。

下一步固定spring2.155/DR.05，用原范围内剩余腿轮动作适应（例如未饱和右active通道），不能继续统一后置PD偏移。当前已验收基线仍v545/spring1.885625、DR.05/24点40回合；新弹簧未接受。目标仍active。

## 2026-10-07 v558未通过，v559腿目标诊断运行中

v558新spring2.155下21/61，旧24丢失4点（.5直行、1/+.8、1/+1.4、.25/-.1）、新增1点，无生存退化。runtime通过但retention失败，promoted=false，保留spring1.885625验收基线。延迟暴露变多，非严格配对，勿断言全是弹簧导致。

v559同v545/新spring2.155/DR.05，4退化命令active腿目标0/+.002245/+.004490rad诊断，1600步/300秒，附leg audit，无训练。+deltaSpring/kp仅未限幅PD补偿假设，动作饱和可能无法靠策略实现，不放宽预算、不进入部署。查status/PID禁止重复；结果只定位，后续无偏置全61点retention仍必须。目标未完成。

## 2026-10-07 当前v558弹簧2.155迁移运行中

v556/v557已结束，DR.05同v545候选24/24各40回合，无生存退化，runtime及质量/惯量/motor读回审计通过；联合报告v556/combined_report.json。仍不是全域或最高DR认证。

下一项v558已通过启动门槛，固定同source/DR.05，只升spring1.885625→2.155Nm/腿，无训练。61点4回合快筛、3200步/600秒，与同DR的v555比retention；先检查v558真实PID再接续，禁止重复。未报告不晋级；后续按弹簧→命令域→DR循环。最终目标保持active。

## 2026-10-07 当前等待v556/v557，v558尚未启动

两评估PID3341535/3341536仍活，约4分半/1000步，GPU0与其他工作区第三个Isaac作业共享99%利用率；不干预他人进程，不因观察慢重启。原session68048/26412，继续核对真实进程，预算各900秒。v558准备下一spring2.155（从1.885625），其余固定DR.05；必须v556/combine.py生成正式24/24且>=32回合、无退化后才能启动，启动器已加门槛。尚未运行，不把准备当结果；目标保持active。

## 2026-10-07 v556/v557 DR.05正式回合运行中

v555快筛24/61，旧24保留，无生存退化/runtime通过。现在同v545候选、固定spring1.885625/DR.05，对24点分两批各12×10环境>=32回合，GPU0并行，各6000步/900秒。完整contract/command/status在v556/v557，查真实PID禁止重复启动。结束后运行v556/combine.py；正式通过后先小步弹簧，再命令域，再DR，按用户确认递归。无网络更新、不生成新checkpoint；最终目标未完成，promoted=false。

## 2026-10-07 最新用户课程顺序：弹簧→命令域→DR循环

当前DR.05完成验收后先小步弹簧迁移，再扩大命令域，再提高DR，按此递归。每步只改变该课程变量并验retention，失败固定该档适应。延迟1→2→3周期单独渐进。v555当前为快筛，不是正式通过；需同候选>=32回合后开始下一弹簧档。所有最终目标保持，不将现24点当全域交付。

## 2026-10-07 v555 DR.05快筛运行中

v545同源、spring1.885625固定，物理/传感器幅度DR.02→.05；观测/通信延迟概率仍.05/.02、幅度固定1周期。source SHA34e9afd05ea3c3b3eddc417d2e6b6126d8c0c53f36f7446e92d32c88221be2fe，无训练。实际质量/摩擦±1%、总惯量约±2%、Dfb[.0495,.0505]/b[.00195,.00215]/C[.000975,.0012]、motor.1–1.5ms，腿与弹簧DR未加。61点4回合/3200步/600秒，原24点及存活retention门槛不变；查v555/evaluation_status.json/PID后接续。无报告不晋级，不误称DR.05完成。

## 2026-10-07 最新v552结束：观测1周期5%概率无退化

v545同源、spring1.885625/jointDR.02，观测概率.05、通信.02、固定1周期；61点各4回合24通过，旧24保留，无存活退化，runtime通过，暴露9004/4503步。报告v552/final_report.json、retention/runtime已生成，进程exit0。无新checkpoint，promoted=false。不要重启本轮或把它当固定1周期全域认证；DR.02正式24×40回合证据仍是v550/v551（概率.02），不要混淆两者。

部署D_comp=.052已核对。后续继续延迟出现比例/DR小步课程，保持物理变量可定位；最终弹簧/全域/最高DR/正式sim2sim未完成。照片与v554不构成真实弹簧安装点、力行程认证，不擅自反转或变更模型。

## 2026-10-07 v552已启动：固定物理，只提高观测延迟概率

v545同源，spring1.885625、DR.02、D反馈中心.05/b.002/C.001不变。观测延迟概率.05、通信.02，命中固定1周期8ms，61点快筛/3200步/600秒、无训练。查v552/evaluation_status.json及真实PID后接续，不重复启动。必须保留24旧通过点和存活，未报告不晋级。

照片不足以确认真实安装点Jacobian；v554仿真负力矩助伸结果不是硬件弹簧方向/幅值认证，暂不改模型或符号。后续力矩方向与幅值不确定性仍需要验证，不能拼接成sim2real通过。

## 2026-10-07 v554弹簧方向几何审计完成

当前MJCF两侧212姿态闭环求解全部收敛，固定髋时dL/dq_active全负[-.14539,-.10131]m/rad，当前负active弹簧力矩因此助伸，未发现方向反；腿长覆盖.231–.402m。见v554/kinematic_report.json。只证明选定闭环几何运动方向，不等于完整站蹲动态轨迹/实机安装或恒幅值准确。无修改动力、无GPU训练。v552观测延迟概率.02→.05的下一实验仍prepared未启动；可继续当前负向等效模型，不需仅凭sin猜测反转。DR.02已有24条件各40回合通过，延迟全域仍未认证。

## 2026-10-07 最新：DR.02各40回合24/24；先核对弹簧方向

v550/v551均结束，v550/combined_report.json通过24/24，无生存退化；仍非固定延迟全域认证。独立观测/通信delay probability已接入，可选字段不改变旧配置，18测试通过。v552仅准备观测概率.05、通信.02/固定1周期，未启动，禁止当运行任务。

用户询问sin(front-back)恒号是否足够。v553已审计：当前等效弹簧实际不使用sin，而是active_link负恒力矩；2048姿态下轴投影/反作用代数通过。旧controller是带encoder零位的电机补偿，左右符号不同，不能直接当机械被动力。当前helper映射下v542部分轨迹sin无明显双符号，但q0的sin=0且未扫完整站蹲/全域，不能认证实机方向。报告v553/spring_direction_report.json记录缺口：机构几何角零位、可达站蹲走转轨迹、虚功/Jacobian方向。保留当前动力不擅自反转；优先最小方向验证再继续DR。

## 2026-10-07 v550/v551 DR.02正式回合正在运行

source v545 SHA34e9afd05ea3c3b3eddc417d2e6b6126d8c0c53f36f7446e92d32c88221be2fe；同v549物理与DR profile，仅增加24条件回合预算。两批各12×10环境、>=32回合，GPU0共享，各6000步/900秒。无网络更新，不应期待新model文件。查两个evaluation_status.json和真实PID，禁止重复。结束后运行v550/combine.py，联合核对retention/延迟exposure；低概率1周期命中不是固定延迟全域认证。目标仍未完成，不能升高弹簧或随机1–3延迟。

## 2026-10-07 v549结束：DR.02快筛24保留，不晋级

v545同candidate/spring1.885625，联合DR.02，全61点各4回合24通过，旧24无丢失、无生存退化、runtime通过。final_report/retention/runtime/delay_coverage均在continuation_v549，exit0。观测/通信延迟只触及4/5命令条件（累计3002/4503步，部分仅2步），幅度都<=1周期，不是全24点延迟验收。下一步当前档32回合并补足固定1周期暴露，不直接升DR或随机1–3。无网络更新，最新候选仍v545 SHA34e9afd05ea3c3b3eddc417d2e6b6126d8c0c53f36f7446e92d32c88221be2fe；最终目标未完成，promoted=false。

## 2026-10-07 v549首档联合DR运行中

v545同源固定spring1.885625，jointDR=.02，61点快筛各4回合、3200步/600秒，无训练。观测/通信命中延迟均固定1周期、命中概率各2%；不要把.02误当延迟幅度，也不宣称每点有延迟覆盖。质量/摩擦±.4%、总惯量约±.8%、Dfb[.0498,.0502]/b[.00198,.00206]/C[.00099,.00108]、motor40–600us。leg DR/spring DR未加。完整profile、contract、command与report.py在continuation_v549；原24点保留/无存活退化门槛，不自动升档。结束后运行report.py核对runtime及实际exposure，最终DR/弹簧/全域/sim2sim仍未完成。

## 2026-10-07 最新24点正式回合通过，候选位置说明

v547/v548均exit0；continuation_v546/combined_report.json汇总24/24、各40回合，同v545 SHA34e9afd05ea3c3b3eddc417d2e6b6126d8c0c53f36f7446e92d32c88221be2fe、runtime通过。当前spring1.885625/D反馈.05/b.002/C.001/DR0/2ms。其他37失败点/动态命令/最高DR/最终弹簧/正式sim2sim未完成，promoted=false。下一阶段固定动力小DR，延迟先固定1周期，禁止直接随机1–3。

最新候选artifacts/continuation_v545/candidate.pt实际mtime15:11:27；logs最新PPO v533/model_7265.pt是13:34:35。此后是CPU拟合、隔离修正、诊断和GPU评估，非连续PPO训练。v544虽15:00更新但retention失败已拒绝，不可误选最新日志或所有candidate。当前两批进程均已结束，不应重复启动。

## 2026-10-07 导出通过，继续等待v547/v548真实进程

v545 export完成6042输入边界检查，JIT0/ONNX2.980232e-7，hash已保存。两批评估未出最终结论。启动时PID v547=3287988、v548=3287987；必须重新检查存活和命令，不据旧PID重启。两批exit0后运行continuation_v546/combine.py，联合核对24条件>=32回合与runtime。缺报告/失败不能推进DR；完整目标仍active。

## 2026-10-07 v547/v548正式回合分批运行中

source v545 SHA34e9afd05ea3c3b3eddc417d2e6b6126d8c0c53f36f7446e92d32c88221be2fe，全部24通过点分成两批各12点×10副本，GPU0共享并行，每点>=32回合，各6000步/900秒。v546原288环境被128上限在仿真前拒绝，无结果；证据不覆盖。spring1.885625/D反馈.05/b.002/C.001/DR0/2ms保持。接续核对v547/v548的PID和日志，禁止重复启动；结束后分别运行report.py并联合核对24点，未完成或任一点失败均不晋级。最终目标保持active，下一DR延迟从固定1周期开始。

## 2026-10-07 最新v545完成：24/61，旧22全保留

候选artifacts/continuation_v545/candidate.pt SHA34e9afd05ea3c3b3eddc417d2e6b6126d8c0c53f36f7446e92d32c88221be2fe。同spring1.885625/D反馈.05/b.002/C.001/DR0/2ms，61点各4回合24通过；新增.5/+.1与1/+.1，yaw.01777/.01264、stable1，无旧点丢失/生存退化，runtime通过。隔离参考修正解决了v544技能干扰；11测试及2080输入恢复/保护审计通过。报告v545/final_report.json，评估exit0。

新候选尚无32回合验证，不能继承旧checkpoint证书，promoted=false。下一步当前候选正式回合复验后，小步DR并固定1周期延迟，弹簧先不变。慢转+.3/静止/全域/最终弹簧/最高DR/正式sim2sim未完成；部署D_comp=.052仍是未饱和参考，不是硬件认证。

## 2026-10-07 v545隔离修正评估中

保留v539为基底，仅在正慢转域混合v544已学参考网络，域外冻结原reference，未增加迁移层数/动作预算。11单测通过、2080输入保护域差0、恢复差0；候选SHA34e9afd05ea3c3b3eddc417d2e6b6126d8c0c53f36f7446e92d32c88221be2fe。spring1.885625/D反馈.05/b.002/C.001/DR0/2ms不变，61点>=4回合无偏置retention已启动；查continuation_v545/evaluation_status.json和真实进程。promoted=false；同输入隔离不能替代闭环结果。延迟仍按先固定1再2/3课程，不直接随机1–3。

## 2026-10-07 v544结束：拒绝退化候选，保留v539

全61点22→17，新增.5/+.1、1/+.1，但丢失7个既有慢转/爬行条件，无存活退化，runtime通过；retention失败，promoted=false。不要将v544作为下一DR源，继续用v539 SHA474a44e8859fa5f01f50968a3c00123c4d69d08e236e2672c9d8e5d5bb5c530f，spring1.885625的八点36回合证据在v541。共享reference分支适应发生技能干扰，合成rehearsal不够；下一步解决更新范围隔离，不盲增训练、不再堆第五层。完整失败条件在continuation_v544/final_report.json/retention_audit.json，评估已exit0。

## 2026-10-07 v544无偏置retention评估中

v543叠层触发深度上限，0更新拒绝。v544只拟合现有physics.frozen_source.reference_network（600CPU步），旧22命令rehearsal，不改共享预算；.5/1m/s四项轨迹用于拟合，1.5m/s两项超预算排除，不代表解决。候选SHA098aad676022143cbce7f39d93ab62ee0dc629d8cf40323a7e88450be135b1ae。固定v541物理契约spring1.885625/D反馈.05/b.002/C.001/DR0/2ms，全61点各>=4回合retention运行中，查continuation_v544/evaluation_status.json和真实PID。新权重不保证其他命令输出一致，旧22点丢失/存活退化即拒绝；未完成前无结论。

## 2026-10-07 v542结束与延迟新要求

六慢转命令轮差分诊断已结束：-.15/+ .15偏置下三速度+.1和1.5/+.3短测通过，.5/1的+.3仍欠转；24条件全存活，2回合/条件。runtime通过，promoted=false，无新checkpoint/无retention认证；证据continuation_v542/final_report.json及paired teacher_probe.startup.pt。慢转至少部分可达，但当前两层network门控0，先选择可训练分支/核对预算再拟合，不盲续训原入口。

用户要求DR与延迟渐进：不直接随机1–3周期。已准备并schema验证六阶段profile：无延迟、观测固定1、联合固定1/2/3、最后混合1–3，未运行；见delay_curriculum.json。strength是出现概率而非幅度，实际抽样必须读回。现有schema共享观测/通信幅度，无独立action-only课程；继承惯量DR=0，非最终最高DR配置。下一实验保持spring1.885625，按行为门槛引入小DR，禁止同时改变弹簧与延迟。

## 2026-10-07 v542查慢转命令覆盖

真实actor门控核对：.5/1/1.5m/s的+.1/+.3六点，外层physics.network和上一层network更新门控均0；不能以继续增加这两层训练次数补慢转。冻结reference/scoped仍有动作输出，不能声称物理不可达。见continuation_v542/gate_audit.json。固定v541物理阶段和v539候选，启动六点轮差分诊断（0/-.05/-.1/-.15），1600步/300秒、无训练；查teacher_status.json和真实PID后接续，禁止重复。偏置结果不认证策略，后续仍需无偏置retention。

## 2026-10-07 v541结束：新弹簧八点36回合通过，但+1.4裕量仍紧

spring1.885625、D反馈.05/b.002/C.001/DR0/2ms，使用v539/candidate.pt SHA474a44e8859fa5f01f50968a3c00123c4d69d08e236e2672c9d8e5d5bb5c530f。八点8/8和正爬行2/2各36回合全存活；+1.4 yaw.049647/stable.571628接近门槛。全61点快筛22点保留见v540，不能借此称全域通过。v541/final_report.json与runtime_verified.json已生成，exit0，无训练；两轮实验契约已清理旧训练模板字段。

接续固定当前弹簧补裕量和慢转域，不重复已完成评估。最终10.775/DR/全域/静止/动态切换/正式sim2sim未完成，promoted=false；部署D_comp=.052仍仅未饱和参考。目标保持active。

## 2026-10-07 v540快筛22点保留，v541准备同候选32回合

spring1.885625下v539同网络61点22通过，旧22点无丢失/生存退化，runtime通过。证据continuation_v540/final_report.json。新档尚未正式接受；v541只增加回合预算，八点+两爬行每点>=32回合，6000步/900秒，参数保持D反馈.05/b.002/C.001/DR0/2ms、弹簧1.885625。状态和完整命令在continuation_v541，检查PID禁止重复。全域、最终弹簧、最高DR、sim2sim和部署认证未完成。

## 2026-10-07 v540评估中：spring1.885625小步迁移

同v539策略474a44e8859fa5f01f50968a3c00123c4d69d08e236e2672c9d8e5d5bb5c530f，仅弹簧1.61625→1.885625，无训练。其余D反馈.05/b.002/DR0/2ms保持，原22点retention门槛不变。查continuation_v540/evaluation_status.json与PID，禁止重复启动。无报告前不接受新档；最终10.775/DR/全域/sim2sim未完成。

## 2026-10-07 最新v539完成：+1.4 yaw.02952/stable1，22点保留

candidate474a44e8859fa5f01f50968a3c00123c4d69d08e236e2672c9d8e5d5bb5c530f，仅原+1.4专家改权重，物理spring1.61625/D反馈.05/b.002/DR0保持。无偏置61点>=4回合22/61无丢失或生存退化，+1.4裕量改善，runtime/freeze通过。新candidate没有36回合认证，不能继承v537证书；下一步小步弹簧筛查、接受新档前同最终candidate>=32回合。bank大/全域/DR/最终弹簧/正式sim2sim仍未完成，promoted=false，无运行任务。

## 2026-10-07 v539运行：针对+1.4欠转巩固裕量

v537同源、同1.61625弹簧，仅小差动力矩轨迹诊断；如有合格示范则只修旧+1.4专家，必须无偏置61点retention，不改其他技能。查continuation_v539/teacher_status.json及PID，不重复。最终DR/10.775/全域/实机仍未完成。

## 2026-10-07 最新v538完成：spring1.61625八点每点36回合8/8

同v537源3a5b024c40f1ef1edc3f6a820660303482fdec5c3cc2ddfa0a20315dcfc00b15，八点8/8及两爬行2/2、各36回合全存活，runtime通过。+1.4 yaw.049546/stable.573857为弱点，下一步先巩固裕量再提高弹簧/DR。selected_for_next_stage=true但promoted=false，不等于最终10.775/DR/全域/多seed/正式MuJoCo/实机认证；bank仍较大。无运行作业，报告continuation_v538/final_report.json。v537新controller契约已按实际参数与hash生成，部署D_comp=.052保持。

## 2026-10-07 v538运行：新弹簧1.61625每点>=32回合

v537源3a5b024c40f1ef1edc3f6a820660303482fdec5c3cc2ddfa0a20315dcfc00b15，10条件12副本，不训练。D反馈.05/b.002/DR0/2ms保持；部署D_comp=.052已核对。查continuation_v538/evaluation_status.json及真实PID，不重复启动。阶段尚未32回合通过，不升弹簧或DR，最终目标未完成。

## 2026-10-07 最新v537完成：spring1.61625恢复22/61

candidate3a5b024c40f1ef1edc3f6a820660303482fdec5c3cc2ddfa0a20315dcfc00b15，仅旧+1.2专家修复，runtime/freeze/导出parity通过。无偏置61点原22全保留、无存活退化，+1.2 yaw.01936/stable1。新弹簧档只有4回合快筛，下一步同candidate新档>=32回合；不借用v535旧弹簧结果、不升DR。checkpoint继承infos可能旧档，必须用本轮experiment/runtime显式弹簧1.61625契约，最终10.775/DR/全域/MuJoCo仍未完成。promoted=false，无运行作业。

## 2026-10-07 v537运行：新弹簧1.61625下修+1.2欠转

源仍v535，右-左+小偏置扫描仅诊断，若有稳定示范只拟合现有+1.2专家，不扩预算/外环，必须无偏置retention原22点。D反馈.05/b.002/DR0/2ms固定；查看continuation_v537/teacher_status.json及PID，禁止重复启动。阶段尚未接受，完整目标未完成。

## 2026-10-07 最新v536已结束：新弹簧1.61625暂不接受

同v535网络春力矩小步迁移，61点22→21，只丢1m/s、+1.2rad/s：yaw.05305/stable.4097、仍存活1；无存活退化。runtime通过但retention失败，stage_accepted/promoted=false。保留v5351.346875八点36回合8/8基线，下一步固定1.61625只修+.1.2局部适应再retention，不全网盲训、不跳DR。证据continuation_v536/final_report.json；无运行任务，完整目标仍未完成。

## 2026-10-07 v536评估中：弹簧1.346875→1.61625，同网络

v535八点36回合8/8后接续弹簧小步迁移。v536同candidate不训练，仅spring1.61625，其余D反馈.05/b.002/C.001/DR0/2ms不变，61点retention原22点需保留。查continuation_v536/evaluation_status.json和PID，不重复启动；无报告前不晋升。最终10.775/DR/全域/sim2sim仍未完成。

## 2026-10-07 最新v535完成：阶段八点8/8，每点36回合

下一阶段源candidate.pt SHA4ed315d258ca9b6475d4a4e5a2c20d1bda50fe77022ceca2d5eb178d0dc91ab6。61点快筛22/61原21无丢失；同candidate历史八点各36回合8/8、两个正爬行点各36回合2/2，全存活。+.8 yaw.00878/stable1已修，不重复同修复。冻结/runtime/5450输入JIT-ONNX通过。

仅当前反馈D.05/b.002/弹簧1.346875/DR0/2ms单seed静态跟踪；bank不作解析门槛，未完成全域/静止/最终弹簧/DR/多seed/正式MuJoCo/部署。selected_for_next_stage=true，promoted=false。无运行作业，证据continuation_v535/{final_report,report32}.json。完整目标保持。

## 2026-10-07 v535快筛22/61、八点8/8，32回合评估正在运行

无偏置+.8 yaw.00823、速度.00873、stable1，原21点保留，八点快筛8/8。正在同candidate4ed315d258ca9b6475d4a4e5a2c20d1bda50fe77022ceca2d5eb178d0dc91ab6跑八点+两爬行各>=32回合；看continuation_v535/evaluation32_status.json及PID，禁止重复启动。快筛最终报告仍promoted=false，静止/全域/目标弹簧/DR/多seed/正式sim2sim未认证。

## 2026-10-07 v535无偏置评估中：修历史+.8

v533源+.8诊断偏置右+.01/左-.01使yaw误差.00859、stable1；仅用作示范，CPU拟合既有local expert5（不是部署偏置），其他网络与预算冻结。候选4ed315d258ca9b6475d4a4e5a2c20d1bda50fe77022ceca2d5eb178d0dc91ab6，验证轮MAE.000767。正在同61点无偏置retention，原21点需保留；查continuation_v535/evaluation_status.json及PID避免重复。DR0/阶段弹簧1.346875，最终全域与sim2sim目标未完成。

## 2026-10-07 最新v534完成：当前阶段每点36回合，八点7/8

同v533候选，无网络更新，八点和两新爬行条件各36回合全存活。历史7/8，仅1m/s+.8 yaw.0702/stable0失败；.25/+.1与+.3均stable1通过。runtime通过、原快筛通过条件无丢失。单seed/DR0/阶段弹簧1.346875，bank未作解析门槛且误差较大，不晋升、不称完整压弯或sim2real。下一目标回到历史+.8缺口，不能靠爬行局部进展代替全域；全部作业结束，证据continuation_v534/final_report.json。

## 2026-10-07 v534评估中：历史八点+两慢转每点至少32回合

使用v533同一候选213516858f9429df455ca57e941e74ec655924e38f632ea1dc596e98bb2dc982，不更新网络。10命令×12副本，至少32回合，DR0/阶段弹簧1.346875/D反馈.05保持。查continuation_v534/evaluation_status.json真实PID，不重复启动。此为静态单seed阶段回合检查，不等于多seed/最高DR/实机认证；完整目标不变。

## 2026-10-07 最新v533完成：21/61，正慢转已获得局部快筛裕量

final SHA213516858f9429df455ca57e941e74ec655924e38f632ea1dc596e98bb2dc982，40更新211秒；82阶段实际leg scale对齐、freeze/runtime通过。同61点原20保留，新增.25/+.1通过（yaw.03131/stable1）；+.3 yaw.01920/stable1明显改善。无生存退化，quick retention=true，promoted=false。负-.3和静止仍失败；直行误差偏大，DR/最终弹簧/全域/正式sim2sim未完成。无运行任务；下一步保持一致映射修负慢转/静止后扩域，不能把v532旧MuJoCo结果当v533认证。证据continuation_v533/final_report.json。

## 2026-10-07 v533运行中：对齐PPO低速腿scale

发现训练阶段静止/.25使用.20，而固定eval/MuJoCo分别.18/.19；v532示范与验收均使用后者，20/61仍是有效快筛。v533只修训练stage0/1/2/7 scale，源v532候选，40更新后同61点retention。网络仍正向爬行隔离，静止/负向不更新；D反馈.05/b.002/弹簧1.346875/DR0不变。查continuation_v533/training_status.json与PID避免重复启动。最终技能域、弹簧、DR、sim2sim仍未完成。

## 2026-10-07 最新v532完成：Isaac20/61，sim2sim未通过

候选f6b74bfb351a559bbc45a5caa079c60367848b2b9273ab86760bae2a7aab4a0f，CPU示范初始化后无偏置61点快筛20通过、旧19全保留；新增.25/+.3 yaw误差.04971/stable.533边缘通过，+.1仍失败。runtime/freeze通过，promoted=false；中断后原评估完成、退出码未知，未重跑。

导出边界覆盖5450输入JIT/ONNX通过；同原rmuc_peak55.xml当前阶段契约MuJoCo两慢转12秒但严重过转，.5直行倾倒，不合格。记录同输入四consumer对照误差<=5.96e-7，仅推理一致。完整报告及controller/MJCF/export hashes在continuation_v532。无运行任务；保留20/61 Isaac候选，继续巩固命令域和跨引擎动力，不升DR或最终弹簧，不宣称部署成功。

## 2026-10-07 v532无偏置评估中：示范初始化而非继续PPO盲训

从v531同物理轨迹选择.25/+ .1与+.3改善动作，用CPU600步拟合现有正向physics.network、其余冻结。保持启动/ramp标签、按副本划分、验证时序/预算，轮MAE约.002；候选f6b74bfb351a559bbc45a5caa079c60367848b2b9273ab86760bae2a7aab4a0f，optimizer移除。正在GPU0跑无动作偏置61点retention，原19点不得丢失。只读并行审计已完成；先看continuation_v532/evaluation_status.json和PID，禁止重复启动。训练契约仍D反馈.05/b.002/弹簧1.346875/DR0，最终物理/DR/全域/部署远未认证。

## 2026-10-07 最新v531结束：19/61，正向慢转尚未修好

40更新209秒，final eeb2136f9c80bb64b2d0c0fc9c951f13bcda6d1de8bb95bdc989156031a999af，runtime/freeze通过。61点仍19无丢失，+.3 yaw误差降到.2595但失败，+.1 .0961失败；不晋升、不再同类PPO短训/std追加。所有进程结束；下一步考虑可行轨迹初始化的受限策略适应，禁止把固定诊断偏置当部署外环。D反馈.05/b.002/弹簧1.346875/DR0仍阶段条件，最终目标未完成。

## 2026-10-07 最新v530结束、v531运行中

v530最终998cbc10fe3b9f1cc1099a587bb6718d5f52cef4e80ca9789a6d421243f5fe86，40更新208秒，61点19保留无新增，正慢转仍失败；隔离确实保留负向/直行输出。中断后原评估继续完成，不重复跑，退出码未知但完整JSON/hash校验通过，详见evaluation_status.json。

v531唯一变化轮训练std.08→.16，40更新再同61点评估；正向更新隔离、物理/奖励/采样不变。查continuation_v531/training_status.json及真实PID禁止重复启动。std不是部署噪声，不扩电磁力矩或外环；最终全域/弹簧/DR/sim2sim未完成。

## 2026-10-07 v530运行中：正向爬行隔离

v527等价派生源4683426c8d19612fe707b61f90cd57ffd428a367066dfda0f0139ab5412bd260，只正向爬行physics更新，负向/直行参考冻结，预算不扩，旧checkpoint兼容。26测试/100输入初始parity通过。GPU0/40更新后自动61点retention，物理/reward/采样与v527相同，原19点需保留。查continuation_v530/training_status.json和真实PID，禁止重复启动。无最终报告前不晋升；全域/最终弹簧/DR/实机要求仍未完成。

## 2026-10-07 v529完成：慢转统计补齐，下一步隔离正负更新

补wheel_advantage_audit的.25m/s ±.1/±.3，9测试通过。固定v527源采集8轮，无actor/critic参数变化，snapshot及hash见continuation_v529/final_report.json。正+.3局部共享网络梯度需要减小右-左差分，全命令混合梯度却增加；这是正负干扰证据但未含Adam且重放均值差.000672，不能称唯一根因已证。下一步做正向慢转更新隔离并校验初始parity/负向和直行retention，不继续加共同奖励。未训练新候选，无运行作业；v527仍19/61、最终物理/DR/实机验收未完成。

## 2026-10-07 v528诊断完成：慢转有轮差分响应

v527候选在.25m/s目标+.3时，无偏置yaw约0；right action-.15/left+.15得到yaw+.2284且存活。目标-.3时right+.05/left-.05得到yaw-.3122且存活。腿单独偏置没修好正转。结论是当前策略未学到合适差分且正负响应不对称，不能宣称轮控物理失效或仅腿clip原因；人工偏置仅诊断，不部署、不晋升。旧快转域有专用网络/扩腿权限，低域旧采样/门控缺失，不能认为快转自然泛化到慢转。下一步针对PPO更新方向/探索与奖励核查，完整目标未完成；v528进程已结束，报告见final_report.json。

## 2026-10-07 最新v527已完成：19/61快筛，未晋升

final SHA 0ceab5c6c70bd6f15afb3ca886c6a54f8b8eff3a3351ace99c041e2b9c188d75，40更新212秒，runtime/freeze/coverage通过。原18点全部保留，新增.25直行通过但速度误差.137仍大；正向+.1/+.3 yaw误差.103/.298仍失败，静止存活0。无同预算reward对照，不将速度改善归因shaping。下一步查低速动作差分到yaw响应，不重复同reward盲训、不升弹簧/DR。无运行作业；仍是阶段弹簧1.346875、D反馈.05、机械b.002、DR0，部署D_comp=.052。全命令域/最终弹簧/DR/sim2sim正式验收未完成。

## 2026-10-07 v526完成诊断、v527奖励单变量训练

v526同契约600步、前500步轨迹：.25m/s五点新增残差截断率0（age>=125），静止轮截断约16.7%/2.9%；不扩大权限。v527从v525最终源直接续40更新，唯一配置变化dense small-yaw weight0→.5，物理/采样/网络保持，自动61点retention，旧18点不得退化。10项相关CPU测试通过。查看continuation_v527/training_status.json及PID避免重复启动；最终气弹簧、DR、全域、实机目标仍未完成。

## 2026-10-07 最新v525已完成，未晋升

final_checkpoint.pt SHA d98f05b1422ef8c61a6e97bcc2ca6f6325c8431ccf373765c452b113082bea0f；60更新305秒，runtime/freeze/采样覆盖通过。61点18通过，原17点全部保留，新增.25m/s五点全存活且仅-.1通过；新增点无旧基线，禁止声称因果增益。静止14次tilt、存活0，.25直行超速约.244，正转+.3实际yaw近0，未完成低速/起停。promoted=false，无运行训练。保持D反馈.05、机械b.002、弹簧1.346875、DR0，不继续同配置盲训；failure_diagnosis.json记录yaw归一化截断与残差预算待验证问题。D_comp=.052已核对；最终弹簧/DR/完整命令域/正式sim2sim目标继续有效。

## 2026-10-07 v525运行中：固定阶段物理补静止和.25m/s

v524快筛17/56保留后，本轮不加弹簧；补齐静止/.25直行/双向.1/.3的采样与可训练gate。新增有界低域层、旧图全冻，>=.45m/s输出受保护；源f0e0e90d771615c620d33c14fec07676adadb2569d84994aec0bf51f89b55a3e。D反馈.05/b.002/C.001/弹簧1.346875/DR0/2ms，60更新，自动61点quick retention。部署D_comp=.052已核对。查continuation_v525/training_status.json及PID，禁止重复启动；尚无最终行为结论，完整目标保持。

## 2026-10-07 最新v524已结束：小步弹簧迁移快筛保留，未正式晋升

从v512派生，弹簧1.0775→1.346875Nm/腿；最终SHA a6628a92319ebc7a92d7ec2b207587e75b6a238388be75ebcb2fb16408857c3b，60更新约298秒，56点至少4回合约178秒。runtime/freeze通过，共同网格17/56全部保留，无新增通过、无生存退化，quick retention=true，promoted=false。报告artifacts/continuation_v524/final_report.json；所有作业结束。先固定本档物理补静止/低速慢转弯，不同时提高弹簧和命令难度。静止仍倾倒，不能称低速交付就绪；最终气弹簧、DR、正式32回合、多seed、MuJoCo/实机验收未完成。26项相关CPU测试通过。

## 2026-10-07 v524接续：小步弹簧迁移与直行优先

v523失败不晋升；v524从v512派生源c04551159bf095cadfae09848786708709d6fe6ab86488ba0179480de5619ad9，弹簧1.346875Nm/腿，反馈D.05/b.002/C.001/DR0/2ms。两次预检正确阻断±.8零梯度目标；补齐positive/scoped门控交集校验（26测试通过），改active1.5直行、保留较低速直行及历史转弯rehearsal。GPU0预算60更新，自动freeze/runtime/56点快筛，retention失败不晋升。当前PID见artifacts/continuation_v524/training_status.json；先查进程禁止重复启动。最终气弹簧、DR、全域、部署仍未完成。首轮失败日志曾被错误覆盖，第二次归档preflight_attempt2，后续恢复独占创建日志。

## 2026-10-07 最新v523已完成：弹簧1.61625暂不晋升，无运行进程

60更新约5分钟，final_checkpoint.pt SHA08b3dbe1add3d3d22d54053c14ce5a5d870d150cf72facc9873ba08fabea431c；同56点16通过，旧v512共同网格17通过，新增1/+.8，丢失1/-.2与1.5/-.1，无新增存活退化。runtime/freeze通过，retention失败，promoted=false。report已修正基线64点混入5个额外通过点导致22的计数错误，正确17→16。8项physics测试通过；未启动后续改门控训练，不重复v523。按用户最新方向继续简单→难和弹簧逐级适应，当前首档需修复/缩小步长，不再扩展sim2sim诊断。DR/最终弹簧/起停/全域目标未完成。

## 2026-10-07 最新v523正在GPU0训练：弹簧1.61625首档，先查进程禁止重启

用户明确停止过度sim2sim排查，要求简单→难命令课程同时开始弹簧迁移。v522正向专训未启动已取消。当前唯一训练v523/run.py，PID见training_status.json、日志training.log；源v512冻结完整图新增零初始化有界层，SHA17448e430c389f6ef751af7968db4e0440a762bfcfbd623feda5154c7aaee2cf。spring1.0775→1.61625Nm，D.05/b.002/C.001/DR0/2ms保持，512env/60更新。先.5–1.5直行+已有1m/s转弯rehearsal，自动56点retention决定是否接下一档，不自动按轮数升弹簧或DR。当前尚无最终报告，不称完成。全部静止/起停/最终弹簧/全DR/全域/正式sim2sim要求继续保留。

## 2026-10-07 最新v520：1ms轮端对比结束，yaw缺口不变

同v512+imp999仅physics/轮2→1ms、leg2ms/policy8ms，.5直行和±1转弯均12s，yaw误差.046/.306/.174，近似原2ms。速度取样点审计确认Isaac和MuJoCo都取机身刚体COM，无需改成原点。v520全部证据已保存、无运行进程、不晋升；不要重复1ms/2ms试验或误称提高频率能解决过跟踪。DR/全域/最终弹簧与正式sim2sim仍未完成。

## 2026-10-07 最新v519：MuJoCo八点单次诊断完成，yaw全部失败

v512+v518/imp999/D.05/弹簧1.0775/DR0/2ms，补齐1m/s±.8/1/1.2/1.4；全部12s存活，负向yaw过跟踪.171–.182，正向.284–.364，速度误差负向.026–.075/正向.160–.246。六新点+复用±1证据在v519，非32回合/多seed验收，promoted=false。下一针对系统性过跟踪/方向不对称修复，不重复存活诊断，不升DR，不加部署任务外环。无运行任务。

## 2026-10-07 最新v518：发现动态闭链软度差异，保留imp999诊断候选

Isaac四闭链启动窗口最大分离.139mm，原MJCF1.445mm。仅将诊断副本equality solimp设常数.999后约.081mm；v512/.5直行由3.728s倒地变12s完整运行，速度/yaw误差.191/.046。±1转弯同候选12s，速度误差.201/.046、yaw误差.306/.174，yaw仍不合格，正转略退。原入口XML未改、不晋升。候选v518/imp999/model.xml+contract.json和完整报告保留；下一在实测软度对齐依据下继续处理yaw与全域，不回到盲训或把12s存活称认证。新增只读Isaac闭链距离记录，当前无运行进程。

## 2026-10-07 最新v517：闭链/接触单变量CPU诊断均未通过

USD主轮/被动连杆作者armature≈.01，null配置不代表0。v512/.5直行baseline3.728s失稳；只加硬闭链至4ms常数7.736s失稳且后半程倒车-1.247m/s；只加硬接触3.808s失稳。闭链误差1.445→.812mm，敏感但无充分修复。源MJCF与策略未改，三个诊断副本保存在v517，无运行任务。新增MuJoCo轨迹闭链/接触距离读回，下一对照Isaac动态闭合误差，禁止把延长存活当通过。完整目标继续未完成。

## 2026-10-07 最新v516：旧视频已找到对应6684记录

桌面视频fixed_mirror对应v423同名报告，其ONNX hash匹配export_locomotion_6684。旧1/+1的4–8秒速度误差.52057、yaw误差.32341；当前v512同窗.37207/.28044，不能称该点较旧视频退化，也不能称旧视频只差yaw。旧6684保留对照；当前.5失稳是真实缺口。v516只读模型库存比对质量/惯量/坐标/初态闭链近一致，未证明runtime动力一致。下一继续查剩余动力项，无新训练、promoted=false，完整报告已保存。

## 2026-10-07 最新v515：电磁执行链路同状态一致，剩余查物理项

v512/.5启动首16物理子步，Isaac输入重算controller，力矩最大差4.3335e-7Nm、目标差3.373e-9rad；原始数据evaluation.startup.pt/evaluation.json及audit.py/final_report在v515。只核实PD/电磁输出，不覆盖被动外力/闭链/接触。前一v514首8ms动力分歧依然真实，下一查剩余项；无运行训练，不重复同配置盲训，目标未完成。

## 2026-10-07 最新v514启动轨迹定位完成

同v512/.5直行、D.05/阶段弹簧1.0775/DR0/2ms，Isaac500步捕获与v513 MuJoCo逐步对齐。初始obs差1.192e-7，首8ms关节速度obs差.02691、腿位置obs差.004323；.4s Isaac前行+.0473而MJ后退-.0205。先有早期动力分歧再闭环放大，尚无唯一根因。数据与final_report在v514，不是完整episode验收；首启动参数拒绝留痕，重试完成，无运行进程。下一应查首动作物理子步执行器/闭链/接触，不能重复归咎导出或只加训练。

## 2026-10-07 最新v513：同输入链路通过，MuJoCo行为未通过

v512完整actor/JIT/ONNX/MuJoCo同输入3467帧误差≤4.768e-7，.5初始观测Isaac/MJ误差1.192e-7。修复显式2ms诊断consumer与腿PD调度（默认1ms不变，诊断必须certified=false），10项controller测试过。同D.05/阶段弹簧1.0775/DR0：1m/s±1转弯12s后半程yaw误差.279/.175；.5直行3.728s倾角/高度失败。全部promoted=false，不得称sim2sim通过。证据在artifacts/continuation_v513；训练数组与评估数组结构不同但所查低速插值腿尺度一致。下一优先查动力轨迹分歧，不能再只调reward或归咎导出。D_comp=.052常量存在，最终DR/1ms/弹簧10.775/多seed/部署仍缺口，无运行任务。

## 2026-10-06 最新v512局部候选完成验收：原56点17通过，未晋升

v509为主，局部平滑选用冻结v511的.5直行physics残差，两网络同预算；候选artifacts/continuation_v512/candidate.pt，SHA989306b78f2832fb89cad1fa0c3dfe0553e4736b9bfcb686ecbed51c61eca2f2。原56点17通过、旧16点全部保留；额外8边界点5过，.3/.4直行仍倒地，.5/+.05失败，静止失败。完整报告/retention/failed_conditions/readback在v512，无运行进程。7项隔离/physics/trace测试通过，推理候选无optimizer状态；ONNX、MuJoCo、动态命令切换、DR/最终动力未认证，promoted=false。下一须补连续低速稳定和正转，不能把局部融合当完整适应。

## 2026-10-06 最新v511已结束并拒绝晋升，保留v509

采样重分配40更新：正转65%、静止/.5直行权重3，其他v509配方/动力保持。56点15通过，新增.5直行，但丢失1m/s-.1/-.2两个旧通过点；静止仍倒地，无存活进一步退化。retention失败，promoted=false。最终SHA162f897892c2c2a044032e3e58bb408ed5f55c560b17b031fc9edf1915793360；v511目录完整证据齐全，无运行进程。下一需对薄弱域更新做技能隔离，禁止继续同采样配置盲训。候选仍v509的16/56，完整任务/DR/MuJoCo未完成。

## 2026-10-06 最新v510响应诊断完成，无运行训练

固定v509扫描六命令×五差分轮偏置，共30条件各2回合；1m/s+.3在bias=-.2可达实际yaw+.358且存活1，1.5m/s可达+.397，但.5m/s同偏置倾倒。证明当前策略周围存在正向响应，不是完全不可控；不可部署固定偏置，30条件均未全部达标。详细见v510/evaluation.json和final_report.json。下一应针对速度相关探索与低速稳定进行学习；v509静止已倒地、.5直行半数存活必须纳入止损，不能只用原通过点retention掩盖。保持v509候选和v501旧基线，完整目标、DR/MuJoCo仍未完成。

## 2026-10-06 最新v509已完成：16/56，新增5点，无旧通过点丢失

不要重启已完成的v509。100更新、512env、dense small-turn yaw weight2真实运行完成；最终checkpoint在artifacts/continuation_v509/final_checkpoint.pt，SHA256 5dcece5322d6240df3bc5895f5071e0a2cf74ec90c575eda4f7fe09a75c48a33。56点每点至少4回合快筛16/56，原v504的11点全部保留。新增(1,-.2)、(1,-.1)、(1.35,0)、(1.5,-.1)、(1.5,0)，survival/stable均1；正向小yaw、静止等仍失败，不可宣称全命令域适应。所有运行均已结束；final_report/freeze/retention/runtime/failed_conditions齐全，promoted=false。D.05/b.002/C.001/阶段弹簧1.0775/DR0/2ms不变，最终动力、DR范围、32回合、多seed、MuJoCo未认证。下一应从实际新增收益出发处理正负转不对称与静止失败，禁止重复已完成100更新或宣称训练无任何进展。

## 2026-10-06 最新v509正在GPU0训练，先查PID禁止重启

用户授权充分GPU并要求reward shaping。v508未启动已取消；v509实际运行命令command_retry.json、状态status_retry.json、日志training_retry.log，源v504SHAf1763c2c9c06512466c70148fb13d3038badebfee2f7178ae43b351c1f5130fa。512env、100更新10critic+90actor、64steps、4epochs、LR3e-4、wheelstd.08、新small-turn dense yaw weight2，speed奖励保持。D.05/弹簧1.0775/DR0/2ms不变，新低速physics可训、旧actor冻结。14命令coverage已过；首启动guard失败保留，重试显式allow_aggressive_highspeed。中途checkpoint须做行为止损、最终56点retention，不自动晋升。尚无最终结论，完整目标未完成。

## 2026-10-06 最新v507：腿目标裁剪假设未得到修复证据

v501同契约1m/s±.3，左active目标0/±.02rad偏移六条件全存活但yaw误差仍约.298–.302；轮载荷近均分、纵滑小。不可据此扩大部署腿限幅，也不能称接触是唯一根因。v507 final_report/完整contact指标齐全，未训练未晋升；下一若做响应诊断需看更充分差分轮力矩而非重复腿偏移。保留v501基线，当前无GPU任务，全域11/56和DR/MuJoCo缺口未变。

## 2026-10-06 最新v505/v506：小yaw不足不是简单的断门控/没更新

v505确认v504小yaw gate有效、optimizer lr1e-4/actor60步、reward日志clip0，初态轮差分权限双向充足。v506当前v501在.5/1/1.5×±.3的0/±.02差分轮偏置18条件全存活，但yaw只有约-.008到.001；偏置进入实际动作、电机不饱和，不能认定接口丢失或策略完全抵消。active-leg槽3在1mps+.3长期action1，下一查腿姿态/接触与轮力矩联合响应。没有新训练/晋升，无运行GPU任务；v505/v506 final_report完整。当前基线仍v501、56点11通过，全域/DR/最终契约未完成。

## 2026-10-06 最新v504：56点仍11通过，不晋升，无运行任务

从v501冻结全图新增域限定physics迁移分支，14低速命令真实覆盖，25更新仅6张量变化。最终SHAf1763c2c9c06512466c70148fb13d3038badebfee2f7178ae43b351c1f5130fa，56点仍11/56、无新增通过；小yaw仍近乎不转，部分直行小改善。大转弯指标逐值保持、原11点retention通过，但不延长此配置，保留v501基线。代码支持显式至两阶段冻结physics迁移，新update_domain小yaw域关闭不影响旧修正，总预算不扩；165输入零初始化一致/21测试通过。v504 final_report/failed_conditions/retention/runtime/freeze齐全。下一核对有效动作余量与学习信号，禁止把coverage通过当已适应；全域/DR/1ms/MuJoCo仍未完成。

## 2026-10-06 最新v503：下一训练须覆盖全低速域，禁止继续expert3-only

CPU完整v501同reset状态敏感性：1m/s yaw-.45→+.45轮差分action只变.009669，导数非零，命令未丢；小yaw的PositionAuthority gate=0，expert3和startup不覆盖。全域失败不能由继续+1.4专家训练修复。下一需站立/.5–1.5直行/双向小yaw的可训练门控和采样coverage，保留大转弯retention；原图冻结不代表动力迁移能力。v503 final_report齐全，无运行GPU任务。当前v501在56点仅11通过，未晋升；历史外环依赖仍需更强因果证据，不据旧参数存在直接下定论。

## 2026-10-06 最新v502：全域56点仅11通过，不能只修1.4

v501新有界启动残差候选SHA426fe8d628d39df7985903820e153205358b501d9a2012b6dde58bfed33ff8fe直接+1.4已4/4存活/yaw.01948/stable1，九点8/9；无计时器/部署外环，旧网络冻结，新预算与门控见rmuc_startup_adaptation.py和fit_report。仍未晋升。

用户要求全已学命令域，v502同候选56点各>=4完整回合raw11/56：历史40点仅1m/s及1.2m/s直行通过；八点转弯7/8通过（+.8失败）；额外探针.75及1.25直行通过。1m/s小角yaw几乎不响应，1.5欠速，2–2.9旧技能失配，站立漂移。当前D.05/b.002/C.001/弹簧1.0775/DR0/2ms，不能宣称全适应。历史v181非零任务反馈参数而当前关闭，要核对旧控制依赖，不只归因D。下一转向全域低速/小yaw迁移，持续保留全旧域retention；动态命令/32回合/multiseed/1ms/DR/MuJoCo未完成。v501/v502 final_report均完整，当前评估已结束，启动前检查PID。

## 2026-10-06 最新v500：停止以原expert3预算模仿启动

同reset观测，成功.25s进入首步右/左轮action=-.06419/.02360；直接+1.4下把local correction全预算用满，右轮最高仍-.41907、左轮最低.21904，差.35489/.19544。local轮预算±.06、hip0，无法复现该成功首动作。不是证明没有其他解，不称唯一根因。v500为CPU容量诊断，未改预算/权重，无训练；final_report齐全。下一需有界且只作用启动状态的策略适应能力与retention设计，不能继续原expert3-only模仿/PPO盲加更新。源v482/v494保留，部署/DR/全域仍未完成。

## 2026-10-06 最新：v499重放差异已定位为TF32/批量数值模式

GPU0纯快照重放，无仿真/训练：相同128batch+TF32与记录动作均值最大误差0，关闭TF32重现.000955差异。参数梯度同精度核验完成，early/middle正相关.519，early/late负相关-.379、middle/late-.568，但合成方向仍与early正相关.626，不能宣称冲突抵消了全部启动学习或据此盲加LR。v499 final_report最新状态completed_capture_and_verified_replay，precision_replay/parameter_gradient_training_precision为证据；早期“未核验”记录仅历史。没有新行为改善，当前仍保留v482/v494，完整目标未完成。

## 2026-10-06 最新v499：capture-only已结束，原始优势可复用

v499从v494采3批，actor/critic逐tensor未变，16384样本与episode age、actions/means/std/advantages已保存在ppo_gradient_snapshot.pt。CPU参数梯度重放失败：mean最大误差.000955>2e-5，不得把日志中的梯度夹角当已验证因果根因。final_report明确梯度未验证。下一从已有快照查CPU/GPU重放差异，不重复GPU采集、不据未验证梯度继续短训。源v482/v494保留，当前无运行训练，D完整适应仍未完成。

## 2026-10-06 最新v498：CPU排除采样状态完全断梯度，未找到唯一根因

v493真实失败16状态上，v494/v497的expert3 gate均1，legs/wheels四输出对expert3参数Jacobian均非零。v497相对v494同输入动作变化腿最大.004824、轮.000575，无法判断PPO方向正确。v498最终报告齐全，无GPU任务。局部expert原来不支持rollout capture，现rmuc_position_authority_training.py已扩展显式RMUC_CAPTURE_PPO_ROLLOUT与ONLY至local，默认关闭；15既有测试通过，但尚未运行native local capture。下一先采启动优势/梯度证据，不继续同配置堆更新。保留v482基线/v494课程点，v497拒绝，全部目标仍未完成。

## 2026-10-06 最新v497失败结项，无运行训练

v496证实v494在0等待+.25s yaw ramp下+1.4四回合全存活、yaw.031472/stable1。随后v497混合50%直接/50%.25s进入完成25更新，只有expert3变化，runtime/freeze/七保护点retention通过；直接命令仍7/9、+1.4存活0，SHA8895e92fc91c1b5e5062ff88f7bd8ffd83a46a348ea26c9341c979b012c61145不晋升、不延长。保留v482与v494，下一检查真实启动学习信号/优势/梯度，而非继续扫描进入时长。v496与v497 final_report均完整；代码新增abrupt_fraction默认0、26相关测试通过。D全适应、1ms最终验收、DR、全域及MuJoCo行为仍未完成。

## 2026-10-06 最新v495：.5s等待+.5s进入通过，无新训练

源v494 SHA d087f0bbb032d625399e339e41a9ef69233f855234b8e2124cbe8c5e1df78742，+1.4 hold.5/ramp.5，4/4存活、速度误差.017053/yaw.039466/stable1；2ms/D.05/弹簧1.0775/DR0不变。只有单seed诊断，不晋升；直接启动仍未恢复。下一继续缩短进入边界，不对已通过档重复PPO。artifacts/continuation_v495/{evaluation_command,probe,final_report,status}.json保存完整记录，进程已终止。v482基线继续保留，D全适应/DR/MuJoCo正式行为均未完成。

## 2026-10-06 最新v494：训练已终止，保留进入课程中间候选

v494从v482完成25更新（10critic+15expert3），2ms/D.05/弹簧1.0775/DR0，hold1s+ramp1s。SHA d087f0bbb032d625399e339e41a9ef69233f855234b8e2124cbe8c5e1df78742，仅expert3变化，runtime/freeze/七点retention通过；直接进入九点仍7/9、+1.4存活0。entry_eval为4/4存活、速度.014855/yaw.042054/stable.778428，课程诊断通过但未晋升。保留v482基线与v494中间checkpoint；下一先短测更短进入边界再决定是否续训，不盲加同档更新。默认训练进入时长为0，只有显式CLI启用；25项相关测试通过。最新报告artifacts/continuation_v494/final_report.json，完整目标未完成。

## 2026-10-06 最新v493：成对证据确认进入方式决定当前+1.4成败

v493直接进入0/4存活，对照v492先直行1秒再yaw ramp1秒4/4存活且跟踪通过。环境origins/root位姿、checkpoint、runtime与物理读回、源码hash全部一致；仅命令进入不同。源仍v482，physics2ms/D.05/弹簧1.0775/DR0，未晋升。下一优先从渐进入可存活状态构建课程并退火到直接进入，同时保留七点retention；最终必须突变/动态命令、1ms、多seed、完整域通过。v493/final_report齐全，任务已结束，启动前查PID。不可将4回合诊断写为D=.05全部适应。

## 2026-10-06 最新v492：渐进进入+1.4有可存活起点

源v482不变，2ms/D=.05/弹簧1.0775/DR0，先1秒直行再1秒yaw ramp至+1.4，4/4存活，速度误差.05784/yaw.01629/stable1。仅诊断，不晋升；与v491直接进入失败的布局不同，不能过度归因。下一步同布局比较突变与渐进入，再从可存活进入课程退火，不再同配置局部PPO盲加轮数。evaluator --diagnostic-turn-entry 明示诊断、report总体false；最终突变/动态命令与完整retention仍必须过。v492 final_report齐全，当前该评估已终止。

## 2026-10-06 最新：v491真2ms完成，仍保留v482，D_comp阶段目标.052

无须重跑v488：固定方向采样已恢复+1.4占比30.2%，但仍7/9，不晋升。v489 MuJoCo±1各12秒均跟踪失败，推理parity通过不代表行为。v490是假2ms：构造函数覆盖到1ms，报告明确无效。现修复显式separated_wheel_physics_dt_s与evaluator实际周期断言，54测试通过。v491真2ms九点7/9、七保护点保持通过；+.8 yaw.07882/stable0；+1.4仅1/29存活，不可挑幸存样本stable1当成功。允许下一次必要训练用显式2ms加速，policy8ms/腿PD2ms保持，最终回1ms验收。源v482 SHA099e9a4b9ceed9eaf027c6ee2a336048ca4acabcbc02d10ed641f3c3b53dc322，当前D.05/弹簧1.0775/DR0，完整适应未完成。

部署新增RMUC_STAGE_D_COMP_NM_S_RAD=.052，用户要求对齐训练.05+.002的未饱和总系数，C测试通过。实机b按零才用全额.052；若有b_real则扣除，饱和区与训练被动项限幅外不同，不能称精确等效。实际左右参数仍由校准调用者显式设置。更新契约及deploy README，不改历史checkpoint/报告。不启动同配置无效长训，接续先查PID及最新目录。

## 2026-10-06 最新结项：v487不晋升，无本项目运行任务

v487完成25更新，runtime/freeze通过，只改expert3。九点4副本仍7/9，+.8跟踪失败、+1.4存活0，七保护点主要指标与v482逐值一致。候选SHAf64659b1db05e9c9b0c31022283e270a79b2e0f9f0a336fdc594d4ccd3a18a0a，promoted=false，禁止自动延长此配置；继续保留v482。实际训练command_retry2.json、实际评估evaluation_command_retry.json、最终final_report.json与completion_status.json优先于早期status。采样发现早倒正向在重置重抽方向后被长存活负向占据，目标样本约8%；若重训须先处理这一已知效率问题并定位启动闭环。全域、正式32回合、DR/MuJoCo行为均未完成。详见improvement及v487证据。

## 2026-10-06 v487正在运行：先核对进程，禁止重复启动

实际命令artifacts/continuation_v487/command_retry2.json，状态status_retry2.json，日志training_retry2.log。源v482，D=.05/弹簧1.0775/DR0，10critic warmup+15expert3局部更新；physics映射/其他actor冻结，不加腿补偿。首次参数解析失败、第二次纯正向采样被双向coverage拦截，均保留日志；目前双向采样已通过前3次coverage。运行结束先执行audit_training.py验证25更新、冻结及实际参数，然后按evaluation_command.json做九点短筛。未有最终报告不得视作完成。当前工程训练readback已覆盖local authority，15项CPU测试通过。

## 2026-10-06 最新：v486诊断结束，禁止把辅助存活当策略通过

v486在v482/D=.05/弹簧1.0775/DR0下，仅增加+.0359167rad腿目标诊断偏移，+1.4完成4/4存活，但yaw=.0751959、stable=0、bank=14.9564deg，未通过。final_report已补齐，promoted=false，无新训练。继续保留v482原checkpoint（7/9快速诊断）；v481/v485拒绝续训。D=.080625当前闭环也有早期pitch发散，故必须先隔离弹簧坐标/轮控/previous-action反馈差异，不能声称单纯D退火即可解决。后续仅在有成对证据与成功锚点时小步退火；任何诊断腿偏移不得进入部署验收。完整命令域及MuJoCo认证未完成。

## 2026-10-06 最新：保留v482，拒绝v485，当前无训练

v483高D参考在正确现物理下+1.4也倒，不能直接把高D数值当成功退火起点。v484诊断+.035917rad active-link目标偏移400步无tilt，但0完整episode；零与反号偏移均反复tilt，非行为通过。v485把偏移拟合进既有+1.4局部expert3，8输出张量改动/其余输出和11组保护命令逐位保持/预算不扩，但+1.4最终4/4倾倒，promoted=false，禁止续该候选。post-policy目标偏移不改上一动作观测，嵌入策略输出会改，previous-action反事实扰动引起源输出最大.1142 action；末期重构yaw门控也关闭。不能忽略此闭环差异或宣称已找唯一根因。当前唯一保留的D=.05诊断候选仍v482 SHA099e9a4b9ceed9eaf027c6ee2a336048ca4acabcbc02d10ed641f3c3b53dc322（7/9，+.8/+1.4未过）；v483-v485报告齐全。继续前查PID，下一修复先核对previous-action反馈和真实门控，不重复已失败配置。

## 2026-10-06 v483已完成，v484仅短诊断

v483发现当前正确物理下高D=.080625的冻结源+1.4也倾倒（首例174步），v482 D=.05首例139步；前71步pitch几乎一致（约20度），早期拟合误差多<.06Nm，后段才变大。因此不能盲目以旧D数值为“已通过起点”。v484正在采0/±.035917rad active-link目标偏移对照（按弹簧旧新初始轴差2.155Nm/PD60计算），6env400步，D=.05/弹簧1.0775、其余契约保持，不训练、不直接部署、不作技能晋升。接续读v484/status及probe，保留v482七点诊断候选和完整目标。

## 2026-10-06 最新结项：v482为7/9诊断候选

当前本项目GPU任务已结束，接续先查PID。v482候选SHA099e9a4b9ceed9eaf027c6ee2a336048ca4acabcbc02d10ed641f3c3b53dc322在activeD=.05/阶段弹簧1.0775/b=.002/C=.001/DR0九点4副本raw7/9；直行、-.8、±1、±1.2、-1.4通过，+.8 yaw=.07075/stable0、+1.4存活0，promoted=false。v481直接PPO2/9且增加正向倾倒，禁止从v481续训。v482 final_report/retention_audit/quick_eval齐全，仅诊断，不是正式32、D随机范围、全域或MuJoCo认证。

用户再次要求避免无效试验：先对照根因，再D小步退火。旧v430与当前除了D还有2→1ms、旧IdealPD→active_motor+b/C、弹簧wxyz→xyzw、DR/seed差异；原四元数问题已在历史文档记录。v482目录保存runtime差异及纯函数方向对照，不能凭名义1.0775称原弹簧动力完全相同，也不能回退错误轴作为部署方案。下一动作针对+.8/+1.4，不再无条件全局PPO；完整目标仍D=.05后迁10.775以及域内各速度压弯/起停/中间速度/MuJoCo。

## 2026-10-06 最新：v481失败，v482评估中

v481九点仅2/9（直行、-.8）通过，+.8存活.1111、+1/+1.2/+1.4存活0，负向其余yaw失败；不可继续从此checkpoint训练或晋升。源v480的补充四点显示+1.4在D=.05本来就倒，其余存活，说明v481还引入额外正向退化。v482用CPU轮力矩匹配初始化新physics六张量，源旧图与腿输出逐位保持、预算不增/不加部署外环，候选SHA099e9a4b9ceed9eaf027c6ee2a336048ca4acabcbc02d10ed641f3c3b53dc322。当前activeD=.05、弹簧1.0775、b=.002/C=.001、DR0、5.5/3、1/2/8ms，GPU0九点快筛中；先查PID和quick_eval，未有最终报告不能称通过。CPU真实输入力矩误差小不证明行为；源码/contract/command保存在artifacts/continuation_v482。完整目标仍是D迁移后10.775弹簧、全命令域及MuJoCo，不缩小为九点。

v480新增CPU MuJoCo同契约基线：1m/s±1各12秒均不通过，速度/yaw误差正.49671/.33384、负.63714/.23001，无轮饱和。3000观测actor/JIT/ONNX/recorded raw parity最大2.384e-7通过，只证推理链一致。证据在artifacts/continuation_v480，v481仍训练中，勿将基线当候选通过。

## 2026-10-06 v481最新执行入口

当前GPU0 trainer1674190，artifacts/continuation_v481/run.py负责25更新及自动九点短筛；启动前/续接必须查真实PID勿重复。固定阶段弹簧1.0775Nm、activeD=.05、b=.002/C=.001、DR0、5.5/3、1/2/8ms、PD60/2/±40、外环关闭，源SHA423570223dc22e478d1d8fd07e2e82807371a39d3c46f5bf58ce3815da486b35。只训练physics残差六张量、旧图冻结预算不扩，1200s预算，先看final_report/retention而非reward。用户强调整个0–1.5命令域掌握：已备69点domain_grid/命令，另需起停与连续切换，尚未执行；八点快筛不是全域认证。完整目标先D=.05，再10.775Nm弹簧，再域内全速度压弯和同checkpoint MuJoCo，未完成项不缩小。

## 2026-10-06 历史八点与新短训门槛

v430历史八点为1m/s双向.8/1/1.2/1.4rad/s，每点32回合，阶段1.0775Nm弹簧/高D=.080625/旧局部DR=.05下tracking8/8；v431/v433旧DR=.1时虽8/8存活但tracking只有4/8，四个正向点失败。v432同策略MuJoCo八点yaw全部未通过。v481在D=.05短训后自动筛直行+八点，4副本只是诊断，不能替代32回合或完整1m/1.5m速度域。

## 2026-10-06 v480完成：D=.05仍未适应

冻结v430源在原阶段弹簧1.0775Nm/腿、activeD=.05、b=.002/C=.001、DR0下五点4副本2400步均存活，但只有直行通过；双向±.8/±1 yaw误差.30581/.09831/.27403/.13340，stable均0。v480/final_report.json为`promoted=false`，不能写成D=.05完成；下一步有界短训并保留双向retention，目标还包括之后迁移10.775Nm弹簧与1m/1.5m全转速域、起停中间速度及MuJoCo。

## 2026-10-06 五小时优先级更新

用户确认保留v430阶段1.0775Nm/腿等效弹簧律，优先迁移主动D=.080625→.05。该阶段弹簧不是最终10.775Nm/腿交付条件。v479终值弹簧探针按新优先级提前停止，没有完整评估JSON，不能下结论；v480准备冻结源在1.0775Nm弹簧、activeD=.05、b=.002/C=.001、DR0、5.5/3、1/2/8ms五点同条件快筛，结果见artifacts/continuation_v480，勿按准备状态当通过。

## 2026-10-06 最新入口：v477未晋升，v478证明负向退化来自更新

v477已完成50更新及五点快筛，仅直行和+1.0通过，负向-.8/-1.0 yaw误差.1621/.1568且stable0；`promoted=false`，不要从其final checkpoint续训。v478同源冻结v430在同一1.5Nm弹簧/activeD=.080625/b=.002/C=.001/DR0条件下直行、-.8、-1.0通过，负向yaw误差.01538/.02221且stable1；+.8/+1.0仍失败。故v477的负向技能退化是PPO更新引入，不能怪弹簧小步本身。当前保留`artifacts/continuation_v476/frozen_source_migration.pt` SHA423570223dc22e478d1d8fd07e2e82807371a39d3c46f5bf58ce3815da486b35，五点仍是短筛，尚未到最终10.775Nm/D=.05；后续先隔离弹簧边界和正向修复，禁止自动升DR。见v477/final_report.json及v478/source_at_spring15.json。

## 2026-10-06 v477最新执行入口

首次启动误触高速guard已修复（扩展level49实际1m/s，不能只按编号判断）；7项测试通过，失败日志在attempt_01。当前重启PID1600428已进入PPO，runtime_verified.json通过，勿重复启动。读取实际status，不把CPU修复或首轮运行称通过。

v476五点全部存活；负向-.8/-1yaw误差.01376/.01794、stable1，正向+.8/+1误差.08078/.13342未过。这是active D=.080625/阶段弹簧1.0775的可学起点，不是交付。v477从完整冻结v430的新零初始化头迁移，源SHA423570223dc22e478d1d8fd07e2e82807371a39d3c46f5bf58ce3815da486b35；唯一物理增量弹簧1.5，其他同v476，源观测yaw clip2保持。50更新/5critic warmup/1200s、主动±1与30%直行/±.8 replay，旧784张量冻结/预算不扩。command/status/evaluation_command位于artifacts/continuation_v477，GPU0勿重复启动；结束自动五点快筛，再依据实测结果决定下一小步，不自动晋升最终D/弹簧/DR。

## 2026-10-06 v476最新入口

v475完成：原v430在当前物理同样失效，+1.4倾倒、±.3不转，不能靠换旧权重解决。v476在v467旧高D/小弹簧条件下只改变D施加位置为active_motor_v1：D=.080625/b=.002/C=.001/弹簧阶段1.0775Nm、5.5/3、1/2/8ms、DR0，五点4副本2400步600s，PID读artifacts/continuation_v476/status.json。只有此处仍有有效yaw和存活才进入小步迁移，不把阶段条件冒充最终10.775Nm/D=.05交付。CPU零初始化迁移已备好frozen_source_migration.pt，完整v430源784张量保留、2156输入输出差0、预算不扩，尚未训练。读probe.json后再决策。

## 2026-10-06 v475当前入口：旧网络对照

目标回归≤1.5全速度压弯：1m/s双向1–2.5rad/s、1.5m/s双向.8–1.8rad/s，小yaw/中间速度/起停也必须覆盖。artifacts/continuation_v475/delivery_target_grid.json为规划非认证。CPU网络审计发现v430的8个local专家在中间版本已变，近期冻结不能代表旧网全保留；position-authority/local分支在1m±.3、1.5m门控0，在1m±1.4门控1，不可盲扩门控。

v475正对原v430/model6835 SHA5960533edf88c8a08a564ab86ca19e679faa49b44074f6cb3ba9873cfa1ad502作当前固定物理7点对照：activeD=.05/b=.002/C=.001、5.5/3、10.775Nm近似、1/2/8ms、DR0/无外环，28env1800steps600s。当前无新训练，PID以该目录status.json为准勿重复。只替换actor，等待old_actor_current_physics.json，不预设旧网能在新物理通过。

## 2026-10-06 v474已结束（覆盖下文“正在训练”状态）

50更新和12点快筛完成，未晋升。固定主动D=.05快筛仅1m直行raw通过（速度误差.06540）；±.3慢转yaw误差.29516/.30511、stable0，扩大wheel std=.18仍未学会确定性转向。零命令存活但漂移速度误差.67436，不能称站立合格；±4自转未学会。全部12点存活只属短筛，不代表D[.04,.06]或联合DR验收。checkpoint SHA760b9a412e10399785141e03787614c992bad4695f9d0addbc383814857f71c7，freeze/coverage通过。完整结果artifacts/continuation_v474/{final_report,quick_eval}.json；不重复同配方长训，下一步分析双向PPO信用分配和命令条件化，尚无最终部署候选。

## 2026-10-06 v474最新入口

v473已结项：轮差0/±.06/±.15十组均4/4存活，零bias无yaw，±.15可产生约+.66/-.89rad/s；物理双向yaw通路存在，弱bias响应弱，不证明摩擦阈值。probe/final_report保留，无晋升。v474正在GPU0接model7050训练，源SHA19e6c1ab146042b733d7c47e019c42d75bf32671dc61be4ac853ecc2b299040a，唯一参数变化wheel exploration std .06→.18；50更新/5critic warmup/1200s、原70/30采样与physics残差限幅/冻结保持。主动D[.04,.06]、b=.002/C=.001、5.5peak/scale与3热参考、10.775Nm近似、1/2/8ms不改，其余DR关闭。status/command/contract在artifacts/continuation_v474，勿重复启动；结束自动12点筛查，未运行正式32、多seed/联合DR/MuJoCo/实机认证。先前“禁止评估interrupted checkpoint”只限制晋升，不妨碍明确诊断。

## 2026-10-06 v473当前诊断入口

v472 freeze源对比需要计入显式action_std_override：std实际[.03,.03,.06,.03,.03,.06]，是启动变化，非PPO偷偷修改。结项run.py及freeze_audit.json已修复，只豁免与命令逐值一致的std；六个physics张量是训练变化。旧训练仍停止。

v472已停；准确计数为149个rollout update，最后checkpoint model7050对源iteration差145。已补freeze_audit.json，仅允许的六个physics张量变化。先前“禁止评估”改为“禁止晋升，允许显式故障诊断”。慢转±.3被wheel advantage审计遗漏，代码现已补齐，8测试通过；尚不能将此日志遗漏当训练失败根因。LR1e-4实际正确，外环关闭动作没有强制轮同步。

v473为model7050冻结轮差探针，2s后1s ramp叠加0/±.06/±.15，1m/s±.3、40env×1800步/600s、主动D=.05固定/b=.002/C=.001/DR0，5.5/3、10.775Nm近似及1/2/8ms保持。artifacts/continuation_v473/{command,experiment_contract,status,probe}.json为入口；运行前先查进程勿重复。无训练/无晋升，等结果区分物理响应与PPO探索/信用分配问题。

## 2026-10-06 v472长适应入口（优先读取）

v472已提前停止，禁止重启或把interrupted checkpoint当候选。145/300更新，命令覆盖和冻结审计通过，D实际[.04,.06]，但±.3 yaw误差EMA=.2973/.3028、actual aligned yaw=-.0005、wheel differential action=.0010，转向没有学习。`interrupted_model_7050.pt` SHA256`19e6c1ab146042b733d7c47e019c42d75bf32671dc61be4ac853ecc2b299040a`只作诊断，`final_report.json=stopped_no_promotion`；grid/32/ONNX/MuJoCo/多seed/联合DR/实机都未运行。下一步先诊断physics residual无法形成有效轮差的原因，再改变训练通道；不再增加该配方更新数。

v471结束，12点仅1m直行通过，慢转yaw误差约.300，未晋升；MuJoCo五项完成运行而非行为验收。当前新任务为GPU0共享的v472，入口artifacts/continuation_v472/run.py，PID/status读取该目录status.json，勿重复启动。源v458 SHA256 7565b5d972c3b3d807078d1ae86b0e4ff65f8f1d19ee1b6b164ddb59f05aa642，300更新/5400s上限、5critic warmup、128env×128steps，physics residual独训，冻结旧图/不扩预算；目标1m/s±.3占70%，站立/.5/1直行rehearsal30%，wheel std=.06，课程不自动升级。

主动D=.05且DR[.04,.06]，机械b=.002/C=.001固定，strength1只指D范围，其余DR关闭；非饱和总D[.042,.062]。执行保持active_motor_v1、5.5峰值/scale、3热参考、1/2/8ms、腿PD60/2±40、10.775Nm/腿近似、无径向/firmware spring/任务外环。command.json与experiment_contract.json保存完整参数；结束自动12点快筛，formal_evaluation_command.json尚未运行。部署六项C边界测试及33项Python测试通过，不构成策略认证。训练后必须读freeze/runtime/evaluation和failed_conditions，再决定是否加联合DR或技能。

## 2026-10-06 v471执行入口（优先于下文v470静态状态）

唯一新增GPU0训练为`active_d05_low_speed_physics_v471`，PID以`artifacts/continuation_v471/status.json`为准，续接前勿重复启动。源v458 SHA`7565b5d972c3b3d807078d1ae86b0e4ff65f8f1d19ee1b6b164ddb59f05aa642`，v470同固定D/弹簧/5.5峰值契约，DR0，不做新弹簧模型。已选覆盖低速目标的physics residual，旧图冻结/预算不扩大，站立、.5/1直行、1m±.3实际训练；rehearsal start=0、fraction=.6，reward turn eligibility=0。预算25更新含5critic warmup，自动12点快筛、不自动晋升；完整命令、契约、preflight、runtime_verified均已落盘。启动后前三轮五命令全覆盖且物理参数读回通过，尚不代表技能通过。

修复了CPU preflight遗漏rehearsal起始level，以及未拒绝目标门控全零的训练配置，13测试通过。后续看最终checkpoint、freeze/retention、快筛和MuJoCo结果决定下一段，不复用v469局部index3慢转配方。正式32回合、D随机范围、联合DR、旧八点retention和实机仍待验证。

## 2026-10-06 当前入口：v470主动阻尼，缩减弹簧工作

结项结果：v470只做固定候选诊断，未训练、未晋升，当前没有本轮训练/评估进程。12点仅1m/s直行raw通过（4/4，速度误差.04023），站立46/46 tilt，慢转及±4没有有效yaw跟踪。MuJoCo三条12s均跑完但转弯/自转失败；同输入72组actor/JIT/ONNX误差≤2.384e-7。正式32回合未启动，未来D随机范围及联合DR未实施。完整结论`artifacts/continuation_v470/final_report.json`。首次MuJoCo因issaclab环境缺onnxruntime退出，原日志保留；使用既有robot环境独立重跑成功，命令见`mujoco_commands_retry.json`。

下一训练不能复用v469的local index3：实际中心+1.4、半宽.15，目标±.3门控为0；1.5m/s/自转同样不激活。已有受限PhysicsAdaptation覆盖这些命令，先用此通道修站立/低速yaw，并设置真正启用的低级rehearsal，启动前做coverage与freeze检查。D由新主动反馈提供，保持本轮同契约；不得把历史高D/不同弹簧下已学技能直接当新契约retention。v468/v469已补最终报告，前者没有合格行为，后者中止无最终checkpoint。

用户改为部署端补主动轮阻尼：机械b先验仍[.001,.005]，未饱和总D目标[.04,.06]，不再继续向纯机械低D下探。反馈D是新设计的1ms低层控制项，不是已辨识的旧固件D，也不是任务级速度/pitch/yaw外环。新显式模式`active_motor_v1`将`-D_fb*omega`加入电磁限幅前，物理b/C在限幅后；净轴力矩不再限幅。Isaac、MuJoCo及C核心已加入此可选路径，历史默认`postclip_loss_v1`保留，旧证据不改写。

v470源`artifacts/continuation_v458/final_checkpoint.pt`，SHA`7565b5d972c3b3d807078d1ae86b0e4ff65f8f1d19ee1b6b164ddb59f05aa642`。短筛固定D_fb=.048/b=.002/C=.001、DR0、10.775Nm/腿旧恒等效近似、firmware_spring关闭/径向0、5.5峰值及scale/3持续热参考、1/2/8ms、PD60/2/±40、外环关闭。唯一干预为新主动阻尼，48env/2400步/600s上限，不训练；运行及结论读取`artifacts/continuation_v470/status.json`、`final_report.json`。CPU86测试和C编译测试通过不构成策略验收。预备32回合命令尚未执行；未来D_fb[.039,.055]+b[.001,.005]才覆盖总D[.04,.06]，饱和时不成立。

气弹簧暂停详细建模，10.775近似不是已辨识实机值，仍保留最小扰动验证。v468的.1缩放探针没有完整episode且±.4 yaw几乎未跟踪；v469因命令覆盖缺失中止，无最终候选。无需继续原配方或重复简单探针。七小时预算优先低速可靠候选与正式验收；2h内无候选则缩小技能，末2h保留验收，不承诺≤1.5压弯和±4“全鲁棒性”。主要参数±20%、观测/通信各1–3周期、旧八点retention、多seed、MuJoCo、controller实机replay仍未完成。详见improvement顶部和部署契约顶部。

## 2026-10-05 v461-v464 接续入口

最新v464固定方向cohort训练首次尝试仅6/25更新后无traceback退出、无checkpoint；独立重跑25/25完成，候选`logs/rsl_rl/rmuc_wheel_leg_course/2026-10-05_14-23-00_turn_fixed_sign_cohorts_retry_v464/model_6930.pt`，SHA`e8145191f54d5eb589b4243edcf1abe488950caff519cfa12cd7e32702942e01`，源v458 SHA`7565b5d972c3b3d807078d1ae86b0e4ff65f8f1d19ee1b6b164ddb59f05aa642`。唯一训练变量为固定方向cohort；DR0、peak/scale5.5Nm、continuous3Nm仅热参考、b=.002、C=.001、D_fb=0诊断、弹簧10.775Nm/腿、1/2/8ms。最终+.8/- .8控制步3078/8275，正向绝对控制步仍几乎未增；DR0十二点快筛1/12，+.8 83/83、+1.0 87/87 tilt，负向-1.0亦tilt。`promoted=false`，停止同配置长训、正式验收及升DR。证据`artifacts/continuation_v464/{final_report,quick_eval,command_retry,experiment_contract,evaluation_command}.json`。

v430历史+1rad通过只适用于旧高D=.080625和1.0775Nm阶段弹簧，实际外倾；v447新postclip轮动力已使同源策略正向终止，不能当最终物理成功。v462把旧高D加到v458最终弹簧仍+.8全部51次tilt。现有旧轮damping DR此前缩放零值；v461已把每轮机械b的[.001,.005]采样接到真实受力并原生读回，C=.001和反馈D=0仍固定，反馈D硬件绝对值未知，最高联合DR未完成。旧固件内部左右符号一致；旧±8192限幅按未校准系数约2.45Nm等效命令、5ms上层、旧腿非PD，必须实机确认，不能靠DR掩盖。

v463新增从训练配置生成带hash的MuJoCo controller v2 contract，修复1.5m/s腿尺度旧固定.22与Isaac .24不符；同checkpoint修正后1.5m/s,+.8 MuJoCo后半速度/yaw误差.808/.700，行为失败。`sim2real_RL/deploy/rmuc_low_level.{h,c}`是独立C参考核心，1ms轮、2ms腿PD60/2/±40、8ms动作、5.5峰值/3热参考，需已校准counts/Nm、符号和接口才输出；未接旧固件或实机。recovery/jump暂缓，当前无合格最终locomotion、最高DR、正式MuJoCo多seed、实机replay或部署证书。

本轮`artifacts/continuation_v464/hardware_profile_audit.json`只证明schema3目标配置算术一致，`measured_calibration=false`、`hardware_certified=false`；反馈D、符号和库仑平滑仍未辨识。128项相关Python测试及C核心编译/测试通过。下一训练不能重复v459/v460/v464同物理残差PPO配置；要先针对正向早倒证明动作通道/课程初态的可学习性，再做有界训练和双向retention。不可因为用户允许长训就无条件拉长失败配置。

## 当前接续入口（2026-10-05，v460已结项）

只使用/修改本工作区；`/home/kellen/wheeled-legged_RL`仅参考。当前无本项目训练/评估进程，启动前仍需重新查PID。用户剩余约两天半，优先可靠的实机/DR和同契约sim2sim；可牺牲技能范围，recovery、jump及全速度暂缓。v278旧条件基线保留；v458/v459/v460均非当前最终硬件/DR合格策略。

最新v459从v458源训练physics residual head 25更新，正负episode50%，候选SHA`a0894f8937474bb8e997b857329f12590926d6938f606b61ae9ee3291748cd9f`，DR0十二点快筛1/12、正向+.8/+1.0全倒。v460的95%正向初次尝试因负向保护命令缺样在critic warmup停止、无checkpoint；80%独立重跑25更新，候选`artifacts/continuation_v460/final_checkpoint.pt`，SHA`87650cf56ea9cb7492293e4e2236d52f4d4d0cf716771041c493f2f5f1b95bbe`，冻结审计通过，但同条件快筛仍1/12：正向+.8/+1.0分别83/86次全tilt，1.5跟踪、自转均不通过。第一次v460评估因代理中断没有JSON，正式结论是`artifacts/continuation_v460/quick_eval_retry.json`。`artifacts/continuation_v459/final_report.json`、`artifacts/continuation_v460/final_report.json`均明确`promoted=false`；不得升DR、实机部署或称sim2sim已通过。

已定位采样问题：v459按episode50%仍只有+.8 1791控制样本、-.8 30165，因为正向约68–72步倾倒；v460按episode80%使+.8达3140但-.8仍11929，闭环没改善。固定轮共同减扭+0.2可让+.8四回合存活，但仅约.24m/s且yaw误差.395rad/s，不是策略能力或可部署外环。下个工作必须先检查早期倾倒轨迹的PPO学习贡献，并修控制步/存活时间分配或早期稳定机制，再做单变量名义训练和双向retention。只有名义闭环通过后才推进目标机械b[.001,.005]、主要物理参数±20%、观测/通信延迟各1–3周期及正式MuJoCo多seed。硬件安装符号、反馈D、转速-力矩包络和热模型仍未辨识；静态MJCF对齐与ONNX parity不保证sim2real。详细物理契约和v458 MuJoCo失败见下节。

## 当前唯一入口（2026-10-05，v458已结束）

本节覆盖下文v457及更早“当前/下一步”。当前没有本项目训练进程；继续前仍须检查PID。唯一写入/运行工作区为本目录，旧`/home/kellen/wheeled-legged_RL`仅参考。用户只剩约两天半，优先实机/DR与同契约sim2sim鲁棒性，可减少技能；recovery、jump、全速度暂缓。

v458重跑最终候选`artifacts/continuation_v458/final_checkpoint.pt`，SHA`7565b5d972c3b3d807078d1ae86b0e4ff65f8f1d19ee1b6b164ddb59f05aa642`，源为v457候选SHA`c26de0ed177e05e8e4ed0d0ddcc875b110b34c72d31cdc50217adad9eccd860a`。20次更新、26个目标有符号命令均实际采样，冻结审计通过；但12点DR0快筛仅旧宽门槛下直行1/12，正向1m/s+.8/+1.0全倒，1.5转向及±4自转失败。`promoted=false`、`certified=false`，不得据此开始DR晋级或实机部署。v278旧条件基线保留，v457与v458均非新契约合格策略。

同checkpoint JIT/ONNX parity通过，MuJoCo直行6秒后半速度误差.444451m/s，正向+.8在.552秒终止；sim2sim行为失败。当前`rmuc_peak55.xml`静态2ms，MuJoCo consumer按contract运行时覆盖至1ms；腿PD2ms、policy8ms。USD/MJCF质量/惯量/轴/闭环初态审计通过静态数值检查，轮gear1、±5.5Nm及腿±40Nm匹配；接触/求解器行为及实机符号/反馈D未认证。现已阻止consumer悄悄覆盖contract摩擦和被动阻尼。全部证据入口`artifacts/continuation_v458/final_report.json`、`quick_eval.json`、`cpu/mjcf_usd_audit.json`、`cpu/mujoco_controller_contract.json`，原始训练日志与前次中断证据同目录。

下一动作：针对正向转向做真实rollout的腿位余量、轮差动和接触载荷定位，先证明名义最小技能集有稳定闭环，再按实际±20%主要参数、b[.001,.005]及各1–3周期延迟递增DR；每级同checkpoint做Isaac正式门槛、MuJoCo和retention。硬件安装符号、反馈D绝对值、峰值转速包络与热时程需实机读回，无法凭仿真保证高成功率。不得重复v458配置或把采样修复称为技能修复。

## 最新接续入口（2026-10-04，v457已结束）

本节覆盖下文旧“当前/下一步”状态。唯一运行/写入工作区仍为本目录，GPU0可与其他工作区共享。短期优先≤1.5m/s稳定转向和±4rad/s原地自转，recovery/jump暂缓。最终peak/scale5.5 Nm、continuous3 Nm仅热参考、机械b先验.002及目标[.001,.005]、Coulomb先验.001 Nm；反馈D独立未辨识，本轮D_fb=0仅诊断；轮1ms/腿PD2ms/policy8ms、腿PD60/2/±40 Nm、等效弹簧10.775 Nm/腿、径向力0、任务外环关闭。质量/惯量/摩擦等最终±20%、观测与通信延迟各1–3周期尚未认证。

v457训练和快速评估均已结束，无本轮待续跑训练；下一代理须先查PID，不能重复启动。候选 `artifacts/continuation_v457/final_checkpoint.pt`，SHA `c26de0ed177e05e8e4ed0d0ddcc875b110b34c72d31cdc50217adad9eccd860a`，`promoted=false`、`certified=false`。保留v450迁移源及历史v278基线，不替换默认策略。v457正向+.8/+1.0早倒、1.5转向和±4自转失败；raw 1/12是旧宽门槛/4回合的直行筛查结果，不是交付通过。正式32回合、闭环retention、DR=.05新契约、最高DR、MuJoCo行为、controller/实机部署及合格demo仍未完成。

下一项必要修正是训练采样：v457 gate列有48/49/50/59/60，但priority仍为旧高速列表，1.0m/s八点实际采样全0。必须同时指定priority IDs和等长weights，验证失败命令确实进入rollout；不能再原样增加更新数。spin还需处理yaw error clip2对±3/±4起步命令的混叠，并在Isaac/ONNX/MuJoCo consumer使用相同显式观测契约。证据先读 `artifacts/continuation_v457/{status,final_report,sampling_audit,training_execution,runtime_verified,freeze_audit,retention_audit}.json`，再读本文件末尾与improvement复盘。v457原始stdout缺失，training.log是有明确标记的TensorBoard事件重建；下一实验必须保存原始stdout。

v458已完成启动前配置：只训练v457中已有local authority experts 0–7，priority/gate同时覆盖48–60并写入等长weights；`--require_command_coverage`在前三个rollout检查实际25D delivered command，缺样即失败。观测契约显式迁移yaw error clip 2→6，要求`--allow_observation_contract_migration --reset_optimizer`，并由checkpoint infos、export metadata、MuJoCo/controller共同读回。执行器仍5.5峰值/3持续参考、10.775Nm/腿、b=.002/Coulomb=.001、D_fb=0诊断、1/2/8ms、DR0、外环关闭。启动前不得把v458当已训练结果。

2026-10-05 v458 attempt01完成5次critic warmup，26个命令对的实际rollout coverage均非零，最低262样本；随后无Python traceback中断，无checkpoint、无actor更新，行为无结论。原日志和状态见`artifacts/continuation_v458/attempt_01_status.json`；本轮只允许同配置重跑并使用新日志/PID。用户当前把实机/DR鲁棒性与sim2sim成功置于技能数量之上；可收窄速度和自转范围，但不能降低最终已声明的DR/实机认证门槛或将诊断写成高成功率。

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

## 本文件职责与接续入口

`inclusion.md`用于交接：先读顶部当前契约、保留基线、运行状态与下一步。`improvement.md`用于证据复盘：按实验编号记录失败、原因假设、验证与教训。两者口径必须一致，但历史实验不能当当前配置执行。

唯一写入/运行权威：`/home/kellen/桌面/11 (1)/11/wheeled-legged_RL`。`/home/kellen/wheeled-legged_RL`仅参考，不修改、不运行、不从中取最终产物。

### 当前保留策略与运行状态（v397，覆盖旧阶段快照）

- 原始保留基线：`logs/rsl_rl/rmuc_wheel_leg_course/2026-09-29_22-15-02_physics_continue_v278/model_6684.pt`，SHA256 `52ffa58ca97db3c2ecb2783d60c101611a3531dc94f0b7afa984ec4c6fa812fe`。它的历史训练实际限幅4.875、动作scale5.5，不把历史成绩重解释为当前最终动力认证。
- v393低DR首段25更新：名义7/8、低DR5/8；v395分支回退候选仍名义7/8、低DR5/8，尚未晋升，路径`artifacts/continuation_v395/candidate.pt`。该混合候选无optimizer状态，续训须reset。
- v396长配对诊断已按用户“加快渐进训练”要求取消，不应重启。v391旧3Nm诊断也已取消。
- 当前v397从v395进行25更新，5.5峰值/scale不变，DR strength=.02；D中心.080625，范围[.07981875,.08143125]，力矩增益[.999,1.001]，阶段助力1.0775Nm、径向0，其他DR关闭。独立local分支训练、主图冻结。进程状态以`artifacts/continuation_v397/status.json`和实际PID为准。
- v397阶段末先做source/candidate配对快速八点筛查（每点4env/至少4回合），仅用于继续/停止；推进阶段或晋升前才做完整32回合验收。最终最高联合DR、真实D目标范围及sim2sim行为均未通过。
- MuJoCo峰值执行器接线已检查，八条12秒诊断未倒但跟踪误差超标；属于链路诊断，不能当正式迁移或验收通过。

### 当前快速课程：v398 DR=.05

v397快速筛查8/8且无保护点丢失，已从model6734直接进入v398 DR=.05的25更新；按用户要求减少重复完整对照，阶段末快速筛查后继续逐级DR。完整32回合、多seed、最高DR及sim2sim仍是晋升门槛。

### 更新：+1.0分支停止延长

v399的+1.0 expert单独53更新后，完整32回合仍yaw误差.065/stable.412，bank误差11.5°，不是随机单回合问题。停止继续跑到100，保留model6810及positive10_full32.json；下一步查bank/腿位/控制路由，再决定是否修改训练因素。

### 更新：+1.0失败定位为bank符号契约问题

+1.0完整32回合的实际策略roll约-5.71°、目标+5.82°、bank error11.53°；静态bank oracle已证明公式正确，负roll代表策略外倾。暂停继续训练，先查正向腿差动/关节轴/动作槽和位置余量，不改bank公式。

### 更新：正向内倾需联合腿姿态/轮补偿可达性

旧可达性诊断显示正向差动方向正确但+.30仍roll -1.08°，目标约+4.68°，且active-leg槽饱和；单差动不是解。下一步做正向共同腿姿态+差动+轮补偿消融，保持bank公式和5.5Nm契约不变。

### 下一步执行顺序

1. 完成v397低DR短段与配对快速筛查；有改善且不退化则按课程逐级推进，不再重复无收益的长诊断；5.5峰值保持。
2. damping DR从低强度逐级增加，按25更新分段检查真实行为；需要修单点时优先已有局部分支并冻结其他actor，每个候选仍做完整八点retention。不可用reward上涨或参数冻结代替行为保持。
3. 尽早将同checkpoint、同观测/动作、相同PD/周期/力矩与damping配置接到MuJoCo复测；数值导出一致、不倒与跟踪达标分别验收，不在部署端加隐藏clip或控制补丁。
4. 最终仍需每腿10.775Nm、最高联合DR实际赛场覆盖、多seed、连续命令与速度包络，以及热/电流/转速能力验证。recovery的真实着地初态、坡度±10度与部署ONNX FSM仍保留，但当前按用户优先locomotion；各策略contract隔离。

相关证据索引：`improvement.md` 第396–407节；具体命令、hash、状态、报告保存在 `artifacts/continuation_v381` 至 `continuation_v392`。不得覆盖或删除历史实验来改变结论。


当前MuJoCo执行器入口：`sim2real_RL/sim2sim/rmuc_peak55.xml`（轮gear1、ctrl/force均±5.5Nm；腿±40Nm；2ms物理步）。`run_mujoco_probe.py`默认已切到此模型及5.5峰值，policy/damping/gas-spring-torque必须显式指定。旧`rmuc_usd_limits.xml`保留为历史3Nm诊断，不作当前默认。原生12项正反输出饱和检查通过，见`sim2real_RL/sim2sim/peak_mjcf_audit.json`；几何/惯量/关节/闭环约束未变。PD60/2在runtime执行，不重复叠加。此为执行器契约检查，不代表最高DR或sim2sim行为验收。


轮端阻尼/速度补充：直接力矩控制、Kp0，显式输出为clip(clip(sign*scale*action,±peak)−D*轮速,±peak)。当前课程D=.080625，v393范围[.07659375,.08465625]；目标先验[.005,.04]中心.0225，尚未覆盖/验收。63.3734rad/s是既有actuator参数，当前IdealPD未实现速度—力矩衰减且无显式63.3734的solver硬限速；不能把它当已验证无载速度或速度闭环。MuJoCo受控关节被动D清零、runtime只扣一次同阶段D。详见docs/rmuc_deployment_control_contract.md“轮端damping与速度语义”。


用户确认接续顺序：当前locomotion八点完成最高联合DR及同契约sim2sim验收后，从通过的checkpoint继续全速度、双向转弯与自转技能；复用执行器/DR/ONNX/MuJoCo链路，保留旧技能retention。新增速度域仍需独立最高DR、多seed、连续命令及sim2sim回归，不把低速验收外推为全速度通过。高速速度—力矩包络、响应/侧滑/持续功耗仍属待验收项。



## 2026-10-02 当前 gate：DR=.05 未通过，停止后续推进

`.05` 仍不是全部通过：v398 有效快速筛查为 7/8，唯一失败 `+1.0`；v399 对该专家追加 53 次更新后完整 32 回合仍未通过（bank error 约 11.53°，stable 约 .412）。因此当前不得进入更高 DR、全速度训练或正式 sim2sim。

本轮 obs/policy/control/actuator 静态契约审计通过：25D training AST 与部署 consumer 最大误差0，JIT/ONNX 最大误差约7.15e-7；Isaac `sim_dt=.002、decimation=4` 与 MuJoCo `policy_dt=.008、physics_dt=.002` 对齐；腿 PD60/2/40Nm 算法对照误差0；轮端5.5Nm峰值、3Nm持续参考、气弹簧10.775Nm/腿及 damping `[.002,.04]` 配置审计通过。静态链路通过不等于行为通过。

此前镜像 probe 已更正为直接复用 `train.py` 的 `_mirror_rmuc_observations/_mirror_rmuc_actions`，并修正了左右 wheel velocity/history 未交换的错误。当前结果仅说明正负 yaw 输出仍存在明显差异；这不是单凭 synthetic probe 就能定位为网络 bug 的充分证据。一次错误的 D 退火修复训练已在首轮前终止，未生成候选 checkpoint。

## 以下为历史交接记录（保留原文，非当前运行指令）

下文旧“正在训练”“下一步”“最终3Nm”等字样仅描述记录当时状态；接续以顶部当前口径、当前代码和实际进程为准。

## 物理分支实际步回退已验证（第304条）

新增rmuc_physics_retention.py并通过RMUC_PHYSICS_RETENTION=1显式接入physical-only训练。每rollout PPO前保存物理参数/对应Adam状态及观测动作优势；PPO实际步后按1/.5/.25/.125/.0625步长检查同rollout逐八命令及other聚合surrogate，各组不得降低且+0.8须改善。无可接受步则回退物理参数；缩步/回退恢复该组Adam状态，critic更新保留。使用相同forward batch计算前后基准，非实际行为保证；other聚合不能替代全技能保留。旧图冻结/限幅审核保持。

v304从v278基线3次原生更新完成，三次均拒绝/回退，完整最终actor_state_dict与源逐值一致，冻结检查通过。verification.json保存。说明实际回退链可运行，但未解决优化可行性、没有策略改善，不安排无意义八点重测或把全拒绝长训浪费GPU。下一步需要构造约束候选方向而不仅缩普通PPO步，再配合同rollout回退及独立八点验收。v303恢复训练继续，不改。

## v303温和渐进第一阶段运行（第303条）

v303正式30度温和恢复课程启动：fresh seed268、128env/128rollout/1000更新，recovery_v3连续撑高/侧倒静止零姿态收益、progress index7、4rad/s目标slew、再跌倒penalty5，腿PD60/2/40、轮3Nm、每腿径向10.775N保持。worker1986859/trainer1986933实际存活，日志已9/1000，params逐项回读runtime_verified.json。新过程奖励可触达，不把非零reward称起稳。

完成自动4方向各4replicas首回合评估，默认读取30度训练角度，禁止将其通过冒充完整倒地恢复，不自动提高倾角或DR；随后需阶段retention/更大倾角和最终90度随机跌倒验证。初态四固定几何尚非多seed泛化。导出/额外MuJoCo最终模型复验须在完成后执行，当前已有v301短模型接线证据，不代替v303最终产物。

## 新恢复控制契约跨引擎接线通过（第302条）

恢复导出支持明确的新policy_kind=rmuc_recovery_v2（区别reward版本），记录progress index7/公式版本、4rad/s目标slew、reset实测q、训练初态角度。旧v1不变；非零数值joint tolerance仍拒绝导出避免未同步保护。MuJoCo consumer验证新契约，按相同姿态/高度/ready保持更新progress，逐physics子步目标slew并从实测q初始化，按记录的30度初态运行，非90度认证。

v301/model2实际完整JIT/ONNX导出通过，v302/export保存。Numpy/Torch phase公式2048状态一致、1000随机slew步边界通过；MuJoCo四30度初态各7.992秒完整执行，实际目标增量<=0.008rad/2ms，均未起稳。verification.json及mujoco.json保存。这是接口/公式/运行证据，不证明接触动力一致、实机接入、恢复成功或DR；实际硬件controller仍需独立实现/验收。下一步可启动30度温和正式课程，严禁直接晋升90度。

## 温和渐进恢复PPO接线通过（第301条）

recovery_v3的supported_height过程项改为supported×clamp((up-0.5)/0.5)×clamp(height/0.28)，倾角60度以内有连续撑高信号，侧倒/倒置/无双轮支撑仍0；严格height_reached/ready/hold成功阈值不变。9项reward测试通过。此为过程塑形候选，不代表所有倒地状态均有密集奖励或已消除全部局部最优。

v301使用30度、4rad/s目标slew、progress index7、refall penalty5、v3完成3次原生128env/128rollout PPO更新；model2有限且actor相对model0有变化，实际params回读正确。supported_height日志0.0964/0.0171/0.0093，证明在该课程可触达，但下降不能解释为效果改善；没有技能验收。runtime_verified.json保存。未启动长训练，下一步仍须同步progress/slew到导出与MuJoCo并验证实际序列，再正式渐进训练。

## 温和动作实测与反思落实（第300条）

用户要求落实反思。docs/rmuc_recovery_contract.md新增已修正与未完成清单：侧倒收益、稀疏高度入口、90度直接探索、动作时间尺度错误、progress部署同步、冻结参数不等于行为retention。明确高度项仍有12度/双接触门槛，不能称过程奖励问题全部解决。

v300非零正反目标切换检查完成：4env、128控制步、512physics子步，每个最终PD目标增量<=0.008000002rad（4rad/s×2ms），包含首动作和第64步显式reset，termination全0。脚本probe_rmuc_recovery.py新增exercise-slew模式及实际writer审计。仅目标slew验证，不证明实际速度/冲击或扶正成功。新正式训练尚未启动，仍需课程过程信号和部署consumer一致性。

## 30度起步与温和目标速率（第299条）

已把几何审核的30/45/60/75/90度初态接入恢复cfg，recovery_initial_tilt_degrees显式选择，旧90度高度保持原数值。评估默认读取checkpoint保存角度，可显式diagnostic override90做最终完整倒地复验，报告实际角度，防止把低倾角通过冒充90度恢复。没有自动晋级，后续须独立验收及已过阶段retention。

v299用recovery_v3/progress/refall5、30度、4rad/s目标变化率做原生4env/128零动作smoke完成，初始倾角四例约30度，termination全0、refall cost0。4rad/s是温和起点候选而非实测硬件最优，零动作检查不能证明随机策略的实际目标限幅或学会扶正；需非零动作子步检查及训练smoke。未启动长训练，deployment progress/限速consumer同步仍待完成。证据v299/smoke.json，USD拓扑/腿PD/轮3Nm不改。

## 渐进恢复、温和动作及progress要求（第298条）

用户明确要求侧倒无收益、加入0–1 progress，并再次要求课程渐进且动作不能过猛。新增显式recovery_v3：静态姿态项在up<=0.5（倾角>=60度）为0，真正翻正的变化仍可获得进展信号；保留站立高度/保持和代价。恢复专用可选progress观测占用原恒零目标vx槽index7，仍25D/6D但语义不同：姿态恢复0–0.5、直立撑高0.5–0.8、满足ready连续保持0.8–1；状态倒退progress可退，不按时钟自动前进。行走/旧恢复默认未变，旧恢复checkpoint不得混用新语义，导出暂拒绝progress模式，部署consumer同步待完成。

上轮再跌倒功能已接入默认关闭：ready连续0.1秒armed，大倾角>60度或root<0.18m持续0.08秒触发一次惩罚，之后重站才rearm，reset清零。v297 penalty5原生4env smoke初始倒地refall_cost0。8项单测通过；v298新v3+progress+refall5原生4env/128零动作检查完成，25D有限、初始progress约0.25276、termination全0，初始倒地再跌倒惩罚0。这些仅接线检查，不是起身或温和动作认证。

v294已正式1000更新+评估完成0/16。下一正式恢复实验必须采用渐进初态和训练内目标变化率约束，先低倾角且验证真实PD目标增量，再提高难度；不能把当前未限速的90度smoke当下一训练方案。8rad/s只为诊断候选，既有对照有方向性退化，不称最优值。渐进课程和部署限速/progress同步尚未完成，本轮不启动全难度长训练。

## 恢复探索瓶颈与再跌倒惩罚（第297条）

用户指出站不稳再次摔倒应额外惩罚。末查v294974/1000，ready/established/新高度奖励均显示0，root均值约0.1302m，尚不能把现状归为已站稳再摔倒。反思v2高度奖励同时要求双轮接触和12度直立，信号入口过近成功条件；侧倒姿态常数正回报仍存在，继续仅调成功奖权重未必解决起身探索。下一版应分开起身过程塑形与严格成功验收，验收阈值不放宽。

新增再跌倒惩罚的设计要求：只有先到达有支撑的站立姿态/高度并短时保持后才armed，真正低高度或大倾角持续越界触发一次；reset初始倒地不罚、瞬时接触丢失不等同跌倒，重站后可重新armed。惩罚参数和滞回时长需显式记录；目前尚未实现/启用，不声称已经修复。应与渐进初态课程和动作平滑一同规划但分对照验证，不能再以奖励上涨代替技能成功。当前v294保留完整预算和原口径最终评估。

## 约束方向有限步长验证（第296条）

使用v296同批八命令梯度解CPU float64约束投影：寻找最接近+0.8归一化梯度且所有八组一阶内积>=0的方向。SLSQP收敛，方向norm0.25713，+0.8内积+0.29276、+1.4+0.13382，其余非负（多项在约束边界）。有限步长TF32/128batch原rollout复验：step0精确一致，1e-5/1e-4/1e-3时+0.8 surrogate改善但部分保护命令出现1e-6量级负变化；不能把一阶可行当实际retention保证，也不能把量化/曲率微差直接当显著行为退化。脚本恢复全部参数且未写checkpoint。constrained_direction.json、direction_line_search.json保存。下一步若接入训练必须检查实际参数步与surrogate回退，且仍需完整行为复验，不能直接对投影后的梯度用Adam便宣称保证。

## 梯度冲突处理可行性检查（第295条）

检查当前scripts/rsl_rl与source/agent_rl未发现可直接复用PCGrad入口（历史日志命名不代表当前路径已启用）。使用v296已严格均值复现的八命令参数梯度，在CPU float64做不改actor的投影可行性分析。按样本数加权的原方向对-1.2一阶内积-0.57164；一轮确定顺序冲突投影后-0.01615仍负，且+1.4从+0.67488变-0.04363，说明简单投影不保证全任务不退化。projection_feasibility.json与command_gradients.pt保存。仅首批局部surrogate，不含其他命令/Adam/物理闭环，不作为实际策略退化量或PCGrad最终效果证明。下一实现应检查约束可行性和实际更新方向，保留完整行为retention；当前未改变训练优化器。

## 梯度复现通过与冲突证据（第294条）

v296离线均值偏差定位：train.py启用CUDA TF32，离线默认未启用；仅改TF32仍受batch形状影响，同时匹配TF32和128环境forward batch后rollout均值逐值复现（max error0）。全16384batch仍约0.000670差异，不能混用精度设置/批形状作严格rollout重放。gradient_audit.py使用128分块前向保留计算图，余弦在CPU float64计算，避免TF32点积误差。

首批真实rollout的完整共享physical network参数梯度显示+0.8与-1.2余弦-0.94038，与+1.2为-0.82713、与+1.4为+0.84612。首次有直接局部参数梯度冲突证据，与多命令退化相符；仍单批启动数据、旧策略surrogate非Adam最终步/稳态因果，不可称唯一根因。gradient_audit.json保存，修正第293条悬而未决的复现问题。下一步应进行有retention的冲突处理/隔离对照，禁止再次仅增epochs或放宽gate。v294恢复训练保持。

## gate复核及梯度一致性待解（第293条）

按用户要求细查退化/gate：评估stable本身含窗口yaw误差<0.05，因此yaw和stable失败不是两个独立原因；还要求姿态/侧速/高度/轮接触。v278/280/295的+0.8及-1.2具体checks与signed yaw已保存v296/gate_audit.json。冻结参数不保证共享物理分支作用域内旧行为保持，这解释为什么权重冻结检查通过仍可行为退化，但具体梯度干扰尚未证实。

v296完成单更新rollout采集，新增可选RMUC_CAPTURE_PPO_ROLLOUT=1仅首更新保存obs/actions/advantages/分布参数和actor快照。离线全actor均值复现最大差0.000619322（CPU/GPU及128batch均存在），严格一致性检查未过，故gradient_audit.py拒绝输出参数梯度冲突结论；须先排查采样状态/模型非state_dict配置差异，不能掩盖误差作因果推断。v294恢复训练不改。

## v295未改善八点，保留v278（第292条）

v295四epochs对照100更新及八点评估完成，八点各32回合全部存活但仅6/8。+0.8 yaw由v278的0.052961恶化至0.063946、stable0；-1.2由0.034578恶化至0.052598、stable0.0953，其他六点通过。initial_state_audit逐值一致，behavior_comparison.json保存。不晋升v295，保留v278/model6684的7/8基线；增加epochs与提高±0.8采样均未解决瓶颈，不再重复同方向盲训。共享物理分支存在任务权衡的迹象，但尚未直接测梯度冲突；下一步应测命令间参数梯度/隔离训练，而非宣称已证实残差爆炸。v294恢复训练仍实查运行。

## 装甲板弹击补充（第291条）

用户补充弹击位置在装甲板、方向随机、持续时间很短（乒乓球弹墙类比），已更新competition disturbance contract。随机点限制装甲有效表面、随机方向限定物理可达入射方向，r×F作用力矩与脉冲冲量同时记录；类比不作为已知毫秒时长。具体装甲几何与冲量分布仍需定位/辨识，尚未接入训练，不冒充DR完成。

v295已100更新、旧图冻结及完整导出通过，八点评估1911259仍存活；v294恢复训练1853303运行。等待结果，不因评估耗时重复启动。

## 赛场联合DR新增硬要求（第290条）

用户新增最终赛场DR要求：弹丸约40–400N、云台世界系瞄准保持/主动转动带来的有效惯量和反力矩、气弹簧精度误差。已写docs/rmuc_competition_disturbance_contract.md，纳入联合最高DR验收而非仅nominal。峰值力不等于冲量，持续时间/作用点/方向待明确；云台静态惯量随机化不替代动态反力矩；气弹簧每腿10.775N，±20%仅既有先验。现有root惯量缩放/弹簧幅值DR/速度push不等于已实现全部要求，保持USD拓扑、25D观测、3Nm及统一PD。后续分项建模/回读再联合训练与MuJoCo验证，不中途改变当前两条对照。

## v295四epochs物理适应对照运行（第289条）

v295从v278/model6684保留optimizer/critic、warmup0启动100更新，唯一学习超参改动PPO epochs1→4，采样仍原[1,1,1]、同物理/奖励/限幅/旧图冻结，优势日志只读。目标为检验更充分优化是否改善7/8瓶颈，非已证明方案。首次启动被仓库high-speed update-size guard拒绝（无run生成），保留first_attempt日志；独立对照显式加--allow_aggressive_highspeed，不改变freeze或验收保护。worker1872239/trainer1872276实际运行，保存params确认为4epochs，前2更新冻结检查全通过，runtime_verified.json。完成自动完整导出/八点验收，无自动晋升。v294恢复奖励对照继续，不修改其预算。

## v294恢复reward_v2正式训练运行（第288条）

v294从fresh seed268启动recovery_v2正式1000更新对照，相同128env/128rollout、新技能PPO、四90度初态与物理，仅切换高度/稳定保持reward版本。不限速、joint tolerance0，避免同时改执行链与奖励。worker1853262/trainer1853303实查存活，首更新完成；params读取profile/standing weights及禁用限速、0容差，runtime_verified.json保存。新三项奖励已在原生日志出现，尚为0，不能称学习成功。完成自动16首回合验收，无自动晋升。

评估入口补读checkpoint保存的slew rate并通过cfg初始化验证（兼容yaml .inf）；导出入口暂拒绝有限slew或非零joint tolerance，避免尚未同步的MuJoCo consumer静默使用旧控制/保护契约。该检查不影响旧默认模型导出；语法检查通过。部署一致性工作仍未完成，不将guard当同步实现。

## recovery_v2高度/保持奖励准备（第287条）

用户建议高度达标和稳定时长奖励，已新增显式recovery_v2，默认仍legacy。v2替换v1高度项为双轮支撑且直立阈值内的高度进度（0.28m封顶、权重2）和高度达标（权重2），增加ready连续保持0→0.5s线性封顶奖励（权重3），失稳hold清零，原established保持奖励保留。姿态/进展/动作/effort代价不改，不增加观测维度。当前直立门槛12度会使高度信号仅在接近直立时生效，尚需学习效果验证，不称解决探索问题。评估读取standing weights、导出记录训练权重。

6项reward单测通过（腾空/倒置不获站立高度分、高度封顶、保持增长与失稳屏蔽）。v293原生4env/128步smoke完成，reward_v2有限且termination均0，站立新项在倒地时为0，artifacts/continuation_v293/smoke.json。未启动长训练，未声称起身成功。v292同checkpoint限速对照两组0/16，前/右倒冲击恶化、后/左倒改善，不能作为旧策略直接部署补丁；comparison.json保留。限速部署同步仍待完成，下一正式实验需明确隔离因素。

## v292修正限速原生验证（第286条）

v292同v279/model999、同4replicas每初态及1e-6数值容差进行串行限速/不限速对照。evaluate_rmuc_recovery新增最终set_joint_position_target写入截获，每physics子步验证实际腿目标增量，包含首动作且只统计首回合。限速组已输出16回合，最大增量0.016000032rad/2ms，8rad/s断言全部通过，起稳仍0。不限速基线待完成，不依据旧checkpoint结果宣称减冲击有效。正式训练未启动，export/MuJoCo契约同步仍需完成；当前限速默认关闭。

## 恢复限速实现待重验（第285条）

v291试运行暴露限速实现错误：physics子步使用step_dt导致速率超设定，首动作直接初始化到请求目标而绕过限幅。已改为cfg.sim.dt并从实测关节位置初始化；评估请求目标跳变字段改名max_requested_leg_target_step_rad，不再冒充PD执行目标。v291已标注invalid_slew_diagnostic.json，不可当8rad/s效果证据；它用v279而旧动作审计用v274，不能比较接触峰值后归因限速。修正后尚待原生重验，正式训练未启动，部署契约同步仍待完成。默认限速关闭，旧策略条件保留。

## v279已结束0/16；v290诊断结束（第284条）

v290完整20更新结束，逐更新冻结通过，最终model6704相对v278仅physical network6张量变化。+0.8欠转1346样本差分surrogate均值-6.520、近目标9259样本-0.973、过转6320样本+1.900；组内中心化后分别-1.103/-0.338/+1.490。过转方向与v285局部有效修正一致，不能把混合负信号称策略学反。误差分组不等于时序稳态、surrogate导数不等于共享网络梯度干扰。完整统计v290/summary.json，未晋升诊断checkpoint。

重新实查v279已跑满1000更新且自动16回合评估complete，recovery_v1仍0/16起稳，所有longest_ready0；原trainer已结束，勿当运行中或重复启动。behavior_summary.json按四初态汇总。该结果不支持仅靠本次奖励改动和预算即可解决恢复；下一步应针对已实测目标跳变/饱和设计一致的动作变化率实验及渐进倒地课程，不重复相同配置盲训。尚无可部署恢复策略，完整目标继续。

## v290分组优势诊断运行（第283条）

优势审计增加按obs8*sign(command yaw)分under_turn/near_target/over_turn，阈值±0.05rad/s；只区分误差，不把near_target自动称稳态。v290从同v278源模型启动20更新诊断，原物理/奖励/采样；进程1770102实查运行，前4更新冻结通过，各组计数和等于命令总数。第4轮+0.8过转组151样本，surrogate+7.267/中心化+7.519，整体+0.921，说明前三轮负方向不应外推为恒定错误梯度。需等待完整20更新统计，避免选单轮作因果结论。当前v279恢复训练继续，未晋升短诊断模型。

## v289短rollout优势方向证据（第282条）

v289从v278源checkpoint原采样配置完成3次实际PPO更新，逐更新冻结检查全通过，wheel_advantage_audit真实字段读取正常。+0.8的差分surrogate上升导数三次-2.056/-4.773/-5.857，命令内中心化后-1.637/-6.469/-6.020，方向与v285稳态正偏置有效相反。仅三个启动rollout，可能包含欠转加速与稳态过转混合；不能断言梯度错误或critic唯一根因。需要按yaw误差符号/阶段细分并延长诊断覆盖稳态，未晋升该短训练模型。summary.json保存，v279继续原预算。

## PPO优势审计准备（第281条）

rmuc_position_authority_training.py增加wheel_advantage_audit，在物理分支PPO update前读取原始rollout obs/actions/Gaussian mean,std/advantages，按八命令记录优势均值、轮差分(+right,-left)score×advantage均值及命令内中心化版本。Gaussian params顺序已核实当前rsl_rl源码为(mean,std)。仅局部旧策略surrogate上升导数，不是实际网络梯度、optimizer方向或物理因果；样本相关不输出虚假独立置信度。

CPU合成数据正/负优势方向检查通过；尚未在实际训练短rollout验证，后续需独立短诊断，不自动加载当前恢复进程。未修改actor/critic损失、门槛或执行器，v279继续原预算。

## v288同样本估值比较不支持晋升（第280条）

v288复验完成，完整actor不变，v287/v288的observations、returns及tail bounds逐值一致，可同样本比较critic。+0.8 bias-69.59→-60.19但RMSE145.78→162.85、EV0.846→0.785；-0.8 RMSE101.81→94.80。整体校准没有改善关键正向质量，不采用v288作为后续优选基线，保留v278/model6684。不把平均bias缩小称校准成功，不继续盲目加critic-only轮数。comparison.json保存，下一步检查实际PPO优势/梯度信号或隔离共享修正影响；v279继续原预算。

## v288校准后估值诊断运行（第279条）

v288已完成50次critic-only更新，model6734。freeze_audit.json逐值确认完整actor_state_dict与v278一致，critic8张量变化，动作策略未改变。worker转入同v287随机策略估值诊断，eval1762010已启动；估值改善待结果，不新增技能认证。v279新奖励复位仍运行。

## v288仅critic校准运行（第278条）

v288从v278/model6684启动50更新critic-only校准，warmup50覆盖全预算，actor/噪声全部冻结，保留优化器、原[1,1,1]复习权重和相同物理/奖励。不是续训v280退化策略。worker1751633/trainer1751669实查运行，首批更新actor_trainable_fraction=0。结束worker逐值比较完整actor_state_dict并要求critic发生更新，再用同v287命令/seed/副本数重新诊断；不执行无意义的确定性技能重测或宣称技能提高。完整actor冻结最终检查尚待结束。v279恢复训练继续。

## v287 critic分方向诊断（第277条）

v287随机策略critic诊断完成，±0.8各14096个满足tail bound<=5的相关状态样本。+0.8预测596.18/回报665.77、bias-69.59、RMSE145.78、EV0.846；-0.8预测623.76/回报566.99、bias+56.77、RMSE101.81、EV0.967。整体bias-6.41掩盖方向差异。按obs yaw误差分层，正向<0.05bias-83.49、0.05–0.1bias-34.29；负向对应+33.27/+56.05。保存critic_stratified.json和原始critic_samples.pt。

该证据提示估值校准偏差，不能证明导致actor停滞：样本相关、条件状态分布不同、常量baseline偏差不必改变期望策略梯度，且尚未检查实际PPO优势及梯度。后续宜做冻结actor的critic校准对照或直接记录训练优势，不能凭整体平均误差正常排除问题，也不据此解冻旧技能图。v279训练继续，无技能晋升。

## v287 critic诊断运行（第276条）

v287使用v278/model6684与实际params奖励配置，1m/s±0.8各8replicas/至少16回合/3500步，随机策略及checkpoint critic，gamma0.99与训练一致。不加动作偏置，使用现有returns/tail bounds诊断比较预测价值与采样回报；仅诊断，不能当确定性技能认证。进程1746270实查存活并完成首动作，command/launch/evaluation.log位于artifacts/continuation_v287。v279自复位仍运行。

## v286奖励扫描结论（第275条）

v286真实训练reward开关扫描完成。当前consistent模式主课程/复习、level26/31固定同态reward差均0。零偏置bucket固定末态yaw误差0/0.025/0.05/0.06/0.075/0.1对应raw=clipped reward约-12.0338/-12.2521/-12.4910/-13.0645/-13.8406/-14.9405，high_clip_fraction均0。说明该样本处降低yaw误差有奖励信号，不支持此处奖励被裁剪为平台的解释。仅末态改yaw传感值反事实，不能代替真实训练轨迹优势估计或排除速度/姿态等权衡。summary.json保存；未改变正式策略或晋升。下一步应检查on-policy优势/共享梯度或安排隔离修正试验，避免继续无根据增大reward。v279继续原预算。

## v286训练奖励配置审计启动（第274条）

检查evaluate_rmuc_grid的audit_training_env_config发现只恢复bank scale/start level/reward clip，漏读训练显式启用的low_speed_turn_yaw_tracking_enabled，默认False会造成奖励审计口径错误。现仅在显式audit配置入口增加恢复该开关、reward_contract_by_command和require_analytic_bank，并记录audit_reward_overrides；普通八点行为评估不改变。语法检查通过。

v286沿用v285偏置诊断，加audit_reward_contract和v278真实params/env.yaml，末态扫描yaw误差与raw/clipped reward。实际进程1741129完成初始化/首动作；结果尚未出。不是优势梯度因果证明或策略通过证书，当前v279训练不改。

## v285局部敏感性结果（第273条）

v285五种偏置各8回合全存活。本诊断布局零偏置速度/yaw误差0.010970/0.057223；+0.01轮差分为0.041986/0.027760、stable1；-0.01 yaw0.125883，+0.02 yaw0.110619，-0.02 yaw0.193487。说明局部小幅轮差分有修正余量，大偏置可过调，不能将误差完全归于物理不可控。每case独立bucket、4replicas且与原八点评估布局不同，零偏置基线也不同；不声称多seed或同轨迹因果认证。未将常量补偿加入策略或任务外环。证据v285/sensitivity.json与summary.json。下一步检查策略在该命令处的梯度/优势及奖励折中，利用这一方向作诊断而非直接硬编码。v279自复位仍运行。

## v285局部动作敏感性诊断运行（第272条）

v285使用v278/model6684固定1m/s、+0.8，五个轮差分动作偏置0/±0.01/±0.02，每种4replicas、至少8回合、3500步；2秒后1秒渐入，腿动作不偏置，同25%动力/DR0/无任务外环。偏置只用于局部闭环敏感性诊断，不写入checkpoint，不作策略通过证据。命令与偏置文件在artifacts/continuation_v285，实际进程1735720初始化及首动作完成，结果待出。v279恢复训练1702567仍存活。

## v280不晋升，保留v278基线（第271条）

v280/model6784八点评估完成，6/8通过；+0.8 yaw0.052765、stable0.225962仍失败，-1.2由源v278的yaw0.034578→0.054937、stable1→0退化。提高±0.8复习覆盖改善-0.8但没有解决正向瓶颈，不能将样本不足视为已证实根因。behavior_comparison.json保存同initial_state_audit配对（逐值一致）。拒绝晋升v280，保留v278/model6684的7/8为后续基线，不继续盲目扩大同采样权重；下一步需要检查正向目标处共享残差与冻结源图反馈的局部敏感性，再决定修改。没有提高动力强度或放宽门槛。v279新奖励复位继续原1000预算，末查300附近。

## v280进入八点验收（第270条）

v280完成100更新，model6784，全部冻结检查通过，仅physical network6张量变化，完整导出parity通过。整轮+0.8样本占15.57%（源v278为5.32%），sampling_audit.json保存计数。当前eval1730024实查存活，按既定八点各至少32回合执行，未产生通过结论；不能以采样增加代替效果证据。v279新奖励自复位仍运行，预算不变。

## 自复位接触峰值部位定位（第269条）

v284原生16首回合接触峰值定位完成。back_down四例最大净接触力15.07–17.03kN均在root body robot，倾角约178–179度、root高度约0.071–0.072m，腿速度约13.5–14.5rad/s且腿力矩40Nm；发生于0.376s或5.48/6.82s。该现象对应倒置机身接触，不是正常直立轮地支撑，但尚未用接触点/穿透深度分离真实冲击与求解器效应。其他姿态峰值分布在轮、平行连杆、下腿和导向轮，不能统一把所有非轮接触都当错误支撑。证据artifacts/continuation_v284/contact_evaluation.json。

评估脚本已新增contact_peak_event（body/时刻/力/姿态/高度/腿速/力矩），原生运行通过，未改变训练奖励。下一步目标变化率对照优先，接触惩罚按部位及峰值分布设计，不以单个17kN作为直接阈值。当前v279/v280继续，完整目标未完成。

## 动作强度与数值容差结果（第268条）

v282与v283完成，v274checkpoint加1e-6rad诊断容差仍0/16起稳；v283剩2例真实joint_limit。排除float32误终止不足以解决恢复。动作审计显示各姿态腿目标最大单步变化1.85–3.08rad（含首动作），实际腿速峰值约17rad/s、16例均触及40Nm，最大单body净接触力约17029N。指标仅控制边界采样，不是substep峰值，也未定位冲击部位/时刻；不能直接拿该峰值设阈值。证据v283/motion_evaluation.json、motion_summary.json。

这与先前v272 model100低动作/未饱和结论不矛盾：当前是v274/model999另一策略，不能沿用早期动作温和判断。下一步准备恢复专用腿目标变化率对照，并同步训练/导出/MuJoCo契约，保持PD60/2/40与轮3Nm不变；当前v279完整reward对照和v280采样对照不中途改变。未宣称根因唯一或恢复技能通过。

## 自复位数值超限与动作审计（第267条）

v281复现v274终止：6次joint_limit中4次仅超2.384e-7rad（float32量级），另外2次right_hip_pitch超0.005596/0.006618rad。recovery_env新增默认0的数值容差配置（允许0至1e-5rad），当前训练默认和已加载进程不变；评估入口支持显式诊断1e-6并记录报告。v282诊断启动，不将其当原口径验收。

用户要求控制自复位动作剧烈程度。evaluate_rmuc_recovery.py新增每回合腿目标单步变化（含首动作跳变）、实际腿速度、施加力矩及最大body净接触力审计；仅控制边界采样，不冒充substep冲击峰值。v283用同v274checkpoint/1e-6容差启动动作审计，launch.json记录PID1713953。语法检查通过，数值容差CPU检查确认屏蔽2.38e-7误差但保留真实0.0066超限。未来若采用目标变化率限制必须同步Isaac/导出/MuJoCo/实机，不能部署单独滤波。当前v279奖励/PD/动作接口不变，待诊断结果决定调整，不以加大惩罚代替物理检查。

## v280采样对照启动（第266条）

核实configs/rmuc_wheel_leg.yaml的stage48为1m/s、0.8rad/s幅值（双向采样）。v280从v278/model6684保留optimizer/critic/iteration、warmup0继续100更新，仅优先rehearsal权重[1,1,1]→[8,1,1]，对应stage[48,49,50]；其余奖励、动力、课程主阶段、连续命令、冻结边界及八点验收不变。此为采样对照，不是已证明根因或晋升。

worker1705362/trainer1705416实际运行，前三更新冻结检查全通过；第3次实际+0.8占2068/16384≈12.62%（原整轮约5.32%），-0.8为1800，采样调整已生效，但短窗不代表整轮分布。证据artifacts/continuation_v280/experiment_contract.json与startup_audit.json。v279新奖励复位trainer1702567同时存活，继续1000预算。

## +0.8停滞审计（第265条）

v270/v275/v278三轮日志和评估配对审计：+0.8样本占比约5.57%/5.57%/5.32%，而±1.4合计约50%；+0.8 yaw误差0.053168→0.052974→0.052961。评估力矩饱和0、平均raw/executed动作相同，physical residual采样峰值未到预算。奖励源码在0.053附近仍保留yaw误差梯度。不能直接归因为力矩饱和或奖励无梯度；采样偏重已过±1.4是可检验假设，下一步核实stage映射后做聚焦采样对照并保留rehearsal/八点验收。证据artifacts/continuation_v278/plateau_audit.json。v279实际进程继续，未改运行中的奖励/预算。

## v279独立奖励正式训练启动（第264条）

v274完整1000更新及16固定倒地首回合评估结束，0/16 recovery_hold通过、longest_ready均0。部分回合best_tilt接近直立，但终态不稳定且部分joint_limit终止；不能宣称学会起身。v278完成100更新，冻结/导出通过，八点仍7/8；+0.8 yaw0.052961、stable_fraction0.207149，与v275几乎停滞，暂不盲目再续相同配置或晋升动力，下一步诊断该命令的学习信号/执行动作。其余七点保持通过。

v279从fresh seed268启动独立recovery_v1正式1000更新对照；与v274相同PPO/四90度初态/物理/PD，只切reward profile并记录新增日志。worker1702547/trainer1702567实际运行，首更新完成；训练日志已实际出现RecoveryReward七分项，补齐第263条仅源码检查的运行证据。未加载行走权重，仍3Nm/每腿10.775N/DR0，自动最终16回合验收，不自动晋升。完整目标保持。

## v275七点通过、+0.8未过；v278续训启动（第262条）

v275/model6584八点各32回合全部存活，7/8通过，v270原六通过点保留，+1.2新增通过（速度0.058671、yaw0.048330、stable_fraction0.544314）。唯一+0.8仍失败（速度0.006027、yaw0.052974、stable_fraction0.206940），不放宽门槛；部分负向误差有所增加，不能称全面改善。behavior_comparison.json与v270逐点对比并确认initial_state_audit完全一致。仍25%中间动力/DR0，非最终硬件和最高DR。

v278从6584保留optimizer/critic/iteration、warmup0继续100更新，仅physical head，旧图冻结、相同rehearsal与奖励/动力；不新增专家或提高物理强度。worker1220980/trainer1221080实际存活，首更新完成；完成后完整导出及八点验收，无自动晋升。v274复位继续原1000预算。

## v275训练/冻结/导出通过，行为验收运行中（第261条）

v275已完成100次更新，最终model6584，freeze_audit.json确认仅rmuc_physics_adaptation.network的6张量变化，全部逐更新旧参数/buffer冻结检查通过。export/export_parity.json完整导出通过。实际worker1180368转入evaluating，eval1213250存活；按原八点、每点16replicas/min32回合、3500步执行，同25%中间动力/DR0，无外环。结果尚未产生，不能把导出成功当行为通过，也未晋升物理强度。

v274独立复位仍legacy奖励的新PPO对照，trainer1180423持续运行，末查218/1000附近；不更改本轮奖励、PD、拓扑或预算。新recovery_v1正式训练仍待该对照完整结果。

## 新复位reward实际PPO/导出接线通过（第260条）

v277 recovery_v1独立3更新完成，run2026-09-29_21-49-56_recovery_reward_smoke_v277/model2；实际params profile/weights及std0.15/lr3e-4读回，参数更新/有限checkpoint和完整导出通过，runtime_verified.json。export契约新增training_reward_profile/weights，不把legacy与新reward候选混称。未证明自主起身，没有启动新的长reward训练。

v274当前仍legacy_v0独立探索对照，末日志75/1000；v275同25%物理续训35/100，所有冻结审计通过，两任务实际存活。原预算/奖励/PD/拓扑不改；后续在完整结果基础上显式开启新reward正式实验。

## 独立自复位reward_v1已原生验证（第259条）

用户明确要求自复位等课程使用独立reward。新增recovery_rewards.py，可配置翻正/高度条件/翻正进展/起稳和动作/effort/时间代价；水平倒置无height奖励，reset进展置0。recovery_env支持legacy_v0/recovery_v1，默认legacy保持当前v274对照，后续训练需显式env.recovery_reward_profile=recovery_v1。4项pytest通过，v276 reward_smoke四90°初态128步有限、无立即终止；不是起身技能证据。

evaluate_rmuc_recovery读取checkpoint params恢复reward profile/weights，旧checkpoint缺字段仍legacy，防止自动评估悄悄改条件。训练PD/拓扑/物理未改，v274/v275真实进程仍存活。小倾角候选20组已CPU生成但未接课程，recovery_curriculum_candidates.json标注applied_to_training=false，当前90°实验保持。

## v270通过6/8；v269全预算复位仍失败；v274/v275已启动（第258条）

实际复核两旧任务均训练+评估complete。v2706484在25%动力八点各32回合全部存活、6点通过，仅+0.8 yaw0.053168与+1.2 yaw0.062187未过；behavior_comparison.json配对源v266（1/8）。v269跑满1000，最终999的16固定倒地回合全部ready0；最终recovery_export已导出验证，但明确为失败候选，不能部署。

v274独立复位从fresh seed268运行1000更新，已核实新std0.15/lr3e-4/epochs4，环境除空间对象序列化字段外无顶层变化；worker1180367/trainer1180423，首动作/首更新已见。只比较整组PPO设置，不声称单std因果。v275从v2706484保留physics optimizer/critic继续100，warmup0，同25%物理/奖励，只新物理分支训练；worker1180368/trainer1180424，初始化和首动作正常。两个任务均先查实际status/PID，勿重复启动。没有提高迁移强度/放宽gate/修改腿PD或拓扑。

## 复位MuJoCo首回合采样时序对齐（第257条）

独立export契约新增first_episode_control_steps=ceil(T/dt)-1，当前999步7.992秒与Isaac timeout一致。runner不在t0累计ready，整数步计数避免累加误差；末积分后刷新并记录terminal，不再推理/记入未执行动作。trace明确previous_executed_action与next_action，terminal的next_action=null。

model100四方向重新导出/各999步MuJoCo实跑，仍ready0；v272/model100_terminal_aligned.json和terminal_timing_audit.json验证全部1000采样、999动作间隔、最终历史来自上一实际动作。仅时序/记账核验，不称接触动力或自主起立通过。v269/v270实际仍运行，未改训练参数。

## 独立探索配置原生验证通过；纠正曲线因果解释（第256条）

v273只做3更新新配置smoke，已complete：init std0.15、lr3e-4、epochs4/entropy0.005实际params读回，末std约0.1474–0.1547，runtime_config_verified.json及update_audit确证，不是起身证书。v269仍原配置跑1000预算，v270继续，未中断或中途改写参数。

reward_counterfactual.json直接执行真实_get_rewards源码：零effort/动作差时倒置height0.07每步0.35，999步约349.65；侧倒2.425、起直未稳定5、满足保持10。加上启动randomize initial episode length，reward/episode length上升不能证明“减少早期危险终止”。第254条相关推断应收回，仅保留model100四姿态未起身的实际证据。后续对照先区分探索和奖励因素，不随意同时改动后声称因果。

## 复位动作未卡限幅；后续实验使用独立PPO配置（第255条）

v272导出model100并实际四姿态MuJoCo各8秒诊断，起稳仍0；动作clip fraction全0，腿峰值均<40Nm，不支持“动作被限幅卡死”解释（仅MuJoCo证据）。action_exploration_audit.json记录。v269实际继承旧行走精调std≈0.004、lr2e-5、epochs1，探索不足是待验证假设，不把它称已证明因果。

新增RmucRecoveryPPORunnerCfg并仅将Recovery任务注册指向该配置，下一次新进程默认std0.15/lr3e-4/epochs4/entropy0.005、仍128rollout/1000预算。同PD/奖励/物理/拓扑不改；v269已加载配置仍原样继续1000预算，不能因短期平台停止（第254条“先决定是否继续”不得覆盖用户预算规则）。当前新配置只做语法检查，未运行新对照训练。v270继续原预算。

## 复位model100曲线提升但四姿态仍0起身（第254条）

v269 model100完整回合评估四方向均timeout、longest_ready0、最终root_z约0.07m、tilt约180度、recovery_hold全false；recovery_behavior_summary.json。TensorBoard快照training_curve_snapshot.json显示episode length56→999、reward72→442，不能把曲线当起身成功，下一步检查动作饱和/腿目标范围/reward塑形后再决定继续预算。

## 复位MuJoCo接触/终止契约补齐（第253条）

export_rmuc_recovery.py将真实params起稳阈值/episode时长及14个USD关节上下限写入contract；run_recovery_onnx.py按契约检查低/高root、越界、线速度、关节范围，轮接触由标量sum改世界净法向力模长。v268模型重新导出、四姿态8秒实际重跑，仍ready0，无伪通过。证据mujoco_recovery_contract_guards.json。未声称接触求解器/PhysX速度约束或末端时序完全等价。

v270末查14/100、冻结审计全通过；v269复位约87/1000后继续，两个trainer仍在跑，先查实际状态，不重复启动/不改训练中环境。

## 复位ONNX已接入MuJoCo诊断链（第252条）

新增sim2sim/run_recovery_onnx.py，独立消费recovery contract，检查policy_kind、ONNX hash、关节顺序、PD60/2/2ms、25D和腿目标范围，拒绝行走模型。export_rmuc_recovery.py新增onnx_sha256。v268模型重新导出后四倒地姿态各8秒实跑，longest_ready全0，与Isaac均未起身；只是链路证据，接触/终止与速度约束尚未完全等价，不能称恢复验收通过。

v269复位仍训练，末日志79/1000；v270行走同阶段续训实际进行，两进程已核实，不重复启动。证据v268/mujoco_recovery_diagnostic.json。拓扑与训练PD不改。

## v266验收完成仍1/8；v270同阶段续训（第251条）

v266八点各32回合生存1，仅-1.2通过；相对v263多数速度与正yaw误差改善，仍不能晋级25%以上。behavior_comparison.json保留配对；model6384完整导出在artifacts/continuation_v266/export。v270保留该checkpoint/optimizer/critic、warmup0继续100更新，物理/奖励/分支不变，只更新physical head；worker904518，查status实际PID勿重复启动。复位v269继续1000预算。

训练日志：logs/rsl_rl/rmuc_wheel_leg_course/*physics_*为行走动力适应，logs/rsl_rl/rmuc_recovery/*recovery_initial_v269为正式复位；artifacts/各版本training.log/status/target_turns为控制和验收记录。v268/recovery_export仅3更新未学会模型验证用，正式v269尚无已认证导出。用户已询问路径，避免将.pt训练checkpoint与ONNX导出混称。

## v266已100更新转验收；复位独立导出入口完成（第250条）

v266训练结束，model6384，仅physical network6张量变化且完整导出通过；eval892858实际进行，已steps2000/min_completed16，最终八点结论未出。v269复位1000更新仍训练，末次日志34次附近，勿重复启动。

新增scripts/export_rmuc_recovery.py：只接受params标识recovery_prototype且无rmuc残差checkpoint，拒绝把行走模型重命名为复位。读取实际params和USD关节范围，完整JIT/ONNX对照后写独立policy_kind/动作范围/PD/轮限制/hash；locomotion_consumer_compatible=false。v268未学会模型导出只用于验证，recovery_export/export_parity及四真实倒地初始观测ORT/JIT对照通过，无行为认证。v269正在运行的worker原本仅训练+评估，结束后需另补该独立导出，不能以修改脚本当已执行。

## 倒地自起立正式训练v269已启动（第249条）

新增scripts/evaluate_rmuc_recovery.py，在_get_dones自动reset之前记录首完整回合、连续ready计数、终止原因，区分曾起稳与结束仍起稳，固定初态结果不当DR/交还证书。v268三更新模型四种初态各完整8秒评估均失败，最长ready0；最终机体约180°，没有误把超时/reset当起身。

v269正式独立复位从新网络启动，128env/128rollout/1000更新预算，DR0、四固定90°姿态、3Nm/damping0.0225/每腿10.775N/腿PD60/2不变。worker891687/trainer891715，run rmuc_recovery/2026-09-29_17-11-32_recovery_initial_v269；初始化和首动作已成功，先查实际status/PID，勿重复启动。结束自动每姿态4replicas的首完整回合验收，不自动晋升；初态扰动、最高DR/多回合统计、完整导出/交还仍待完成。v266行走动力适应独立继续，末查仍training。

## 独立复位PPO三更新接线通过（第248条）

v268新网络、不加载行走checkpoint，128env/64步rollout/3更新已完成，run rmuc_recovery/2026-09-29_17-05-40_recovery_smoke_v268/model_2.pt。worker已结束；update_audit.json确认参数更新、有限checkpoint，不是起身验收。runtime_contract_readback确认PD60/2、轮3Nm、damping0.0225/每腿10.775N、tracking_error_with_target、无外环。

docs/rmuc_recovery_contract.md新增输入/腿范围映射/执行器/交还条件。复位command=0不代表观测速度误差槽为0；复位与行走腿动作尺度不同，不能直接混历史action或consumer。尚需独立deterministic起身与交还评估、完整导出，未开启长复位训练。v266仍运行，末查75更新及之后继续，先读实际jsonl与PID。

## 独立复位环境原生smoke通过（第247条）

新增recovery_env.py及注册Robotics-RMUC-WheelLeg-Recovery-v0，独立复位奖励/reset/终止，未改行走类。25D/6D、原USD拓扑/腿PD60/2保持，轮端3Nm、damping0.0225、每腿10.775N。腿目标为关节可用范围映射，与行走0.22尺度不同，必须独立动作契约，不能混用ONNX。

scripts/probe_rmuc_recovery.py在4env/128步实际执行，四初始倾角约90°且不立即done，观测/奖励有限、终止数全0，实际PD60/2和轮3Nm读回核实。artifacts/continuation_v267/recovery_native_smoke.json为证据；首次sim.physics=None故障已修复并保留失败日志。仅环境接线成功，未训练recovery、未证明起身或交还行走。原v266训练仍独立继续，下一步奖励/动作与接触验收后才开始复位探索。

## v266到28更新冻结正常；复位初态CPU检查完成（第246条）

v266真实trainer867839仍在运行，最新28/100更新（含10critic预热），全部frozen_changed/buffers_changed为空，新物理分支已学习；最终验收未到，不称已恢复技能。

新增sim2sim/audit_recovery_starts.py与settle_recovery_starts.py及两JSON报告。在不改拓扑条件下生成前倒/后倒/两侧轮组件朝下四类姿态，初始地面间隙3mm、四闭链误差<1e-6；重力下各做0/每腿10.775N共8个2秒无电机驱动试验，全部finite，峰值闭链误差<=1.502mm，主动腿峰值<=12.747rad/s。部分状态仍有轮/导向轮转动，不当静止初态；有弹簧两侧案例达到本次静止诊断阈值。未运行复位策略或证明可达，独立recovery ONNX待训练。下一步接触分布/初态课程与recovery独立终止/动作契约。

## 物理适应原生smoke通过；v266正式100更新已启动（第245条）

新增--train_physics_adaptation，prepare_authority_optimizer物理分支独立组，冻结整个旧actor；审计新增physical预算/有效样本与域外Adam动量保护。27项相关测试通过。v265实际3更新（1critic预热），仅新network6张量改变、旧参数/buffer完全保持，model6287完整导出通过；仅接线证据。

v266从v264零分支（sha43287819…）重新正式100更新/10critic预热，25%目标动力（轮4.875Nm、damping0.080625、每腿2.69375N、scale5.5、DR0），旧图与PD不改。worker867784/trainer867839真实运行，已看到首更新；artifacts/continuation_v266/status/command/experiment_contract可核实。完成后自动freeze审计、完整导出、同v263八点32回合验收，不自动晋升。不得重复启动或把smoke/首更新当新技能掌握。

## v263已结束；v264物理分支完整图恢复与导出通过（第244条）

25%迁移八点各32回合全生存，仅-1.2通过；其余七点有真实跟踪缺口。实际effort4.875/damping0.080625/每腿2.69375N，readback_summary.json确认。无本任务评估在跑，勿重复v263。

物理分支现已在rmuc_command_adapter完整forward中位于authority后、Gaussian前，install_position_authority按checkpoint键显式恢复；旧checkpoint不自动添加。v264/zero_physics.pt来源v261，旧state和512观测零组合/严格restore逐值一致；nonzero_diagnostic.pt仅人工非零接线测试，不作训练起点。两者完整JIT/ONNX含新gate边界2409探针均通过，最大ONNX误差2.38e-7，export contract记录physical预算/支持域。7项相关测试通过。尚未接入独立PPO optimizer/native smoke；下一步完成训练隔离和consumer预算审计后，用零分支做25%迁移动力适应，不能直接加载旧local optimizer。

## v263迁移25%评估运行；有界物理适应分支CPU预检（第243条）

worker857699/eval857700，artifacts/continuation_v263/status.json；同v261八点/128env/至少32回合，硬件fraction0.25，名义effort4.875Nm、damping0.080625、每腿弹簧2.69375N、动作尺度5.5、DR0。已实际初始化并steps1000；结束读target_turns/summary，勿重复启动。只是迁移中间态，不是最终3Nm认证。

新增scripts/rsl_rl/rmuc_physics_adaptation.py原型，一个共享零初始化MLP而非逐命令堆叠；低速支持域|vx|<=1.5向1.7平滑退出，|yaw|<=4向4.2退出；有界动作修正预算[.15,0,.35,.15,0,.35]，最终consumer/物理限幅仍需保持。3项CPU测试通过；真实v261完整actor零组合逐值相同，合成一步更新只改变新head、旧全部state不变，physics_head_cpu_preflight.json记录。尚未接入actor安装/恢复/导出/PPO，不是已训物理策略。下一步依据v263缺口完成完整图集成与native smoke后训练。

## v262已完成：目标动力八点0通过（第242条）

worker/eval已结束，status complete/passed_rows0，目标八点不是后台运行。七点各32回合生存1但速度/yaw失败，速度误差0.659–0.887、yaw0.178–0.381；-1.4共44回合全部tilt，coverage0，误差null不得当0。summary.json/target_turns.json/source_target_comparison.json保存。

与v261源nominal8/8通过相比checkpoint与记录初态完全相同；实际轮上限3Nm、damping0.0225、每腿10.775N，motor response虽安装但DR0使腿轮tau全部0，故不是滞后鲁棒性验证。MuJoCo同中心0.0225/3Nm/10.775N也在约5秒触发停止。目标物理适应缺口已在两引擎出现，不能仅归咎于obs或MJCF；多项变化同时发生，不单独归因某参数。下一步渐进物理适应与保留验证，旧nominal证据仍有效，未训练新策略。

## v262目标动力Isaac八点评估已实际运行（第241条）

artifacts/continuation_v262/worker.py已启动，worker852280、eval852281；先读status.json并查实际PID，勿重复启动。来源v2616284 sha240ddf1a…，同原八命令/128env/每点至少32回合/3500步上限，DR0；仅评估、不训练/自动晋升。profile显式保留5.5动作尺度，轮端3Nm、径向每腿10.775N、damping中心0.0225（候选区间0.005–0.04）、启用既有电机响应先验0.002–0.030s；所有未实测，非最高DR。

启动/首动作成功，已观察1000步；-1.4在episode898步出现tilt终止，外环反馈均0，完整八点结果未出。experiment_contract.json/profile/command保存证据，结束看target_turns.json和summary.json。另一fudan训练832110未动；4080有余量后启动本小规模评估。下一步根据真实结果决定物理适应训练，不重复已通过原nominal。

## 腿PD实机一致性成为明确硬约束（第240条）

用户要求训练/部署腿PD完全相同。已新增sim2real_RL/sim2sim/leg_pd.py、verify_leg_pd.py、LEG_PD_CONTRACT.md，MuJoCo调用共享PD参考。对本机真实Isaac IdealPD.compute及_clip_effort源码AST，1024四关节输入（含正负饱和）误差0；leg_pd_verification.json存源码hash，hardware_controller_verified=false。60/2输出侧PD，目标速度/前馈当前为0，限幅40Nm，默认PD2ms/策略8ms；不叠加隐藏环或重复乘9:1，实机协议/周期/滤波仍待核实。目标动力Isaac八点评估尚未启动，不能把计划当进程；GPU发现另一个fudan训练，不干预它。

## damping敏感性确认；Isaac迁移接口允许保留动作单位（第239条）

默认USD限制模型/3Nm/每腿10.775N、同v261测试原用户暂定0.005–0.08区间两端及中点：0.005在4.344秒停止，0.0425在6秒停止，0.08跑满但后半程平均速度0.020614，均不达标。probes/damping_sensitivity.json记录，measured_calibration=false。这是宽区间敏感性探针，没有覆盖/修改现有hardware_target中依据4m/s提出的0.005–0.04候选范围，不能称最高DR完成。

scripts/rmuc_hardware_profile.py解除“action scale必须等于3”的不必要耦合：正有限动作尺度独立插值，实际effort目标仍严格3Nm且禁止重复gear。旧scale3配置保持兼容；支持显式scale5.5保留旧actor单位。18项相关测试通过，含部分迁移物理上限不随尺度放宽和非法值无部分写入。未改默认硬件目标文件、旧策略或启动训练。

## 默认切到USD轮端3Nm口径（第238条）

用户确认gear细节交由工程处理，只要求USD执行器对齐。默认probe/audit模型已切rmuc_usd_limits.xml，轮上限默认3Nm、腿40Nm、gear1，动作尺度保留5.5作为旧ONNX单位；原参考模型不改。新增加载MJCF能力与声明限幅校验，默认3Nm模型请求5.5会拒绝；默认0.08秒smoke和模型审计通过。gear_unit_verification.json实测源gear15.7647换算ctrl与新gear1在轮端±3Nm范围输出力矩等价。仍不代表速度限制/接触/整段行为及DR都已对齐。

## 每腿10.775N径向气弹簧接入MuJoCo（第237条）

新增radial_spring.py、verify_radial_spring.py；runner参数--radial-force按每腿N解释，hip锚点与wheel轴心成对外力，mj_applyFT正确位置施力，不改拓扑。512组对训练force helper误差3.55e-15，广义力虚功与腿长有限差分检查通过，radial_spring_verification.json。

3Nm输出上限/旧5.5尺度/摩擦0.5/2ms配置下10.775N每腿跑满10秒，后半程速度/yaw误差0.258780/0.111332；无弹簧0.211308/0.151626，仍未通过。报告probes/v261_radial10775.json；这是目标物理适应缺口，不是新认证。实际轮峰值1.35752/1.19864Nm。仍缺实测damping/完整DR/复位与最终跨引擎验收，不盲目复训旧nominal。

## 保持腿轮拓扑；生成显式Nm执行器配置（第236条）

用户要求腿轮拓扑不要改。原wheeled_infantry.xml仍与来源逐字节相同，rmuc_training_v261.xml/rmuc_usd_limits.xml也保持body层级/位姿、joint连接/轴/范围、四闭链约束完全一致，topology_verification.json记录，build_mjcf_profiles.py加不变检查。

两生成配置gear1，ctrl为输出轴Nm；训练对照轮5.5Nm、USD限制版轮3Nm/腿40Nm，依据实际USD与v261参数生成。几何惯量逐数组一致、正负饱和力矩探针通过，mjcf_profiles.json记录来源。USD限制版同v261摩擦0.5跑满10秒，实际两轮峰值1.34758/1.19888Nm，速度/yaw误差0.211308/0.151626未过。probes/v261_usd_limits_local.json。原策略未改，继续接触动力差异/DR与复位主线。

## MJCF及网格已本地化（第235条）

用户要求MJCF放sim2sim并与USD对齐。现模型sim2real_RL/sim2sim/wheeled_infantry.xml及meshes_mj/21网格均为本地真实文件，22文件hash与原来源一致，model_source_manifest.json记录。audit_model/run_mujoco_probe/run_dynamics_ablation默认切到本地模型，重新审计与0.08秒加载闭环smoke完成。未静默改gear/摩擦等物理参数，运行时接触/执行器最终对齐仍待完成；源模型对照与训练runtime覆盖必须分别核查。

## 3Nm限幅与动作尺度分离诊断完成（第234条）

probe新增--wheel-action-scale/--wheel-effort-limit和实际qfrc_actuator上限检查。摩擦0.5下同v261三组：5.5尺度/5.5限幅与5.5尺度/3限幅均10秒，完整trace逐值相同，实际最大轮力矩1.34233/1.19844Nm，说明本点没有用到>3Nm；3尺度/3限幅在1.984秒tilt45.85°终止。证据probes/torque_migration_comparison.json与三trace。不能把动作尺度视为物理限幅，也不能直接改ONNX后乘数到3；优先验证保留旧尺度、严格轮端3Nm裁剪路线，再处理最终damping/每腿10.775N/DR。原Isaac和策略未改，该点原有yaw偏差仍在，无新增行为认证。

## 实际reset obs对照通过；硬件与力矩口径再次确认（第233条）

用户强调obs/策略对齐，硬件两RM3508、四DM8009（此前完整型号DM-J8009P-2EC9:1）。读取USD轮maxForce3Nm、腿40Nm，当前Isaac运行轮覆盖5.5Nm，probe也是旧5.5Nm诊断；MJCF ctrl±3乘gear15.7647不是输出轴±3Nm，最终每轮持续3Nm未完成迁移。

新增compare_initial_observation.py及probes/initial_observation_comparison.json，实际v261 Isaac初态112–127与MuJoCo摩擦0.5 probe首帧25D最大差1.1921e-7（重力），同ONNX动作差3.5763e-7。初态接口证据通过，不外推完整状态/轨迹/DR。继续物理差异定位，无重训。

## 根坐标系惯量/关节轴核查与摩擦对照（第232条）

audit_model.py扩展初态body位置/COM/完整惯量张量/关节轴和锚点对照，39体位置及COM最大分量误差5.87e-7m、惯量3.69e-7kg m²、关节轴2.22e-16，未发现明显错配。运行时接触和惯量读回尚未完全核实。

v261 final_turns.json实际机器人材料读回0.5，训练地面配置0.5；MuJoCo XML轮地1.1。新增--sliding-friction 0.5并读取实际contact friction确认为0.5。10秒同checkpoint/初态单因素对照跑满，后5秒速度/yaw误差0.212264/0.152709，比原1.1更差，仍未通过。证据probes/v261_friction05.json。不把更好成绩当物理正确；后续匹配初态/接触求解和动力配置排查，不重训。

## 步长/被动阻尼成对对照完成，未解释主要缺口（第231条）

run_mujoco_probe.py新增--physics-dt/--passive-damping，只改模型内存并记原值，不改资产。四组各10秒同ONNX/初态：基线、2ms步长、被动阻尼0.01、两者组合，全跑满，yaw误差均约0.103；组合速度误差0.156872。汇总probes/dynamics_ablation.json，新增run_dynamics_ablation.py供复现。尚未跟踪通过，不重训。

检查本机IsaacLab base_articulation_data.py确认root_lin_vel_w指root COM；MuJoCo objectVelocity(BODY)与jacBodyCom*qvel数值一致，当前取点正确。下一步接触几何/材料、关节坐标和匹配初态对照，不能靠调摩擦到单点通过就冒充物理一致。goal保持。

## 首个v261 MuJoCo十秒闭环已运行，跟踪未通过（第230条）

新run_mujoco_probe.py使用同一v261完整ONNX，右先映射、canonical=(-source_z,source_x,source_y)、8ms推理、1ms候选XML物理步、输出轴Nm除gear。内存清除主动关节原阻尼，腿PD60/2/40Nm/scale0.22，轮旧5.5Nm/-action符号/0.1显式阻尼；原XML与USD未改。只做诊断，尚无ROS与最终物理对齐。

probes/v261_turn_10s.json：1m/s,-1.4rad/s跑满10秒，最大倾角5.048°、腿速度2.557rad/s；后5秒625样本平均速度0.84238、yaw-1.50297，绝对误差0.15762/0.10297，yaw未通过。先mj_forward刷新派生量避免观测时间错位。不要把不倒当跟踪达标，也不要因跨引擎缺口马上重训；下一步做被动阻尼/接触/初态/坐标与驱动的成对对照。goal继续，无新训练。

## 模型初审与轮输出力矩单位核验（第229条）

sim2real_RL/sim2sim/audit_model.py已实际编译候选/home/kellen/wheel_leg_sim2sim/assets/wheeled_infantry.xml并读取当前训练USD，39刚体名称一致，总质量19.22113998/19.22113937kg，四闭链初始误差<1.2e-16m。旧asset文档18.970kg是历史值，勿当当前质量。模型仍待COM/惯量方向/关节轴/碰撞/运行时驱动审计，未执行ONNX物理闭环。

关键：候选轮gear15.7647058824，ctrl3对应约47.294Nm，必须把输出轴Nm除gear后写ctrl；六执行器逐项mj_forward/qfrc_actuator验证通过，轮3Nm对应ctrl约0.1902985。原MJCF未改。报告model_audit.json。

部署consumer已改为与训练env._compute_policy_action相同的直接条件，verify_interface.py直接提取实际执行分支AST并加float32相邻边界，1024组观测/consumer误差0、ONNX/JIT误差7.15e-7。仍只有接口和模型初审证据，非DR或sim2sim行为证书。goal继续，无新训练。

## v261八点nominal通过；goal恢复，推进RMUC部署接口（第228条）

v261最终6284完成100更新，八点各32回合全部通过；-1.4速度/yaw误差0.081163/0.008047，七保护行与v260逐值一致，完整导出通过。证据artifacts/continuation_v261/final_evidence_audit.json。只认定nominal固定点，不是最高DR或最终3Nm认证。

用户确认近期按行走＋独立复位两个ONNX推进，后续增加越障；全速度/高yaw/自转/最高DR长期范围保留，十月八日前优先MuJoCo、复位与限定行走范围。部署位置sim2real_RL/sim2sim，上游wheelbipe_ros2_sim2sim固定dd367bf78c7e393d811595edeb4affe253156c95，35D normal-only框架需要适配为RMUC25D/右先关节顺序/8ms策略周期/轮力矩。不可混用旧Fudan25D历史策略接口。

已新增rmuc_policy_interface.py与verify_interface.py；1024状态对真实env.py AST观测前缀、训练authority consumer及完整ONNX/JIT，前两者误差0，ONNX/JIT最大7.15e-7。报告interface_verification.json。issaclab环境缺onnxruntime，改用已安装依赖的robot环境完成CPU验证，没有安装依赖或启动训练。找到/home/kellen/wheel_leg_sim2sim/assets/wheeled_infantry.xml作为待审计闭链模型候选，不能直接当作与USD一致。下一步核查模型/执行器/坐标映射后做MuJoCo基础闭环；本轮尚无MuJoCo行为证书。

## v260正1.4通过；v261负向专家独立训练运行（第227条）

v2606184 sha02aae0ae…八点256回合全存活，7点通过；+1.4速度/yaw0.015351/0.042934，-1.4仍yaw0.095168失败。七非训练行与v259逐值一致。v261只训专家4，正向及其他图冻结，reset optimizer（参数集合变更）、保留critic/warmup0，100预算不改门槛。worker710166/trainer710167，run2026-09-29_13-03-23_negative_yaw_v261；启动实查9更新，冻结通过。先查v261/status/live_check及进程，不重复启动；结束自动导出八点32回合与保护行对照。全任务未完成。

## v260正向+1.4续训运行（第226条）

v259验收已结束：+1.4速度/yaw0.020596/0.064681仍未过，七保护行与v258逐值相同，原六点通过、负1.4仍yaw0.095168。v260从6084继续100，保留optimizer/critic、warmup0、只训indices[3]，门槛不改。worker164930/trainer164931，启动实查11更新、冻结通过。先查continuation_v260/status/live_check及进程；结束自动导出/八点32回合，读evaluation_status与final_evidence_audit。全任务未完成。

## v259训练完成，正确八点评估已启动（第225条）

v258结果：六旧点通过，+1.4速度/yaw0.03813/0.13437未过，-1.4仍0.09517；七保护行与v257相同，审计已补存。v259继续100更新仅专家3变化，6084 sha ed1769fa…导出通过，未修改奖励。此前错用train.py做评估已失败退出，证据归档failed_evaluation_entrypoint。当前正确run_evaluation.py worker158750运行，读continuation_v259/evaluation_status.json和实际进程，勿重复启动；训练status complete不代表验收完成。结束看final_evidence_audit.json及七保护行比较。全任务未完成。

## v258正向+1.4专家独立诊断训练中（第224条）

更正：专家3/4本来参数和gates独立，v257不证明方向互相干扰。v258从v257 model5884只训专家3，首个参数解析失败在训练前已隔离，修正后真实worker3898086/trainer3898087，run2026-09-28_13-12-12_positive_yaw_v258，已8更新、冻结通过。先查continuation_v258/status与进程，结束后分别验收+1.4/保护六点/-1.4；当前没有新技能结论。

## v257结束：±1.4仍失败，出现速度/yaw权衡（第223条）

v257额外100更新结束、导出和八点各32回合审计完成；六保护点保持通过，±1.4全生存但失败。+1.4速度0.09981变好、yaw0.33609变坏；-1.4 yaw0.09517改善仍超限。继续共享两个专家会互相干扰，下一步拆分正负目标或独立命令损失，不能自动晋升。全DR/动态/部署目标仍未完成。

## v257仅±1.4专家续训中（第222条）

v256匹配对照已结束：六保护行与v255逐值一致；正1.4速度0.270→0.198但yaw0.024→0.059，负1.4 yaw0.169→0.147，仍未达标。v257从v2555784继续100更新，保留optimizer/critic，warmup0、仅indices3/4，不改奖励门槛/预算。worker3877041/trainer3877042，启动实查6更新，冻结通过；先查continuation_v257/status.json/live_check及进程，勿重复启动。结束自动导出/八点32回合，读evaluation_status.json。当前无新增技能认证。

## v255已验收，v256源策略同场景对照进行中（第221条）

v255八点256回合全存活，原六点通过，+1.4速度/yaw0.19808/0.05889，-1.4 yaw0.14673，两个新增点均未过。无新训练。v256 worker3873139冻结v251源5684，匹配八点布局/seed67/32回合，核实保护行为及新训练实际收益；先查continuation_v256/status.json和进程，勿重复启动。v255 evaluation_status已complete，不是全任务完成。

## v255训练/导出完成，八点评估运行中（第220条）

100更新结束，final model_5784 sha5f3bf716…，仅专家3/4变化，完整导出通过。已显式启动finish_evaluation.py，worker3869764/eval child3869908；先查evaluation_status.json与进程，不重复启动。status.json complete只表示训练结束。八点各32回合结果尚待生成，未认证±1.4/原六点新checkpoint行为，更未完成全任务。

## v255实际训练中，结束后需显式启动验收（第219条）

真实run2026-09-28_12-06-38_higher_yaw_v255，worker3853959/trainer3853960，100预算仅训indices3/4。先查status/live_check及实际进程，勿按旧日期猜路径或重启。v254短程audit已补存，仅新40张量变化。

注意：正在运行的run_train_final.py只训练，status complete只代表训练结束！之后执行artifacts/continuation_v255/finish_evaluation.py，读取evaluation_status.json，进行完整导出和八点各32回合验收。该脚本已编译、尚未运行。不同batch布局不承诺与旧六点完整行相等；要看保护行为及新增±1.4 checks。当前没有新的技能通过结论。

## v253扩展±1.4原生/导出通过，待新专家训练（第218条）

LocalAuthorityCorrection可显式追加中心、按checkpoint动态恢复。v253/zero_extended.pt sha01c6cc36…从v2515684增加索引3/4（+1.4/-1.4）；原677张量仅centers追加，原前缀/原参数/half_width/总budget保持。15测试通过、边界导出1403观测通过；原生三组source/zero/人工非零各八命令4回合：zero所有行与源相同，人工非零仅改变±1.4，六保护行完全相同。全部后台已结束，尚无新训练。下一步只训indices3/4，先短原生optimizer验证，再完整新技能预算；勿用人工非零checkpoint训练。

## v252已结束：六点仅nominal重复通过，±1.4以上均有缺口（第217条）

同5684第二seed六点32回合全过，但初态/材质/延迟及完整行与seed67完全相同，DR0下不是独立扰动覆盖。高yaw±1.4/±1.6/±1.8/±2/±2.5十点8回合全存活却全部跟踪失败；+1.4速度误差0.2726，-1.4 yaw误差0.1681。证据artifacts/continuation_v252/assessment.json。全部后台已结束，无训练；下一步隔离扩展±1.4局部专家，保护六固定点，不解冻已通过专家；恢复/导出/原生验证后再训练新技能。后续鲁棒性测试必须实际改变条件，不能重复无效seed。

## v251六点全通过；v252验证/高yaw初筛运行（第216条）

v251最终审计已执行，seed67六点各32回合全生存全通过；-0.8 yaw0.011328，五保护行与v247逐值一致。同一5684 sha3333631e…进入v252：先seed75六点32回合，再1m/s双向1.4/1.6/1.8/2/2.5各8回合初筛。worker3063263，先查其status/进程；当前无训练，不重复训练六点。仍非最终动力/DR/动态/部署认证，完整目标保持。

## v251训练与导出完成，六点评估进行中（第215条）

100更新完整结束，源及其他专家冻结检查通过，仅专家0变化；最终model_5684 sha3333631e…，完整导出通过。worker3045520/eval child3057080实际存活，先查artifacts/continuation_v251/status.json和进程。等结果后运行audit_final.py核对-0.8及五保护点，不重复启动。当前没有新技能结论，以下训练中记录为历史快照。

## v251仅-0.8局部专家训练运行中（第214条）

v248第二seed确认同v247五点通过、仅-0.8失败，已结束。新增--local_authority_train_indices 0，v250原生3更新仅专家0的20张量变化、其他源图冻结、200保护观测一致、导出通过。v249曾因0.8rehearsal+0.15continuous违反10%fixed限制在训练前失败，保留证据；已改0.7/0.15及priority48/49权重9:1。

v251从正式v2475584跑100（10critic），worker3045520/trainer3045521，run2026-09-27_13-34-12_local_single_v251；启动实查11更新、源冻结通过。先查artifacts/continuation_v251/status.json与进程，不重复启动。结束自动导出/六点32回合，再执行audit_final.py确认只expert0变化、五保护点对照。15项相关测试通过，尚无新技能认证。

## v247五点通过；quota已修，v248第二seed进行中（第213条）

v247最终审计已实际执行，192全生存，仅-0.8 yaw/stable失败（0.17067）；+0.8/±1/±1.2通过，三保护行与v243逐值相同。验收结束后才改env逐batch舍入credit，防focus变化重算历史；修前5失败，修后36相关测试通过。v248 worker3034988冻结同5584做seed75六点8回合，先查其status/进程；当前无训练。确认结果后只修剩余-0.8，避免继续训练已改善局部专家。完整目标和最终动力/DR范围不减。

## v247训练/导出已完成，六点验收进行中（第212条）

100更新全完成，源冻结检查通过，model_5584 sha f76fd74d…完整导出通过。worker3001300/eval child3020179实际存活；先查continuation_v247/status.json和进程，勿重复启动。结果出来后执行audit_final.py，检查保护点完整行一致性及失败点改善，再修第211条方向quota问题。当前无训练，只有独立评估；以下训练进度为旧快照。

## v247仍运行；方向quota历史重算问题待本轮后修复（第211条）

最近第三局部专家+1.2出现probe=0且主命令全部-1.2；不是三个专家始终有样本。实际_rebalance_turn_assignments用最新focus比例重算全部历史配额，CPU执行源码复现[800,200]历史转20%目标后新128回合全负。证据artifacts/continuation_v247/sampling_focus_review.json。未改运行实验，未停100预算；先等待本轮验收，再改逐batch累计期望credit，测试focus反转/单env reset。源冻结检查正常，勿重复启动。

## v247局部专家正式训练运行中（第210条）

续查已进入局部策略学习，最新更新数见live_check.json。最终audit已增加三个保护点相对v243完整行为行/物理极值逐值对照，编译通过、待训练结束后执行。

v246原生3更新及完整导出通过，只变local experts60张量，源图冻结、192保护观测输出相同。独立optimizer/无样本清grad/真实合计预算审计已接入，12测试通过。v247从v245 zero_local重新跑100（10critic），worker3001300/trainer3001301，run2026-09-27_12-49-25_local_turn_v247；启动实查5更新、三个gate都有样本、冻结检查通过。先查artifacts/continuation_v247/status.json与进程，不重复启动。结束自动导出/六点32回合，随后执行audit_final.py；当前未验收新技能。以下未接optimizer等为历史状态。

## v245局部模块原生/导出通过，待独立训练接线（第209条）

zero_local.pt位于artifacts/continuation_v245，sha01088f29…；源v243615张量保持、完整actor恢复/零与非零导出通过。原生两次六点seed75：零候选所有行与源v244逐值一致；人工非零只改变目标三个命令，保护+0.8/+1/-1.2行含物理极值仍逐值一致。所有进程已结束，无训练。共享authority optimizer现在拒绝局部候选，不能直接沿用--train_position_authority；下一步接只训练local experts的optimizer及inactive清grad、实际合计budget审计、短原生更新。详见docs/rmuc_local_authority_contract.md。人工非零checkpoint只诊断，训练应从zero_local.pt开始。

## v244已结束，局部修正仅CPU原型（第208条）

第二seed75仍+0.8/+1/-1.2通过，其余失败，与seed67一致；同一v243 checkpoint两seed证据见continuation_v244/cross_seed_audit.json。无后台任务。新增scripts/rsl_rl/rmuc_local_authority.py、4项CPU测试通过，设计冻结v243、只在-0.8/-1/+1.2邻域学习，总budget不增且保护点gate=0。尚未接actor/optimizer/导出/原生仿真，不要直接启动训练。下一步按docs/rmuc_local_authority_contract.md完成接线与验证，完整最终目标不变。

## v243已验收结束；v244冻结策略第二seed验证（第207条）

v243最终审计真实通过工程检查，192回合全生存但只+0.8/+1/-1.2通过。-0.8/-1过度转向，+1.2速度偏低，整体未晋升；配置修正有效但共享head仍有命令间干扰。证据artifacts/continuation_v243/final_evidence_audit.json。v244从同一v2435484做seed75六点各8回合检查，worker2990342；先查artifacts/continuation_v244/status.json及进程，不重启。当前无训练，只有独立评估。以下v243运行中均为旧快照。

## v243已完成100更新和导出，六点评估进行中（第206条）

worker2972759/eval child2986826，先查status与实际进程。run2026-09-27_12-07-47_rehearsal_fix_v243/model_5484.pt sha d310f4dd…；不要混用v242同编号。旧图冻结审计全通过、完整导出通过，最终六点结果尚待生成。结束后执行artifacts/continuation_v243/audit_final.py并检查v238/v242逐方向对照与载荷极值，未自动晋升。以下训练中进度为历史快照。

## v243配置修正对照实际运行（第205条）

续查已越过10critic预热，至少15次总更新、actor_update_l2非零，旧图冻结检查通过；live_check.json记录实际进程和累计观测命令计数。docs/rmuc_skill_status.md已补最新逐方向/跨seed表，避免按历史bank要求或单seed通过误判掌握情况。

已观察48/49双向retention统计。最终audit脚本补入v238/v242逐方向对照与上下文差异检查，编译通过、尚待结束后执行；本轮继续原进程，不重复启动。

同v2385384起点/seed66，保留v242奖励与100预算（10critic），仅显式优先weights=[1,1]和retention加入48/49，未继承v242退化权重。worker2972759/trainer2972760，run2026-09-27_12-07-47_rehearsal_fix_v243；最近至少4预热更新，params实际读回通过，六命令完整rollout计数都有覆盖，旧图不变。先查artifacts/continuation_v243/status.json与进程，不重复启动。新增计数及相关22测试通过；训练结束自动导出和六点各32回合评估，随后运行audit_final.py。该配置对照不是已解决遗忘或完成技能。以下无后台/完成等属于旧版本快照。

## v242全部结束，未晋升；rehearsal配置缺陷已修防护（第204条）

final_evidence_audit.json已真实通过工程完整性审计，192回合全生存但仅+1.2通过跟踪；−1.2改善，±1/±0.8退化。无本任务后台进程。实存优先ids[48,49]配旧13项weights，zip实际取[.02,.12]形成1:6，retention列表又缺48/49。env.py已在仿真初始化前拒绝此类override不匹配，20测试通过。后续命令必须成对覆盖ids/weights，补对应retention；门槛只能阻止晋级，不能阻止梯度遗忘，勿认为仅改gate即可解决。对照证据tracking_and_rehearsal_audit.json；最终权重暂不采用为整体改进，未原样续训。以下“运行中”均为历史快照。

## v242训练完整结束，最终评估运行中（第203条）

100更新全部完成，model_5484，旧参数/buffer检查全部通过。完整JIT/ONNX导出通过，sha8e5f7bac…；worker2951897已自动转入评估，当前child2965963。六点各16replicas/3500steps/至少32回合，seed67，operator_tracking_v1，无外环、authority consumer和修正TGS。最近日志到1000steps，结果尚未生成；先核查实际进程，勿重复训练或评估。完成后执行artifacts/continuation_v242/audit_final.py并分析失败checks及载荷/侧滑极值。当前未晋升技能，也未完成全任务。

## v242稳定跟踪训练实际运行（第202条）

从v2385384继续100更新/10critic，解析bank关闭、无外环、authority consumer与修正TGS开启。主±1.2、复训±0.8/±1跟踪缺口；旧图冻结，不按平台早停。worker2951897/trainer2951898，先查artifacts/continuation_v242/status.json和进程，勿重复启动。尚未行为验收，也未完成真实遥控/动态场地部署。

最近实查至少81次更新，冻结检查均通过且两个进程存活。audit_final.py已补充checkpoint hash及六点物理极值数据完整性检查，编译通过但尚未执行最终审计；不要把此处进度当作实时值或技能完成比例，以live_check/status和进程核查为准。以下历次“无后台训练”等状态属于对应历史版本。


## 最新执行状态：操作手稳定跟踪契约已落地（第201条）

用户明确姿态自主。默认关闭解析bank奖惩/gate/方向focus，历史严格模式保留；原物理行为与v240逐值一致。v241 seed75仅−1/+1.2在现有跟踪checks通过，其余仍真实失败；不认证整体技能。新增接触/侧滑/proxy极值与事件时长诊断。32测试通过，当前无本任务后台训练；下一步补动态操作手命令/多seed及确有跟踪缺口，勿再单为解析内倾开训。


**用户确认（2026-09-27）：RMUC轮腿步兵由操作手控制，以稳定跟踪转弯为主，姿态由策略选择；载荷、侧滑和稳定余量需验收。解析atan倾角不是必选目标。** 默认已关闭解析bank奖惩/硬gate/方向焦点，历史可显式复现；目标速度、自转、全DR、3Nm及E2E范围全部保留。新契约见docs/rmuc_operator_tracking_contract.md（docs内同目录）。


## 验收语义复核优先（第200条）

v238已完整结束。原checks中±1及+1.2仅bank失败，其他3点仍有真实跟踪失败；不是所有转弯能力为零。解析atan目标±3°对双轮独立腿机体姿态的必要性未被验证，用户验收偏好正在确认。未删除gate/改奖励/重写历史passed，未启动新训练。详见docs/rmuc_bank_gate_review.md及artifacts/continuation_v239/acceptance_audit.json；当前无本任务后台进程。


## v238位置余量正式训练已启动（第199条）

v237三次原生更新通过、旧图冻结与完整导出核验通过。v238从零head开始100更新（10critic），仅新位置余量head可训，consumer开启/外环关闭/TGS True/wrapper clip=None。worker2928772/trainer2928773；先查artifacts/continuation_v238/status.json与实际进程，勿重复启动。结束自动同consumer六点评估，无新技能认证。


## v236更新：非零余量与PPO样本语义通过

人工非零head原生目标到0.365rad，证明执行链未吞余量；不是行为认证。CPU执行真实PPO.act/仓库credit wrapper，边界采样仍保留raw logprob且ratio1。6测试通过。待接trainer新head优化器/consumer及原生少量更新验证；未启动长训练，无后台任务。见improvement第198条及artifacts/continuation_v236/summary.json。


## v235环境consumer与零head原生对照通过（第197条）

新head需要显式--position-authority-consumer，wrapper不统一clip、env按实际输入观测局部限幅。双向各4回合与旧策略初态及行为/物理目标力矩逐值相等。尚待非零余量原生验证和PPO优化器/采样链；未训练，无后台任务。证据artifacts/continuation_v235/comparison.json。


## v234完整策略重建/导出已接入位置余量head（第196条）

zero_authority.pt保留旧573张量、1131状态执行一致；零/非零head完整JIT/ONNX通过，24测试通过。环境/训练consumer仍待接通，旧uniform clip1评估器显式拒绝新head，勿直接训练或部署。下一步统一wrapper/env限幅、历史动作与PPO采样语义。无后台任务。证据artifacts/continuation_v234。


## v233位置余量原型可审阅，未接入训练（第195条）

采用旧clip执行值+双向独立可学习修正，active额外0.15rad、轮小预算；1131真实策略探针零初始化执行一致，3测试通过。必须先完成wrapper/env限幅、checkpoint/完整导出、历史动作和PPO分布语义，勿直接挂原型开训。详见docs/rmuc_position_authority_contract.md和artifacts/continuation_v233/audit.json。无后台训练、无新行为认证。


## v232位置目标余量诊断完成（第194条）

负转向允许左主动腿目标0.22→0.37rad后bank8.63→3.19°，未饱和力矩，但速度/yaw恶化；正向需相反差动，亦未通过。证实应做位置目标余量与腿轮协同适应，不直接保留诊断偏置当E2E。当前无后台任务。证据artifacts/continuation_v232/summary.json。


## v231已结束：修正物理下内倾仍失败（第193条）

100更新/90策略学习，192回合均生存，但六点bank仍失败。最终5284 sha a0e50e48…；速度一致性改善保留，六点左主动腿动作100%饱和而腿力矩未满。下一步解决局部位置指令余量，勿照旧配置盲目续训。当前无本任务后台进程，证据artifacts/continuation_v231/{final_evidence_audit,leg_physics_summary}.json。


## v231修正物理适应实际运行（第192条）

worker2447478/trainer2447479，从v223最终5184继续100更新（前10critic），显式TGS external forces=True、无任务外环、双向局部分支，腿17/0.010、DR0。params读回通过；勿重复启动，先看artifacts/continuation_v231/status.json与实际进程。结束自动在相同TGS设置评估六点；非转弯E2E仍是未完成项。


## v230物理扩展对照已结束（第191条）

第二seed双向转弯再次确认TGS开关显著降低速度—位置差分偏差；倒退命令组同样改善但没有跟踪倒退成功。直行两设置都无稳态样本，零速新设置失稳，不能宣称全任务验证通过。默认solver仍未切换，当前无本任务后台进程。下一步在修正物理下明确适应与非转弯E2E缺口，勿沿用旧带外环技能证书。证据artifacts/continuation_v230/comparison.json。


## v228/v229关键物理发现（第190条）

腿目标饱和但力矩未满；稳态joint_vel与位置差分明显不一致。仅开启TGS external_forces_every_iteration后偏差明显下降，但旧策略bank仍失败。默认solver未切换；当前无本任务后台进程。先核验物理一致性再训练，勿用DR掩盖求解偏差。证据artifacts/continuation_v229/comparison.json。


## v227腿反馈分离诊断结束（第189条）

相同初态对比实时腿反馈与2秒后固定腿参考：部分失稳被改变，但正yaw无偏置基线固定腿也会倒，不能简单移除旧腿反馈；全部压弯门槛仍失败。当前无本任务后台进程。下一步测量腿目标角/实际角/计算和实际力矩，查0.22动作尺度下的位置命令余量，不再扩展同类常数偏置扫描。证据artifacts/continuation_v227/comparison.json。


## 腿电机契约更新（第188条）

用户确认DM-J8009P-2EC、9:1。默认实际solver限速已修为17rad/s；armature原本是USD继承0.010，现显式保持。腿DR随现有strength覆盖速度[12,17]和armature[0.008,0.024]，工程先验待辨识，物理写入已原生验证。当前±1任务腿速峰值仅2.78，参考惯量0.0158也未解决外倾；勿将问题归因“跑到60/没有armature”。v226及docs/rmuc_leg_actuator_audit.md保存证据。旧无限速/无腿DR历史报告不是新条件认证。


## v224/v225姿态诊断结束（第187条）

四关节9组±0.3基底与起稳后平滑对照均未通过；平滑可救回部分起步tilt，但出现正倾角的案例实际倒退，不能当正向压弯成功。诊断内倾符号已考虑实际速度方向，原始报告另附勘误。当前无本任务后台进程；无新策略晋升。证据artifacts/continuation_v225/{comparison,inward_semantics_correction}.json。


## v223完成：yaw退化受控，内倾仍未通过

100更新/90策略学习、六点192回合全部生存；与上一轮相同评估初态，+1yaw误差0.529→0.029、+1.2为0.438→0.052，但全部bank仍失败（约7.35–13.89°），不晋升。当前无后台任务。下一步核查四腿动作组合与执行饱和下的姿态可达性，勿继续盲目加权或重复启动旧worker。证据artifacts/continuation_v223/final_evidence_audit.json，improvement第186条。


## v223新奖励对照已启动（第185条）

v222证实旧奖励允许用yaw换速度（合成对照收益+5.78），新增显式低速大转弯相对+绝对yaw约束后变−0.80；16测试通过。v223从v220转向初始化起跑100更新，保持无外环/双向新分支，仅启用新yaw约束；保存间隔10。新状态artifacts/continuation_v223/status.json，勿重复启动；完整压弯尚未通过。


## v221已结束：完整预算、192回合均生存但六点未通过

最终5184（sha b0505769…）保留100更新/90策略更新，正向yaw显著退化（+1误差0.529，+1.2误差0.438），全部bank失败。不晋升、不直接续训同配置；先审计低速大转弯fine-yaw资格和速度/yaw取舍。v220初始化保留可回退。状态及证据artifacts/continuation_v221_resume/{status,final_evidence_audit,initial_final_comparison}.json；第184条。所有worker已结束，勿误判旧PID仍活。


## v221中断恢复（第183条）

原worker/trainer已消失，59次日志仅41次保存于model_5125.pt；18次未保存更新已记录损失。从5125保留optimizer恢复剩余59次，不再预热。最新状态以artifacts/continuation_v221_resume/status.json及实际进程为准，原v221状态已标中断，勿重启旧worker。


## v221实际训练中：无外环双向腿轮协同

worker2012827/trainer2012829，100更新预算（前10 critic warmup），主1m/s±1.2，复训±0.8/±1，外环False，DR0，腿std0.03/轮0.012。仅两E2E分支可训练；不按平台早停。状态见artifacts/continuation_v221/status.json，运行日志及自动导出/六点评估由train_e2e_bank.py负责。第182条记录了首轮std保护拒绝及恢复；勿重复启动，先核实进程与jsonl。


## v220：开始将轮转向学进策略

双向E2E局部分支已完成500次轮动作监督初始化（非PPO），旧参数冻结、同输入腿输出保持，配对数据9024条，验证RMSE0.182→0.016；22测试及完整导出通过。六命令点无外环48回合已结束，全部生存1，±1rad yaw误差0.0293/0.0399，初步学入转向；但全部bank仍失败，正向还欠速。结果见artifacts/continuation_v220/behavior_summary.json。尚未认证压弯；详见第181条。


## v219完整对照：先学E2E转向，再联合内倾

同初始状态双向各16回合，关闭任务外环后1m/s±0.8实际yaw近0，误差约0.8；带外环约0.04。两者均未过正确bank。详见第180条及artifacts/continuation_v219/comparison.json。±1/±1.2四目标按最终契约尚无通过证据；没有正在运行的适应训练，不误报训练完成率为技能完成率。


**最新约束：最终E2E RL。任务级速度/pitch/yaw/推进外环只作过渡teacher，最终必须移除；不得复现外环冒充策略能力。** 第179条已增加显式无外环路径及短探针；尚无E2E技能认证，当前仍集中克服压弯饱和与腿轮协同瓶颈。

## 当前优先：先克服低速压弯瓶颈（第178条）

v217已定位并修复远端bank惩罚平台：6°内保持，远端有界连续引导。14测试通过；原生80回合已完成但全部综合失败：负转向腿动作触及clip1，正转向姿态改善伴随速度误差增加；见artifacts/continuation_v217/reachability_summary.json。先解决正确内倾，不推进轮速迁移、部署扩展或盲目增加critic/侧滑奖励。详细见docs/rmuc_bank_bottleneck_review.md；尚无新策略内倾通过证据。

## 用户最新决定：暂缓速度环，RL直接轮力矩，LQR仅对照或备用接管

此决定优先于前文速度环提案。当前默认力矩控制从未切换，v216已结束且不自动恢复；无迁移训练继续运行。确认部署不是RL→LQR内环，也不假设现成闭环轮速。deployment_contract见configs/rmuc_hardware_target.yaml和docs/rmuc_deployment_control_contract.md。

v216连f0行为基线也失败，不能把v214短时数学等效当完整迁移成功；原型仅存档，暂不排查扩展。主线回到物理内倾修复、直接轮力矩3Nm/径向助力/最高DR。注意导出actor后仍有仿真外环与限幅，最终要复现或迁出这些环节，不能只乘3就部署。详见第177条。

## 用户选择轮子部分迁移：v216分段速度环评估中

第176条：只改轮执行器控制，腿位置控制/参数、旧策略权重、气弹簧和USD保持。v215直接Kp0.2/3Nm三组均早tilt，但原生2001次计算真实限幅3Nm、无旧D叠加；不代表速度环不可行。新增fraction0..1轮子渐进迁移，f0等效旧0.1/5.5，f1目标0.2/3；过渡>3Nm仅仿真。15控制器相关测试通过，腿参数与gas/响应标志保持。

v216 worker1982500串行评估f0/0.05/0.1，每批三命令24env，旧5084不改、无训练不晋升，最新看status。物理内倾腿姿态缺口仍独立存在，不把速度环当压弯修复。默认训练未切换。

## 轮速控制原型可审阅：等效链路已原生验证，默认训练未切换

第175条：wheel_velocity_control.py实现离散PI/限幅/antiwindup/filter/reset，独立rmuc_wheel_velocity_migration.yaml给出工程提案，Kp与机械阻尼分开。显式eval等效诊断8env128steps得到513次physics计算，最大力矩差1.192e−7Nm；这是旧5.5Nm路径的P环等效，不是3Nm目标控制或实机标定。v214已complete，无自动训练。默认25D/6D/旧策略仍保持。

v213腿差动所有组均未通过新内倾门槛，幅度0.24最多把外倾减小或接近直立，尚非物理内倾。先确定可复现的最终速度环及姿态学习接口，勿将控制器单位误当机械damping或直接替换原两轮action语义。

## bank修复已落地：新契约world_heading_inward_v2，原策略仍须姿态适应

第174条：bank_geometry共享物理参考统一reward/课程gate/eval，旧25D与动作/USD保持，25项测试通过。v212同5084、同初态、四点各64回合复测：速度/yaw/生存全部与旧报告精确一致，新bank误差6.735–11.993°全部不合格；不能继续使用旧符号通过证书。报告provenance_version2和bank_contract已显式记录。

v213物理内倾腿差动诊断已实际运行worker1963206/child1963207（1m/s±0.8、差动0/±0.12/±0.24、120env）；不训练、不晋升、不修改生产预算。最新核对artifacts/continuation_v213/status.json。先确定内倾动作与速度耦合，再充分训练实际姿态缺口。

## 最高优先级纠正：当前bank奖励/验收鼓励物理外倾，暂停按旧gate宣称压弯掌握

用户要求对照原V14并检查物理/观测，已定位确定坐标问题：canonical[-z,x,y]与−atan(v*yaw/g)组合方向错误；独立世界几何核验内倾被罚、外倾被奖，v211原生世界向内倾角正负方向均负（−6.18°/−3.31°）确认。旧速度/yaw/生存报告保留，但bank通过只代表旧符号gate，不能当物理内倾证书。原v177审计缺独立世界几何oracle，结论需按此纠正。

原V14实际actor原始command+height等，critic另有线速度/高度特权状态；默认轮速控制、reward乘dt；未见RMUC解析bank目标。RMUC25D缺vy/COM，gyro0时yaw2/2.5/3/10 clip成2。当前1m/s额外fine-yaw奖励未开，高yaw仍受旧1rad超速罚。真实COM约0.40m而root约0.315m；旧0.26轮滑移指标混正常内外轮差，局部比较约0.048。net_forces_w仅NORMAL force，勿当摩擦利用率。

完整报告docs/rmuc_v14_reward_observation_review.md；证据artifacts/continuation_v210及v211，详见improvement第173条。下一步先统一物理bank坐标、reward/gate/eval及观测范围契约，保留旧25D与旧checkpoint，再训练，不继续盲加原bank权重或按旧gate晋升v209。当前本组诊断均已完成，无新训练。

## v208完整100次训练及验收结束：+1.0仍失败，v209局部诊断接续

v208实际100updates/90学习、5084完整图导出与同hash审计通过。±0.8/−1.0各64回合保留通过；+1.0仍linear0.2238/yaw0.05389失败，未晋升。无平台早停，勿重新跑同一实验。v209在5084上做+1.0四组轮偏置小样本诊断（64env/1500steps），worker1939387，最新看status；不自动训练、不把diagnostic行内通过当完整策略认证。详见第172条、v208/final_evidence_audit.json。

## v208预算内训练继续，最终只读审计已挂接

v208仍原worker1916755/trainer1916756运行，无STOP或重启。额外只读worker1923559等待训练、导出及评估结束后核对同hash、更新预算、参数隔离、64回合及实际行为指标；最终报告为v208/final_evidence_audit.json，尚未生成。CPU防误报检查通过，不代替技能验收。详见第171条。实际进度仍看run/state_router_training.jsonl。

## v208仍训练：5000已学checkpoint仅新分支改变

最新核实至5003，共19updates/9次非warmup学习，正1.0尚未过门槛；继续100预算，不平台早停。worker1916755/trainer1916756实存。5000的1091保护输入输出一致，旧参数/buffer不变，新分支非零学习，完整JIT/ONNX导出通过；证据v208/learning_snapshot_5000.json和export_snapshot_5000。status的training_updates=0是初始字段，实际进度读取run/state_router_training.jsonl。详见第170条。

## v208正1.0新分支已实际训练，100次上限无平台早停

v205的±0.6/±0.8四点64回合通过；v206−1.0也64回合通过，+1.0与±1.2失败。v207零初始化正1.0专用新层保护全部负yaw/正≤0.8，40测试、完整导出通过。v208修复启动编排后worker1916755/trainer1916756已4985实际更新；100次含10critic预热，无平台期早停，结束自动导出/验±0.8和±1.0。run2026-09-26_10-15-14_positive_1p0_turn_v208。最新status在v208；失败启动记录归档保留。详见第169条，当前没有新技能训练通过声明。

## v205新增0.8局部可学习分支，完整图评估中

v204正向共同−0.02+差动0.03诊断行内通过；结合负向共同+0.02，创建新独立双向层，gate保护≤0.6，seed与学习量共享cap。旧v202全部tensor与1067保护输入不变，38项测试、完整导出通过。worker1348690/child1348805评估±0.6/±0.8各至少64回合，最新看artifacts/continuation_v205/status.json；无新训练、不自动晋升。详见第168条。用户允许新技能跑满原预算，勿再套用旧平台期STOP。

## 用户最新预算规则：新技能充分探索，不因短平台期早停

用户允许新技能跑满原定训练上限；避免GPU浪费针对相同条件下已充分稳定掌握技能的重复训练。后续勿重启v198平台期monitor或单凭短期改善小提前STOP。保留旧技能rehearsal/retention与最终验收；单seed nominal并非全部条件稳定。规则已置顶docs/rmuc_skill_status.md与docs/rmuc_long_term_goal.md，详见improvement第167条。

v203±0.8共同模式诊断已完；v204正向共同−0.02+差动局部诊断接续运行，最新看artifacts/continuation_v204/status.json。不复训已过±0.6，不把小样本诊断当正式能力。

## v202本组评估全部结束：±0.45保留、±0.6新增通过

同完整候选±0.45和±0.6各64回合通过；±0.8仍失败，overall False。无新训练更新，本组进程已结束，下一步只针对±0.8并保护已过±0.6。证据v202/{status,new_turns,retained_turns,nominal_progress}.json，限定旧动力单seed DR0。

## v202±0.6各64回合nominal通过；±0.8仍失败

同seeded完整图候选，+0.6 linear0.06585/yaw0.04760、−0.6 linear0.11911/yaw0.03052，生存1/stable1且全部行检查通过，无新训练。±0.8速度/稳定仍失败，勿外推或复训已过±0.6。证据v202/new_turns.json与nominal_progress.json；旧±0.45保留复验运行child1338884，先核对status。仅旧5.5Nm/0.1单seed DR0，不是目标动力或最高DR证书。详见第166条。

## v202局部轮seed候选正在正式网格评估，无新增训练

v201正向共同−0.04加差动0.01小样本行内通过；结合反向共同+0.04，构建保留4984全部参数的局部seed候选。新轮seed独立cap0.06，局部总轮残差cap0.10，旧图/旧budget不变；36测试通过、909保护输入输出一致。worker1336754先完整导出，再四新点+两保留点串行各64回合，检查v202/status.json。候选未晋升，正式参考仍4913，无新训练。详见第165条。

## 速度误差方向更正：正转慢、反转快；v201正向差动诊断

此前把绝对速度误差统称欠速不准确。原signed结果显示正yaw慢、负yaw快。v200共同−0.04改善正向速度但yaw0.0503略超0.05；负向共同+0.04/+0.08小样本行内通过，整体仍诊断不可认证。当前v201只对正yaw作共同−0.04+小差动敏感性（64env），worker1332944；勿直接加腿预算、放宽门槛或把偏置行通过当策略通过。详见第164条、v200/directional_findings.json及v201/status.json。

## 4984新点失败、旧点保留；v200轮共同模式诊断运行

v198恢复评估完成：±0.6/±0.8各64回合全存活但欠速，±0.45各64回合保留通过。真实稳态快照回放一致，已验证局部clip余量不足的采样线索。v199左腿−0.16使负方向error0.244→0.180，正方向反而0.197→0.269，无通过；不能统一扩大预算。v200仅串行轮共同偏置敏感性（120env，0/±0.04/±0.08），无训练或晋升，worker1330854/child1330855，先核对真实状态。详见第163条。

## v198已停滞停止4984，拆分评估正在接续

71updates/61实际学习后正常STOP保存4984，节省29次预算；旧图冻结、最终导出通过。原6×32评估192env被128上限拒绝，并非训练失败。新resume_evaluation.py串行4新点128env+2保留点64env；worker1322265/child1322266已实际运行。最新看artifacts/continuation_v198/resume_status.json，旧status保留启动失败。CPU快照分析随后自动执行，不重训、不自动晋升。详见第162条。

## 欠速诊断补充：基线左腿action3持续clip

v196四点steady raw action3饱和率1，±0.6 raw均值1.083/1.089，执行为1；轮动作/力矩未饱和。现新腿预算0.08在这些固定状态上的平均负修正仍不能完全解除clip，但未证明欠速因果。v198继续原100updates，先取得最终快照，失败后优先局部敏感性诊断，勿直接扩大全部预算。详见第161条及v198/baseline_action_headroom_audit.json。

## v198运行中：已学checkpoint隔离/导出通过，欠速尚未解决

至4936训练仍活跃，正负linear EMA约0.201/0.246，旧图/缓冲不变；不提前晋升或追加预算。真实4925只新分支改变，909保护输入输出差0，新域变化非零；JIT/ONNX一致。为避免全存活欠速没有终止轨迹，最终既定评估新增有界steady_tracking_probes，CPU实际代码块验证完成、原生待跑。详见improvement第160条及artifacts/continuation_v198。

## v196新转弯确认欠速，v198仅新双向分支训练中

v196正式4913的1m/s±0.6/±0.8各64回合全存活但四点跟踪失败（linear0.199–0.302）；v198新增局部双向零初始化分支，旧全部tensor与2948输入输出一致，35测试通过、JIT/ONNX初始化完整图一致。已启动worker1224256/trainer1224257，最多100updates只学习±0.6，旧图冻结，stage47只是命令坐标并非此前能力已过。最终自动验六点和学习后导出，不自动晋升；详见第159条、artifacts/continuation_v198/status.json。

已审计当前IdealPD不使用velocity_limit63.3734做速度力矩限制；物理velocity_limit_sim默认来自USD，启动读回约1.7e4rad/s。因此3.80m/s不能当现仿真硬上限。

## 4m/s硬件速度依据：独立迁移damping先验更新

用户报告原车可达4m/s；按r0.06m/每轮3Nm，持续无滑动无额外推进条件下黏性阻尼上界0.045。迁移DR暂取0.005–0.04、中心0.0225，非实测，历史0.005–0.08保留；当前训练动力和USD未热改。审计v197/speed_conditioned_damping_audit.json，11项profile测试通过。分数迁移入口已支持但未原生验证。详见improvement第158条。

v195已complete，5135仍欠速（0.162740/0.183243>0.15），不能晋升，不原样追加训练；正式参考4913。v196新转弯基线串行评估中，请先核对status及进程。

## v195沿已验证斜坡路径定向学习欠速；新硬件中心直接迁移失败

v194目标动力探针读回wheel D0.0425/limit3Nm；1.0/2.0/3.0全部早期tilt，旧策略不能直接替换动力。径向10.775N配置已设，报告按CLI记录导致每env力值缺失，已修入口，未回填旧数据。见第156条。

v195 worker1186105/trainer1186106已实际训练，从5035+匹配Adam继续，只有新直行分支可学习，2秒真实零速起步命令斜坡、strict0.15/1200 gates、最多100updates/进18停。斜坡期间不计最终跟踪，早期失败仍计survival；32相关测试通过。最终斜坡两点通过才追加固定3.0检验，不重复已过低速。先核对artifacts/continuation_v195/status.json及真实进程。详见第157条。

## v193起步斜坡使3.0全存活，仍欠速；目标动力诊断入口已接

相同5035、相同物理初态/布局，1.0→最终目标2秒斜坡后2.95/3.0各64回合survival1；error0.163892/0.185375仍超0.15，不能宣称通过。3.0固定阶跃全tilt记录保留；后续应针对可存活的命令进入路径学习稳态欠速，不原样冷启动复训。证据artifacts/continuation_v193/comparison.json，详见第154条。

--diagnostic-hardware-profile已可显式把目标3Nm/0.0425中心与给定DR区间、每腿10.775N、独立电机响应写入仿真配置并读回；尚未原生目标动力评估，诊断不作证书。20项相关测试通过，见第155条。目前v193已结束，无新训练。

## v190提前停止且独立验收失败；v192电机响应原生接口通过

v190在5035按STOP保存退出，共122updates，因>100次实际学习仍survival0而节省剩余78次预算。2.95实测64回合生存1但speed error0.221539失败；3.0的128回合全tilt。5035不替代已验收4913，不原样续训。证据artifacts/continuation_v190/{early_stop_decision,final_straight,status}.json。

串行8env/128step独立电机响应探针完成：六路T实际在2–30ms，response状态与真实actuator.applied_effort一致且限幅有效；诊断passed=False，不是技能/最终3Nm验收。v192/native_motor_response_probe.json；当前这组worker正常complete，无新训练。详见improvement第152–153条。

## v192独立电机响应可选实现，默认关闭，原生验证待做

新motor_response.py覆盖六路电机的physics-step一阶施加力矩响应，独立于观测/通信FIFO；工程探索T=2–30ms，非RM3508实测，随DR渐进，reset独立采样。默认关闭，当前v190未启用。20项CPU相关测试通过，尚未原生仿真或鲁棒性验收。诊断评估写实际T并强制不可认证，observation_only关闭该物理模型。见improvement第151条、artifacts/continuation_v192/implementation_status.json。

v1905021仍运行，继续原预算，无新增并行GPU任务。

## v191修复damping bridge饱和语义，当前v190训练继续

已确认旧bridge速度补偿在effort target限幅后不再等效目标D，可能出现实际力矩与日志相反。现改为直接插值explicit actuator.damping tensor，最终effort限幅保留；未启bridge路径保持。11项实际源代码CPU测试通过，尚未原生仿真迁移验收，见improvement第150条和artifacts/continuation_v191/damping_bridge_audit.json。

v190仍为原5.5Nm/0.1、bridge关闭的有界直行恢复；不会热改其动力。最终目标为用户确认每腿10.775N、3Nm、damping估计区间及DR，不把旧训练条件当最终硬件通过。

## 用户动力参数最新澄清：每腿10.775N，允许工程估计+DR

RM3508轮端持续最大3Nm；damping估计0.005–0.08；气弹簧**每条腿各10.775 N径向力**，不是10.775Nm或40.93N。用户允许用DR覆盖差距，不因缺完整实测暂停仿真。独立目标配置configs/rmuc_hardware_target.yaml，尚未热改v190；拟damping中心0.0425覆盖给定区间，推力±20%仅工程提案。

CPU力矩边界显示：若0.08是未补偿粘性损耗，0.06m轮在3.2m/s需要4.2667Nm抵消阻尼，超过3Nm。需审查阻尼层级与真实执行器语义及验证可达性，不删端点或宣称DR能突破动力限制。见improvement第149条和artifacts/continuation_v190/hardware_target_screen.json。

## v190早期尚未恢复：继续原有界实验，不追加GPU任务

1140476/1140477本轮已实核存活。4942最近10完整窗口active vx均值−0.4788、survival EMA0；参数在更新但尚无驾驶改善证据。不能将旧retention min survival1解读为全部gate通过。证据artifacts/continuation_v190/learning_window.json；最多200updates/实际进18停止和最终2.95/3.0验收不变。详见improvement第148条，最终全目标保持active。

## v188八点DR0.05全过；v190开始只恢复3.0直行

正式4913同checkpoint八点nominal与DR0.05均各64回合全过、survival1；这一强度不再重复训练，最高DR仍未过。技能表docs/rmuc_skill_status.md已更新。

v190从4913新增零输出直行分支，2948输入不变、原tensor保持、完整导出一致。worker1140476/trainer1140477实到4927；model4925核对仅新直行分支20参数变化，旧图/buffer保持；最多200updates/进18停，仅recovery_straight_3p0训练，其他router也显式冻结。选择旧Level17的3.0命令并收紧gate为0.15/1200，DR0；最终只评估2.95/3.0受影响区间。先查artifacts/continuation_v190/status.json及实际进程，不重复启动。首轮CLI吞Hydra参数的启动故障已修复并保留*_failed_cli，无训练更新，不算策略失败。见improvement第146–147条。

## 技能分类已审计：v188训练60updates停止，4913八点nominal全过

用户要求避免误判技能而浪费GPU。最新清单docs/rmuc_skill_status.md；机器证据artifacts/continuation_v189/skill_status.json，CPU脚本可更新。1.0/1.5旧转弯、过渡直行、2.2/2.4旧小yaw及2.5–2.9直行有nominal通过和当前图保持证据，暂不从头复训。4913直接八点各64回合全过，2.0/+0.3已不是nominal失败；3.0直行仍明确待恢复，最高DR和新大yaw另列。

v188进19于4913自动停训，仅60updates；worker1119770当前final_low_dr/child1132078独立评估DR0.05，先核对真实进程，不再追加训练。低DR起点未先测是本轮遗漏，不能把后来通过都归功学习；后续训练先定位确实失败的条件，未测试不当失败。见improvement第145条。

## v189扩展课程渐进化；v188真实4875仅恢复分支更新

v188低DR训练仍在运行，model4875逐tensor核对仅2.0正向恢复分支20张量改变，旧actor/router/buffer保持；最终独立验收仍等待。先查artifacts/continuation_v188/training_status.json和实际PID，勿重复启动。

磁盘大转速课程已由15扩展改为35阶段（47–81），原0–46完整保持，各速度yaw增量≤0.25。扩展DR单独从约2.78%渐进至1，旧调度和显式评估强度保持；17项相关测试通过。82总阶段是配置，不是技能通过；新专家支撑、yaw编码/reward及最终联合DR仍待完成。见improvement第143条与artifacts/continuation_v189。

## v188八点nominal全过，正式低DR学习已启动

显式seed保留正式4853全部旧张量：八点各64完整回合、survival全1，2.0/+0.3 yaw0.028999、bank1.11250°，全部checks通过。证据artifacts/continuation_v188/seed_nominal_retention.json。不是全域/最高DR证书。

worker1119770、trainer1119771已实际训练，DR0.05固定、最多200updates至5053/实际19entry，随后最终同checkpoint八点nominal与DR0.05评估；不自动升DR/续训。先检查training_status.json、日志和实际进程，不按历史PID重复启动。长期稳定完成全部任务及高鲁棒性目标保持active，见docs/rmuc_long_term_goal.md。详见improvement第142条。

## 2026-09-25 接续v188：长期目标已建立，正式4853显式种子足量复测中

长期最终目标见docs/rmuc_long_term_goal.md，Codex goal保持active。v188保留正式4853所有已学actor张量，显式加入局部轮差分seed；旧加载等价、2112输入域外保持、JIT/ONNX验证通过。八点nominal评估PID1114805已实际到3000steps，报告待定；先查artifacts/continuation_v188/launch.json及进程身份，勿重复启动，不用v186 test-only续训。

大转速final_turn_envelope已接env reset连续采样，旧0–46阶段分布逐位保持，只在已开放扩展锚点间扩大范围；12项采样/包络测试通过。尚缺渐进课程/DR调度、观测reward与新专家支撑，不能视为技能完成。详见improvement第140–141条。

## 最新约束与审计：延迟 DR 必须保留，旧 checkpoint 不得隐式改动作

观测与动作通信延迟已实现为独立 reset 采样的 1–3 控制周期，当前各 8–24 ms，闭环两段可合计 16–48 ms；低→高 DR 渐进，最终须联合最高 DR 验收。独立电机响应滞后尚未建模/标定，不能用动作 FIFO 或 damping/gain DR 冒充完成。v187 最高 DR 仍失败。

已修正旧 recovery checkpoint 缺失 wheel_seed 时自动注入非零种子的回归：默认及缺失种子均为零，显式保存种子不变。第137条兼容性表述及基于该错误加载的等价对照应以本条纠正为准。34项相关测试通过，无新增训练成功证据。

新增较大角速度目标仍在准备，旧 Level46不再代表整个扩展目标完成；追加课程配置不等于连续随机命令与新专家训练已完成。详见 improvement 第139条。

## v187最高DR单点失败：固定轮差分未达鲁棒

2.0/+0.3 DR1生存0.9844、yaw0.06384、1次tilt，coverage/linear/yaw/stable失败；nominal种子仅证明动作方向，不是最终策略。full flat DR实际开启并记录。后续应在隔离分支内渐进DR训练，不能部署该candidate。详见improvement第138条。

## v186轮差分种子修复2.0/+0.3 nominal，v187最高DR单点运行

种子−0.01/+0.01使2.0/+0.3 yaw0.02975、bank1.11863，邻近7点保持；v186 episodes不足仅因诊断采样数，非行为失败。v187 PID1047515以64replicas和DR1复测2.0/+0.3，结果待定。候选不部署、不替代全DR。详见improvement第137条。

## v183八点评估完成：7通过，2.0/+0.3仍欠速；v185轮差分诊断运行

model4853八点中7点全通过；2.0/+0.3 yaw0.08485、signed−0.10152，生存1但yaw/stable失败。2.0/+0.25 bank2.88660°仍通过。PID925773进行5档轮差分诊断，非验收/非训练。详见improvement第136条。

## v183训练已结束，4853旧actor等值审计通过；八点评估运行中

100updates完成、冻结违规0，最终old_actor_changed=[]。worker910584当前eval PID921344已实际执行首动作；等待recovery_retention_grid.json，勿重复启动/原样续训。当前不能宣称2.0恢复、3.0或最高DR通过。审计training_audit_summary.json，反思第135条。

## v183至4829冻结审计正常；v184高速直行分支仅完成实现验证

可选recovery_straight_3p0只在speed>2.9近直行生效，零初始化、独立参数/限幅，15项测试和JIT/ONNX验证通过。v184 checkpoint仅测试，不可部署或续训。当前仍以v183 trainer910585和其八点最终评估为主，未启动第二个训练。3.0实际能力/最高DR均未完成。见improvement第134条。

## v181结束39/41通过；v183修复启动后已真实更新

失败点2.0/+0.3 yaw/stable、3.0直行pitch倾倒（survival0）。完整统计artifacts/continuation_v181/final_summary.json，不是全项目完成度。v183首轮anchor结构不匹配无更新，已保留日志并改用v182零扰动同结构anchor；当前worker910584/trainer910585实到4765、预热结束、新恢复分支实际更新、旧图冻结审计通过。等待最多100updates及八点独立验收；3.0与最高DR仍待解决。详见improvement第132条。

## v181完成28点：27通过、2.0/+0.3重复跟踪失败；v183等待PID更新

2.0/+0.3 yaw误差0.09025、生存1但yaw/stable失败；新通过2.0/−0.25、−0.3和2.2/+0.15。当前grid07 PID810946。恢复调度已更新为810909，仍等v181，训练后验收含2.0±0.25共八点；旧804225仅为历史。新增steady signed误差供后续报告诊断，gates不变。详见improvement第131条。

## v181前24点通过，2.0直行/±0.15/+0.25也保留

grid05已通过、生存1；2.0/+0.25 bank误差2.88733°接近3°门槛，恢复+0.3时需保护邻近点。当前grid06 PID809001已核实，v183仍排队。最高DR尚未过，不把小角度通过外推为全速度完成。详见improvement第130条。

## v181过渡速度复测继续，v183仍为等待状态

grid04已完成：[(1.2, 0.0, 1.0, []), (1.35, 0.0, 1.0, []), (1.6, 0.0, 1.0, []), (1.75, 0.0, 1.0, [])]；以grid_04_report.json为准。 当前grid_05.json子PID806813已核实，v183尚未训练。新增router/layout/命令清单source hash供后续报告。详见improvement第129条。

## 低速nominal复测更新：1.0/1.5全部16个课程命令已通过

v181 grid00–03全部通过，每点至少64回合、生存1；覆盖1.0直行/±0.1/±0.2/±0.3/±0.45，1.5直行/±0.1/±0.2/±0.3。当前grid04子PID804334继续其它命令，v183排队等待。不是最高DR认证。

## v183恢复实验已排队，等待v181完整复测结束

v181已确认1.0全部9个课程命令通过，1.5直行/±0.1通过；剩余继续。v183 worker804225等待v181真实退出，随后单次最多100updates的2.0正转恢复、旧actor张量等值审计及2.0/2.2/2.4双向独立验收。未开始训练前勿称能力恢复；状态artifacts/continuation_v183/status.json，勿重复启动。13项router/training测试通过，详见improvement第128条。

## v182隔离恢复候选准备完成，v181旧技能复测继续

zero_recovery_candidate.pt为4753零扰动增加2.0正转分支：399旧tensor相等，2048输入差0，尚未训练，后续需--reset_optimizer。12测试及非零分支JIT/ONNX一致性通过，nonzero_recovery_export_test_only.pt不可部署/续训。v181前8个1.0命令已全部过，grid02运行；nominal runtime摩擦shape静/动均0.5。全41点/最高DR/最终3Nm仍未完成。详见improvement第127条。

## v181首批1.0旧技能已过，其余继续；新训练须覆盖真实缺口

grid00的1.0×{0,±0.1,+0.2}全部checks通过、生存1。当前grid01子PID797803，worker788585，勿重复启动。实际router支撑审计发现41个固定点只有2.4±0.3可适配，无法原样训练修复2.0正转；router_command_support.json记录。应按复测缺口加隔离适配而非无效延训，尚未改网络。见improvement第126条。

## v180结束：最高DR未过、2.0正转retention未过；v181低速复测运行

nominal2.4双向通过；2.0/+0.3虽生存1但yaw误差0.09061导致yaw/stable失败。fullDR2.4正/负生存0.83077/0.44444，均未过；不能宣称鲁棒完成。v181 worker788585/current grid00 child788985实核运行，以status.json为准。新增评估运行时摩擦读回供后续报告使用。用户最终全速度连续随机命令及切换泛化要求已记入，当前采样不等于该目标完成。见improvement第125条。

## v181补齐全部老技能：41命令点排队复测

用户指出1.0/1.5缺口，已按实际Level0–28展开41个去重命令，优先低速直行/双向转弯，manifest位于artifacts/continuation_v181。worker788585等待v180真实退出后串行执行，每点至少64回合，同4753/正式flat网格。只能根据各grid_report判通过，冻结参数不代表行为保留；本轮nominal不替代最高DR。详见improvement第124条。

## v180正式2.4双向已通过，retention/最高DR队列继续

nominal_large_turns.json passed=True，每向64回合survival1，bank2.59899/0.15419°；复现诊断结果。worker784180当前子PID785739为nominal_retention，随后full_dr_large_turns；以status.json及PID复核为准。USD摩擦属性未显式写入，需运行时材质读回，尚未通过多摩擦/最高DR全目标。详见improvement第123条。

## v180生产flat修正已接入，串行验收运行中

flat_mesh_subdivisions默认5、0可回退，原平面/材质与USD保持。16项几何/reward/layout测试通过。worker784180串行同4753的2.4双向nominal、2.0/2.2retention、2.4最高DR；以artifacts/continuation_v180/status.json子PID和实测JSON为准，勿重复启动。当前只flat，多摩擦联合DR、全速度及3Nm部署仍未完成。详见improvement第122条。

## v179已结束：原材质细分mesh双向全部单项通过

同4753/seed44/初态，仅三角细分，2.4±0.3各64回合survival1、bank2.59899/0.15419°，原材质静/动摩擦0.5。报告artifacts/continuation_v179/subdivided_mesh_grid.json。原大面mesh失败而解析plane/细分mesh均成功，下一步候选修正flat网格并做retention、多摩擦联合DR；最终目标未验收，无活跃trainer/评估。详见第121条。

## 最终范围更新：flat多摩擦鲁棒性必验；v179网格细分运行中

用户只要求flat，不新增rough地形；最终同policy必须跨平地摩擦并联合最高DR。现有DR在机器人shape上缩放静/动摩擦0.85–1.15，需核实combine后的有效范围，不能当作已完成多地面验收。v179 PID780690、artifacts/continuation_v179/launch.json，mesh五次细分保留原材质，已执行首动作。详见improvement第120条。

## v178成对对照已完成：同初态下mesh失败、解析plane双向通过

777950及776109均已结束。原mesh精确复现正/负survival0.869565/0.96875；解析plane两向1且单项全过。checkpoint、记录源码hash、所有环境初始位置/姿态/25D观测均完全相同。plane_mesh_comparison.json提供证据。当前首要排查转为地面接触表示/材质绑定差异，勿原样重复加训；具体机制未定、训练地面未改、全目标未验收。反思第119条。

## v178解析plane双向单项通过，原mesh正在同脚本复核

local_plane_direction_grid.json每向64回合、survival1、bank2.61734/0.12652°、所有单项checks通过，无倾倒；整体passed=False为诊断标志，非验收。PID776109已结束。当前PID777950复核原mesh，记录mesh_recheck_launch.json，待核对初始状态和地面差异。未改训练地面、未宣称全速度/最高DR/3Nm通过。详见improvement第118条。

## v178初始化修复后重试：以PID776109为准

旧772907/775037/775702均已退出，无评估JSON，不是策略失败。当前local_plane_spaced_launch.json记录776109；本地解析plane避免远程视觉USD，terrain间距显式等于scene，异常在Kit关闭前打印。结果待定，不重复启动。见improvement第117条，旧段落PID仅为历史。

## v178地面表示诊断正在运行，结果待定

PID772907已核实运行，解析plane对照原mesh，固定4753/seed44/原命令布局、128env、2.4±0.3、DR0。日志与启动artifacts/continuation_v178；报告强制diagnostic、不用于验收。评估新增reset初态和env_origins取证。无正式trainer；不得重复启动或将此当新技能通过。详见improvement第116条。

## v177审计完成：布局影响生存差异，倾角奖励已存在且同状态引导明确

同4753交换命令位置后正/负survival为0.96923/0.90625，原布局0.869565/0.96875；全部布局整体均未通过，不能将原10个百分点差距全归因于正向网络。正向bank偏差仍需处理。真实reward AST在2.4/±0.3固定状态下，正确4.197668°相对直立净reward增加5.728996，竞争直立项仅抵消0.009665；非动态/最高DR验证。侧向力引导、主动降低质量加权COM尚未实现；取消固定姿态偏好不等于已降重心。证据artifacts/continuation_v177/{layout_comparison,bank_reward_audit}.json，反思improvement第115条。

## v176已结束，最终新技能未过、旧四点通过

当前无正式trainer，最终4753正/负survival0.869565/0.96875，正bank3.03524°失败，负向coverage仍失败。旧2.0/2.2双向四点各64回合全部通过、生存1。报告artifacts/continuation_v176/final_decision.json。用户取消的固定高度/腿长偏好保持关闭；不按下方历史进程状态重复启动。

## v176/model4700实测未通过，固定姿态偏好保持关闭

正向survival0.83099、bank3.05369°失败；负向survival0.953125但coverage失败。root均值约0.307188/0.304538m，没有显著降姿证据。报告artifacts/continuation_v176/model4700_large_turns.json。trainer742541继续，worker743504等待最终4753及retention；不恢复用户已取消的0.4m高度/0.34m腿长偏好，不重复启动。

## 用户最终行驶高度要求：取消固定高度/腿长偏好

行驶reward不追求0.4m root高度，也不追求0.34m腿长；command_leg_length_preference_scale默认0已同时写入YAML及env_cfg。最低运行高度/height penalty阈值保持0.15m，姿态、接触及其他能力gates仍须通过。v176原已scale0运行，无需重启。站立课程专用目标不代表行驶目标，勿恢复这两项行驶偏好。

## 当前v176：隔离行驶站立腿长偏好，随机命令继续

trainer742541已实跑至4659（critic预热），从v175final4653续训100updates；command_leg_length_preference_scale=0取消行驶0.34m腿长吸引，0.4m高度吸引仍已取消，接触/姿态/最低高度/gates不变。随机命令0.15、lambda0.99和完整旧图冻结保持。23项测试通过，尚无效果结论。

run_dir=/home/kellen/桌面/11 (1)/11/wheeled-legged_RL/logs/rsl_rl/rmuc_wheel_leg_course/2026-09-25_10-59-15_free_leg_posture_continuous_v176。worker743504等4700及最终4753/实际29entry做新技能和旧技能独立验收；状态artifacts/continuation_v176/evaluation_status.json，勿重复启动。详见improvement.md第109条。

## 当前v175已结束：新技能未过，旧技能四点通过

最终4653正/负生存0.869565/0.9375，正bank3.04857°且两侧coverage失败；2.0/2.2双向旧四点全部checks通过、生存1。报告artifacts/continuation_v175/final_large_turns.json、final_retention_turns.json。trainer273763和worker275512均已结束，当前无正式trainer；勿按下方历史状态重复启动。

取消0.4m行驶高度吸引保持有效，但本轮未出现显著降姿或2.4稳定改善；随机命令实现仍保留，尚未全域最高DR验收。下一步需分离剩余腿长/姿态引导因素，而非原样重复训练。详见improvement.md第108条。

## v175/model4600中途实测未过，未见明显机身降低

2.4正向survival0.85507、bank3.00713°失败；负向survival0.953125但coverage失败。两侧root mean约0.307181/0.304415m，与修改前几乎一致，不能称主动降低重心。证据artifacts/continuation_v175/model4600_posture_comparison.json。trainer273763继续既定100updates，worker275512等待最终4653/entry29与retention；不重复启动，取消0.4m高度吸引保持有效。

## v175预热结束，独立验收已安排

4563结束预热，4574 router更新正常，旧graph/buffers变化0。worker275512绑定trainer273763，等4600与最终4653/实际29entry双向实测，最终再旧四点retention。状态artifacts/continuation_v175/evaluation_status.json；随机命令保持0.15。尚未证明取消高度偏好改善驾驶或实际降低COM，勿重复启动。

## 当前v175：行驶取消0.4m高度吸引，随机命令继续启用

用户明确不要求靠拢0.4m，env.py已移除行驶reward的height_score，保留最低高度/姿态/接触gates与独立腿长项。17项测试通过。v175 PID273763从v174 candidate099/model4553续训100updates、10critic预热、stop29，实际到4560，旧图审计无变化；配置continuous_sampling_fraction0.15，日志实际19/128=0.1484。

启动入口artifacts/continuation_v175/launch.json、日志train.log，勿重复启动。当前随机命令只覆盖已开放速度/转弯包络，尚未全域/动态切换/最高DR通过。旧站立课程高度项与腿长偏好未全删，不混称行驶rootz就是COM。详见improvement.md第105条。

## v174候选lambda0.99组已运行

基线2.4双向失败，2.0/2.2双向retention四点全部通过、生存1。worker251795已启动候选trainer261310，从共同4453/Adam开始；runtime YAML确认lambda0.99，其余实验参数一致，记录中另保留Gym space RNG元数据差异。当前候选早期冻结审计正常，效果尚未评估；以artifacts/continuation_v174/status.json为运行入口，勿重复启动。详见improvement.md第103条。

## v174进度：基线训练及2.4双向评估完成，旧技能评估中

baseline lambda0.95最终4553正/负survival0.842857/0.90625，正bank3.04367°且两侧coverage失败；报告artifacts/continuation_v174/baseline095_large_turns.json。worker251795继续baseline retention，之后顺序启动candidate lambda0.99；以status.json及实际进程为准，勿重复启动。尚无完整A/B结论，不能称时间尺度改动已有效。

## 当前v174：v173已结束失败，执行同起点GAE对照

v173/model4653正/负生存0.869565/0.890625，正bank3.05668°，整体未过；2.0/2.2双向retention四点通过。提前pitch奖励未显示双向改善，已返回v172/model4453及原reward。

v174串行A/B：同4453/Adam、seed47、每组100updates、10critic预热，lambda0.95对0.99，gamma0.99不变；完整旧图冻结、router预算/gates不变。worker251795，当前baseline trainer251796已实跑至4461（warmup中），候选尚未启动；状态artifacts/continuation_v174/status.json。每组结束独立评估新技能及旧技能；不要重复启动。详见improvement.md第101条，全速度/最高DR/硬件部署目标仍未完成。

## v173/model4500独立评估未通过

正向survival0.84286、bank3.00554°且coverage失败；负向survival0.96875，其余steady指标通过但coverage失败。证据artifacts/continuation_v173/model4500_large_turns.json，不四舍五入bank过线，不把负向survival过线当整体通过。trainer1084482继续，worker1085971等待最终4653/实际29entry及旧技能评估，保持reward和gates不变。尚无新reward双向突破证据。

## v173已结束预热，独立评估worker运行

4463结束critic warmup，router恢复更新，旧图审计无变化。runtime_reward_audit.json确认大转弯pitch4°/8°而普通8°/20°，未混用旧配置。worker1085971绑定trainer1084482，等待4500及最终4653或实际29entry，再做旧技能retention；状态artifacts/continuation_v173/evaluation_status.json。尚无新reward能力通过证据，勿重复启动。

## 当前v173：结构实验已收尾，启动早期pitch奖励试验

v172最终4453：2.4±0.3正/负survival0.86957/0.9375，未通过；同checkpoint的2.0/2.2×±0.3四旧锚点各64回合全部checks通过、生存全1。200updates完整旧actor逐位不变，证据artifacts/continuation_v172/final_integrity.json及final_*turns.json。旧技能局部保留有证据，新2.4恢复仍待攻克。

v173 PID1084482已实际到4459，当前critic预热，旧图审计无变化。run_dir=/home/kellen/桌面/11 (1)/11/wheeled-legged_RL/logs/rsl_rl/rmuc_wheel_leg_course/2026-09-24_10-40-53_isolated_router_early_pitch_v173；launch在artifacts/continuation_v173/launch.json，日志train_isolated_router_early_pitch_v173.log。仅改大转弯pitch tracking容差8→4°、scale20→8°，恢复4453的Adam状态；保持归一化state router/参数隔离/预算、旧图冻结、全部gates和物理配置。10critic warmup后只router+critic训练，最多200updates/进29停止。不要重复启动；需最终新技能与旧技能实测，不能宣称新reward已有效。最高DR与最终硬件/全速度目标未完成。

## v172/model4300独立评估未过，继续既定实验

正/负survival0.88235/0.90625，均coverage/survival失败；bank2.99494°/0.66871°通过。参考4253为0.82857/0.96875，尚无双向稳定改善。报告artifacts/continuation_v172/model4300_large_turns.json。实际checkpoint合成输出隔离验证通过，不等于物理技能已过。trainer1067090继续，worker1068788等待最终及retention；reward保持原样，不重复启动。

## v172评估worker已绑定，4275参数隔离检查通过

trainer1067090持续运行；model4275与zero_router初始化对比，仅router40参数变化，旧actor全部state逐位相同。新修正最大约0.002464，尚无能力改善证据。worker1068788已绑定训练身份，等待4300及最终4453/实际29entry，独立评估2.4±0.3，并对最终模型做2.0/2.2双向retention。状态artifacts/continuation_v172/evaluation_status.json；勿重复启动。详见improvement.md第94条。

## 当前v172：独立状态router正式能力实验已启动，reward保持原样

PID1067090已实际训练到4258，run_dir=/home/kellen/桌面/11 (1)/11/wheeled-legged_RL/logs/rsl_rl/rmuc_wheel_leg_course/2026-09-24_10-12-58_isolated_state_router_learning_v172，启动记录artifacts/continuation_v172/launch.json。128env最多200updates，Level28起、进29即停。只训练新router/专家与critic，完整旧actor及std冻结；全局L2系数0但完整性审计保留，归一化状态router、aux0.01、独立方向optimizer隔离生效。4258旧图参数/buffers变化0、新修正有限且在预算内。不要重复启动。

真实PPO接入及Adam恢复已通过32env短程smoke和21项测试，详见improvement.md第93条。尚未验证改善生存/压弯；router仅当前局部命令域，非全速度最终架构完成，新预算也不覆盖旧图原有残差。reward新选项未启，LoRA无；最高DR、最终3Nm/径向气弹簧和部署目标仍待完成。

## v170状态router推理链路已接，PPO接入仍待完成

zero_router_candidate.pt保持4253原图全部状态不变，新增修正零初始化；2048探针保存/恢复逐位一致。非零test-only模型完整JIT/ONNX通过（最大2.384e−7），19项单测通过；8env/100step真实恢复运行通过接口检查，不是技能证书。证据artifacts/continuation_v170。play支持加载router，尚无新正式训练；候选移除旧optimizer状态，必须显式新建并隔离旧图参数。不要直接按旧resume流程启动。新预算约束新增适配，未修复旧图所有累加残差；reward实验未启。

## 最新：正则梯度诊断完成，未支持输出层主导压制（v169）

8updates/32minibatches真实测量anchor/PPO动作均值梯度norm比约4.925e−6至7.811e−6，median7.026e−6；整体cosine近0，冻结weights/routing delta0。证据artifacts/continuation_v169/gradient_summary.json。限制为全命令混合输出空间，不能排除参数组或局部任务冲突。没有删除L2，没有部署诊断模型，状态router/LoRA尚未实施。v169短程诊断已结束，当前无trainer；最终能力仍以v166/model4253失败报告为准。

## 2026-09-24 当前：v166已结束，正向恢复瓶颈未解决

本条覆盖下方旧“运行/等待”快照。v166完成4253，训练与评估worker均结束，当前无正式trainer。最终2.4±0.3独立报告正向survival0.82857/12tilt、coverage失败；负向0.96875全部checks通过。报告artifacts/continuation_v166/model4253_large_turns.json。不原样自动追加训练，不部署失败候选。

v167为评估诊断：关闭高速速度反馈使正向survival降至0（191重复失败回合），负向0.921875；已拒绝采用，正式参数和checkpoint未改。证据artifacts/continuation_v167/feedback_ablation_decision.json。说明外环是当前控制的重要部分，不能凭pitch失稳前的速度反馈关联直接删除；完整根因仍待证实。最高DR、3Nm径向气弹簧与最终全速度/自转/部署目标仍未完成。

## 中间点对照已完成：属于保留能力，不是本轮新增突破

3953前置模型在同2.1/2.3×±0.225、seed46/layout2028四点也全部通过、survival1。因此4000四点通过仅支持此范围能力保留，不能归功于连续采样。证据artifacts/continuation_v166/intermediate_grid_comparison.json。2.4大turn仍未通过，trainer561199继续，最终4253评估计划不变。

## v166/model4000四个中间点通过（范围有限）

2.1/2.3m/s×yaw±0.225，seed46/layout2028，每点64完整回合，全部checks通过、survival1；2.3正向bank2.97147°余量很小。报告artifacts/continuation_v166/model4000_intermediate_grid.json。固定2.4/±0.3仍失败、最高DR仍未过；不能扩张证书范围。3953前置模型同网格对照已启动reference_intermediate_launch.json，尚不归因于连续随机训练。

## v166/model4000独立评估未过（最新）

正向survival0.80282、bank3.05224°失败；负向survival0.9375失败，两侧coverage均未过。报告artifacts/continuation_v166/model4000_large_turns.json，比v165final3953的0.88235/0.953125未改善。base参数逐位不变，13个变化张量仅在允许模块，不能据此声称行为不遗忘。trainer561199继续原定300updates，worker562628等待4253，当前不采用为最终策略。采样器修正只在磁盘，运行进程仍旧包络。

## 连续采样覆盖修正（尚未进入运行中的v166）

发现直行专属课程节点会把相邻转弯采样界清零；已改command_sampling.py，使用独立的已开放转弯节点构造包络。5项测试通过，离线同100000点straight比例从79.974%降至29.899%，未扩展超过当前2.4m/s转弯范围。运行v166仍使用启动时旧采样器，后续续训才生效，详见improvement.md第78条。不能把修复后的离线覆盖当已训练能力。

## v166监控与评估入口

连续命令审计已捕获非课程点，early_sampling_audit.json仅证明采样发生，不认证全域能力。绑定评估worker PID562628等待实际model4000与训练结束后的model4253，状态artifacts/continuation_v166/milestone_evaluation_status.json；不重复启动。当前采样最低速度来自课程约1m/s，0速自转尚未覆盖。

## 当前v166已启连续随机命令，v165最终仅负向通过

v165/model3953最终双向实测：正向survival0.88235且bank3.03654°失败，负向survival0.953125全部checks通过，报告artifacts/continuation_v165/final3953_large_turns.json。v165及最终评估均已结束。

v166从3953续训，PID561199已实跑至3961；连续随机命令cohort实际19/128=0.1484，rehearsal0.60，固定active/gates不混入随机cohort。保持原L2 anchor、LR、冻结图与物理配置，最多300updates；3963已结束critic warmup，3966 actor恢复更新，冻结weight/routing delta0。run_dir=/home/kellen/桌面/11 (1)/11/wheeled-legged_RL/logs/rsl_rl/rmuc_wheel_leg_course/2026-09-23_19-48-38_continuous_commands_local_turn_v166；启动记录artifacts/continuation_v166/launch.json，日志train_continuous_commands_local_turn_v166.log。不要重复启动，不将低DR/当前5.5Nm阶段当最终验收。

## 最新验证：v165/model3800未通过，训练继续既定200轮

独立128env、2.4±0.3、DR0报告artifacts/continuation_v165/model3800_large_turns.json：正向survival0.81429、负向0.921875，均coverage/survival失败，不能称突破。trainer PID544451仍存活。最终评估worker PID551989已运行，绑定训练进程身份并等待结束后评估model3953；状态artifacts/continuation_v165/final_evaluation_status.json，不重复启动，不自动续训。连续随机命令尚未启于此运行，后续方案必须依据最终报告。

## 连续随机命令采样已接入（默认关闭）

用户希望泛化线速度/角速度。新增env.command_continuous_sampling_fraction，建议后续续训0.15；独立reset采样cohort，当前开放速度域内连续采样vx、受保守局部yaw包络约束，episode内保持目标。固定active与rehearsal gates排除该cohort，不将泛化样本冒充课程证书。33项测试和16env/2updates真实smoke通过，实际观测到非课程命令；尚未完成泛化训练。当前v165已运行进程未启新选项，不重复启动。详见improvement.md第72条与artifacts/continuation_v165/continuous_sampling_smoke_evidence.json。

## 当前实验续训v165（未验收，2026-09-23）

v165从local_2p4_positive_candidate.pt实验初始化启动Level28，最多200updates（至3953附近），run_dir=logs/rsl_rl/rmuc_wheel_leg_course/2026-09-23_19-23-45_local_head15_posture_learning_v165，启动记录artifacts/continuation_v165/launch.json，PID544451已在3760实际运行。候选保留全部网络参数，仅2.4正向大yaw附近stage15输出乘0.3和active差分−0.08/+0.08；54个其他命令锚点probe逐位一致、21项测试及JIT/ONNX通过，但128env正向survival0.84286仍失败，绝非认证策略。

当前stage LR提高到1e−4，L2 anchor固定新初始化；保留所有冻结审计/rehearsal/retention/gates，10critic warmup。3763已结束warmup，3770的actor动作已更新（对anchor最大差0.002198），冻结权重及routing变化仍为0。正式交付、最高DR、3Nm径向气弹簧与自转目标均未完成；旧文档无trainer状态由本条更新。日志train_local_head15_posture_learning_v165.log；不要重复启动。

## 2026-09-23 v163已结束，Level28最终复测未通过

v163完成3753轮并保存最终checkpoint，当前无正式训练。3400正/负生存0.81944/0.96875；最终3753同seed/grid为0.80556/0.89063，coverage失败，虽steady tracking/bank通过仍不能验收。证据artifacts/continuation_v164/final3753_large_turns.json。关闭Level28 head的对照已失败：正向survival0.74026、负向0.9375，未采用。共同轮偏置小样本+0.02/+0.04双向通过，但+0.02在seed44双向各64replicas复测时正向0.89706且coverage失败、负向0.96875；不能宣布解决。+0.04同规模诊断也已结束：正向survival0.88235、8次tilt、coverage失败，负向survival1通过。证据artifacts/continuation_v164/common004_confirm64.json；两个常数偏置均不采用。所有外部偏置诊断不能替代最终完整策略验收，尚无正式续训。
## 最新动作归因与诊断（v164）

原3753失败轨迹已补齐25D动作输入与外环修正，12个失败env/36状态完整策略重放误差≤4.768e-7。部分状态base轮输出已达1.25–1.45，但这不是减小轮推力的充分依据：base轮预限幅0.9诊断使正向survival降至0.52128、负向0.875，拒绝采用。报告artifacts/continuation_v164/failure_action_attribution.json、base_wheel_clip090.json；仍无已验证修复或正式trainer。评估器新诊断开关默认关闭且禁止输出整体通过证书，正式策略权重未变。

## v164旧修复head外推的生存/压弯权衡

stage15为2p2_large_repair，speed≥2.2时gate一直为1；在2.4正向失败态给双hip约+0.21。完整head消融诊断（disable_stage15.json）使正/负survival升至0.96923/0.96875，但正向bank4.17185°失败、负向yaw0.05163失败且coverage不够，故未采用。关闭该head并加右active−0.08/左+0.08的2.4正向64replicas组合诊断已完成：survival1、coverage及跟踪通过，但bank3.88314°失败，未采用。报告artifacts/continuation_v164/disable15_active_difference_positive64.json。不能将此诊断扩写为删除旧技能或全速度通过。正式trainer仍未恢复。

## 后期自转目标更新（2026-09-23）

用户要求先完成全速度行走/压弯，再将原地小陀螺自转逐步提高到双向10rad/s；0–3rad/s为中间能力，不再是最终自转上限。自转奖励按目标方向比较：目标至目标+0.5rad/s允许超速带不罚yaw，转慢强罚；零目标仍要求停转，不允许以该宽限带持续自转。姿态/生存/位置漂移/3Nm轮端持续上限及最高DR验收仍保留。

已有默认关闭的spin_asymmetric_reward_enabled实现此非对称奖励；欠速0.5rad/s默认额外扣10（有界），不是已验证最优权重。当前v163行驶训练未启用，不改变原行驶gate。后期旋转还需单独课程、足够yaw观测范围（±10换向至少覆盖20.5误差）、启停换向及相应验收指标。
> 原地±3rad/s基础支持已加入兼容选项：command_yaw_observation_error_clip可单独设6，sensor/延迟同步；command_yaw_overspeed_relative_to_target可避免合法高速旋转被旧1rad/s规则惩罚。默认保留旧行为，v163未启用；9项相关测试通过，不代表原地旋转已学会。详见improvement.md第54条。
## 用户补充最终能力：原地双向旋转0–3rad/s与广泛速度下转弯

原地vx=0时yaw必须可从0连续调到±3rad/s，包含启停/换向且限制位置漂移；行进状态须适应大部分速度下的转弯/压弯，需明确并实测速度—yaw能力边界。必须由同一最终策略在最高DR及最终动力配置下验收，不能用原±0.15/±0.30课程替代。当前尚无该能力证书。
> 用户确认采用低DR逐步推进最高DR（2026-09-23）。当前v163实际配置start29/full46、strength_override=-1：Level29约5.26%起步，阶段transition内升至10.53%，后续逐级增加，到46达到100%；环境reset才重采样。阶段进阶由技能/retention gates约束，但阶段内部强度爬升按transition时间，不是完整的性能自适应强度控制。后续若出现持续退化，应在有证据的强度区间巩固而非跳过gate；最终最高DR验收不变。
> 最新最高DR实测已完成且未通过：entry3051在2.4m/s、yaw±0.15，同seed nominal通过而strength1下survival0.5676/0.4337，yaw0.0701/0.0749、stable0.1800/0.0388均不达标。报告确认观测与指令延迟各实际采到1/2/3周期，并记录新IMU/编码器配置与source hash；见artifacts/continuation_v163/level27_skill_full_latest_dr.json。仍是5.5Nm迁移前动力、气弹簧关闭，不是最终硬件验收。v163继续Level28 nominal训练，最高DR通过仍为强制交付要求。
## 2026-09-23 当前Level28；Level27新技能独立通过

v163在3051保存model_level_28_entry_3051.pt，进入2.4m/s、±0.30大转弯；对应2p4_large head可训练，其他已认证head冻结。进阶日志retention speed max0.143/yaw max0.047/stable min0.666/survival min0.958。

该entry checkpoint的2.4m/s、±0.15独立复测各64回合全部通过：survival1，linear0.02159/0.02394、yaw0.02185/0.01991、bank2.12960°/0.47836°、stable0.87202/0.85708。证据artifacts/continuation_v163/level27_skill_2p4_small_turns.json。3064权重/冻结head/路由buffer审计全零变化。

训练不停；已安排首次Level29 entry或3400 checkpoint的2.4×±0.30独立验收，worker325742，level28_skill_status.json。最高DR新传感器延迟尚未训练完成，3Nm/径向气弹簧与部署迁移仍待完成。
> Level27 entry2866固定checkpoint已在seed43与seed45/layout2027两轮独立nominal网格通过2.0/2.2×±0.3各64回合检查；seed45的2.2负向survival0.96875，其余三桶1。证据汇总artifacts/continuation_v163/level26_repeated_nominal_evidence.json，不代表fullDR/全速度/部署通过。当前继续Level27，已安排首次Level28 entry或3200检查点独立验证2.4×±0.15（level27_skill_status.json）。
## Level27 entry独立四桶验收通过（seed43，nominal）

model_level_27_entry_2866.pt固定策略在2.0/2.2m/s、yaw±0.30四桶均完成64回合并全部checks通过，survival均1。2.0正向bank2.0033°、yaw0.04924、stable0.47053，余量小；负向yaw0.04305/stable0.53595。2.2正负yaw0.02553/0.02722、stable0.87291/0.79693。证据artifacts/continuation_v163/milestone_turns_nominal_seed43.json。

同checkpoint seed45/layout2027重复验证已启动，尚无结论；不把单seed当充分鲁棒。v163继续Level27，未重启。最高DR、全速度、3Nm径向气弹簧迁移/部署目标仍未完成。
## 2026-09-23 已通过Level26，当前Level27

v163在iteration2866实际进阶到Level27：bank_2p4_turn_0p15，保存model_level_27_entry_2866.pt。进阶日志active双向yaw max0.035、stable min0.708、bank max1.936°、survival min0.971；retention speed max0.144、yaw max0.046、stable min0.595、survival min0.985。进阶后日志计数归零是新阶段重置，不是丢失checkpoint或训练回退。

自动worker已针对该entry checkpoint启动2.0/2.2双向独立64回合复测，尚未判通过；训练PID314868继续Level27，不重启。关键base/expert/修正buffer共32个张量与初始化逐位一致，见artifacts/continuation_v163/level27_entry_integrity.json；完整运行审计继续覆盖其他冻结head。当前仍未最高DR/3Nm/气弹簧/部署验收。
> v163校准在2763结束，actor恢复更新；完整组合策略导出数值一致性验证通过。首次Level27 entry或model3000的独立双向验证worker已启动等待（artifacts/continuation_v163/milestone_validation_status.json）。早期2.0正向bank改善不等于完整进阶，仍以真实checkpoint与完整回合证据为准。
## 2026-09-23 当前v163：局部双向修正初始化后续训

从v161最终2753及v162验证过的修正方向构造both_direction_seed.pt，模型参数未变；正向2.0active差分-.12/+.12、负向wheel差分+.02/-.02只在speed(1.75,2.2)、对应大yaw平滑生效。负向64回合通过；正向单seed64通过但seed45网格2.0/.3 yaw/stable失败，所以这是学习初始化而非稳健证书。v163已实际运行，启动与run目录见artifacts/continuation_v163/launch.json。L2参考显式固定到该初始化，10critic校准后续训，其他LR/gate/physics不变；最新1–3周期sensor/action DR配置按课程启用（当前26仍DR0），气弹簧仍关闭、5.5Nm为迁移前阶段。
> 2026-09-23 用户更新DR契约：观测上行与控制指令下行分别随机1–3控制周期（当前8–24ms），取代先前5–25ms方案。最高DR全部环境分别reset采样并固定回合内延迟，低强度按strength提高启用概率。两条链路可叠加为2–6步总滞后；不是总链路最多3步。新增按物理单位IMU零偏/随机游走、加速度计对gravity估计的等效噪声、编码器角度/速度噪声。25-D/6-D不变。既有旧DR通过记录不认证新传感器/通信契约；新契约仅完成接入测试，尚未训练通过。
> 2026-09-23 当前无正式训练进程：v161已完成2753，最终复测2.0正向bank/负向yaw-stable仍失败。定向差分0.12经seed44/64回合诊断逐桶通过，已构造所有权重不变、仅2.0正向大yaw附近平滑固定修正的候选；完整候选物理验证进行中。详情improvement.md第32–36条、artifacts/continuation_v162。不是全目标完成或新训练已启动。
## 当前运行：v161，定向加强2.0转向奖励

v160已保存model_requested_stop_1753.pt退出；v161从该checkpoint接续，唯一奖励资格变化为turn_yaw_tracking_start_level26→18（速度阈值2.2→2.0），相同L2锚点、nominal直行保护、rehearsal0.60、学习率、gate与物理。10轮critic校准已结束，actor正在更新。启动/目录见artifacts/continuation_v161/launch.json。自动复测等待首次Level27 entry或model2050，状态milestone_validation_status.json，尚未出新证书。

v160/model1700四桶各64回合全存活：2.2双向全部通过；2.0正向bank3.4466°失败，负向yaw0.05926/stable0.31637失败。已有奖励代码counterfactual显示加强起点移至2.0增强该速度yaw/bank纠偏信号而抽查1.5/2.2/2.5/2.9不变，并非已证明训练成功。
> 气弹簧口径更正（用户要求）：应按小腿径向力及其作用点/Jacobian建模，不能将固定active-link -10.775Nm直接等同实机气弹簧。此前探针只证明该力矩通路实现正确，不证明机械等效。该近似现限制为play诊断，禁止用于训练；v161当前辅助幅度0不受影响。实际等效方向、推力与安装点待确认。
> 气弹簧接入进展：RMUC已新增默认关闭的独立机械力矩对（gas_spring_assist_torque_nm=0保持旧动力），45测试及短时PhysX/composer实测通过；尚未迁移训练或证明稳定性提升。v160继续原动力，model1700独立双向复测已自动启动。详见improvement.md第22条。
> 2026-09-22 电机/气弹簧参考已核对：用户确认RM3508持续最大3Nm是减速器之后的轮端扭矩。参考/home/kellen/wheel_leg_mjrl-lqr/wheeled_infantry.xml把±3再乘gear15.7647，MuJoCo实际前向得到±47.294Nm，不能照搬。保留gear时输入限幅折算±0.1902985Nm，内存修正模型已验证轮端±3Nm。参考气弹簧在lqr_deploy.py外加左右active_link各-10.775Nm，非XML弹簧；当前RMUC缺少对应辅助力路径，必须纳入后续动力迁移和最高DR验收。详见docs/rmuc_motor_spring_reference.md；原XML/USD/运行物理参数未修改。
> 硬件参数最终澄清：RM3508 的 **3 Nm 为持续最大扭矩**，表示持续输出上限，不表示任何工况均输出3 Nm。最终策略不得依赖超过此值的持续输出；短时峰值能力未提供，不自行假定。
> 2026-09-22 用户确认：轮电机为RM3508，3 Nm是持续输出扭矩，不是短时峰值。最终actuator/deployment按这一用户确认参数建立验收；不能再称其仅为USD元数据。当前5.5 Nm仿真训练仍为迁移前阶段。real_hardware配置0.8比例是另一个软件限幅，需在最终控制契约中显式核对；不得把用户确认3 Nm误当成该限幅已自动修改。
> v160固定里程碑独立验证已安排：首次Level27 entry或model1700，验证2.0/2.2双向各至少64完整回合。worker状态见artifacts/continuation_v160/milestone_validation_status.json；当前waiting_checkpoint，不能当作通过。保持训练连续，不改基线。
## 最终启用策略的硬条件（2026-09-22 用户明确）

最终control/controller及部署启用的必须是**同一份完整专家混合残差组合策略**，对应唯一checkpoint hash与导出制品；必须经过最高DR（strength=1.0）训练/验证并具备足够鲁棒性，同时覆盖全速度直行、双向压弯行进和转弯。不能只认证最高速度一个点、只验一个转向、只用nominal/早期DR结果，也不能拼接不同checkpoint的通过记录作为最终证书。

稳定掌握需在同一最终策略上重复独立seed、连续速度/yaw变化和持续行驶，逐点检查tracking、bank、survival、接触/姿态与用户认可的高度口径；完成后续actuator迁移后需在最终部署控制契约和最终动力参数下重新验证最高DR。全程保持25-D/6-D、closed-loop USD与frozen skill graph/rehearsal/retention。导出需完整组合图及数值一致性校验，control默认策略只能在上述证据齐全后切换。

若随机化范围与动力能力存在物理矛盾，必须明确诊断、解决并验证；不得静默降低DR范围、放宽性能gate或掩盖失败来交付。当前候选尚未满足这些条件，不能作为最终启用策略。
> 2026-09-22 稳定掌握要求更新：同一策略须通过重复/双向/fullDR/持续命令及部署验证。固定v158候选fullDR seed45已实测未通过（69完成回合、33tilt、survival52.17%、speed error0.3356），不是高度拒绝，也不能单凭动力估算归因。v160保持连续训练，未变基线或削减DR。详见improvement.md第14条与artifacts/continuation_v160/full_dr_straight64_seed45.json。
## 2026-09-22 当前v160：补齐retention bank，原checkpoint接续

v159完成50轮critic warmup并保存model_requested_stop_1409.pt正常退出。真实checkpoint比较确认actor恢复更新后，2.9straight四个受保护head切片全部逐位不变。v160从1409继续actor/critic/optimizer，不重复warmup，固定candidate L2锚点、奖励、学习率、物理参数不变；加载retention双向bank EMA/3°gate、进阶与actuator推进检查、bank deficit方向采样。实际迭代与新bank日志已出现，启动记录artifacts/continuation_v160/launch.json。初期出现一次2.2正向tilt终止（约85.5°），不能把之前deterministic64回合通过当成随机探索100%安全，继续以完整计数判定，不因单次事件再改基线。
## 2026-09-22 当前v159：恢复直行路由并修复retention

v157已STOP保存model_requested_stop_1643.pt退出。v158候选2.9直行nominal/DR2/19通过；2.2双向nominal各64回合通过；2.0正向bank和负向yaw/stable未通过。v159从已测candidate(model1350来源)续训，启用nominal2.9straight局部head/Adam切片保护，rehearsal0.60，用户高度口径0.15，固定L2参考显式改为候选；50critic warmup，保持原LR与全部其他gate。已看到真实1353迭代，PID216377，启动与run路径见artifacts/continuation_v159/launch.json。尚未通过Level26/全目标；详细证据及反思见improvement.md第10/11条。
> 2026-09-22 用户新增要求：每次瓶颈、反思、攻克方法及证据更新到 [improvement.md](improvement.md)。已补录本轮8类问题，区分代码修复、局部行为通过与未闭环；后续持续追加，不覆盖历史证据。
## 2026-09-22 直行候选DR2/19通过，转向复测进行中

候选exact_base_2p9_candidate.pt在seed44、DR2/19、64环境/64完整回合通过2.9直行：survival1.0、linear_error0.146764、stable0.505539、root_min0.291888。证据为artifacts/continuation_v158/exact_base_straight64_dr_seed44.json。不代表fullDR通过。观察期间曾误把1000步进度当作结束；重复启动被输出文件防覆盖检查拒绝，JSON保持完整，但原.log被重复命令重定向覆盖成拒绝信息，日志不再代表原运行。

官方v157 iteration1556仍Level26：turn yaw0.0693、stable0.4357；此前1544日志还明确显示retention未通过（Level23 speed error0.2746/stable0，另有2.0负向yaw/stable失败）。所以此前只列当前stage yaw/stable为阻塞不完整。已启动候选2.0/2.2、yaw±0.30、seed43、nominal、每bucket8环境/12000步、64回合要求的独立验证，输出exact_base_turns_nominal_seed43.json。官方训练不变。
## 2026-09-22 2.9直行恢复候选通过单seed nominal复测

`artifacts/continuation_v158/exact_base_2p9_candidate.pt`恢复2.9直行精确base路由，332个受保护tensor不变，44个其他锚点输出不变（过渡区仍改变）。seed43/64回合 nominal：survival1.0、linear_error0.146924、stable0.604515、root_min0.294759，全部checks通过；与v130同条件结果一致，不依赖root门槛降至0.15。完整JIT/ONNX导出通过，24项相关测试通过。详见`artifacts/continuation_v158/exact_base_straight64_nominal_height015.json`和`control_contract_audit.md`。

官方v157保持运行且仍在Level26，未采用此候选。seed44/DR2/19复测已启动（输出`exact_base_straight64_dr_seed44.json`），完成与否以文件和进程为准。候选尚缺过渡区、跨速度、双向、fullDR和部署验收。
## 2026-09-22 用户更新：允许低姿态行驶

撤销行驶 root z >0.20 m 的硬要求，暂用 >=0.15 m，保持速度、yaw、bank、接触和生存 gate。奖励 minimum height 从0.36改至0.15 m；站立任务的目标高度和行驶中的软姿态偏好未删除。0.15 m 并非已验证的导轮触地高度，需实测姿态/接触；不追溯改写旧报告。v157 已启动进程仍使用旧0.20 m配置，新配置从下一次启动/复测生效。


> **最新运行 v157（2026-09-22）**：v155已保存1152轮停止。发现critic的绝对value clipping复用0.1 actor比例阈值，与数百量级回报误差失配；源码梯度检查与留出环境校准支持关闭critic clipping。v157从1153开始，50轮多任务critic-only校准后恢复残差学习；actor PPO clipping仍0.1，gate不变。详见[停滞审查末尾](docs/rmuc_stall_audit_v154.md)，启动记录`artifacts/continuation_v157/launch.json`。下文v154/v155状态均为历史快照。

> **最新运行 v155（2026-09-22 19:39）**：在奖励一致性修复基础上，将reward cap从10修正为20，避免精细yaw跟踪奖励被截断；验收阈值未变。v154已保存765轮，v155从766轮续训。详见[停滞审查补充](docs/rmuc_stall_audit_v154.md)。下面v154状态是历史快照。

> **2026-09-22 停滞审查更新**：确认 active/rehearsal 奖励不一致，并用同物理状态 A/B 验证修复。v151 已保存624轮后停止；v154启用按command一致的奖励，从625轮续训，经gate连续推进至Level31，停止于Level32 entry。此前“到Level27就停止”的运行配置已纠正。详见 [停滞审查](docs/rmuc_stall_audit_v154.md)；是否恢复进阶仍须以实际日志与checkpoint证明，不能视为项目完成。

> 更新时间：2026-09-21（Asia/Shanghai）  
> 用途：新对话接管时首先阅读。这是当前工程状态、已验证结论、失败经验和后续路线的索引，不是最终实验报告。

> **本轮接续状态已更新**：v139已停止。最新执行记录见 [docs/rmuc_continuation_progress.md](docs/rmuc_continuation_progress.md)，实现审阅见 [docs/rmuc_continuation_review.md](docs/rmuc_continuation_review.md)。下文§11的v139运行状态属于历史快照；不要直接执行其中的旧PID/续训建议。当前主线先做一致控制接口下的严格再认证，再推进全速度与DR。新增代码、诊断证据和运行日志以执行记录与实际进程为准。

## 0. 新对话必须先做的事

1. 工作目录：`/home/kellen/桌面/11 (1)/11/wheeled-legged_RL`。
2. 先检查运行状态：

   ```bash
   pgrep -af 'scripts/rsl_rl/train.py'
   tail -n 450 train_level29_progressive_flat_dr_domain_bank_v139.log
   ```

3. 先读当前 source of truth，不要根据旧对话自行猜测：

   - `configs/rmuc_wheel_leg.yaml`
   - `source/agent_tasks/agent_tasks/direct/rmuc_wheel_leg/env.py`
   - `source/agent_tasks/agent_tasks/direct/rmuc_wheel_leg/env_cfg.py`
   - `scripts/rsl_rl/rmuc_command_adapter.py`
   - `scripts/rsl_rl/train.py`
   - `scripts/evaluate_rmuc_stage.py`
   - `docs/rmuc_curriculum.md`
   - `docs/rmuc_wheel_leg_asset.md`

4. 工作树很脏，大量文件是用户和先前工作的成果。不要 reset/checkout/revert 不相关修改。
5. 训练使用 RTX 4080：`CUDA_VISIBLE_DEVICES=0`。
6. 本机应直接使用：

   ```bash
   /home/kellen/anaconda3/envs/issaclab/bin/python
   ```

   不要在未激活 Conda 环境时盲用 `/home/kellen/桌面/IsaacLab-3.0/isaaclab.sh -p`；它曾选到 `/usr/bin/python3.12` 并缺少 `lazy_loader`。

## 1. 最终目标

最终不是“进入某个 level”，而是得到一个可控、不遗忘、可 sim2real 的单策略：

- 从静止起步，正向、反向、加速、减速。
- `1.0` / `1.5` / `2.0` / `2.5` / `2.9` / `3.0` m/s 附近稳定直行。
- `3.0 m/s` 以下大部分速度点都能跟踪 `yaw=0.15/0.30 rad/s`，左右两个方向均需通过。
- 最后覆盖 `3.0/3.2 m/s` 直行和高速小/大 yaw 压弯。
- 行驶阶段 root z 工作高度暂定不低于 `0.15 m`（2026-09-22 用户允许低姿态；导轮触地边界待物理验证），生存率和 speed tracking 优先于能耗、对称和观感。
- 转弯允许 roll/腿长差，不能用强对称惩罚压制压弯。
- 地形始终保持 flat；鲁棒性通过渐进 domain randomization 训练，不使用 rough terrain 替代本轮目标。
- 完成后导出可部署 policy/ONNX，并接入 `control.py` 的 W/S/A/D 渐进速度/yaw 控制器。

## 2. 不可再改错的硬约束

### 2.1 USD、根坐标和 reset

- 当前 runtime 使用仓库内 closed-loop USD，不改 mesh、joint axis、limit 和闭链结构。
- 当前配置的 `initial_pos_z_m=0.40`、`asset_frame_rpy_deg=[90,0,0]` 是这个 composed USD 的实际约定。
- USD 的 world 是 Z-up，但 authored rigid-body frame 自带 `+90 deg X`；`env.py::_state()` 用 `(-source_z, source_x, source_y)` 映射到 policy canonical forward/lateral/up。
- 早期“去掉 90° X 旋转”的推断已不适用于当前 composed USD。当前训练 root z 约 `0.307 m`、survival `1.0`，不得再未做 asset probe 就删除这个旋转或把 root z 改成 `0.06 m`。
- `0.06 m` 是轮心高度量级，不是 chassis root z。

### 2.2 机器人质量

- Isaac articulation audit：39 个 rigid bodies。
- 总质量：`18.970039 kg`，可简称 `19 kg`。
- root/chassis 单体质量：`16.227064 kg`。
- 不要再把 `16.2 kg` 说成整车质量。

### 2.3 执行器与 sim2real

- USD authored wheel drive cap 是 `3.0 Nm`。
- 当前 checkpoint lineage 的 Isaac explicit `IdealPDActuator` 使用：
  - `wheel_torque_scale_nm=5.5`
  - `effort_limit_nm=5.5`
  - `wheel damping=0.1`
- 重要：explicit actuator 会使用 task config 的 `5.5 Nm`，不能再说“USD 3 Nm 会自动把 Isaac 5.5 Nm 截断”。
- 最终 sim2real 版必须迁移到 `3.0 Nm`，并围绕实测 closed-loop damping 训练。不允许把 5.5 Nm 伪装成真机约束。
- 但不能从 5.5/0.1 突然跳到 3.0/0.01。实测过静态切换会让 2.9 m/s 策略崩溃。必须在名义 locomotion 技能完成后用独立 actuator bridge + domain randomization 渐进迁移。

### 2.4 Policy contract

- 当前和历史有效 checkpoint 都是 `25-D observation / 6-D action`。
- 不要再随意改成 26/28 维，否则旧 actor first layer、teacher dataset、controller 和 checkpoint 都不兼容。
- 6 个 action 语义顺序：
  `right_active, right_hip, right_wheel, left_active, left_hip, left_wheel`。
- 不要在部署端再重排 action。数字 joint index 应在 runtime 按名字查找，不要硬编码依赖。

### 2.5 闭链/并联 USD 资产的正确用法

这个机器人的腿是并联闭链，不能当成普通开链 URDF 随意简化。

- 训练 runtime 必须使用：
  `source/agent_world/agent_world/assets/usd_files/rmuc_wheel_leg/rmuc_robot_closed_loop.usd`。
- `rmuc_robot_only.usd` 是 open-chain 诊断 fallback，不是正式 PPO 训练资产。
- 当前 `RMUC_WHEEL_LEG_CFG` 明确指向 closed-loop USD，并用 `copy_from_source=True` 克隆到每个 env。
- USD 必须只有一个 `ArticulationRootAPI`，放在真正的 physical `robot` prim，不能放在 zero-mass world/helper prim 上。
- 闭链由 4 个 `PhysicsSphericalJoint` 补回：
  - `left_node_link <-> left_upper_leg`
  - `left_parallel_link <-> left_lower_leg`
  - `right_node_link <-> right_upper_leg`
  - `right_parallel_link <-> right_lower_leg`
- 这 4 个 constraint 必须 `enabled=True`、`collisionEnabled=False`、`excludeFromArticulation=True`。它们关闭机械环，但不能再被算成可控 articulation DOF。
- 只有 6 个 authored motor joints 接受 policy command：每侧 active、hip、wheel。
- 8 个 linkage joints 是 passive：在 Isaac command layer 用 `stiffness=0, damping=0.01` 的 passive group 防止 importer 遗留高 drive 把闭链锁死，但不给 policy action。
- guide-wheel joints 是 passive rolling：`stiffness=0, damping=0, effort_limit=0`。
- 不要为了套 V14 env 而伪造 `front/rear/spring` 关节别名，否则会产生重复/模糊物理对象。V14-style task ID 只是语义 alias，实际应使用 RMUC adapter。
- 不要直接手改 96 MB USD 来调 reward/control。如确需从 open chain 重建，只用 `scripts/prepare_rmuc_closed_loop_usd.py`，它只移动 articulation-root schema 并增加 4 个 loop constraints，不改 mesh/mass/inertia/drive/limit。
- 资产改动后必须按顺序做：

  ```bash
  /home/kellen/anaconda3/envs/issaclab/bin/python scripts/validate_rmuc_usd.py \
    --usd source/agent_world/agent_world/assets/usd_files/rmuc_wheel_leg/rmuc_robot_closed_loop.usd \
    --allow-authored-zero-mass

  /home/kellen/anaconda3/envs/issaclab/bin/python scripts/audit_rmuc_articulation.py
  ```

  然后再做 single-env zero-action/constant-wheel probe，确认 loop residual、wheel contact、mass、forward sign 和 reset 有限。

### 2.6 `3 Nm` 与 `0.1 damping` 必须分层理解

这两个值不是同一层的事，也不是二选一：

1. **USD authored `maxForce ~= 3.0 Nm`**：这是资产/硬件来源的轮驱动上限，最终 sim2real 必须尊重它。
2. **Isaac explicit wheel actuator `effort_limit=5.5 Nm`**：这是现有 checkpoint lineage 实际训练时的 clip。`IdealPDActuator` 使用 config 中的值，所以当前 policy 确实可以依赖超过 3 Nm 的 simulation effort。
3. **Wheel damping `0.1`**：这是现有 checkpoint-compatible explicit actuator D 项。轮子 `stiffness=0`，当只发 effort target 时，它会产生类似 `tau_applied = tau_ff - 0.1*qdot` 的高速反向力矩。
4. **USD authored drive damping `0.25`**：当前 explicit actuator 配置会清掉/取代其驱动语义，不能将 `0.25+0.1` 简单相加当成实际 damping。
5. **XML passive wheel-joint damping `0.01`**：这只描述机械/建模层的被动阻尼，不等于真机电机+驱动器+电流/速度环的 closed-loop effective damping。

因此正确路线是：

- 当前 Level29-46 技能训练：保持 lineage-compatible center `5.5 Nm / 0.1`，只做小幅对称 DR，避免再次冲掉 locomotion 技能。
- 名义技能认证后：用 actuator bridge 同步、平滑地将 torque scale 和 explicit effort limit 迁到 `3.0 Nm`。
- damping 不应盲目直达 `0.01`；先用真机空载/带载衰减、速度-电流-力矩数据识别 effective damping 中心，再退火到该中心。
- actuator bridge 过程将 frozen nominal skill graph 作为锚点，训可控 residual；只有 active tracking、retention tracking 和 survival 稳定时才允许 bridge fraction 上升。
- 最终交付的 policy 必须在 `3.0 Nm` center 上通过全速度/yaw grid，而不是只在 5.5 Nm 上训完再宣称可上真机。

## 3. 当前 observation、command 与方向约定

### 3.1 25-D observation 组成

`env.py::_get_observations()` 当前输出：

- body angular velocity `3`，乘 `0.25`。
- projected gravity `3`。
- command observation `3`。当前 mode 是 `tracking_error_with_target`：
  `[target_vx - actual_heading_vx, target_vx, target_yaw - actual_yaw_rate]`。
- four controlled leg position offsets `4`。
- six controlled joint velocities `6`，乘 `0.05`。
- previous executed policy actions `6`。

总计 `3+3+3+4+6+6=25`。线速度闭环通过 speed error + absolute target 实现，不需要为了看到 vx 把 observation 硬扩成 28 维。

### 3.2 Forward/yaw 定义

- reward/gate 的 actual vx 使用 world velocity 在车体水平 heading 上的投影，不是容易受 pitch/roll 垂直耦合干扰的 raw body x。
- Probe 证明当前 USD 正轮力矩使 canonical forward 为负，所以 config 明确使用 `wheel_action_sign=[-1,-1]`。
- 左右轮直行共模、yaw 差模的符号已在 env residual 中处理，controller 不应再额外翻轮子 action。
- A/D 应改 `command yaw`，不是直接混控两个 wheel action。

## 4. 这段长对话已经找到的主要根因

### 4.1 早期“存活约 30 steps”

- 历史上确实出现过 reset/坐标/command/reward 不一致、初始姿态快速失稳的运行。
- 但当前 composed USD + reset 已经长时稳定，root z 约 `0.30-0.34 m`，episode 可走满，不得再直接把新问题归因为 `+90 deg X`。
- 如今后再出现 20-30 steps，必须先看 termination reason、root z、tilt、joint limit 和 reset probe，不能套用旧结论。

### 4.2 早期 command 总是低速/反向

已确认过多个独立问题：

- wheel action sign 与 canonical forward 反向。
- 低速 target 搭配过宽 velocity tolerance，静止也能拿到近似满分。
- policy 只看 error 时，不同 target 的同样 error 无法区分，导致混合速度策略非 Markov。因此保持 25 维，但 command slots 改成 error + absolute target。
- `torch.all(128 envs stable for 300 steps)` 的旧门控几乎不可达。现在改为 ready/stable 比例、完整 episode survival、完整 episode tracking 和左右 yaw 独立 gate。
- 早期 aggregate yaw/speed 会用一个好方向掩盖另一个失败方向。现在 positive/negative yaw 必须独立通过。
- 早期 survival 只看 episode age，会把“活到门槛后立即摔倒”误认为成功。现在 unsafe termination 即使在 survival step 后发生也算失败。

### 4.3 高速长期卡在 1.1-1.8 m/s

根因不是一个 reward 权重，而是组合问题：

- speed tracking kernel 过窄时，远离 target 几乎没有梯度；过宽时，低速又成为局部最优。
- energy/action/posture 成本与 speed reward 不对称，策略会选择低速保守解。
- action std 过大会破坏已学平衡，过小或全冻结又不能探索新速度。现在 leg std 小并冻结，wheel std 保留最低探索。
- 32-step PPO rollout 对 inverted-pendulum 的 pitch-then-drive 时序责任分配太短。高速训练改用 `128 steps/env`。
- 训练中的随机 episode phase、加速瞬态、rehearsal cohort 会拉低全局 `mean actual_vx`。必须看 `command_active_actual_vx`、分方向 error 和 completed-episode metrics。
- 单纯“速度差 >0.25 扣15分”不能解决访问不到目标状态的问题，还会让 critic 更难拟合。现在使用有界 tracking/progress/acquisition、filtered speed error、severe underspeed、survival 和 pitch/yaw shaping 组合。

### 4.4 damping/扭矩与物理上限

- `IdealPDActuator` 计算为 `effort_target - damping * wheel_velocity`。高速下 `damping=0.1` 会吃掉较多净力矩。
- 早期 damping compensation 符号曾经可能反向，当前代码使用正确的 velocity feed-forward 补偿方向。
- 直接将 damping 或 torque limit 跳到硬件值会破坏旧策略。当前 `wheel_damping_anneal_target=-1` 表示默认禁用 actuator bridge。
- 可用的 bridge 代码已存在：cosine anneal 同时迁移 damping、torque scale 和 effort limit，且受 speed/survival gate 控制。但它应在名义技能训完后单独启用。

### 4.5 灾难性遗忘和 teacher 冲突

- 只续训一个 shared MLP 会在新速度/yaw 上改写低速技能。
- 固定 teacher 权重过大会让 student 无法超越 teacher；权重退火过快又会遗忘。
- 多 teacher 同时对 wheel common mode、yaw differential 和 leg posture 施加相反梯度时，直接求和会互相抵消。
- 现在的主思路不是无限增大 distillation weight，而是：冻结已认证技能 + command-gated residual head + on-policy rehearsal + strict retention gate。

## 5. 当前 curriculum 结构

Command level 从 0 开始，当前总共 0-46：

| Level | 技能 |
|---|---|
| 0-4 | 1.0 m/s：直行，yaw 0.10/0.20/0.30/0.45 |
| 5-8 | 1.5 m/s：直行，yaw 0.10/0.20/0.30 |
| 9-13 | 1.2 -> 1.35 -> 1.5 -> 1.6 -> 1.75 m/s 严格直行恢复 |
| 14 | 2.0 m/s 直行 |
| 15 | 2.0 m/s + 0.25 yaw |
| 16 | 2.5 m/s 直行 |
| 17 | 3.0 m/s 直行（早期版） |
| 18 | 2.0 m/s + 0.30 yaw 压弯强化 |
| 19-23 | 2.5/2.6/2.7/2.8/2.9 m/s 逐点直行认证 |
| 24 | 2.0 m/s + 0.15 yaw |
| 25-38 | 2.2/2.4/2.5/2.6/2.7/2.8/2.9 m/s 的 0.15/0.30 yaw 密集压弯点 |
| 39-41 | 3.0/3.1/3.2 m/s 直行再认证 |
| 42-45 | 3.0/3.2 m/s 的 0.15/0.30 yaw |
| 46 | terminal 3.2 m/s + 0.30 yaw 重复认证点 |

当前训练在 Level29：`2.5 m/s + 0.15 rad/s`，正负 yaw 都必须通过。

## 6. Gate 的真实含义

不要因为一次 mean vx/yaw 看起来好就宣布进阶。当前 gate 同时要求：

- transition 完成后再 hold `500000 global env steps`。
- `300-step` rolling speed/yaw window。
- ready fraction `>=0.70`。
- stable fraction `>=0.45`。
- posture `>=0.90`。
- wheel contact `>=0.55`。
- survival `>=0.95`，默认需活到 `1200 control steps`。
- high-speed linear error 一般 `<=0.15/0.20 m/s`，每个 stage 可 override。
- yaw error `<=0.05 rad/s`。
- 正/负 yaw 每个方向至少 `300` 个 window samples。
- 完整 unassisted episode 的 tracking 和 survival，不让 reset bootstrap 或幸存者偏差伪通过。
- 连续 `125` 个 control steps 满足所有条件才进阶。
- retention gate 要求 rehearsal 中已认证技能也同时通过。

当前 retention gate stage IDs：
`[8,15,18,23,26,28,30,32,34,36,38,43,45]`。

当前 rehearsal fraction 为 `0.40`，priority fraction 为 `0.85`。这会导致全局 `command_target_vx` 低于 active target，是正常的混合现象。

## 7. 当前 policy 架构

### 7.1 不是多个独立 ONNX

- 当前 checkpoint 是一个 actor state dict，最终应导出一个 policy/ONNX。
- 它内部有 command-gated expert/residual modules，但不是“多个完全独立策略由控制器手动切换”。
- 输入仍是同一个 25-D obs，输出仍是同一个 6-D action。

### 7.2 模块分工

1. Frozen base MLP：`25 -> 128 -> 64 -> 32 -> 6`。
2. Frozen low/medium turn expert：保留早期 1.0/1.5 m/s 转弯基线。
3. Cumulative turn residual bank：从 Level26/28 到 Level45 的新速度/yaw 点各有速度/yaw gated residual。
4. Directional split：每个 turn head 内 positive/negative yaw 分路，避免左右转梯度相互抵消。
5. Frozen 2.9 m/s straight expert：只在 target vx 约 `2.88-2.92` 且 yaw 近零时混入。
6. Bounded domain residual bank：专门吸收 domain randomization 差异，不覆盖已认证技能。
7. Optional actuator bridge adapter：最后 5.5 -> 3.0 Nm / damping 迁移时使用，当前 v139 未启用。

### 7.3 Domain residual bank 的冲突隔离

已经发现“只给 2.9 straight expert 留 residual”仍然会让其他冻结技能无法适应随机动力学。现在改为 5 个互斥/平滑路由的 domain heads：

- `straight`
- `positive_small_yaw`
- `negative_small_yaw`
- `positive_large_yaw`
- `negative_large_yaw`

每次只有对应 command regime 收到梯度，已做单样本梯度隔离测试。输出用 `0.04*tanh(...)` 限幅，即使 domain head 漂移也不能取代 frozen skill graph。

### 7.4 HIM / DreamWaQ / NP3O 现状

- 仓库内有 HIM、DreamWaQ、NP3O 的 algorithm/module/runner 源码和可选导入。
- 当前 RMUC 主线实际使用 stock `OnPolicyRunner + PPO + feed-forward ActorCritic`。
- 启动时会看到可选 HIM/DreamWaQ/NP3O runner 因 installed `rsl_rl.modules.ActorCritic` API 版本不匹配而跳过的 warning。这不会影响当前 stock PPO。
- “HIM + DreamWaQ 共享主干、history length=5”只进行过讨论，不应宣称已经在当前 checkpoint 里实现。
- 不要在当前 Level29-46 中途切换 algorithm/actor contract；应先完成当前 PPO 技能线，再做独立 ablation。

## 8. Teacher distillation / rehearsal 的准确状态

### 8.1 Residual 解决什么

- Residual 是“在已认证策略上增加受控容量”，不是另外训一个无约束 actor。
- Frozen base/expert 保证旧能力的基线输出不被 PPO 改写。
- Turn delta residual 只在对应 speed/yaw support 区间激活，且 positive/negative yaw 分路。
- Domain residual bank 不学新 command 技能，主要学相同 command 在 mass/friction/damping/gain/noise/delay 改变时的小修正，总输出限幅 `0.04 action`。
- Actuator bridge residual 是第三种 residual：它在最后 5.5 -> 3 Nm / damping-center 迁移时使用，不应和当前 command skill residual 混为一个概念。

### 8.2 Rehearsal 解决什么

- Rehearsal 是当前 on-policy rollout 中专门分出一部分 env，让它们执行旧速度/yaw command。
- 当前通常为 40% env，其中 85% 优先分配到关键 anchors。
- 它不只是“看一眼旧任务”：这些 env 产生真实 PPO reward/termination/physics trajectory，同时为 retention gate 提供 speed、yaw、stable、survival 数据。
- 当新任务梯度过强时，rehearsal 将旧状态分布重新带回 on-policy buffer，防止“新任务压过旧技能”。
- Rehearsal 本身不保证不遗忘；必须同时有 frozen skill graph/residual routing 和 retention gate。

### 8.3 Teacher distillation 的 loss 与输入

- 训练脚本已支持 online frozen teacher + fixed replay dataset。
- Teacher 和 student 都使用同一个 25-D policy observation；当前不存在 teacher 吃特权观测、student 吃非特权观测的非对称蒸馏。
- 直行 teacher 对齐 deterministic action，使用 `SmoothL1 loss`（不是 KL，不直接对齐 Gaussian std/distribution）。
- Turn teacher 不默认对 6 个 action 全量做同权误差：
  - wheel differential loss 是主项，保留 yaw steering。
  - wheel common-mode weight 可单独设定，避免低速 teacher 把高速 student 拉回低速。
  - leg-action weight 可单独设定，避免低速 posture 锁死高速平衡。
- Fixed replay dataset 存 observation/action/command labels，左右 yaw 会做 balanced sampling。
- Online teacher 只能监督它已认证的 target 区间，不得在超出自身速度范围时外推 common-mode action。

### 8.4 蒸馏退火的准确机制

- CLI 参数：
  - `--teacher_distill_weight`：初始强度。
  - `--teacher_distill_final_weight`：最终强度，必须不大于初始值。
  - `--teacher_distill_anneal_iterations`：退火的 trainable actor updates 数。
  - `--teacher_distill_learning_rate`：独立蒸馏 optimizer LR。
  - `--teacher_distill_updates`：每次 PPO update 附加的 teacher updates 数。
  - `--teacher_distill_only_iterations`：早期可只训 critic + teacher actor step，暂不放 PPO actor gradient。
- Weight 用 cosine 从 initial 退火到 final：
  `w(t)=w_final+(w_initial-w_final)*0.5*(1+cos(pi*t/T))`。
- 仅缩放 loss 对 Adam 实际 step 影响不足，所以代码还同步令 distill LR 按 `effective_weight/initial_weight` 缩放。
- 目的是训练前期保住 teacher 技能，后期逐步让 PPO/student 超越 teacher；不要再长期固定 `0.35` 高权重。
- 当 speed/turn teacher gradient cosine < 0 时可启用 `--teacher_gradient_projection`，对冲突分量做投影，而不是简单相加互相抵消。
- 要同时记录 `teacher_distill_effective_weight`、`effective_lr`、`anneal_fraction`、`gradient_cosine`、`gradient_conflict`，否则无法判断是 teacher 压制 student 还是 PPO 遗忘。

### 8.5 三者的边界

- **Residual**：提供可控、受 gate 限制的新参数容量。
- **Rehearsal**：把旧技能状态分布带回当前 on-policy 训练和验收。
- **Distillation**：用已验证 action labels 约束 student，应退火，不能永久主导。
- **Retention gate**：是最后的行为证据；上面三种手段都不能代替它。
- 固定 teacher 不是当前 v139 的主保留机制；v139 主要依赖 frozen skill graph + on-policy rehearsal + retention gate + domain residual bank。
- PPO rollout buffer 不被跨轮重用；“经验回放”在本项目主要指 fixed teacher observation/action dataset，不是 off-policy PPO replay。

## 9. Progressive flat domain randomization

已接入 Level29-46，地形始终为 128 m flat plane。

随机化参数：

- base mass + inertia，终局最大 `+/-5%`。
- robot collision friction，终局最大 `+/-15%`。
- wheel damping，终局最大 `+/-10%`。
- wheel torque gain，终局最大 `+/-5%`。
- normalized observation noise std，终局最大 `0.01`。
- observation delay，最多 1 control step，终局最大抽样概率 `0.35`。

强度不按 iteration 盲目增长，而跟随已通过 gate 的 command level 和当前 transition fraction。Level29 目前 strength `0.1053`，实际范围约：

- mass `+/-0.53%`
- friction `+/-1.58%`
- wheel damping `+/-1.05%`
- wheel gain `+/-0.53%`

只在 episode reset 时重采样，不在轨迹中间突然改物理。Playback/确定性 evaluation 默认关闭 DR。

注意：日志中 `domain_*_scale` 的 mean 应该接近 `1.0`，因为是对称采样；mean 接近 1 不代表“没有随机化”，要看 strength 和分布。

## 10. 关键 checkpoint 和 teacher dataset

### 10.1 建议保留的 checkpoint

- 1.5/2.0 m/s 转弯 teacher：
  `logs/rsl_rl/rmuc_wheel_leg_course/2026-09-16_17-28-19_command_level15_deterministic_gate_certification_v61b/model_level_16_entry_33.pt`
- 2.9 m/s straight frozen expert 来源：
  `logs/rsl_rl/rmuc_wheel_leg_course/2026-09-19_20-50-50_command_level21_v87_survival_refine_v98b/model_125.pt`
- 当前 Level29 净基线（v139 的起点）：
  `logs/rsl_rl/rmuc_wheel_leg_course/2026-09-21_15-19-35_level28_strict_to29_v130/model_125.pt`

### 10.2 重要 dataset

- `artifacts/rmuc_teacher_datasets/level8_gatepassed_v61_currentobs_balanced_v1.pt`
- `artifacts/rmuc_teacher_datasets/level15_gatepassed_v61_currentobs_balanced_v1.pt`
- `artifacts/rmuc_teacher_datasets/level23_2p9_v98b_model125_gatepassed_v1.pt`
- 其他 level12/13/16/21 数据集也在同目录，但启用前必须看 schema、gate_passed 和当前 obs semantics。

### 10.3 不可盲信的 checkpoint

- `2026-09-20_20-05-26_level28_directional_bank_v113/model_level_29_entry_124.pt` 属于旧 gate 语义，不能当作严格 Level29 证书。旧 aggregate yaw 和 survivor-biased survival 可以伪通过。
- 旧 `2999.pt` 只能结合对应 run 的 env.yaml/obs semantics 使用，不能只看文件名就续训。

## 11. 当前正在运行的训练

### 11.1 运行状态

- Run name：`level29_progressive_flat_dr_domain_bank_v139`
- PID（会变，以 pgrep 为准）：`1475479`
- Log：`train_level29_progressive_flat_dr_domain_bank_v139.log`
- Run directory：
  `logs/rsl_rl/rmuc_wheel_leg_course/2026-09-21_19-15-18_level29_progressive_flat_dr_domain_bank_v139`
- 起点：v130 Level29 entry + external v98b 2.9 straight expert。
- 目标：严格进入 Level30 后自动保存 entry checkpoint 并停止。
- Checkpoint 每 `125 iterations` 保存。

### 11.2 启动命令

```bash
cd "/home/kellen/桌面/11 (1)/11/wheeled-legged_RL"

CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH="$PWD/source/agent_rl:$PWD/source/agent_tasks:$PWD/source/agent_world${PYTHONPATH:+:$PYTHONPATH}" \
/home/kellen/anaconda3/envs/issaclab/bin/python scripts/rsl_rl/train.py \
  --task Robotics-Wheelbipe-V14-RMUC-Command-v0 \
  --num_envs 128 \
  --max_iterations 1500 \
  --num_steps_per_env_override 128 \
  --command_level 29 \
  --start_command_at_target \
  --stop_at_command_level 30 \
  --checkpoint logs/rsl_rl/rmuc_wheel_leg_course/2026-09-21_15-19-35_level28_strict_to29_v130/model_125.pt \
  --speed_expert_checkpoint logs/rsl_rl/rmuc_wheel_leg_course/2026-09-19_20-50-50_command_level21_v87_survival_refine_v98b/model_125.pt \
  --load_critic_for_finetune \
  --turn_residual_bank \
  --turn_directional_residual_bank \
  --action_std_override 0.004,0.004,0.006,0.004,0.004,0.006 \
  --freeze_leg_action_std \
  --min_wheel_action_std 0.006 \
  --learning_rate_override 2e-6 \
  --num_learning_epochs_override 1 \
  --highspeed_regression_guard \
  --highspeed_regression_guard_warn_only \
  --allow_curriculum_override \
  --run_name level29_progressive_flat_dr_domain_bank_v139 \
  --viz none
```

### 11.3 2026-09-21 约 iteration 205 的快照

- `command_level=29`
- active target：`2.5 m/s`, `|yaw|=0.15 rad/s`
- active actual vx 当轮：约 `2.0-2.2 m/s`，还有短时波动，不能只看这个 mean。
- active yaw aligned：约 `0.13 rad/s`。
- active turn gate max linear EMA：约 `0.03 m/s`。
- active turn gate max yaw EMA：约 `0.02 rad/s`。
- active survival EMA：`1.0`。
- root z：约 `0.307 m`。
- tilt：约 `3.85 deg`。
- tilt termination：约 `0.0001`。
- DR strength：`0.1053`。
- 当前 active gate 基本是通过状态，真正阻塞是 retention：
  - retention ready fraction `0.8333`
  - retention max linear error 约 `0.18`
  - retention max yaw error 约 `0.10`
  - retention stable/survival 还有未收集完的 bucket，显示 0 时不能简单理解为技能完全失败。

结论：v139 目前不是“Level29 active skill 不会”，而是 retention 中 Level23 样本为零的原因尚未确定，且还有几个已确认的薄弱 bucket。不要为了快速进阶直接删掉 retention gate。

### 11.4 Level23 straight retention 零样本：现象确认，根因待定

约 iteration 205 之后可以确定的只有日志现象：

- Level23 是 `2.9 m/s, yaw=0` 直行。
- 它被列入 `rehearsal_gate_stage_ids`，用来保住 2.9 straight。
- `level23_positive_sample_count=0`。
- `level23_negative_sample_count=0`。
- 同时两个 survival episode count 已经超过 250，证明 Level23 rehearsal episode 确实被分配和结束。
- 两个 survival fraction 仍为 0，与 tracking sample 为 0 同时出现。
- retention ready fraction 长期为 `0.8333`，Level23 是当前未 ready 的 stage。

不能再直接将它归因为“遗漏 straight 分桶”：

- v139 实际加载的 `env.py` 在训练启动前已有 straight compatibility：
  `direction_match = turn_direction_match | (~self._rehearsal_gate_turning)[None, :, None]`。
- 对 yaw=0 的 gate stage，`_rehearsal_gate_turning=False`，理论上两个兼容 slots 都应能匹配。
- `env.py` 修改时间约为 18:55，v139 约 19:15 启动，因此也不能在没有更多证据时断言“只是没有重启训练”。

当前更值得怀疑的路径是 `window_ready` 之前就终止：

- tracking sample 需要 `window_ready=True`，即先积累完整 300-step window。
- survival episode count 对任何已结束 rehearsal episode 都会增加；它不要求先活满 300 steps。
- Level23 survival fraction 长期为 0，所以完全可能是 2.9 straight rehearsal episode 在 tracking window ready 之前就因 tilt/root/contact/其他 unsafe reason 终止，从而 sample count 始终为 0。
- 这仍是基于统计结构的推断，不是已证实根因。

下一步应先增加/检查 Level23 stage-local 证据：

- complete episode length/age 分布。
- stage-local termination reason：tilt、root low/high、joint limit、xy distance、linear speed、nonfinite。
- reset 后到 termination 前的 vx、pitch、root z、wheel effort/slip。
- speed expert 和 bounded domain residual 在 Level23 rehearsal 上的实际 action/common-mode。
- `window_ready` 达到率和首次 ready step。
- 用 fixed Level23 evaluation 在 nominal 和当前 Level29 DR strength 下分别对照，确认是技能退化、DR 超出 expert 鲁棒范围，还是 gate bookkeeping 问题。

在这些证据完成前，文档只保留“Level23 sample=0 且 survival=0”这个事实，不把它宣称为已定位的 straight-bucket bug。

除了这个待查问题，iteration 205 还有需要真实修复的 retention 薄弱点：

- Level8 positive：yaw error 约 `0.0975 > 0.05`，stable fraction 约 `0`。
- Level26 negative：yaw error 约 `0.0544 > 0.05`，stable fraction 约 `0.3515 < 0.45`。
- Level28 negative：yaw error 约 `0.0667 > 0.05`，survival 约 `0.898 < 0.95`。
- Level8/15/18/26/28 的某些方向 complete-episode count 仍小于 64，需区分样本不足和指标真失败。

排查顺序应是：先为 Level23 补齐 stage-local termination/window-ready 证据并做 nominal/DR A/B，确定根因后再修；同时用 rehearsal priority/domain head 针对 Level8+/Level26-/Level28-，不要放宽全局 gate。

## 12. 当前 controller 状态

- `control.py` 已存在，是 Isaac Lab keyboard controller entry point。
- W/S 控制前进/后退，A/D 控制左/右 yaw，command 有渐进加减速，避免阶跃。
- 它当前默认还指向 Level18 checkpoint：
  `2026-09-18_21-50-10_command_level17_stage_survival_cert_v87/model_level_18_entry_36.pt`。
- 历史 GUI 暴露过：前进可能反向、A/D 响应弱、镜头不易观察。其中一部分是旧 checkpoint 能力不足，一部分是 command/obs 语义和视角。
- 不要现在就把 v139 中间 checkpoint 替换为最终控制器策略。先进入 Level46 并做 nominal + randomized grid evaluation，再更新 controller 默认 checkpoint/ONNX。

## 13. 接下来的执行顺序

### Phase A：完成 Level29 retention，进入 Level30

1. 继续观察 v139。
2. 如 retention 长时卡住，从日志中找出具体 `levelX_positive/negative` bucket，不要只看 max 聚合值。
3. 区分“sample/episode 数不足”和“已有足够数据但 error/survival 失败”。
4. 只对真正失败的 bucket 增加 rehearsal priority 或定向 residual 容量。
5. 出现 `model_level_30_entry_*.pt` 后，用 fixed-policy evaluator 独立验证 Level29 正/负 yaw 和 Level23 2.9 straight，再作为下一轮基线。

### Phase B：密集完成 2.5-2.9 m/s 小/大 yaw

- Level30-38 逐点训练，每个 level 都使用前一个 entry checkpoint。
- 已认证 turn residual head 冻结，只训当前 head + bounded domain residual bank。
- DR 强度随 gate 通过逐级上升，不按 iteration 强推。
- 每轮都保留 Level8/15/18/23 等关键 rehearsal anchors。

### Phase C：3.0-3.2 m/s 直行和压弯

- Level39-45：先再认证 3.0/3.1/3.2 straight，再做 3.0/3.2 小/大 yaw。
- Level46 只能在 Level45 所有 active + retention + survival gate 通过后进入。

### Phase D：真机执行器迁移

- 以 Level46 的名义动力学 checkpoint 为基线。
- 开启 actuator bridge adapter。
- 渐进将 wheel torque scale/effort limit 从 5.5 迁到 `3.0 Nm`。
- damping 中心应来自实测 motor/current-loop effective damping，不能盲目把 XML passive damping `0.01` 当作完整电机阻尼。
- 每个 bridge 强度都必须在 active + rehearsal survival/tracking 稳定时才增加；失败则暂停增强，不继续改变动力学。

### Phase E：最终验收与部署

- nominal flat 确定性 grid。
- 中强度 randomized flat multi-seed grid。
- 从零速起步和已在运动两种 reset。
- 正/负 yaw 分别统计，不允许 aggregate 抵消。
- 导出单 policy/ONNX，更新 `control.py`。
- 用 W/S/A/D 做加速、减速、低速转弯、中速转弯、2.9/3.0 高速直行和高速压弯可视化。

## 14. 最终验收矩阵

至少要覆盖：

| vx target (m/s) | yaw target (rad/s) |
|---|---|
| 1.0 | 0, +0.15, -0.15, +0.30, -0.30 |
| 1.5 | 0, +0.15, -0.15, +0.30, -0.30 |
| 2.0 | 0, +0.15, -0.15, +0.30, -0.30 |
| 2.5 | 0, +0.15, -0.15, +0.30, -0.30 |
| 2.9 | 0, +0.15, -0.15, +0.30, -0.30 |
| 3.0 | 0, +0.15, -0.15, +0.30, -0.30 |
| 3.2 | 0, +0.15, -0.15, +0.30, -0.30 |

每个格子都要看：

- temporal absolute speed error。
- temporal absolute yaw error。
- positive/negative direction success fraction。
- full-episode survival。
- root z minimum/mean。
- tilt/roll/pitch 及 termination。
- torque target/applied saturation。
- wheel slip。
- randomization seed 间方差。

建议最终硬条件：

- root z 暂按 `>=0.15 m` 验收；允许低姿态，仍须检查倾覆及导轮触地。
- survival `>=0.95`，最终部署候选应尽量接近 1.0。
- speed error 按 stage 不高于 `0.15-0.20 m/s`。
- yaw error `<=0.05 rad/s`。
- 正/负 yaw 均通过，不能只看 mean。

## 15. 日志解读中最容易误判的点

1. `command_target_vx` 是 active + rehearsal 的混合均值；当前阶段目标要看 `command_active_target_vx`。
2. `command_active_actual_vx` 是当前 rollout 短时均值；gate 主要看 rolling window EMA 和 completed episode error。
3. 某个 direction 的 instantaneous speed error 可因当轮样本少而大幅摆动；要对照 sample count。
4. retention stable/survival 为 0 可能是还没有足够 complete episodes，不一定是全部摔倒。
5. domain scale 的 mean 接近 1 是对称随机化的正常结果。
6. `Mean action std: 0.00` 可能是终端显示精度造成；实际参数是 leg `0.004`、wheel `0.006` 级别，要看完整参数日志。
7. 开始数轮从静止 reset 加速，actual vx 从 0 上升是正常瞬态，不能用 iteration 0-3 判断失败。
8. 训练进入下一 level 后 command 会重新 transition，短期 mean 下降不代表策略立即遗忘。

## 16. 已经证伪/不要重复的方案

- 不要因为早期分析就直接删除当前 USD reset 的 `+90 deg X`。
- 不要把 root z 改成轮心高度 `0.06 m`。
- 不要为了加入 vx 就把 obs 从 25 改成 28 并直接加载旧 actor。
- 不要用全 env `torch.all` 连续稳定作为唯一 gate。
- 不要用 aggregate yaw 掩盖某一方向失败。
- 不要只加大 underspeed 负奖励而不解决访问目标状态、rollout horizon 和 action capacity。
- 不要继续用固定高权重 teacher 直接压 actor；会锁死 student 上限。
- 不要让 low-speed turn teacher 直接监督 high-speed wheel common mode。
- 不要把 frozen expert 完全锁死后又要求它适应 DR；现在由 bounded domain residual bank 解决。
- 不要用 rough terrain 代替当前 flat-domain robustness 目标。
- 不要直接把 5.5 Nm / damping 0.1 切成 3 Nm / 0.01。
- 不要在没有同 stage critic 时默默 fresh critic 续训高速 actor；优先 `--load_critic_for_finetune`，除非明确做重建 critic 实验。
- 当前 installed RSL-RL `runner.load()` 不接受旧版 `load_optimizer=` 参数；脚本已有兼容处理，不要恢复旧调用。

## 17. 常用操作

### 查当前训练

```bash
pgrep -af 'scripts/rsl_rl/train.py'
tail -n 450 train_level29_progressive_flat_dr_domain_bank_v139.log
```

### 只停训练，不动 TensorBoard

```bash
pid=$(pgrep -f '/home/kellen/anaconda3/envs/issaclab/bin/python scripts/rsl_rl/train.py' | head -1)
test -n "$pid" && kill -INT "$pid"
```

### TensorBoard

训练与 TensorBoard 是两个独立进程。当前未确认 TensorBoard 是否在运行；需要时检查：

```bash
pgrep -af tensorboard
```

常用 URL 是 `http://127.0.0.1:6006`，但返回给用户前必须先确认进程、端口和 logdir。

## 18. 关键文件职责

- `configs/rmuc_wheel_leg.yaml`：USD/runtime、action/control、47 个 command stages、gate、rehearsal、reward、PPO、safety、DR 的主配置。
- `env_cfg.py`：将 YAML 转成 Isaac Lab cfg，强校验参数，维持 25-D/6-D contract。
- `env.py`：reset、canonical state、command curriculum、rehearsal、active/retention gate、reward、termination、执行器、DR 和 diagnostics。
- `rmuc_command_adapter.py`：frozen experts、turn residual bank、directional heads、2.9 speed expert、domain residual bank、actuator bridge adapter。
- `train.py`：CLI、checkpoint 兼容加载、teacher distillation、residual 安装/冻结、gate sidecar、regression guard。
- `play.py`：加载 checkpoint 内的 expert/residual modules，GUI/playback。
- `evaluate_rmuc_stage.py`：fixed-policy 独立验证，不要只用 training mean 代替它。
- `control.py`：W/S/A/D Isaac controller，目前仍是 Level18 旧默认 checkpoint。
- `docs/rmuc_curriculum.md`：当前 curriculum/gate/DR 设计说明。
- `docs/rmuc_wheel_leg_asset.md`：USD 拓扑、质量和 asset 约定。

## 19. 新对话接管检查表

- [ ] 确认 v139 进程是否存活。
- [ ] 确认最新 iteration 和 command level。
- [ ] 分析 active gate 和 retention gate，不看单一 mean。
- [ ] 如已进 Level30，找 `model_level_30_entry_*.pt` 并做 fixed-policy 独立评估。
- [ ] 如未进阶，定位具体 retention stage/direction 是样本不足还是指标失败。
- [ ] 修改时保持 25-D/6-D contract、flat terrain、root z >=0.15（暂定）、正/负 yaw 独立验收。
- [ ] 已认证 skill graph 保持冻结；新能力进 residual head，DR 适应进 bounded domain bank。
- [ ] 名义技能完成前不启用 5.5 -> 3 Nm 大幅 actuator migration。
- [ ] 名义技能完成后必须做 3 Nm + measured damping 的渐进 sim2real 认证。
- [ ] 最终用全速度/全 yaw/双方向/multi-seed grid 验收后再更新 controller/ONNX。

## 自复位 reward 训练日志通道已补齐（第263条）

RSL-RL 仅读取 `extras['episode']`/`['log']`，旧 recovery_env 的自定义 `extras['recovery']` 不会进入 TensorBoard。现已在 recovery_env 写入独立 `Recovery/*` 状态项及 recovery_v1 的 `RecoveryReward/*` 七项，并每步创建新字典。源码方法审计保存在 `artifacts/recovery_logging_audit/source_method_check.json`；py_compile 和 4 项 reward 测试通过。该改动不改变奖励、PD、观测、拓扑或当前已加载训练进程。

## v303中段诊断（第306条）

v303约492/1000更新时，支撑姿态过程项已上升，但机体高度仍低于严格站立门槛，`ready/established/height_reached/stable_hold`全为0。该现象按疑似局部 reward-hacking 记录，不能把过程奖励或训练均值当作自复位成功证据；后续保留高度和稳定保持门槛并单独审计。

## v303提前停止与八点保护（第307条）

v303从早期约0.136m提升到中后段约0.24m，但300–839更新在0.208–0.257m平台，严格0.28m高度和ready/height_reached/established/stable_hold均未通过；保留model_800并停止末期GPU。下一轮从该checkpoint接续，过程高度目标设为0.36m而成功门槛保持0.28m。恢复 residual 合入组合策略后必须重新执行 locomotion 八点 gate，恢复checkpoint不得冒充行走结果。

## 稳定优先的恢复续训（第308条）

model_800接续3次预检出现一次0.3383m瞬时高度，但随后回落且ready/stable_hold全零，不能算成功。已加入stable_pose连续塑形，要求双轮支撑、直立、低平面速度和低角速度共同改善，严格stable gate不放宽；八点 locomotion gate仍须在 residual 合入后执行。

## 稳定高度课程档位（第309条）

严格稳定高度与过程塑形高度解耦，支持0.25/0.28/0.36m课程档位；稳定判据仍包含姿态、速度、角速度、双轮支撑和连续保持。恢复分支重新合入 locomotion 后仍须执行八点 gate。

## 0.25m稳定档位验收与安全惩罚（第310条）

0.25m严格门槛四方向×4 replicas仍为0/16稳定通过，侧倒出现关节限位和数kN冲击；降低高度不是充分条件。已加入限位屏障与接触冲击软惩罚，后续采用更低slew短窗口验证后再长训。八点 locomotion gate保持不变。

## 1.5rad/s直接迁移失败（第311条）

model_800直接改1.5rad/s后3次更新出现高度和姿态快速崩溃，ready全零。后续必须训练内渐进slew退火，并在每个阶段做稳定保持和八点保护检查。

## slew退火而非直接跳到1.5（第312条）

1.5rad/s只是最终候选终点，恢复环境已支持4→1.5的显式训练退火；每一级需稳定保持、限位和冲击验收后才能继续，默认退火关闭以保持旧checkpoint兼容。

## slew退火公式验证（第313条）

4.0→1.5的起点、中点、终点和越界保持均有单测，12项恢复reward测试通过；退火长训待稳定短窗口通过后再启动。


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

## 四方向standing reward bank（第337条）

恢复奖励可按front/back/left/right独立配置，默认值保持兼容；用于下一阶段分别处理后倒翻正和侧倒双轮支撑，仍保持25D/6D契约并要求四方向完整验收。


### 337. 四方向reward bank索引修复并通过原生smoke

首次bank smoke发现多环境权重索引产生[env,env]形状，已修复为按case与env逐项取权重。修复后recovery_v3/30度/4rad/s/progress/refall配置原生4环境初始化通过，实际初始倾角30度、final_ready全false（仅初始化/合成动作），腿PD读回60/2、轮限幅3Nm、25D progress观测保持；13项测试、py_compile和`artifacts/continuation_v337_smoke.json`核验通过。该smoke仅验证接线和reset，不是自复位行为认证，case bank默认值与旧全局权重一致，未启动新训练。

## 双轮支撑转换事件（第338–339条）

侧倒恢复新增一次性双轮支撑转换奖励，默认关闭；v339 30度短预检事件为0，未证明行为改善。正式实验需要侧倒初态课程，不能用30度课程代替。

## 气弹簧单位勘误（第340条）

当前最终口径为每腿10.775N·m等效力矩；10.775N径向力属于旧错误假设和历史实验，不能用于最终执行器契约。径向力版本保留为待辨识/DR分支。

### 422. 镜像注入和D敏感度扫描完成，暂不迁移

model_6684的25D synthetic mirror probe显示输出不能按简单左右交换复现；不同关节轴符号假设的误差不同，最终物理结论需结合关节轴oracle，不能直接改bank公式。固定 `vx=1,yaw=+1` 的D扫描显示：`.005/.020` 直接倾覆，`.050` 存活但速度/偏航误差和stable不达标，`.080` 通过短诊断。该结果支持D退火与正yaw定向微调路线，但仍不是最高DR或sim2sim验收。证据在 `artifacts/continuation_v421/mirror_report.json` 和 `artifacts/continuation_v422/d_scan_summary.json`；当前不宣称策略已对D鲁棒、不迁移。


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


## 2026-10-03 v443/v446：轴向错误使新动力结果失效

v443完成25更新（10critic-only+15新残差），源784tensor映射冻结、初始化输出相同及warmup审计成立。候选model6860 SHA2ac906d0ea5db23940a8146fffec8ffebf5da14e98c89e9577f55fa820297f07。旧DR=.05 retention八点各32回合/8通过/全存活；但不得晋升为新契约通过。v444同候选JIT误差0、ONNX最大2.98e-7，MuJoCo两命令12s存活但跟踪失败。

独立只读核查发现本机IsaacLab3 body_quat_w采用xyzw，而新增wheel passive轴转换和历史gas_spring helper按wxyz解释。实际USD关节父局部+X经过错误转换变到错误世界轴；MuJoCo xmat路径正确，不能修改其wxyz约定来迎合错误。原v440 native测试只验证提交与buffer自洽，不能证明正确关节轴，其物理有效性结论撤销。v443新动力训练/评估标invalid_physical_axis_not_certified，已停止仅本项目GPU0新评估，保留日志/候选/旧retention证据。

修复策略：新separated模式轮与气弹簧明确使用Isaac xyzw；历史legacy spring保留wxyz路径并明示为旧数值复现，绝非正确物理。新检查必须用安装Isaac API+实际USD轴作为独立oracle，并核对真实刚体相对角速度功率。修复后重新做短native验证和有界迁移，不重写历史报告为通过。最高DR和正式sim2sim仍未完成。


## 2026-10-03 v448 修正轴后完整结果：未通过，不升级DR

v447原生独立API/USD轴与真实功率检查通过后，v448从原v430的等价冻结迁移源46010fbe38f73a883297905526f4c6853b67c6842cf31deb1a69e8440ac19517完成25更新（10critic-only+15新残差）。最终model6860 SHA6b1f0526549938928a81825a869bd1c65c4c682aececb6cb7a5cb9dd83ff0608；仅新增物理残差6个tensor改变，旧actor冻结及warmup审计通过。

新separated契约（xyzw轮/气弹簧，轮1ms/腿PD2ms/policy8ms，b=.002/Fc=.001/反馈D=0诊断，5.5峰值/3持续/scale5.5，阶段弹簧1.0775Nm，DR0）完整八点至少32回合：0/8。正向+.8生存约.0044，+1/+1.2/+1.4生存0；负向-.8/-1/-1.2虽全部存活但yaw误差约.2094/.2169/.2943，-1.4也失败。旧legacy DR=.05 retention八点各32回合均存活、8/8通过，仅证明旧条件保留。promoted=false。

同candidate CPU导出parity通过；MuJoCo直行12s后半均速-.03893m/s，yaw+1均速.32843m/s/yaw1.35619rad/s，跟踪失败。不能把能跑完写为sim2sim合格。训练、双GPU评估和CPU流程均已完成，无需重训本轮；完整证据artifacts/continuation_v448/{final_report.json,evaluation.json,retention_legacy_dr005.json,cpu/summary.json}。

历史DR=.1（v431/v433）为八点各32回合全存活、tracking4/8，四个正yaw失败；通过8/8的是旧DR=.05 retention，并非.1。最高DR不能跳级。v445已准备非对称机械b与独立总惯量DR纯函数/16项CPU测试，但尚未接env，不能宣传目标DR覆盖。后续优先用匹配物理条件定位移除旧feedback D及修正spring轴后的迁移缺口，不盲增同配置更新或升级DR。

## 2026-10-04 v454-v456 接续状态

v455 的原生等效弹簧审计在 Isaac xyzw 下通过：500 steps 的 `max_world_torque_error_nm=1.7166e-5`、`max_net_world_torque_nm=2.0981e-5`。v456 使用 v450 `final_checkpoint.pt`（源 SHA `a9feac926c66f657ba6e1df5d4138d706f972096d01858096a60e7d210082dd4`）完成 25 updates，候选 SHA `de127c2902306daab90f6fb24421fdda63e8e266fe46f40c5e09427c8360d822`。freeze audit 通过，只有 physics adaptation 6 个 tensor 改变。

v456 实际契约：5.5 Nm 峰值、3 Nm 持续热参考、action scale 5.5、轮 actuator damping 0、b=.002、库仑=.001、反馈 D=0 诊断、10.775 Nm/腿等效弹簧、径向力0、腿PD60/2/40、轮1 ms、腿PD2 ms、policy8 ms、外环关闭。快速 Isaac 12 点网格全部 `passed=false`，所以不进入正式32回合、不升DR、不晋升。JIT/ONNX parity 最大误差 `2.384e-7`，只代表导出一致；同契约 MuJoCo 直行后半速度0.5289 m/s（目标1.0），+0.8转向触发 tilt/height diagnostic limit，均 `certified=false`。

当前保留 v450 作为迁移源；v456 只证明契约、轴和导出链路可运行，未恢复最终动力下的locomotion。DR=.05正式八点、最高联合DR、全速度压弯、±4自转、MuJoCo行为认证、controller部署和demo视频仍未完成。机器可读证据在 `artifacts/continuation_v456/{experiment_contract,training_execution,runtime_verified,freeze_audit,quick_eval,final_report}.json`。

v457 启动前唯一变量为 physics residual 腿通道预算从 `[.15,0,.35,.15,0,.35]` 扩到 `[.681818,0,.35,.681818,0,.35]`，由 v450 派生源 SHA `c8bfb90e8fcfc304fd79699e0e18268ca16fddfc488c60414bd98885d3e43812`。5.5 Nm 峰值、3 Nm 持续热参考、action scale5.5、后置 b=.002/Coulomb=.001、反馈D=0诊断、10.775 Nm/腿、轮1ms/腿PD2ms/policy8ms、DR0、外环关闭全部保持。预算迁移会改变 residual 输出，v457 必须独立做 retention 与快速网格；未通过不得升 DR 或改写 v456 结论。
# 2026-10-05 v465-v467 最新接续

用户确认气弹簧以只读旧`combat-dm-balance-main/application/chassis/chassis.c`实时逻辑为主：幅值`clip((-83.121*L0+40.242)*0.8,16.5,27.0)`乘`sin(front-back)`，固件以左右有符号前后电机补偿实施。它不是当前Isaac/MuJoCo每腿10.775Nm active-link恒定近似；新纯函数已复现补偿公式且33项相关测试通过，但L0/轴/符号/被动力映射和跨引擎行为尚未完成，不能部署或称sim2real对齐。旧10.775Nm是历史等效近似验证条件，不应覆盖用户新指定的主要参考。

v465 full tail reward使前期低端裁剪约16–18%，10/25更新停止，无最终策略结论。v466尾段gain0.1完成25更新、冻结审计通过，DR0十二点快筛仍1/12、正向+.8 83/83和+1.0 85/85 tilt，`promoted=false`；reward-only路线停止。v467同v430 checkpoint/seed/五点网格及分离轮动力，旧1.0775Nm阶段弹簧下只改postclip反馈D：D0正向+.8/+1.0分别58/58、69/69 tilt；D=.080625五点均存活却正向yaw/stable不达标。高D未实测，仅供小步迁移起点。当前无正式32回合、目标弹簧模型、DR、MuJoCo或实机认证。机器报告`artifacts/continuation_v466/final_report.json`与`continuation_v467/final_report.json`；不得从旧条件拼接完成声明。


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
