# RMUC world-model pilot

This is an independent SuperMimic-inspired research backend, **not the official implementation**, and not yet a replacement for the existing trainer.

## Frozen deployment boundary

- Actor input: existing25D observation, one frame. Actor output: existing6D action.
- Deployment target: STM32H723 + CubeAI. World model, critic and privileged histories are training-only.
- Continue the existing approximately30k-parameter student export/distillation route when a new deployment is requested. Actual flash/RAM/latency must still pass CubeAI Analyze and on-device timing.
- No new command slots, velocity/pitch/yaw outer loops, runtime world-model inference, or implicit actor history.
- Keep the current4.89Nm peak/3Nm continuous reference/action scale5.5 and staged4.31Nm spring; final10.775 adaptation remains unfinished.

## Implemented now

`replay.py` imports verified policy traces with strict timestep and hash checks. It pairs post-state[t-1] with raw_action[t] and post-state[t]. Windows crossing resets are rejected; only the first episode uses initial DR metadata. The16D reduced physical state intentionally omits continuous wheel angle, but is not complete physical state. Root mass/inertia context and four-frame physical/action history are internal to the model only.

`dynamics.py` implements a residual differentiable MLP and autoregressive rollout. Training8-step rollouts use no intermediate truth-state injection. Normalization is fit on training environments only.

`train_pilot.py` runs a bounded CPU identification experiment, saves model/hash/data provenance and1/8/16/32-step results versus persistence. It never emits an actor checkpoint. Fixed env IDs are held out across all rollout sources to reduce physical-draw leakage; this still does not constitute independent-seed validation.

`audit_counterfactual.py` compares baseline and unseen intervention-family trajectories with the same initial physical draw. It uses recorded action streams; future actor feedback differs between streams, so this is a necessary action-effect test, **not** an on-policy imaginary rollout or a true simulator-Jacobian proof.

`capture.py` is opt-in through `evaluate_rmuc_grid.py --world-model-replay-dir ...`. It captures pre-step state and post-step terminal state before reset separately from the returned next observation. It also records motor/transport histories, sensor/action queues, full joint state and contact forces. Terminal next observations returned by Gym belong to a reset episode and must not be treated as terminal targets. Physics solver caches and random-number states are not serialized; capture is not an exact simulator restart snapshot.

## Current experimental evidence

Artifacts: `artifacts/supermimic_pilot_20261027/`.
Early reduced models failed action-effect checks. The richer model now supports a limited BPTT pulse direction in paired Isaac tests (proposal7/8→8/8; opposite5/8→2/8). The first isolated actor update nevertheless regressed the primary stop gate22/32→21/32 and was not promoted. Production policy optimization remains unqualified; bounded candidates are quarantined experiments, not mainline replacements.

## Next implementation gates

1. Validate capture against the unchanged frozen actor's original trajectory and actions; check reset alignment and all delay units.
2. Train with richer physical/controller state and actual per-episode DR context; add diverse bounded action excitation, with held-out interventions, environments, seeds and command families. Do not silently modify deploy actor observations.
3. Require multi-step prediction and action-effect direction/magnitude agreement at intended imagination horizon. Establish a numerical gate against real simulator repeatability before policy optimization; a differentiable graph alone is insufficient.
4. Add analytic observation reconstruction consistent with the25D contract and explicit delayed queues/noise, differentiable task rewards and termination, and bootstrap critic. Preserve existing behavior gates unchanged. Confirm predicted improvements in simulator before expanding scope.
5. Start from a frozen teacher, allow only isolated bounded residual updates, add rehearsal/retention and compare against PPO using both equal interaction counts and equal wall time. Count collection, model training, actor updates and evaluation separately.
6. Switch mainline only if fixed-checkpoint independent-seed Isaac closed-loop results improve without unacceptable regressions. Maximum DR/full-speed/spin/deployment remain separate outstanding gates.

No claim is made that a model trained on these narrow stop traces can control the full robot or reproduce the paper's speedup.

### Closed-loop audit (2026-10-27)

`audit_closed_loop.py` restores the immutable teacher, verifies action/consumer and
sensor-queue parity, then drives the learned model with teacher actions generated
from predicted physical state. Commands and recorded additive sensor residuals
are held fixed; this is a conditional development diagnostic, not independent
simulator acceptance. Results distinguish held-out environment IDs from all IDs.
The native captured `root_state_w` quaternion is **xyzw**, while actor observation
math and the environment's `root_quat_w` accessor use **wxyz**. Use
`native_root_to_wxyz` once at this boundary; do not reorder native training data.
Primary reported speed/pitch must derive from predicted root physics, not the
separately learned auxiliary metric heads. Earlier invalid audits are identified
in `artifacts/supermimic_pilot_20261027/QUATERNION_AUDIT_ERRATA.md`.

`train_pilot.py --actor-state-loss-weight 1` adds an explicit normalized loss on
root state and controlled-joint sin/cos/velocity (default0 preserves prior runs).
`audit_counterfactual.py --capture-inputs --physical-root-metrics` evaluates
root-derived action effects. Both audits still leave policy-update qualification
false. A finite gradient alone does not establish an accurate simulator gradient.

### BPTT proposal and MC bootstrap pilot

`value.py` / `train_stop_value.py` fit the frozen teacher's final stop cost-to-go
from real replay, with held-out environment IDs. This critic is training-only,
narrow-scope and finite-horizon; it is not yet the production reward/GAE critic.
`propose_bounded_probe.py` backpropagates through analytic observations, actor,
consumer, queues and model into a single bounded action-pulse amplitude. It
checks autograd against finite differences, preserves all teacher parameters,
and selects the lowest training objective among evaluated iterates. An optional
MC value supplies the H32 bootstrap. This is a directional physics probe, **not
a trained deployable actor**. The evaluator's existing stop-budget projection
is reused, and Isaac sign/dose/zero-control comparisons are mandatory.

First pilot artifacts: `bptt_probe_v1`, `stop_value_v1`,
`bptt_probe_value_v1`, `bptt_probe_value_physics_v1`. Record collection/model/value/
proposal/evaluation time separately; small model cost improvements can be below
model/value errors and must never be promoted without physical confirmation.

`--train-stop-head` performs an isolated actor pilot through the frozen world
model/value: only rows0/3 of the top-level stop head final layer may change,
all other parameters remain bit-identical, and replay deviations are bounded.
It emits an actor-only checkpoint with the stale PPO optimizer removed. Do not
resume it as an exact PPO optimizer checkpoint. `--actor-lr` defaults1e-5.
A stopping-specific reverse-rebound shaping term can be fit consistently into
the MC critic; acceptance thresholds remain unchanged.

### Dynamic speed/yaw and delay expansion

`--condition-commands` adds current vx, yaw and the command-dependent leg scale
as **training-only** time-varying model context, read from hash-verified actual
world command traces. The physics consumer scales leg targets with current
command speed, so old history alone is insufficient at a command change. Actor
observation remains25D. Models with35D context are not interchangeable with old
32D-context proposal code. `--resume` preserves source-model normalization and
requires matching state schema; incompatible context dimensions fail loading.

`prepare_dynamic_course.py` constructs vx AND yaw ramps on8ms policy ticks using
the existing command interface: fixed-vx yaw acceleration/reversal, joint speed
and yaw changes, reverse and spin. Prepared sequences are not accepted skills.
`analyze_dynamic_domain.py` counts first-episode coverage, missing events, true
command parity, speed/yaw/bank diagnostics. Turning never invokes the straight
leg-symmetry gate. It deliberately does not assert full dynamic-yaw acceptance
from steady-tail scores. `audit_joint_delay.py` checks actual sampled channel
exposure before any delayed behavior interpretation.

First combined profile is sensor8ms + wheel2ms/leg2ms in parallel, action0;
nominal pure-transport longest path10ms excludes motor response. It is not a
hardware end-to-end latency measurement. Phase/clock age and motor dynamics
remain distinct. The12ms goal forbids casually stacking sensor8 + action8.

### Device and current-delay actor pilots

`train_stop_value.py --condition-commands --device cuda:0` uses the matching
command-conditioned critic schema. Its final stopped segment has constant
zero command; next-command context therefore equals current-command context,
and no unavailable post-timeout context is fabricated.
`propose_bounded_probe.py --device cuda:0` stages replay and frozen model/actor/
critic on GPU. Context dimensions and source actor hash must agree before use.
Only explicitly emitted quarantined candidates modify stop-head rows0/3;
production source weights remain immutable.

`train_pilot.py --device cuda:0 --batch-size N` additionally supports device-
resident world-model datasets. Default CPU/batch128 preserves old runs. GPU
speedup depends on batch/rollout/workload; report actual wall time rather than
assuming a small network is always faster on GPU. GPU model-fitting option has
syntax/CPU regression checks; value and actor GPU runs have real execution proof.

### Objective/gate audit

`diagnose_stop_objective.py` evaluates the exact legacy shaped cost on real
trajectories, separately from behavior gates. The first joint-delay actor slightly
reduced that cost yet failed more arrival holds. gamma.99 at8ms discounts a1.5s
settling interval to15.2%, and2.5s to4.3%. The isolated `settling_v1` pilot uses
monotone pseudo-Huber speed error, yaw error and active/hip posture terms, a
reverse rebound margin, and gamma.998. Production simulator rewards and actual
acceptance gates are unchanged. This is a tested hypothesis, not a proven fix;
paired joint-delay and sensor-only retention are both required.

External-pulse proposals now default toH64 and reject an imagination horizon
that ends before the400ms pulse. A baseline-teacher bootstrap otherwise omits
the remaining intervention, so its predicted total effect is not comparable to
full-pulse physics. The guard applies to pulse diagnostics, not autonomous actor
updates. H64 remains beyond the model's32-step training horizon; true behavior
and old-skill retention are still necessary even when effect signs agree.

`--phase-retention` is an opt-in controlled experiment after H64 actor regression.
It balances four offline trajectory phases: ordinary rehearsal, initial idle,
all stop tails and the final early braking interval. Age labels never enter the
actor. Protected idle/tail raw-action differences are bounded at.0005 versus.01
for ordinary/early phases. Sampled candidate selection is followed by a full
recorded-input phase audit before emission. This protects recorded inputs only;
actual off-trajectory behavior and all-domain retention still need evaluation.


### Scalable model-based loop (2026-10-28)

The new active development path is `train_pilot.py` -> `train_imagination.py`
-> `refresh_cycle.py`. It is SuperMimic-inspired, not an exact reproduction.
The original two-layer128 checkpoints remain loadable. Width/depth are now
explicit metadata;2048x3 is tested on GPU. `gpu_replay.WindowReplay` gathers
complete batches without per-example Python/GPU synchronization and preserves
uniform-trajectory sampling plus reset-safe valid starts.

`train_imagination.py` updates complete selected startup/stop modules (all six
outputs, all layers), preserving all other expert weights. Initial states cover
all recorded phases and command changes, not only the final parking event.
A256x128 return critic trains every iteration using detached on-policy GAE
lambda targets, five epochs/four shuffled minibatches. The actor uses BPTT
through a frozen model and critic input derivatives. Fresh sensor noise uses
the simulator sensor function; explicit sensor/action histories remain present.
The actor remains25D single-frame/6D. gamma.998 matches this8ms adaptation;
the paper's50Hz/gamma.99 is not silently copied to a different physical horizon.

The new monotone task objective covers vx/yaw, straight-only active/hip symmetry,
pitch and straight roll. It includes an explicit failure cost to avoid rewarding
early death in a negative-cost objective. Its smooth survival is only a model
proxy, not an exact simulator termination implementation or an acceptance gate.
There is no claim that this proxy/reward is identical to the paper or production
PPO reward. Initial actor lr2e-5, batch256/H32 and20updates are integration
settings, not the paper's4096rollouts. Extending domains requires their captures
and real retention checks, not an all-skills claim from this short sequence.

`refresh_cycle.py` evaluates the candidate in joint-delay and sensor-only physics,
captures fresh joint-delay trajectories, and refits the model with those data.
Sensor-only buffers are not mixed because actuator queue shapes differ.
It never auto-promotes an actor. Historical replay is retained for rehearsal;
a long-running FIFO scheduler and full-domain independent-seed acceptance are
still outstanding. See `artifacts/supermimic_online_20261028` for actual runs.

Directional step-overshoot shaping is optional (`--overshoot-weight`, default0).
It uses training-only command-event metadata, with an18% margin ahead of the
unchanged20% physical gate. It distinguishes overshooting the destination from
still approaching it. Smooth ramps have no discrete event penalty. The reward
does not add time, previous command, or a navigation slot to the deployed actor.
`--retention-teacher` can keep the original accepted teacher as a rehearsal anchor
while continuing a quarantined candidate; source freeze audit remains relative
to that candidate. Actor/critic/model optimizer continuation is now explicit.

Second-cycle audit found a critic conditioning defect: directional overshoot
reward used previous-step direction/magnitude, but critic saw current command
only. Future overshoot-enabled runs append these two privileged reward-context
fields to critic input only; actor remains25D. Loading a critic from another
reward/schema is now rejected. The failed second-cycle critic is retained as
evidence, not reused. This fix needs a new critic training run and physical
acceptance; passing syntax tests does not certify behavioral improvement.

`--balance-delay-domains` stratifies initial imagined states by actual leg/wheel
transport activation. With sensor-only and joint-delay replay, each batch uses
50% of each condition, independent of capture counts. Default legacy sampling
remains uniform by trajectory. Runtime contract records per-domain valid-window
counts. `queue_schema.py` permits only zero padding unused queue tails when
transport is inactive; it rejects physical-field fabrication and active-queue
padding. Queue normalization must be fit on both conditions before policy use.

`evaluate_rmuc_grid.py --world-model-action-noise-std .03` is an opt-in
simulator-only identification capture. Gaussian action-unit noise is clipped
at3sigma and uses a separate reproducible RNG, so it does not consume the DR
sensor RNG stream. The unchanged actuator consumer still enforces physical
limits. This option requires privileged replay and rejects distillation/critic
audit combinations; output is flagged ineligible for deterministic acceptance.
`analyze_dynamic_domain.py` refuses noisy captures. Default0 preserves exact
legacy evaluation behavior. This is a data-collection capability, not proof
that exploration has improved a policy.

`--response-loss-weight 1 --response-batch-size 4096` adds an experimental
randomized-action moment loss alongside the unchanged rollout MSE. It restores
the immutable collection teacher to calculate actual additive excitation and
uses only training env IDs. Physical vx/yaw/pitch and active/hip differences
come from root/joint state, not auxiliary metric heads. The loss penalizes
residual response correlated with independent random actions; constant bias is
not mistaken for action gain. Held-out response audits are not used as labels.
This identifies average local effects, not a complete nonlinear Jacobian or a
policy qualification. Both physical prediction and response audits must be
reported before any BPTT reuse. It is a local RMUC adaptation, not a component
claimed to exist in the original SuperMimic implementation.

`--tracking-version bounded_physical_slope_v2` is an isolated reward experiment.
The original tolerance-normalized pseudo-Huber cost has large-error derivative
about100 for zero-speed commands versus20 at0.5m/s. Multiplying by tolerance/.05
bounds the asymptotic derivative at20 per m/s across commands while retaining
higher local curvature at idle. This does not relax any behavior gate. Critic
checkpoints store the reward version and incompatible versions cannot resume.
Critic errors under different reward scaling must not be compared as evidence
of improved calibration without accounting for that scaling.

`--project-gate-gradients` is an experimental constrained update, not a relaxed
acceptance gate. On imagined H32 windows it computes separate per-delay-domain
worst-quartile window-peak violations: overshoot18% margin, late speed/yaw and
straight active/hip differences. Late-phase labels use existing1.5/2s step
budgets and remain training-only. The actual Adam parameter proposal is projected
onto local non-increasing halfspaces for active violations. This operates after
Adam preconditioning; projecting raw gradients alone would not guarantee the
same step direction. Numerical nonconvergence falls back to zero update and
restores optimizer state. It does not ensure nonlinear losses decrease, does
not cover an entire stopping event, and does not certify real retention. All
candidate releases still require unchanged full physical gates. Default off.
