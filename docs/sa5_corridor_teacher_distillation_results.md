# SA5 Corridor Teacher Distillation Results

Date: 2026-07-24 (Asia/Taipei)

## Decision

Freeze c12 as the current joint candidate:

`logs/rnn_car/sa5_e2e_k8_obb_corridor_teacher_distill_c12_stepnorm_s42/checkpoint_128.pt`

Do not use c11 or c13. Do not continue global corridor-frame distillation from
c12. c13 failed the 1.2 m narrow gate immediately. The later
intervention-only and projection-only branches are also rejected.

The c12 policy meets the requested joint boundary:

- Gate2 pooled collision rate < 9%.
- Gate5 1.2 m crossing rate >= 95%.
- The 4 m x 10 m, 2-static + 1-dynamic corridor improves over c10.

It does not graduate the corridor gate yet. The formal corridor threshold is
SR >= 90%, CR <= 10%, TO <= 5%; c12 remains below that threshold.

## Fixed Evaluation Results

| Candidate | Corridor projection | Gate2 SR / CR / TO | Gate5 crossing / CR / TO | 2S+1D corridor SR / CR / TO | Decision |
|---|---:|---:|---:|---:|---|
| c10 baseline | none | 91.40 / 8.53 / ~0% | 100 / 0 / 0% | 79.5-80.7 / 19.3-20.5 / 0% | Original fallback |
| c11 | 16 optimizer steps, joint agreement 23.9% | 89.34 / 10.33 / 0.37% | 100 / 0 / 0% | 68.32 / 31.68 / 0% | Reject |
| c12 | 4 full-batch steps, joint agreement 5.5% | 91.53 / 8.37 / 0.10% | 96.71 / 0 / 3.40% | 84.12 / 15.88 / 0% | Freeze |
| c13 | one more 4-step update, agreement 10.4% | Not run | 22.77 / 0 / 77.23% | 83.41 / 16.59 / 0% | Reject: no corridor gain |
| intervention + PPO | unsafe-action frames only, lr 0.02 | Not run | 0 / 0 / 100% | Not run | Fail-fast reject |
| intervention projection-only | unsafe-action frames only, no PPO | Not run | 32.66 / 0 / 67.74% | 83.43 / 16.57 / 0% | Reject: harms Gate5 and corridor |

Gate2 uses deterministic seeds 101, 202, and 303, 64 envs x 1200
steps. c12 per-seed collision rates were 7.9%, 8.9%, and 8.3%.

c12 Gate5 yaw diagnostics:

- p50: 0.97 degrees
- p90: 7.02 degrees
- p95: 7.29 degrees
- within 10 degrees: 97.75%

## Mechanism

The original `4096` corridor distillation batch size made the update strength
depend on the number of corridor labels:

- 64-env calibration: about 1,225 labels -> 1 minibatch per epoch -> 4 steps.
- 1024-env c11: about 16,320 labels -> 4 minibatches per epoch -> 16 steps.

That unintentionally multiplied the effective teacher update by four. c11
overfit toward the corridor teacher, increased ordinary-wall collisions, and
also made the corridor policy worse.

c12 sets `corridor_teacher_distill_batch_size=65536`, so all corridor labels
fit in one batch and four epochs remain exactly four optimizer steps. This
restored Gate2, retained Gate5 above 95%, and improved corridor performance.

c13 shows that repeated global corridor-frame projection is not safe even
when each individual update is step-normalized. The second update doubled
teacher agreement and caused a freeze/timeout collapse in the narrow gate.

The intervention-only audit then isolated the update source:

- With ordinary PPO enabled, the 1.2 m Gate5 crossing rate fell to zero.
- With PPO disabled, the crossing rate was still only 32.66%.
- The projection-only corridor result was 83.43% SR, below c12's 84.12%.
- c13's diagnostic corridor result was also only 83.41%.

Therefore the intervention projection is not a useful direction that merely
needs a stronger narrow-retention beta. It changes the policy enough to damage
the narrow approach distribution while supplying no measurable corridor gain.

Same-frame Gate5 comparison localizes the regression. Relative to the c10
narrow teacher, c12 retained linear/angular argmax agreement of 87.3%/77.3%
on approach frames and crossed 97.71%. The projection-only branch fell to
70.5%/65.3% agreement on approach frames and crossed only 49.14% in the
600-step comparison (32.66% in the full 1200-step gate). Throat agreement
remained comparatively high, so the failure is accumulated approach behavior,
not an OBB collision at the gap.

## Frozen Follow-up

Keep c12 immutable as the current SA5 Pareto candidate. Keep the original c20
checkpoint as the narrow teacher for SA6-SA8:

`logs/rnn_car/sa5_e2e_k8_obb_narrow_recovery_seg2_c1280_s42/checkpoint_1280.pt`

The original SA6 continuation started with the fixed 78/12/10 mix:

- 78% native stage scenes.
- 12% 1.2-1.4 m narrow replay, yaw limited to +/-4 degrees, with c20
  narrow-only forward KL at the calibrated beta of 0.30.
- 10% exact 4 m x 10 m deployment corridors with moving obstacles.

Do not increase corridor teacher epochs/lr, repeat c12 distillation, add a
stronger beta to rescue the rejected intervention update, or use c13.

## SA6 Retention Calibration

The beta sweep must use the production 1024-env PPO batch. A 64-env one-update
smoke collapsed Gate5 for every tested setting because its minibatches were
only one sixteenth of the production size; it is valid for runtime/NaN checks,
not policy-quality selection.

All formal candidates below start from c12 and perform exactly one 1024-env
SA6 PPO update with identical seed and scene mix:

| beta | Gate2 pooled CR | Gate5 crossing / CR / TO | 4S+2D corridor SR | Decision |
|---:|---:|---:|---:|---|
| 0.10 | Not run | 94.01 / 0 / 6.69% | Not run | Reject: Gate5 below 95% |
| 0.20 | Not run | 94.77 / 0 / 5.88% | Not run | Reject: Gate5 below 95% |
| 0.25 | about 10.0% (all three seeds fail) | 98.46 / 0.11 / 1.65% | Not run | Reject: Gate2 regression |
| 0.30 | 9.23% | 100 / 0 / 0% | 65.44% | Pareto continuation point |
| 0.50 | 9.50% | 100 / 0 / 0% | 65.61% | Reject: no corridor gain over 0.30 |

The beta response is not monotonic because changing the retention gradient can
change PPO KL early-stop and clipping paths. Do not infer an untested beta from
linear interpolation. Beta 0.30 supplies the same corridor gain as 0.50 while
preserving materially better Gate2 behavior, so it is frozen for the next
short SA6 continuation. It has not graduated yet: Gate2 remains 0.23
percentage points above its hard collision threshold, and the deployment
corridor remains well below its 90% SR gate.

## Ten-update SA6 Continuation

The uninterrupted beta 0.30 continuation reached checkpoint c10:

`logs/rnn_car/sa6_k8_obb_beta0p3_joint_replay_cont10_s42/checkpoint_1280.pt`

Its fixed joint evaluation was:

- SA5 Gate2: SR 89.51%, CR 10.46%, TO 0.03% -- fail.
- 1.2 m Gate5: SR 98.8%, crossing 99.15%, CR 0.5%, TO 0.7% -- pass.
- 4 m x 10 m, 4S+2D corridor: SR 78.78%, CR 21.22%, TO 0% -- fail.

The corridor improved by 16.70 percentage points over c12's 62.08% 4S+2D
baseline, so corridor replay is supplying useful gradient. Narrow retention
also recovered by c10. The remaining regression is specific to the old SA5
general distribution: the SA6 mix had no ordinary SA5 replay at all.

Intermediate Gate5 results were non-monotonic (c2 28.99%, c4 84.38%, c6
74.32%, c8 78.03%, c10 99.15% crossing), so rollout retention KL/agreement
cannot replace fixed deterministic gates.

## SA5 General Replay

SA6-SA8 now use a mutually exclusive 68/10/12/10 scene mix:

- 68% native current-stage scenes.
- 10% SA5-general scene replay: 10 static + 3 dynamic obstacles and 2-3
  internal walls near 4 m long.
- 12% fixed 1.2-1.4 m narrow replay with the c20 beta 0.30 teacher.
- 10% exact 4 m x 10 m deployment corridor with 4 static + 2 dynamic
  obstacles.

The SA5 replay intentionally changes only per-env obstacle and internal-wall
geometry. Reward, discount, episode horizon, goal count, and the shared outer
arena remain owned by the current stage. This avoids changing the frozen
optimization recipe or rebuilding per-env outer boundaries.

Runtime verification on 64 envs passed both initial-reset and one-update
smokes:

- event order: native reset -> SA5 replay -> corridor -> narrow;
- scene masks are disjoint;
- SA5 replay allocates exactly 10 static and 3 dynamic obstacle slots;
- 83D observations, K8 CNN, measured OBB, clean-progress reward, future 0.10,
  anti-spin 0.15, and narrow beta 0.30 remain unchanged;
- no pending installs, traceback, NaN, or OOM.

The next policy-quality experiment must use 1024 envs. Start from c10 to test
whether the 10% replay recovers Gate2 while retaining its corridor gain, save
every two updates, and run Gate2 before Gate5/corridor at each checkpoint.

## SA5-General Replay Result

The formal 1024-env recovery started from c10 with its Adam state and ran two
updates. The checkpoints were evaluated independently rather than assuming
that the later checkpoint was better:

| Candidate | SA5 Gate2 SR / CR / TO | Decision |
|---|---:|---|
| c10 input | 89.51 / 10.46 / 0.03% | Failing baseline |
| replay c128 | 90.37 / 9.63 / 0% | Best recovery point, still above 9% CR |
| replay c256 | 89.80 / 10.20 / 0.03% | Reject: second update regressed |

The replay-c128 checkpoint is:

`logs/rnn_car/sa6_k8_obb_sa5replay10_recovery_c10_2u_s42/checkpoint_128.pt`

Its fixed retention gates were:

- Gate5 1.2 m: crossing 99.20%, SR 98.0%, CR 1.2%, TO 0.8%.
- 4 m x 10 m, 4S+2D corridor: SR 77.99%, CR 22.01%, TO 0%.
- Gate2 per-seed CR: 9.3%, 10.3%, and 9.3%.

Therefore 10% SA5 replay supplies a real but insufficient correction. One
update improved pooled Gate2 CR by 0.825 percentage points; the next update
reversed most of it. More uninterrupted updates are not justified.

## Dual-Teacher Retention Result

A second frozen policy was added without replacing the narrow c20 teacher:

`L = L_PPO + 0.30 KL(c20 || policy)|narrow + beta5 KL(c12 || policy)|SA5-replay`

The two masks are disjoint. Both teachers have independent observation
normalizers and K8 histories. Ordinary SA6 and corridor frames remain pure
PPO.

A 64-env runtime calibration found that the c12 disagreement was much larger
than the c20 disagreement:

- c20 narrow KL: 0.0036 linear + 0.0075 angular.
- c12 SA5-replay KL: 0.0638 linear + 0.1386 angular.
- At beta5=0.30 the c12 weighted term was 0.0607, about sixteen times the
  observed PPO policy loss. Beta 0.30 was therefore rejected before formal
  training as an effective freeze.

All formal points below started from replay-c128 with the same Adam state and
ran exactly one 1024-env update:

| beta5 | Gate2 SR / CR / TO | Gate2 per-seed CR | Gate5 | 4S+2D corridor | Decision |
|---:|---:|---:|---:|---:|---|
| 0 (replay-c128) | 90.37 / 9.63 / 0% | 9.3 / 10.3 / 9.3% | crossing 99.20% | SR 77.99% | Current Pareto |
| 0.02 | 90.37 / 9.53 / 0.10% | 8.8 / 10.3 / 9.5% | Not run | Not run | Reject: no hard-gate pass |
| 0.05 | 90.27 / 9.63 / 0.03% | 9.3 / 9.9 / 9.7% | Not run | Not run | Reject: no improvement |
| 0.10 | 90.50 / 9.47 / 0% | 9.3 / 9.7 / 9.4% | crossing 84.87%, TO 14.5% | SR 78.65% | Reject: narrow collapse |

The response is not a stable monotonic trade. Increasing the c12 anchor
slightly improves the worst Gate2 seed but updates shared CNN/policy
parameters and eventually damages narrow approach behavior despite the
disjoint scene masks and the c20 anchor. Do not increase beta5 further and do
not continue any dual-retention branch.

Freeze replay-c128 as the current SA6 Pareto fallback. It is not graduated:
Gate2 remains 0.63 percentage points above its collision threshold and the
deployment corridor remains about 12 percentage points below its SR target.
The dominant capability gap is now the 4S+2D corridor, not narrow retention.

## Observation-Gated Residual Adapter

The corridor scene itself is observable without privileged inputs. A held-out
probe trained on the deployment observation achieved 95.43% balanced accuracy
using the current normalized 83D policy observation. This justified a
deployable scene-gated residual policy:

- the c12 K8 CNN and policy head are structurally frozen;
- an 83D MLP gate is supervised by the corridor replay mask;
- PPO updates only a zero-initialized 38-logit residual adapter;
- the gate probability is detached from PPO, so PPO cannot open the gate;
- the final policy is exactly c12 before the first adapter update.

The first formal 1024-env update used an 83D residual input. Runtime checks
passed: gate recall/specificity were 95.0%/94.7%, residual L2 was 0.204 on
corridor frames versus 0.014 elsewhere, and the base policy/CNN gradients were
exactly zero. Its 3000-step deterministic 2S+1D gate nevertheless failed:

- 3832 episodes, SR 84.79%, CR 15.21%, TO 0%;
- 72% of obstacle collisions were with the moving patrol obstacle;
- versus frozen c12 on identical frames, linear/angular argmax agreement was
  95.4%/90.6%;
- brake rate was 11.2% versus 11.0%, and strong-turn rate was 24.6% versus
  24.7%.

Because the residual saw only the current 83D observation, a second ablation
fed it the full deployable 179D policy feature vector: 83D current observation
plus the frozen 8-frame LiDAR CNN's 96D embedding. The gate remained 83D. This
also passed all runtime checks, but its matched 1200-step gate was worse:

- 1491 episodes, SR 82.90%, CR 17.10%, TO 0%;
- 70% of obstacle collisions were dynamic;
- linear/angular argmax agreement with c12 was 96.4%/87.6%;
- brake rate was 12.3% versus 12.1%, and strong-turn rate was 26.6% versus
  26.5%.

Therefore both PPO-only adapter variants are rejected. They change low-margin
discrete bins but do not change the action class needed for dynamic avoidance.
Do not continue either adapter with more PPO updates or run Gate2/Gate5 for
them. The next justified adapter experiment is privileged-teacher
distillation into the isolated residual, because the v4 teacher passes the
same 2S+1D corridor and a held-out 179D probe predicts its speed intent and
turn direction materially above their majority baselines.

## Narrow c20 Retention Decision

The proposed narrow-only objective was implemented and calibrated with the
production 1024-env batch:

`L = L_PPO + beta * KL(policy_c20 || policy_current)|narrow`

The prerequisite c20-versus-c30 diagnostic used 76,800 identical Gate5
frames generated by c20. The frozen checkpoints were:

- c20: `sa5_e2e_k8_obb_narrow_recovery_seg2_c1280_s42/checkpoint_1280.pt`
- c30: `sa5_e2e_k8_obb_narrow_recovery_seg3_c2560_s42/checkpoint_1280.pt`

The comparison used `KL(policy_c20 || policy_c30)` independently for the
linear and angular categorical heads:

| Region | Frames | KL linear / angular | Argmax agreement linear / angular | c20 / c30 brake | c20 / c30 strong turn |
|---|---:|---:|---:|---:|---:|
| All | 76,800 | 0.1032 / 0.1020 | 70.2% / 64.0% | 23.9% / 28.3% | 40.8% / 44.2% |
| Approach | 64,474 | 0.1062 / 0.1121 | 69.1% / 67.5% | 28.4% / 33.5% | 48.1% / 50.3% |
| Throat | 7,038 | 0.1493 / 0.0802 | 86.4% / 39.2% | 0.5% / 1.3% | 4.7% / 21.8% |
| Exit | 5,288 | 0.0048 / 0.0072 | 62.7% / 54.6% | 0.0% / 0.0% | 0.1% / 0.1% |

c20 crossed on 97.08% of its 718 sampled trajectories. The decisive c30
regression is concentrated at the throat: angular agreement falls to 39.2%
while strong-turn actions rise from 4.7% to 21.8%. The failure is therefore
policy drift in the successful narrow behavior, not insufficient Gate5
training exposure or an OBB geometry failure.

This is a forward categorical KL on both action heads and applies only to the
mutually exclusive narrow-replay mask. Ordinary, SA5-general, and corridor
frames remain free to follow PPO. The measured beta sweep established:

- beta 0.10 and 0.20 did not reliably retain the 95% Gate5 crossing gate;
- beta 0.30 retained Gate5 at 100% while preserving the best measured Gate2
  trade-off;
- beta 0.50 did not improve the corridor or retention result over beta 0.30.

At the short calibration horizon, c20 therefore remained frozen as the narrow
teacher and beta 0.30 was selected for SA6-SA8. The long-continuation result
below supersedes any claim that beta 0.30 is sufficient over a long training
horizon: Gate5 eventually collapsed even while the logged average KL remained
small. Keep c20 frozen, but do not rely on the joint PPO KL alone as the final
retention mechanism.

## Privileged-Teacher Distillation Result

The v4 privileged teacher was captured on the exact 4 m x 10 m 2S+1D
corridor, using c12's observation history and seed 515:

- 160 closed-loop episodes: SR 95.625%, CR 4.375%, TO 0%;
- teacher brake 29.747%, strong turn 0.178%;
- 99.24% of frames had at least one feasible candidate action.

The aligned 179D probe confirmed that teacher intent is partially recoverable
from deployment observations on teacher-visited states. Held-out-env balanced
accuracy was 90.06% for linear intent and 79.81% for angular direction.

An isolated residual adapter was then fitted offline with the c12 base, K8
encoder, and corridor gate frozen. On held-out teacher-trajectory frames it
reached:

- linear/angular exact accuracy 79.03% / 84.18%;
- joint exact accuracy 67.48%, joint within-one-bin 89.20%;
- brake 27.33% versus teacher 28.97%;
- strong turn 0.84% versus teacher 0.08%.

Checkpoint:

`logs/rnn_car/sa6_c12_corridor_adapter_teacher_v4_s515/checkpoint_offline.pt`

This apparently strong offline result failed the required out-of-seed
closed-loop test. On seed 616, 64 envs x 1200 steps:

- 1,210 episodes: SR 60.99%, CR 36.12%, TO 2.89%;
- obstacle CR 35.79%, wall CR 0.33%;
- brake 27.71%, coast 49.89%, strong turn 1.75%;
- the frozen c12 shadow policy on the same student-visited frames had brake
  11.23%, coast 0.49%, and strong turn 28.19%.

The failure is behavioral-cloning covariate shift. The adapter was trained on
states visited by the privileged teacher; once its small action errors move
the robot away from those trajectories, it receives observations outside the
training distribution and compounds those errors. Offline action agreement
does not imply closed-loop navigation performance.

Do not tune more epochs or a larger logit delta on the same dataset. The next
necessary test is DAgger-style aggregation: execute the student policy, query
the privileged teacher only for labels on student-visited states, aggregate
those frames with the original teacher dataset, retrain, and judge first on
an unseen-seed closed-loop corridor gate. If this fails, the remaining issue
is not ordinary supervised coverage; investigate observation sufficiency or
move the privileged planner into an explicit deployment-time control layer.

## Training-Volume Decision

A teacher-shadow mode was implemented and verified so that the policy, not
the teacher, controls the simulator while privileged labels are recorded.
Two 16-env x 600-step round-1 datasets were captured:

- seed 515: student SR 58.19%, CR 41.24%, TO 0.56%; teacher feasible on
  84.83% of student-visited frames;
- seed 616: student SR 67.13%, CR 31.47%, TO 1.40%; teacher feasible on
  86.34% of student-visited frames.

These datasets confirm the covariate-shift diagnosis and are retained, but
the DAgger branch is paused before fitting. The adapter family failed its
required first closed-loop window, while ordinary SA6 PPO had already shown
a large corridor improvement over ten uninterrupted updates. The next
decisive experiment therefore tests training volume without another method
change.

The fixed long-continuation config is:

`e2e_sa6_k8_obb_long_cont300`

It starts from replay-c128 with its Adam state and changes only the training
budget:

- 300 PPO iterations, 1024 envs, rollout length 128;
- checkpoints every 10 iterations;
- 68% native SA6, 10% SA5-general, 12% narrow, 10% exact 4S+2D corridor;
- c20 narrow-only forward KL beta 0.30;
- unchanged K8 CNN, OBB, future 0.10, anti-spin 0.15, reward, LR, and horizon;
- no residual adapter and no second teacher.

### Fixed corridor-goal reset semantics

Before the long continuation, the deployment-corridor evaluator exposed a
reset-order bug. Isaac Lab runs reset events before `command_manager.reset()`.
The corridor event therefore wrote the far-end goal first, after which
`MultiGoalCommand.reset()` replaced it with a random goal. The
`no_goal_movement` flag only disabled later movement and could not prevent
this reset-time overwrite. The interval event usually restored the policy goal
after the first step, but the initial policy observation was wrong and the GUI
marker could remain stale.

Fixed-scene environments now bypass random command resampling and reassert the
same goal into `goal_pos_w`, `all_goals_pos_w`, `_local_goal_world`, and the
visual marker. Training audits command and local-goal alignment every frame
and fails immediately above `1e-5 m`.

Verification on the corrected evaluator:

- 34 fixed-goal, corridor-geometry, and replay-config tests passed;
- a 4-env headless run completed 21 episodes with command and local-goal
  maximum errors both exactly `0.0 m`;
- the 1024-env production smoke and the formal run both reported initial
  alignment errors of `0.0 m`;
- corridor geometry, 4S+2D allocation, and dynamic-obstacle movement all
  passed.

Pre-fix corridor results are useful only as exploratory evidence. They are not
formal baselines for the long-continuation A/B because at least the initial
observation and visual marker had different semantics. Formal comparisons
must rerun both checkpoints under the corrected evaluator with fixed seeds.

Use the 2S+1D corridor as an intermediate learning-curve diagnostic. The
4S+2D corridor remains the final deployment gate, but its SA8-like density
must not turn every early SA6 checkpoint into a reason to change recipes.

## Corrected Long-Continuation and Projection Result

All numbers in this section use the corrected fixed-goal evaluator. Corridor
geometry, obstacle movement, and goal alignment passed in every run; both
`goal_command_max_error_m` and `local_goal_max_error_m` were exactly `0.0`.

The formal long continuation produced this fixed-seed curve:

| Checkpoint | Gate2 SR / CR / TO | Gate5 SR / CR / TO / crossing | 4S+2D corridor SR / CR | 2S+1D corridor SR / CR |
|---|---:|---:|---:|---:|
| replay-c128 baseline | 90.37 / 9.63 / 0.00 | 98.0 / 1.2 / 0.8 / 99.20 | 77.57 / 22.43 | 90.53 / 9.47 |
| long c40 | 89.43 / 10.53 / 0.00 | 99.8 / 0.2 / 0.0 / 99.85 | 75.47 / 24.53 | 89.15 / 10.85 |
| long c50 | 90.63 / 9.33 / 0.03 | 73.5 / 9.6 / 16.9 / 78.81 | 86.80 / 13.20 | 94.40 / 5.60 |
| long c60 | not rerun | 0.0 / 0.5 / 99.5 / 0.0 | not rerun | not rerun |

The long run was stopped at c60. c50 had learned a real corridor skill, but
the same update window destroyed the narrow skill; c60 then froze completely
in Gate5. Training-time SR, KL, and teacher agreement remained apparently
healthy at c60, so they cannot replace deterministic closed-loop Gate5.

This rejects the hypothesis that narrow-only beta 0.30 is a hard retention
constraint. It is only an average on-policy regularizer. Critical throat
states and deterministic argmax flips can be diluted by many low-KL frames.

### Projection-only causal tests

Starting from long c50, three projection-only tests used one 1024-env rollout,
no PPO/value update, the same frozen c20 teacher, forward KL, and an argmax
margin. The only variables were which student parameters could update and the
number of full-batch SGD steps:

| Projection | Trainable parameters | Steps | Gate5 SR / CR / TO | Crossing |
|---|---|---:|---:|---:|
| head-only | policy head | 4 | 77.3 / 9.1 / 13.6 | 79.15 |
| full actor | LiDAR CNN + policy head | 4 | 92.7 / 6.2 / 1.1 | 93.11 |
| full actor | LiDAR CNN + policy head | 8 | 98.4 / 0.9 / 0.7 | 98.45 |

Head-only projection barely moved the closed-loop result despite linear and
angular teacher argmax agreement of 93.6% and 87.3% on sampled narrow frames.
Allowing gradients through the LiDAR CNN produced the decisive recovery.
Narrow behavior is therefore partly encoded in the K8 LiDAR representation,
not only in the policy head.

The accepted eight-step projection was then evaluated outside Gate5:

- Gate2, three seeds and 7,897 episodes: SR 90.006%, CR 9.928%, TO 0.066%;
- 4S+2D corridor, 1,340 episodes: SR 85.149%, CR 14.851%, TO 0%;
- 2S+1D corridor, 1,540 episodes: SR 94.805%, CR 5.195%, TO 0%.

It passes Gate5 and the intermediate 2S+1D corridor, and retains most of c50's
4S+2D gain, but it misses the Gate2 CR and final 4S+2D corridor gates. This is
reverse forgetting in the other direction: repairing the shared encoder for
c20 shifts some c50 behavior.

The next justified experiment is a dual functional anchor on the same
student-visited rollout:

`L_project = L_narrow(c20, student) + lambda * L_non_narrow(c50, student)`

The c20 term repairs narrow frames while the frozen pre-projection c50 policy
anchors ordinary and corridor frames. This differs from the rejected
dual-retention PPO probes: it is a post-update supervised projection with no
reward or PPO gradient mixed into the causal test. Do not resume the stopped
long continuation, increase beta blindly, or add more single-teacher
projection steps; those directions are already bounded by the results above.

### Dual projection result and revised sequence

The dual projection was implemented through the full actor, with disjoint
c20 narrow and c50 anchor masks. The loss averages each domain independently,
so the larger anchor set cannot dominate merely through frame count.

| c50 anchor | Gate2 SR / CR / TO | Gate5 SR / CR / TO / crossing | 4S+2D corridor SR / CR | 2S+1D corridor SR / CR |
|---|---:|---:|---:|---:|
| non-narrow, lambda 0.5 | 90.60 / 9.33 / 0.07 | 98.8 / 0.6 / 0.6 / 99.40 | 84.02 / 15.98 | 95.32 / 4.68 |
| non-narrow, lambda 1.0 | not run | 93.2 / 4.1 / 2.7 / 94.80 | 85.71 / 14.29 | not run |
| corridor only, lambda 0.5 | not run | 93.1 / 5.1 / 1.8 / 95.39 | 85.69 / 14.31 | not run |

Lambda 0.5 successfully merged c20 narrow behavior with c50 ordinary and
intermediate-corridor behavior. It did not preserve the dense 4S+2D closed
loop. Increasing lambda or making the corridor an independent anchor mean
improved dense-corridor CR only to about 14.3% and crossed the Gate5 failure
boundary. This is a real Pareto tradeoff, not an optimizer or mask bug.

Projection can recover a skill already present in a teacher, but it cannot
push dense-corridor CR below the unprojected teacher. The missing step is to
create a stronger corridor teacher through continued PPO training, then use
the proven full-actor projection to repair narrow behavior.

The previously unevaluated unprojected c60 checkpoint confirmed this
sequence. Under the corrected fixed-goal 4S+2D evaluator it achieved
SR 88.96%, CR 11.04%, and TO 0%, improving substantially over c50 CR 13.20%.
Its Gate5 behavior is collapsed, but that is no longer a reason to stop
corridor training because full-actor projection has already demonstrated
post-training narrow recovery.

Revised sequence:

1. Continue the exact SA6 recipe from c60 with Adam state preserved.
2. Save every ten updates and select the first fixed-seed 4S+2D checkpoint
   with CR at or below 10%.
3. Apply the full-actor c20 projection to that stronger corridor teacher.
4. Run Gate2, Gate5, 4S+2D, and 2S+1D jointly before accepting the checkpoint.

This section supersedes the earlier instruction not to resume long
continuation. That instruction was valid before dual projection and the c60
corridor measurement established the two-stage recovery sequence.

## C105 Joint-Pass Result

The long-training-first decision was completed without changing the SA6
recipe. Starting from c60, the corrected fixed-goal 4S+2D corridor curve was:

| Checkpoint | Corridor SR | Corridor CR | Gate5 crossing |
|---|---:|---:|---:|
| c70 | 86.80% | 13.20% | not run |
| c80 | 87.28% | 12.72% | not run |
| c90 | 90.83% | 9.17% | 48.02% |
| c100 | 91.99% | 8.01% | 91.02% |
| c105 | 91.59% | 8.41% | 100.00% |

The narrow result was strongly non-monotonic. The first corridor-passing
checkpoint, c90, had severely collapsed Gate5 behavior. Ten more ordinary PPO
updates restored most of it at c100, and five additional updates produced a
clean joint solution at c105. This directly rejects checkpoint selection from
training SR, average retention KL, or corridor CR alone.

Projection was not needed for the accepted checkpoint. Fixed-seed Gate5
crossing from c100 was:

| c100 variant | Gate5 SR / CR / TO | Crossing | Decision |
|---|---:|---:|---|
| unprojected | 90.8 / 4.0 / 5.2 | 91.02% | baseline |
| dual c20+c100, anchor 0.75 | 93.5 / 2.2 / 4.3 | 93.69% | fail |
| second dual cycle | 88.2 / 2.3 / 9.6 | 89.17% | fail |
| c20-only full actor | 86.1 / 3.4 / 10.5 | 87.05% | fail |
| c20 teacher-forced rollout | 75.4 / 5.1 / 19.5 | 76.05% | fail |
| dual c20+c100, anchor 0.50 | 86.6 / 4.5 / 8.9 | 87.47% | fail |

Teacher forcing failed because it trained only on teacher-visited states. The
student then left that trajectory during deterministic deployment and had no
correction data for its own off-trajectory states. Do not continue teacher
forcing or projection-lambda sweeps from c100.

The accepted c105 checkpoint is:

`logs/rnn_car/sa6_k8_obb_long_c100_to_c110_s42/checkpoint_640.pt`

SHA-256:

`c6a20524e5e4c300eb3b55d0df0a9fbee61970f3f6af51880e594e6b96a656f2`

Formal fixed-seed acceptance:

- Gate2, seeds 101/202/303, 7,717 episodes:
  SR 91.766%, CR 8.201%, TO 0.033% -- PASS.
- Gate5 1.2 m, seed 404, 718 episodes:
  SR 100%, CR 0%, TO 0%, crossing 100% -- PASS.
- 4 m x 10 m, 4S+2D corridor, seed 515, 1,308 episodes:
  SR 91.590%, CR 8.410%, TO 0% -- PASS.
- Corridor geometry, fixed-goal alignment, constructive solvability, obstacle
  allocation, and dynamic-obstacle movement all passed.

The later c110 checkpoint exists only because c105 could not be evaluated
without stopping the training process. It is not the accepted checkpoint and
must not replace c105 without a complete independent gate suite.
