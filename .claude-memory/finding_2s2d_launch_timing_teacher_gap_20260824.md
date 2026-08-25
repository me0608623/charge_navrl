# 2S2D launch timing and privileged-teacher result (2026-08-24/25)

## Observation

- GUI run: `logs/gui_capture/corridor_2s2d_gui_nogizmo_20260824.log`.
- Checkpoint: `logs/rnn_car/sa5_v3_c500_parent_control_from_sa4v3_ne1024_s42_p50_r1/checkpoint_6400.pt`.
- Fixed long corridor, 2S2D, `mixed`, pedestrian speed 0.25-0.45 m/s, `speed_rate=0.7`, full VLP-16 noise with `valid_return_only`, fixed `d1=200 ms`.
- User observed repeated hesitation when a lateral pedestrian crosses left-to-right and then reverses. The desired human-like behavior is to use the side just vacated by the pedestrian when that side is jointly safe, then commit long enough to pass.
- Fixed-scene goals are restored every step by `goal_movement.py`; this observation is not explained by goal drift.

## What the current teacher can see

- `privileged_corridor_teacher.py` reads exact robot, obstacle, patrol waypoint, speed, pause, and wall state.
- `predict_patrol_obstacle_paths()` explicitly models patrol waypoint transitions/reversals.
- It filters a reachable 19x19 action grid against dynamic obstacles and wall geometry.
- Existing formal teacher-only closed-loop evidence is limited to 2S1D lateral with actuator DR off: historical R=1 pooled across three evaluator seeds gave SR 95.910%, CR 4.090%, TO 0%. This cannot be extrapolated to 2S2D+d1.

## Why the pre-diagnostic teacher was not yet a solution

1. The teacher is a per-frame 2 s argmin with no explicit `WAIT -> COMMIT_LEFT/RIGHT -> PASS` state.
2. Robot candidates are single constant `(v, omega)` paths, not wait-then-launch action sequences.
3. A lateral patrol spans 2.6 m and takes roughly 5.8-10.4 s at 0.25-0.45 m/s, longer than the 2 s planning horizon.
4. If all 361 actions are infeasible, override falls back to the original policy and no valid teacher label exists for that frame.
5. Before this work, `play_rnn_car.py` rejected teacher mode when actuator DR was enabled because the teacher did not model the delay queue. Those older teacher-only tests cannot be accepted as exact d1 evidence.
6. The GUI checkpoint and current v3/B3 lineage use `corridor_teacher_distill_epochs=0`; the observed behavior is PPO behavior, not a teacher failure.
7. The GUI checkpoint training mix covers 0S1D, 1S1D, 2S1D, 3S2D, and 4S2D, but not exact 2S2D. Missing-combination exposure is a hypothesis, not an established sole cause.
8. In legacy `mixed`, the two dynamic obstacles are commonly assigned different motion families for one-env resets. "Go left after a left-to-right pedestrian" is valid only if the left side is also feasible for the second pedestrian, static obstacles, and walls.
9. Old 2S1D learnability data show the linear go/wait label is the difficult student-observation channel: mutual information fell about 37% on student trajectories and linear aliasing rose from 5.5% to about 21-27%. Teacher closed-loop success would not by itself prove K8 distillability.

## Implemented diagnostic

- `privileged_corridor_teacher.py` now predicts candidate paths through the exact fixed-d1 queue: the pending command executes first, then the candidate command.
- Teacher override is inserted before `process_actions`; the normal decode -> d1 queue -> simulator path remains intact.
- `teacher_interaction_metrics.py` records safe-gap launch latency, vacated-side choice, stop/go, post-launch side switches, bounded reverse, U-turn, and cumulative 360-degree rotation with d1-aligned labels.
- Frozen memoryless protocol: `5608b21a50929cf76044905bf72796aacd8e947310079527d29ae17ae25c84f8`.
- Frozen stateful protocol: `0d608c1e61111b339847e3fa68f9563dae10eb8e86107274f24c8634d196af84`.
- Checkpoint SHA-256: `4bc1744bb134688179ad2dfdc858bec245194d205ce61570f94bd2a0d726a99c`.

## Memoryless formal result

Output: `logs/gates/teacher_2s2d_d1_screen/screen_20260824_r1/`.

- lateral+lateral: n=2042, SR=62.10%, CR=37.90%, TO=0%, vacated-side=9.09%, side-switch p95=3, stop/go p95=6.
- mixed: n=1681, SR=46.34%, CR=49.32%, TO=4.34%, vacated-side=3.23%, side-switch p95=3, stop/go p95=6, full-rotation=0.77%.
- Verdict: FAIL; redesign as WAIT -> COMMIT_SIDE -> PASS.

## Stateful implementation and invalid development runs

- The FSM waits near an interaction, requires two consecutive feasible-side confirmations, prefers the recently vacated side, forbids side switching after commit, caps reverse at 0.10 m/s and 0.30 m, and limits passage heading to 90 degrees.
- `screen_20260824_r1` is `INCOMPLETE_NO_VERDICT`: fixed launch thresholds were unreachable from rest; 192 episodes all timed out.
- `screen_20260824_r2` is `INCOMPLETE_NO_VERDICT`: 2500 steps produced 876 episodes, below the preregistered minimum 1000.
- Neither incomplete run may be cited as a teacher verdict.

## Stateful formal result

Output: `logs/gates/teacher_2s2d_stateful_d1_screen/screen_20260824_r3/`.
Status: `COMPLETE_VALID_DIAGNOSTIC_EVIDENCE`; source fingerprint stable; shared-GPU timing excluded.

- lateral+lateral: n=1436, SR=17.97%, CR=53.06%, TO=28.97%, obstacle CR=52.44%, wall CR=0.63%.
- mixed: n=1549, SR=22.01%, CR=53.52%, TO=24.47%, obstacle CR=52.49%, wall CR=1.03%.
- FSM committed-side match reached 99.87% / 99.89%; launch p95=0.4 s; side-switch p95=1 in both cells. This uses the d1-aligned FSM side hint, whereas memoryless uses its selected endpoint, so the two percentages are not an identical estimator and 99.9% does not prove successful physical passage.
- U-turn fractions were 0.07% / 0.19%; full-rotation fractions were 0% / 0.13%. The undesirable looping symptom was strongly suppressed.
- This did not produce safe navigation. Wait actions occupied 73.06% / 70.87% of frames; episode wait p95 was 70.8 / 73.36 s; collisions remained about 53%.
- FSM state fractions were WAIT/COMMIT/PASS=13.68/7.32/79.00% lateral and 39.02/8.86/52.12% mixed. A likely mechanism is that a committed side often loses the constant-action 2 s feasible path, causing repeated waiting while in COMMIT/PASS. This is a diagnostic hypothesis, not a proven sole cause.

## Decision

`teacher_pass=False`, `next_step=HOLD_AND_ANALYZE_STATEFUL_TEACHER_FAILURE`.

- Do not run K8 label-inferability yet: a failed teacher is not a valid label source.
- Do not distill, start SA6, or authorize new training from this result.
- Do not increase distillation weight. First stratify collisions by FSM state and determine whether a multi-stage spatiotemporal passage planner is required instead of another one-step 19x19 selector adjustment.

Obsidian note: `/home/aa/Documents/Obsidian Vault/isaaclab_v4/Gate結果/59_2S2D折返行人的出發時機與特權教師適用性_20260824.md`

## r4 per-step mechanism diagnostic (2026-08-25)

Output: `logs/gates/teacher_2s2d_stateful_d1_screen/screen_20260825_r4/`.

- r4 is record-only. For both cells, `teacher.json`, `corridor.json`, and `cell.json` are byte-identical to r3. Protocol SHA remains `0d608c1e...`; source fingerprint is stable.
- NPZ accounting is valid: shapes are 4000x64, episode-end counts are 1436/1549, terminal rows align to episode_step=n-1, and the next row resets to zero.
- Timeout terminal median abs(lateral x) is 0.370 m / 0.388 m and median applied speed is 0. The robot usually stops near the corridor center rather than against a wall.
- Among committed-invalid frames, opposite-side-open/no-switch accounts for 10.86% / 9.79%; both-sides-without-passage accounts for 89.14% / 90.21% (lateral/mixed). No-switch deadlock exists but is secondary.
- Lateral obstacle collisions: 753/753 episodes experienced both-sides-blocked and 747/753 terminated there. Mixed: 807/813 experienced it and 765/813 terminated there.
- The 1.0 s prediction residual p95 is 0.174/0.177 m overall and 0.117/0.129 m on obstacle-collision terminal slot samples. There is no broad collision-linked residual increase.
- Critical limitation: both-sides-blocked is a committed-invalid frame-level property of the frozen 19x19/2 s teacher model. It is not a physical inevitability or collision proportion. Prediction samples are not matched to the contacting obstacle.
- Next: record-only decomposition of passage kinematics/progress/heading, obstacle collision, and wall collision masks under the same frozen two cells. Do not alter clearance, horizon, FSM, reward, training, distillation, or SA6 before this split.
- Durable analyzer: `scripts/reinforcement_learning/skrl/rnn_car_wdclean/analyze_stateful_teacher_step_diagnostic.py`; report JSON/Markdown are in the r4 output directory.
