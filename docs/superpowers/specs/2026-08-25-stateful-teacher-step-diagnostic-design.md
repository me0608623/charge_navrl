# Stateful teacher per-step diagnostic instrumentation

Date: 2026-08-25
Status: complete, r4 valid diagnostic evidence
Scope: record-only instrumentation + one re-run of the frozen stateful screen

## Problem

The r3 screen (`logs/gates/teacher_2s2d_stateful_d1_screen/screen_20260824_r3/`)
ruled the stateful `WAIT → COMMIT_SIDE → PASS` teacher a FAIL
(SR 17.97% / 22.01%, CR ≈ 53%, TO 24–29%) but could not explain it. Per-episode
log parsing narrowed the failure to two disjoint buckets:

| bucket | share of episodes | share of frames | behaviour |
|---|---:|---:|---|
| obstacle collision | 52% | 24% | dies early (p50 74 steps) while moving at 0.20 m/s, 93% forward |
| timeout | 29% | **63%** | 100% of them reached ≥ 0.15 m/s at some point, then travelled ~1 m total |
| goal | 18% | 12% | ~every one reaches the 0.675 m/s speed cap; a clear-corridor sprint |

Three questions remained unanswerable from the recorded data:

1. Where does a frozen robot end up laterally? (no lateral coordinate was recorded)
2. When the committed side has no path, is the *other* side open?
3. How large is the pedestrian prediction residual just before a collision?

Question 2 is the decisive one. `committed_valid == False` alone cannot separate
a **commitment-rule failure** (other side open, the no-switch rule forbids using
it) from a **geometry failure** (both sides blocked, the 0.20 m effective margin
is too thin). The two have opposite fixes.

## Design

### Guarantee: no behaviour change

Three independent defences, in order of strength:

1. **Opt-in flag.** `--stateful_teacher_diagnostic_output` defaults to `""`.
   When empty, no new code executes.
2. **Return-only FSM change.** `committed_valid`, `left_valid`, `right_valid`
   are already computed inside `StatefulCorridorTeacher.select()`
   (`stateful_corridor_teacher.py` L367-368, L573-575). They are added to the
   return dict. No new computation, no RNG consumption.
3. **Regression check.** r4 must reproduce r3's `episodes / SR / CR / TO`
   exactly, since `deterministic=True`, `seed=818` and the protocol hash are
   unchanged. Any difference invalidates the run.

The protocol payload is untouched, so the frozen protocol SHA-256 stays
`0d608c1e61111b339847e3fa68f9563dae10eb8e86107274f24c8634d196af84` — verified,
identical to r3. Only the source fingerprint changes; the runner captures the
fingerprint at start and checks for drift *during* the run, so it does not
compare against r3 and will not block.

### Recorded fields

Corridor frame: local **x is lateral** (walls at ±`free_width/2` = ±2.10 m),
local **y is longitudinal**. Confirmed empirically in the smoke run (3.3 m of y
travel vs 1.2 m of x over 50 steps) and from
`long_corridor_replay_geometry.inner_half_width`.

Arrays are stacked `[rollout_step, env]` so one episode is a contiguous run
within an env column.

| field | dtype | answers |
|---|---|---|
| `lateral_m`, `longitudinal_m` | f32 | Q1 |
| `state`, `committed_side` | i8 | segmentation |
| `committed_valid`, `left_valid`, `right_valid` | bool | **Q2** |
| `used_wait`, `emergency_brake`, `interaction_active` | bool | cross-check vs r3 ledger |
| `applied_v_mps`, `applied_omega_rps` | f32 | cross-check vs log |
| `pred_err_1step_m`, `pred_err_5step_m` | f32 `[T,E,slot]` | **Q3** |
| `episode_step` | i16 | segmentation |
| `episode_end` | i32 `[M,4]` | `(rollout_step, env_id, cause, episode_steps)` |

Prediction error uses the **pause=0 branch** (`_teacher_obstacle_paths_moving`),
which is the teacher's point prediction. The pause=5 branch only widens the
blocked set for feasibility and is not a prediction.

Rows where the episode reset inside the lag window are **NaN, not 0** — a reset
teleports the pedestrian and a zero would silently drag the mean down.

Total ≈ 19 MB per cell.

## Verification performed

- TDD: two RED tests (`test_select_reports_left_and_right_side_feasibility`,
  `test_committed_valid_exposes_no_switch_deadlock`) failed with
  `KeyError: 'committed_valid'`, then passed after the FSM change. The second
  test encodes the deadlock signature directly: committed left, left candidate
  removed, right still open → `committed_valid False`, `right_valid True`,
  `used_wait True`.
- 41 teacher-related tests pass.
- Protocol SHA-256 verified identical to r3.
- Smoke run (4 envs × 50 steps): npz written with correct keys, shapes, dtypes;
  NaN masking behaves as designed; analysis script runs end to end.

## r4 result

Output:

`logs/gates/teacher_2s2d_stateful_d1_screen/screen_20260825_r4/`

The regression gate passed before reading the new fields. For both
`lateral_lateral` and `mixed`, the r4 `teacher.json`, `corridor.json`, and
`cell.json` are byte-identical to r3. The protocol SHA-256 remains
`0d608c1e61111b339847e3fa68f9563dae10eb8e86107274f24c8634d196af84`, and the
r4 source fingerprint is stable from suite start to suite end.

Both archives have shape `4000 x 64`; their 1,436 and 1,549 episode-end rows
respectively join to the terminal action row with no off-by-one or reset error.

| scenario | committed-invalid frames | opposite side open | both sides blocked |
|---|---:|---:|---:|
| lateral+lateral | 157,272 | 17,083 (10.86%) | 140,189 (89.14%) |
| mixed | 107,627 | 10,533 (9.79%) | 97,094 (90.21%) |

The no-switch deadlock exists, but it is not the dominant recorded mode. Most
committed-invalid frames have no left or right passage candidate under the
frozen 19x19 teacher model. This does **not** mean those physical scenes are
unavoidable; horizon, discretization, kinematic passage filters, obstacle
clearance, and wall clearance are still combined in this observation.

Timeouts generally finish stopped near the corridor center. The median
absolute lateral position is 0.370 m in `lateral_lateral` and 0.388 m in
`mixed`; the median applied speed is 0 m/s. They are not predominantly robots
wedged against a wall.

The 1.0 s point-prediction residual p95 is 0.174 m / 0.177 m over all dynamic
slot samples and 0.117 m / 0.129 m at obstacle-collision terminal rows. No
general collision-linked residual increase is observed. The archive does not
identify which obstacle caused contact, so this is not a per-contact predictor
accuracy claim.

Machine-readable and rendered analyses:

- `STATEFUL_STEP_DIAGNOSTIC.json`
- `STATEFUL_STEP_DIAGNOSTIC.md`
- `scripts/reinforcement_learning/skrl/rnn_car_wdclean/analyze_stateful_teacher_step_diagnostic.py`

Decision: the r3 teacher FAIL remains unchanged. Training, distillation, and
SA6 remain unauthorized. The next diagnostic should decompose the passage
candidate mask into kinematic/progress, obstacle-collision, and wall-collision
loss before changing FSM rules, clearance, horizon, reward, or training.

## Known unrelated issue

Collecting the whole `rnn_car_wdclean` package with pytest fails on a
pre-existing `SA5-R2 sealed source drift` guard in
`rnn_car_modular/configs/e2e_sa5_r2_sealed_from_sa4r3_it125_actdelay12.py`.
The working tree was clean before any edit, so this is not caused by this
change. Workaround: run pytest from inside the package directory with
`PYTHONPATH` pointing at `scripts/reinforcement_learning/skrl`.

## Out of scope

No teacher logic change, no threshold change, no training, no distillation.
r4 is a diagnostic re-run; the r3 FAIL verdict stands regardless of outcome.
