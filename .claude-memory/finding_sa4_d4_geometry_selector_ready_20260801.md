---
name: SA4-D4 geometry-feasible 19x19 selector implementation and completed comparison
description: Delay-aware pedestrian plus LiDAR static/wall geometry selector, frozen protocol, and completed fixed comparison with no observed improvement
type: design
date: 2026-08-01
updated: 2026-08-01
status: complete-not-passed
---

# SA4-D4 geometry-feasible 19x19 selector

## Authority and current state

- Selector, play integration, recorder compatibility, protocol freeze, and
  fixed comparison runner are implemented.
- The Isaac Sim/GPU comparison completed as valid privileged diagnostic
  evidence. Geometry-feasible showed no observed improvement and both arms
  failed the SA4 Gate.
- `SA4 HOLD`, `accepted_parent=null`, and `SA5_NOT_AUTHORIZED` remain in force.
- Design note:
  `/home/aa/Documents/Obsidian Vault/isaaclab_v4/Gate結果/22_SA4_D4_GeometryFeasibleSelector設計_20260801.md`
- Result note:
  `/home/aa/Documents/Obsidian Vault/isaaclab_v4/Gate結果/23_SA4_D4_GeometryFeasible固定比較結果_20260801.md`

The historical filename contains `ready`, but this file now records the
completed result. Use
`finding_sa4_d4_geometry_selector_result_20260801.md` for the concise verdict.

## Why this exists

D3 best-turn reduced dynamic CR from 15.79% to 2.28% but moved failures to
static/wall CR (57.53%/25.39%). D4 tests whether selecting only actions that
are jointly feasible against predicted pedestrian motion and visible
static/wall geometry avoids that failure transfer.

## Frozen selector

- Protocol schema: `sa4_d4_geometry_selector/v1`.
- Protocol SHA-256:
  `1729fe3a878b1a1c64f172bb7b3bb0a23f6553385c8b8ca9e982e0cf97170e9d`.
- Machine freeze: `docs/freeze/sa4_d4_geometry_selector_v1.json`.
- Implementation:
  `scripts/reinforcement_learning/skrl/rnn_car_wdclean/d4_geometry_selector.py`.
- Action order:
  `policy index -> selector index -> decode -> d1 queue -> applied command`.
- The first 0.2 s of each candidate path uses the command already pending in
  the d1 decoded-command queue; reset rows use zero pending command.
- Candidate horizon: 2.4 s, 12 samples, all 19x19 actions.
- Action contract: 19 bins; v_max=1.0 m/s; reverse scale=0.2;
  a_max=0.5 m/s2; omega_max=1.2 rad/s; alpha_max=3.0 rad/s2.
- Robot OBB: half `(0.35,0.30)` m, x offset `-0.128` m.
- Dynamic check: constant-velocity obstacle circles, minimum OBB-to-circle
  surface clearance >=0.10 m.
- Static check: current raw 72-bin LiDAR surface points, after privileged
  removal of returns attributed to dynamic circles, minimum swept OBB
  clearance >=0.10 m.

## Deterministic selection

1. Preserve policy action if jointly feasible.
2. Otherwise minimize L1 distance in the two action indices among feasible
   candidates.
3. Tie by maximum bottleneck dynamic/static margin, then forward speed, then
   smallest flattened index.
4. If none is feasible, preserve policy and mark the frame unresolved. This
   makes no protection claim.

The existing frozen two-frame radial-TTC trigger is retained to isolate the
selector mechanism from trigger changes. Every active-frame grid still checks
all valid dynamic obstacles.

## Fixed comparison result

- Runner:
  `scripts/reinforcement_learning/skrl/rnn_car_wdclean/run_sa4_d4_geometry_suite.py`.
- Fresh arms in fixed order: identity baseline, then `geometry_feasible`.
- Checkpoint: SA4-R2 c6400, SHA
  `c6dbd94bcbd6cc17ceca3cf8d5abfb4d956c98ed173c0dd890b9fc45959a2197`.
- Cell: stage4 lateral, seed818, d1=200 ms, 64 env x 2500 steps/arm.
- Runner preregisters protocol before the first rollout, refuses overwrite,
  fingerprints source before/after each arm, and fails on delay/ledger drift.
- Baseline: SR `83.05%`, obstacle CR `16.45%`, wall CR `0.51%`, total CR
  `16.95%`, TO `0%`.
- Geometry-feasible: SR `82.44%`, obstacle CR `16.90%`, wall CR `0.65%`,
  total CR `17.56%`, TO `0%`.
- Active frames: `24,414`; jointly feasible `4,426` (`18.1%`); no feasible
  `19,988` (`81.9%`); overrides `2,455`.
- Status: `COMPLETE_VALID_DIAGNOSTIC_EVIDENCE`, not passed.

## Validity and limits

- Targeted contracts: `51 passed`.
- Full rnn_car_wdclean regression: `817 passed, 27 subtests passed`.
- 64-env synthetic batch shapes are valid; CPU peak RSS about 678 MB.
- Dynamic state and dynamic-return attribution are privileged.
- Static geometry includes only currently visible LiDAR surfaces; occluded
  geometry cannot be reconstructed.
- Constant-velocity prediction, one checkpoint, one evaluator seed.
- The `81.9%` denominator is active environment frames, not collisions or
  episodes. It is not a physical collision-unavoidability rate.
- No-feasible means no candidate under the frozen discrete action, prediction,
  horizon, visible-LiDAR, and clearance model. It does not prove absolute
  physical infeasibility.
- The `+0.61 pp` total-CR point difference is single-seed descriptive evidence;
  conclude no observed improvement, not proven degradation.
- Next diagnostic: baseline-only shadow-mode feasibility frontier to locate
  when jointly feasible actions disappear before collisions and successful
  passes. Do not change reward or start SA5.
