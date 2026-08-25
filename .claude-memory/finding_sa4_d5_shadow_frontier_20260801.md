---
name: SA4-D5 baseline-only shadow feasibility frontier
description: D5 validly located the model-relative feasibility frontier; radial TTC is later than path-intersection warning, while static feasibility is the dominant frozen-model bottleneck
type: finding
date: 2026-08-01
status: complete-diagnostic
---

# SA4-D5 shadow frontier verdict

- Output: `logs/gates/sa4_d5/shadow_r1_20260801/`.
- Status: `COMPLETE_VALID_DIAGNOSTIC_EVIDENCE`.
- Fixed c6400, stage4 lateral, seed818, d1=200 ms, 64 env x 2500 steps.
- Checkpoint SHA: `c6dbd94bcbd6cc17ceca3cf8d5abfb4d956c98ed173c0dd890b9fc45959a2197`.
- Protocol SHA: `3ff3513423bd132a7dba7261bf3d3d4cf0f738d9d7f5b1f79e6ed40b40bde1b1`.
- Shadow only: policy actions unchanged; no training and no SA5.

## Validity

1,976 episodes; SR/CR/TO `83.05% / 16.95% / 0%`; 160,000/160,000
records evaluated; 312 dynamic collisions; D3 delay alignment 157,960 samples / 0
errors; all D3/D5 reconciliation true; source fingerprint stable.

## Frontier

- Collision event no-feasible: 99.68%; successful closest-approach event: 67.18%.
- Feasible seen in prior 5 s: collision 97.76%, success control 97.75%.
- Entire evaluated 5 s no-feasible: 2.24% / 2.25%.
- Collision last-feasible p50 2.4 s; persistent collapse p50 2.2 s.
- Control last-feasible p50 0.2 s; persistent collapse p50 0.4 s.

Controls are per-obstacle closest approaches in successful episodes, not independent episodes.
No-feasible is relative to the frozen D4 model, not physical unavoidability.

## Timing

- Reconstructed two-frame D4 radial trigger: all collision events, first lead p50 2.4 s;
  51.09% of successful close approaches, p50 2.0 s among warned.
- Linear intersection proxy: all collisions, p50 3.8 s; 68.52% controls. It was
  earlier than radial in 93.91% of paired collisions, median advantage 1.2 s.
- Delay-aware policy-path conflict: all collisions, p50 2.6 s; 32.54% controls;
  earlier than radial in 64.10%, median advantage 0.4 s.
- Warning before/at persistent collapse: radial 58.84%, linear 87.78%, path 65.92%.

This supports useful lateral-intersection timing beyond radial TTC, but linear conflict alone
is too non-specific to freeze as an intervention trigger from this run.

## Dominant limitation

All frames: dynamic-feasible 98.03%, static-feasible 34.53%, joint-feasible 33.68%.
At collision persistent-collapse onset: dynamic-feasible 90.03%, static-feasible 9.65%.
Thus frozen-model static feasibility is the dominant numerical bottleneck. D5 cannot decide
whether this is real geometry or conservatism from visible LiDAR surfaces, 2.4 s horizon,
0.10 m margin, or the 19x19 grid.

## Decision

SA4 HOLD and SA5 unauthorized remain. Do not just lower TTC or rerun D4. Next audit the
static-feasibility bottleneck; only after calibration should an intersection-aware fresh A/B be frozen.

## Follow-up

D6 v2 completed the static sensitivity audit. Known active walls are not the dominant blocker;
residual unmatched LiDAR points dominate the frozen model. See
`finding_sa4_d6_static_feasibility_20260801.md`.

Obsidian authority: `Gate結果/24_SA4_D5_ShadowFeasibilityFrontier_20260801.md`.
