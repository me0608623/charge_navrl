---
name: SA4-D4 geometry selector fixed comparison result
description: Valid single-checkpoint diagnostic found no improvement and 81.9 percent no-feasible active frames under the frozen model
type: finding
date: 2026-08-01
updated: 2026-08-01
status: complete-not-passed
---

# SA4-D4 fixed comparison verdict

## Identity

- Output: `logs/gates/sa4_d4/geometry_suite_r1_20260801/`.
- Status: `COMPLETE_VALID_DIAGNOSTIC_EVIDENCE`.
- Checkpoint: SA4-R2 c6400, SHA-256
  `c6dbd94bcbd6cc17ceca3cf8d5abfb4d956c98ed173c0dd890b9fc45959a2197`.
- Protocol SHA-256:
  `1729fe3a878b1a1c64f172bb7b3bb0a23f6553385c8b8ca9e982e0cf97170e9d`.
- Fixed cell: stage4 lateral, seed818, d1=200 ms, 64 env x 2500 steps/arm.
- No training or next-stage launch occurred; source fingerprints were stable.

## Results

| metric | baseline | geometry-feasible | delta |
|---|---:|---:|---:|
| SR | 83.05% | 82.44% | -0.61 pp |
| obstacle CR | 16.45% | 16.90% | +0.45 pp |
| wall CR | 0.51% | 0.65% | +0.14 pp |
| total CR | 16.95% | 17.56% | +0.61 pp |
| TO | 0% | 0% | 0 pp |

Both arms failed `SR >= 90%, CR <= 10%, TO <= 5%`. The valid conclusion is
no observed improvement. The single-seed point estimates do not prove that the
selector causes degradation.

## Mechanism accounting

- trigger activations: `2,332`.
- active environment frames: `24,414`.
- jointly feasible: `4,426` (`18.1%`).
- policy already feasible: `1,971`.
- overrides: `2,455`.
- no feasible: `19,988` (`81.9%`).
- delay alignment errors: `0` over `157,949` samples.
- dynamic LiDAR returns removed: `200,917`.

The `81.9%` denominator is active frames, not collisions, episodes, or trigger
events. No-feasible is relative to the frozen 19x19 action grid, 2.4-second
constant-velocity prediction, current visible LiDAR surfaces, and 0.10 m
clearance. It is not a physical unavoidability claim.

## Decision and completed follow-up

- `accepted_parent=null`; SA4 remains HOLD; SA5 remains unauthorized.
- Do not tune selector weights or reward from this result.
- D5 subsequently completed in baseline-only shadow mode. Collision last-feasible
  p50 was 2.4 s and linear intersection warning preceded radial TTC by a paired
  median 1.2 s, but frozen static-feasible coverage was only 34.53%.
- The next step is therefore a static-feasibility bottleneck audit, not an
  immediate early-trigger A/B. See
  [finding_sa4_d5_shadow_frontier_20260801.md](finding_sa4_d5_shadow_frontier_20260801.md).

## Obsidian authority

`/home/aa/Documents/Obsidian Vault/isaaclab_v4/Gate結果/23_SA4_D4_GeometryFeasible固定比較結果_20260801.md`
