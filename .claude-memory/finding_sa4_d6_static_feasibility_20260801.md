---
name: SA4-D6 static-feasibility source and sensitivity finding
description: Corrected all-wall baseline shadow identifies residual LiDAR points as the frozen D4 model's dominant numerical blocker
type: finding
date: 2026-08-02
status: complete-valid-diagnostic
---

# SA4-D6 static-feasibility source and sensitivity finding

## Fixed run

- checkpoint: SA4-R2 c6400, SHA-256 `c6dbd94bcbd6cc17ceca3cf8d5abfb4d956c98ed173c0dd890b9fc45959a2197`
- stage4 lateral, seed818, d1=200 ms, 64 env x 2500 steps
- output: `logs/gates/sa4_d6/static_sensitivity_r3_20260801/`
- completed: 2026-08-02 00:18 CST; output suffix is a preregistered batch label
- protocol: `sa4_d6_static_feasibility_sensitivity/v2`, SHA-256 `acab10b5417c7ee1e79b95dbd94dd9adfe9055cbb5a9e5ff99aafc79ad164620`
- action identity errors 0/160,000; source partition errors 0; D3 delay errors 0
- 1,976 episodes; SR/CR/TO 83.05%/16.95%/0.00%; no training or SA5

## Finding

Frozen h2.4/c0.10 static-any feasibility is 34.53% (104,759 blocked frames).
Corrected attribution includes long-corridor and arena boundary walls:

- known active walls: 61.08% of points; removing rescues only 0.96% blocked frames
- active static obstacles: 13.70%; removing rescues 0.93%
- residual unmatched LiDAR points: 25.22%; removing rescues 97.65%

Thus known walls are not the dominant numerical blocker. A binding subset of residual
LiDAR points dominates the min-over-points static constraint.

Horizon and clearance are secondary interacting amplifiers:

- h2.4 -> h0.4 at c0.10 rescues 24.35% blocked frames
- c0.10 -> c0.05 at h2.4 rescues 21.41%
- c0.10 -> c0.01 rescues 38.07%
- c0 is a degenerate tautology because point-to-OBB distances are non-negative; do not use it for safety calibration

## Scope

Residual means unmatched within 0.20 m by the current wall-AABB/static-circle source model.
It does not prove sensor noise. Candidate mechanisms include full-mode distractors/noise,
5-degree bin-centre reconstruction, dynamic-return subtraction mismatch, or missing source
geometry. Source/horizon/clearance rescue rates are coupled and cannot be added.

The earlier D6 v1 run omitted arena boundary walls from attribution. Its factor matrix remains
valid, but its wall/residual point fractions are superseded by v2. SA4 remains HOLD and SA5
unauthorized. Obsidian authority:
`Gate結果/25_SA4_D6_StaticFeasibility敏感度與來源分解_20260801.md`.

## D7 follow-up

D7 completed the origin audit in `logs/gates/sa4_d7/residual_origin_r2_20260802/` and reconciled
all 2,603,039 points exactly. Final mechanisms are 59.95% distractor winners, 37.63%
actual-ray-to-bin-centre representation/dynamic-attribution artifacts, and 1.76% raw geometry
unmatched. See `finding_sa4_d7_residual_lidar_origin_20260802.md`. These are point-source shares,
not exact feasibility rescue or collision-cause shares.
