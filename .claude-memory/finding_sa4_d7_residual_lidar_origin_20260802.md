---
name: SA4-D7 residual LiDAR origin finding
description: Exact realized-trace audit attributes D6 residual points primarily to distractor winners and 1-degree to 5-degree angle representation
type: finding
date: 2026-08-02
status: complete-valid-diagnostic
---

# SA4-D7 residual LiDAR origin finding

## Fixed valid run

- output: `logs/gates/sa4_d7/residual_origin_r2_20260802/`
- status: `COMPLETE_VALID_DIAGNOSTIC_EVIDENCE`
- checkpoint: SA4-R2 c6400, SHA-256
  `c6dbd94bcbd6cc17ceca3cf8d5abfb4d956c98ed173c0dd890b9fc45959a2197`
- stage4 lateral, evaluator seed818, d1=200 ms, 64 env x 2500 steps
- 1,976 episodes; SR/CR/TO 83.05%/16.95%/0%
- protocol: `sa4_d7_lidar_residual_origin/v2`, SHA-256
  `8f50b1689da8f575d9c1b0760115b2bc0e7be3f5622eaa70783c0bed7cdf38ca`
- action identity 160,000/160,000; exact policy trace 2,500/2,500; ambiguous 0
- all D7 reconciliations/self-checks true; source fingerprint stable
- D3/D6/corridor outputs match the previous D6 baseline, so tracing did not alter RNG or rollout

## Finding

D6 final residual points total 2,603,039. Mutually exclusive source mechanisms:

- distractor winner: 1,560,617 (59.95%)
- bin-centre reconstruction: 802,188 (30.82%)
- actual-angle dynamic attribution missed by centre-angle reconstruction: 177,250 (6.81%)
- raw geometry unmatched: 45,873 (1.76%)
- range corruption + raw dynamic unmatched + unresolved: 17,112 (0.66%)

Angle-representation mechanisms total 37.63%. The main origin is therefore the simulated
distractor/min-pool pipeline plus loss of the winning raw ray's azimuth, not a large missing-wall
model.

Noise amplification mechanism: the configured base rate 0.002515 is applied independently to
5,760 raw rays, with replacement range U(0.2,2.0)m, before 72-bin minimum pooling. Expected
distractor draws per scan are 14.4864; observed final residual distractor winners are 9.75/frame.
In frozen-static-blocked frames, the nearest residual is a distractor in 99.16% of frames and lies
at 0.4-0.6m in 90.20%.

## Interpretation limits

- Point-source share is not collision-cause share or exact blocked-frame rescue share.
- A residual point present in a blocked frame need not be the unique binding point.
- `raw_geometry_unmatched` inherits D6's 0.20m matching tolerance and does not identify a USD prim.
- Single checkpoint/training seed42/evaluator seed818; diagnostic, not population evidence.
- r1 is invalid and preserved at
  `logs/gates/sa4_d7/residual_origin_r1_20260802_INVALID_final_range_roundtrip/`; one float32
  boundary point failed reconciliation. Cite only r2.

## Decision

1. Audit/recalibrate the distractor contract, especially per-raw-ray sampling and U(0.2,2.0)m.
2. Preserve each 5-degree bin's true argmin azimuth for static geometry, or use higher-resolution
   points while keeping 72-bin policy input.
3. If exact rescue shares are required, run same-frame mechanism-specific feasibility shadows.
4. Do not first expand wall geometry, disable all noise, lower clearance, accept SA4 parent, or
   launch SA5.

Obsidian authority:
`Gate結果/26_SA4_D7_ResidualLiDAR點來源稽核_20260802.md`.

