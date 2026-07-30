# SA1-SA8 Sim-to-Real Scene Curriculum V2

## Decision

The canonical curriculum separates two independent axes:

1. Sim-to-real realism stays fixed from SA1: measured VLP-16 full noise,
   decoded-command delay `U{0,1,2}` (`0/200/400 ms`), K8/83D issued-action
   history, and OBB collision.
2. Task difficulty increases only at stage boundaries: room size, narrow-gap
   geometry, corridor share, density, speed, motion family and interaction
   complexity.

The running `sa1_sim2real_v1_ne1024_s42_r1` experiment predates this decision.
It remains an all-at-once robustness arm with 12% deployment narrow replay and
10% full deployment-corridor replay. It must not be relabelled as canonical
navigation-first SA1.

## Frozen Stage Table

Reset shares below are nominal episode-reset probabilities. They are not
rollout-frame shares: longer corridor episodes can occupy more rollout frames
than their reset fraction suggests.

| Stage | Arena | Native / narrow / corridor | Narrow replay | Corridor replay |
|---|---:|---:|---|---|
| SA1 | 20 x 20 m | 94 / 4 / 2% | 1.80-2.00 m, yaw +/-2 deg | 5.0 m, 2S1D, 0.15-0.25 m/s, lateral |
| SA2 | 18 x 18 m | 90 / 6 / 4% | 1.60-1.80 m, yaw +/-3 deg | 4.8 m, 3S1D, 0.18-0.30 m/s, lateral/longitudinal |
| SA3 | 17 x 17 m | 86 / 8 / 6% | 1.50-1.70 m, yaw +/-4 deg | 4.6 m, 3S1D; native SA3 introduces crossing |
| SA4 | 16 x 16 m | 82 / 10 / 8% | 1.40-1.60 m, yaw +/-4 deg | 4.4 m, 3S2D; crossing retained |
| SA5 | 15 x 15 m | 78 / 12 / 10% | 1.30-1.50 m, yaw +/-4 deg | 4.2 m, 50% 3S2D + 50% 4S2D; random-2D patrol begins |
| SA6 | 14 x 14 m | 78 / 12 / 10% | 1.20-1.40 m, yaw +/-4 deg | 4.0 m, near-deployment density mix; random-2D patrol |
| SA7 | 13 x 13 m | 78 / 12 / 10% | 1.20-1.40 m, yaw +/-6 deg, 25% exact 1.20 m | audited deployment density mix, wander and interactions |
| SA8 | 12 x 12 m | 78 / 12 / 10% | 1.20-1.40 m, yaw +/-8 deg, 50% exact 1.20 m | stress density mix, wander and interactions |

`S` means static obstacles and `D` means dynamic obstacles.

## Density Mixes

```text
SA5:
  50% 3S2D / 50% 4S2D

SA6:
  30% 3S2D / 50% 4S2D / 20% 4S3D

SA7:
  25% 3S1D / 35% 4S2D / 20% 4S3D / 15% 5S3D / 5% 5S5D

SA8:
  15% 3S1D / 25% 4S2D / 20% 4S3D /
  20% 5S3D / 10% 5S4D / 10% 5S5D
```

SA1-SA4 use the legacy fixed-count corridor path. SA2-SA4 assign only lateral
and longitudinal motion. The native `e2e_final20_v1` curriculum introduces
its dedicated crossing scene at SA3. Variable-density corridor allocation,
including sampled interactions and random-2D family slots, begins at SA5.
`wander` is delayed until SA7 so random-2D motion complexity does not arrive at
the same time as its first exposure.

## Lineage and Launch Rules

- Canonical config names are
  `e2e_sa{1..8}_k8_obb_sim2real_curriculum_v2`.
- SA1 starts from scratch.
- SA2-SA8 contain nonexistent parent-checkpoint sentinels and therefore fail
  closed until an accepted preceding-stage checkpoint is written explicitly.
- Optimizer state does not resume at a stage boundary.
- A stage advances only after global and per-scene gates pass. Training
  completion or a high aggregate SR alone is insufficient.
- The current v1 R1 checkpoint may be evaluated as an experimental parent, but
  using it would create a non-canonical lineage because it saw deployment
  narrow/corridor difficulty during SA1.

