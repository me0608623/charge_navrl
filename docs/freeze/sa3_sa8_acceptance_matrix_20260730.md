# SA3-SA8 Formal Acceptance Matrix and Threshold Freeze

Date: 2026-07-30

Status: design frozen before SA3-SA8 training or acceptance results.

This is the graduation contract for the canonical
`sim2real_curriculum_v2` lineage. A stage ending normally or reporting a high
mixed training SR is not sufficient for promotion.

The machine-readable companion is
`scripts/reinforcement_learning/skrl/rnn_car_wdclean/sa3_sa8_acceptance_contract.py`.
Supervisors must import that contract instead of scraping this Markdown.

## Invariants

All stages use:

- K8 / 83D issued-action history;
- measured VLP-16 full noise;
- OBB collision;
- decoded-command delay `U{0,1,2}` during training;
- fixed-delay acceptance cells d0/d1/d2 = 0/200/400 ms;
- actuator profile `sa1_delay_only`;
- held-out formal seeds 515, 616 and 717;
- deterministic policy evaluation;
- at least 1000 completed episodes per performance cell;
- one independently scored `(checkpoint, scenario, delay, seed)` cell.

No result may be pooled across seed, delay, scene, density or motion family to
rescue a failed blocking cell.

## Stage Scene Contracts

The executable source of truth is
`rnn_car_modular/configs/sim2real_stage_curriculum_v2.py`.

| Stage | Arena | Narrow range | Exact narrow | Corridor | Speed | Random-2D |
|---|---:|---|---|---|---|---|
| SA3 | 17 x 17 m | 1.50-1.70 m, yaw +/-4 deg | none | 4.6 m, fixed 3S1D | 0.20-0.35 m/s | absent |
| SA4 | 16 x 16 m | 1.40-1.60 m, yaw +/-4 deg | none | 4.4 m, fixed 3S2D | 0.22-0.40 m/s | absent |
| SA5 | 15 x 15 m | 1.30-1.50 m, yaw +/-4 deg | none | 4.2 m, 50% 3S2D + 50% 4S2D | 0.25-0.45 m/s | patrol |
| SA6 | 14 x 14 m | 1.20-1.40 m, yaw +/-4 deg | none | 4.0 m, 30% 3S2D + 50% 4S2D + 20% 4S3D | 0.27-0.50 m/s | patrol |
| SA7 | 13 x 13 m | 1.20-1.40 m, yaw +/-6 deg | 25% at 1.20 m | deployment mix | 0.30-0.60 m/s | wander |
| SA8 | 12 x 12 m | 1.20-1.40 m, yaw +/-8 deg | 50% at 1.20 m | stress mix | 0.30-0.60 m/s | wander |

SA7 deployment mix:

```text
25% 3S1D / 35% 4S2D / 20% 4S3D / 15% 5S3D / 5% 5S5D
```

SA8 stress mix:

```text
15% 3S1D / 25% 4S2D / 20% 4S3D /
20% 5S3D / 10% 5S4D / 10% 5S5D
```

## Blocking Layers

### G0 Training Health

Over the final 50 iterations:

- all logged scalar metrics are finite;
- median approximate KL <= 0.015;
- no two consecutive 10-iteration windows have median KL > 0.0225;
- clip fraction < 0.25;
- no three-window value-loss explosion relative to the preceding stable
  window;
- exact target iteration and final checkpoint exist;
- trainer exits normally without traceback, OOM or forced termination.

G0 identifies optimizer/runtime corruption. It does not prove scene ability.

### G1 Navigation Core

`nav_clean` is a hard gate at every stage. It uses the stage arena, random
start pose and static target, with no walls or obstacles:

| Metric | Threshold |
|---|---:|
| episodes | >= 1000 |
| SR | >= 0.98 |
| CR | <= 0.015 |
| TO | <= 0.005 |

`nav_native` is also blocking from SA3 onward and uses the exact native
stage distribution. Its per-cell thresholds are:

| Stage | SR min | CR max | TO max |
|---|---:|---:|---:|
| SA3 | 0.94 | 0.05 | 0.03 |
| SA4 | 0.92 | 0.07 | 0.035 |
| SA5 | 0.90 | 0.09 | 0.04 |
| SA6 | 0.88 | 0.11 | 0.045 |
| SA7 | 0.86 | 0.13 | 0.05 |
| SA8 | 0.90 | 0.10 | 0.05 |

These are the existing `STAGE_THRESH` deterministic thresholds. SA8 tightens
again because it is the deployment graduation stage.

### G2 Stage Narrow Passage

Every stage-range cell must satisfy:

| Metric | Threshold |
|---|---:|
| episodes | >= 1000 |
| SR | >= 0.90 |
| CR | <= 0.05 |
| TO | <= 0.05 |
| crossing rate | >= 0.95 |
| direct crossing rate | >= 0.95 |

SA7 and SA8 additionally run a fixed 1.20 m lower-edge cell at the stage yaw
limit with the same thresholds. Passing only the random 1.20-1.40 m mixture
cannot hide failure at the deployment minimum.

Runtime structure must prove the built width, yaw, arena, unclamped layout,
sensor mode and actuator pipeline.

### G3 Stage Corridor

Every corridor performance cell must satisfy:

| Metric | Threshold |
|---|---:|
| episodes | >= 1000 |
| SR | >= 0.90 |
| CR | <= 0.10 |
| TO | <= 0.05 |

All structural flags are hard:

- requested and measured width/length/counts agree;
- requested motion and measured slot motion agree;
- goal/local-goal alignment passes;
- no constructive unsolvable scene;
- zero dynamic-static penetrations with complete audit coverage;
- speed does not exceed the stage upper bound;
- no inactive or unclassified slot is included in an outcome ledger.

### G4 Distribution and Interaction Integrity

This layer applies to count-mix stages SA5-SA8.

For a configured category with probability `p`, `N` audited installations and
observed share `p_hat`, the allowed density/family deviation is:

```text
abs(p_hat - p) <= max(0.03, 3 * sqrt(p * (1 - p) / N))
```

Interaction checks use:

```text
abs(p_hat - p) <= max(0.05, 3 * sqrt(p * (1 - p) / N_group))
```

Expected interaction policy:

| Dynamic count | independent | crossing | side-by-side |
|---:|---:|---:|---:|
| D < 2 | 1.00 | 0.00 | 0.00 |
| D = 3 | 0.75 | 0.25 | 0.00 |
| D = 2 or D >= 4 | 0.60 | 0.20 | 0.20 |

Additional hard conditions:

- no `0S0D` or unconfigured density is present;
- lateral/longitudinal/random-2D family accounting is complete;
- crossing pairs are lateral + longitudinal;
- side-by-side pairs are longitudinal + longitudinal;
- random-2D slots never occupy a forced interaction pair;
- realized interaction geometry passes for every forced pair;
- every configured density and every nonzero interaction category is observed.

The exact-mix result is scored per seed and delay. Tolerance is not computed
after merging cells.

### G5 Previous-Stage Retention

The accepted parent and the candidate are evaluated on the same previous-stage
scene, seed and delay. Each paired cell must:

- still pass the previous stage's absolute threshold;
- have candidate SR drop no more than 0.02;
- have candidate CR increase no more than 0.02;
- preserve the same scene and runtime contract hashes.

This gate prevents a harder stage from buying its new capability by erasing
navigation already accepted.

## Scenario Matrix

`range` means the stage's narrow width distribution. `exact` means fixed
1.20 m. Pure corridor family cells use the hardest density listed for that
stage; the exact-mix cell separately tests the real sampler.

| Stage | Blocking scenarios per checkpoint |
|---|---|
| SA3 | nav_clean, nav_native, native_crossing, narrow_range, corridor_lateral_3S1D, corridor_longitudinal_3S1D |
| SA4 | nav_clean, nav_native, native_crossing, narrow_range, corridor_lateral_3S2D, corridor_longitudinal_3S2D |
| SA5 | nav_clean, nav_native, narrow_range, corridor_lateral_4S2D, corridor_longitudinal_4S2D, corridor_random2d_patrol_4S2D, corridor_exact_mix, corridor_crossing_4S2D, corridor_side_by_side_4S2D |
| SA6 | nav_clean, nav_native, narrow_range, corridor_lateral_4S3D, corridor_longitudinal_4S3D, corridor_random2d_patrol_4S3D, corridor_exact_mix, corridor_crossing_4S3D, corridor_side_by_side_4S2D |
| SA7 | nav_clean, nav_native, narrow_range, narrow_exact_1p20, corridor_lateral_5S5D, corridor_longitudinal_5S5D, corridor_random2d_wander_5S5D, corridor_exact_mix, corridor_crossing_5S5D, corridor_side_by_side_5S5D |
| SA8 | nav_clean, nav_native, narrow_range, narrow_exact_1p20, corridor_lateral_5S5D, corridor_longitudinal_5S5D, corridor_random2d_wander_5S5D, corridor_exact_mix, corridor_crossing_5S5D, corridor_side_by_side_5S5D |

`native_crossing` in SA3-SA4 is the native stage crossing scene, not a
count-mix corridor interaction. These are different mechanisms and their
results must not be merged.

## Candidate Selection and Formal Matrix

### Stage A: Screening

- candidates: every saved checkpoint plus the exact final checkpoint;
- seed: 818;
- delay: d1 (200 ms);
- scenarios: every blocking scenario for that stage;
- rank: minimum signed margin over all valid cells;
- promotion: best two structurally valid checkpoints only.

A screening pass is not a stage pass.

### Stage B: Formal Confirmation

For each of the two Stage A candidates:

- seeds: 515, 616, 717;
- delays: d0, d1, d2;
- scenarios: every blocking scenario for that stage.

Cells per candidate:

| Stage | Scenarios | Formal cells per candidate |
|---|---:|---:|
| SA3 | 6 | 54 |
| SA4 | 6 | 54 |
| SA5 | 9 | 81 |
| SA6 | 9 | 81 |
| SA7 | 10 | 90 |
| SA8 | 10 | 90 |

The candidate passes only if every required cell and G0-G5 pass. If both
candidates pass, choose the larger minimum signed margin, then lower CR as the
tie breaker. Never choose by mean SR.

## Automatic Advancement Contract

An automated supervisor may start SA(N+1) only after all of the following are
machine-verified:

1. the current training unit is inactive and its process is gone;
2. GPU evaluation jobs have exited;
3. exact final and candidate checkpoints are readable;
4. expected cell IDs equal recorded cell IDs with no duplicates;
5. every cell is structurally valid and individually passes;
6. G0-G5 pass and the summary recomputes all verdicts from raw metrics;
7. source/config hashes match this acceptance run's manifest;
8. the next-stage config contains the selected checkpoint explicitly and no
   pending-parent sentinel;
9. a CPU config-lock test and a short simulator smoke both pass;
10. only one next-stage training unit is launched.

Any missing field, unsupported scenario, stale schema, hash mismatch, process
ambiguity or non-finite value means HOLD. It must never be interpreted as
PASS-by-default.

## Implementation Boundary

As of this freeze:

- the SA2 fixed-count runner is implemented and tested;
- the play path supports parameterized corridor width and speed;
- SA5-SA8 exact density/interaction performance attribution is not yet a
  complete formal runner.

Therefore this document freezes what those future runners must prove; it does
not authorize SA3-SA8 auto-advance until the corresponding executable runner,
JSON schema and CPU tests exist and match this matrix.
