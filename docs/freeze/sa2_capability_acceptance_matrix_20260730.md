# SA2 Narrow and 3S1D Capability Acceptance Freeze

Date: 2026-07-30

Status: protocol frozen before SA2 capability results are inspected.

This document freezes the SA2-specific narrow-passage and corridor capability
test. It does not by itself graduate SA2: navigation-core and training-health
gates remain separate blocking layers.

## Question

The test answers only:

1. Can the checkpoint traverse the narrow distribution used by SA2?
2. Can it traverse the fixed 3-static + 1-dynamic corridor used by SA2 under
   both motion families that SA2 sampled?
3. Does each capability remain valid at fixed decoded-command delays of
   0, 200 and 400 ms?

It must not use the historical deployment gates:

- fixed 1.2 m Gate5 is not the SA2 narrow distribution;
- 4S2D at 4.0 m and 0.30-0.60 m/s is not the SA2 corridor;
- `random_2d` is absent from SA2 and is not an SA2 blocking cell.

## Source Contract

The runner reads all scene values from
`STAGE_SCENE_CURRICULUM[2]`. Constants copied into a command or report are not
an independent source of truth.

| Field | Frozen SA2 value |
|---|---:|
| stage | 2 |
| arena | 18 x 18 m |
| VLP-16 noise | full |
| narrow width | U[1.60, 1.80] m |
| narrow yaw | U[-3, +3] deg |
| narrow segment length | 9.0 m |
| corridor free width | 4.8 m |
| corridor length | 10.0 m |
| corridor density | fixed 3S1D |
| dynamic speed | U[0.18, 0.30] m/s |
| motion families | lateral, longitudinal |
| random-2D | absent |
| actuator profile | `sa1_delay_only` |
| fixed delays | d0=0 ms, d1=200 ms, d2=400 ms |

The executable is:

`scripts/reinforcement_learning/skrl/rnn_car_wdclean/run_sa2_capability_suite.py`

One invocation is exactly one
`(checkpoint, scenario, delay, seed)` cell. The runner intentionally has
`--seed`, not `--seeds`.

The fail-closed matrix driver is:

`scripts/reinforcement_learning/skrl/rnn_car_wdclean/sa2_capability_queue.py`

It runs Stage A as 18 cells and Stage B as 27 cells per candidate. It refuses
stale cell directories, fingerprints the queue/runner/play/curriculum sources
for every cell, and selects a checkpoint only from candidates whose complete
27-cell conjunction passes.

## Cell Thresholds

No cross-seed, cross-delay or cross-scenario average may turn a failed cell
into a pass.

### Narrow

Every individual narrow cell must satisfy:

| Metric | Threshold |
|---|---:|
| completed episodes | >= 1000 |
| SR | >= 0.90 |
| CR | <= 0.05 |
| TO | <= 0.05 |
| barrier crossing rate | >= 0.95 |
| direct crossing rate | >= 0.95 |

The runtime contract must also prove the built geometry, stage, full VLP-16
noise, unclamped replay layout and fixed actuator pipeline.

### 3S1D Corridor

Every lateral and longitudinal cell must independently satisfy:

| Metric | Threshold |
|---|---:|
| completed episodes | >= 1000 |
| SR | >= 0.90 |
| CR | <= 0.10 |
| TO | <= 0.05 |

The following structural checks are hard gates:

- measured free width and length equal 4.8 m and 10.0 m;
- measured obstacle counts equal 3S1D in every environment;
- requested and measured motion family agree, with family fraction 1.0;
- all dynamic slots move and stay below the requested speed upper bound;
- goal and local-goal alignment pass;
- constructive unsolvable count is zero;
- dynamic-static penetration is zero and audit coverage is complete;
- full VLP-16 and fixed actuator runtime markers are present.

Missing, non-finite or contradictory fields void the cell. They are not
converted to zero.

## Candidate Protocol

The completed 300-iteration SA2 run has six saved checkpoints:
`c50/c100/c150/c200/c250/c300`.

### Stage A: Screening

- checkpoints: all six saved checkpoints;
- scenarios: narrow, lateral, longitudinal;
- seed: 818;
- delay: d1 (200 ms);
- cells: `6 x 3 = 18`.

Rank checkpoints by their minimum signed margin over all three cells. A
lower-bound margin is `value - bound`; an upper-bound margin is
`bound - value`. A structurally invalid cell ranks below every valid cell.
The nominal carry-forward count is two. If the second/third-place margin gap
is smaller than `TIE_MARGIN=0.005`, the cut is unsafe and every checkpoint
within `0.005` of the second-place margin proceeds. A tie is expanded, never
broken by checkpoint order or by a metric selected after seeing the screen.
Screening never declares SA2 PASS.

The pass/fail checks remain blocking even when they are saturated and therefore
carry no ranking information. The report must name saturated checks explicitly.

### Stage B: Formal Confirmation

- candidates: every checkpoint carried by the Stage A tie rule;
- scenarios: narrow, lateral, longitudinal;
- seeds: 515, 616, 717;
- delays: d0, d1, d2;
- cells per candidate: `3 x 3 x 3 = 27`.

A candidate passes the capability layer only if all 27 of its cells pass.
If multiple candidates pass, select the one with the larger minimum signed
margin, then the lower worst-cell collision rate.
Aggregate rates may be reported for diagnosis, but they do not participate in
the verdict.

For the 2026-07-30 Stage A result, `c50/c100/c150/c200` tied at `+0.0500`.
`c250` and `c300` were outside the tie band, at `+0.0380` and `+0.0327`.
The formal Stage B matrix is therefore `4 x 27 = 108` cells.

### Stage A Aggregation Erratum

The original Stage A summary compared checkpoint paths as strings. The queue
requested paths below `/home/aa/IsaacLab/logs`, while the runner recorded the
resolved symlink target below `/home/aa/logs`; all 18 valid cells therefore
folded into zero per-checkpoint cells. The immutable cell payloads remain valid.
The corrected aggregation resolves both paths before comparing them and raises
if a requested checkpoint matches no cells. The original summary is preserved;
the corrected report is
`stage_a_screen_summary_regraded_20260730.json`.

## SA2 Graduation Boundary

SA2 may advance only when all of these independent facts are true:

1. training health is valid and the checkpoint is readable;
2. the stage-aligned navigation-core gate passes;
3. this SA2 capability matrix passes;
4. the result set has no missing or duplicate cell IDs;
5. source/config hashes match the freeze used to construct the cells.

The SA2 training aggregate SR is not a substitute for item 2 or 3 because it
mixes native, narrow and corridor episodes with different dwell times.

The current SA2 parent came from the non-canonical SA1-R1 c1700 lineage, which
had already seen harder narrow/corridor replay. A pass therefore establishes
SA2 capability and retention, not that SA2 was the first stage to learn it.

## Execution Example

This example runs one cell only and does not launch a matrix:

```bash
/home/aa/miniconda3/envs/env_isaaclab/bin/python \
  scripts/reinforcement_learning/skrl/rnn_car_wdclean/run_sa2_capability_suite.py \
  logs/rnn_car/sa2_sim2real_v2_from_sa1r1_c1700_ne1024_s42_p300_r1/checkpoint_38400.pt \
  --output-dir logs/gates/sa2_capability/c300 \
  --scenario narrow \
  --seed 515 \
  --actuator-delay-steps 1 \
  --actuator-profile sa1_delay_only
```

Exit status is 0 for a valid PASS, 1 for a valid threshold FAIL, and greater
than 1 or an exception for an invalid/void cell.
