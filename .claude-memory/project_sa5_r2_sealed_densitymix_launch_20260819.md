# SA5-R2 sealed low-density-mix corrective lineage

Date: 2026-08-19
Status: TRAINING; formal run active

## Decision and evidence boundary

- The legacy SA5 corridor had a real 2.0 m end bypass. Its corridor results
  remain historical/diagnostic evidence only.
- The sealed r3 screen was stopped early after 9 valid cells and preserved as
  `STOPPED_EARLY_SUFFICIENT_DIAGNOSTIC_EVIDENCE`; no ranking or parent was
  accepted from it.
- The formal SA5-R2 run contains two declared interventions: sealed corridor
  geometry and low-density corridor coverage. Future claims must not attribute
  change to geometry alone.
- Training may create checkpoints only. It cannot accept an SA5 parent or
  launch SA6.

## Frozen lineage

- Parent: original SA4-R3 conceptual it125
  `logs/rnn_car/sa4_r3_cont25_from_it100_ne1024_s42_p25_r1/checkpoint_3200.pt`
- Parent SHA-256:
  `57f43d07255971ad607e0bf3234dd7d46984c9c71cc9550e2e9461fbc95ab71d`
- Optimizer reset (`no_resume_optimizer=True`)
- Config:
  `e2e_sa5_r2_sealed_densitymix_from_sa4r3_it125_actdelay12`
- Config SHA-256:
  `85e16878bf9af731ed95290e4cb1c3af77117b87858ff262345ab8f7491f2757`
- Protocol:
  `docs/freeze/sa5_r2_sealed_densitymix_correction_v1.json`
- Protocol SHA-256:
  `5721a57191802510bb52c71ed479a0243f330b76ad1d794f8044a4b2ddca4ca0`

## Training distribution

- Native/narrow/corridor reset shares remain 0.78/0.12/0.10.
- Corridor density mix: 0S1D 10%, 1S1D 10%, 2S1D 15%, 3S2D 30%,
  4S2D 35%.
- Thus 35% of corridor resets contain one dynamic obstacle and 65% retain the
  moderate/hard two-dynamic cases. Across all resets, single-dynamic corridors
  are nominally 3.5%.
- Geometry remains interaction length 10 m, physical wall span 15 m and 0.5 m
  boundary overlap.
- Full VLP-16 noise + valid_return_only; actuator delay U{1,2}; scale 1.0;
  motor lag 1.0; reward/model/PPO unchanged.

## Pre-training baseline

Exact parent, sealed fixed 4S2D, seed818, d1, 64 env, 2500 steps, 4/4 valid,
source fingerprint stable:

| family | n | SR | CR | wall CR | obstacle CR |
|---|---:|---:|---:|---:|---:|
| lateral | 2823 | .5675 | .4325 | .0375 | .3950 |
| longitudinal | 2483 | .5429 | .4571 | .0395 | .4176 |
| random2d | 1644 | .1746 | .8254 | .0134 | .8120 |
| mixed | 2291 | .4033 | .5967 | .0148 | .5818 |

The baseline is diagnostic only. It shows the exact parent is weak on hard
dynamic interaction while sealed wall collisions are not the dominant term.
Output: `logs/gates/sa5_r2_parent_baseline/baseline_20260819_r2/`.

## Validation and launch

- Focused config/protocol tests: 6 passed.
- Full wdclean CPU suite: 1001 passed + 27 subtests.
- Density-mix 64-env/1-iteration GPU smoke: PASS; reconciliation=1; no strict
  anomaly; checkpoint_128 SHA
  `5bb567f46fe66e3c1790fc8199d57c6ac77da69971309b381e639b0a5700f363`.
- Formal unit:
  `sa5-r2-sealed-ldmix-from-sa4r3-it125-p300-r1.service`
- Run:
  `sa5_r2_sealed_ldmix_from_sa4r3_it125_actd12_ne1024_s42_p300_r1`
- Started 2026-08-19 16:20:04 CST, 1024 env, 300 iterations, save every 50.
- First runtime density realization over 92 corridor envs:
  0S1D/1S1D/2S1D/3S2D/4S2D = 9/9/14/28/32; constructive solvability 100%.
- First iteration: SR .6913, CR .0980, TO 0, entropy 3.8389,
  KL 7.36e-08, fps 2856, corridor-family reconciliation=1.
- Supervisor expected_run and PID matched in a real cron invocation;
  auto-advance remains HALTED_ALERT.

## Launch health snapshot

At 2026-08-19 16:26:50 CST the formal unit remained active at iteration
6/300. Metrics were SR .8802, CR .1131, TO 0, entropy 3.9093, KL .002852,
clip fraction .02346, VF 11.4954 and fps 2809. Corridor-family
reconciliation remained 1.0. This is a launch-health snapshot, not Gate or
checkpoint-selection evidence; the first checkpoint is expected at it50.

## Post-training requirements

1. Screen every saved checkpoint on fixed sealed 4S2D lateral,
   longitudinal, random2d and mixed cells.
2. Check sealed 0S1D and 1S1D speed/stop behavior to detect unnecessary
   stop-and-go in easy corridors.
3. Verify native and narrow retention for leading checkpoints.
4. Do not select from training metrics alone and do not auto-launch SA6.
