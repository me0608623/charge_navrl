# D1 weighted corridor failure and random_2d pause-model mismatch

**Date**: 2026-07-26  
**Status**: D1 closed; random_2d root-cause diagnosis active  
**Authority**: Canonical decision memory for the next corridor experiment

## Decision

- Permanently archive D1 `30/10/60`; do not run c25, extend D1, or start D2.
- Keep D0 as the main baseline:
  `logs/rnn_car/sa6_k8_obb_corridor_env_stratified_probe_c100_s42/checkpoint_3840.pt`
- Keep D1 c10 only as a diagnostic comparison:
  `logs/rnn_car/sa6_k8_obb_corridor_weighted_stratified_d1_s42/checkpoint_1280.pt`
- Training metrics are health signals only. Fixed-seed deterministic gates decide promotion.

## Verified evidence

D1 changed only the motion-family split inside the fixed 10% corridor replay:

`lateral / longitudinal / random_2d: 1:1:1 -> 30:10:60`

This corresponds to about 3% / 1% / 6% of all training environments.

| checkpoint | lateral CR | longitudinal CR | random_2d CR | mixed CR | note |
|---|---:|---:|---:|---:|---|
| D0 | 7.41 | 2.23 | 33.79 | 19.15 | main baseline |
| D1 c10 | 6.87 | 5.98 | 32.90 | 18.60 | diagnostic only |
| D1 c20 | 9.59 | 9.66 | 35.18 | 21.56 | stop line |
| D1 c30 s616 | 11.18 | 18.99 | 39.47 | 25.74 | bounded sentinel; all failed |

D0 to c10 approximate 95% confidence intervals:

- lateral: -0.54pp `[-1.74,+0.66]`
- longitudinal: +3.74pp `[+2.75,+4.74]`, clear regression
- random_2d: -0.88pp `[-3.71,+1.96]`
- mixed: -0.55pp `[-2.58,+1.48]`
- Gate2: -0.64pp `[-1.46,+0.19]`

No apparent c10 improvement is statistically established. The only clear change is worse longitudinal behavior. At c29-c30, train SR/CR improved while every fixed gate worsened.

## Mixed evaluator caveat

Legacy `mixed` uses balanced flat-slot allocation, not per-obstacle IID. With the common `count=1`, it forces heterogeneous pairs; homogeneous pairs are about 11.2% instead of the IID 33.3%.

At c10, pair-weighted IID reconstruction changes mixed CR only from 18.60% to 18.08%. The bias is real but not the main failure. Add a separate `mixed_iid`; never redefine legacy `mixed`.

## Mechanism that exists in code, but is not yet causal

1. `random_2d` uses a short diagonal patrol, about 1.49m, so it reaches waypoints and reverses frequently.
2. `rule_behaviors.py` pauses 0-5 steps at a waypoint and sets velocity exactly to zero.
3. `clean_progress.py` future occupancy:
   - predicts `p + v*t`;
   - ignores obstacles with `||v|| < 0.1m/s`.
4. A paused dynamic obstacle therefore contributes zero future risk. Its subsequent reversal also violates constant-velocity prediction.

Estimated paused/invisible exposure is about 3.2% longitudinal, 6.4% lateral, and 13.1% random_2d. This ordering matches CR ordering, but does not prove causality or explain the full gap.

## Next authorized experiment

Evaluator-only changes first:

1. Add `mixed_iid`, preserve legacy `mixed`.
2. Add eval-only `pause_mode={default,zero}` and serialize the actual range.
3. Snapshot pre-step motion phases: `paused`, `post_switch_or_resume_1s`, `pre_waypoint_1s`, `steady`.
4. Attribute dynamic collisions and exposure to each phase; record waypoint, pause, distance, velocity, and closing speed.
5. Reuse future-occupancy counterfactual audit and privileged teacher shadow. Keep teacher override separate.
6. After code review, compare D0 random_2d pause default vs zero on seeds 515/616/717.

Pause is causal only if:

- aggregate CR improves by at least 10pp;
- no seed worsens;
- at least two seeds improve by at least 8pp;
- TO rises by no more than 2pp.

Only after that repeat on D1 c10. Teacher override must independently pass `SR>=90%, CR<=10%, TO<=5%` before considering distillation.

## Guardrails

- No reward, network, training-config, replay-weight, or D2 changes before the evaluator A/B.
- Do not promote c10.
- Do not claim "more exposure never works"; only this 30/10/60, one-seed, 30-iter experiment failed.
- Do not claim pause mismatch is the cause before the thresholded A/B.

## Sources

- `docs/corridor_d1_sweep_result_for_codex.md`
- Obsidian: `isaaclab/課程學習/2026-07-26_D1加權走廊失敗與random2d機制稽核.md`
- run commit `8b9312e3bfa`, W&B `4vrqd6nv`

