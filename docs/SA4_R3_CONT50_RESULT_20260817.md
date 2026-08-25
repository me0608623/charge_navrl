# SA4-R3 c6400 Exact Continuation Result

Date: 2026-08-17  
Status: COMPLETE; NO ACCEPTED SA4 PARENT; SA5 NOT STARTED

## Question

Does the SA4-R3 policy continue improving when training is resumed from its
conceptual iteration-50 checkpoint with the same optimizer, noise, delay,
reward, scene, and network, or has lateral CR plateaued around 15-20%?

## Frozen continuation

- Parent: `sa4_r3_validreturn_from_sa3r1_c100_ne1024_s42_p50_r1/checkpoint_6400.pt`
- Parent SHA-256: `fa51b5a312d38b7aa1ee93c3464b087b9b9cb985a7137eefdc8fecf40fed166d`
- Optimizer: restored; checkpoint contained 38 non-empty RL optimizer states
- Added budget: 50 iterations / 6,400 steps; saves at conceptual it75 and it100
- Fixed: stage4, 1,024 env, seed42, rollout128, K8, full VLP-16 noise,
  `valid_return_only`, actuator delay U{0,1,2}, future-occupancy weight 0.15,
  corridor family weights 50/50, and all PPO/network fields
- Config lock: every non-lineage/non-metadata field equals the original R3 config
- Freeze: `docs/freeze/sa4_r3_cont50_from_r3_c6400_v1.json`

## Training health

The run completed 50/50 with exit status 0. Both checkpoints contain the
restored optimizer state. There were no non-finite metrics and corridor-family
reconciliation passed 50/50 iterations.

| Metric | Result |
|---|---:|
| max KL | 0.01044 |
| max clip fraction | 0.14095 |
| entropy, first to last | 4.0201 to 4.0223 |
| lateral CR, first 10 to last 10 | 27.63% to 18.85% |
| longitudinal CR, first 10 to last 10 | 23.57% to 5.96% |
| native CR, first 10 to last 10 | 8.31% to 8.49% |

The final iterations shared the GPU with an independently launched old R3 c6400
layout-split evaluation. FPS fell from about 4,400-4,600 to about 1,780-2,000,
so wall time increased to about 32.5 minutes. This is recorded as a timing
confound, not a recipe or lineage change; all frozen source hashes remained
stable.

Checkpoint hashes:

- it75: `checkpoint_3200.pt`, SHA-256
  `7bec45d4c86d7dd1819412a1bf14b1adcf0acb661d8f0fbb6c775f867136f9c2`
- it100: `checkpoint_6400.pt`, SHA-256
  `aeddf32b3e41d1920bfc839651f18e09d7384a8f6ab3a5ac6de3c11628520d7d`

## Fixed six-cell screen

Protocol: seed818, fixed d1=200 ms, 64 env, stage4,
`valid_return_only`, minimum 1,000 completed episodes. Corridor thresholds are
SR >= 90%, CR <= 10%, TO <= 5%; native thresholds are SR >= 92%, CR <= 7%,
TO <= 3.5%.

| checkpoint | scenario | n | SR | CR | TO | Gate |
|---|---|---:|---:|---:|---:|---|
| it75 | lateral | 2,257 | 83.96% | 16.04% | 0.00% | FAIL |
| it75 | longitudinal | 2,496 | 84.94% | 15.06% | 0.00% | FAIL |
| it75 | native | 2,246 | 94.57% | 5.43% | 0.00% | PASS |
| it100 | lateral | 2,190 | 88.08% | 11.92% | 0.00% | FAIL |
| it100 | longitudinal | 3,044 | 94.68% | 5.29% | 0.03% | PASS |
| it100 | native | 2,251 | 93.60% | 6.40% | 0.00% | PASS |

The six-cell bundle is complete, source-stable, and has no incomplete marker.
Its protocol SHA-256 is
`9ec8876c4d3e5362f02d608efd99320dd673bfbd912298e6aa31e0b402477a14`.

## Interpretation

The pre-specified plateau branch was not observed. Lateral CR moved from 18.87%
at the original it50 screen to 16.04% at it75 and 11.92% at it100. Longitudinal
CR moved from 21.73% to 15.06% and then 5.29%. This is evidence that exact
optimizer continuation was still learning under this single training lineage.

However, the acceptance contract is absolute. It100 misses both lateral limits
by 1.92 percentage points: SR is 88.08% and CR is 11.92%. Native also moved from
5.43% CR at it75 to 6.40% at it100, leaving only a 0.60 percentage-point margin
to the native CR limit. Therefore:

1. `selected_checkpoint=null`; no checkpoint passes all three cells.
2. it100 is the lowest-worst-CR diagnostic checkpoint tested here, not an
   accepted SA4 parent.
3. SA5 must remain stopped.
4. This result does not yet establish that lateral exposure is insufficient;
   lateral improved materially rather than plateauing in the specified band.
5. Any further run requires a new bounded authorization. The smallest defensible
   next probe is an exact +25-iteration continuation to conceptual it125, with
   the same three-cell screen and a hard stop if lateral or native fails.

## Evidence boundary

This is one training seed and one fixed evaluator seed. It supports the observed
within-lineage trajectory and the fixed Gate verdict; it does not establish a
general causal learning curve. Training-window scene metrics are not Gate
substitutes, and the it100 policy must not be described as SA4-qualified.

## Artifacts

- Training: `logs/rnn_car/sa4_r3_cont50_from_r3c6400_ne1024_s42_p50_r1/`
- Console: `logs/rnn_car/sa4_r3_cont50_from_r3c6400_ne1024_s42_p50_r1.console.log`
- Screen: `logs/gates/sa4_r3_cont50_checkpoint_screen/r1_20260817/`
- Training freeze: `docs/freeze/sa4_r3_cont50_from_r3_c6400_v1.json`
- Screen freeze: `docs/freeze/sa4_r3_cont50_checkpoint_screen_v1.json`
