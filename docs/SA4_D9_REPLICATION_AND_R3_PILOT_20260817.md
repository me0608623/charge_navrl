# SA4-D9 evaluator-seed replication and SA4-R3 pilot

Date: 2026-08-17  
Status: `COMPLETE_VALID_EVIDENCE`, `SA4_HOLD`  
Scope: one frozen checkpoint across three evaluator seeds, followed by one
training-seed SA4-R3 short pilot and a six-cell fixed screen

## 1. Decision question

The work was frozen as a two-stage decision:

1. Repeat the D9 `all_rays` versus `valid_return_only` closed-loop comparison
   on evaluator seeds 515 and 616, then combine those descriptive outcomes with
   the immutable seed-818 result.
2. Only if at least two of three seeds show lower total CR, wall-CR increase no
   greater than 0.5 percentage points, and TO increase no greater than 0.5
   percentage points, authorize one 50-iteration SA4-R3 pilot.

The rule authorized only the short pilot. It never authorized an automatic
training extension or SA5 launch.

## 2. D9 replication

Fixed checkpoint: SA4-R2 `checkpoint_6400.pt`  
Checkpoint SHA-256:
`c6dbd94bcbd6cc17ceca3cf8d5abfb4d956c98ed173c0dd890b9fc45959a2197`  
Cell: stage-4 lateral, d1=200 ms, 64 env, 2,500 steps per arm  
Protocol SHA-256:
`c1e1345e972aeeb5432b28819ff116c179f504ebecf9f448d3d867563d849ed1`

| seed | A CR all_rays | B CR valid_return_only | B-A CR | B-A wall CR | qualifies |
|---:|---:|---:|---:|---:|---:|
| 515 | 16.3951% | 15.4146% | -0.98 pp | +0.37 pp | yes |
| 616 | 18.5787% | 16.7074% | -1.87 pp | +0.18 pp | yes |
| 818 | 16.9534% | 15.8277% | -1.13 pp | +0.27 pp | yes |

All three fixed evaluator seeds qualified. TO was 0% in every arm. Descriptive
episode pooling gives:

| outcome | A all_rays | B valid_return_only | B-A |
|---|---:|---:|---:|
| episodes | 5,910 | 6,163 | +253 |
| SR | 82.6904% | 84.0175% | +1.33 pp |
| total CR | 17.3096% | 15.9825% | -1.33 pp |
| obstacle CR | 16.8359% | 15.2361% | -1.60 pp |
| wall CR | 0.4738% | 0.7464% | +0.27 pp |
| TO | 0.0000% | 0.0000% | 0.00 pp |

This establishes a replicated direction across three **evaluator seeds** for
this frozen checkpoint and lateral cell. It is not training-seed replication,
not a per-episode paired counterfactual, and not validation of a faithful
VLP-16 sensor model.

## 3. SA4-R3 training contract

Run: `sa4_r3_validreturn_from_sa3r1_c100_ne1024_s42_p50_r1`  
Parent: clean SA3 c100 `checkpoint_12800.pt`  
Parent SHA-256:
`e7da9aa0771252966f4d3cc7081adcbfdb35d9eaf828432ea8cd2fb50e453f88`

The config derives structurally from SA4-R2. Excluding metadata, its only
behavioral config difference is:

```text
lidar_distractor_eligibility: all_rays -> valid_return_only
```

Fixed conditions: stage 4, 1,024 env, training seed 42, rollout 128, K8,
full-noise preset, actuator delay U{0,1,2}, future-occupancy weight 0.15,
no optimizer resume, 50 iterations, checkpoints at it25 and it50.

Training completed 50/50 in 1,437 s (23.9 min), with 147,951 completed
episodes. It had no NaN, OOM, traceback, or corridor-family reconciliation
failure. KL max was 0.00807 and clip fraction max was 0.10134. The final
training iteration reported overall SR 90.55%, CR 9.39%, but this mixed
training metric is not a graduation result.

Checkpoints:

- it25 `checkpoint_3200.pt`, SHA-256
  `197dbbd2270a6f41ae917ff839950dcb8620329e415a85c21efc1ee21538e9f3`
- it50 `checkpoint_6400.pt`, SHA-256
  `fa51b5a312d38b7aa1ee93c3464b087b9b9cb985a7137eefdc8fecf40fed166d`

## 4. Fixed six-cell screen

Protocol: seed818, d1=200 ms, 64 env, stage-4 geometry,
`valid_return_only` at runtime. Minimum 1,000 completed episodes per cell.
Protocol SHA-256:
`8a5ec62fe473dcc86d5bca3b752466bf8556f4f708fe8634d12a958369844a51`.

| ckpt | scenario | n | SR | CR | TO | gate |
|---|---|---:|---:|---:|---:|---:|
| it25 | lateral | 2,122 | 78.75% | 21.25% | 0.00% | FAIL |
| it25 | longitudinal | 1,556 | 50.39% | 48.97% | 0.64% | FAIL |
| it25 | native | 2,242 | 95.09% | 4.91% | 0.00% | PASS |
| it50 | lateral | 2,215 | 81.13% | 18.87% | 0.00% | FAIL |
| it50 | longitudinal | 1,753 | 78.21% | 21.73% | 0.06% | FAIL |
| it50 | native | 2,222 | 94.06% | 5.94% | 0.00% | PASS |

Corridor threshold: SR >= 90%, CR <= 10%, TO <= 5%. Native threshold:
SR >= 92%, CR <= 7%, TO <= 3.5%.

It50 improved both corridor directions relative to it25, especially
longitudinal, but neither direction reached the absolute gate. Native retention
passed at both checkpoints. No checkpoint passed all three required cells, so
the fixed selector returned `selected_checkpoint=null`.

## 5. Decision

1. D9's observation-time improvement replicated on 3/3 evaluator seeds and
   validly authorized the bounded R3 pilot.
2. R3 learned during the 50-iteration budget, but the improvement was
   insufficient: it50 lateral CR remained 18.87% and longitudinal CR 21.73%.
3. SA4-R3 is not an accepted parent. Do not extend this recipe and do not start
   SA5 under the frozen decision rule.
4. The result does not prove that `valid_return_only` is harmful or ineffective
   in general. There is one training seed, and evaluator-seed replication does
   not measure training variance.
5. The next technical question is why a consistent inference-time sensitivity
   gain did not produce a gate-passing retrained policy. That question requires
   a separately frozen experiment; it must not be answered by silently
   extending this failed pilot.

## 6. Evidence paths

- D9 replication:
  `logs/gates/sa4_d9/noise_closed_loop_ab_replication_s515_s616_r1_20260817/`
- R3 run:
  `logs/rnn_car/sa4_r3_validreturn_from_sa3r1_c100_ne1024_s42_p50_r1/`
- R3 console:
  `logs/rnn_car/sa4_r3_validreturn_from_sa3r1_c100_ne1024_s42_p50_r1.console.log`
- R3 screen:
  `logs/gates/sa4_r3_checkpoint_screen/r1_20260817/`
- freezes:
  `docs/freeze/sa4_d9_seed_replication_v1.json`,
  `docs/freeze/sa4_r3_valid_return_noise_pilot_v1.json`, and
  `docs/freeze/sa4_r3_checkpoint_screen_v1.json`

## 7. Verification

- D9 replication: four new cells valid, prior seed818 manifest hash fixed,
  source fingerprints stable, no incomplete marker.
- R3 source hashes remained equal to the pre-training freeze.
- R3 tests: 891 passed + 27 subtests.
- Screen: 6/6 cell JSON present, `source_fingerprint_stable=true`, status
  `COMPLETE_VALID_SCREEN`.
- No evaluator or trainer process remained; GPU returned to the desktop daemon.
- `training_extension_started=false` and `sa5_started=false`.
