# SA4-D9 replication and SA4-R3 pilot result

Date: 2026-08-17
Status: COMPLETE; SA4 HOLD; SA5 NOT STARTED

## Authoritative decision

D9 `all_rays` versus `valid_return_only` was repeated on pre-fixed evaluator
seeds 515 and 616 and combined with immutable seed818. All 3/3 seeds had lower
total CR under valid-return-only while wall-CR increases stayed below the
pre-registered +0.5 pp tolerance and TO stayed unchanged. This authorized only
one 50-iteration SA4-R3 pilot.

Per-seed B-A total CR / wall CR:

- s515: -0.98 pp / +0.37 pp
- s616: -1.87 pp / +0.18 pp
- s818: -1.13 pp / +0.27 pp

These are evaluator-seed replications of a frozen checkpoint, not independent
training-seed replications and not proof of a faithful VLP-16 model.

SA4-R3 started from clean SA3 c100 SHA
`e7da9aa0771252966f4d3cc7081adcbfdb35d9eaf828432ea8cd2fb50e453f88`.
Relative to SA4-R2, the only non-metadata config difference was
`lidar_distractor_eligibility: all_rays -> valid_return_only`. It used 1024 env,
seed42, K8, rollout128, full noise, actuator delay U{0,1,2}, no optimizer resume,
and ran 50/50 iterations in 1437 s without NaN/OOM/traceback or reconciliation
failure.

Fixed seed818/d1 screen:

| checkpoint | lateral SR/CR/TO | longitudinal SR/CR/TO | native SR/CR/TO |
|---|---|---|---|
| it25 | 78.75/21.25/0.00% FAIL | 50.39/48.97/0.64% FAIL | 95.09/4.91/0.00% PASS |
| it50 | 81.13/18.87/0.00% FAIL | 78.21/21.73/0.06% FAIL | 94.06/5.94/0.00% PASS |

No checkpoint passed lateral + longitudinal + native. `selected_checkpoint=null`,
`training_extension_started=false`, `sa5_started=false`. Do not extend the same
recipe or use R3 as parent without a new explicitly frozen decision.

Evidence:

- `logs/gates/sa4_d9/noise_closed_loop_ab_replication_s515_s616_r1_20260817/`
- `logs/rnn_car/sa4_r3_validreturn_from_sa3r1_c100_ne1024_s42_p50_r1/`
- `logs/gates/sa4_r3_checkpoint_screen/r1_20260817/`
- `docs/SA4_D9_REPLICATION_AND_R3_PILOT_20260817.md`
