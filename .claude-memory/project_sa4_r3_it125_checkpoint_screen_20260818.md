# SA4-R3 conceptual it125 three-scenario checkpoint screen

Date: 2026-08-18
Status: COMPLETE_VALID_SCREEN; frozen screen FAIL; HUMAN WAIVER RECORDED; SA5 EXPLICITLY STARTED

The user authorized the fixed three-scenario acceptance screen after the exact
optimizer continuation from conceptual it100 to it125 completed. The candidate
is exclusively:

- run: `sa4_r3_cont25_from_it100_ne1024_s42_p25_r1`
- checkpoint: `checkpoint_3200.pt`
- conceptual iteration: 125
- SHA-256: `57f43d07255971ad607e0bf3234dd7d46984c9c71cc9550e2e9461fbc95ab71d`

The training freeze confirms the it100 optimizer was restored and the recipe
remained stage4 / 1024 env / seed42 / rollout128 / 83D-K8 / full VLP-16 noise /
`valid_return_only` / actuator delay U{0,1,2}. Training completed 25/25 with a
final checkpoint and no NaN/OOM/traceback.

Frozen screen:

| Scenario | Meaning | Steps | Absolute gate |
|---|---|---:|---|
| corridor_lateral | dynamic obstacles cross the corridor width and robot path | 2500 | n>=1000, SR>=0.90, CR<=0.10, TO<=0.05 |
| corridor_longitudinal | dynamic obstacles travel along the corridor, including approaching and same-direction lanes | 4000 | n>=1000, SR>=0.90, CR<=0.10, TO<=0.05 |
| nav_native | unforced stage4 native distribution, testing general-navigation retention | 1200 | n>=1000, SR>=0.92, CR<=0.07, TO<=0.035 |

All three cells must independently pass. No averaging can rescue a failed cell.
A three-cell pass only permits consideration of the larger formal SA4 matrix;
the screen itself does not accept a parent or start SA5.

Evidence boundary: the training-time config, trainer, env override, LiDAR
observation and actuator hashes still match exactly. `play_rnn_car.py` has
additive default-off diagnostic and teacher instrumentation since the it100
screen, so old it100-to-new-it125 deltas are descriptive rather than a strict
same-source paired estimate. The it125 absolute Gate verdict remains the
preregistered decision rule.

Runtime blocker observed before launch: NVML reports two dead process IDs
(`2283874`, `2300635`) retaining about 30 GiB VRAM. They belong to two other-user
(`cm`) YOLO runs whose main processes died while 17 orphaned `yolo detect train`
/ `pt_data_worker` children remained under PPID 1. Host memory is also exhausted
(60/62 GiB RAM, 4/4 GiB swap). This is not an SA4 process and must not be killed
without owner/admin authorization.

The transient user unit `sa4-r3-it125-screen-wait.service` was started at
2026-08-18 14:24:31 CST. It performs no GPU work while waiting. It launches the
frozen queue only after the other-user YOLO workers disappear, MemAvailable is
at least 16 GiB, and GPU free memory is at least 8 GiB. The queue then repeats
the GPU-busy, checkpoint-hash, training-source and evaluation-source checks.
Waiter log: `logs/gates/sa4_r3_it125_checkpoint_screen/waiter_20260818.log`.

Protocol sources:

- `scripts/reinforcement_learning/skrl/rnn_car_wdclean/sa4_r3_it125_checkpoint_screen.py`
- `scripts/reinforcement_learning/skrl/rnn_car_wdclean/run_sa4_r3_it125_checkpoint_screen.py`
- `scripts/reinforcement_learning/skrl/rnn_car_wdclean/sa4_r3_it125_checkpoint_screen_queue.py`
- `docs/freeze/sa4_r3_it125_checkpoint_screen_v1.json`

Planned output:
`logs/gates/sa4_r3_it125_checkpoint_screen/r1_20260818/`.

## Final result and user decision

Resources recovered at 14:41:31 CST and the queue completed all three cells at
14:57:50 CST. There were no invalid cells, source fingerprints were stable,
and no NaN/OOM/traceback occurred.

| Scenario | Episodes | SR | CR | TO | Frozen result |
|---|---:|---:|---:|---:|---|
| corridor_lateral | 2236 | 0.8895349 | 0.1104651 | 0.0 | FAIL |
| corridor_longitudinal | 3254 | 0.9698832 | 0.0301168 | 0.0 | PASS |
| nav_native | 2208 | 0.9393116 | 0.0602355 | 0.0004529 | PASS |

The preregistered verdict remains `screen_pass=false`, `accepted_parent=false`,
because lateral missed both SR and CR limits by 0.010465, or about 1.05
percentage points. The original summary JSON must remain immutable.

After disclosure of the complete result, the user explicitly judged this
approximately 1.05 pp lateral miss acceptable. Record this as a human waiver
for subsequent engineering decisions and treat it125 as a conditionally
acceptable candidate. Do not rewrite the frozen screen as PASS, claim
cross-seed robustness, or automatically start SA5. A later launch still
requires an explicit operational decision.

Final summary:
`logs/gates/sa4_r3_it125_checkpoint_screen/r1_20260818/sa4_r3_it125_checkpoint_screen_summary.json`.

## Subsequent explicit SA5 decision

The user later explicitly authorized SA5 after config lock, CPU tests and GPU
smoke. The first formal SA5 run used the inherited actuator delay U{0,1,2}, but
was deliberately stopped at iteration 5 when new real-robot actuator evidence
was raised. Its console and five metric rows remain preserved; it produced no
checkpoint.

Reanalysis of the raw robot CSVs rejected the session-wide 0.28--0.44 angular
ratio as a steady-state actuator gain: it mixed a pre-motion interval and a
likely manual/mux takeover interval and did not align command/response delay.
Five clean moving windows give aligned static gains 0.983--1.046 (median about
1.013) and best lags 350--400 ms. Therefore the replacement SA5 keeps angular
scale 1.0 and changes only the delay support from U{0,1,2} (mean 200 ms) to
U{1,2} (mean 300 ms).

Active replacement:

- run: `sa5_sim2real_v2_from_sa4r3_it125_hwaiver_actd12_ne1024_s42_p300_r1`
- unit: `sa5-sim2real-v2-from-sa4r3-it125-actd12-p300-r1.service`
- config: `e2e_sa5_k8_obb_sim2real_from_sa4r3_it125_hwaiver_actdelay12`
- W&B: `2j76v4he`
- budget: 300 iterations; checkpoints every 50 iterations
- first metric: it1, SR 0.67966, CR 0.10701, TO 0, reconciliation 1.0
- auto-advance: HALTED_ALERT; SA6 is not authorized

This operational launch does not alter the frozen SA4 screen FAIL.
