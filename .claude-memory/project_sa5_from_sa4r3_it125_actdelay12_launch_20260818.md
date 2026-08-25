# SA5 from human-waived SA4-R3 it125 with delay U{1,2}

Date: 2026-08-18
Status: COMPLETE 300/300; AWAITING FIXED CHECKPOINT SCREEN

Parent:

- checkpoint: `logs/rnn_car/sa4_r3_cont25_from_it100_ne1024_s42_p25_r1/checkpoint_3200.pt`
- conceptual iteration: 125
- SHA-256: `57f43d07255971ad607e0bf3234dd7d46984c9c71cc9550e2e9461fbc95ab71d`
- frozen SA4 screen remains FAIL because lateral missed SR/CR by 1.0465 pp
- user explicitly accepted that miss as a human waiver for engineering progression

The first formal SA5 run, without the delay correction, was stopped at it5 after
new robot actuator evidence was raised. It is preserved as diagnostic evidence:

- run: `sa5_sim2real_v2_from_sa4r3_it125_hwaiver_ne1024_s42_p300_r1`
- 5/300, 640 steps, no checkpoint
- stop requested 15:31:34 CST; unit stopped 15:33:04 CST

Raw CSV reanalysis found near-unity angular gain and a 300--400 ms delay. The
replacement changes only `actuator_delay_range` from (0,2) to (1,2). It retains
angular scale (1,1), motor lag alpha 1.0, parent, model, reward, PPO, sensor,
scene, budget and teacher settings.

Active formal run:

- run: `sa5_sim2real_v2_from_sa4r3_it125_hwaiver_actd12_ne1024_s42_p300_r1`
- unit: `sa5-sim2real-v2-from-sa4r3-it125-actd12-p300-r1.service`
- config: `e2e_sa5_k8_obb_sim2real_from_sa4r3_it125_hwaiver_actdelay12`
- W&B: `2j76v4he`
- 1024 env, seed42, rollout128, 300 iterations, checkpoint every 50 iterations
- cross-stage optimizer reset; policy/model weights loaded from the parent

Preflight:

- corrected config tests: 19 passed
- full CPU suite: 965 passed, 27 subtests
- GPU smoke: 64 env, 1/1, exit 0, checkpoint_128.pt, reconciliation 1.0
- formal startup markers: Stage5, 15x15 m, full VLP16, valid_return_only,
  narrow 1.3--1.5 m, corridor 4.2 m, 3S2D/4S2D, delay U{1,2}, scale 1.0

Formal first metric:

- it1/300, total_steps 128, total_episodes 2925
- SR 0.679658, CR 0.107009, TO 0
- entropy 3.823215, KL 8.26e-8, clip 0, vf 13.6839, fps 2882
- native SR/CR 0.89069/0.10931
- narrow SR/CR 1.0/0
- corridor SR/CR 0.32911/0.67089
- corridor reconciliation_ok 1.0

The first metric is a Stage5 starting point, not checkpoint acceptance evidence.
Supervisor manually verified the expected run, active trainer and healthy latest
line. Auto-advance remains HALTED_ALERT and must not launch SA6.

Metadata:
`logs/rnn_car/sa5_sim2real_v2_from_sa4r3_it125_hwaiver_actd12_ne1024_s42_p300_r1/run_metadata.yaml`.

## Completion

Training completed normally at 2026-08-18 19:45:07 CST:

- 300/300, 38,400 steps, 14,480 s (241.3 min)
- six checkpoints at it50/100/150/200/250/300
- final checkpoint SHA-256:
  `5cc978df7ce38cbf9412844734377fc089cb389a791165cc747bb81575b53f79`
- final row: SR 0.93263, CR 0.06430, TO 0, entropy 4.09195,
  KL 0.004365, clip 0.04379, vf 5.3915, fps 2807
- no NaN/OOM/traceback, 0 reconciliation failures, 0 nonfinite actions
- supervisor transitioned COMPLETE_TO_IDLE at 19:50:02 and cleared expected_run
- auto-advance remains HALTED_ALERT; SA6 did not start

Episode-weighted first50 -> last50:

- overall SR 0.8685 -> 0.9297; CR 0.1219 -> 0.0662
- native CR 0.0915 -> 0.0840
- narrow CR 0.00136 -> 0.00074
- corridor CR 0.5632 -> 0.0495
- lateral CR 0.4361 -> 0.0501
- longitudinal CR 0.6251 -> 0.0398
- random2d CR 0.5691 -> 0.0394
- mixed CR 0.5760 -> 0.0599

This is strong training-distribution learning evidence, not a fixed Gate verdict.
Do not assume c300 is best. The next valid decision is a frozen checkpoint screen
covering each corridor family plus native and narrow retention.

## Checkpoint screen queued (2026-08-19)

See `project_sa5_checkpoint_screen_20260819.md`. The frozen two-phase queue is
running as `sa5-fixed-checkpoint-screen-r1.service`, currently waiting for the
shared GPU because cm's YOLO process is active. No Isaac cell, training, or SA6
has started yet.
