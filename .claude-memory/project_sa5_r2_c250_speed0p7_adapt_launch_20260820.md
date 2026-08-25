# SA5-R2 c250 speed0p7 adaptation pilot launch

## Status

- Formally launched: 2026-08-20 16:36:53 CST.
- Unit: `sa5-r2-c250-speed0p7-adapt-p50-r1.service`.
- Run: `sa5_r2_c250_speed0p7_adapt_ne1024_s42_p50_r1`.
- Config: `e2e_sa5_r2_c250_speed0p7_adapt_p50`.
- Obsidian canonical note: `/home/aa/Documents/Obsidian Vault/isaaclab_v4/Gate結果/40_SA5_R2_c250_speed0p7可學性Pilot啟動_20260820.md`.
- SA6 not started; auto-advance remains `HALTED_ALERT`.

## Lineage and intervention

- Exact parent: `logs/rnn_car/sa5_r2_sealed_ldmix_from_sa4r3_it125_actd12_ne1024_s42_p300_r1/checkpoint_32000.pt` (conceptual c250).
- Parent SHA-256: `df14c45d8dd27b0e327f9447a9a613bf62eaa850af542b474aad682dcd937a4a`.
- Both RL and auxiliary optimizer states are resumed; this is true continuation, not weights-only transfer.
- The only behavioral intervention is fixed `speed_rate=0.7` with `speed_rate_obs=ego` and downstream `deployment_speed_scale=1.0`.
- Effective limits: linear velocity 0.7 m/s, linear acceleration 0.35 m/s², angular velocity 0.84 rad/s, angular acceleration 2.1 rad/s².
- LiDAR is not scaled. Raw ego/goal/action-history observations are converted to the 0.7 policy frame before normalization.
- Existing rate-1.0 c300 `checkpoint_38400.pt` is the natural +50-iteration continuation control, SHA `97a4ccdf929a0fcab9adff5d80ae7a1c7475067e3049f11571e04fd164a524db`.

## Fixed contract

- Stage 5, 1024 env, seed 42, rollout 128, PPO, K8/83D.
- Full VLP-16 `valid_return_only` noise.
- Actuator delay U{1,2}, scale 1.0, lag 1.0.
- Sealed density-mix scene and reward/network remain unchanged.
- Budget: 50 iterations / 6,400 steps; save adaptation it25 and it50.
- Protocol: `docs/freeze/sa5_r2_c250_speed0p7_adapt_pilot_v1.json`, SHA `2f76986eed931ab89aefa4042c0bddf3bd4a6534a61452d93de1e4d4caf65871`.
- Run metadata: `logs/rnn_car/sa5_r2_c250_speed0p7_adapt_ne1024_s42_p50_r1/run_metadata.yaml`.

## Verification and evidence boundary

- Related CPU tests: 37 passed.
- GPU smoke: 64 env, one PPO iteration, both optimizer states loaded, checkpoint written, no NaN/OOM/traceback.
- Formal run initial health was normal; at iteration 9: SR 89.11%, CR 10.35%, TO 0%, KL 0.00824, clip 0.09642, entropy 4.021, FPS 3067, reconciliation_ok=1.
- These are training-distribution startup metrics, not a fixed Gate or efficacy conclusion.
- At adaptation it25/it50, compare c250 baseline, natural c300 control, and adaptation checkpoints under identical `speed_rate=0.7` fixed lateral/longitudinal/native/narrow screens.
- Do not assume it50 is best and do not authorize SA6 from training metrics.

## Progress 2026-08-20 17:00 CST

- Cron snapshot: 30/50, active/running, PID 1334524, no recent NaN/OOM/traceback.
- `checkpoint_3200.pt` (adaptation it25) saved successfully at 16:56:15.
- During data collection the run advanced to 32/50.
- At it30: overall SR/CR/TO 86.32/12.97/0.00%; native CR 11.92%, narrow CR 0%, corridor CR 42.34%.
- Last-five episode-weighted CR at the snapshot: overall 12.01%, native 11.00%, corridor 40.13%.
- Health remains GREEN: KL max 0.00857, clip max 0.10050, no reconciliation failures, GPU normal.
- Efficacy remains unresolved/YELLOW: corridor is noisy and has not shown stable improvement. Let the frozen 50-it budget finish; fixed screens, not training metrics, decide learnability.

## Completion 2026-08-20 17:14 CST

- Completed 50/50 and 6,400 steps in 2,140 s (35.7 min); systemd result success.
- Checkpoints: `checkpoint_3200.pt` (adapt-it25) and `checkpoint_6400.pt` (adapt-it50). Final SHA: `812a495902f7c54c638096ba113b1b44266ddcc41431d476e6919302ee136f1c`.
- No NaN/OOM/traceback; source/protocol hashes stable; all reconciliation checks passed.
- Supervisor auto-transitioned COMPLETE_TO_IDLE at 17:20, cleared expected_run, and did not launch SA6.
- Final single iteration: overall SR/CR/TO 88.07/10.96/0.00%; native CR 10.91%, narrow CR 0%, corridor CR 38.17%.
- Episode-weighted first10 to last10 CR: lateral 34.20% -> 30.53% (-3.67pp), longitudinal 26.48% -> 28.57% (+2.09pp), corridor overall 37.23% -> 37.27% (+0.04pp), native 8.66% -> 10.44% (+1.78pp).
- Decision: training-health PASS only. Speed-0.7 learnability is not yet accepted because lateral improvement was offset by longitudinal/native regression and corridor overall was flat. Run identical fixed-rate-0.7 screens for c250 baseline, natural c300 control, adapt-it25, and adapt-it50 before choosing a parent.
