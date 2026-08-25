# SA5-R2 c250 speed-rate screen and 0.7 learnability decision

## Authority

- Obsidian canonical note: `/home/aa/Documents/Obsidian Vault/isaaclab_v4/Gate結果/39_SA5_R2_c250速度縮放Screen與0p7可學性決策_20260820.md`
- Frozen protocol: `docs/freeze/sa5_r2_c250_speed_scale_screen_v2.json`
- Output: `logs/gates/sa5_r2_speed_scale_screen/screen_20260820_r3_vehicle_semantics/`
- Status: `COMPLETE_VALID_SINGLE_SEED_DIAGNOSTIC`

## Fixed checkpoint and evidence boundary

- SA5-R2 c250: `sa5_r2_sealed_ldmix_from_sa4r3_it125_actd12_ne1024_s42_p300_r1/checkpoint_32000.pt`
- SHA-256: `df14c45d8dd27b0e327f9447a9a613bf62eaa850af542b474aad682dcd937a4a`
- Fixed sealed Stage-5 lateral 4S2D, seed 818, 64 env, 3000 steps, d1=200ms, full VLP-16 + valid-return-only.
- Single checkpoint, evaluator seed, and high-density lateral scenario. Diagnostic only; no accepted parent and no SA6 authorization.

## Result

| rate | SR | CR | TO | body speed | body stop | warning p50 | response delay p50 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.0 | 61.18% | 38.82% | 0% | 0.4113 m/s | 18.01% | 1.8s | 0.6s |
| 0.8 | 51.19% | 48.81% | 0% | 0.3604 m/s | 18.06% | 2.0s | 0.6s |
| 0.7 | 41.23% | 58.77% | 0% | 0.3323 m/s | 18.40% | 2.0s | 0.8s |
| 0.6 | 32.22% | 67.73% | 0.06% | 0.2939 m/s | 21.28% | 1.8s | 0.8s |

At 0.7 vs 1.0: actual speed -19.2%, SR -19.95pp, CR +19.95pp, body stop only +0.39pp. Lower post-hoc rate did not make the mature policy safer.

Current rover semantics jointly scale four action limits by rate and ego/goal/action-history observations by 1/rate while leaving LiDAR unchanged. This is not a simple downstream velocity multiplier. Exact cause of degradation remains unresolved.

## Decision

- Deployment requirement updated: real operation is a tunable `speed_rate` band 0.6～0.8, with 0.7 as the current nominal value; user reports 0.7 felt usable in a real test and 0.6 felt too slow. This qualitative usability observation is not a stress-test safety result.
- Withdraw the old preliminary claim that training at 0.7 is unnecessary.
- The screen contains no gradient updates and cannot show whether RL can learn earlier deceleration/turning at 0.7.
- Do not restart SA1-SA5 yet and do not start an unbounded c250 continuation.
- The proposed experiment was formally launched on 2026-08-20 16:36:53 CST: exact c250 + optimizer, only current vehicle `speed_rate=0.7`, 50-iteration learnability pilot, save adaptation it25/it50. See [project_sa5_r2_c250_speed0p7_adapt_launch_20260820.md](project_sa5_r2_c250_speed0p7_adapt_launch_20260820.md).
- Existing same-lineage c300 is the natural rate=1.0 +50-iteration continuation control. Evaluate c250 baseline, c300 control, and 0.7 adaptation checkpoints under the same fixed rate=0.7 protocol.
- Track lateral/longitudinal/native/narrow plus first sustained deceleration and turn after warning. Stop fraction alone is insufficient.
- If adaptation clearly beats baseline/control without retention loss, extend a new speed-aware lineage. If not, introduce rate earlier at the SA3→SA4 dynamic-obstacle boundary; do not repeat pure-navigation SA1/SA2 by default.
- Even though final deployment spans 0.6～0.8, keep the first 50-it learnability pilot fixed at 0.7 to preserve causal interpretation. If it passes, formal training should use preregistered episode-stratified rates, provisionally 0.6/0.7/0.8 = 20/60/20, and rank checkpoints by worst rate × required family rather than average performance.
- If operators change rate during an episode, add an explicit rate-transition curriculum and transition screen; per-episode randomization alone is insufficient evidence.
- The bounded learnability pilot is running. SA6 remains unauthorized and auto-advance remains HALTED_ALERT.
