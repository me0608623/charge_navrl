# SA5-R2 fixed speed-0.7 four-checkpoint screen

## Status

- r1 started 2026-08-20 18:08:28 CST and fail-closed at 18:24:55.
- r1 output: `logs/gates/sa5_r2_speed0p7_checkpoint_screen/screen_20260820_r1/`.
- r1 completed only baseline-c250 lateral and longitudinal. Random-2D produced 645 completed episodes at 2,500 steps, below the frozen n>=1,000 floor. `INCOMPLETE_NO_VERDICT.json` exists; do not combine r1 cells with r2.
- r2 started 2026-08-20 18:28:33 CST and fail-closed at 19:53:06.
- r2 output: `logs/gates/sa5_r2_speed0p7_checkpoint_screen/screen_20260820_r2/`.
- r2 completed 13 valid cells. Adapt-it25 longitudinal produced n=829 at 2,500 steps, so cell 14 failed the n>=1,000 floor. Fingerprint at failure exactly matched launch (`0d3553cb...45d`). No verdict.
- v3 uses frozen sample-size-only tiers: lateral/longitudinal 2500/5000/7500; random-2D/mixed 5000/7500; native/narrow 1200/2400/3600. A retry occurs only for n<1,000, never based on outcomes.
- v3 resume verification imports 13 v2 first-tier cells only after checking protocol, cell hashes, source fingerprint, unchanged measurement sources, exact first-tier steps, and n>=1,000. Remaining 11 cells are fresh.
- Obsidian: `/home/aa/Documents/Obsidian Vault/isaaclab_v4/Gate結果/41_SA5_R2_speed0p7四Checkpoint六題型固定驗收_20260820.md`.
- r3 imported 13 cells and reached 19/24. It then fail-closed on intentional queue source drift while changing only the resource scheduler to the already-approved shared-memory guard.
- r4 started 2026-08-20 21:07:59 CST, imported 19 valid v3 cells, ran the final 5 under shared-memory-guarded mode, and completed 22:59:10.
- Final output: `logs/gates/sa5_r2_speed0p7_checkpoint_screen/screen_20260820_r4/`.
- Final status complete; before/after source fingerprint identical (`0ac85304...c99f`). No training or SA6.

## Frozen comparison

- Candidates: rate-1-trained baseline c250, natural rate-1 c300 control, fixed-rate-0.7 adapt-it25, adapt-it50.
- Scenarios: lateral, longitudinal, random-2D, mixed, native, narrow.
- Every cell uses speed_rate=0.7, speed_rate_obs=ego, deployment scale=1.0, full VLP-16 valid-return-only, fixed d1=200ms, evaluator seed 818.
- 24 cells with the v3 frozen tiers above; minimum 1000 completed episodes each.
- Protocol SHA: `1eaf8a42c2c8c0d86b2ed48032c8e57824ec8307d2bba3a9901b97bb394333ec`.
- CPU tests: 12 passed. Resume dry-run verified 13 imports. GPU smoke 32x64 passed earlier against the unchanged runtime contract.

## Preregistered decision

- Rank by min worst CR across four corridor families; secondary mean family CR.
- Clear difference is 0.005 absolute CR.
- Retention requires existing native/narrow hard gates and <=2pp regression relative to rate-0.7 baseline c250.
- Extend adapt-it50 only if it clearly beats all three comparators and retention passes.
- If adapt-it25 clearly beats it50, stop for late forgetting.
- If neither adaptation checkpoint clearly beats both controls with retention, move speed-rate earlier to SA3->SA4.
- Any missing/invalid/drifted cell produces no verdict. The screen never accepts a parent or starts training/SA6.

## Final result

- Original preregistered rule outcome (audit only): `STOP_IT50_LATE_FORGETTING`.
- Corrected evidence label: `ADAPTATION_DEGRADED_NO_IMPROVEMENT_AT_ANY_CHECKPOINT`.
- Ranking by worst-family CR: baseline c250 81.37%, c300 81.44%, adapt-it25 83.62%, adapt-it50 86.03%.
- Mean CR alone would misleadingly favor it50 (66.85%) because it improved lateral/longitudinal while degrading random-2D/mixed.
- Native retention: c250 90.69/8.94 SR/CR PASS; c300 89.30/10.48 FAIL; it25 87.31/12.62 FAIL; it50 86.58/13.36 FAIL. Narrow is saturated at 100/0 for all.
- Conservative n=1200 standardized comparisons: control vs baseline 0.04 SE; it25 vs baseline 1.45 SE worse-direction; it50 vs baseline 3.10 SE worse; it50 vs it25 1.65 SE worse. The 1.65-SE it25/it50 gap does not establish late forgetting because it25 never demonstrated improvement.
- The natural rate-1.0 c300 control is indistinguishable from c250 while the fixed-rate-0.7 it50 degrades. This supports attributing the observed it50 degradation to the changed speed contract rather than merely adding 50 training iterations, within the single-training/evaluator-seed boundary.
- c300 relative retention passes; its overall retention FAIL comes only from the native absolute gate (SR 89.30% <90%, CR 10.48% >10%), a roughly 1.2-SE boundary result. Preserve the hard-gate outcome but do not call it proven control degradation.
- Neither adaptation beats both controls. Do not extend it50. Before rebuilding SA3->SA4, run the preregistered no-training pedestrian-speed sweep so robot/pedestrian speed ratio is defined once rather than rebuilding the lineage twice.
- c250 is only the relative leader among four bad corridor results; it is not accepted as parent or deployment candidate.
- Single-seed boundary: paper-level causal claims still require replication. Original result files are preserved as `SUMMARY_PREREGISTERED_ORIGINAL.{json,md}`; corrected `SUMMARY.{json,md}` changes interpretation only, not measurements or protocol hash.
