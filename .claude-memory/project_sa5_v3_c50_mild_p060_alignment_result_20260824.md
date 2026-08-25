# SA5-v3 c50 mild weighting and P060-alignment result (2026-08-24)

- Stage-2 A2/B2 parent: equal-control it25 checkpoint `d9095e7d...3419a` with
  exact RL optimizer resume.
- A2 kept equal/default motion allocation; B2 changed only motion slot weights to
  `(0.30, 0.30, 0.40)` for lateral/longitudinal/random-2D and ran 25 iterations.
- Fixed 4S2D B2-minus-A2 CR effects: lateral `-4.17 pp`, longitudinal `-4.64 pp`,
  random-2D `-4.36 pp`, mixed `-2.66 pp`. This passed the paired development rule,
  but all absolute CR values remained `54-70%`.
- B2 retention: native CR `14.28%` FAIL, narrow `0%` PASS, P060 0S1D `8.71%`
  PASS, P060 1S1D `13.36%` FAIL. Relative to original c50, P060 CR worsened by
  `+3.61/+4.93 pp` (descriptive z `4.50/5.05`), so B2 was rejected as parent.
- Stage-3 paired arms both restarted from B2 SHA `45c972f8...aee30` with the same
  38-entry PPO optimizer and ran 25 iterations. A3 retained P080 `0.70-0.90 m/s`
  for 0S1D/1S1D. B3 changed only those two ranges to P060 `0.50-0.70 m/s`.
- A3 checkpoint SHA `ae6c2495...dde13`; B3 SHA `5645f846...04f2b8`. Both 25/25,
  no non-finite metrics, no reconciliation failures, normal completion.
- Frozen Phase-A, seed 818/d1/speed-rate 0.7, A3 vs B3:
  - P060 0S1D CR `12.05% -> 7.09%` (`-4.96 pp`, z `-5.39`), B3 PASS.
  - P060 1S1D CR `20.21% -> 14.10%` (`-6.11 pp`, z `-5.30`), B3 FAIL.
  - P080 0S1D CR `16.50% -> 14.22%` (`-2.27 pp`, z `-2.06`).
  - P080 1S1D CR `30.99% -> 25.61%` (`-5.38 pp`, z `-4.09`).
  - All TO were zero; P080 relative retention passed.
- Decision: `REJECT_P060_ALIGNMENT_NO_PHASE_B`. Alignment improved every fixed cell
  and caused no observed P080 tradeoff, but P060 1S1D remained `4.10 pp` above the
  absolute CR gate. Speed mismatch was a real contributor, not the sole bottleneck;
  static-dynamic interaction remains.
- Queue incident: all 8 cells, SUMMARY, and stable before/after fingerprints were
  written, then a presentation-only legacy-field KeyError occurred. Original
  `INCOMPLETE.json` retained. Independent recovery revalidated all cells, all recorded
  source hashes, fingerprints, and exact recomputed verdict; authority status is
  `COMPLETE_VALID_POSTPROCESS_RECOVERED`.
- No Phase-B, no selected parent, no extension, no SA5 graduation, no SA6.
- B3 training-distribution corridor CR by five-iteration window was
  `39.48/41.29/39.15/34.73/36.64%`: best at it16-20, then rebounded `+1.91 pp`.
  Continued improvement is not established, and these mixed-profile windows cannot isolate
  P060 1S1D.
- Fixed-evaluator Wilson intervals sharpen the absolute result: B3 P060 0S1D is
  `7.09% [6.03, 8.31]`, while P060 1S1D is `14.10% [12.65, 15.68]`.
  This rules out a threshold-edge episode-sampling ambiguity for this evaluator seed, but does
  not replace training/evaluator-seed replication.
- Keep matched causality separate from lineage recovery. B3 beats matched A3 by `6.11 pp` on
  P060 1S1D, but is `+0.74 pp` worse than its B2 starting checkpoint and `+5.67 pp` worse than
  original c50. It reduced further degradation; it did not restore the historical capability.
- B3 P060 1S1D is strongly layout-stratified: layout 100 CR `22.55%` versus layout 101
  `5.70%`. Obstacle CR is `13.70%` and wall CR `0.39%`. Console scheduler counters further
  record 272 dynamic-patrol and 16 static slot-contact events (94.44% dynamic) for B3 P060
  1S1D. These are non-exclusive events: 288 events exceed 278 obstacle-terminal episodes by
  10, so 94.44% must not be reported as an episode fraction. The evidence supports a static
  geometry constraint followed mainly by pedestrian contacts, but does not provide exclusive
  episode-level or layout-specific collision attribution.
- Source geometry identifies layout 100 as the robot-left first static obstacle and layout 101
  as the right-side static skeleton. These are not exact episode-paired mirrors: dynamic start,
  active target direction and patrol phase are sampled separately and are absent from the layout
  key. Across 13 runtime-compatible unique checkpoint SHAs, 13/13 have higher
  layout-100 CR (median gap `11.67 pp`). Angular bins and slew clamps are symmetric, ruling out
  a static decoder mapping bug but not policy/observation/training asymmetry or an unreported
  dynamic-state imbalance. All use evaluator seed 818 and span six historical protocol hashes,
  so this remains retrospective diagnosis. A causal left-right claim requires full-scene paired
  mirrors and additional evaluator seeds.
- B3 exposure accounting: long-corridor resets total `6107`, completed corridor episodes `5910`,
  and mean corridor active fraction `18.94%`. P060 1S1D is 10% of a 10% corridor reset branch,
  nominally 1% of all assignments and about `611` expected resets. The failed fixed gate is the
  lateral family, weighted at 30%, so exact gate-matched exposure is only about `182-183` resets
  (`0.3%` overall), roughly 91 expected static-left resets. This makes low exposure a live
  hypothesis, not a cause: profile-by-family active steps and training SR/CR/TO were not logged.
  Add density-speed-by-family accounting before any authorized continuation.
- Suggested only with separate authorization: exact B3 optimizer continuation for at most
  25 iterations, save every five iterations, and fixed-screen intermediates before choosing
  one. If P060 1S1D remains above 10%, stop adding iterations and test staged 0S1D -> 1S1D
  density curriculum.
- Authority:
  `logs/gates/sa5_v3_c50_stage3_phase_a_screen/screen_20260824_r1/SUMMARY.json`
  SHA `cf707656...b71`, recovery SHA `2b7f93cc...e8fa`, completion record
  `docs/freeze/sa5_v3_c50_stage3_p060_alignment_completion_20260824.json`, and locked
  posthoc supplement `docs/freeze/sa5_v3_c50_stage3_posthoc_diagnostic_supplement_20260824.json`
  SHA `5da94c77...0d823`.
