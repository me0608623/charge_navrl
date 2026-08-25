---
name: SA4-D8 mixed-pixel eligibility paired-shadow result
description: No-return/dropped-ray near ghosts are a major contributor to frozen-D4 no-feasible active frames, but not the full cause
type: finding
date: 2026-08-17
status: complete-diagnostic
---

# SA4-D8 current vs valid-return-only noise result

## Fixed protocol

- SA4-R2 c6400, SHA `c6dbd94bcbd6cc17ceca3cf8d5abfb4d956c98ed173c0dd890b9fc45959a2197`
- stage4 lateral, seed818, fixed d1=200ms, 64 env x 2500 steps
- one fresh identity-baseline rollout; policy/simulator always use current full noise
- paired shadow reuses identical ray hits, bias, sigma, dropout/distractor masks and values
- counterfactual distractor eligibility = `raw_valid AND NOT hole`; ineligible replacements restore the dropout-stage range
- both sweeps use actual winner-ray angles and frozen D4 geometry on the same active frames
- no action modification, training, SA5, or corrected observation fed to policy
- protocol `sa4_d8_noise_eligibility_sensitivity/v1`, SHA `1fcce8d769fef404d01304b184eeef96a073cce7b4275849184a1a850e701185`

## Validation

- 864 tests + 27 subtests pass
- 160,000/160,000 D3 records, 0 action identity errors
- 0/157,960 d1 alignment errors
- 2,500/2,500 exact policy-trace calls
- 0 corrected-range monotonicity errors; current/corrected dynamic grids bitwise equal
- all reconciliation true, source fingerprint stable, no NaN/OOM/traceback
- current policy exactly reproduces baseline n=1976, SR=83.05%, CR=16.95%, TO=0%; these are not corrected-noise outcomes

## Noise ledger

- raw ray slots: 921,600,000
- realized distractors: 2,316,395 = 0.2513%, matching configured 0.2515%
- eligible existing surviving returns: 1,097,155 = 47.36% of distractors
- removed no-return/dropped-ray distractors: 1,219,240 = 52.64%
- current winner bins removed: 1,028,455
- changed 72-bin values: 1,028,144 = 8.93% of all bin slots
- frames with at least one changed bin: 159,808/160,000 = 99.88%

## Paired frozen-D4 result

- active frames: 24,534
- both feasible 4,202; current-only 34; corrected-only 6,157; both-no-feasible 14,141
- current jointly feasible 17.27%, no-feasible 82.73%
- valid-return-only jointly feasible 42.22%, no-feasible 57.78%
- rescued current no-feasible = 6,157/20,298 = 30.33%
- net recovery = 6,123 frames = +24.96pp of active
- static-any feasible 21.42% -> 50.63%
- feasible action count 583,183 -> 1,894,010 (3.25x)
- preregistered descriptive `major_fake_obstacle_contributor=True` (required rescue>=25%, net>=5%, >=100 frames)

Conclusion: under this fixed cell and frozen D4 model, near ghosts created on no-return or dropped rays are a major source of no-feasible states. They are not the full cause: corrected no-feasible remains 57.78%. Do not claim corrected-policy collision improvement because the shadow was not fed to policy. Do not call valid-return-only a faithful VLP-16 model; U(0.2,2m), edge/ring/range/material conditions and the per-return interpretation remain unsupported.

Decision update: the separately frozen D9 fresh closed-loop A/B has completed. In the same fixed c6400 lateral/d1/s818 cell, valid-return-only produced SR +1.13pp and CR -1.13pp, but remained below SA4 thresholds and is single-seed evidence. Keep the historical full default unchanged pending replication/model validation. SA4 HOLD, c6400 rejected, SA5 not started.

Evidence: `logs/gates/sa4_d8/noise_eligibility_r1_20260817/`.

Authority:
- `docs/SA4_D8_NOISE_ELIGIBILITY_SENSITIVITY_20260817.md`
- Obsidian `Gate結果/28_SA4_D8_目前與有效回波限定雜訊比較_20260817.md`
