---
name: SA4-D4-r2 VLP-16 noise semantics and argmin-angle correction
description: The measured 0.2515% metric does not justify uniform whole-scan near-ghost injection; corrected D4 arm now uses exact winning raw-ray angles with paired center shadow
type: finding
date: 2026-08-16
status: superseded-by-finding-sa4-d4r2-argmin-result-20260817
---

# SA4-D4-r2 noise semantics and argmin-angle correction

> Superseded pre-run snapshot. The valid fixed comparison is recorded in
> [finding_sa4_d4r2_argmin_result_20260817.md](finding_sa4_d4r2_argmin_result_20260817.md).
> Do not use the `READY_NOT_RUN` statements below as current status.

## Noise audit

- `0.002515` originated as per-frame `MAD outliers removed / existing points in the filtered
  white-wall ROI`, then averaged. It is not a measured probability that any of all 5,760 possible
  rays creates a near phantom return.
- Current `wd_like_sweep_72` applies independent Bernoulli draws to every raw ray, replaces selected
  ranges with `U(0.2,2.0)m`, then takes the minimum of about 80 rays per 5-degree bin.
- This yields 14.4864 expected distractor rays and 13.137 expected bins with at least one distractor
  per scan. The min-pool therefore amplifies the small raw-ray rate.
- The simulator semantics are not faithful to the measured statistic. Do not silently call the
  current distractor path measured VLP-16 mixed-pixel fidelity.
- The preserved six guided white-wall JSONs recompute to 0.002373665 unweighted and 0.002839469
  ROI-weighted, neither equal to the FIXED 0.002515. Exact July source inputs are not reproducible
  from currently preserved JSONs.
- Do not change noise in the D4-r2 angle rerun; it would confound two factors. Any noise correction
  requires a separate frozen protocol. Aggregate files do not contain the edge/range/intensity
  distribution needed to synthesize a physical mixed-pixel replacement model.

## Argmin-angle correction

- Historical `geometry_feasible` remains protocol v1 SHA
  `1729fe3a878b1a1c64f172bb7b3bb0a23f6553385c8b8ca9e982e0cf97170e9d`.
- New mode: `geometry_feasible_argmin`.
- New protocol: `sa4_d4_geometry_selector_argmin/v1`, SHA
  `aca15fbdc2a2a7545dbc3d4b6dfba86549e1fac9e39ae0e4c24549abe6facdcd`.
- It bitwise-matches the current policy 72-bin observation to the realized trace and passes
  `winner_actual_angle_rad` into static point reconstruction. Missing/ambiguous trace, bad shape,
  nonfinite angle or angle outside [-pi,pi] fails closed; there is no center-angle fallback.
- Action insertion remains before `process_actions`: policy indices -> selector -> decode -> d1
  queue -> applied command.
- Every active argmin frame also computes a legacy 5-degree-center feasibility shadow on the same
  state. Only argmin actions may be applied. Four paired quadrants reconcile exactly to active
  frames.

## Fixed pending comparison

- checkpoint: SA4-R2 c6400, SHA
  `c6dbd94bcbd6cc17ceca3cf8d5abfb4d956c98ed173c0dd890b9fc45959a2197`
- stage4 lateral, evaluator seed818, fixed d1=200ms
- fresh baseline then argmin arm, each 64 env x 2500 steps
- output: `logs/gates/sa4_d4/argmin_angle_r2_20260816/`
- noise unchanged; no training; no SA5
- preregistered descriptive clear recovery: paired net feasible gain >=5 pp and >=100 frames
- code verification: full rnn_car_wdclean 855 passed + 27 subtests
- at 2026-08-16 17:28 GPU was occupied by another user's two jobs (31.5/32.6GiB); no rollout was
  launched and no incomplete evidence directory exists.

Authority:
- `docs/SA4_D4_R2_VLP16_NOISE_AND_ARGMIN_AUDIT_20260816.md`
- Obsidian `Gate結果/27_SA4_D4R2_VLP16雜訊語意與Argmin角度修正_20260816.md`
