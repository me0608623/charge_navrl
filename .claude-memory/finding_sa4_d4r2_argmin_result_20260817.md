---
name: SA4-D4-r2 actual winner-angle fixed comparison result
description: Actual raw-ray angles did not recover D4 feasible candidates; obstacle CR remained unchanged and SA4 stays on hold
type: finding
date: 2026-08-17
status: complete-not-passed
---

# SA4-D4-r2 actual winner-angle result

## Fixed protocol

- SA4-R2 c6400, checkpoint SHA `c6dbd94bcbd6cc17ceca3cf8d5abfb4d956c98ed173c0dd890b9fc45959a2197`
- stage4 lateral, evaluator seed818, fixed d1=200ms
- fresh baseline then `geometry_feasible_argmin`, each 64 env x 2500 steps
- full VLP-16 noise unchanged; no training; no SA5
- selector protocol `sa4_d4_geometry_selector_argmin/v1`, SHA `aca15fbdc2a2a7545dbc3d4b6dfba86549e1fac9e39ae0e4c24549abe6facdcd`

## Valid result

- baseline: n=1976, SR=83.05%, obstacle CR=16.45%, wall CR=0.51%, total CR=16.95%, TO=0%
- actual-angle: n=1989, SR=83.36%, obstacle CR=16.44%, wall CR=0.20%, total CR=16.64%, TO=0%
- total CR delta is -0.31pp, almost entirely wall CR; obstacle counts are 325 -> 327 and obstacle-rate delta is only -0.007pp. Do not claim demonstrated performance improvement from this single checkpoint/seed point estimate.
- all recorder reconciliation true, delay alignment errors=0, no NaN/OOM/traceback, source fingerprints stable, 856 tests +27 subtests passed.

## Paired geometry result

- active frames: 24,665
- both feasible: 3,846 (15.59%)
- actual-angle only: 469 (1.90%)
- center-angle only: 567 (2.30%)
- both no-feasible: 19,783 (80.21%)
- actual-angle feasible: 4,315 / 24,665 = 17.49%
- center-angle feasible: 4,413 / 24,665 = 17.89%
- paired net recovery: -98 frames = -0.40pp
- actual-angle no-feasible: 20,350 / 24,665 = 82.51%
- preregistered descriptive `clear_candidate_recovery=False` (required >=100 frames and >=5pp).

Conclusion: the 5-degree center approximation is not the main cause of D4 no-feasible frames. Stop tuning winner-ray angle. c6400 remains rejected (SR<90%, CR>10%); SA4 HOLD and SA5 not started.

## Runtime incident retained

The first formal attempt completed baseline but failed argmin at step0 because a no-hit ray made near by mixed-pixel injection had `atan2(inf, inf)=NaN` when angle was derived from hit position. Trace now uses RayCaster `_ray_directions_w`, preserving the emitted-ray direction without changing policy observation/noise/binning. Failed evidence remains at `logs/gates/sa4_d4/argmin_angle_r2_20260816/INCOMPLETE_RUNTIME_ERROR.md` and must not be pooled.

Valid evidence: `logs/gates/sa4_d4/argmin_angle_r2_20260817/`.

Separate unresolved issue: measured `0.002515` is a white-wall ROI MAD outlier fraction over existing returns, not evidence for uniform Bernoulli near-ghost injection across all 5,760 simulated rays. Any noise correction requires a separate frozen protocol and physically conditioned source data; do not fold it into D4-r2.

Authority:
- `docs/SA4_D4_R2_VLP16_NOISE_AND_ARGMIN_AUDIT_20260816.md`
- Obsidian `Gate結果/27_SA4_D4R2_VLP16雜訊語意與Argmin角度修正_20260816.md`
