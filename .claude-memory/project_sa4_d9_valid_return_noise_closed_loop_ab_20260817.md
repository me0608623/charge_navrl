# SA4-D9 valid-return-only noise closed-loop A/B

Date: 2026-08-17
Status: COMPLETE_VALID_DIAGNOSTIC_EVIDENCE

## Subsequent authority (2026-08-17)

The single-seed result below was replicated on pre-fixed evaluator seeds 515
and 616. Together with seed818, 3/3 seeds showed lower total CR and wall-CR
increase below +0.5 pp. This authorized one SA4-R3 50-iteration pilot.

R3 completed normally but failed its fixed screen: it50 lateral
SR/CR=81.13/18.87%, longitudinal=78.21/21.73%, native=94.06/5.94%.
No checkpoint was selected. Do not extend, accept a parent, or launch SA5.

Authoritative continuation:
`project_sa4_d9_replication_r3_pilot_result_20260817.md`.

## Why this experiment exists

SA4-D8 was a paired shadow sensitivity audit. It showed that removing mixed-pixel distractors created on no-return or already-dropped rays changed the frozen D4 geometry result materially:

- current-noise no-feasible active frames: 82.73%
- valid-return-only shadow no-feasible active frames: 57.78%
- rescued current-no-feasible frames: 30.33%

D8 did not feed the corrected sweep to the policy, so it could not answer whether navigation SR/CR improves. D9 is the required closed-loop test.

## Frozen A/B question

Under the same SA4 c6400 lateral evaluation cell, does feeding a valid-return-only mixed-pixel sweep to the policy improve realized episode SR/CR relative to the historical full-noise rule?

## Arms and sole behavioral difference

- A, `current_full`: every raw LiDAR ray slot remains eligible for the existing mixed-pixel/distractor replacement.
- B, `valid_return_only`: a distractor is eligible only when that ray had a finite physical return and survived dropout.

All other LiDAR terms remain unchanged, including bias, Gaussian displacement, dropout probabilities, human-ray handling, distractor rate, and `U(0.2, 2.0 m)` replacement values. Physics, scene, checkpoint, policy, action delay, seed, environment count, and rollout length are fixed.

## Fixed cell

- checkpoint: SA4-R2 `checkpoint_6400.pt`
- checkpoint SHA-256: `c6dbd94bcbd6cc17ceca3cf8d5abfb4d956c98ed173c0dd890b9fc45959a2197`
- geometry/stage: SA4
- corridor family: lateral
- evaluator seed: 818
- actuator delay: d1, 200 ms
- environments: 64
- rollout: 2,500 steps per arm
- action override/shield: none
- training: forbidden
- SA5 launch: forbidden

## Validity requirements

1. Protocol and checkpoint hashes are frozen before either rollout.
2. A and B run from fresh processes with identical CLI arguments except the explicit noise-eligibility mode and output paths.
3. Both cells must produce valid corridor reports and meet the existing episode accounting checks.
4. Source fingerprints must remain unchanged before, between, and after arms.
5. Runtime must confirm the selected mode; a missing or unknown mode fails closed.
6. Any NaN, OOM, traceback, missing output, hash drift, or partial arm invalidates the A/B verdict.
7. Results are preserved and never overwritten.

## Primary outcomes and interpretation

Primary descriptive outcomes are episode-weighted SR, total CR, obstacle CR, wall CR, and TO. Report `B - A` in percentage points.

- A lower B-arm CR and higher B-arm SR would establish a closed-loop improvement for this fixed checkpoint/cell/seed.
- It would not validate `valid_return_only` as a faithful VLP-16 noise model.
- It would not establish generalization across checkpoints, evaluator seeds, corridor families, or stages.
- If B does not improve, D8's geometry recovery remains real but does not translate into policy performance under this fixed test.

No graduation or SA5 decision may be based on this single-seed diagnostic alone.

## Completed result

- protocol SHA-256: `e7ad89bc996d9c012515e25746dd6162bf52c13a61a5f407945ae5f2dba41ca4`
- A `current_full`: n=1,976, SR=83.0466%, CR=16.9534%, obstacle CR=16.4474%, wall CR=0.5061%, TO=0%
- B `valid_return_only`: n=2,066, SR=84.1723%, CR=15.8277%, obstacle CR=15.0532%, wall CR=0.7744%, TO=0%
- B-A: SR +1.13pp, total CR -1.13pp, obstacle CR -1.39pp, wall CR +0.27pp, TO unchanged
- preregistered directional and >=0.5pp material descriptive improvement rules both passed
- A exactly reproduced the prior fixed baseline, confirming backward-compatible `all_rays` behavior
- 871 tests + 27 subtests passed; runtime modes, d1, corridor contracts, hashes, source stability and cleanup all verified

Interpretation: corrected noise produced an actual closed-loop point-estimate improvement in this fixed cell, but the effect is single-seed and not statistically established. B still fails SA4 (SR 84.17% <90%; CR 15.83% >10%). Keep SA4 HOLD, do not accept c6400, and do not launch SA5. Valid-return-only remains a sensitivity rule, not a validated VLP-16 model.

Protocol wording clarification: v1's "same mask/value" means the eligibility branch preserves the random-draw operations while RNG/state histories are aligned. It does not mean masks remain bitwise paired after closed-loop actions and reset timing diverge. D9 is same-seed/same-sampling-law, not a per-frame paired counterfactual. See `PROTOCOL_INTERPRETATION_ERRATUM.md` in the evidence directory.

Evidence: `logs/gates/sa4_d9/noise_closed_loop_ab_r1_20260817/` and `docs/SA4_D9_VALID_RETURN_NOISE_CLOSED_LOOP_AB_20260817.md`.
