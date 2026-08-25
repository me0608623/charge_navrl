# SA5-v3 c50 matched-control result (2026-08-24)

- Authority: `logs/gates/sa5_v3_c50_equalweight_control_screen/screen_20260824_r1/SUMMARY.json`
- Status: `COMPLETE_VALID_SINGLE_SEED_MATCHED_THREE_ARM_DIAGNOSTIC`.
- Anchor c50 SHA: `4bc1744b...a99c`.
- Equal-weight it25 SHA: `d9095e7d...419a`.
- Random-2D 70% it25 SHA: `cbc8a91a...670f`.
- All four new cells valid; source fingerprint before/after exact.
- Generic continuation (equal control - c50): only longitudinal changed materially,
  CR `81.48% -> 47.50%` (`-33.98 pp`, descriptive z `-22.43`).
- Weight-specific effect (70% weighted - equal control):
  lateral CR `+5.88 pp`, longitudinal `+21.79 pp`, random-2D `-5.65 pp`,
  mixed `+5.01 pp`. Random-2D SR rose only `+3.75 pp`, below the frozen +5 pp
  target. All three retention families failed.
- Decision: `WEIGHT_SPECIFIC_CAPABILITY_REDISTRIBUTION_OBSERVED_REJECT_CURRENT_WEIGHTS`.
  Do not extend weighted it25, select it as parent, graduate SA5, or launch SA6.
- Corrected interpretation: the earlier apparent longitudinal improvement versus c50
  was generic continuation; 70% random-2D weighting suppressed most of it.
- Empirical active-step shares, equal vs weighted:
  `21.59/17.32/26.50/34.59%` vs `1.40/11.82/69.00/17.78%`
  for lateral/longitudinal/random-2D/mixed.
- Formal allocator CPU replay reproduces equal and 70% reset shares. A mild
  `(0.30,0.30,0.40)` slot-weight candidate predicts reset shares near
  `16.96/16.12/34.88/32.03%`; it is a calibration candidate, not accepted treatment.
- Next bounded experiment: from equal-control it25, run matched A2 equal and B2
  `(0.30,0.30,0.40)` for 25 iterations each, then fixed four-family screen. No SA6.
