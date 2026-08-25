# SA4-R3 exact optimizer continuation result

Date: 2026-08-17
Status: COMPLETE; SA4 HOLD; SA5 NOT STARTED

The user explicitly overrode the earlier R3 no-extension decision and authorized
an exact 50-iteration continuation from R3 c6400. Parent SHA was
`fa51b5a312d38b7aa1ee93c3464b087b9b9cb985a7137eefdc8fecf40fed166d`.
The continuation preserved the full R3 recipe and restored 38 RL optimizer
states. It completed 50/50 without non-finite metrics or accounting failures.

Conceptual checkpoint mapping and hashes:

- it75 = new `checkpoint_3200.pt`, SHA `7bec45d4...136f9c2`
- it100 = new `checkpoint_6400.pt`, SHA `aeddf32b...8520d7d`

Fixed seed818/d1/valid-return-only screen:

| checkpoint | lateral SR/CR/TO | longitudinal SR/CR/TO | native SR/CR/TO |
|---|---|---|---|
| it75 | 83.96/16.04/0.00% FAIL | 84.94/15.06/0.00% FAIL | 94.57/5.43/0.00% PASS |
| it100 | 88.08/11.92/0.00% FAIL | 94.68/5.29/0.03% PASS | 93.60/6.40/0.00% PASS |

`selected_checkpoint=null`. It100 is the best diagnostic candidate in this run,
but it is not an accepted parent. The planned 15-20% plateau condition did not
occur: lateral kept improving through it100. Therefore the result does not yet
justify changing lateral exposure. If the user authorizes another experiment,
prefer a bounded exact +25 iterations to conceptual it125, then rerun all three
fixed cells; stop if lateral or native fails. Never auto-launch SA5.

During final training iterations an external old R3 c6400 layout-split evaluator
shared the GPU, reducing fps only. Source fingerprints and recipe were stable.

Authoritative report: `docs/SA4_R3_CONT50_RESULT_20260817.md`.
Screen: `logs/gates/sa4_r3_cont50_checkpoint_screen/r1_20260817/`.
