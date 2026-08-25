---
name: project_sa5_low_density_speed_ratio_screen_20260821
description: Frozen c250 P100 low-density speed-rate cross-screen result and interpretation limits
type: project
---

# SA5 c250 low-density P100 × speed-rate screen (2026-08-21)

Status: `COMPLETE_VALID_SINGLE_SEED_DIAGNOSTIC`.

Formal output:
`logs/gates/sa5_low_density_speed_ratio/screen_20260821_r1/`

Protocol SHA-256:
`ddc37d497f55cf5eb1a9a845f7ede2a5077b7f0af1f477ad6d3621e9de16f243`

Fixed checkpoint: SA5-R2 c250 `checkpoint_32000.pt`, SHA-256
`df14c45d8dd27b0e327f9447a9a613bf62eaa850af542b474aad682dcd937a4a`.
Evaluator seed 818, P100 = 0.90–1.10 m/s, lateral, d1, full VLP-16 noise,
`valid_return_only`, 64 env × 3000 steps.

Results:

| density | speed_rate | n | SR | CR | TO |
|---|---:|---:|---:|---:|---:|
| 0S1D | 1.0 | 4266 | 46.20% | 53.80% | 0% |
| 0S1D | 0.7 | 4563 | 10.72% | 89.28% | 0% |
| 1S1D | 1.0 | 4229 | 47.60% | 52.40% | 0% |
| 1S1D | 0.7 | 4549 | 11.41% | 88.59% | 0% |

Within-density CR deltas, `0.7 - 1.0`:

- 0S1D: +35.49 pp = 19.13 conservative SE.
- 1S1D: +36.19 pp = 19.33 conservative SE.

Decision: `P100_NOT_ENTRY_READY_BEGIN_WITH_P060_P080`. Stop the old speed0.7
post-hoc adaptation. A policy trained under rate 1.0 cannot be made deployment-ready
by simply scaling its action contract to 0.7.

Scope: this is one checkpoint and evaluator seed. It does not prove P100 is
unlearnable or that 0.7 is generally less safe. No parent was accepted and SA6 was
not started.

