# Teacher Goal Denominator Closed-Loop A/B (2026-08-17)

## Decision

Keep the historical teacher default `goal_denominator_floor_m=1.0`.
The `R=10` candidate did not pass the preregistered comparative checks, so no
new SA5 distillation branch was created and no SA5/SA6 training was started.

The tested goal term was:

```text
C_goal = ||goal - predicted_position(action)|| / max(initial_goal_distance, R)
```

The parameter is configurable in training and evaluation, while its default
remains `1.0` for backward compatibility.

## Protocol

- Teacher-only closed loop with teacher action override.
- Historical checkpoint:
  `logs/rnn_car/sa5_e2e_k8_obb_corridor_teacher_distill_c12_stepnorm_s42/checkpoint_128.pt`
- Checkpoint SHA-256:
  `bbde43317a9ce2bc508a2e82042913a7838fe4b2690274fe98c3ebf321335598`
- Three evaluator seeds: `515`, `616`, `717`.
- Each arm and seed: `64 env x 700 steps`, at least 500 completed episodes.
- Scene: 4 m x 10 m corridor, 2 static + 1 dynamic obstacle, lateral patrol,
  dynamic speed 0.30-0.60 m/s, default pause behavior.
- Deterministic evaluation, no actuator domain randomization.
- Protocol SHA-256:
  `dcf42d302a95c0f7c2e240ecf67aba03fbd0dcd3a71efb58a7f4fa4296872b90`

Source fingerprint, checkpoint hash, phase accounting, and finite-value checks
all remained valid.

## Pooled Results

| metric | R=1 historical | R=10 candidate | R10 - R1 |
|---|---:|---:|---:|
| episodes | 2,225 | 1,774 | -451 |
| SR | 95.910% | 92.616% | -3.29 pp |
| CR | 4.090% | 7.384% | +3.29 pp |
| TO | 0.000% | 0.000% | 0.00 pp |
| obstacle CR | 4.090% | 7.328% | +3.24 pp |
| wall CR | 0.000% | 0.056% | +0.06 pp |
| mean absolute speed | 0.6495 m/s | 0.5141 m/s | -0.1354 m/s |
| stop-command fraction | 2.554% | 7.887% | +5.33 pp |
| near-goal completion | 100.000% | 100.000% | 0.00 pp |
| near-goal collision | 0.000% | 0.000% | 0.00 pp |
| near-goal mean selected clearance | 2.5158 m | 2.5717 m | +0.0559 m |
| near-hard clearance fraction (<0.11 m) | 0.000% | 0.000% | 0.00 pp |

All three evaluator seeds showed higher CR with `R=10`: `+3.25 pp`,
`+3.63 pp`, and `+2.99 pp`. The per-seed near-goal fifth-percentile clearance
improvements were only `+0.0425 m`, `+0.0318 m`, and `+0.0419 m`; none reached
the preregistered `+0.05 m` threshold.

## Safety-Distance Answer

The recorded clearance is not the simulator's realized closest separation. It
is the teacher's **minimum predicted OBB-to-obstacle surface clearance along
the selected two-second action path, after robot buffer**.

For states with `D_goal <= 3 m`, `R=10` changed that predicted clearance in the
intended direction: pooled mean increased by `0.0559 m`, and per-seed p05
increased by `0.0318-0.0425 m`.

However, a post-hoc check of the already-recorded all-state distribution shows
the opposite global result:

| selected predicted clearance | R=1 | R=10 | change |
|---|---:|---:|---:|
| frame-weighted mean | 1.0104 m | 0.9070 m | -0.1034 m |
| per-seed p05 range | 0.1905-0.2014 m | 0.1044-0.1049 m | lower in all seeds |
| fraction below 0.11 m | 1.691% | 6.855% | +5.164 pp |

Thus `R=10` modestly increases the teacher's predicted clearance near the goal,
but does not increase predicted clearance globally. The all-state result is
descriptive and post-hoc; it was not a preregistered decision endpoint.

An exact claim about **realized physical minimum clearance** would require a
new recorder based on simulator OBB geometry at every step. This A/B did not
record that distribution.

## Interpretation

The earlier static synthetic probe showed that `R=10` could remove a narrow
near-goal clearance band. That band was not active in the near-goal subset of
this closed-loop test: both arms had zero near-goal selections below the 0.11 m
near-hard threshold, and every episode that entered the near-goal region
completed without collision or timeout. Across all states, by contrast, the
candidate spent more selected-path frames close to the 0.10 m hard floor.

`R=10` also changes the goal term for every state with initial goal distance
below 10 m. In this test it reduced mean speed, increased stopping, and reduced
the number of completed episodes at the same frame budget. This is consistent
with longer dynamic-obstacle exposure, but the A/B does not establish that as
the sole causal mediator.

Therefore the closed-loop evidence does not support changing the global
denominator floor. A future intervention should target the realized failure
state conditionally instead of globally weakening goal progress.

## Evidence

- Valid result: `logs/gates/teacher_goal_denominator_ab_r3_20260817/`
- Manifest: `suite_manifest.json`
- Human-readable summary: `SUMMARY.md`
- Frozen protocol: `PROTOCOL.json`
- Retained incomplete attempts:
  - `logs/gates/teacher_goal_denominator_ab_20260817/`
  - `logs/gates/teacher_goal_denominator_ab_r2_20260817/`
