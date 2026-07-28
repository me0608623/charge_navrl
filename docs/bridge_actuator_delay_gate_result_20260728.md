# Actuator-delay bridge gate result (2026-07-28)

## Decision

Neither corrected bridge c10 nor its five-iteration continuation c15 is a
deployment candidate. Both fail the fixed-200 ms Stage-5 Gate2 absolute
threshold. The expensive corridor and Gate5a sweeps were stopped by the
preregistered rule.

## Checkpoints

- W1 baseline:
  `logs/rnn_car/sa6_k8_obb_corridor_w1_wander_s42/checkpoint_1280.pt`
- Corrected bridge c10:
  `logs/rnn_car/bridge_actdelay_w1c10_s42_r2_cmdqueue/checkpoint_1280.pt`
- Continued bridge c15:
  `logs/rnn_car/bridge_actdelay_w1c10_s42_r3_c10_to_c15/checkpoint_640.pt`

The continuation loaded both `charge_opt_rl` and `charge_opt_aux` optimizer
states, ran five iterations (640 steps), and saved every iteration. It was not
a restart.

## Gate2

All formal rows use seeds 101/202/303, 64 envs, 1200 steps per seed.

| Condition | Policy | n | SR | CR | TO | Verdict |
|---|---|---:|---:|---:|---:|---|
| Actuator off | W1 | 7164 | 92.9% | 7.0% | 0.1% | PASS |
| Actuator off | bridge c10 | 7218 | 94.3% | 5.7% | 0.0% | PASS |
| Fixed 200 ms bundle | W1 | 4422 | 86.6% | 13.2% | 0.2% | FAIL |
| Fixed 200 ms bundle | bridge c10 | 4445 | 88.5% | 11.3% | 0.1% | FAIL |
| Fixed 200 ms bundle | bridge c15 | 4582 | 88.5% | 11.5% | 0.0% | FAIL |

Absolute Gate2 thresholds are SR >= 90%, CR <= 9%, TO <= 4%.

c10 learned a consistent but insufficient adaptation: relative to W1 at the
same fixed-delay bundle, SR improved by 1.93 pp and CR improved by 1.90 pp.
Five more PPO iterations did not improve the primary gate.
They did reduce omega RMS from 0.613 to 0.591 rad/s and full-steer usage from
28.70% to 26.08%, which further separates the failure from action jitter:
the policy became smoother without reducing collisions enough to pass.

## Action stability at fixed 200 ms

| Metric | W1 | bridge c10 |
|---|---:|---:|
| omega RMS | 0.623 | 0.613 rad/s |
| Worst-seed sign-flip p95 | 0.151 | 0.165 |
| Full-steer fraction | 28.10% | 28.70% |
| Worst-seed dominant-frequency p95 | 0.227 | 0.215 Hz |
| Worst-seed same-sign-turn p95 | 11.43 | 11.44 s |

The bridge did not fail because of a newly introduced action oscillation. It
failed because collision performance remained below the deployment gate.

## Model-specification issue discovered

The fixed-delay evaluation is not a pure 200 ms delay test. It also applies:

- velocity scale sampled from U(0.9, 1.1) per episode;
- one scalar motor-lag coefficient alpha=0.3 to both linear and angular speed.

For the implemented filter

`y[t] = alpha*u[t] + (1-alpha)*y[t-1]`

at `dt=0.2 s`:

`tau = -dt / ln(1-alpha) = 0.56 s`

and:

`t90 = -dt*ln(0.1) / ln(1-alpha) = 1.29 s`.

Adding a 200 ms dead time makes this much harsher than a pure 200 ms actuator
delay. Historical notes state that the real angular channel used approximately
alpha=0.5 while the linear channel used approximately alpha=0.3, and explicitly
mark the time constants as not calibrated. The current scalar action-term
parameter cannot represent that split.

## Next valid experiment

Do not continue training the current bundle blindly.

1. Measure separate linear and angular step responses on the vehicle.
2. Estimate per-channel alpha with
   `alpha = 1 - 0.1^(dt/t90)` (or fit the full first-order-plus-dead-time model).
3. Add separate linear/angular lag coefficients if the measurements differ.
4. Keep the user-frozen delay support U{0,1,2}, then rerun a short bridge.
5. Repeat the same actuator-off and fixed-200 ms paired Gate2 before any
   corridor sweep or model export.

Raw reports are under `logs/gates/bridge/fixed_delay_pair/`.
