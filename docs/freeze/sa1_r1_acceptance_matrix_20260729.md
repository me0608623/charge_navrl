# SA1 R1 Acceptance Matrix - Frozen Specification

> **SUPERSEDED on 2026-07-29.** This document tested Stage-5 deployment
> obstacle avoidance and is not an SA1 promotion specification. Use
> `sa1_r1_nav20_acceptance_matrix_20260729.md`. Existing results from this
> matrix are retained only as deployment stress diagnostics.

- Initially frozen at: `2026-07-29T14:45+08:00`
- Pre-execution audit revision: `2026-07-29T15:23+08:00`
- Target run: `sa1_sim2real_v1_ne1024_s42_r1`
- Config: `e2e_sa1_k8_obb_sim2real_v1`
- Source freeze: `docs/freeze/sa1_sim2real_v1_full_r1_20260728.md`
- Status: **queued only - no GPU evaluation may start before R1 is inactive**

This file fixes every free parameter of the acceptance run. Any deviation
invalidates comparison across checkpoints and across delay conditions.

## Fail-open defaults that MUST be overridden

The three existing gate runners default to the historical W1 bridge configuration.
Running them without these flags produces numbers that look valid but describe
a different actuator model.

```text
--actuator-profile sa1_delay_only     # default is "bridge": U(0.9,1.1) scale + alpha=0.3
--actuator-delay-steps {0,1,2}        # default None means actuator DR fully OFF, not d=0
--seeds 515,616,717                   # runners default to three different seed sets
--mode sealed                         # narrow runner defaults to "legacy" = Gate5b, not the gap test
--modes lateral,longitudinal,random_2d,mixed_iid   # corridor default uses legacy "mixed"
```

`--actuator-delay-steps` semantics, from `fixed_actuator_eval.py`:

```text
omitted / None -> fixed_actuator_cli_args returns []      -> actuator DR OFF
0              -> --actuator_delay_range 0 0              -> DR ON, zero delay
1              -> --actuator_delay_range 1 1              -> DR ON, 200 ms
2              -> --actuator_delay_range 2 2              -> DR ON, 400 ms
```

Under `sa1_delay_only`, d=0 is an identity transform at the action-term level:
velocity scale is `(1.0, 1.0)` and motor lag alpha is `1.0`. This identity is
guarded by the deterministic `ActionDelaySemanticsTest` and fixed-profile
wiring tests.

Actuator OFF is **not** included as a fourth delay in the formal performance
matrix. Enabling d=0 still executes `torch.randint` and `uniform_` calls inside
the action term. Those calls can advance the shared RNG stream and change later
scene draws, so OFF and d=0 full rollouts are not guaranteed to be trajectory
pairs. A score difference between them would therefore be ambiguous.

The true-clean OFF run remains a separate single-seed diagnostic (Stage O). It
must not be pooled with, or used to replace, the fixed d=0/1/2 acceptance cells.

## Fixed parameters

```text
num_envs        = 64
steps           = 1200
seeds           = 515, 616, 717
delay conditions= 0, 1, 2
actuator profile= sa1_delay_only
narrow mode     = sealed          (Gate5a)
narrow stage    = 5
narrow arena    = 10.0
corridor modes  = lateral, longitudinal, random_2d, mixed_iid
corridor kinematics = wander      (matches the R1 training distribution)
task            = Isaac-Navigation-Charge-VLP16-Curriculum-WD
curriculum      = warp_drive_e2e_final20_v1
deterministic   = true
```

## Evaluation protocol

### Stage A — candidate screening

Single delay `d=1` (200 ms, the measured vehicle dead time), single seed `515`.
Applied to the late checkpoints only. Screening exists to cut cost, not to
select a winner: its ranking is advisory.

```text
checkpoints: it1500, it1700, it1900, it2109(final)
scenarios:   gate2, narrow_sealed, lateral, longitudinal, random_2d, mixed_iid
rollouts:    4 ckpt x 6 scenarios = 24
```

Fixed-interval checkpoints, not "highest training SR". Training SR across
it1261-2109 sits inside a 0.36 pt band, so selecting by training score is
sampling noise. SA7 already produced the counter-example where the lowest
training SR gave the strongest corridor behaviour.

### Stage B - fixed-delay acceptance

Top 1-2 candidates from Stage A only.

```text
scenarios: 6   (gate2, narrow_sealed, lateral, longitudinal, random_2d, mixed_iid)
delays:    3   (0, 1, 2)
seeds:     3   (515, 616, 717)
rollouts per candidate = 6 x 3 x 3 = 54
```

### Stage O - true-clean actuator-off diagnostic

This is reported separately and is not an additional deployment delay:

```text
scenarios: 6
delay:     actuator DR off
seed:      515
rollouts per candidate = 6
```

Do not require numerical equality between Stage O and Stage B d=0. The direct,
deterministic action-term tests establish d=0 identity without scene-RNG
confounding.

### Non-GPU gates

The currently frozen evaluator emits the following action-stability fields in
every Gate2 cell:

```text
action stability : sign-flip rate, full-steer fraction, omega RMS,
                   dominant frequency, same-sign turn duration
```

`reverse_fraction` was required by the training plan but is not emitted by the
currently frozen `play_rnn_car.py`. It must be added as a logging-only evaluator
change after R1 exits, then frozen and tested before deployment acceptance can
finish. It is not silently treated as zero.

Python/TorchScript parity is also a separate mandatory gate:

```text
parity : 83D action-history + K8 frame-stack Python checkpoint vs the
         vehicle-side TorchScript export on one fixed observation sequence
```

R1 is not an RNN policy: the checkpoint records
`end_to_end_frame_stack=True`, `lidar_frame_stack=8`,
`use_action_history=True`, and no active `use_rnn` flag. The historical
`e2e_parity_oracle.py` is stateless but hard-coded to 79D/K4, so it is still not
valid for this 83D/K8 checkpoint without an explicit audited extension.
Until the vehicle-side 83D parity JSON is supplied, every queue summary must
remain `deployable=false`, even if all GPU cells pass.

## Reporting rules

1. **Every cell reports independently.** No averaging across delay conditions.
   A single SR that pools d=0/1/2 is not a valid summary; the delay is the
   variable under test.
2. Per-scenario results are reported per scenario. The three-scene training
   mixture already hides corridor behaviour behind an aggregate; the gate must
   not repeat that.
3. Each cell writes its own JSON. Schema below.
4. Runner `rc=0` means policy gate PASS. Runner `rc=1` means a valid policy gate
   FAIL only when the expected runner JSON exists, parses, matches the requested
   cell, and contains complete metrics/runtime evidence. That FAIL is recorded
   and the matrix continues.
5. Missing files, malformed reports, runtime/config mismatches, structural
   failures, `rc>1`, or an `rc`/gate-verdict mismatch stop the queue.
6. No partial matrix is reported as a completed result.

## JSON schema (one file per cell)

```json
{
  "schema": "sa1_r1_acceptance/v2",
  "cell_id": "checkpoint_269952__corridor_random_2d__d1__seed515",
  "stage": "B",
  "checkpoint": "logs/rnn_car/sa1_sim2real_v1_ne1024_s42_r1/checkpoint_269952.pt",
  "scenario": "corridor_random_2d",
  "delay_condition": "d1",
  "actuator_profile": "sa1_delay_only",
  "seed": 515,
  "num_envs": 64,
  "steps": 1200,
  "episodes_completed": 4096,
  "sr": 0.91,
  "cr": 0.08,
  "timeout": 0.01,
  "gate_pass": true,
  "structural_pass": true,
  "thresholds": {},
  "action_stability": null,
  "actuator_eval": {
    "enabled": true,
    "profile": "sa1_delay_only",
    "delay_steps": 1,
    "delay_ms": 200,
    "velocity_scale_range": [1.0, 1.0],
    "motor_lag_alpha": 1.0
  },
  "runner": "run_corridor_motion_suite.py",
  "runner_argv": [],
  "source_report": "/abs/path/corridor_motion_suite.json",
  "runtime_markers": {
    "actuator_dr": "[SIM2REAL] Actuator DR: delay=(1, 1) steps, ...",
    "vlp16_noise": "[SIM2REAL][VLP16-ablation] mode=full ..."
  },
  "exit_code": 0,
  "started_at": "2026-07-29T08:00:00+00:00",
  "finished_at": "2026-07-29T08:06:00+00:00"
}
```

`runtime_markers` captures the `[SIM2REAL]` lines from the rollout log. For
d=0/1/2, an empty actuator marker is a failure. For Stage O, the actuator marker
must be absent as negative evidence that DR stayed off. The VLP-16 `mode=full`
marker is mandatory in every cell.

## Preconditions (all must hold before any GPU work)

```text
1. systemd unit is inactive, or its transient unit has been collected
2. training console contains `Training complete: 270,000 steps`
3. exact `checkpoint_269952.pt` exists and is readable by `torch.load`
4. checkpoint reports `iteration + 1 == 2109`, `total_steps == 269952`, and
   the frozen run/config/seed arguments
5. no python process matching train_rnn_car_wdclip.py
6. nvidia-smi succeeds and shows no compute process other than the desktop daemon
```

Fail closed: if any precondition is unmet, the queue waits. It never starts a
GPU rollout to "check" whether the GPU is free.

## Explicitly out of scope

- No SA2 launch under any outcome of this matrix.
- No modification to any file listed in the R1 SHA-256 freeze.
- No checkpoint is deployable because training completed or because a gate
  passed; the staged vehicle acceptance sequence is separate.
- Reverse-action fraction and final-checkpoint 83D/K8 Python/TorchScript parity
  remain explicit deployment blockers.
- This R1 run is the all-at-once v1 robustness arm. It is not retroactively
  relabelled as the canonical navigation-first SA1 in curriculum v2, and the
  queue never launches SA2.
