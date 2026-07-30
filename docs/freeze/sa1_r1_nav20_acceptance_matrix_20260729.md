# SA1 R1 Nav20 Acceptance Matrix

- Frozen at: `2026-07-29T16:35+08:00`
- Target run: `sa1_sim2real_v1_ne1024_s42_r1`
- Runner: `run_sa1_nav20_suite.py`
- Queue: `sa1_r1_acceptance_queue.py`
- Canonical schema: `sa1_r1_acceptance/v3`
- Supersedes: `sa1_r1_acceptance_matrix_20260729.md`

## Decision

SA1 promotion measures basic goal navigation in the 20 x 20 m room. The old
Gate2 was a Stage-5 deployment evaluation (`15 x 15 m`, `10S+3D`) and therefore
cannot block SA1. Narrow-passage and deployment-corridor suites remain useful
stress diagnostics, but they are outside this promotion matrix.

The currently running R1 is retained as an all-at-once robustness arm. Its
training recipe contains narrow and deployment-corridor reset injections, but
that does not redefine the SA1 acceptance target.

## Layer 1 - Nav20-clean (blocking)

```text
arena                    = 20 x 20 m
random robot spawn/yaw   = enabled
stationary random goals  = 1 per episode
goal distance            = 2.0-9.0 m
episode length           = 60 s
static obstacles         = 0
dynamic obstacles        = 0
internal walls           = 0
outer boundary walls     = enabled
policy                   = deterministic argmax
```

Hard thresholds:

```text
SR >= 98.0%
CR <= 1.5%     # only boundary-wall collisions are possible
TO <= 0.5%
```

All three conditions must pass. This is the only blocking scene in SA1.

## Layer 2 - Nav20-native (advisory)

Same contract as Nav20-clean, except:

```text
static obstacles = 2
```

Advisory labels:

```text
SR >= 90%
CR <= 8%
TO <= 3%
```

A native failure is reported but cannot make `all_blocking_gates_pass=false`.
Its purpose is to expose whether the policy has begun basic static bypass
without requiring deployment-grade obstacle avoidance.

## Actuator and observation conditions

Formal acceptance evaluates each scene independently under:

```text
d0 = 0 ms
d1 = 200 ms
d2 = 400 ms
actuator profile = sa1_delay_only
velocity scale   = 1.0
motor lag alpha  = 1.0
VLP-16 mode      = full measured-noise preset
```

No score may pool delay conditions. Actuator-off remains a separate
single-seed diagnostic because enabling the d0 action term advances RNG state.

## Evaluation stages

### Stage A - checkpoint screening

```text
checkpoints = it1500, it1700, it1900, it2109
scenes      = Nav20-clean, Nav20-native
delay       = d1
seed        = 515
rollouts    = 4 x 2 = 8
```

Ranking uses Nav20-clean first. Nav20-native is shown beside it but cannot
promote or eliminate a checkpoint.

### Stage B - fixed-delay acceptance

For one or two explicitly selected candidates:

```text
scenes = 2
delays = 3
seeds  = 3 (515, 616, 717)
rollouts per candidate = 2 x 3 x 3 = 18
```

Only the nine Nav20-clean cells are blocking. Every clean delay/seed cell must
pass independently. Nav20-native remains advisory in all nine cells.

### Stage O - actuator-off diagnostic

```text
scenes = 2
delay  = actuator DR off
seed   = 515
rollouts per candidate = 2
```

Stage O is not pooled with d0.

## Runtime and schema guards

Every rollout must prove from its runtime log:

```text
stage=1
arena=20 x 20 m
goal count=1
goal distance=2.0-9.0 m
episode=60 s
dynamic obstacles=0
internal walls=0
static obstacles=0 (clean) or 2 (native)
VLP-16 mode=full
requested fixed actuator delay/profile reached the action term
```

Each canonical JSON includes:

```text
scenario
blocking
delay_condition
seed
SR / CR / TO
action-stability metrics
runtime markers
source report and runner argv
```

The currently frozen evaluator records deterministic full-steer, sign-flip,
omega RMS, dominant-frequency, and same-sign-turn metrics. These are required
diagnostics but have no hard threshold yet. `reverse_fraction` and final
83D/K8 Python/TorchScript parity remain separate deployment blockers.

## Parallel sentinel rule

Single-seed intermediate Nav20 sentinels may run beside training when GPU
memory permits. They must be labelled `interim` and are not formal Stage B
cells. The full post-training queue retains its completion and checkpoint
preconditions so the final matrix is reproducible and resumable.

