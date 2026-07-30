# SA1 Sim-to-Real V1 Pilot-30 Result

## Identity

- Run: `sa1_sim2real_v1_ne1024_s42_pilot30_20260728`
- W&B: `4p4dfa1z`
- Config: `e2e_sa1_k8_obb_sim2real_v1`
- Source freeze: `docs/freeze/sa1_sim2real_v1_pilot30_source_20260728.md`
- Runtime: 30/30 iterations, 3,840 steps, 974 s
- Status: learning-health PASS; not a deployment candidate

## Runtime Contract

- 1024 environments, seed 42, fresh model and optimizer
- 83D policy and critic observations
- K=8 end-to-end LiDAR frame stack; RNN unused
- issued-action history normalization `(0.5, 1.2)`
- measured VLP-16 `full` noise
- actuator delay `U{0,1,2}` with neutral scale and lag
- 78% native / 12% narrow / 10% corridor reset assignment
- narrow first batch: 125/125 constructively solvable
- corridor first batch: 100/100 constructively solvable

At 500 corridor assignments:

```text
density=25/35/20/15/5 percent exactly
family fractions=1/3,1/3,1/3 exactly
crossing geometry=94/94
side-by-side geometry=34/34
random_2d wander=375/375
s_spacing=0
s_speed_delta=0
```

## Learning Health

```text
iteration 1:  SR= 4.3%  CR= 4.3%  TO= 0.0%
iteration 10: SR=73.3%  CR=10.3%  TO=15.0%
iteration 20: SR=78.9%  CR=14.5%  TO= 8.5%
iteration 30: SR=80.6%  CR= 7.6%  TO=12.1%
last-5 mean: SR=80.1%  CR= 8.7%  TO=11.1%
```

No Python traceback, RuntimeError, NaN, OOM, or allocator overflow dump
occurred. Iteration 30 triggered the configured KL early-stop guard after three
accepted minibatch updates; iterations 1-29 completed all 32 updates. This is
treated as a guard activation, not a crash.

Six checkpoints were produced through `checkpoint_3840.pt`. Every checkpoint
in this pilot directory is explicitly non-deployable.

## Decision

The exact recipe is released for a fresh full SA1 training run. Fixed
`d=0/1/2` performance gates are deferred until checkpoint screening because a
30-iteration policy is intentionally immature. Each selected mature checkpoint
must still pass separate delay-condition gates, action-stability analysis,
Python/TorchScript parity, and the staged vehicle acceptance sequence.
