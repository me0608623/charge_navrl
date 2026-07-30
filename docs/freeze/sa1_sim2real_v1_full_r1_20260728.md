# SA1 Sim-to-Real V1 Full R1 Freeze

- Run name: `sa1_sim2real_v1_ne1024_s42_r1`
- Config: `e2e_sa1_k8_obb_sim2real_v1`
- Source hashes: identical to
  `docs/freeze/sa1_sim2real_v1_pilot30_source_20260728.md`
- Pilot gate: PASS, recorded in
  `docs/sa1_sim2real_v1_pilot30_result_20260728.md`
- Start state: fresh model and optimizer; no pilot checkpoint is loaded

## Run Contract

```text
num_envs=1024
seed=42
timesteps=270000
rollout_length=128
effective_iterations=2109
effective_steps=269952
save_interval=50
checkpoint=None
resume_optimizer=False
parallel_eval=off
```

The 48-step remainder in `270000 / 128` is not executed because the trainer
uses complete rollouts. Checkpoints are saved every 50 iterations (6,400
steps), plus the final checkpoint.

No checkpoint is deployable merely because training completes. Mature
checkpoint screening must include fixed `d=0`, `d=1`, and `d=2` gates reported
separately, action-stability metrics, Python/TorchScript parity, and the staged
vehicle acceptance sequence.
