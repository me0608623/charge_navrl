# Training Snapshot Summary — user-5090

> Quick comparison sheet. For details see `TRAINING_SNAPSHOT_user-5090_20260325_1751.md`

## Machine
| | Value |
|---|---|
| Host | user-5090 |
| GPU | RTX 5090 (32GB) |
| CPU | Intel Core Ultra 7 265K, 12 cores |
| RAM | 47 GiB |
| OS | Ubuntu 24.04.3, kernel 6.17.0-19 |

## Software
| | Value |
|---|---|
| Python | 3.11.14 (conda: env_isaaclab) |
| PyTorch | 2.7.0+cu128 |
| CUDA | 13.0 |
| Driver | 580.126.09 |
| Isaac Sim | 5.1.0.0 |
| Isaac Lab | 0.52.1 (editable) |
| SKRL | 1.4.3 |
| WandB | 0.23.1 |

## Git
| | Value |
|---|---|
| Branch | abl |
| Commit | d03c17c4bb |
| Clean? | Yes |
| Remote | charge_skrl → me0608623/charge_skrl.git |

## PPO
| | Value |
|---|---|
| rollouts | 128 |
| epochs | 6 |
| mini_batches | 16 |
| γ | 0.990 (curriculum adjusts) |
| LR | 1e-4 → 3e-5 (LinearLR) |
| grad_clip | 1.0 |
| ratio_clip | 0.2 |
| entropy | 0.01 |
| timesteps | 234,375 |
| num_envs | 6144 (CLI) |
| seed | 1 (CLI) |

## Reward (V6)
| Term | Weight |
|------|:---:|
| reaching_goal | +500 |
| goal_velocity | +2.0 (curriculum) |
| goal_progress | +3.0 (curriculum) |
| static_safety | +2.0 (front_block=0.4) |
| dynamic_safety | +2.0 (curriculum) |
| smoothness | -0.1 |
| collision | -50 |
| alive | 0 |

## Curriculum (open_ended_v1)
| Phase | Stages | Obstacles | Walls |
|-------|:---:|---------|:---:|
| Bootstrap | B1-B6 | 0→8S+3D | 0 |
| Open-ended | L0+ | 8S+3D→20S+5D | 0 |

## Scene
| | Value |
|---|---|
| Size | 20×20m |
| Obstacle entities | 20 |
| Wall slots | 8 (disabled with --no_walls) |
| Spawn safe dist | 1.0m |

## Observation: 139D
ego(4) + goal(2) + LiDAR_72bins(72) + obstacles_top10×6D(60) + time(1)

## Action: MultiDiscrete([19,19])
v_max=1.0, a_max=0.5, ω_max=0.25π, dt=0.2s, reverse=Yes

## Network
- Actor: 139D → 3-branch FE (128D) → 128→38 logits
- Critic: 139D → 3-branch FE (128D) → 64→32→1
- Separate actor/critic (not shared)

## Latest Run
```
rw_groundv6_openendedv1__seed1_nowalls
reward=v6, curriculum=open_ended_v1, --no_walls, seed=1, envs=6144
```
