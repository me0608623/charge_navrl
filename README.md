# Charge-SKRL

> **Safe Navigation for Differential-Drive Robots via Discrete PPO in Isaac Lab**

[![English](https://img.shields.io/badge/lang-English-blue)](README_en.md)
[![繁體中文](https://img.shields.io/badge/lang-繁體中文-red)](README_zh_TW.md)

---

## What is Charge-SKRL?

A sim-to-real navigation framework for differential-drive mobile robots using:
- **Proximal Policy Optimization (PPO)** with discrete action space
- **3D LiDAR perception** (VLP-16 → 72-bin 2D projection)
- **Potential-based reward shaping** for policy invariance
- **4-Phase Curriculum Learning** for progressive skill acquisition

---

## Quick Links

| Document | Description |
|----------|-------------|
| [README_en.md](README_en.md) | Full English documentation |
| [README_zh_TW.md](README_zh_TW.md) | 完整繁體中文文檔 |
| [README_CURRICULUM.md](README_CURRICULUM.md) | Curriculum learning design (English) |
| [README_CURRICULUM_zh_TW.md](README_CURRICULUM_zh_TW.md) | 課程學習設計（繁中） |

---

## Quick Start

```bash
# Train with curriculum learning
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train_charge_ac.py \
    --task Isaac-Charge-Navigation-Curriculum-v0 \
    --num_envs 256 \
    --headless
```

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Discrete(361) Action** | 19×19 center-symmetric grid for linear × angular velocity |
| **139D Observation** | Ego(4) + Goal(2) + LiDAR(72) + Obstacles(60) + Time(1) |
| **3-Branch Network** | Conv1d(LiDAR) + MLP(Obstacles) + MLP(State) → 128D |
| **Mixed Parallel** | 50% empty / 30% static / 20% dynamic obstacle environments |
| **Domain Randomization** | Physics, sensor noise, external disturbances |

---

## 4-Phase Curriculum

| Phase | Goals | Distance | Obstacles | Focus |
|-------|-------|----------|-----------|-------|
| 1: Dense Exploration | 8 | 2-5m | 0 | Learn to reach targets |
| 2: Sparse Navigation | 3 | 4-8m | 3 | Long-range stability |
| 3: Safe Avoidance | 2 | 3-8m | 8 | Obstacle avoidance |
| 4: Ultimate Challenge | 1 | 3-8m | 13 | Dynamic environments |

See [README_CURRICULUM.md](README_CURRICULUM.md) for details.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Isaac Lab Environment                     │
│  Scene │ VLP-16 LiDAR │ Contact Sensor │ Goal Command       │
│       ┌───────────────────┬────────────────────────────┐    │
│       │    Observation Manager (139D)                  │    │
│       └───────────────────┬────────────────────────────┘    │
└───────────────────────────┬─────────────────────────────────┘
                            │ obs (139D)
                            ▼
┌──────────────────────────────────────────────────────────────┐
│                    SKRL PPO Agent                            │
│  VLP16FeatureExtractor: LiDAR(64D) + Obs(32D) + State(32D)  │
│                          = 128D                              │
│       ┌──────────────────┴──────────────────┐               │
│       ▼                                     ▼               │
│   Actor (128→361)                    Critic (128→1)         │
│   Categorical logits                 Scalar V(s)            │
└──────────────────────────────────────────────────────────────┘
```

---

## Performance

| Metric | Target |
|--------|--------|
| Success Rate (Phase 4) | > 50% |
| Collision Rate | < 20% |
| Training Time | ~60M env steps |

---

## References

- [NVIDIA Isaac Lab](https://github.com/isaac-sim/IsaacLab)
- [SKRL](https://github.com/Toni-SM/skrl)
- [PPO Paper](https://arxiv.org/abs/1707.06347)

---

## License

This project is built on the Isaac Lab framework and follows its license terms.
