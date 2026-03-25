# Charge Robot Project - AI Quick Reference Guide

> 📌 **Read this first at the start of every new session!**

---

## 🎯 Project Core Info

### What is this project?
**Sim-to-Real Autonomous Navigation Robot** - Training the Charge robot for mapless navigation using Deep Reinforcement Learning.

### Robot Specifications
- **Differential Drive** chassis
- **2D LiDAR** sensor
- **End-to-End Learning**: LiDAR + State → Neural Network → Velocity Command

### Project Path
```
/home/aa/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge_sb3
```

---

## 📊 Training Phases (Current Status)

| Phase | Description | Obstacles | Status |
|:-----:|-------------|:---------:|:------:|
| **0** | Basic Kinematics | None | Basic Training |
| **1** | Static Avoidance | 3 static | ✅ Complete |
| **2** | Complex Static | 5 static | ⚠️ Snake behavior |
| **3** | Long-Range Nav | None (2-15m adaptive) | 🔄 **In Progress** |
| **4** | Dynamic Env | Dynamic | Pending |

**Current Focus**: Phase 3 - Build strong "navigation will", eliminate fear

---

## 🗂️ Key Files Quick Reference

```
charge_sb3/
├── __init__.py                    ⭐ All environment registrations (entry point!)
│
├── cfg/
│   ├── charge_cfg.py              🤖 Robot physics configuration
│   ├── charge_env_cfg.py          Phase 1 config
│   ├── charge_env_cfg_v2.py       Phase 2 config
│   ├── charge_env_cfg_v3.py       Phase 3 config (adaptive curriculum)
│   ├── charge_env_cfg_phase0.py   Phase 0 config (basic kinematics)
│   └── charge_env.py              HierarchicalChargeNavigationEnv
│
├── agents/
│   ├── sb3_ppo_cfg.yaml           SB3 PPO parameters
│   ├── sb3_ppo_cfg_v2.yaml
│   └── sb3_ppo_cfg_v3.yaml
│
└── mdp/
    ├── observations/              👁️ Observation functions
    ├── rewards/                   🎁 Reward functions
    ├── terminations/              🛑 Termination conditions
    ├── events/                    📅 Events/Curriculum learning
    └── path_planner/              🗺️ AIT* path planner
```

---

## 🎮 Registered Environment IDs

### Phase 0 (Basic Kinematics)
```
Isaac-Navigation-Charge-Phase0           # Train
Isaac-Navigation-Charge-Phase0-Play      # Test
```

### Phase 1-3 (RSL-RL)
```
Isaac-Navigation-Charge-v0/v1/v2/v3      # Train
Isaac-Navigation-Charge-Play-v0/v1/v2/v3 # Test
```

### Stable-Baselines3
```
Isaac-Navigation-Charge-SB3-v0/v1/v2/v3       # Train
Isaac-Navigation-Charge-SB3-Play-v0/v1/v2/v3  # Test
```

### Hierarchical Navigation (AIT*)
```
Isaac-Navigation-Charge-Hierarchical-v0/v1
```

---

## 🔧 Standard Configuration Modification Workflow

### 1️⃣ Modify Observations

```python
# 1. Define function in mdp/observations/functions.py
def your_new_obs(env: ManagerBasedEnv, dt: float) -> torch.Tensor:
    """Your observation function"""
    return some_value

# 2. Export in mdp/observations/__init__.py
__all__ = [..., "your_new_obs"]

# 3. Use in cfg/charge_env_cfg_*.py
from ..mdp.observations import your_new_obs

class ObservationsCfg:
    policy = ObsGroup(
        observations={
            "your_obs_name": ObsTerm(func=your_new_obs),
        },
    )
```

### 2️⃣ Modify Rewards

```python
# 1. Define function in mdp/rewards/your_rewards.py
def your_reward(env: ManagerBasedEnv) -> torch.Tensor:
    """Your reward function, returns [num_envs] tensor"""
    return reward_values

# 2. Export in mdp/rewards/__init__.py
# 3. Use in cfg/charge_env_cfg_*.py
class RewardsCfg:
    your_reward_term = RewTerm(
        func=your_reward,
        weight=1.0,  # Reward weight
        params={},    # Optional params
    )
```

### 3️⃣ Modify Terminations

```python
# Same flow, use DoneTerm
class TerminationsCfg:
    your_termination = DoneTerm(func=your_term_func)
```

### 4️⃣ Create New Environment Version

```python
# 1. Copy cfg/charge_env_cfg_v3.py → cfg/charge_env_cfg_v4.py
# 2. Rename class: ChargeNavigationEnvCfgV4
# 3. Export in cfg/__init__.py
# 4. Register in main __init__.py
gym.register(
    id="Isaac-Navigation-Charge-SB3-v4",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.charge_env_cfg_v4:ChargeNavigationEnvCfgV4",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)
```

---

## 🎁 Common Reward Functions Reference

| Function | Location | Purpose |
|----------|----------|---------|
| `velocity_toward_goal` | goal_rewards.py | Velocity projection toward goal |
| `reaching_goal` | goal_rewards.py | Reward for reaching goal |
| `collision_penalty` | safety_rewards.py | Collision punishment |
| `forward_velocity_reward` | motion_rewards.py | Forward velocity reward |
| `action_rate_penalty` | motion_rewards.py | Prevent jerky movements |
| `progress_along_path` | goal_rewards.py | Path following (AIT*) |

---

## 📝 Environment Config Structure Template

```python
@configclass
class ChargeNavigationEnvCfgVX:
    """Phase X Environment Configuration"""

    # Scene configuration
    scene: MySceneCfg = MySceneCfg(num_envs=4096)

    # Observation configuration
    observations: ObservationsCfg = ObservationsCfg()

    # Action configuration
    actions: ActionsCfg = ActionsCfg()

    # Reward configuration
    rewards: RewardsCfg = RewardsCfg()

    # Termination configuration
    terminations: TerminationsCfg = TerminationsCfg()

    # Event configuration (curriculum learning)
    events: EventCfg = EventCfg()

    # Command configuration (goal generation)
    commands: CommandsCfg = CommandsCfg()
```

---

## 🚀 Training Commands

```bash
# SB3 Training
./isaaclab.sh -p scripts/reinforcement_learning/sb3/train_charge.py \
    --task Isaac-Navigation-Charge-SB3-v0 \
    --num_envs 256 --headless

# Test Model
./isaaclab.sh -p scripts/reinforcement_learning/sb3/play.py \
    --task Isaac-Navigation-Charge-SB3-Play-v0 \
    --load_path path/to/model.zip
```

---

## ⚠️ Known Issues

| Issue | Solution |
|-------|----------|
| **Snake behavior** | Reduce collision penalty, increase goal-oriented rewards |
| **Tipping to cheat** | Use posture gating |
| **PPO std>=0** | Use ChargeNavigationEnv custom environment class |
| **Fear of obstacles** | Phase 3 removes obstacles first to build navigation will |

---

## 💡 Development Principles

1. **Read before modifying**: Check existing implementation first
2. **Modular design**: Put new functions in corresponding mdp subdirectories
3. **Remember exports**: Update `__init__.py` after modifications
4. **Backward compatibility**: Use v4/v5 for new features, be careful with overrides
5. **Test first**: Small-scale testing before large training

---

## 📚 More Information

Full Manual: `AI_SESSION_MANUAL.md` (Chinese)
Design Docs: `md/` directory
