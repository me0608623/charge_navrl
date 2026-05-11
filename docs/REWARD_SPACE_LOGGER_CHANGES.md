# Reward Space Logger — 修改摘要

## 新增檔案

### `scripts/reinforcement_learning/skrl/diagnostics/reward_space_logger.py`

**完整新檔案**，提供 `RewardSpaceLogger` 類別：

```python
class RewardSpaceLogger:
    """記錄每個 episode 的關鍵 reward space 指標"""

    # 配置常量
    NUM_LIDAR_BINS = 72
    COLLISION_THRESHOLD = 0.45  # m
    FRONT_WARN_DIST = 1.2  # m
    FRONT_HALF_ANGLE_DEG = 30.0  # ±30°
    FRONT_NEAREST_K = 5

    # 主要方法
    def step(env_ids, lidar_72)
    def finalize_episodes(reset_env_ids, infos)
    def get_rollout_metrics() -> dict
    def flush()
    def close()
```

**輸出 CSV 格式**：
```
episode_id,env_id,num_steps,lidar_min_mean,lidar_min_min,d_front_mean,
collision_count,front_block_count,collision_trigger_ratio,front_block_trigger_ratio,
success,collision_done,timeout
```

---

## 修改的檔案

### 1. `diagnostics/__init__.py`

添加導出：
```python
from .reward_space_logger import RewardSpaceLogger

__all__ = [..., "RewardSpaceLogger"]
```

### 2. `wandb_trainer.py`

**a) __init__() 添加成員變數**：
```python
self.reward_space_logger = None  # Line ~87
```

**b) _single_agent_train_with_wandb() 添加 per-step 調用**：
```python
# 在 ablation_logger.step() 之後 (Line ~575)
if self.reward_space_logger is not None:
    lidar_72 = next_states[..., 6:78] if next_states.dim() > 2 else next_states[None, 6:78]
    self.reward_space_logger.step(env_ids=torch.arange(self.env.num_envs, device=self.env.device), lidar_72=lidar_72)
```

**c) 所有 reset() 處添加 finalize_episodes()**（3 處）：
```python
# 在 self.env.reset() 之前
if self.reward_space_logger is not None:
    done_mask = (terminated | truncated).squeeze(-1) if (terminated | truncated).dim() > 1 else (terminated | truncated)
    reset_ids = torch.nonzero(done_mask).squeeze(-1)
    self.reward_space_logger.finalize_episodes(reset_env_ids=reset_ids, infos=infos)
```

**d) _log_to_wandb() 之前的 flush 區塊添加指標聚合**：
```python
# 在 ablation_logger.get_and_reset() 之後 (Line ~695)
if self.reward_space_logger is not None:
    rs_metrics = self.reward_space_logger.get_rollout_metrics()
    tracking_data_snapshot.update({k: [v] for k, v in rs_metrics.items()})
```

### 3. `train_charge_ac.py`

**在 ablation_logger 初始化之後添加** (Line ~971)：
```python
# Reward space logger (always enabled with ablation_logger)
try:
    from diagnostics import RewardSpaceLogger
    log_dir = getattr(trainer, "_csv_path", None)
    if log_dir:
        log_dir = str(Path(log_dir).parent)
    trainer.reward_space_logger = RewardSpaceLogger(env, output_dir=log_dir)
    print(f"[INFO] RewardSpaceLogger enabled (CSV: {trainer.reward_space_logger._csv_path})")
except Exception as e:
    print(f"[WARN] RewardSpaceLogger failed to init: {e}")
```

---

## 完整 Patch

```diff
--- a/scripts/reinforcement_learning/skrl/diagnostics/__init__.py
+++ b/scripts/reinforcement_learning/skrl/diagnostics/__init__.py
@@ -7,10 +7,13 @@ from .module_entropy import (
     classify_module_entropy,
 )
+from .reward_space_logger import RewardSpaceLogger

 __all__ = [
     "ModuleEntropyMonitor",
     "compute_grad_norm",
     "compute_module_entropy",
     "classify_module_entropy",
+    "RewardSpaceLogger",
 ]
```

```diff
--- a/scripts/reinforcement_learning/skrl/wandb_trainer.py
+++ b/scripts/reinforcement_learning/skrl/wandb_trainer.py
@@ -84,6 +84,7 @@ class WandBSequentialTrainer(SequentialTrainer):
         self.debug_logger = debug_logger
         self.ablation_logger = None  # Set externally for NavRL0* tasks
+        self.reward_space_logger = None  # Set externally for reward space logging
         self._initial_timestamp = None
         self._last_log_time = None

@@ -572,6 +573,13 @@ class WandBSequentialTrainer(SequentialTrainer):
                 if self.ablation_logger is not None:
                     self.ablation_logger.step(actions, rewards, terminated, truncated, infos)

+                if self.reward_space_logger is not None:
+                    lidar_72 = next_states[..., 6:78] if next_states.dim() > 2 else next_states[None, 6:78]
+                    self.reward_space_logger.step(
+                        env_ids=torch.arange(self.env.num_envs, device=self.env.device),
+                        lidar_72=lidar_72
+                    )
+
                 # 保存tracking_data（在record_transition清空之前）
                 tracking_data_snapshot = self._capture_tracking_data(single_agent=True)

@@ -532,7 +534,14 @@ class WandBSequentialTrainer(SequentialTrainer):
             # reset environments
             if terminated.any() or truncated.any():
-                with torch.no_grad():
+                if self.reward_space_logger is not None:
+                    done_mask = (terminated | truncated).squeeze(-1) if (terminated | truncated).dim() > 1 else (terminated | truncated)
+                    reset_ids = torch.nonzero(done_mask).squeeze(-1)
+                    self.reward_space_logger.finalize_episodes(reset_env_ids=reset_ids, infos=infos)
+                with torch.no_grad():
                     states, infos = self.env.reset()
             else:
                 states = next_states
@@ -691,6 +700,13 @@ class WandBSequentialTrainer(SequentialTrainer):
                         {k: [v] for k, v in abl_metrics.items()}
                     )

+                # Reward Space 指標
+                if self.reward_space_logger is not None:
+                    rs_metrics = self.reward_space_logger.get_rollout_metrics()
+                    tracking_data_snapshot.update(
+                        {k: [v] for k, v in rs_metrics.items()}
+                    )
+
                 # Module Entropy 指標：flush 累積的 mini-batch 數據，取平均後注入
                 if self._module_entropy_monitor is not None:
```

```diff
--- a/scripts/reinforcement_learning/skrl/train_charge_ac.py
+++ b/scripts/reinforcement_learning/skrl/train_charge_ac.py
@@ -968,6 +968,18 @@ def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
             except Exception as e:
                 print(f"[WARN] AblationMetricsLogger failed to init: {e}")

+            # Reward space logger (always enabled with ablation_logger)
+            try:
+                from diagnostics import RewardSpaceLogger
+                log_dir = getattr(trainer, "_csv_path", None)
+                if log_dir:
+                    log_dir = str(Path(log_dir).parent)
+                trainer.reward_space_logger = RewardSpaceLogger(env, output_dir=log_dir)
+                print(f"[INFO] RewardSpaceLogger enabled (CSV: {trainer.reward_space_logger._csv_path})")
+            except Exception as e:
+                print(f"[WARN] RewardSpaceLogger failed to init: {e}")
     else:
         runner = Runner(env, agent_cfg)
         trainer = None
```

---

## 使用方式

**自動啟用**：當 task 名稱包含 "NavRL" 時，自動啟用（與 ablation_logger 相同邏輯）

**輸出位置**：
- CSV: `logs/reward_space/reward_space_episodes_<timestamp>.csv`
- WandB 指標: `reward_space/lidar_min_mean`, `reward_space/d_front_mean`, `reward_space/collision_trigger_ratio`, `reward_space/front_block_trigger_ratio`

**關鍵計算**：
- `lidar_min`: 72 bins 全局最小
- `d_front`: 前方 ±30° (12 bins) 取最近 5 bins 平均
- `collision_flag`: lidar_min <= 0.45m
- `front_block_flag`: d_front < 1.2m
