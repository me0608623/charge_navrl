---
name: charge-debug
description: 處理 Charge 專案的調試任務，包含常見問題診斷、錯誤排查、訓練問題解決等。在遇到訓練問題或異常行為時使用。
---

# Charge 專案 - 調試專用 Skill

## 🐛 常見問題診斷流程

### 問題診斷檢查清單

```python
✅ 環境導入正確？
✅ 觀測維度匹配？
✅ 獎勵函數返回有效值？
✅ PPO 超參數合理？
✅ Episode 長度適當？
✅ 終止條件正常工作？
✅ Isaac Sim 版本兼容？
```

---

## 🔴 高頻問題與解決方案

### 問題 1：PPO StdDev = 0 錯誤

**錯誤訊息**：
```
RuntimeError: The standard deviation of the action distribution is zero.
This can happen when the action distribution is too deterministic.
```

**根本原因**：觀測值方差為零（所有值相同）

**診斷步驟**：
```python
# 1. 檢查觀測值範圍
obs, _ = env.reset()
print(f"Obs range: [{obs.min()}, {obs.max()}]")
print(f"Obs std: {obs.std(dim=0)}")

# 2. 檢查各個觀測分量
print(f"LiDAR mean: {obs[:, :120].mean()}, std: {obs[:, :120].std(dim=0)}")
print(f"Goal position: {obs[:, 120:122]}")
```

**常見原因**：

| 原因 | 症狀 | 修復方法 |
|------|------|----------|
| LiDAR 返回相同距離 | obs[0:120].std() ≈ 0 | 檢查傳感器配置，添加隨機性 |
| 目標位置總是 [0, 0] | goal 不變 | 修復 goal_command.py |
| 觀測未歸一化 | 方差過大，梯度問題 | 添加歸一化到 [0, 1] |

**解決方案**：
```python
# 解決方案 1：給 LiDAR 添加噪聲
lidar_raw = ray_caster.data.out_dist
lidar_noisy = lidar_raw + torch.randn_like(lidar_raw) * 0.01

# 解決方案 2：檢查歸一化
normalized = torch.clamp(raw_value / max_range, 0.0, 1.0)

# 解決方案 3：使用自定義環境類檢查觀測
# 在 cfg/charge_env.py 中的 ChargeNavigationEnv
```

---

### 問題 2：蛇行行為（Snake Behavior）

**症狀**：
- Agent 在障礙物附近左右擺動
- 路徑呈現正弦波模式
- 進度顯著減慢

**根本原因分析**：
```python
# 計算目標:避障比例
goal_weight = 5.0  # distance_to_goal
avoidance_weight = 0.4  # progressive_collision
ratio = goal_weight / avoidance_weight  # = 12.5:1

if ratio < 25:
    print(f"❌ Snake behavior expected: ratio too low ({ratio:.2f}:1)")
else:
    print(f"✅ Ratio healthy: {ratio:.2f}:1")
```

**解決方案**：

```python
# 方案 1：增加目標權重
class RewardsCfg:
    velocity_to_goal = RewTerm(
        func=velocity_toward_goal,
        weight=10.0,  # 從 5.0 提高到 10.0
    )

# 方案 2：使用距離加權
# 遠距離時更關注目標，近距離時更關注避障
class RewardsCfg:
    velocity_to_goal_dw = RewTerm(
        func=velocity_toward_goal_distance_weighted,
        weight=1.0,
        params={"far_weight": 2.0, "near_weight": 0.5},
    )

# 方案 3：添加朝向獎勵
class RewardsCfg:
    heading_to_goal = RewTerm(
        func=heading_to_goal_distance_weighted,
        weight=1.0,
    )
```

---

### 問題 3：Agent 翻車騙分（Tipping Hack）

**症狀**：
- Agent 故意翻倒
- 翻車後滑行到目標
- 利用物理漏洞

**診斷**：
```python
# 檢查姿態分佈
up_vector = env.robot.data.root_state_w[:, 6:9]  # [num_envs, 3]
uprightness = up_vector[:, 2]  # z 分量

print(f"Upright ratio: {(uprightness > 0.5).float().mean()}")
# 如果 < 0.8，說明很多 Agent 在翻車狀態
```

**解決方案 - 姿態門控**：
```python
def forward_velocity_reward_gated(env: ManagerBasedEnv) -> torch.Tensor:
    # 基礎獎勵
    reward = forward_velocity_reward(env)

    # 檢查是否正立
    up_vector = env.robot.data.root_state_w[:, 6:9]
    is_upright = up_vector[:, 2] > 0.5  # z 軸向上

    # 翻車則取消所有移動獎勵
    reward = reward * is_upright.float()

    return reward

# 在配置中使用
class RewardsCfg:
    forward_velocity = RewTerm(
        func=forward_velocity_reward_gated,  # 使用門控版本
        weight=1.0,
    )
```

---

### 問題 4：恐懼障礙物（Fear of Obstacles）

**症狀**：
- Agent 停留在原地不敢移動
- 繞大圈遠離障礙
- 訓練收斂極慢

**根本原因**：碰撞懲罰過高，Agent 學會「不做不错」

**解決方案**：

```python
# 方案 1：降低碰撞懲罰
class RewardsCfg:
    collision_penalty = RewTerm(
        func=collision_penalty,
        weight=-0.5,  # 從 -5.0 降到 -0.5
    )

# 方案 2：使用漸進式懲罰
class RewardsCfg:
    progressive_collision = RewTerm(
        func=progressive_collision_penalty,
        weight=-1.0,
        params={
            "near_threshold": 0.3,  # 0.3m 內開始懲罰
            "far_threshold": 1.0,   # 1.0m 達到最大
        },
    )

# 方案 3：Phase 3 策略（移除障礙）
# 先訓練純淨的導航意志，再加回障礙
class MySceneCfg:
    num_static_obstacles = 0  # Phase 3: 無障礙物
```

---

### 問題 5：訓練不收斂

**診斷檢查清單**：

```python
# 1. 檢查學習率
learning_rate = 3e-4  # 標準值
# 如果 < 1e-5: 太小，收斂慢
# 如果 > 1e-3: 太大，不穩定

# 2. 檢查獎勵尺度
total_reward = env.reward_manager.compute()
print(f"Total reward: mean={total_reward.mean():.2f}, std={total_reward.std():.2f}")
# 正常範圍：[-50, +50] per episode
# 如果絕對值 > 1000: 獎勵設計有問題

# 3. 檢查 Episode 長度
max_episode_length = 500  # steps
# 如果太短: Agent 沒時間學習
# 如果太長: 訓練效率低

# 4. 檢查環境數量
num_envs = 256  # SB3 推薦值
# 如果 < 64: 採樣效率低
# 如果 > 4096: 可能 GPU 記憶體不足
```

---

## 🔧 調試工具與指令

### 查看觀測值
```python
# 在訓練腳本中添加
obs, info = env.reset()
print("Observation shape:", obs.shape)
print("Observation range:", obs.min(), obs.max())
print("Per-dimension std:", obs.std(dim=0))
```

### 查看獎勵分佈
```python
# 在環境中添加日誌
from ..mdp.rewards import _print_diagnostics

# 在 reward term 中啟用
def your_reward(env):
    reward = calculate_reward(env)
    _print_diagnostics("your_reward", reward)
    return reward
```

### TensorBoard 監控
```bash
# 啟動 TensorBoard
tensorboard --logdir ~/isaaclab/isaac-navigator-charge-sb3-v0/logs

# 觀察指標
- rollout/ep_rew_mean      # Episode 平均獎勵
- train/learning_rate      # 學習率
- train/policy_loss        # Policy loss
- train/value_loss         # Value loss
- train/policy_entropy     # 熵（探索程度）
```

### 錄製訓練過程
```bash
# 啟用可視化訓練（移除 --headless）
./isaaclab.sh -p scripts/reinforcement_learning/sb3/train_charge.py \
    --task Isaac-Navigation-Charge-SB3-v0 \
    --num_envs 64  # 降低環境數以適配渲染

# 錄製影片
--video  # 自動保存 MP4
```

---

## 📊 性能基準值

### 健康訓練指標

| 指標 | Phase 0 | Phase 1 | Phase 2 | Phase 3 |
|------|:-------:|:-------:|:-------:|:-------:|
| **成功率 (1M steps)** | >90% | >70% | >50% | >80% |
| **Episode 長度** | <200 | <300 | <400 | <500 |
| **平均獎勵** | >100 | >50 | >20 | >80 |
| **Policy Entropy** | >0.5 | >0.3 | >0.2 | >0.4 |

### 異常警報

```python
# 警報條件
if ep_rew_mean < -100:
    print("🚨 獎勵過低！檢查獎勵函數設計")

if policy_entropy < 0.1:
    print("🚨 熵過低！探索不足，增加 ent_coef")

if success_rate < 0.1 and steps > 1e6:
    print("🚨 訓練失敗！檢查環境配置")

if episode_length > max_length * 0.95:
    print("⚠️ Episode 過長！可能需要調整終止條件")
```

---

## 🚨 緊急修復指令

```bash
# 訓練崩潰時
# 1. 殺掉進程
pkill -f train_charge.py

# 2. 清理快取
rm -rf ~/.isaac/cache/*

# 3. 降低複雜度重啟
./isaaclab.sh -p ... --num_envs 128  # 降低環境數

# 4. 從檢查點恢復
--load_path path/to/checkpoint.zip
```
