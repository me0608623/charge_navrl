# Plan: 修復碰撞偵測 + Robot 重生安全檢查

## Context
三個同時存在的 bug 導致：(1) 碰撞不觸發 terminate (2) robot 重生在障礙物裡面

### Bug 1: LiDAR 碰撞偵測失效
- LiDAR 在 **1.6m** 高，obstacles 最高 **1.5m**
- 近距離時 LiDAR 最低射線（-15°）從障礙物上方通過
- `collision_occurred()` 的 `min_distance` 永遠是遠距離值

### Bug 2: PhysX Contact Sensor 無效
- Obstacles 設為 `kinematic_enabled=True`
- PhysX 不對 kinematic body 產生接觸力 → `force_matrix_w` 永遠是 0

### Bug 3: Robot 重生只檢查前 20 個障礙物
- `reset.py` L434: `for i in range(20)` 硬編碼
- 場景有 100 個 obstacle entity，obstacle_20~99 不被檢查

---

## 修復方案

### Fix 1: 碰撞偵測 — 改用 robot 位置 vs obstacle 位置的直接距離計算
不依賴 LiDAR 或 contact sensor，改用最可靠的方式：直接計算 robot center 到每個 visible obstacle center 的距離。

**檔案**: `source/.../charge_skrl/mdp/rewards/safety_rewards.py`

新增函數 `obstacle_proximity_termination()`：
```python
def obstacle_proximity_termination(env, threshold=0.45, max_obstacles=100):
    robot_pos = env.scene["robot"].data.root_pos_w[:, :2]  # [N, 2]
    for i in range(max_obstacles):
        obs_name = f"obstacle_{i}"
        if obs_name not in env.scene.keys():
            continue
        obs_pos = env.scene[obs_name].data.root_pos_w
        visible = obs_pos[:, 2] > 0.0
        dist = torch.norm(robot_pos - obs_pos[:, :2], dim=1)
        # 取得 obstacle radius
        obs_r = env._obstacle_sizes[i] / 2.0 if hasattr(env, '_obstacle_sizes') else 0.3
        # 碰撞 = robot center 到 obstacle center 距離 < threshold + obs_radius
        collision |= visible & (dist < threshold + obs_r)
    return collision
```

**檔案**: `source/.../charge_skrl/cfg/charge_env_cfg_vlp16.py` ~L623
```python
collision = DoneTerm(
    func=obstacle_proximity_termination,
    params={"threshold": COLLISION_THRESHOLD, "max_obstacles": MAX_OBSTACLES},
)
```

保留 LiDAR 碰撞作為輔助（可偵測牆壁等 LiDAR 目標），保留 wall_collision。

### Fix 2: Robot 重生安全檢查 — range(20) → 動態讀取
**檔案**: `source/.../charge_skrl/mdp/events/reset.py` ~L434

```python
# 改前: for i in range(20):
# 改後:
num_obs = getattr(env, "_num_obstacles", 20)
for i in range(num_obs):
```

`_num_obstacles` 由 `set_obstacle_metadata()` 設定，值為 100。

---

## 修改檔案清單

| 檔案 | 修改 |
|------|------|
| `mdp/rewards/safety_rewards.py` | 新增 `obstacle_proximity_termination()` |
| `cfg/charge_env_cfg_vlp16.py` | collision DoneTerm 改用新函數 |
| `mdp/events/reset.py` L434 | `range(20)` → `range(num_obs)` |

## 驗證
```bash
./isaaclab.sh -p scripts/reinforcement_learning/skrl/play_charge_ac_curriculum.py \
  --checkpoint .../agent_58590.pt \
  --no_curriculum --num_static 50 --num_dynamic 5 --num_walls 0 --num_envs 4
```
- Robot 碰到 obstacle → episode 立即 reset ✅
- Robot 不會重生在 obstacle 裡面 ✅
- 場景有 55 個可見障礙物 ✅
