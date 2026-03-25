# Curriculum Learning 設計 v2.0 - 碩士研究生級優化版

## 設計理念

根據 RL 訓練的經驗法則，優化課程學習設計以避免常見陷阱：
- 避免過度擬合 (Overfitting)
- 避免過度依賴全域規劃器
- 避免獎勵衝突
- 避免災難性遺忘 (Catastrophic Forgetting)

---

## 四階段課程設計

### Stage 1: Kinematics & Path Interception (運動學與路徑攔截)

**原版問題**：Phase 0 只訓練直線追隨，導致 Agent 學會「油門焊死」策略。

**優化方案**：
```yaml
路徑類型:
  - S 型曲線 (S-curve): 40%
  - 直角轉彎 (90° turn): 25%
  - 圓形路徑 (Circular): 20%
  - 直線 (Straight): 15%

初始姿態隨機化:
  - 位置偏移: ±2 米
  - 姿態偏移: ±45°
  - 路徑起始點隨機
```

**實作重點**：
1. AIT* 生成的路徑必須包含多樣性
2. 強迫 Agent 學會「切回軌跡」而非「從起點開始」

**觀測空間增強**：
```python
# 原有觀測保持，新增：
- path_curvature: 路徑曲率 (1維)
- distance_to_path: 到路徑的距離 (1維)
- lateral_error: 橫向誤差 (1維)
```

---

### Stage 2: Perception-Aware Tracking (感知感知追蹤)

**原版問題**：Agent 過度依賴 AIT*，忽略 LiDAR 數據。

**優化方案**：
```yaml
路徑噪點 (Path Noise):
  - 30% 的步驟: 路徑偏移 ±0.5 米
  - 10% 的步驟: 路徑點暫時丟失 (模擬 sensor 失效)

LiDAR 感知獎勵:
  - clearance_reward: 離障礙物 > 安全距離 時給獎勵
  - too_close_penalty: 離障礙物 < 20cm 時給懲罰
  - lidar_utilization: 根據 LiDAR 數據使用程度給獎勵
```

**獎勵函數調整**：
```python
def perception_aware_reward(env, ...):
    # 原有的路徑追隨獎勵
    path_reward = path_following_reward(...) * 0.7

    # 新增：LiDAR 感知獎勵
    lidar_reward = lidar_clearance_reward(...) * 0.3

    return path_reward + lidar_reward
```

---

### Stage 3: Precision Control in Constrained Spaces (約束空間精確控制)

**原版問題**：窄門場景中獎勵衝突 - 防碰撞懲罰高則不敢動，路徑獎勵高則硬擠。

**優化方案**：
```yaml
動態廊道 (Dynamic Corridor):
  - 寬廣區域 (width > 2m): cross_track_error 容忍 ±1.0m
  - 窄門區域 (width < 1m): cross_track_error 容忍 ±0.2m
  - 過渡區域: 線性插值

CBF 保護:
  - 在窄門處啟用 CBF 安全過濾器
  - 允許「貼著牆走」但防止碰撞
  - penalty_coeff 隨廊道寬度動態調整
```

**動態廊道實作**：
```python
def dynamic_corridor_width(env, ...):
    # 根據周圍障礙物密度調整容許誤差
    obstacle_density = calculate_obstacle_density(lidar)
    allowed_error = 1.0 / (1.0 + obstacle_density)
    return allowed_error
```

---

### Stage 4: Dynamic Reactivity & Latency Compensation (動態反應與延遲補償)

**原版問題**：AIT* 更新延遲 (0.5秒) 導致 RL 照著舊路徑撞向移動障礙物。

**優化方案**：
```yaml
Frame Stacking:
  - 輸入: 當前幀 + 過去 3 幀的 LiDAR
  - 格式: [N_envs, 4, 72] 或 [N_envs, 288]
  - 目的: 讓 CNN 學會預測障礙物運動

動態障礙物模式:
  - 直線移動 (Linear): 30%
  - 圓弧移動 (Arc): 20%
  - 隨機停止-啟動 (Stop-Go): 30%
  - 追蹤模式 (Chasing): 20%
```

**Frame Stacking 實作**：
```python
# 在觀測配置中加入
frame_stack_cfg = ObservationTermCfg(
    func=mdp.observation_functions.frame_stack_lidar,
    params={"stack_size": 4},
    history=4,  # 保留過去 4 幀
)
```

---

## 混合訓練 (Mixed Training) - 防止災難性遺忘

**原版問題**：訓練到高階段時忘記低階段的基本能力。

**優化方案**：
```yaml
# 訓練階段混合比例
Stage 1: 100% Stage 1 環境
Stage 2: 80% Stage 2 + 20% Stage 1
Stage 3: 60% Stage 3 + 30% Stage 2 + 10% Stage 1
Stage 4: 40% Stage 4 + 30% Stage 3 + 20% Stage 2 + 10% Stage 1

# 持續訓練 (Fine-tuning)
最終階段: 混合所有環境 + 30% 隨機重播舊數據
```

**實作方式**：
```python
class MixedCurriculumScheduler:
    def __init__(self):
        self.schedule = {
            0:    [1.0, 0.0, 0.0, 0.0],  # Stage 1 only
            1000: [0.2, 0.8, 0.0, 0.0],  # Stage 1+2
            5000: [0.1, 0.3, 0.6, 0.0],  # Stage 1+2+3
            10000: [0.1, 0.2, 0.3, 0.4], # All stages
        }

    def get_env_distribution(self, iteration):
        # 根據訓練進度返回環境分布
        ...
```

---

## Phase 對應表

| 階段 | 原版名稱 | 優化名稱 | 重點能力 | 成功指標 |
|------|---------|---------|---------|---------|
| Stage 1 | Phase 0 (直線追隨) | Kinematics & Path Interception | 運動控制、路徑攔截 | Success > 95% |
| Stage 2 | Phase 1 (靜態避障) | Perception-Aware Tracking | LiDAR 感知、局部修正 | Success > 70% |
| Stage 3 | Phase 2 (窄門迷宮) | Precision Control | 窄門通過、精確控制 | Success > 60% |
| Stage 4 | Phase 3 (動態環境) | Dynamic Reactivity | 預判、動態避障 | Success > 50% |

---

## 訓練命令建議

```bash
# Stage 1: 基礎運動控制
python train_charge.py --task Isaac-Charge-Kinematics-v0 \
    --num_envs 128 --max_iterations 5000

# Stage 2: 感知感知追蹤 (從 Stage 1 模型開始)
python train_charge.py --task Isaac-Charge-Perception-v1 \
    --checkpoint logs/stage1/model.zip \
    --mix_envs Isaac-Charge-Kinematics-v0:0.2

# Stage 3: 精確控制 (混合訓練)
python train_charge.py --task Isaac-Charge-Precision-v2 \
    --checkpoint logs/stage2/model.zip \
    --mix_envs Isaac-Charge-Kinematics-v0:0.1,Isaac-Charge-Perception-v1:0.3

# Stage 4: 動態反應 (完全混合)
python train_charge.py --task Isaac-Charge-Dynamic-v3 \
    --checkpoint logs/stage3/model.zip \
    --mix_envs Isaac-Charge-Kinematics-v0:0.1,Isaac-Charge-Perception-v1:0.2,Isaac-Charge-Precision-v2:0.3
```

---

## 論文/報告使用建議

在論文或實驗報告中，使用以下術語會顯示專業度：

1. **Path Interception Capability**: 路徑攔截能力 (而非單純的追隨)
2. **Perception-Aware Tracking**: 感知感知追蹤 (強調 LiDAR 使用)
3. **Dynamic Corridor Navigation**: 動態廊道導航 (而非簡單的窄門)
4. **Latency-Aware Planning**: 延遲感知規劃 (處理全域規劃延遲)
5. **Continual Learning with Replay**: 持續學習與重播 (防止災難性遺忘)

這些術語展現了對 RL 與控制理論整合的深刻理解。
