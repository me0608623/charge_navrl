# Domain Randomization 域隨機化模組

## 功能定位
通過在訓練中隨機化環境參數，提高策略的 Sim-to-Real 遷移魯棒性。
核心思想：如果策略能在大量隨機化的模擬環境中都表現良好，
那麼它在真實世界（只是隨機化分佈中的一個樣本）也能工作。

## 物理意義與參考文獻
- **Tobin et al. (2017)** "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World"
- **Tan et al. (2018)** "Sim-to-Real: Learning Agile Locomotion for a Quadruped Robot" — 最完整 DR 實踐
- **Peng et al. (2018)** "Sim-to-Real Transfer of Robotic Control with Dynamics Randomization"
- **OpenAI (2019)** "Solving Rubik's Cube with a Robot Hand" — 大規模 DR（100+ 參數）
- **LiDARsim (CVPR 2020)** — 學習真實 LiDAR 仿真
- **Towards Zero Domain Gap (ICCV 2023)** — LiDAR sim-to-real 差距系統研究
- **DROPO (2023)** — 數據驅動的 DR 參數分佈估計
- **MMDR (2024)** — 多模態延遲隨機化
- **Isaac Lab Discussion #2813** — 社區 DR 實踐經驗

## 隨機化類別

### 1. 感測器隨機化 (sensor_dr.py)
模擬真實 LiDAR 的各種非理想行為：
| 技術 | 參數 | 物理意義 |
|------|------|----------|
| 射線丟失 (ray dropout) | `dropout_rate=0.05` | 真實 LiDAR 5-10% 射線因反射/遮擋丟失 |
| 距離高斯雜訊 | `distance_noise_std=0.03` | 測距精度 ±3cm |
| 系統性偏差 | `bias_range=(-0.05, 0.05)` | 每通道固定偏差（安裝誤差） |
| 幽靈點 | `ghost_rate=0.002` | 多路徑反射產生的虛假近距離讀數 |
| 串擾 (crosstalk) | `crosstalk_rate=0.01` | 相鄰通道干擾 |
| 角度偏移雜訊 | `angle_offset=±0.5°` | 掃描角校準誤差 |
| 光束強度隨機 | `return_prob=0.85~1.0` | 表面材質影響回波強度 |

### 2. 動力學隨機化 (physics_dr.py)
模擬真實機器人的物理不確定性：
| 技術 | 參數 | 物理意義 |
|------|------|----------|
| 質量隨機化 | `mass_scale=(0.85, 1.15)` | 載重變化 ±15% |
| 質心偏移 | `com_offset=0.05` | 組裝誤差/載重不均 |
| 地面摩擦力 | `friction_scale=(0.7, 1.3)` | 不同地面材質 |
| 慣性矩 | `inertia_scale=(0.9, 1.1)` | 形狀/質量分佈差異 |

### 3. 致動器隨機化 (actuator_dr.py)
模擬馬達和傳動系統的非理想行為：
| 技術 | 參數 | 物理意義 |
|------|------|----------|
| 動作延遲 | `action_delay_steps=(0,2)` | 通訊/處理延遲 1-2 步 |
| 速度縮放 | `velocity_scale=(0.9, 1.1)` | 馬達效率差異 ±10% |
| 回應滯後 | `response_lag=0.1` | 一階低通濾波 τ=0.1s |
| 死區 | `deadzone=0.02` | 低速時摩擦導致不動 |
| 馬達反沖 | `backlash=0~2°` | 齒輪傳動反向死區 |

### 4.5 通訊/時序隨機化 (尚未實作，未來擴充)
模擬真實系統的時序不確定性：
| 技術 | 參數 | 物理意義 |
|------|------|----------|
| 觀測延遲 | `obs_delay=0~20ms` | 感測器處理時間 |
| 多模態延遲 (MMDR) | `lidar_delay=0~200ms` | LiDAR 全旋轉週期延遲 |
| 控制頻率抖動 | `ctrl_period=±20%` | 非 RTOS 系統時序抖動 |
| IMU 偏差 | `bias=±0.05rad, σ=0.05` | 陀螺儀漂移與雜訊 |

### 4. 外部擾動 (disturbance_dr.py)
模擬真實環境中的不可控因素：
| 技術 | 參數 | 物理意義 |
|------|------|----------|
| 隨機推力 | `push_force=(10, 30)` | 被人碰撞/地面不平 |
| 持續風力 | `wind_force=(0, 5)` | 室外風/通風系統 |
| 推力間隔 | `push_interval_s=(5, 15)` | 平均 10 秒推一次 |

### 5. 初始狀態隨機化
已整合在 `events/reset.py` 中：
- 位置：隨機安全位置 + 牆壁碰撞檢查
- 速度：±0.5 m/s（模擬非靜止啟動）
- 角速度：±0.5 rad/s
- 朝向：-π ~ +π

## 訓練中的啟用方式
```python
# 在環境配置的 EventCfg 中：
domain_randomization = EventTerm(
    func=apply_domain_randomization,
    mode="reset",
    params={
        "enable_physics": True,
        "enable_sensor_noise": True,
        "enable_external_force": True,
    },
)

# interval 事件（每步執行）：
random_push = EventTerm(
    func=apply_random_push,
    mode="interval",
    interval_range_s=(5.0, 15.0),
    params={"force_range": (10.0, 30.0)},
)
```

## 漸進式啟用策略 (Curriculum DR)
**重要原則：不要一開始就使用強隨機化，否則策略會崩潰。** (Isaac Lab Discussion #2813)

建議分階段增加隨機化強度：
1. **Phase 0**: 僅初始狀態隨機 + LiDAR 噪聲（基礎策略學習）
2. **Phase 1**: 加入動作延遲 + 速度縮放（適應致動器不確定性）
3. **Phase 2**: 加入質量/摩擦隨機 + 外部推力（完整 DR）
4. **Phase 3**: 加入極端場景（大推力、高噪聲、多延遲）→ 魯棒性壓測

進階方法：**Automatic Domain Randomization (ADR)** — 僅在性能維持閾值以上時自動擴大隨機化範圍（Isaac Lab 2.3+ 支援）。
