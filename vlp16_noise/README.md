# VLP-16 經驗雜訊模型 — 訓練端使用說明

> **給訓練端 Claude Code**：這包是感測端（rover）實測的 VLP-16 雜訊模型，要在 Isaac Lab 訓練時注入到光達觀測，縮小 sim-to-real gap。本檔說明**資料來源、數學公式、如何接進訓練、ablation**。
> 檔案：`isaac_lab_noise_params.py`(參數) + `vlp16_noise.py`(代碼) + 本 README。

---

## 1. 資料來源（怎麼量出來的）

- 感測器：真實 **Velodyne VLP-16**（903nm，rover 從機，topic `/velodyne_points`）。
- 方法：對**平整白牆** 0.5→3.0m（每 0.5m，共 6 站）各錄 ~35s rosbag，離線重放萃取。
- **σ（隨機雜訊）= 點到平面殘差 std**：每幀對 ROI 點雲用 PCA 擬合平面，取點到平面殘差的標準差。
  - ⚠ 這不是「每幀中位數的時間 std」——那個是 σ/√N，會低估真 σ 約 10–13×（舊 bug 版 `isaac_lab_noise_params.py` 的 0.0007 就是錯的；本包 `..._FIXED` 的 0.0087 才對）。
- **bias（系統偏差）**：量測平均 − 捲尺/雷射真距（有 GT 基準）。per-ring bias = 16 條光束各自的出廠校正偏差。
- **dropout**：intensity < 閾值 的丟點率（目前為場景級近似）。
- 實測結果：σ ≈ **8.2–9.3mm**（幾乎不隨距離變 → R² 低 → 用固定 σ）；bias ≈ 1.5cm。

---

## 2. 數學模型（公式）

單條 ray 的量測距離：

```
r_measured = r_true + bias(ring, d) + ε ,   ε ~ N(0, σ)       (+ dropout: 有機率無回波)
```

| 項 | 公式 | 物理 |
|---|---|---|
| σ 隨機雜訊 | `ε ~ N(0, σ)`；σ 固定或 `σ(d)=slope·d+intercept` | 光子散粒+電子+時間抖動 |
| 全域 bias | `b(d)=slope·d+intercept`（本資料為 fixed）| ToF 固定時間偏移 |
| per-ring bias | 每 ring 一個 offset（16 維）| 各光束出廠校正不同 |
| dropout | Bernoulli(p)，命中則該 ray 無回波 | 回波能量 ∝ 反射率/距離² |

**注入方向**：模擬器（RayCaster）給的是 `r_true`（理想幾何射線）；上式把它加工成「像真實 VLP-16」。

---

## 3. 參數語意與可信度（`isaac_lab_noise_params.py`）

| 常數 | 意義 | 可信度 |
|---|---|---|
| `LIDAR_NOISE_STD_MEAN` | 固定 σ（m）| ★ 高 |
| `LIDAR_NOISE_SLOPE / _R2 / FORCED_CONSTANT` | σ(d) 模型；R² 低時 slope=0（用固定）| — |
| `LIDAR_BIAS_MEAN / _TYPE` | 全域 bias（m）/ fixed | ★ 高（有 GT）|
| `LIDAR_PER_RING_BIAS{0..15}` | 每 ring 偏差（m）| ★ 高 |
| `LIDAR_DROPOUT_RATE` | 丟點率 | 中（**場景級**，非材質級）|

> 本包只含 **white_wall**（σ/bias/per-ring/dropout 的乾淨來源）。human/acrylic 材質的 σ 為 null（非測距精度目標），acrylic 的 dropout(θ) 尚在改進，暫勿用。

---

## 4. 如何接進 Isaac Lab 訓練 ⚠ 兩個必確認

1. **RayCaster 是理想幾何射線**（Warp GPU，無 intensity、無內建雜訊/dropout）→ 雜訊要在「打完 ray→進 policy 前」加。
   - 取距離：`ranges = ‖data.ray_hits_w − data.pos_w‖`（`ray_hits_w` 形狀 `[N,B,3]`）；未命中 inf/nan 要 `nan_to_num` 成 `max_range`。
2. **flat ray → ring 對應**：RayCaster 把 pattern 攤平成 B 條 ray，**無 ring 索引**。依你的 `LidarPatternCfg`（channels=16 × 水平取樣 H）推：
   - 「同 ring 掃完 H 方位再換 ring」→ `ring_idx = arange(B) // H`
   - 「每方位掃 16 rings 再換方位」→ `ring_idx = arange(B) % 16`
   - **務必用一幀驗證**（把 per-ring bias 設成 ring 編號，看輸出分層對不對）。

### 建議：自訂 observation 函式（manager-based）

```python
# your_task/mdp/observations.py
import torch
from isaaclab.managers import SceneEntityCfg
from vlp16_noise import load_vlp16_params, apply_vlp16_noise   # 本包

_PARAMS = None; _RING = None
MAX_RANGE = 100.0                                # 依 LidarPatternCfg.max_distance
PARAMS_PATH = "/home/aa/IsaacLab/vlp16_noise/isaac_lab_noise_params.py"

def vlp16_lidar_ranges(env, sensor_cfg: SceneEntityCfg = SceneEntityCfg("lidar")):
    global _PARAMS, _RING
    s = env.scene.sensors[sensor_cfg.name]
    r = torch.norm(s.data.ray_hits_w - s.data.pos_w.unsqueeze(1), dim=-1)   # [N,B]
    r = torch.nan_to_num(r, nan=MAX_RANGE, posinf=MAX_RANGE).clamp_(max=MAX_RANGE)
    if _PARAMS is None:
        _PARAMS = load_vlp16_params(PARAMS_PATH, env.device)
        B = r.shape[1]; H = B // 16                 # ⚠ 見上；必要時改 % 16
        _RING = (torch.arange(B, device=env.device) // H).clamp_(max=15).long()
    mode = getattr(env.cfg, "vlp16_noise_mode", "full")   # ablation 開關
    return r if mode == "ideal" else apply_vlp16_noise(r, _RING, _PARAMS, MAX_RANGE, mode)
```
把光達那項 obs 換成此函式；`ObservationTermCfg.noise` 留 None（雜訊已在函式內處理）。

### 簡版（只要 σ+全域 bias，用內建）
```python
from isaaclab.utils.noise import NoiseModelWithAdditiveBiasCfg, GaussianNoiseCfg
observation_noise_model = NoiseModelWithAdditiveBiasCfg(   # DirectRLEnvCfg 欄位
    noise_cfg      = GaussianNoiseCfg(mean=0.0, std=LIDAR_NOISE_STD_MEAN, operation="add"),
    bias_noise_cfg = GaussianNoiseCfg(mean=0.0, std=abs(LIDAR_BIAS_MEAN), operation="add"))
```
（不含 per-ring/dropout；且作用於整段 obs，僅適用「obs 幾乎只有光達」。）

---

## 5. Ablation（論文核心，用 `vlp16_noise_mode` 切）
| mode | 內容 |
|---|---|
| `ideal` | 理想光達（baseline）|
| `sigma` | 只加 N(0,σ) |
| `bias` | 只加 per-ring+全域 bias |
| `dropout` | 只加丟點 |
| `full` | σ+bias+dropout |

各變體同 seeds 訓練 → 部署真實 rover 量 **成功率/碰撞率/SPL** → 分析哪個成分對遷移最關鍵。
**主張 (full)>(ideal) 必須真機驗證，不能只憑「塞進去了」。**

## 6. Checklist
- [ ] `PARAMS_PATH` 指到本包 `isaac_lab_noise_params.py`（σ=0.0087 的 FIXED 版，**別用舊的 0.0007**）
- [ ] 確認 sensor 名稱、`max_distance`、pattern channels/H → 設 `MAX_RANGE` 與 ring 映射
- [ ] 用一幀驗證 ring 映射
- [ ] 接 `vlp16_lidar_ranges` 進 obs，加 `vlp16_noise_mode` 到 env cfg
- [ ] 跑 (a)–(e) ablation，真機評估
