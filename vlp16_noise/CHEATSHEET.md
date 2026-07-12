# Isaac Lab 雜訊 / 域隨機化 速查表（VLP-16）

> 心智模型：**Isaac Lab 給旋鈕，量測給旋多少。** 雷達雜訊的值用本包實測；物理 DR 沒實測就用慣例範圍。

---

## A. 雷達雜訊（讓雷達不乾淨）— `isaaclab.utils.noise`

| 雜訊 | 工具 | **設多少（實測）** | 內建? |
|---|---|---|---|
| σ 隨機手抖 | `GaussianNoiseCfg(mean=0.0, std=0.0087, operation="add")` | std=**0.0087** m | ✅ |
| 全域固定偏 | `ConstantNoiseCfg(bias=0.0148, operation="add")` | bias=**0.0148** m | ✅ |
| σ+bias 一起(每回合重抽bias) | `NoiseModelWithAdditiveBiasCfg(noise_cfg=Gaussian(std=0.0087), bias_noise_cfg=Gaussian(std=0.0148))` | 同上 | ✅ |
| per-ring bias(16線各異) | 內建做不到 → `vlp16_noise.apply_vlp16_noise` | 讀 `LIDAR_PER_RING_BIAS` | ❌自訂 |
| dropout 點消失 | 內建沒有 → 同上 | rate=**0.195** | ❌自訂 |
| 距離相關 σ(d) | 內建 std 固定 | slope≈0 → 用固定 | 不需要 |

**掛法**
- manager-based：`ObservationTermCfg(func=<雷達obs>, noise=GaussianNoiseCfg(mean=0.0, std=0.0087))`
- DirectRLEnv：env cfg 設 `observation_noise_model = NoiseModelWithAdditiveBiasCfg(...)`
- 完整(σ+bias+dropout+per-ring)：用 `vlp16_noise.apply_vlp16_noise(mode="full")`，見 README

⚠️ RayCaster 是理想射線、**無內建雜訊/dropout**；dropout+per-ring 一定走 `vlp16_noise.py`。

---

## B. 域隨機化 events（隨機化物理，**非**雷達）— `isaaclab.envs.mdp.events`

| 對象 | 函式 | 慣例範圍（非實測，需自己定）|
|---|---|---|
| 地面摩擦/彈性 | `randomize_rigid_body_material` | friction [0.6, 1.2] |
| 車體質量 | `randomize_rigid_body_mass`(舊 add_body_mass) | ±5–15% |
| 外力/推擾 | `apply_external_force_torque` / `push_by_setting_velocity` | 小幅 |
| 致動器增益 | `randomize_actuator_gains` | ±10–20% |

**掛法**：`EventTermCfg(func=..., mode="reset", params={...})`；mode = `startup`/`reset`/`interval`
**注意**：這是物理/動力學 DR，跟雷達雜訊兩套機制，可選；先不加也能跑。

---

## C. Route A 快速上手（最省）

```python
# 1) 最省：只加 σ（2 行）
from isaaclab.utils.noise import GaussianNoiseCfg
# 在 ObservationsCfg 的雷達項：
lidar = ObservationTermCfg(func=<你的雷達obs>, noise=GaussianNoiseCfg(mean=0.0, std=0.0087))

# 2) 完整：σ+bias+dropout+per-ring（用本包）
from vlp16_noise import load_vlp16_params, apply_vlp16_noise
#   在雷達 obs 函式內對 ranges 套 apply_vlp16_noise(..., mode="full")；細節見 README
```

**PARAMS_PATH** = 本資料夾 `isaac_lab_noise_params.py`（σ=0.0087 FIXED；**別用舊的 0.0007**）。

---

## D. 一句話總結
- σ / 全域 bias → 內建 noise cfg 就能做，填實測值即可
- per-ring / dropout → 內建做不到，用 `vlp16_noise.py`
- 物理 DR（摩擦/質量）→ 另一套 events，可選、值靠慣例
