# Dense-Mixed 部署導航 Implementation Plan（密度 ramp + shield，方向 A）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 讓單幀 LiDAR reactive policy 在高密度（12×12 正方 + 走廊型，15 靜 + 6–8 動）場景**對靜態障礙提早避障**、動態靠 shield 補尾，達堪用部署（+shield 總 SR 75–80%+）。

**Architecture:** 靜態靠 curriculum 密度非線性 ramp（SA1-8, 2→15 靜, 0→6-8 動, 正方→走廊 DR）逐階畢業（det SR≥90）；動態靠部署 shield 三層（減速閘/α slew/FGM 或 ORCA）。所有「新做」前置 kill-gate / 診斷，不盲燒。**北極星＝高密度提早避障。**

**Tech Stack:** Isaac Lab 5.1 + SKRL / `train_rnn_car_wdclip.py` / warp_drive curriculum phases / `play_rnn_car.py` det eval + `CHARGE_GRAD_ATTR` / wandb / 車端 pyrvo2(ORCA)。

**Spec:** `docs/superpowers/specs/2026-07-12-dense-deploy-density-ramp-shield-design.md`

## Global Constraints

- **部署目標**：15 靜 + 6–8 動；arena 正方 12×12m **且** 走廊型長方（暫定 ~20×7m，尺寸待量測）；goal 7–9m 近對角（**不縮短**）。
- **障礙尺寸**：靜態半徑 ≈ 0.3m；機器人半徑 0.35m；碰撞 LiDAR ≤ 0.45m。
- **config 硬約束**：`max_active_obstacles ≥ 24`（SA8 最多 15+8=23）。
- **命名**：`sa{N}_deploy_dense_ne1024_s42`（依 run_name 規範，只 stage 數字變）。
- **畢業硬門檻**：每階段 det eval **SR≥90** 才畢業，切換前必跑 det eval。
- **固定變因**：seed 42 / vf_coeff / max_grad_norm / rollout / batch / epochs / lr / entropy floor / obs 維度 — 只動 stage 密度與 arena 形狀。
- **穩定化**：高密度階段 KL early-stop（target_kl 0.015）。
- **語言**：註解/文件繁中，代碼英文。
- **監控**：訓練每 ~25 分一輪，五段式繁中 + 台北時間；別 kill 別人 GPU job。
- **LV-DOT**：K=5，位置欄保留、速度欄不動（不指望，ORCA 部署層才用）。

---

## Phase 0 — Step 0：bang-bang 成因診斷（先做，最省，gate 一切）

> **為何先做**：若 rmd `|ω|~0.68` 全時抖是**結構性 bang-bang**（obs_delay 類，[[finding_bangbang_degenerate_penalty0]]），則任何 reward shaping / 提早避障訓練都會被這個退化行為污染 → 先確認再投入。判別法：**空曠場（0 障礙）deterministic play 量 |ω|**——空曠也飽和 = 病態結構性；空曠≈0、近障礙才升 = 健康可 reward 引導。

### Task 0.1: 檢查 rmd 訓練 config 的 obs_delay 設定

**Files:**
- Read: `scripts/reinforcement_learning/skrl/rnn_car_modular/configs/wd_sa3_rmd_enc24.yaml`
- Read: 其 base config 鏈（grep `obs_delay`）

- [ ] **Step 1: 查 obs_delay_steps 是否啟用**

Run:
```bash
cd /home/aa/IsaacLab
grep -rn "obs_delay\|actuator_dr\|delay_steps" scripts/reinforcement_learning/skrl/rnn_car_modular/configs/wd_sa3_rmd_enc24.yaml
# 追 base config
grep -rn "obs_delay_steps\|obs_delay" source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge_skrl/ | grep -iv "def \|#" | head
```
Expected: 確認 rmd 是否 `obs_delay_steps=[0,0]`（無觀測延遲，健康）還是 `[0,1]`（有延遲，病態根因）。記錄結果。

- [ ] **Step 2: 記錄結論**

若 `obs_delay_steps` 非零 → 結構性 bang-bang 高度可疑，Step 0.2 空曠測會確認。若為零 → bang-bang 可能是 reward/exploration，reward shaping 有望。

### Task 0.2: 空曠場 deterministic |ω| 診斷（病態判別）

**Files:**
- Create: `/tmp/step0_bangbang_diag.sh`

**Interfaces:**
- Produces: 兩個 det eval log — 空曠場（0 障礙）+ 稀疏場（2 靜），各含反應曲線「前錐距離 → |ω|」。

- [ ] **Step 1: 寫診斷腳本（空曠 + 稀疏各一跑）**

```bash
cat > /tmp/step0_bangbang_diag.sh <<'EOF'
#!/bin/bash
cd /home/aa/IsaacLab
source /home/aa/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab
CKPT=logs/rnn_car/sa3_rmd_enc24_ne1024_s42/checkpoint_60000.pt
run_case () {  # $1=label $2=num_static
  CHARGE_USE_ACT_HIST=0 CHARGE_USE_LVDOT_OBS=1 PYTHONUNBUFFERED=1 \
  ./isaaclab.sh -p scripts/reinforcement_learning/skrl/play_eval/play_rnn_car.py \
    --task Isaac-Navigation-Charge-VLP16-Curriculum-WD --checkpoint $CKPT \
    --curriculum_version warp_drive_single_agent_v3e_rmd --stage 3 \
    --deterministic --num_envs 128 --steps 1200 --seed 42 --headless --feat_norm \
    --num_static_obs $2 --num_dynamic_obs 0 --obstacle_behavior static \
    --num_walls 2 --num_goals_override 1 --goal_distance_min 5.0 --goal_distance_max 9.0 \
    --arena_size 16 --react_curve_interval 0 \
    > /tmp/step0_$1.log 2>&1
  echo "=== $1 (static=$2) 反應曲線 ==="
  grep -A12 "反應曲線" /tmp/step0_$1.log | tail -14
}
run_case empty 0     # 空曠場: |ω| 是否仍飽和 ~0.68?
run_case sparse 2    # 稀疏場: |ω| 是否隨障礙距離變化?
echo "=== STEP0 DONE ==="
EOF
chmod +x /tmp/step0_bangbang_diag.sh
```

- [ ] **Step 2: 執行（背景，~5-8 分）**

Run: `nohup bash /tmp/step0_bangbang_diag.sh > /tmp/step0_wrap.log 2>&1 &`
Expected: 產生 `/tmp/step0_empty.log`、`/tmp/step0_sparse.log`，wrap 尾出現 `STEP0 DONE`。

- [ ] **Step 3: 判讀（病態 vs 健康）**

```
空曠場(0 障礙) |ω| 平均:
  ≳ 0.5 rad/s (仍飽和)  → ★結構性 bang-bang(退化) → reward shaping 白搭
                          → 先修結構(查 obs_delay / 用 noobsdelay 版重訓 base),
                            §5.6 Step 1 暫緩
  ≈ 0 (空曠直行)         → 健康, |ω| 由障礙觸發 → reward shaping 有望
                          → 放行 §5.6 Step 1, 且提早避障訓練不被污染
```

- [ ] **Step 4: 寫結論到 spec + 記憶**

把判讀結果（病態/健康 + obs_delay 設定）記到 spec §5.6 Step 0 欄下方，並更新記憶 [[finding_bangbang_degenerate_penalty0]]（附 rmd checkpoint_60000 空曠 |ω| 實測值）。

- [ ] **Step 5: Commit**

```bash
cd /home/aa/IsaacLab
git add docs/superpowers/specs/2026-07-12-dense-deploy-density-ramp-shield-design.md
git commit -m "diag(step0): rmd bang-bang 成因診斷結果(空曠場 |ω| 實測)"
```

**★ 決策點**：Step 0 病態 → 先處理結構（開新子任務修 obs_delay/base），Phase 1 仍可平行進行（curriculum 不受影響）。Step 0 健康 → 全速進 Phase 1。

---

## Phase 1 — 建 deploy_dense curriculum + arena 形狀 DR + config

### Task 1.1: 建 `wd_single_agent_v3e_deploy_dense.py` curriculum（非線性密度 ramp）

**Files:**
- Create: `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge_skrl/curriculum/phases/wd_single_agent_v3e_deploy_dense.py`
- Modify: `.../curriculum/phases/__init__.py`（註冊）
- Reference（照抄結構）：`.../phases/wd_single_agent_v3e_vdec2.py`

**Interfaces:**
- Produces: module-level `CONFIG = {..., "stages": [...]}`（**匯出名 = `CONFIG`，與 vdec2 一致，非 `CURRICULUM`**），8 stage，各 stage `scene` dict 含 `static_obstacles / dynamic_obstacles / dynamic_obstacles_min / arena_shape`。
- 註冊：`phases/__init__.py` 的 `PHASE_REGISTRY` dict，key = `"warp_drive_single_agent_v3e_deploy_dense"`。

- [ ] **Step 1: 寫 curriculum（在 vdec2 上改密度 ramp + arena DR）**

依 spec §3.1 表（靜態 2,4,6,8,10,12,14,15；動態 0,0,1,2,3,4,6,(6-8)；arena SA1-3 正方、SA4 引入走廊、SA5+ 正方/走廊混合）。骨架照抄 `wd_single_agent_v3e_vdec2.py` 的 `_build_vdec2_stages()` + `_flatten_phase` + 檔尾 `CONFIG = {...}` 模式，改 `_STATIC_RAMP` / 加 `_DYNAMIC_RAMP` / 加 `_ARENA_SHAPE`。

⚠️ **key 名用 vdec2 的完整 stage 名**（非 "SA1" 短名）：
```python
# vdec2 實際 stage 名: SA1_nav_bootstrap, SA2_nav_static, SA3_walls_crossing,
#   SA4_spatial_plan, SA5_endurance, SA6_dense_avoid, SA7_high_pressure, SA8_final
_STATIC_RAMP = {"SA1_nav_bootstrap":2,"SA2_nav_static":4,"SA3_walls_crossing":6,
  "SA4_spatial_plan":8,"SA5_endurance":10,"SA6_dense_avoid":12,"SA7_high_pressure":14,"SA8_final":15}
_DYNAMIC_RAMP = {"SA1_nav_bootstrap":0,"SA2_nav_static":0,"SA3_walls_crossing":1,
  "SA4_spatial_plan":2,"SA5_endurance":3,"SA6_dense_avoid":4,"SA7_high_pressure":6,"SA8_final":8}
_DYNAMIC_MIN  = {"SA8_final":6}  # SA8 動態 min6 max8；其餘 min=max
_ARENA_SHAPE = {"SA1_nav_bootstrap":"square","SA2_nav_static":"square","SA3_walls_crossing":"square",
  "SA4_spatial_plan":"mix","SA5_endurance":"mix","SA6_dense_avoid":"mix","SA7_high_pressure":"mix","SA8_final":"mix"}
# for name in _STATIC_RAMP: by_name[name]["scene"]["static_obstacles"]=..., ["dynamic_obstacles"]=..., ["arena_shape"]=...
```

- [ ] **Step 2: 註冊到 PHASE_REGISTRY**

`phases/__init__.py`：加 `from .wd_single_agent_v3e_deploy_dense import CONFIG as _wd_sa_v3e_deploy_dense`（照 line 15-16 vdec2/rmd 樣式）+ 在 `PHASE_REGISTRY` dict 加 `"warp_drive_single_agent_v3e_deploy_dense": _wd_sa_v3e_deploy_dense,`（照 line 34-35）。

- [ ] **Step 3: 測 curriculum 載入 + 密度正確（失敗測先行）**

```bash
cd /home/aa/IsaacLab
# ⚠️ PHASE_REGISTRY 走完整 package import 會觸發 isaaclab_tasks __init__ → 需 pxr(Isaac Sim)。
# 兩種驗法:① 在 Isaac Sim 環境內跑(./isaaclab.sh -p) ② headless 直接 import 該 module(不觸發 package __init__)。
# ★flatten 後的 CONFIG['stages'] 用 flat key num_obstacles_static/num_obstacles_dynamic(非 nested scene.*)。
/home/aa/miniconda3/envs/env_isaaclab/bin/python -c "
import sys; sys.path.insert(0,'source/isaaclab_tasks')
from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_skrl.curriculum.phases import wd_single_agent_v3e_deploy_dense as C
st = C.CONFIG['stages']
assert len(st)==8, f'stage 數 {len(st)}'
static = [s['num_obstacles_static'] for s in st]          # ★flat key
dyn    = [s.get('num_obstacles_dynamic') for s in st]
arena  = [s['scene'].get('arena_shape') for s in C.STAGES]  # nested STAGES 才有 arena_shape
assert static==[2,4,6,8,10,12,14,15], static
print('static ramp OK', static)
print('dynamic ramp', dyn)
print('arena_shape', arena)
assert static[-1]+dyn[-1] <= 24, 'max_active cap 不足'
print('cap check OK: SA8 total', static[-1]+dyn[-1])
"
```
Expected: `static ramp OK [2, 4, 6, 8, 10, 12, 14, 15]` + `dynamic ramp [0, 0, 1, 2, 3, 4, 6, 8]` + cap check OK（SA8 total 23 ≤ 24）。若 KeyError/AssertionError → 修 curriculum。

- [ ] **Step 4: Commit**

```bash
git add source/.../curriculum/phases/wd_single_agent_v3e_deploy_dense.py source/.../curriculum/phases/__init__.py
git commit -m "feat(curriculum): deploy_dense 非線性密度 ramp(靜2→15/動0→6-8) + arena DR 骨架"
```

### Task 1.2（已改範圍 2026-07-12：只做正方 12×12，走廊 DR 延後至 SA4 前）

> **用戶決策**：先建正方版 + 開訓 SA1（SA1-3 本就純正方），走廊 DR 等真實尺寸量測後、SA4 前再補（Task 1.2b）。
> **Recon 已定位**：arena 尺寸 = curriculum stage 的 `scene.boundary`（flatten 於 `wd_single_agent_v3.py:937` `scene.get("boundary", 8.5)`）。現行 stage 用 7.0–8.5（=14–17m）。**12×12 正方 → boundary=6.0**。走廊非對稱 boundary 基礎設施已存在（`walls.py:38` room_boundary 收 (float,float)、`wall_layout.py:415` T-corridor、`wd_sparse:79` (28,9)）——留給 Task 1.2b。

**Files:**
- Modify: `.../curriculum/phases/wd_single_agent_v3e_deploy_dense.py`（在 `_build_deploy_dense_stages()` 迴圈加 `sc["boundary"]=6.0`）

**Interfaces:**
- Consumes: 無新輸入（沿用 Task 1.1 的 stage 迴圈）
- Produces: deploy_dense 8 stage 全部 `scene.boundary = 6.0`（12×12 正方）。`arena_shape` 已存在（Task 1.1 設），Task 1.2b 才消費它做走廊 DR；本 task **不消費 arena_shape**（"mix" 目前等同正方 fallback）。

- [ ] **Step 1: 在 stage 迴圈加 boundary=6.0**

在 `wd_single_agent_v3e_deploy_dense.py` 的 `_build_deploy_dense_stages()` for-loop 內（設 static/dynamic/arena_shape 處）加一行 `sc["boundary"] = 6.0  # 12×12 正方 (半徑 6)`。

- [ ] **Step 2: 驗證 flatten 後 boundary=6.0（headless）**

```bash
cd /home/aa/IsaacLab
/home/aa/miniconda3/envs/env_isaaclab/bin/python -c "
import sys; sys.path.insert(0,'source/isaaclab_tasks')
from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_skrl.curriculum.phases import wd_single_agent_v3e_deploy_dense as C
b=[s.get('boundary') for s in C.CONFIG['stages']]
assert all(abs(x-6.0)<1e-9 for x in b), b
print('boundary OK (12x12 正方):', b)
"
```
Expected: `boundary OK (12x12 正方): [6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0]`。

- [ ] **Step 3: SA1 煙測（Isaac Sim，場景生成不崩）**

```bash
CHARGE_USE_ACT_HIST=0 PYTHONUNBUFFERED=1 timeout 300 ./isaaclab.sh -p scripts/reinforcement_learning/skrl/play_eval/play_rnn_car.py \
  --task Isaac-Navigation-Charge-VLP16-Curriculum-WD \
  --checkpoint logs/rnn_car/sa5_v3f_react_ne1024_s42/checkpoint_60000.pt \
  --curriculum_version warp_drive_single_agent_v3e_deploy_dense --stage 1 \
  --num_envs 4 --steps 10 --headless --feat_norm 2>&1 | tail -20
```
Expected: 場景生成、無 NaN/crash、boundary/room 約 ±6。（obs 維度不符可忽略，只驗場景生成。）

- [ ] **Step 4: Commit**

```bash
git add source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge_skrl/curriculum/phases/wd_single_agent_v3e_deploy_dense.py
git commit -m "feat(curriculum): deploy_dense boundary=6.0 (12×12 正方); 走廊 DR 延後 Task 1.2b"
```

### Task 1.2b（延後：走廊 DR，SA4 前補）

> 走廊真實尺寸量測後啟動。用既有非對稱 boundary 基礎設施（room_boundary tuple / T-corridor）把 `arena_shape="mix"` 接成 per-env 正方/走廊隨機。不在當前執行範圍。

### Task 1.3: 8 階段 config yaml

**Files:**
- Create: `scripts/reinforcement_learning/skrl/rnn_car_modular/configs/wd_sa{1..8}_deploy_dense.yaml`
- Reference: `configs/wd_sa3_rmd_enc24.yaml`

- [ ] **Step 1: 寫 SA1 config（其餘照改 stage/resume）**

照 `wd_sa3_rmd_enc24.yaml`，改：`curriculum_version: warp_drive_single_agent_v3e_deploy_dense`、`initial_stage: 1`、`fixed_stage: true`、`max_active_obstacles: 24`、KL early-stop `target_kl: 0.015`。SA2-8 各自 `initial_stage: N` + `resume` 前一階 checkpoint。

- [ ] **Step 2: 測 config 解析**

```bash
/home/aa/miniconda3/envs/env_isaaclab/bin/python -c "
import yaml; c=yaml.safe_load(open('scripts/reinforcement_learning/skrl/rnn_car_modular/configs/wd_sa1_deploy_dense.yaml'))
assert c['curriculum_version']=='warp_drive_single_agent_v3e_deploy_dense'
assert c.get('max_active_obstacles',0)>=24
print('SA1 config OK')"
```
Expected: `SA1 config OK`。

- [ ] **Step 3: Commit**

```bash
git add scripts/reinforcement_learning/skrl/rnn_car_modular/configs/wd_sa*_deploy_dense.yaml
git commit -m "feat(config): deploy_dense SA1-8 yaml(cap24 + KL early-stop)"
```

### Task 1.4: goal 放置可行性驗證（Phase 0 of spec，防飽和）

- [ ] **Step 1: SA8 密度空跑量 goal 放置成功率**

```bash
# 128 env reset 多次,統計 Goal resample fallback 比例
CHARGE_USE_ACT_HIST=0 PYTHONUNBUFFERED=1 timeout 400 ./isaaclab.sh -p scripts/reinforcement_learning/skrl/play_eval/play_rnn_car.py \
  --task Isaac-Navigation-Charge-VLP16-Curriculum-WD \
  --checkpoint logs/rnn_car/sa3_rmd_enc24_ne1024_s42/checkpoint_60000.pt \
  --curriculum_version warp_drive_single_agent_v3e_deploy_dense --stage 8 \
  --num_envs 128 --steps 300 --headless --feat_norm --goal_distance_min 7 --goal_distance_max 9 \
  > /tmp/goal_feasibility.log 2>&1
grep -c "Goal resample fallback\|fallback still in wall" /tmp/goal_feasibility.log
```
Expected: fallback 比例 <10%。若 >10% → 放寬 `obs_near_goal_count/radius`（**不縮 goal 距離**），重測。

- [ ] **Step 2: 記錄 + 若需要則調 near-goal 參數 + Commit**

---

## Phase 2 — SA1–SA8 逐階訓練（gated milestone，模板 × 8）

> 每階同一模板：launch → 監控 → det eval 畢業 gate。SA{N} 詳細密度見 §3.1。**SA_next 的細節待 SA_prev 畢業結果再定**（RL gating）。

### Task 2.N: 訓練 SA{N}（模板）

**Files:**
- Config: `configs/wd_sa{N}_deploy_dense.yaml`
- Log: `/tmp/sa{N}_deploy_dense.log`

- [ ] **Step 1: Launch**

```bash
PYTHONUNBUFFERED=1 CHARGE_USE_ACT_HIST=0 ./isaaclab.sh -p \
  scripts/reinforcement_learning/skrl/train/train_rnn_car_wdclip.py \
  --experiment_config wd_sa{N}_deploy_dense --headless --feat_norm \
  --run_name sa{N}_deploy_dense_ne1024_s42 > /tmp/sa{N}_deploy_dense.log 2>&1 &
```
（SA≥2 加 resume 前階 checkpoint；SA3 起若放行則掛 §5.6 靜態早避 reward，見 Phase 3。）

- [ ] **Step 2: 監控協定（每 ~25 分，五段式繁中）**

拉 wandb（SR/value_loss/approx_kl/entropy/gate）；崩盤判定：entropy>4.5 持續 + value_loss>1.0 持續 + SR<55% 持續 = 惡性 → KL early-stop 應已擋，未擋則保守處理；KL 單點尖峰回穩 = 續跑。

- [ ] **Step 3: det eval 畢業 gate**

```bash
# 訓到收斂後 det eval,SR≥90 才畢業
# (用 play_rnn_car.py --deterministic 在 SA{N} 密度 128 env)
```
Expected: **SR≥90**。未達 → 診斷（撞靜態為主=容量牆提早/撞動態為主=reactive 地板預期/撞牆凍結=獨立 bug），依 spec §5 決策樹處理，不盲進下一階。

- [ ] **Step 4: 記錄 run↔wandb id + 進下一階**

---

## Phase 3 — 靜態早避 reward 子實驗（§5.6，SA3 起，Step 0 健康才放行）

### Task 3.1: 加單一 `early_gap_reward` term（非-invariant，平滑 gate）

**Files:**
- Modify: `.../charge_skrl/mdp/rewards/gap_rewards.py`（gap_heading 現成，加 distance_gate 版）
- Modify: reward cfg（掛 term + 權重）

**Interfaces:**
- Consumes: `min_static_dist`、`gap_heading`、`v_fwd`、`omega`（現有 obs/reward 中已算）
- Produces: reward term `early_gap = gap_heading * sigmoid((min_static_dist-3.0)/1.8) * 0.9`；efficiency anchor `v_fwd - 0.3*|omega|`。

- [ ] **Step 1: 寫 term（平滑 gate，無 cliff）**

```python
import torch
def early_gap_reward(env, w: float = 0.9):
    d = env.min_static_dist               # [E] 前錐最近靜態距離
    gate = torch.sigmoid((d - 3.0) / 1.8) # 越遠越大,平滑
    return env.gap_heading * gate * w      # 非-invariant(明知改最優,這正是目的)
```

- [ ] **Step 2: 單元測（gate 單調 + 邊界）**

```python
# gate(1m) < gate(3m) < gate(6m); 全在 (0,1)
import torch
for d in [1.0,3.0,6.0]:
    g = torch.sigmoid((torch.tensor(d)-3.0)/1.8); print(d, float(g))
# 期望: 0.13 < 0.5 < 0.83
```
Expected: 單調上升、皆 ∈ (0,1)。

- [ ] **Step 3: 掛進 SA3 reward cfg，shaping/goal 比控 0.3–0.6，Commit**

### Task 3.2: SA3 高密度跑 + kill gate 驗收

- [ ] **Step 1: 訓 SA3 with early_gap（vs 無 early_gap baseline 同 seed）**
- [ ] **Step 2: det eval 量 d_safe(p5/median) + 淨路徑轉向 onset + SR + 路徑長**（用 `/tmp/analyze_traj.py` 淨路徑法）
- [ ] **Step 3: Kill gate 判定**

```
滿足任一 → 停,回 curriculum+shield,記錄「reward 路線 07-06 結論在高密度也成立」:
  d_safe p5 改善 <0.15m / onset 沒提前 / SR 掉 >3pp 或路徑明顯變長
三者同時滿足(ΔSR≥0 且 onset 提前 且 d_safe 動) → 才加第2 term(clearance)
```

- [ ] **Step 4: 記錄結論到 spec §5.6 + 記憶 [[finding_reward_economics_determines_style]]**

---

## Phase 4 — 容量牆診斷（§5.5，SA8 撞牆才做）

### Task 4.1: 刀0 per-beam LiDAR 梯度歸因（零成本先做）

- [ ] **Step 1: 復用 `CHARGE_GRAD_ATTR` hook，改測高密度各 LiDAR beam |∂/∂beam|**
- [ ] **Step 2: 判讀**：梯度集中少數 beam → MaxPool 混疊（感知瓶頸）；分散 → 決策瓶頸。記錄。

### Task 4.2:（條件）刀1 感知 encoder swap / 刀2 policy head swap

> 僅在 SA8 純 policy SR 卡 <60% 且刀0 指向對應瓶頸時做。刀1：MaxPool→CNN/Transformer(LiDAR-only)；刀2：RNN→attention。詳細待刀0 定調後開子計畫。

---

## Phase 5 — Shield / ORCA 校準（§3.2，逐層拆貢獻）

### Task 5.1: 減速閘(①) + α slew(②) 校準

- [ ] **Step 1: baseline(policy only) 部署密度 det eval 記 動態 CR / SR / freeze_ratio**
- [ ] **Step 2: +減速閘 校準 d_slow/d_stop，量 動態 CR↓ vs freeze_ratio↑ 邊際**
- [ ] **Step 3: +α slew 量 bang-bang↓ / SR 影響**

### Task 5.2: FGM(3a) / ORCA(3b) 條件接管 + H2 對照

- [ ] **Step 1: ORCA baseline（pyrvo2）量動態 CR vs reactive policy → 證「速度可用只是 RL 不學」（H2）**
- [ ] **Step 2: 逐層邊際貢獻表，部署只留正貢獻層；盯 freeze_ratio 不暴增**

---

## Phase 6 — 部署密度總驗收（§5）

### Task 6.1: 正方 + 走廊 各自 det eval 驗收表

- [ ] **Step 1: 12×12 正方 15靜+6-8動 det eval（policy only / +shield 分開）**
- [ ] **Step 2: 走廊型 同密度 det eval（正方/走廊分開報，不平均）**
- [ ] **Step 3: 對驗收門檻**：撞靜態<5% / +shield 動態 CR<10-15% / +shield 總 SR 75-80%+ / freeze_ratio 不暴增
- [ ] **Step 4: 更新記憶（run↔wandb lineage、部署配方）+ Obsidian 筆記 + Commit**

---

## Self-Review 註記

- **Spec coverage**：§3.1 密度/arena→Task1.1-1.2；§3.2 shield→Phase5；§4 訓練→Phase2；§5 驗收→Phase6；§5.5 診斷→Phase4；§5.6 reward 子實驗→Phase3；Step0→Phase0。✅ 全覆蓋。
- **待填佔位**：Task 1.2 Step 2（arena 代碼）、Phase 2 各階密度、Phase 4.2 encoder 細節——**皆標「待前置結果定位」**，因 RL gating / 代碼定位必須先探再填，非可預寫的佔位。執行到該 task 時先跑定位 step 再填實作。
- **granularity**：code artifact（curriculum/config/reward term/diag script）走 TDD-ish（測載入/密度/gate）；training run 走 launch→monitor→det-eval-gate milestone（SR≥90 gate = 驗收）。
