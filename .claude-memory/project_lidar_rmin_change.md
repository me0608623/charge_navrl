---
name: future-lidar-r-min-change-0-9-0-2
description: 用戶實測 LiDAR 邊緣到人物中心最近 0.2m，sim 目前 r_min=0.9 要降到 0.2，所有 normalize / safety / reward 連動修改
metadata: 
  node_type: memory
  type: project
  originSessionId: 3ce3483d-ce16-48c2-a4e6-11086ed2cc3e
---

未來計劃把 LiDAR `r_min` 從 0.9m 改成 **0.2m**，需同步更新所有依賴此值的常數。

**Why:** 用戶 2026-06-08 實測：VLP-16 從感測器**表面**到人物中心**最近可看到 0.2m**，目前 sim 用的 r_min=0.9 嚴重高估盲區。先前 2026-04 估算的 0.5m 也不夠低。差距 0.7m 會讓 sim policy 對近距離物體誤判（sim 認為 < 0.9m 全是 r_max 盲，policy 學會「~1.0 = 遠/盲」，但真實 LiDAR 0.2-0.9m 是真實值，policy 會以為突然出現一堆近障礙物 → 抽動反應）。

**幾何修正（很重要！）：**
- VLP-16 物理直徑 ∅103mm → 半徑 0.0515m
- 用戶量的「0.2m」是 LiDAR 表面到人物中心
- sim 的 r_min 是 LiDAR optical center 到障礙物的距離
- 所以 **sim r_min = 0.2 + 0.0515 ≈ 0.25m**，不是 0.20m
- 寫成 0.20 會讓 sim 比實機多看到一圈（0.05m），policy 部署到實機反應變晚

**How to apply:** 改動時必須一次性同步以下位置（grep 確認沒有遺漏）：

1. **LiDAR sensor cfg** — `charge_env_cfg_vlp16.py:414, 476`（Policy + Critic 兩處）
2. **Curriculum cfg** — `charge_env_cfg_vlp16_curriculum.py` 內 `r_min=0.9` 處
3. **DirectMARL env** — `direct_marl/charge_marl_env.py:266` `lidar_r_min: 0.5` → 0.2
4. **obs_functions.py:176** 註解 "r_min=0.9 對應實測值" 改 0.2
5. **CLI 預設值** — `train_charge_ac.py` / `train_rnn_car_wdclip.py` `--lidar_r_min` argparse default
6. **Safety shield** — `safety_shield.py` 距離門檻 (d_safe_min, d_stop)，與 r_min 有耦合的要重算
7. **Reward gates** — `navrl_rewards.py` / dynamic_safety 的 `dmin/dmax/d_danger/d_attenuate`，原本是相對 0.9 設計的
8. **Body radius + buffer** — body_radius=0.35 + buffer=0.10 = 0.45 碰撞門檻。**注意**：r_min=0.2 < 0.45 表示感測器看得到自己身體外緣附近，需確認 LiDAR mount 位置與 collision threshold 的關係不會誤觸發
9. **Ablation metrics** — `ablation_metrics.py` 內 `d_safe_mean` 統計範圍
10. **CLAUDE.md** — 「真實 LiDAR 校準: min_range = 0.9m」要更新為 0.2m
11. **Memory** — `MEMORY.md` 內 SHOWSTOPPER bug #7 要更新

**部署 timing:**
- ❌ 不要在 v2 鏈中途改（會破壞 SA1_v2 → SA7_v2 obs distribution 延續性）
- ✅ 規劃放到 **v3 起始點**，或 SA7_v2 完訓後當乾淨重啟
- ✅ 改完後所有舊 checkpoint 不可用，必須重訓 baseline

**Related:** [[finding_policy_oscillation_stuck]] — policy 在某些 obs 配置下會震盪，r_min 修正後可能改善
