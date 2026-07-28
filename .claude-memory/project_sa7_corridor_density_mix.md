---
name: project-sa7-corridor-density-mix
description: SA7(1024env/s42)已啟動：走廊改逐env密度混合+行人互動；凍結表平均動態2.25
metadata:
  type: project
---

★★ **SA7 已啟動**（2026-07-27，run `sa7_wander_w1c10_ne1024_s42`，PID 2185472）
`--num_envs 1024 --seed 42 --timesteps 38400 --save_interval 10`，不並行 eval。
commit `5a7acb2e600`，已推 `charge_navrl/wdclean-repro-20260429-pcB`。

與 `e2e_sa7_k8_obb` 的**三項**差異：暖啟動自 W1-c10、random_2d 用 wander、
**走廊障礙改逐 env 密度混合**（原固定 4S+2D）。

**凍結密度表**（唯一字面值來源 = config 的 `SA7_DENSITY_MIX`）：
3S1D 25% / 4S2D 35% / 4S3D 20% / 5S3D 15% / 5S5D 5%，**平均動態 2.25**。
定位是 **retention replay 不是高密壓測** —— 保留 D1/D2 才不會訓成遇人就停車；
5S5D 佔 5% 維持曝光，SA8 升到 10%。（曾誤用平均 3.70 的版本被退回。）

**行人互動**：independent/crossing/side_by_side = 60/20/20（D=3 為明定政策 75/25/0）。
關鍵設計見 [[finding_interaction_sampling_order]]。

**1024 env 首批複查**（vs 64 env smoke）：family `[0.3327,0.3354,0.3319]`、
density 誤差 0.13pp、wander 374/374、geometry 74/74+34/34、間距與速差違規 0。規模放大 16 倍未退化。

**晉級 SA8 條件**：一般導航 + 長走廊 + **Gate5a**（`--narrow_gap_mode sealed`）三者皆通過；
舊 Gate5b 不參與判定。

**N1 與 λ=0.067 維持封鎖**（λ 量自 SA6/14m，SA8/12m 須重量才能翻 `_LAMBDA_IS_CALIBRATED`）。
