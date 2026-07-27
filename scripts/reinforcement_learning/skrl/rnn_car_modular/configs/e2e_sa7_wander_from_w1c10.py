"""SA7：13x13 m，延續 W1 wander 配方並持續重播窄縫與長走廊。

2026-07-27 路線圖：SA6 -> SA7(13 m) -> SA8(12 m 部署尺寸)。核心要求是
**窄縫與走廊不能只當一次補課，必須在 SA7、SA8 持續重播**，否則後續密度訓練
很容易把能力洗掉。

與歷史 `e2e_sa7_k8_obb` 的差異共**三項**，其餘（stage 7、room_size 6.5、
68/10/12/10 replay 比例）完全沿用：

1. **暖啟動自 W1-c10**（不再從零）。W1-c10 在所有四個走廊模式與 Gate2 上都優於
   D0，是 SA6 血緣目前最好的 checkpoint：

   | 指標 | D0 | W1-c10 |
   |---|---|---|
   | Gate2 | 91.9 / 8.1 | **92.9 / 7.0** |
   | lateral CR | 7.41 | **5.53** |
   | longitudinal CR | 2.23 | **1.41** |
   | mixed_iid CR | 17.11 | **15.49** |
   | random_2d CR | 29.85 | **27.91** |

2. **走廊 random_2d 動力學 patrol -> wander**，並改用 env_stratified 均衡取樣。
   W1 已證實部署行人是連續有界隨機走而非 ping-pong 巡邏：同一顆 D0 在 patrol 下
   CR 33.79%、在 wander 下 21.94%（3 seed、4203 回合）。

replay 比例維持不動（本 config 未覆寫，全部繼承）：

    68% 當前階段一般場景
    10% 前階段一般場景          previous_stage_replay_fraction = 0.10
    12% 窄縫（1.2-1.4 m）        narrow_passage_fraction        = 0.12
    10% 真實長走廊 4x10 m        long_corridor_fraction         = 0.10
                                 (障礙數量改為**逐 env 抽樣**，見下方第 3 項)

晉級條件（進 SA8 前）：一般導航、長走廊、以及**修正版窄縫閘 Gate5a**
（`--narrow_gap_mode sealed`，牆隨場地延伸到外牆、只留中央窄縫）三者皆通過。
舊的 Gate5b（10 m 邊界死角壓測）不再參與晉級判定 —— 它量的是小場地邊界死角
行為，不是直穿能力。

3. **走廊障礙數量改為逐 env 混合分佈**（原本固定 4 靜態 + 2 動態）。
   2026-07-27 凍結表，平均動態數 2.25：

   | 組合 | 比例 |
   |---|---|
   | 3 靜態 + 1 動態 | 25% |
   | 4 靜態 + 2 動態 | 35% |
   | 4 靜態 + 3 動態 | 20% |
   | 5 靜態 + 3 動態 | 15% |
   | 5 靜態 + 5 動態 | 5%  |

   這是 **retention replay 不是高密壓測** —— 保留 D1/D2 才不會把 policy 訓成
   遇到人就停車；5S5D 只佔 5% 維持持續曝光，SA8 再升到 10%。
   行人之間另有互動分佈（independent / crossing / side_by_side），
   D=2 與 D>=4 為 60/20/20，D=3 為明定政策 75/25/0。

`test_sa6_sa8_replay_configs.py` 鎖住這三項允許差異；密度分佈的唯一字面值
來源是本檔的 `SA7_DENSITY_MIX`。
"""

from dataclasses import replace

from rnn_car_modular.configs.e2e_sa7_k8_obb import CONFIG as _SA7_BASE


# W1 wander 臂的 c10。W1 每 5 iter 存檔，故 c10 = checkpoint_1280。
_W1_C10_CHECKPOINT = (
    "/home/aa/IsaacLab/logs/rnn_car/"
    "sa6_k8_obb_corridor_w1_wander_s42/checkpoint_1280.pt"
)


#: 2026-07-27 凍結的 SA7 走廊密度分佈（唯一真相來源）。
#: 任何使用者都必須 import 這個常數，不得各自重打字面值 ——
#: 之前就是因為兩份測試各寫一張表，兩張都自稱「裁決」而互相矛盾。
SA7_DENSITY_MIX = (
    ((3, 1), 0.25),
    ((4, 2), 0.35),
    ((4, 3), 0.20),
    ((5, 3), 0.15),
    ((5, 5), 0.05),
)


CONFIG = replace(
    _SA7_BASE,
    name="e2e_sa7_wander_from_w1c10",
    description=(
        "SA7 at 13x13 m warm-started from the W1 wander arm's c10 checkpoint. "
        "The corridor random_2d family runs the deployment wander kinematics "
        "with balanced env-stratified sampling; the 68/10/12/10 replay mix "
        "(narrow 12%, corridor 10%) is inherited unchanged so neither "
        "capability is washed out by the denser SA7 scenes."
    ),
    checkpoint=_W1_C10_CHECKPOINT,
    no_resume_optimizer=True,
    long_corridor_random_2d_kinematics="wander",
    long_corridor_dynamic_motion_mode="env_stratified",
    # 2026-07-27 凍結的 SA7 分佈，逐字對齊 experiment_config.py 的規格註解：
    #   25% 3S1D / 35% 4S2D / 20% 4S3D / 15% 5S3D / 5% 5S5D
    # 平均動態數 2.25。這是 retention replay，不是高密壓測 —— 保留 D1/D2
    # 才不會把 policy 訓成遇到人就停車；5S5D 只佔 5% 維持持續曝光，
    # SA8 再升到 10%。
    long_corridor_obstacle_count_mix=SA7_DENSITY_MIX,
    tags=_SA7_BASE.tags + ("wander_corridor_kinematics", "sa7_13m"),
    notes=(
        "Warm start from W1-c10 (best SA6-lineage checkpoint: beats D0 on all "
        "four corridor modes and Gate2). Stage 7, room_size 6.5 (13x13 m) and "
        "the 68/10/12/10 replay mix come straight from e2e_sa7_k8_obb and are "
        "not touched. Three changes: the warm-start checkpoint, "
        "the per-env corridor density mix (25% 3S1D / 35% 4S2D / 20% 4S3D / "
        "15% 5S3D / 5% 5S5D, mean 2.25 dynamics, replacing the fixed 4S+2D), "
        "and "
        "random_2d kinematics patrol -> wander with env_stratified sampling. "
        "Narrow (12%) and corridor (10%) replay must stay on for the whole of "
        "SA7 and SA8 - the 2026-07-27 roadmap is explicit that treating them "
        "as a one-off remedial block lets the denser scenes wash the "
        "capability out. Graduation to SA8 requires general navigation, the "
        "long corridor, and the corrected narrow gate Gate5a "
        "(--narrow_gap_mode sealed, wall extends to the outer boundary so only "
        "the central opening is passable). The legacy Gate5b (10 m boundary "
        "dead-corner stress test) no longer gates advancement: it measured "
        "small-arena dead-corner behaviour, not direct-crossing ability."
    ),
)
