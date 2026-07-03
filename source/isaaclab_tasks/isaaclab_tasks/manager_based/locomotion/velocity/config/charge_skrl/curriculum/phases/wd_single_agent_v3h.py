"""warp_drive_single_agent_v3h — v3e + 小比例 head_on(從 SA3),部署主線用 (v3h, 2026-07-01)

★ 基於 v3e(penalty_smoothness=0 + ent_ang floor 0.02),把 SA3/SA4/SA5 的 head_on 比例
  從 0.20 降到 0.05(小比例),多出的比例分給 horizontal_crossing(繞行指引)。

理由(2026-07-01 用戶決策):
- v3f_vaux 主線 SA4 卡 ~61% head_on 平台(overnight 裁決:reactive+gap 即使給足探索也
  破不了,~61% 是天花板;RNN 學位置不學運動)。
- 重訓部署主線,head_on 只保留【小比例 0.05】從 SA3 起,避免 head_on 主導把 SR 壓死;
  head_on 後續在 SA6/SA7 再 fine-tune 加重(此檔先不動 SA6/7,待訓到再調)。
- 主要目標=【堪用的部署 policy】(不是刷 head_on 分數)。

其餘完全沿用 v3e(單一真實來源:scene/transition/trainer/reward 都不動)。
- position aux(predict_dim 7)保留 → 這是 train flag(--aux_velocity_topk 0),非 curriculum。
- 已證不可行的 velocity-aux / hybrid / 多幀 一律不帶。

⚠️ 從頭訓練 SA1_v3h。curriculum behavior 改變,不可 resume v3f_vaux ckpt。
⚠️ train/play 用 CHARGE_USE_ACT_HIST=0(obs 79D,沿用 v3f)。
"""

from __future__ import annotations

import copy

from .wd_single_agent_v3 import GLOBAL, _flatten_phase
from .wd_single_agent_v3e import STAGES as _V3E_STAGES

# v3h:head_on 小比例(原 v3/v3e 在 SA3-5 是 0.20)
HEAD_ON_SMALL_V3H = 0.05
# v3h (2026-07-03 用戶決策, oracle 100k 裁決後):
#   1. goal_move 全移除 — 「goal 主動移動(尤其 toward_obstacle 拖進障礙群)」是人工加難裝置,
#      真實部署 goal 不會動;oracle 證實此類碰撞多屬物理不可避,移除以對齊真實場景。
#   2. head_on 速度 (0.30,0.60)→(0.30,0.50) — 對齊真實行人衝撞速度,並把 closing speed
#      上限從 1.6 降到 1.5 m/s,留出物理可反應窗(head_on 觸發距離推導 ~3m)。
HEAD_ON_SPEED_V3H = (0.30, 0.50)


def _patch_head_on(stage: dict) -> dict:
    """deep-copy v3e stage 後套 v3h 補丁:
    (a) head_on 比例降到 HEAD_ON_SMALL_V3H(多出比例給 horizontal_crossing/patrol);
    (b) goal_move 移除(goal_move_speed=0);
    (c) head_on speed_range 壓到 HEAD_ON_SPEED_V3H。
    無該欄位的 stage(SA1/2 等)原封不動。"""
    s = copy.deepcopy(stage)
    mix = s.get("behavior", {}).get("behavior_mix")
    if isinstance(mix, dict) and mix.get("head_on", 0.0) > HEAD_ON_SMALL_V3H:
        excess = mix["head_on"] - HEAD_ON_SMALL_V3H
        mix["head_on"] = HEAD_ON_SMALL_V3H
        if "horizontal_crossing" in mix:
            mix["horizontal_crossing"] = round(mix["horizontal_crossing"] + excess, 6)
        elif "patrol" in mix:
            mix["patrol"] = round(mix["patrol"] + excess, 6)
    # (b) goal_move 修正 (2026-07-04 用戶更正): 目標【保持隨機移動】，只移除
    #     「朝障礙物移動」(toward_obstacle, 僅 SA3 有,人工加難裝置)。
    #     舊版(07-03)誤把全部歸零成靜止目標 — 已改回:速度/半徑/角速度保留原 stage 值,
    #     只把 toward_obstacle → random_walk。SA4+ 本來就是 random_walk/patrol,不動。
    scene = s.get("scene")
    if isinstance(scene, dict) and scene.get("goal_move_behavior") == "toward_obstacle":
        scene["goal_move_behavior"] = "random_walk"
    # (c) head_on 速度壓到 0.3-0.5 (2026-07-03)
    overrides = s.get("behavior", {}).get("speed_overrides")
    if isinstance(overrides, dict) and "head_on" in overrides:
        overrides["head_on"]["speed_range"] = HEAD_ON_SPEED_V3H
    return s


# v3h 的 stage 列表(巢狀 schema,供人類閱讀/再調整)
STAGES = [_patch_head_on(stage) for stage in _V3E_STAGES]


# 供 phases registry 匯出的標準格式(與 v3 結構一致)。
CONFIG = {
    "obstacle_mode": GLOBAL["obstacle_mode"],
    "upgrade_pass_required": GLOBAL["upgrade_pass_required"],
    "clear_window_on_promote": GLOBAL["clear_window_on_promote"],
    "trainer_sync_allowlist": GLOBAL["trainer_sync_allowlist"],
    "stages": [_flatten_phase(stage) for stage in STAGES],
}
