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


def _patch_head_on(stage: dict) -> dict:
    """deep-copy v3e stage 後,把 head_on 比例降到 HEAD_ON_SMALL_V3H;
    多出的比例分給 horizontal_crossing(沒有則給 patrol),保持 mix 總和不變。
    只動 head_on > 小比例 的 stage(SA3/4/5);其餘(SA1/2/6/7 無 head_on)原封不動。"""
    s = copy.deepcopy(stage)
    mix = s.get("behavior", {}).get("behavior_mix")
    if isinstance(mix, dict) and mix.get("head_on", 0.0) > HEAD_ON_SMALL_V3H:
        excess = mix["head_on"] - HEAD_ON_SMALL_V3H
        mix["head_on"] = HEAD_ON_SMALL_V3H
        if "horizontal_crossing" in mix:
            mix["horizontal_crossing"] = round(mix["horizontal_crossing"] + excess, 6)
        elif "patrol" in mix:
            mix["patrol"] = round(mix["patrol"] + excess, 6)
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
