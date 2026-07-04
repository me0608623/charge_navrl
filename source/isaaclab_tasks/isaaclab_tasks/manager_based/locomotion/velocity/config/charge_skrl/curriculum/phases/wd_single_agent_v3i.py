"""warp_drive_single_agent_v3i — v3h + 收緊 actor update cap (8→4), aux1 深記憶配方專用 (2026-07-05)

★ 場景/雜訊/reward 與 v3h 完全相同（單一真實來源: 直接 re-patch v3h STAGES）。
  唯一差異: 全 stage trainer 的 wd_actor_update_clip 8.0 → 4.0。

理由 (2026-07-05 凌晨, SA4 v3i 第二次崩盤後):
- aux_epochs=1 配方在難 stage 反覆出現同款死亡螺旋: 連續 norm 5-7 的大 actor update
  → critic VE 先斷 → policy 崩 (SA3 v3i 75k / SA4 v3i 48-52k / SA4 ent-floor 版 82k, 三次同簽名)。
- ent floor (0.0075/0.03) 只延後崩點 (48k→82k) 沒根治 → 預案第二級: 直接掐死殺手 burst。
- aux@8 時代從不崩: 8 遍 aux 更新在 RL update 間充當「參數空間阻尼」; aux1 拿掉阻尼後,
  8.0 的 cap 太鬆。4.0 仍在健康 update norm (1-2.5) 的 1.6-4 倍,不影響正常學習。
- ⚠️ 為何要動 curriculum 而非 CLI: trainer sync 每 iter 從 curriculum 讀 cap 並覆寫 CLI 值
  (train_rnn_car_wdclip.py:2994-2996), CLI --wd_actor_update_clip 會被踩掉。

⚠️ 只給 v3i (aux_epochs 1) 血緣用; aux@8 血緣繼續用 v3h (有 aux 阻尼,8.0 沒問題)。
"""

from __future__ import annotations

import copy

from .wd_single_agent_v3 import GLOBAL, _flatten_phase
from .wd_single_agent_v3h import STAGES as _V3H_STAGES

# v3i: aux1 配方無 aux 阻尼 → actor update cap 收緊
WD_ACTOR_UPDATE_CLIP_V3I = 4.0


def _patch_actor_cap(stage: dict) -> dict:
    """deep-copy v3h stage 後把 trainer.wd_actor_update_clip 收緊到 4.0。"""
    s = copy.deepcopy(stage)
    trainer = s.get("trainer")
    if isinstance(trainer, dict) and "wd_actor_update_clip" in trainer:
        trainer["wd_actor_update_clip"] = WD_ACTOR_UPDATE_CLIP_V3I
    return s


STAGES = [_patch_actor_cap(stage) for stage in _V3H_STAGES]


CONFIG = {
    "obstacle_mode": GLOBAL["obstacle_mode"],
    "upgrade_pass_required": GLOBAL["upgrade_pass_required"],
    "clear_window_on_promote": GLOBAL["clear_window_on_promote"],
    "trainer_sync_allowlist": GLOBAL["trainer_sync_allowlist"],
    "stages": [_flatten_phase(stage) for stage in STAGES],
}
