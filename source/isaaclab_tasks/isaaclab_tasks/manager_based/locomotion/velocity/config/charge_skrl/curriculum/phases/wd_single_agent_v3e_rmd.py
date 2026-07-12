"""warp_drive_single_agent_v3e_rmd — reactive-must-die (crossing 主導).

★ 目的 (2026-07-12): 逼 policy 用 LV-DOT 速度做「預判式」避障。
  v3e_vdec (速度決定性, head_on 主導) 已證 velocity 仍沒被用 (ablation ΔSR≈0)
  + encoder gate 幾乎不開 (敏感度 0.4%)。根因: head_on / 高密度是 b1/b2 型
  (反應式也能避、或都不能避), 非 b3 (只有 velocity 能避)。

★ b3 型 = 快速「橫向 / 路徑穿越」:
  反應到障礙**當前位置**去閃 → 閃進它的**未來路徑**被撞;
  只有 velocity (往哪走、多快) 才知道要「從後方 / 正確時機」通過。
  → crossing 是 velocity **因果必要** 的場景, 這才能讓 ablation ΔSR 變大。

★ 本變體 (在 v3e_vdec 基礎上, 只改 QUALITATIVE behavior mix, 不加 raw 難度避免崩盤):
  - crossing 主導: horizontal_crossing 0.30 + path_crossing 0.30 = **0.60** (原 SA3 只 0.20)
  - head_on **微降 0.10** (v3e_vdec 是 0.25); patrol/random_walk/static 各 0.10 (壓低雜訊行為)
  - crossing 速度**適度** (0.35, 0.65): 夠快逼預判, 但不極端 → 避免 v3e 激進版的崩盤
  - 密度 / dynamic count **沿用 v3e_vdec 不加** (b3 來自 geometry 而非 count, 也避免密度崩)

⚠️ 從 SA2 clean ckpt + 24D LV-DOT encoder 暖啟 (config wd_sa3_rmd_enc24.yaml, --use_lvdot_encoder)。
⚠️ 第一步 fixed_stage SA3 測: 每 30k steps velocity-zero ablation。
   ★判準: ΔSR 明顯 > enc24 baseline +1.4pp = b3 成立 (velocity 因果必要); 停滯 = 早收手。
"""

from __future__ import annotations

import copy

from .wd_single_agent_v3 import GLOBAL, _flatten_phase
from .wd_single_agent_v3e_vdec import STAGES as _VDEC_STAGES

# reactive-must-die crossing-dominated behavior mix (套用於有 crossing 能力的 stage, SA3+)
RMD_MIX = {
    "horizontal_crossing": 0.30,   # 橫向穿越
    "path_crossing": 0.30,         # ★robot→goal 路徑交叉 (最強 b3: 必須預判才能從後方過)
    "head_on": 0.10,               # 微降 (迎面左右閃都行, 非 b3, 只留少量)
    "patrol": 0.10,
    "random_walk": 0.10,
    "static": 0.10,
}
RMD_CROSS_SPEED = (0.35, 0.65)     # crossing 速度: 夠快逼預判 (原 ~0.25-0.60)
RMD_HEADON_SPEED = (0.30, 0.55)    # head_on 保守


def _patch_rmd(stage: dict) -> dict:
    """deep-copy v3e_vdec stage 後把 behavior mix 改成 crossing 主導 (只改有 crossing 能力的 stage)."""
    s = copy.deepcopy(stage)
    beh = s.get("behavior", {})
    mix = beh.get("behavior_mix")
    # 只改「已引入 crossing / head_on」的 stage (SA3+); SA1-2 無此欄位 → 原封不動
    if isinstance(mix, dict) and ("head_on" in mix or "horizontal_crossing" in mix):
        beh["behavior_mix"] = dict(RMD_MIX)
        ov = beh.setdefault("speed_overrides", {})
        ov["horizontal_crossing"] = {"speed_range": RMD_CROSS_SPEED}
        ov["path_crossing"] = {"speed_range": RMD_CROSS_SPEED}
        ov["head_on"] = {"speed_range": RMD_HEADON_SPEED}
        # patrol / random_walk 的 speed_override 沿用 v3e_vdec 既有值 (不動)
    return s


# v3e_rmd 的 stage 列表 (巢狀 schema, 供人類閱讀 / 再調整)
STAGES = [_patch_rmd(stage) for stage in _VDEC_STAGES]


# 供 phases registry 匯出的標準格式 (與 v3 結構一致)
CONFIG = {
    "obstacle_mode": GLOBAL["obstacle_mode"],
    "upgrade_pass_required": GLOBAL["upgrade_pass_required"],
    "clear_window_on_promote": GLOBAL["clear_window_on_promote"],
    "trainer_sync_allowlist": GLOBAL["trainer_sync_allowlist"],
    "stages": [_flatten_phase(stage) for stage in STAGES],
}
