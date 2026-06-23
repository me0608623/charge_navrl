"""warp_drive_single_agent_v3g — 小 penalty_smoothness 抗 bang-bang 塌縮 (v3g, 2026-06-20)

★ 根因定案（SA2_v3f/SA3_v3f deterministic eval + lidar regime 分析）：
   v3e/v3f 殘留 sin 波的兇手是 penalty_smoothness=0，不是 ent_angular。

機制：penalty_smoothness 罰 |Δω|（角速度逐幀變化）。設 0 時，bang-bang ±最大角速度
（左右抵銷=平均直行）與「平順小修正直行」拿到完全相同 reward → policy 沒理由不
bang-bang → 訓練久了塌縮到這個「免費」degenerate 解（target ω 死貼 ±1.20 滿舵）。
bang-bang = 每幀最大 Δω，所以 penalty_smoothness 直接命中它。

證據：
- SA2_v3f（ent_ang 0.05, penalty 0）：飽和率 iter700~0% → iter1000 100%（漸進塌縮）。
- SA3_v3f（ent_ang 0.02, penalty 0）：iter100 就 100% 飽和（降 ent 反而更快）。
- 三重確認：stochastic 也飽和、無牆也飽和、空曠(lidar>1.2)也飽和 → 真病態，非 argmax/牆。
- 對照 sa2_v3f/210000：空曠飽和 0%（乾淨）。

v3g 與 v3e 差異（單一變因）：
1. penalty_smoothness  0.0 → **0.003**（全 stage，小阻尼讓 bang-bang 有代價）。
   ★ 0.003 ≪ v3d 的 0.015（後者太高會逼 hold-direction 慢弧 weave）。甜蜜點。
2. ent_angular floor 0.02 維持不變（非兇手）。

⚠️ reward shaping 改變，從 sa2_v3f/checkpoint_210000.pt（iter700 乾淨）重訓 SA3。
⚠️ train/play 需 CHARGE_USE_ACT_HIST=0（obs 79D，v3f 架構無 act_hist）。

其餘 stage 定義完全沿用 v3，維持單一真實來源；本檔只 deep-copy 後改
penalty_smoothness 一個欄位 + 沿用 ent floor。
"""

from __future__ import annotations

import copy

from .wd_single_agent_v3 import GLOBAL, STAGES as _V3_STAGES, _flatten_phase

# v3g 補丁常數
PENALTY_SMOOTHNESS_V3G = 0.003   # ★ 小阻尼抗 bang-bang 塌縮（v3e 是 0.0，v3d 是 0.015）
ENT_ANGULAR_FLOOR_V3G = 0.02     # 沿用 v3d/v3e：防 angular collapse（非 sin 波兇手）


def _patch_stage(stage: dict) -> dict:
    """deep-copy v3 stage 後套用 v3g 補丁：penalty_smoothness=0.003 + ent_ang floor 0.02。"""
    s = copy.deepcopy(stage)

    reward = s.setdefault("reward", {})
    reward["penalty_smoothness"] = PENALTY_SMOOTHNESS_V3G

    expl = s.setdefault("exploration", {})
    expl["entropy_angular"] = max(
        ENT_ANGULAR_FLOOR_V3G, float(expl.get("entropy_angular", 0.0))
    )

    return s


# v3g 的 stage 列表（巢狀 schema，供人類閱讀/再調整）
STAGES = [_patch_stage(stage) for stage in _V3_STAGES]


# 供 phases registry 匯出的標準格式（與 v3 結構一致）。
CONFIG = {
    "obstacle_mode": GLOBAL["obstacle_mode"],
    "upgrade_pass_required": GLOBAL["upgrade_pass_required"],
    "clear_window_on_promote": GLOBAL["clear_window_on_promote"],
    "trainer_sync_allowlist": GLOBAL["trainer_sync_allowlist"],
    "stages": [_flatten_phase(stage) for stage in STAGES],
}
