"""warp_drive_single_agent_v3e — 移除 penalty_smoothness 的 sin 波修正版 (v3e, 2026-06-15)

★ 根因更正：sin 波不是「缺抗抽動」，而是 penalty_smoothness 本身造成的。

決定性證據（sa3_v2 baseline 對照，2026-06-15）：
- sa3_v2（penalty_smoothness=0、無 act_hist）視覺乾淨直行，但 ratio_flip p95=0.362（高）。
- v3d（penalty_smoothness=0.015）視覺 sin 波，ratio_flip p95=0.127（低）。
→ flip rate 是「反指標」：
    高 flip = 每步快抖 → 平均抵銷成直線 → 乾淨 ✅
    低 flip = 持住同方向幾步才換 → slew 累積成大弧 → 慢頻大 weave = sin 波 ❌
→ penalty_smoothness 罰 |Δratio_ang|（角速度改變）= 直接罰 flip → 逼 policy 持住同
   方向 → 低 flip → slew 累積成 sin 波。當初為「抗單幀抽動」加的懲罰，反而把無害
   的快抖壓成了有害的慢 sin 波。

v3e 與 v3d 的差異（方案 A：最小隔離 penalty_smoothness 單一變因）：
1. penalty_smoothness  0.015 → **0.0**（全 stage，移除元兇）。
2. 其餘維持 v3d：entropy_angular floor=0.02（防 angular collapse，與 sin 波無關）、
   act_hist action_error 編碼（CHARGE_ACT_HIST_MODE=action_error，obs 仍 83D）保留，
   待確認 smoothness 單獨是否為主因後，再決定 act_hist 去留。

⚠️ 從頭訓練（SA1_v3e）。reward shaping 改變，不可 resume v3d ckpt。
⚠️ train/play 仍需 CHARGE_ACT_HIST_MODE=action_error（act_hist 保留）。

其餘 stage 定義（scene / behavior / transition / trainer）完全沿用 v3，維持單一
真實來源；本檔只 deep-copy 後改 penalty_smoothness 一個欄位 + 沿用 v3d 的 ent floor。
"""

from __future__ import annotations

import copy

from .wd_single_agent_v3 import GLOBAL, STAGES as _V3_STAGES, _flatten_phase

# v3e 補丁常數
PENALTY_SMOOTHNESS_V3E = 0.0     # ★ 移除 sin 波元兇（v3d 是 0.015）
ENT_ANGULAR_FLOOR_V3E = 0.02     # 沿用 v3d：防 angular 分布過尖 collapse（與 sin 波無關）


def _patch_stage(stage: dict) -> dict:
    """deep-copy v3 stage 後套用 v3e 補丁：penalty_smoothness=0 + ent_ang floor 0.02。"""
    s = copy.deepcopy(stage)

    reward = s.setdefault("reward", {})
    reward["penalty_smoothness"] = PENALTY_SMOOTHNESS_V3E

    expl = s.setdefault("exploration", {})
    expl["entropy_angular"] = max(
        ENT_ANGULAR_FLOOR_V3E, float(expl.get("entropy_angular", 0.0))
    )

    return s


# v3e 的 stage 列表（巢狀 schema，供人類閱讀/再調整）
STAGES = [_patch_stage(stage) for stage in _V3_STAGES]


# 供 phases registry 匯出的標準格式（與 v3 結構一致）。
CONFIG = {
    "obstacle_mode": GLOBAL["obstacle_mode"],
    "upgrade_pass_required": GLOBAL["upgrade_pass_required"],
    "clear_window_on_promote": GLOBAL["clear_window_on_promote"],
    "trainer_sync_allowlist": GLOBAL["trainer_sync_allowlist"],
    "stages": [_flatten_phase(stage) for stage in STAGES],
}
