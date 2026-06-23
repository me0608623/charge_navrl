"""warp_drive_single_agent_v3d — v3c 鏈的抗抽動強化版 (v3d, 2026-06-12)

這個檔案不重新定義 stage，而是在 `warp_drive_single_agent_v3` 的基礎上做兩處
最小化補丁，專門解決 SA2/SA3 出現的「目標在正前方仍走 sin 波」抽動問題：

v3d 與 v3 的差異：
1. penalty_smoothness  0.005 → 0.015（全 stage）
   - v3 的 0.005 已證明擋不住 dense reward；加重 |Δratio_ang| 懲罰作為更強的
     inductive bias，從 SA1 第一天就讓「平滑」成為 reward landscape 的一部分。
2. entropy_angular floor = 0.02（全 stage）
   - v3 SA3 把 ent_ang sync 壓到 0.01，是 angular 動作分布過尖、policy 收斂進
     limit cycle（左右抖）的直接導火線。v3d 對所有 stage 設 floor=0.02，撐住
     idx 9（零角速度）的 probability mass，避免鎖死。

搭配（非本檔，但屬同一 v3d 方案）：
- obs act_hist 改 delta 編碼（CHARGE_ACT_HIST_MODE=delta）— 斷 "copy 上一步" shortcut
- model act_hist dropout（experiment_config act_hist_dropout=0.2）— 保險絲

其餘 stage 定義（scene / behavior / transition / trainer）完全沿用 v3，
維持單一真實來源；本檔只 deep-copy 後改兩個欄位。

⚠️ 從頭訓練（SA1_v3d）。obs 維度仍 83D，但 act_hist 語義改變（delta）+ reward
   shaping 改變，不可 resume v3/v3c ckpt。
"""

from __future__ import annotations

import copy

from .wd_single_agent_v3 import GLOBAL, STAGES as _V3_STAGES, _flatten_phase

# v3d 補丁常數
PENALTY_SMOOTHNESS_V3D = 0.015   # v3: 0.005 → 加重抗抽動
ENT_ANGULAR_FLOOR_V3D = 0.02     # 所有 stage 的 entropy_angular 下限（防 collapse 鎖死）


def _patch_stage(stage: dict) -> dict:
    """deep-copy v3 stage 後套用 v3d 抗抽動補丁。"""
    s = copy.deepcopy(stage)

    reward = s.setdefault("reward", {})
    reward["penalty_smoothness"] = PENALTY_SMOOTHNESS_V3D

    expl = s.setdefault("exploration", {})
    expl["entropy_angular"] = max(
        ENT_ANGULAR_FLOOR_V3D, float(expl.get("entropy_angular", 0.0))
    )

    return s


# v3d 的 stage 列表（巢狀 schema，供人類閱讀/再調整）
STAGES = [_patch_stage(stage) for stage in _V3_STAGES]


# 供 phases registry 匯出的標準格式（與 v3 結構一致）。
CONFIG = {
    "obstacle_mode": GLOBAL["obstacle_mode"],
    "upgrade_pass_required": GLOBAL["upgrade_pass_required"],
    "clear_window_on_promote": GLOBAL["clear_window_on_promote"],
    "trainer_sync_allowlist": GLOBAL["trainer_sync_allowlist"],
    "stages": [_flatten_phase(stage) for stage in STAGES],
}
