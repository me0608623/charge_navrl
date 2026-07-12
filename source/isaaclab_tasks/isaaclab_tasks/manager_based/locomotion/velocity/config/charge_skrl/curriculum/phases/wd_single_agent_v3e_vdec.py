"""warp_drive_single_agent_v3e_vdec — v3e + 速度決定性 (velocity-decisive) 強化.

★ 目的 (2026-07-10 用戶決策):逼 policy 真的用 LV-DOT 速度資訊 (vx,vy) 做「提早避讓」,
  而非用位置代理 (「近了晚點側閃」) 繞過 TTC 稅 → 避免重演 velocity 頭 KL≈0.

★ 病根 (用戶推論,已查證):v3e 在 20×20 太稀疏 (stage3-6 只有 4~6 個動態障礙) →
  「有空間晚點側閃」的位置策略就夠用 → 沒有「同位置、不同速度 → 不同正確動作」
  (B迎面/C背向/D橫穿) 的密集相遇 → policy 學不到用速度.

★ 本變體製造大量速度決定性相遇 (維持 20×20,不縮 arena 避開牆碰主宰坑):
  (a) head_on 0.20 → 0.35     — 迎面高 closing speed 常態化 (最強的速度決定性場景);
  (b) 動態障礙數 +2/stage      — 減少晚閃的逃生空間,逼提早繞;
  (c) 障礙速度 ×1.25 + 全域 obstacle_speed→1.0
                              — closing speed 上限拉高 → 晚反應 (1m) 來不及 → 必須 2~3m 就預判.
  其餘完全沿用 v3e (scene/transition/trainer/reward/exploration 不動).

⚠️ 從頭訓練 SA1 (curriculum behavior 改變,不可 resume 舊 ckpt).
⚠️ 配 config wd_sa1_lvdot_ttc (109D LV-DOT + full_material) + CLI TTC 稅 + gap 正向.
   max_active_obstacles 需 ≥13 (動態 +2 後 stage 末 static+dynamic 可達 ~13).
"""

from __future__ import annotations

import copy

from .wd_single_agent_v3 import GLOBAL, _flatten_phase
from .wd_single_agent_v3e import STAGES as _V3E_STAGES

# 速度決定性補丁參數
# ★07-11 放緩(SA3 v3e_vdec 全激進版訓練不穩:卡67%平台+週期性崩到19%,難度斷崖太陡)。
#   用戶拍板降難度:head_on 0.35→0.25、速度×1.25→1.1、動態+2→+1、全域速度1.0→0.9。
#   仍保有速度決定性(head_on 0.25 > v3e 原 0.20、closing 仍高於原版),只是斷崖變緩、可學+穩定。
HEAD_ON_VDEC = 0.25          # v3e 原 0.20 → 0.25 (溫和提高迎面比例)
SPEED_SCALE = 1.10          # 動態障礙 speed_range ×1.10 (closing 溫和↑)
SPEED_CAP = 0.80            # 速度上限
GLOBAL_OBS_SPEED = 0.9      # 全域 obstacle_speed cap → 0.9
DYN_BUMP = 1               # 動態障礙數 +1 (只對已有動態相遇的 stage≥3)
DYN_MIN_BUMP = 1


def _patch_vdec(stage: dict) -> dict:
    """deep-copy v3e stage 後套速度決定性補丁 (無該欄位的 stage 原封不動)."""
    s = copy.deepcopy(stage)
    beh = s.get("behavior", {})

    # (a) head_on 比例 → HEAD_ON_VDEC (多出的從 patrol/random_walk/static 扣)
    mix = beh.get("behavior_mix")
    if isinstance(mix, dict) and 0.0 < mix.get("head_on", 0.0) < HEAD_ON_VDEC:
        excess = round(HEAD_ON_VDEC - mix["head_on"], 6)
        mix["head_on"] = HEAD_ON_VDEC
        for k in ("patrol", "random_walk", "static"):
            if k in mix and mix[k] >= excess:
                mix[k] = round(mix[k] - excess, 6)
                break

    # (b) 障礙速度 ×SPEED_SCALE (cap),全域 obstacle_speed → GLOBAL_OBS_SPEED
    if isinstance(beh.get("obstacle_speed"), (int, float)):
        beh["obstacle_speed"] = GLOBAL_OBS_SPEED
    ov = beh.get("speed_overrides")
    if isinstance(ov, dict):
        for k, v in ov.items():
            if isinstance(v, dict) and "speed_range" in v:
                lo, hi = v["speed_range"]
                v["speed_range"] = (
                    round(min(lo * SPEED_SCALE, SPEED_CAP), 3),
                    round(min(hi * SPEED_SCALE, SPEED_CAP), 3),
                )

    # (c) 動態障礙數 +DYN_BUMP (只對已有實質動態相遇的 stage: dynamic≥3)
    scene = s.get("scene", {})
    if isinstance(scene, dict) and scene.get("dynamic_obstacles", 0) >= 3:
        scene["dynamic_obstacles"] = scene["dynamic_obstacles"] + DYN_BUMP
        if "dynamic_obstacles_min" in scene:
            scene["dynamic_obstacles_min"] = scene["dynamic_obstacles_min"] + DYN_MIN_BUMP

    return s


# v3e_vdec 的 stage 列表 (巢狀 schema,供人類閱讀/再調整)
STAGES = [_patch_vdec(stage) for stage in _V3E_STAGES]


# 供 phases registry 匯出的標準格式 (與 v3 結構一致)
CONFIG = {
    "obstacle_mode": GLOBAL["obstacle_mode"],
    "upgrade_pass_required": GLOBAL["upgrade_pass_required"],
    "clear_window_on_promote": GLOBAL["clear_window_on_promote"],
    "trainer_sync_allowlist": GLOBAL["trainer_sync_allowlist"],
    "stages": [_flatten_phase(stage) for stage in STAGES],
}
