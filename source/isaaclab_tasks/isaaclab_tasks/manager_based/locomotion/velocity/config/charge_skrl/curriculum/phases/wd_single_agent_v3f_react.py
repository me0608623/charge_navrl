"""warp_drive_single_agent_v3f_react — SA5「反應訓練」版 (v3f-react, 2026-06-22)

★ 目的：對症 SA4_v3f 診斷出的「動態障礙晚反應碰撞」。
  診斷（deterministic 2019 回合，記憶 project_sa4_v3f_completion_diagnosis）：
  - CR 11.5%，其中撞障礙 10.9%（撞牆僅 0.6%）為唯一死因。
  - 89% 碰撞為全速行進（>0.4 m/s）撞動態障礙，97% 在 LiDAR ≤0.5m 才偵測，
    0% stuck、0% 追 goal → 死因 = 動態(patrol/crossing)障礙的「反應距離不足碰撞」。

設計兩軸（皆只作用在 SA5，其餘 stage 與 v3e 完全相同）：

【軸一 Reward】新增 clearance-gated 減速懲罰 penalty_speed_near_obs：
  障礙進入反應區 [d_stop=0.45, d_react=1.2]m 時，懲罰「快速」而非「接近」：
    逐步懲罰 = -(w/fps) * p^2 * max(v_forward, 0)，p=clip((d_react-d)/(d_react-d_stop),0,1)
  → 補上 sparse reward「接觸前無梯度」的根因：給 policy 平滑訊號「靠近就放慢」換反應時間。
  → p^2 聚焦內側危險區、慢下來即免罰（鼓勵減速通過而非凍結）。
  （實作在 rnn_car_wdclean/rewards.py，預設 w=0 → 不影響其他 stage / curriculum。）

【軸二 場景】把 SA5 動態速度壓回「可反應」範圍（反應訓練哲學）：
  現況 SA5 動態比 SA4 更快（patrol 0.30-0.65），一邊加 reward 一邊加難度會 confound、
  且若 v_rel 超過轉向能力則物理上避不掉。故先壓低速度讓 policy 學會減速避讓，
  corridor_crossing 比例降低（先練開放動態避障），難度留 SA6 加回。

⚠️ 從 SA4_v3f checkpoint_420000 接續訓練（initial_stage 5）。reward shaping 改變，
   penalty_speed_near_obs 只在 SA5 啟用；obs 維度不變（79D），可 resume SA4_v3f ckpt。
⚠️ train/play 仍需 CHARGE_USE_ACT_HIST=0（obs 79D，與 v3f 一致）。

其餘 stage 沿用 v3（經 v3e patch：penalty_smoothness=0 + ent_ang floor 0.02），維持單一真實來源。
"""

from __future__ import annotations

import copy

from .wd_single_agent_v3 import GLOBAL, STAGES as _V3_STAGES, _flatten_phase

# --- v3e 沿用補丁（全 stage）---
PENALTY_SMOOTHNESS_V3E = 0.0     # 移除 sin 波元兇（與 v3e 一致）
ENT_ANGULAR_FLOOR_V3E = 0.02     # 防 angular collapse（與 v3e 一致）

# --- v3f-react SA5 專屬補丁常數（v2「遭遇密集」版，2026-06-23）---
# ★ 設計更正：v1 同時壓低速度+稀疏化，導致「光轉向就能避→不需減速→reward 沒東西可咬」
#   （speed_x_near_obs 不降證實）。v2 反向操作：用密度+距離製造「非減速不可」的遭遇，
#   速度回到「需減速但減速有用」的中段。見 [[project_sa5_v3f_react_design]]。
SA5_PENALTY_SPEED_NEAR_OBS = 0.8     # clearance-gated 減速權重（主調參旋鈕；0=關閉）

# 場景：goal 更遠更少 + 障礙更密更近 → 逼長距離穿越密集障礙場
SA5_GOALS = 2                        # ↓ from 4：拿掉近 goal 捷徑，逼長導航（像 SA4）
SA5_GOAL_DISTANCE = (5.0, 9.0)       # ↑ 下限 from 2.0：無近 goal 速成
SA5_DYNAMIC_OBSTACLES = 7            # ↑ from 5：加密動態遭遇（+static 3 = 10 = max_active 上限）
SA5_DYNAMIC_OBSTACLES_MIN = 5        # ↑ from 3
SA5_OBS_NEAR_GOAL_COUNT = 2          # ↑ from 1：障礙擋在 goal 前
SA5_OBS_NEAR_GOAL_RADIUS = 2.0       # 障礙就生在反應區（製造遭遇，非避開）

# 速度回中段：快到「轉向來不及、必須減速換反應時間」，但慢到「減速真的避得掉」
SA5_SPEED_OVERRIDES = {
    "patrol": {"speed_range": (0.30, 0.60)},
    "random_walk": {"speed_range": (0.25, 0.55)},
    "horizontal_crossing": {"speed_range": (0.35, 0.65)},
    "path_crossing": {"speed_range": (0.30, 0.60)},
    "corridor_crossing": {"speed_range": (0.25, 0.50)},
}

# behavior_mix：加重「主動擋必經路徑」的行為（path/horizontal/corridor crossing）
SA5_BEHAVIOR_MIX = {
    "patrol": 0.15,
    "random_walk": 0.10,
    "static": 0.10,
    "horizontal_crossing": 0.20,
    "path_crossing": 0.25,
    "corridor_crossing": 0.20,
}


def _patch_stage_v3e(stage: dict) -> dict:
    """deep-copy v3 stage 後套用 v3e 補丁：penalty_smoothness=0 + ent_ang floor 0.02。"""
    s = copy.deepcopy(stage)
    reward = s.setdefault("reward", {})
    reward["penalty_smoothness"] = PENALTY_SMOOTHNESS_V3E
    expl = s.setdefault("exploration", {})
    expl["entropy_angular"] = max(
        ENT_ANGULAR_FLOOR_V3E, float(expl.get("entropy_angular", 0.0))
    )
    return s


def _patch_stage_sa5_react(stage: dict) -> dict:
    """SA5 專屬（v2 遭遇密集版）：clearance-gated 減速 reward + 製造「非減速不可」的密集遭遇。"""
    s = copy.deepcopy(stage)

    # 軸一：reward 加 clearance-gated 減速懲罰
    reward = s.setdefault("reward", {})
    reward["penalty_speed_near_obs"] = SA5_PENALTY_SPEED_NEAR_OBS

    # 軸二：場景製造密集近障礙遭遇（goal 遠少 + 障礙密近 + 擋路行為 + 速度中段）
    scene = s.setdefault("scene", {})
    scene["goals"] = SA5_GOALS
    scene["goal_distance"] = SA5_GOAL_DISTANCE
    scene["dynamic_obstacles"] = SA5_DYNAMIC_OBSTACLES
    scene["dynamic_obstacles_min"] = SA5_DYNAMIC_OBSTACLES_MIN
    scene["obs_near_goal_count"] = SA5_OBS_NEAR_GOAL_COUNT
    scene["obs_near_goal_radius"] = SA5_OBS_NEAR_GOAL_RADIUS

    behavior = s.setdefault("behavior", {})
    behavior["behavior_mix"] = dict(SA5_BEHAVIOR_MIX)
    behavior["speed_overrides"] = copy.deepcopy(SA5_SPEED_OVERRIDES)

    return s


# 先全 stage 套 v3e 補丁，再對 SA5_endurance 疊加 react 補丁。
def _build_stages() -> list[dict]:
    out = []
    for stage in _V3_STAGES:
        s = _patch_stage_v3e(stage)
        if s.get("name") == "SA5_endurance":
            s = _patch_stage_sa5_react(s)
        out.append(s)
    return out


STAGES = _build_stages()


CONFIG = {
    "obstacle_mode": GLOBAL["obstacle_mode"],
    "upgrade_pass_required": GLOBAL["upgrade_pass_required"],
    "clear_window_on_promote": GLOBAL["clear_window_on_promote"],
    "trainer_sync_allowlist": GLOBAL["trainer_sync_allowlist"],
    "stages": [_flatten_phase(stage) for stage in STAGES],
}
