#!/usr/bin/env python3
"""
play_rnn_car.py — Charge RL 模組化 RNN 模型 Play 視覺化腳本

功能:
  - 載入訓練完的 checkpoint 並在 Isaac Lab GUI 中即時運行
  - 支援 BEV 俯視圖、LiDAR 光線、RNN aux 預測 debug
  - 支援 VO Safety Shield 後處理
  - 所有場景參數（障礙物數/行為、牆壁、目標）均可 CLI 覆寫

預設:
  - 自動尋找 logs/rnn_car 下最新的 checkpoint
  - 使用 1 個環境
  - 固定於指定 curriculum stage 進行觀察

用法範例:
  # 基本 play（自動找最新 checkpoint）
  python play_rnn_car.py --bev_vis --bev_frame world --deterministic

  # 指定 checkpoint + phase 3 場景
  python play_rnn_car.py --checkpoint xxx.pt --stage 3 --bev_vis

  # 覆寫場景參數
  python play_rnn_car.py --checkpoint xxx.pt --num_dynamic_obs 6 --num_walls 3 --wall_length 4.0

  # 啟用 VO Shield + 動態障礙物
  python play_rnn_car.py --checkpoint xxx.pt --use_vo_shield --scripted_obstacles
"""

import argparse
import glob
import os
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher


# ============================================================================
# ★★★ 場景參數設定區 — 直接在這裡改，不用打 CLI ★★★
# ============================================================================
# 數字直接填數字，文字要加引號 "..."，True/False 不加引號。
#
# --- 基礎設定 ---
CHECKPOINT       = None    # checkpoint .pt 路徑。None=自動搜尋最新
STAGE            = 1       # 固定 curriculum 階段（1-indexed）
STAGE_PARAMETER  = True    # True=完整載入訓練時 STAGE 的場景參數（下方場景元素區全部忽略）
                           # False=使用下方手動設定區 / CLI 覆寫
STEPS            = 3000    # 最大 play 步數
CAMERA           = "top"   # "top"=俯視 / "follow"=跟隨 / "side"=側視
DETERMINISTIC    = True    # True=確定性動作 / False=隨機探索
REAL_TIME        = False   # True=以真實時間步進（插入 sleep）

# --- 場景元素（STAGE_PARAMETER=True 時此區全部忽略，直接用 stage 定義） ---
NUM_GOALS        = 1       # 目標數量
NUM_STATIC_OBS   = 1       # 靜態障礙物數量
NUM_DYNAMIC_OBS  = 2       # 動態障礙物數量
NUM_WALLS        = 1       # 內部牆壁數量（min=max=N）
WALL_LENGTH      = 3.0     # 牆壁長度 m
GOAL_DIST_MIN    = 2.0     # 最小目標距離 m
GOAL_DIST_MAX    = 6.0     # 最大目標距離 m
EPISODE_LENGTH_S = None    # episode 秒數。None=沿用 env 預設
OBSTACLE_SPEED   = 0.8     # 障礙物速度倍率（0.0=靜止, 1.0=全速）
OBSTACLE_BEHAVIOR = "patrol"  # 障礙物行為模式:
                           #   "static"            — 靜止不動
                           #   "patrol"             — 2-4 waypoint 巡邏
                           #   "random_walk"        — 隨機改方向
                           #   "horizontal_crossing"— 橫向穿越
                           #   "path_crossing"      — 路徑交叉
                           #   "near_miss"          — 擦過機器人
                           #   "corridor_crossing"  — 走廊穿越
                           #   "occlusion"          — 多物體遮擋
                           #   "mixed"              — 用 stage config 的比例混合
                           #   None                 — 不變更
NO_WALLS         = False   # True=移除所有內部牆壁
USD_SCENE        = None    # USD 場景路徑：None=程式化迷宮（預設）
                           #   "warehouse"  → Isaac Sim 倉庫
                           #   "hospital"   → Isaac Sim 醫院
                           #   "/abs/path"  → 自訂 USD 檔案

# --- 目標附近障礙物 ---
OBS_NEAR_GOAL_COUNT = 1    # 在 goal 附近強制生成的障礙物數量（0=關閉）
OBS_NEAR_GOAL_RADIUS = 2.0 # goal 附近多少米範圍內生成障礙物

# --- 障礙物運動 ---
SCRIPTED_OBSTACLES = False # True=啟用 interval events 讓障礙物動
                           # False=凍結不動（與訓練一致）

# --- LiDAR Distance Bias (Play-only) ---
LIDAR_DIST_BIAS  = 0.0     # 加到 LiDAR 距離的偏移量 (m)，讓 agent 覺得障礙物更遠
                           # 例如 0.2 → 實際 0.25m 碰撞邊界 agent 感知為 0.45m
                           # 0.0 = 關閉（預設）

# --- Safety Shield ---
USE_SAFETY_SHIELD = False  # True=距離安全護盾（限速/停止）
USE_VO_SHIELD    = False  # True=VO 預測式護盾（預設關閉，避免污染純 policy play）
SHIELD_MODE      = "soft"  # "soft"=線性降速 / "hard"=強制停止

# --- RVO2 (ORCA) Safety Filter ---
USE_RVO2_FILTER      = False  # True=啟用 RVO2 ORCA 多障礙物安全過濾
RVO2_TIME_HORIZON    = 2.5    # ORCA 動態障礙物預測時域（秒）
RVO2_TIME_HORIZON_STATIC = 0.3  # ORCA 靜態障礙物預測時域（秒）
RVO2_ANGLE_THRESHOLD = 60.0   # 非全向 fallback 角度閾值（度）
RVO2_SAFETY_MARGIN   = 0.10   # 安全邊距（加在 robot_radius 上）
RVO2_CULLING_RADIUS  = 5.0    # 障礙物篩選半徑（超出不考慮）
RVO2_NEIGHBOR_DIST   = 10.0   # ORCA neighbor 搜尋距離
RVO2_MAX_NEIGHBORS   = 20     # ORCA 最大鄰居數
RVO2_TACTICAL_RETREAT = True   # 側向威脅強制倒車（Zone B 戰術後退）
RVO2_DISP_THRESHOLD  = 0.3    # displacement stuck 偵測閾值（m）
RVO2_DISP_WINDOW     = 15     # displacement stuck 偵測視窗（步數）

# --- LiDAR ---
LIDAR_NO_NOISE   = False   # True=關閉 LiDAR 雜訊
LIDAR_VIS        = True    # True=顯示 LiDAR 射線
LIDAR_R_MIN      = 0.1     # LiDAR 最小量測距離 m（盲區）
COLLISION_DIST   = 0.45    # 碰撞判定距離 m（= body_radius 0.35 + buffer 0.10）

# --- 診斷工具 ---
AUX_DEBUG        = True   # True=印出 RNN aux 7D 預測 vs 真實值
AUX_DEBUG_INTERVAL = 25    # 每 N 步印一次 aux debug
PLAY_DIAG        = False   # True=累積航向/速度/距離導航診斷
DIAGNOSTIC       = False   # True=LiDAR 可觀察性診斷

# --- BEV 俯視圖 ---
BEV_VIS          = True    # True=開啟 BEV 俯視圖視窗
BEV_FRAME        = "world"  # "body"=車體座標（前方為上）/ "world"=世界座標
BEV_UPDATE_INTERVAL = 5    # 每 N 步更新一次 BEV（matplotlib 是純 CPU，太頻繁會拖慢 GPU）
BEV_MAX_RANGE    = 20.0    # BEV 最大顯示範圍 m


# ============================================================================
# CLI 參數定義（上方設定區的值會作為 default，CLI 可再覆寫）
# ============================================================================
parser = argparse.ArgumentParser(description="Charge RL — 模組化 RNN Play 腳本")


_original_argv = list(sys.argv)

# --- 基礎設定 ---
parser.add_argument("--task", type=str, default="Isaac-Navigation-Charge-VLP16-Curriculum-NavRL",
                    help="Gymnasium 環境 ID（對應 __init__.py 註冊名稱）")
parser.add_argument("--num_envs", type=int, default=1,
                    help="並行環境數量（play 通常用 1）")
parser.add_argument("--checkpoint", type=str, default=CHECKPOINT,
                    help="checkpoint .pt 路徑。None 時自動搜尋最新")
parser.add_argument("--stage", type=int, default=STAGE,
                    help="固定 curriculum 階段（1-indexed）用於觀察")
parser.add_argument("--curriculum_version", type=str, default=None,
                    help="覆寫 curriculum 版本（如 warp_drive_single_agent_v1）")
parser.add_argument("--steps", type=int, default=STEPS,
                    help="最大 play 步數")
parser.add_argument("--camera", type=str, default=CAMERA, choices=["top", "follow", "side"],
                    help="攝影機視角：top=俯視 / follow=跟隨 / side=側視")
parser.add_argument("--deterministic", action="store_true", default=DETERMINISTIC,
                    help="動作選擇用 argmax（確定性）而非 sampling（探索）")
parser.add_argument("--seed", type=int, default=None,
                    help="環境隨機種子（控制 obstacle/goal 生成順序，None=使用 env config 預設 42）")
parser.add_argument("--real_time", action="store_true", default=REAL_TIME,
                    help="以真實時間步進（插入 sleep 模擬 dt）")

# --- 場景覆寫 ---
# 上方設定區的值作為 default；CLI 參數可再覆寫
# 注意：是否使用 stage 場景參數請改檔案上方 STAGE_PARAMETER，不提供 CLI flag。
parser.add_argument("--num_goals_override", type=int, default=NUM_GOALS,
                    help="覆寫目標數量")
parser.add_argument("--num_static_obs", type=int, default=NUM_STATIC_OBS,
                    help="覆寫靜態障礙物數量")
parser.add_argument("--num_dynamic_obs", type=int, default=NUM_DYNAMIC_OBS,
                    help="覆寫動態障礙物數量")
parser.add_argument("--num_walls", type=int, default=NUM_WALLS,
                    help="覆寫內部牆壁數量（設為 walls_min = walls_max = N）")
parser.add_argument("--wall_length", type=float, default=WALL_LENGTH,
                    help="覆寫牆壁長度 (m)")
parser.add_argument("--goal_distance_min", type=float, default=GOAL_DIST_MIN,
                    help="覆寫最小目標距離 (m)")
parser.add_argument("--goal_distance_max", type=float, default=GOAL_DIST_MAX,
                    help="覆寫最大目標距離 (m)")
parser.add_argument("--episode_length_s", type=float, default=EPISODE_LENGTH_S,
                    help="覆寫 episode 長度（秒）。STAGE_PARAMETER=True 且未指定時使用 stage 的 episode_s")
parser.add_argument("--obstacle_speed", type=float, default=OBSTACLE_SPEED,
                    help="覆寫障礙物速度倍率 (0.0=靜止, 1.0=全速)")
parser.add_argument("--obstacle_boundary", type=float, default=None,
                    help="覆寫障礙物生成邊界 (m)。例如 5.0 表示障礙物只在機器人 ±5m 內生成")
parser.add_argument("--obstacle_behavior", type=str, default=OBSTACLE_BEHAVIOR,
                    choices=["static", "patrol", "random_walk", "horizontal_crossing",
                             "path_crossing", "near_miss", "corridor_crossing", "occlusion",
                             "mixed"],
                    help="覆寫障礙物行為模式（需要 BehaviorScheduler）。"
                         "mixed=使用 stage config 的 behavior_mix 比例")

# --- Reward 模式（消融實驗用） ---
parser.add_argument("--reward_mode", type=str, default="current",
                    choices=["current", "navrl_ground_v1", "navrl_ground_v2", "navrl_ground_v3",
                             "navrl_ground_v4", "navrl_ground_v5", "navrl_ground_v6",
                             "navrl_ground_v7", "navrl_ground_v8"],
                    help="Reward 計算版本（current=env_cfg 預設）")
parser.add_argument("--v_gate_mode", type=str, default="baseline",
                    choices=["baseline", "floor", "softer"],
                    help="v_gate 消融：goal attraction 衰減模式")
parser.add_argument("--progress_gate_mode", type=str, default="baseline",
                    choices=["baseline", "delayed_negative", "weaken_negative"],
                    help="progress_gate 消融：danger zone 門檻模式")
parser.add_argument("--use_gap_reward", action="store_true", default=False,
                    help="啟用 gap-seeking reward")
parser.add_argument("--gap_reward_type", type=str, default="heading",
                    choices=["heading", "clearance", "both"])
parser.add_argument("--gap_reward_weight", type=float, default=5.0)
parser.add_argument("--directional_gate", action="store_true", default=False,
                    help="啟用方向性 gate（v16 用）")
parser.add_argument("--gate_cone_half_bins", type=int, default=6)
parser.add_argument("--gate_cone_bottom_k", type=int, default=3)
parser.add_argument("--gate_omni_blend", type=float, default=0.2)

# --- Reward 權重微調 ---
parser.add_argument("--dynamic_safety_mode", type=str, default="log_distance",
                    choices=["log_distance", "closing_risk"])
parser.add_argument("--goal_vel_gate_beta", type=float, default=0.2)
parser.add_argument("--progress_scale_gamma", type=float, default=0.3)
parser.add_argument("--w_goal", type=float, default=500.0, help="到達目標獎勵權重")
parser.add_argument("--w_vel", type=float, default=10.0, help="速度獎勵權重")
parser.add_argument("--w_prog", type=float, default=12.0, help="進度獎勵權重")
parser.add_argument("--w_ss", type=float, default=3.0, help="靜態安全獎勵權重")
parser.add_argument("--w_ds", type=float, default=4.0, help="動態安全獎勵權重")
parser.add_argument("--w_smooth", type=float, default=-0.05, help="平滑度懲罰權重")
parser.add_argument("--w_time", type=float, default=-0.1, help="時間懲罰權重")
parser.add_argument("--w_collision", type=float, default=-100.0, help="碰撞懲罰權重")
parser.add_argument("--w_alive", type=float, default=0.2, help="存活獎勵權重")
parser.add_argument("--goal_vel_use_soft_gate", action="store_true", default=False)
parser.add_argument("--reward_speed_v05", action="store_true", default=False)
parser.add_argument("--ds_weight_boost", type=float, default=1.0)
parser.add_argument("--risk_sigma", type=float, default=None)
parser.add_argument("--b_risk", type=float, default=None)
parser.add_argument("--ss_lower_mode", type=str, default=None, choices=["aggressive", "moderate"])
parser.add_argument("--ss_raise_mode", action="store_true", default=False)

# --- LiDAR Distance Bias ---
parser.add_argument("--lidar_dist_bias", type=float, default=LIDAR_DIST_BIAS,
                    help="LiDAR 距離偏移 (m)：加到 LiDAR 觀測值上，讓 agent 覺得障礙物更遠。"
                         "例如 0.2 → 實際碰撞邊界 0.25m 的位置 agent 感知為 0.45m。0=關閉")

# --- Safety Shield ---
parser.add_argument("--use_safety_shield", action="store_true", default=USE_SAFETY_SHIELD,
                    help="啟用距離安全護盾（action 後處理）")
parser.add_argument("--no_use_safety_shield", dest="use_safety_shield", action="store_false",
                    help="關閉距離安全護盾（純 policy play/eval 用）")
parser.add_argument("--use_vo_shield", action="store_true", default=USE_VO_SHIELD,
                    help="啟用 VO (Velocity Obstacle) 預測式護盾")
parser.add_argument("--no_use_vo_shield", dest="use_vo_shield", action="store_false",
                    help="關閉 VO 護盾（純 policy play/eval 用）")
parser.add_argument("--shield_mode", type=str, default=SHIELD_MODE, choices=["soft", "hard"],
                    help="護盾模式：soft=線性降速 / hard=強制停止")
parser.add_argument("--vo_horizon", type=float, default=1.0,
                    help="VO 預測時域 (秒)")
parser.add_argument("--vo_safety_radius", type=float, default=0.45,
                    help="VO 安全半徑 (m)，= body_radius + buffer")
parser.add_argument("--vo_evade_gain", type=float, default=1.0,
                    help="VO 角度迴避增益")

# --- RVO2 (ORCA) Safety Filter ---
parser.add_argument("--use_rvo2_filter", action="store_true", default=USE_RVO2_FILTER,
                    help="啟用 RVO2 ORCA 多障礙物安全過濾（需要 pyrvo2）")
parser.add_argument("--rvo2_time_horizon", type=float, default=RVO2_TIME_HORIZON,
                    help="ORCA 動態障礙物預測時域（秒）：越長越保守")
parser.add_argument("--rvo2_time_horizon_static", type=float, default=RVO2_TIME_HORIZON_STATIC,
                    help="ORCA 靜態障礙物預測時域（秒）")
parser.add_argument("--rvo2_angle_threshold", type=float, default=RVO2_ANGLE_THRESHOLD,
                    help="非全向 fallback 角度閾值（度）：v_safe 方向偏離 heading 超過此值時減速轉向")
parser.add_argument("--rvo2_safety_margin", type=float, default=RVO2_SAFETY_MARGIN,
                    help="ORCA 安全邊距（m），加在 robot_radius(0.35) 上")
parser.add_argument("--rvo2_culling_radius", type=float, default=RVO2_CULLING_RADIUS,
                    help="ORCA 障礙物篩選半徑（m），超出此距離的障礙物不送入 solver")
parser.add_argument("--rvo2_neighbor_dist", type=float, default=RVO2_NEIGHBOR_DIST,
                    help="ORCA neighbor 搜尋距離（m）")
parser.add_argument("--rvo2_max_neighbors", type=int, default=RVO2_MAX_NEIGHBORS,
                    help="ORCA 最大鄰居數")
parser.add_argument("--rvo2_obs_inflation", type=float, default=1.8,
                    help="非合作障礙物半徑膨脹倍率（動態障礙物不參與 ORCA 50/50 避讓，需膨脹補償）")
parser.add_argument("--rvo2_tactical_retreat", action="store_true", default=RVO2_TACTICAL_RETREAT,
                    help="側向威脅戰術後退：60~120 度時強制倒車避讓（關閉則慢速前進+轉向）")
parser.add_argument("--no_rvo2_tactical_retreat", dest="rvo2_tactical_retreat", action="store_false",
                    help="關閉側向威脅戰術後退")
parser.add_argument("--rvo2_disp_threshold", type=float, default=RVO2_DISP_THRESHOLD,
                    help="displacement stuck 偵測閾值（m），若在 disp_window 步內移動小於此值判定 stuck")
parser.add_argument("--rvo2_disp_window", type=int, default=RVO2_DISP_WINDOW,
                    help="displacement stuck 偵測視窗（步數）")

# --- 障礙物運動控制 ---
parser.add_argument("--scripted_obstacles", action="store_true", default=SCRIPTED_OBSTACLES,
                    help="啟用 scripted 動態障礙物運動（interval events）。"
                         "關閉時障礙物凍結不動。"
                         "注意：未來將被 BehaviorScheduler 取代")
parser.add_argument("--no_walls", action="store_true", default=NO_WALLS,
                    help="移除所有內部牆壁")
parser.add_argument("--usd_scene", type=str, default=USD_SCENE,
                    help="使用 Isaac Sim USD 場景取代程式化迷宮。"
                         "Nucleus: warehouse, hospital, grid, simple_room, rough_plane | "
                         "本機: 3floor, 3floor_v1, 3floor_v2 | 或自訂 USD 絕對路徑。"
                         "啟用後自動停用程式化牆壁並擴展 LiDAR 偵測範圍")

# --- 目標附近障礙物 ---
parser.add_argument("--obs_near_goal_count", type=int, default=OBS_NEAR_GOAL_COUNT,
                    help="在 goal 附近強制生成的障礙物數量（0=關閉）")
parser.add_argument("--obs_near_goal_radius", type=float, default=OBS_NEAR_GOAL_RADIUS,
                    help="goal 附近多少米範圍內生成障礙物")

# --- LiDAR 設定 ---
parser.add_argument("--lidar_no_noise", action="store_true", default=LIDAR_NO_NOISE,
                    help="關閉 LiDAR 雜訊（distractor + Unoise）")
parser.add_argument("--lidar_r_min", type=float, default=LIDAR_R_MIN,
                    help="LiDAR 最小量測距離（盲區）m。實機 VLP16 ≈ 0.9")
parser.add_argument("--collision_dist", type=float, default=COLLISION_DIST,
                    help="碰撞判定距離 m（= body_radius + buffer）。預設 0.45")
parser.add_argument("--no_goal_movement", action="store_true", default=False,
                    help="強制關閉 goal movement（即使 stage config 有設定）")
parser.add_argument("--lidar_vis", action="store_true", default=LIDAR_VIS,
                    help="啟用 LiDAR 光線視覺化（預設開啟）")
parser.add_argument("--no_lidar_vis", action="store_true", default=False,
                    help="關閉 LiDAR 光線視覺化")

# --- 診斷工具 ---
parser.add_argument("--lidar_sanity", action="store_true", default=False,
                    help="執行 LiDAR 幾何完整性測試（z_filter 驗證），測完即退出")
parser.add_argument("--diagnostic", action="store_true", default=DIAGNOSTIC,
                    help="印出 LiDAR 可觀察性診斷（前 10 步 + 每 500 步）")
parser.add_argument("--play_diag", action="store_true", default=PLAY_DIAG,
                    help="累積 play 時的目標對齊診斷（heading error / velocity-to-goal）")
parser.add_argument("--aux_debug", action="store_true", default=AUX_DEBUG,
                    help="印出 RNN aux 7D 預測 vs simulator ground truth")
parser.add_argument("--aux_debug_interval", type=int, default=AUX_DEBUG_INTERVAL,
                    help="每 N 步印一次 aux debug")
parser.add_argument("--no_domain_randomization", action="store_true", default=False,
                    help="關閉 domain randomization")

# --- BEV 俯視圖 ---
parser.add_argument("--bev_vis", action="store_true", default=BEV_VIS,
                    help="開啟即時 BEV 俯視圖視窗（72-bin LiDAR + 目標 + 障礙物）")
parser.add_argument("--bev_update_interval", type=int, default=BEV_UPDATE_INTERVAL,
                    help="每 N 步更新一次 BEV 圖")
parser.add_argument("--bev_max_range", type=float, default=BEV_MAX_RANGE,
                    help="BEV 最大顯示範圍 (m)")
parser.add_argument("--bev_frame", type=str, default=BEV_FRAME, choices=["body", "world"],
                    help="BEV 座標系：body=車體座標（前方為上）/ world=世界座標")
parser.add_argument("--bev_trail_length", type=int, default=500,
                    help="BEV 歷史軌跡最大點數（0=關閉）")

# --- RSGS-Lite (Recovery-only Safer-Gap Goal Selector) ---
parser.add_argument("--use_rsgs", action="store_true", default=False,
                    help="啟用 RSGS-Lite 卡住恢復模組（play-only，預設 OFF）")
parser.add_argument("--rsgs_stuck_window", type=int, default=15,
                    help="RSGS stuck 偵測視窗步數（15 = 3s @5Hz）")
parser.add_argument("--rsgs_stuck_threshold", type=float, default=0.3,
                    help="RSGS stuck 位移門檻 (m)：視窗內移動 < 此值判定卡住")
parser.add_argument("--rsgs_gap_min_width", type=int, default=3,
                    help="RSGS 最小 gap 寬度 (LiDAR bins)")
parser.add_argument("--rsgs_gap_clear_threshold", type=float, default=0.10,
                    help="RSGS LiDAR clear 門檻（normalized，0.10 ≈ 2.35m）")
parser.add_argument("--rsgs_recovery_distance", type=float, default=2.0,
                    help="RSGS 恢復目標距離 (m)")
parser.add_argument("--rsgs_max_recovery_steps", type=int, default=50,
                    help="RSGS 恢復最長步數（50 = 10s）")
parser.add_argument("--rsgs_exit_displacement", type=float, default=1.0,
                    help="RSGS 脫困成功位移門檻 (m)")
parser.add_argument("--rsgs_goal_bias", type=float, default=0.3,
                    help="RSGS gap 選擇時偏向 final goal 的權重 [0,1]")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.stage_parameter = STAGE_PARAMETER  # 只由檔案上方設定區控制，不做 CLI 參數

# 非 headless 模式需要啟用攝影機
if not getattr(args_cli, "headless", False):
    args_cli.enable_cameras = True

# 啟動 Isaac Lab 應用程式（必須在所有 isaaclab import 之前）
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# ============================================================================
# Post-AppLauncher imports（Isaac Sim 啟動後才能 import 的模組）
# ============================================================================
import gymnasium as gym
import torch
from torch.distributions import Categorical

from isaaclab.envs import ViewerCfg

import isaaclab_tasks  # noqa: F401 — 觸發 gymnasium 環境註冊

# 設定模組搜尋路徑，讓 play 腳本能 import 訓練用的模組
_skrl_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))         # play_eval/ 本身
sys.path.insert(0, str(_skrl_root))                    # skrl/ 根目錄（obstacle_agent package）
sys.path.insert(0, str(_skrl_root / "models"))         # skrl/models/（modular_rnn_models.py）
sys.path.insert(0, str(_skrl_root / "utils"))          # skrl/utils/（charge_env_overrides.py, wd_aux_targets.py）

from charge_env_overrides import apply_charge_env_overrides  # env_cfg 覆寫工具
from modular_rnn_models import LidarStateExtractor, PolicyHead, PreprocessRNN, RNNStateManager, ValueHead  # 模型元件
from wd_aux_targets import build_wd_preprocess_targets  # RNN aux 7D target 計算

# 139D 觀測中，policy 使用的 79 維索引：
#   ego(4) + goal(2) + LiDAR(72) + time(1) = 79D
#   跳過 obs/Top10×6D(60) 這段由 extractor 處理
POLICY_OBS_INDICES = list(range(0, 78)) + [138]


def _cli_flag_provided(flag_names: tuple[str, ...] | list[str]) -> bool:
    """Return True if a CLI flag was explicitly provided in the original argv."""
    for arg in _original_argv[1:]:
        for flag in flag_names:
            if arg == flag or arg.startswith(flag + "="):
                return True
    return False


def _scene_arg(cli_args, attr: str, default_value, flag_names: tuple[str, ...] | list[str]):
    """stage_parameter=True 時只接受明確 CLI 覆寫；False 時使用設定區/default。"""
    if getattr(cli_args, "stage_parameter", False):
        return getattr(cli_args, attr) if _cli_flag_provided(flag_names) else default_value
    return getattr(cli_args, attr) if getattr(cli_args, attr) is not None else default_value


# ============================================================================
# 場景參數覆寫函式 — 讓使用者調整所有場景元素
# ============================================================================
def configure_play_scene(env_cfg, stage_cfg: dict | None, cli_args) -> dict:
    """
    依 stage_parameter / CLI 參數覆寫場景配置。

    - stage_parameter=True：預設使用訓練時該 stage 的場景參數；只有明確 CLI flag 會覆寫。
    - stage_parameter=False：使用檔案最上方手動設定區的 default / CLI 覆寫（舊行為）。

    可覆寫的場景參數:
      --num_goals_override : 目標數量
      --num_static_obs     : 靜態障礙物數量
      --num_dynamic_obs    : 動態障礙物數量
      --num_walls          : 內部牆壁數量
      --wall_length        : 牆壁長度 (m)
      --goal_distance_min  : 最小目標距離 (m)
      --goal_distance_max  : 最大目標距離 (m)
      --obstacle_speed     : 障礙物速度倍率
      --obstacle_behavior  : 障礙物行為模式 (需要 BehaviorScheduler)

    回傳: 最終生效的場景參數 dict（用於 print 確認）
    """
    # 從 stage_cfg 讀取訓練時 stage 預設值；stage_parameter=False 時則用手動設定區/default。
    stage_defaults = {
        "num_goals": stage_cfg["num_goals"] if stage_cfg else 10,
        "num_static": stage_cfg["num_obstacles_static"] if stage_cfg else 0,
        "num_dynamic": stage_cfg["num_obstacles_dynamic"] if stage_cfg else 2,
        "walls_min": stage_cfg["min_walls"] if stage_cfg else 0,
        "walls_max": stage_cfg["max_walls"] if stage_cfg else 2,
        "wall_length": stage_cfg.get("target_wall_length", 5.0) if stage_cfg else 5.0,
        "goal_dist_min": stage_cfg["goal_distance"][0] if stage_cfg else 2.0,
        "goal_dist_max": stage_cfg["goal_distance"][1] if stage_cfg else 10.0,
        "episode_length_s": stage_cfg.get("episode_length_s", getattr(env_cfg, "episode_length_s", 60.0)) if stage_cfg else getattr(env_cfg, "episode_length_s", 60.0),
        "obstacle_speed": stage_cfg.get("obstacle_speed_rate", 0.8) if stage_cfg else 0.8,
    }
    manual_defaults = {
        "num_goals": NUM_GOALS,
        "num_static": NUM_STATIC_OBS,
        "num_dynamic": NUM_DYNAMIC_OBS,
        "walls_min": NUM_WALLS,
        "walls_max": NUM_WALLS,
        "wall_length": WALL_LENGTH,
        "goal_dist_min": GOAL_DIST_MIN,
        "goal_dist_max": GOAL_DIST_MAX,
        "episode_length_s": EPISODE_LENGTH_S if EPISODE_LENGTH_S is not None else getattr(env_cfg, "episode_length_s", 60.0),
        "obstacle_speed": OBSTACLE_SPEED,
    }
    defaults = stage_defaults if cli_args.stage_parameter else manual_defaults

    # CLI 覆寫：stage_parameter=True 時只接受「明確寫在 CLI」的覆寫，避免上方手動 default 蓋掉 stage。
    final = {}
    final["num_goals"] = _scene_arg(cli_args, "num_goals_override", defaults["num_goals"], ("--num_goals_override",))
    final["num_static"] = _scene_arg(cli_args, "num_static_obs", defaults["num_static"], ("--num_static_obs",))
    final["num_dynamic"] = _scene_arg(cli_args, "num_dynamic_obs", defaults["num_dynamic"], ("--num_dynamic_obs",))
    _num_walls_override = _scene_arg(cli_args, "num_walls", None, ("--num_walls",))
    final["walls_min"] = _num_walls_override if _num_walls_override is not None else defaults["walls_min"]
    final["walls_max"] = _num_walls_override if _num_walls_override is not None else defaults["walls_max"]
    final["wall_length"] = _scene_arg(cli_args, "wall_length", defaults["wall_length"], ("--wall_length",))
    final["goal_dist_min"] = _scene_arg(cli_args, "goal_distance_min", defaults["goal_dist_min"], ("--goal_distance_min",))
    final["goal_dist_max"] = _scene_arg(cli_args, "goal_distance_max", defaults["goal_dist_max"], ("--goal_distance_max",))
    final["episode_length_s"] = _scene_arg(cli_args, "episode_length_s", defaults["episode_length_s"], ("--episode_length_s",))
    final["obstacle_speed"] = _scene_arg(cli_args, "obstacle_speed", defaults["obstacle_speed"], ("--obstacle_speed",))
    final["obstacle_behavior"] = _scene_arg(cli_args, "obstacle_behavior", None, ("--obstacle_behavior",))  # None = 不變更

    # --- 套用到 env_cfg ---

    # 目標數量
    env_cfg.commands.goal_command.num_goals = final["num_goals"]

    # 障礙物數量與比例（套用到 randomize_obstacles 事件）
    total_obs = final["num_static"] + final["num_dynamic"]
    # Play 模式 (num_envs=1): 永遠從用戶設定的數量計算 ratio，
    # 避免 stage 的 empty_ratio 導致唯一的 env 被擲骰為 empty。
    # 訓練時的 ratio 是給 512 envs 做混合分佈用的，1 env 不適用。
    is_single_env = getattr(env_cfg.scene, "num_envs", 1) <= 1
    if cli_args.stage_parameter and stage_cfg is not None and not is_single_env:
        # 多環境訓練：嚴格對齊訓練 stage 的 ratios
        empty_ratio = stage_cfg["empty_ratio"]
        static_ratio = stage_cfg["static_ratio"]
        dynamic_ratio = stage_cfg["dynamic_ratio"]
    else:
        # Play / 手動模式：從 counts 確定性推導 ratios
        # mixed_ratio 不需要顯式傳入，它是 1 - (empty + static + dynamic) 的餘量
        if total_obs > 0:
            empty_ratio = 0.0
            if final["num_dynamic"] > 0 and final["num_static"] > 0:
                # 兩者都有 → mixed (remainder=1.0: 同場景靜態+動態)
                static_ratio = 0.0
                dynamic_ratio = 0.0
            elif final["num_dynamic"] > 0:
                static_ratio = 0.0
                dynamic_ratio = 1.0
            else:
                static_ratio = 1.0
                dynamic_ratio = 0.0
        else:
            empty_ratio = 1.0
            static_ratio = dynamic_ratio = 0.0

    for evt_attr in ["randomize_obstacles", "randomize_obstacles_startup"]:
        evt_term = getattr(env_cfg.events, evt_attr, None)
        if evt_term is not None:
            evt_term.params.update({
                "empty_ratio": empty_ratio,
                "static_ratio": static_ratio,
                "dynamic_ratio": dynamic_ratio,
                "num_obstacles_static": final["num_static"],
                "num_obstacles_dynamic": final["num_dynamic"],
            })

    # 牆壁數量
    wall_evt = getattr(env_cfg.events, "randomize_wall_positions", None)
    if wall_evt is not None:
        wall_evt.params["min_walls"] = final["walls_min"]
        wall_evt.params["max_walls"] = final["walls_max"]
        if final["wall_length"] is not None:
            wall_evt.params["target_wall_length"] = final["wall_length"]

    # 目標距離範圍
    env_cfg.commands.goal_command.ranges.distance = (final["goal_dist_min"], final["goal_dist_max"])
    env_cfg.commands.goal_command.num_obstacles = total_obs
    if final["episode_length_s"] is not None:
        env_cfg.episode_length_s = float(final["episode_length_s"])

    # 障礙物生成邊界覆寫（讓障礙物在機器人附近生成）
    obs_boundary = getattr(cli_args, "obstacle_boundary", None)
    if obs_boundary is not None:
        for evt_attr in ["randomize_obstacles", "randomize_obstacles_startup"]:
            evt_term = getattr(env_cfg.events, evt_attr, None)
            if evt_term is not None:
                evt_term.params["boundary"] = obs_boundary
        # 同步縮小機器人生成範圍，確保機器人和障礙物在同一區域
        reset_evt = getattr(env_cfg.events, "reset_base", None)
        if reset_evt is not None:
            spawn_limit = max(1.0, obs_boundary - 1.5)
            reset_evt.params["pose_range"]["x"] = (-spawn_limit, spawn_limit)
            reset_evt.params["pose_range"]["y"] = (-spawn_limit, spawn_limit)
        print(f"[PLAY] obstacle_boundary={obs_boundary:.1f}m (robot spawn ±{spawn_limit:.1f}m)")

    # 印出最終場景配置
    changed = []
    scene_flag_map = {
        "num_goals": ("num_goals_override", ("--num_goals_override",)),
        "num_static": ("num_static_obs", ("--num_static_obs",)),
        "num_dynamic": ("num_dynamic_obs", ("--num_dynamic_obs",)),
        "walls_min": ("num_walls", ("--num_walls",)),
        "walls_max": ("num_walls", ("--num_walls",)),
        "wall_length": ("wall_length", ("--wall_length",)),
        "goal_dist_min": ("goal_distance_min", ("--goal_distance_min",)),
        "goal_dist_max": ("goal_distance_max", ("--goal_distance_max",)),
        "episode_length_s": ("episode_length_s", ("--episode_length_s",)),
        "obstacle_speed": ("obstacle_speed", ("--obstacle_speed",)),
        "obstacle_behavior": ("obstacle_behavior", ("--obstacle_behavior",)),
    }
    for key, (_attr, _flags) in scene_flag_map.items():
        if cli_args.stage_parameter:
            if _cli_flag_provided(_flags):
                changed.append(key)
        else:
            cli_val = getattr(cli_args, _attr, None)
            if cli_val is not None:
                changed.append(key)

    print(f"[PLAY] 場景配置 (stage_parameter={cli_args.stage_parameter}):")
    print(f"  目標數={final['num_goals']}  靜態障礙={final['num_static']}  動態障礙={final['num_dynamic']}")
    print(f"  牆壁={final['walls_min']}~{final['walls_max']}  牆長={final['wall_length']:.1f}m")
    print(f"  目標距離={final['goal_dist_min']:.1f}~{final['goal_dist_max']:.1f}m  "
          f"episode={final['episode_length_s']:.1f}s  障礙速度={final['obstacle_speed']:.2f}")
    if changed:
        print(f"  CLI 覆寫: {', '.join(changed)}")
    if final["obstacle_behavior"]:
        print(f"  障礙物行為: {final['obstacle_behavior']}（需要 BehaviorScheduler）")
    if cli_args.obs_near_goal_count > 0:
        print(f"  目標附近障礙: {cli_args.obs_near_goal_count} 個在 goal ≤{cli_args.obs_near_goal_radius:.1f}m 內")

    return final


# ============================================================================
# USD 場景切換
# ============================================================================

# 預設場景別名 → Nucleus 路徑
_USD_SCENE_ALIASES: dict[str, str] = {
    "warehouse": "Environments/Simple_Warehouse/warehouse.usd",
    "warehouse_full": "Environments/Simple_Warehouse/full_warehouse.usd",
    "hospital": "Environments/Hospital/hospital.usd",
    "simple_room": "Environments/Simple_Room/simple_room.usd",
    "grid": "Environments/Grid/default_environment.usd",
    "grid_black": "Environments/Grid/gridroom_black.usd",
    "rough_plane": "Environments/Terrains/rough_plane.usd",
}

# 本機自訂 USD 場景（不透過 Nucleus，直接使用絕對路徑）
_USD_LOCAL_ALIASES: dict[str, str] = {
    "3floor": "/home/aa/usd/charge/3floor_ver_1.usd",
    "3floor_v1": "/home/aa/usd/charge/3floor_ver_1.usd",
    "3floor_v2": "/home/aa/usd/charge/3floor_ver_2 .usd",
}


def apply_usd_scene(env_cfg, usd_scene: str) -> str:
    """將 env_cfg 的 terrain 替換為 USD 場景，停用程式化牆壁，擴展 LiDAR 偵測。

    Args:
        env_cfg: 環境配置物件
        usd_scene: 場景名稱 (warehouse/hospital/grid) 或自訂 USD 路徑

    Returns:
        最終使用的 USD 路徑字串
    """
    from isaaclab.terrains import TerrainImporterCfg
    import isaaclab.sim as sim_utils
    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

    # 解析場景路徑：本機 alias → Nucleus alias → 自訂絕對路徑
    if usd_scene in _USD_LOCAL_ALIASES:
        usd_path = _USD_LOCAL_ALIASES[usd_scene]
    elif usd_scene in _USD_SCENE_ALIASES:
        usd_path = f"{ISAAC_NUCLEUS_DIR}/{_USD_SCENE_ALIASES[usd_scene]}"
    else:
        usd_path = usd_scene  # 使用者提供的絕對路徑

    # 1. 替換 terrain: plane → usd
    env_cfg.scene.terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="usd",
        usd_path=usd_path,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )

    # 2. 停用程式化牆壁（外牆 + 內牆）
    wall_attrs = []
    for attr in list(vars(env_cfg.scene)):
        if attr.startswith("wall_"):
            delattr(env_cfg.scene, attr)
            wall_attrs.append(attr)

    # 停用牆壁隨機化事件
    wall_evt = getattr(env_cfg.events, "randomize_wall_positions", None)
    if wall_evt is not None:
        wall_evt.params["min_walls"] = 0
        wall_evt.params["max_walls"] = 0

    # 3. 擴展 LiDAR raycast targets — 加入 USD 場景內所有 mesh
    #    保留原有的 Obstacle_* target（程式化障礙物仍可用）
    if hasattr(env_cfg.scene, "lidar"):
        from isaaclab.sensors.ray_caster import MultiMeshRayCasterCfg
        # 新的 mesh_prim_paths：
        #   - /World/ground/.* → 涵蓋 USD 場景內的所有 geometry（牆壁/貨架/地面等）
        #   - {ENV_REGEX_NS}/Obstacle_.* → 保留程式化障礙物偵測
        env_cfg.scene.lidar.mesh_prim_paths = [
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="/World/ground",
                track_mesh_transforms=False,
            ),
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="/World/ground/.*",
                track_mesh_transforms=False,
            ),
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="{ENV_REGEX_NS}/Obstacle_.*",
                track_mesh_transforms=True,
            ),
        ]

    print(f"[PLAY] USD 場景: {usd_scene} → {usd_path}")
    if wall_attrs:
        print(f"[PLAY]   停用程式化牆壁: {len(wall_attrs)} 個 ({', '.join(wall_attrs[:6])}{'...' if len(wall_attrs) > 6 else ''})")
    print(f"[PLAY]   LiDAR raycast 擴展至 /World/ground/.* (涵蓋 USD 場景所有 mesh)")

    return usd_path


def sample_action(logits: torch.Tensor, deterministic: bool) -> torch.Tensor:
    """從 policy logits 取樣動作。MultiDiscrete([19, 19]) = 線性加速 × 角速度。

    logits 前 19 維 = 線性加速度（idx 9 = 零加速），後 19 維 = 角速度。
    deterministic=True 時取 argmax（確定性動作），否則用 Categorical 隨機抽樣。
    """
    logits_a, logits_w = logits[:, :19], logits[:, 19:]  # 拆分為線性/角度兩組 logits
    if deterministic:
        # 確定性模式：選機率最高的動作
        act_a = logits_a.argmax(dim=-1)
        act_w = logits_w.argmax(dim=-1)
    else:
        # 探索模式：依 softmax 機率分佈隨機抽樣
        act_a = Categorical(logits=logits_a).sample()
        act_w = Categorical(logits=logits_w).sample()
    return torch.stack([act_a, act_w], dim=-1)  # [N, 2] 組合為雙動作


def find_latest_rnn_checkpoint() -> str | None:
    """搜尋 logs/rnn_car/ 下最新的 checkpoint 檔案，依修改時間排序。"""
    candidates = glob.glob("logs/rnn_car/*/checkpoint_*.pt")
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)  # 最新在前
    return os.path.abspath(candidates[0])


def resolve_env_cfg(task: str):
    """依 task ID 建立對應的環境配置物件。

    支援的 task:
      - VLP16-Baseline / VLP16: 基礎版本（無 curriculum）
      - VLP16-Curriculum: 舊版 8 階段 curriculum
      - VLP16-Curriculum-NavRL: NavRL dense reward + curriculum（消融實驗用）
      - VLP16-Curriculum-WD: WD sparse reward + curriculum（RNN 訓練用）
    """
    if task == "Isaac-Navigation-Charge-VLP16-Baseline":
        from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_skrl.cfg.charge_env_cfg_vlp16 import (
            ChargeNavigationEnvCfgVLP16Baseline,
        )
        return ChargeNavigationEnvCfgVLP16Baseline()
    if task == "Isaac-Navigation-Charge-VLP16":
        from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_skrl.cfg.charge_env_cfg_vlp16 import (
            ChargeNavigationEnvCfgVLP16,
        )
        return ChargeNavigationEnvCfgVLP16()
    if task == "Isaac-Navigation-Charge-VLP16-Curriculum":
        from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_skrl.cfg.charge_env_cfg_vlp16_curriculum import (
            ChargeNavigationEnvCfgVLP16Curriculum,
        )
        return ChargeNavigationEnvCfgVLP16Curriculum()
    if task == "Isaac-Navigation-Charge-VLP16-Curriculum-NavRL":
        from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_skrl.cfg.charge_env_cfg_vlp16_curriculum import (
            ChargeNavigationEnvCfgVLP16CurriculumNavRL,
        )
        return ChargeNavigationEnvCfgVLP16CurriculumNavRL()
    if task == "Isaac-Navigation-Charge-VLP16-Curriculum-WD":
        from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_skrl.cfg.charge_env_cfg_wd_sparse import (
            ChargeNavigationEnvCfgVLP16CurriculumWD,
        )
        return ChargeNavigationEnvCfgVLP16CurriculumWD()
    if task == "Isaac-Navigation-Charge-VLP16-Curriculum-WD-TCorridor":
        from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_skrl.cfg.charge_env_cfg_wd_sparse import (
            ChargeNavigationEnvCfgVLP16CurriculumWD_TCorridor,
        )
        return ChargeNavigationEnvCfgVLP16CurriculumWD_TCorridor()
    if task == "Isaac-Navigation-Charge-VLP16-Curriculum-NavRL-TCorridor":
        from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_skrl.cfg.charge_env_cfg_vlp16_curriculum import (
            ChargeNavigationEnvCfgVLP16CurriculumNavRL_TCorridor,
        )
        return ChargeNavigationEnvCfgVLP16CurriculumNavRL_TCorridor()
    if task == "Isaac-Navigation-Charge-VLP16-Curriculum-NavRL-Play":
        from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_skrl.cfg.charge_env_cfg_vlp16_curriculum import (
            ChargeNavigationEnvCfgVLP16CurriculumNavRL_PLAY,
        )
        return ChargeNavigationEnvCfgVLP16CurriculumNavRL_PLAY()
    if task == "Isaac-Navigation-Charge-VLP16-Curriculum-NavRL-Play-TCorridor":
        from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_skrl.cfg.charge_env_cfg_vlp16_curriculum import (
            ChargeNavigationEnvCfgVLP16CurriculumNavRL_PLAY_TCorridor,
        )
        return ChargeNavigationEnvCfgVLP16CurriculumNavRL_PLAY_TCorridor()
    raise ValueError(f"不支援的 task: {task}")


def apply_stage_to_env_cfg(env_cfg, curriculum_version: str | None, stage: int, apply_scene_params: bool = True):
    """固定 curriculum stage，並可選擇是否把該 stage 的場景參數套用到 env_cfg。

    apply_scene_params=True：完全帶入訓練時該 stage 的場景參數。
    apply_scene_params=False：只讀取 stage_cfg / 關閉自動晉級；實際場景由手動設定區或 CLI 決定。
    並將 env_cfg.curriculum 設為 None（關閉自動晉級）。

    回傳: (resolved_version, stage_cfg_dict) 或 (None, None) 若無 curriculum。
    """
    # 嘗試從 env_cfg 取得 curriculum 設定
    cur = getattr(env_cfg, "curriculum", None)
    term = getattr(cur, "goal_obstacle_curriculum", None) if cur is not None else None
    if term is None:
        return None, None  # 此 task 沒有 curriculum

    # 覆寫 curriculum 版本（若 CLI 有指定）
    if curriculum_version is not None:
        term.params["curriculum_version"] = curriculum_version
    resolved_version = term.params.get("curriculum_version", "baseline_v1")

    # 載入該版本的所有 stage 定義
    from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_skrl.curriculum.goal_obstacle_curriculum import (
        _load_stages,
    )

    stages, max_stage, _ = _load_stages(resolved_version)
    stage = max(1, min(stage, max_stage))  # 限制在合法範圍
    cfg = stages[stage]

    if not apply_scene_params:
        env_cfg.curriculum = None
        return resolved_version, cfg

    # --- 套用障礙物參數 ---
    for evt_attr in ["randomize_obstacles", "randomize_obstacles_startup"]:
        evt_term = getattr(env_cfg.events, evt_attr, None)
        if evt_term is not None:
            evt_term.params.update({
                "empty_ratio": cfg["empty_ratio"],
                "static_ratio": cfg["static_ratio"],
                "dynamic_ratio": cfg["dynamic_ratio"],
                "num_obstacles_static": cfg["num_obstacles_static"],
                "num_obstacles_dynamic": cfg["num_obstacles_dynamic"],
            })

    # --- 套用牆壁參數 ---
    wall_evt = getattr(env_cfg.events, "randomize_wall_positions", None)
    if wall_evt is not None:
        wall_evt.params["min_walls"] = cfg["min_walls"]
        wall_evt.params["max_walls"] = cfg["max_walls"]
        if cfg.get("target_wall_length") is not None:
            wall_evt.params["target_wall_length"] = cfg["target_wall_length"]

    # --- 套用目標/回合參數 ---
    env_cfg.commands.goal_command.num_goals = cfg["num_goals"]
    env_cfg.commands.goal_command.num_obstacles = cfg["num_obstacles_static"] + cfg["num_obstacles_dynamic"]
    env_cfg.commands.goal_command.ranges.distance = cfg["goal_distance"]
    env_cfg.episode_length_s = cfg["episode_length_s"]

    # Play 模式關閉自動晉級
    env_cfg.curriculum = None
    return resolved_version, cfg


def apply_play_goal_override(env_cfg, num_goals_override: int | None):
    """在 stage config 套用後，再覆寫目標數量（純 play 診斷用）。"""
    if num_goals_override is None:
        return
    n = max(1, int(num_goals_override))
    env_cfg.commands.goal_command.num_goals = n
    print(f"[PLAY] num_goals_override={n} (play 診斷用覆寫)", flush=True)


def configure_camera(env_cfg, camera: str):
    """設定攝影機視角。

    top:    正上方俯視（z=30m），適合觀察全場導航路徑
    follow: 從後方跟隨機器人（asset_root 追蹤模式）
    side:   側面遠距觀察
    """
    if camera == "top":
        env_cfg.viewer = ViewerCfg(
            eye=(0.0, 0.0, 30.0), lookat=(0.0, 0.0, 0.0),
            origin_type="env", env_index=0, resolution=(1920, 1080),
        )
    elif camera == "follow":
        env_cfg.viewer = ViewerCfg(
            eye=(-3.0, 0.0, 3.0), lookat=(2.0, 0.0, 0.0),
            origin_type="asset_root", env_index=0, asset_name="robot",
            resolution=(1920, 1080),
        )
    else:  # side
        env_cfg.viewer = ViewerCfg(
            eye=(15.0, -15.0, 15.0), lookat=(0.0, 0.0, 0.0),
            origin_type="env", env_index=0, resolution=(1920, 1080),
        )


def run_lidar_sanity_test(env, raw_env):
    """LiDAR 射線原點 Bug 驗證 — 確認 z_filter 在 per-ray origin 下正常工作。

    驗證 4 個面向:
      1. 射線原點 z ≈ 1.6m（VLP16 感測器高度）
      2. 原始 hit 分析（finite / ground-like / max-range 比例）
      3. z_filter 效果（地板射線被正確排除）
      4. 72-bin sweep 不受 5.6m 地板回波主導
    """
    import math
    from isaaclab.managers import SceneEntityCfg
    from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_skrl.mdp.observations.obs_functions import (
        wd_like_sweep_72,
    )

    print("\n" + "=" * 70)
    print("  LiDAR Ray Origin Bug 驗證測試")
    print("=" * 70)

    sensor = raw_env.scene.sensors["lidar"]
    pos_w = sensor.data.pos_w            # [N, 3] — base_link (z≈0)
    hit_points = sensor.data.ray_hits_w  # [N, R, 3]
    N, R, _ = hit_points.shape

    # ==================================================================
    # 測試 1: Ray Origin — 確認 _ray_starts_w 存在且 z ≈ 1.6
    # ==================================================================
    print(f"\n[測試 1] Ray Origin 驗證 (env 0):")
    print(f"  pos_w (base_link) = [{pos_w[0, 0].item():.3f}, "
          f"{pos_w[0, 1].item():.3f}, {pos_w[0, 2].item():.3f}]")

    has_ray_starts = hasattr(sensor, "_ray_starts_w") and sensor._ray_starts_w is not None
    if has_ray_starts:
        ray_starts_w = sensor._ray_starts_w  # [N, R, 3]
        ray_z = ray_starts_w[0, 0, 2].item()
        print(f"  _ray_starts_w shape = {list(ray_starts_w.shape)}")
        print(f"  _ray_starts_w[0,0] = [{ray_starts_w[0, 0, 0].item():.3f}, "
              f"{ray_starts_w[0, 0, 1].item():.3f}, {ray_z:.3f}]")
        t1_pass = abs(ray_z - 1.6) < 0.3
        print(f"  判定: {'PASS' if t1_pass else 'FAIL'} — "
              f"ray_origin z = {ray_z:.3f}m {'≈' if t1_pass else '≠'} 1.6m")
    else:
        print(f"  WARNING: _ray_starts_w not available! Estimating z = base + 1.6")
        ray_z = pos_w[0, 2].item() + 1.6
        ray_starts_w = pos_w.unsqueeze(1).expand_as(hit_points).clone()
        ray_starts_w[..., 2] += 1.6
        t1_pass = False
        print(f"  判定: FAIL — _ray_starts_w 不存在")

    # ==================================================================
    # 測試 2: Raw Hit 分析 — per-ray origin
    # ==================================================================
    print(f"\n[測試 2] Raw Hit 分析 — per-ray origin (env 0, {R} rays):")

    hits_0 = hit_points[0]           # [R, 3]
    origins_0 = ray_starts_w[0]      # [R, 3]
    rel_hits = hits_0 - origins_0    # [R, 3]

    dist_2d = torch.linalg.norm(rel_hits[:, :2], dim=-1)  # [R]
    rel_z = rel_hits[:, 2]           # [R]

    finite_mask = torch.isfinite(dist_2d) & torch.isfinite(rel_z)
    near_max = dist_2d >= 19.5
    valid = finite_mask & ~near_max

    n_finite = finite_mask.sum().item()
    n_ground_like = (finite_mask & (rel_z < -1.0)).sum().item()
    n_zf_pass = (finite_mask & (rel_z.abs() <= 0.5)).sum().item()
    n_zf_reject_ground = (finite_mask & (rel_z < -0.5)).sum().item()
    n_zf_pass_mid = (finite_mask & (rel_z.abs() <= 0.5)).sum().item()

    print(f"  總射線: {R}")
    print(f"  finite: {n_finite}")
    print(f"  valid (finite & dist < 19.5m): {valid.sum().item()}")
    print(f"  near max-range (≥ 19.5m): {(finite_mask & near_max).sum().item()}")
    print(f"  ---")
    print(f"  ground-like (rel_z < -1.0): {n_ground_like}")
    print(f"  z_filter pass (|rel_z| ≤ 0.5): {n_zf_pass}")
    print(f"  z_filter reject below (rel_z < -0.5): {n_zf_reject_ground}")
    print(f"  z_filter pass mid-height (|rel_z| ≤ 0.5): {n_zf_pass_mid}")

    if valid.sum() > 0:
        valid_dist = dist_2d[valid]
        valid_rel_z = rel_z[valid]
        hit_z_valid = hits_0[valid, 2]
        print(f"\n  Valid hit 統計:")
        print(f"    dist_2d: min={valid_dist.min().item():.3f}  "
              f"mean={valid_dist.mean().item():.3f}  max={valid_dist.max().item():.3f}")
        print(f"    rel_z:   min={valid_rel_z.min().item():.3f}  "
              f"mean={valid_rel_z.mean().item():.3f}  max={valid_rel_z.max().item():.3f}")
        print(f"    hit_z (world): min={hit_z_valid.min().item():.3f}  "
              f"mean={hit_z_valid.mean().item():.3f}  max={hit_z_valid.max().item():.3f}")

    # ==================================================================
    # 測試 3: Z-filter 效果驗證
    # ==================================================================
    print(f"\n[測試 3] Z-filter 效果 (z_filter=0.5, 基於 per-ray origin):")

    zf_pass_mask = finite_mask & (rel_z.abs() <= 0.5)
    zf_reject_mask = finite_mask & (rel_z.abs() > 0.5)
    zf_reject_below = finite_mask & (rel_z < -0.5)
    zf_reject_above = finite_mask & (rel_z > 0.5)

    print(f"  z_filter PASS:   {zf_pass_mask.sum().item()} rays")
    print(f"  z_filter REJECT: {zf_reject_mask.sum().item()} rays")
    print(f"    below sensor (rel_z < -0.5): {zf_reject_below.sum().item()}")
    print(f"    above sensor (rel_z > +0.5): {zf_reject_above.sum().item()}")

    if zf_pass_mask.sum() > 0:
        pass_dist = dist_2d[zf_pass_mask]
        pass_rel_z = rel_z[zf_pass_mask]
        print(f"  PASS rays dist_2d: min={pass_dist.min().item():.3f}  "
              f"mean={pass_dist.mean().item():.3f}  max={pass_dist.max().item():.3f}")
        print(f"  PASS rays rel_z:   min={pass_rel_z.min().item():.3f}  "
              f"mean={pass_rel_z.mean().item():.3f}  max={pass_rel_z.max().item():.3f}")

    # Ground rejection verdict
    t3_pass = n_ground_like > 0 and n_zf_reject_ground >= n_ground_like
    if n_ground_like == 0:
        print(f"\n  判定: INFO — 無 ground-like rays (rel_z < -1.0), z_filter 未觸發 ground rejection")
        t3_pass = True  # no ground hits to reject = OK
    else:
        print(f"\n  判定: {'PASS' if t3_pass else 'FAIL'} — "
              f"ground-like={n_ground_like}, z_filter rejected below={n_zf_reject_ground}")

    # ==================================================================
    # 測試 4: wd_like_sweep_72 + 5.6m 地板回波檢查
    # ==================================================================
    print(f"\n[測試 4] wd_like_sweep_72 輸出 (z_filter=0.5, r_min=0):")
    sensor_cfg = SceneEntityCfg("lidar")
    sensor_cfg.resolve(raw_env.scene)
    sweep = wd_like_sweep_72(
        raw_env, sensor_cfg,
        num_bins=72, r_max=20.0, r_robot=0.3,
        r_min=0.0, z_filter=0.5,
    )
    sweep_0 = sweep[0]
    real_dist = sweep_0 * 20.0

    # 5.6m ground echo parameters
    ground_echo_dist = 1.6 / math.tan(math.radians(15.0)) - 0.3
    echo_lo = ground_echo_dist - 0.3
    echo_hi = ground_echo_dist + 0.3

    n_close = (real_dist < 2.0).sum().item()
    n_mid = ((real_dist >= 2.0) & (real_dist < 5.0)).sum().item()
    n_echo = ((real_dist >= echo_lo) & (real_dist <= echo_hi)).sum().item()
    n_max_range = (real_dist >= 19.5).sum().item()

    min_bin = real_dist.argmin().item()
    min_angle = -180.0 + min_bin * 5.0

    print(f"  72-bin 摘要:")
    print(f"    min  = {real_dist.min().item():.2f} m  (bin #{min_bin}, {min_angle:+.0f}°)")
    print(f"    mean = {real_dist.mean().item():.2f} m")
    print(f"    max  = {real_dist.max().item():.2f} m")
    print(f"    bins < 2m:                  {n_close:3d} / 72")
    print(f"    bins 2~5m:                  {n_mid:3d} / 72")
    print(f"    bins near 5.6m [{echo_lo:.1f},{echo_hi:.1f}]: {n_echo:3d} / 72")
    print(f"    bins >= 19.5m (max range):  {n_max_range:3d} / 72")

    # Top 10 closest bins
    sorted_indices = real_dist.argsort()
    print(f"\n  最近 10 個 bins:")
    print(f"    {'Bin':>4s}  {'角度':>7s}  {'距離m':>7s}")
    for i in range(min(10, 72)):
        idx = sorted_indices[i].item()
        angle = -180.0 + idx * 5.0
        d = real_dist[idx].item()
        print(f"    {idx:4d}  {angle:+7.1f}°  {d:7.2f}")

    # 5.6m ground echo verdict
    print(f"\n  地板回波檢查 (預期回波距離 = {ground_echo_dist:.2f} m):")
    if n_echo >= 30:
        t4_echo = False
        print(f"    FAIL — {n_echo}/72 bins 集中在 ~5.6m，地板回波仍主導 sweep")
    elif n_echo >= 10:
        t4_echo = False
        print(f"    SUSPICIOUS — {n_echo}/72 bins 落在 ~5.6m 附近")
    else:
        t4_echo = True
        print(f"    PASS — 僅 {n_echo}/72 bins 在 ~5.6m，地板回波已被 z_filter 排除")

    # 72-bin scan table
    print(f"\n  72-bin 掃描表 (每 3 bin 或 < 3m 全印):")
    print(f"  {'Bin':>4s} {'角度':>7s} {'距離m':>7s} {'圖示'}")
    for i in range(72):
        angle = -180.0 + i * 5.0
        d = real_dist[i].item()
        bar_len = min(int(d / 0.5), 40)
        bar = "█" * bar_len if d < 19.5 else "·" * 5
        if i % 3 == 0 or d < 3.0:
            print(f"  {i:4d} {angle:+7.1f}° {d:7.2f} {bar}")

    # r_min comparison
    sweep_rmin = wd_like_sweep_72(
        raw_env, sensor_cfg,
        num_bins=72, r_max=20.0, r_robot=0.3,
        r_min=0.1, z_filter=0.5,
    )
    real_dist_rmin = sweep_rmin[0] * 20.0
    diff = (real_dist_rmin - real_dist).abs()
    changed = diff > 0.01
    print(f"\n  r_min=0.5 效果:")
    print(f"    變更 bin 數: {changed.sum().item()} / 72")
    if changed.sum() > 0:
        for i in range(72):
            if changed[i]:
                angle = -180.0 + i * 5.0
                print(f"      bin {i} ({angle:+.0f}°): "
                      f"{real_dist[i].item():.2f}m → {real_dist_rmin[i].item():.2f}m")

    # ==================================================================
    # 綜合判定
    # ==================================================================
    print(f"\n{'=' * 70}")
    print(f"  綜合判定")
    print(f"{'=' * 70}")
    verdicts = [
        ("ray_origin z ≈ 1.6m", t1_pass),
        ("ground rays 被 z_filter reject", t3_pass),
        ("72-bin 不集中在 ~5.6m 地板回波", t4_echo),
    ]
    all_pass = True
    for label, passed in verdicts:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {label}")

    has_near_object = real_dist.min().item() < 5.0
    if has_near_object:
        print(f"  [INFO] 偵測到近距離物體: min = {real_dist.min().item():.2f}m")
    else:
        print(f"  [INFO] 空曠場景 — 無近距離物體 (min = {real_dist.min().item():.2f}m)")

    overall = "PASS" if all_pass else "FAIL"
    print(f"\n  === 總結: {overall} ===")
    if all_pass:
        print(f"  Ray Origin Bug 修復驗證通過。地板回波已被 z_filter 正確排除。")
    else:
        print(f"  仍有問題，請檢查上方 FAIL 項目。")

    print("=" * 70 + "\n")


def print_lidar_diagnostic(raw_env, step: int):
    """每步 LiDAR 可觀察性輕量診斷 — 用 per-ray origin 計算真實距離。

    印出: 射線高度 / 有效射線數 / 地板射線數 / 72-bin 最近距離 / 地板回波數
    """
    import math as _math
    from isaaclab.managers import SceneEntityCfg
    from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_skrl.mdp.observations.obs_functions import (
        wd_like_sweep_72,
    )
    sensor = raw_env.scene.sensors["lidar"]
    hit_points = sensor.data.ray_hits_w  # [N, R, 3] — 世界座標下的射線撞擊點

    # 取得 per-ray 射線原點（含 OffsetCfg z=1.6 的偏移）
    if hasattr(sensor, "_ray_starts_w") and sensor._ray_starts_w is not None:
        ray_starts = sensor._ray_starts_w  # [N, R, 3]
        ray_z = ray_starts[0, 0, 2].item()
    else:
        # fallback: 用 base_link 位置 + 估計偏移
        ray_starts = sensor.data.pos_w.unsqueeze(1).expand_as(hit_points)
        ray_z = sensor.data.pos_w[0, 2].item()

    # 計算相對撞擊向量
    rel_hits = hit_points[0] - ray_starts[0]  # [R, 3]
    dist_2d = torch.linalg.norm(rel_hits[:, :2], dim=-1)  # 水平距離
    rel_z = rel_hits[:, 2]  # 垂直偏移（負 = 打到地板）
    finite = torch.isfinite(dist_2d) & torch.isfinite(rel_z)
    valid = finite & (dist_2d < 19.5)  # 排除 max-range 無效射線
    n_ground = (finite & (rel_z < -1.0)).sum().item()  # 地板射線數
    n_zf_pass = (finite & (rel_z.abs() <= 0.5)).sum().item()  # 通過 z_filter 的射線數

    # 重新計算 72-bin sweep（與訓練相同的處理流程）
    sensor_cfg = SceneEntityCfg("lidar")
    sensor_cfg.resolve(raw_env.scene)
    sweep = wd_like_sweep_72(raw_env, sensor_cfg, num_bins=72, r_max=20.0, r_robot=0.3, r_min=0.5, z_filter=0.5)
    real_dist = sweep[0] * 20.0  # 反正規化為公尺

    # 計算 5.6m 地板回波（VLP16 在 z=1.6m、15° 下射角時的幾何距離）
    ground_echo_dist = 1.6 / _math.tan(_math.radians(15.0)) - 0.3
    n_echo = ((real_dist >= ground_echo_dist - 0.3) & (real_dist <= ground_echo_dist + 0.3)).sum().item()

    # 最近障礙物 bin
    near_wall = real_dist.min().item()
    near_bin = real_dist.argmin().item()
    near_angle = -180.0 + near_bin * 5.0

    print(f"  [診斷 step={step}] ray_z={ray_z:.2f}m | "
          f"有效={valid.sum().item()}/{hit_points.shape[1]} "
          f"地板={n_ground} zf通過={n_zf_pass} | "
          f"72bin: 最近={near_wall:.2f}m @{near_bin}({near_angle:+.0f}°) "
          f"平均={real_dist.mean().item():.2f} "
          f"回波5.6={n_echo} <2m={int((real_dist < 2.0).sum())} "
          f"max={int((real_dist >= 19.5).sum())}")


def print_aux_debug(raw_env, step: int, obs_tensor: torch.Tensor, aux_pred: torch.Tensor, max_obstacles: int):
    """印出 RNN aux 7D 預測 vs simulator ground truth（env 0）。

    7D 目標格式:
      [near1_x, near1_y, near1_d, near2_x, near2_y, near2_d, timestep]
      near1/near2 = 最近兩個障礙物的車體座標相對位置 (m) 和距離
      timestep = 當前時間步（訓練時 loss weight = 0，不影響梯度）
    """
    from isaaclab.managers import SceneEntityCfg
    from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_skrl.mdp.observations.obs_functions import (
        wd_like_sweep_72,
    )

    # 從 simulator 重新計算真實 7D 目標
    target = build_wd_preprocess_targets(
        raw_env,
        max_obstacles=max_obstacles,
        device=aux_pred.device,
    )

    pred0 = aux_pred[0].detach()   # RNN 預測值
    tgt0 = target[0].detach()      # simulator 真實值
    err0 = pred0 - tgt0            # 預測誤差

    def _fmt_triplet(v):
        """格式化 [x, y, d] 三元組"""
        return f"x={v[0].item():+6.2f} y={v[1].item():+6.2f} d={v[2].item():5.2f}"

    # 同時重算 LiDAR 作為交叉驗證
    sensor_cfg = SceneEntityCfg("lidar")
    sensor_cfg.resolve(raw_env.scene)
    sweep = wd_like_sweep_72(
        raw_env, sensor_cfg,
        num_bins=72, r_max=20.0, r_robot=0.3, r_min=0.1, z_filter=0.5,
    )
    lidar_m = sweep[0] * 20.0  # 反正規化
    near_bin = int(lidar_m.argmin().item())
    near_angle = -180.0 + near_bin * 5.0

    # 從觀測中取出 LiDAR 部分（obs[6:78] = 72-bin 正規化距離）
    obs_lidar = obs_tensor[0, 6:78].detach()
    obs_near_bin = int(obs_lidar.argmin().item())
    obs_near_angle = -180.0 + obs_near_bin * 5.0
    obs_near_m = obs_lidar[obs_near_bin].item() * 20.0  # obs 中 LiDAR 已除以 20.0

    print(f"\n[AUX 除錯 step={step}] env0 RNN 預測 vs simulator 真實值")
    print("  近1 預測:  " + _fmt_triplet(pred0[0:3]))
    print("  近1 真實:  " + _fmt_triplet(tgt0[0:3]))
    print("  近1 誤差:  " + _fmt_triplet(err0[0:3]))
    print("  近2 預測:  " + _fmt_triplet(pred0[3:6]))
    print("  近2 真實:  " + _fmt_triplet(tgt0[3:6]))
    print("  近2 誤差:  " + _fmt_triplet(err0[3:6]))
    print(f"  時間步 預測={pred0[6].item():.2f} 真實={tgt0[6].item():.0f} "
          f"(訓練時 loss weight = 0)")
    print(f"  觀測中 LiDAR obs[6:78]: 最近={obs_near_m:.2f}m "
          f"@local_bin={obs_near_bin} global_bin={obs_near_bin + 6} ({obs_near_angle:+.0f}°)")
    print(f"  重算 LiDAR:             最近={lidar_m[near_bin].item():.2f}m "
          f"@bin={near_bin} ({near_angle:+.0f}°)\n")


class LiveBEVVisualizer:
    """即時 BEV 俯視圖視窗 — 顯示 RL 實際使用的 72-bin LiDAR sweep。

    功能:
      - 72-bin 極座標點雲（綠=安全 / 橙=警告 / 紅=危險）
      - 機器人位置/朝向箭頭
      - 目標位置（cyan X）+ 當前導航目標（黃色箭頭）
      - 終止判定目標（洋紅 / 綠色 P 標記 + 虛線圓）
      - 距離環（2/5/10/15/20m）
      - 上方獨立資料面板（Step/Action、目標終止、LiDAR、ORCA 指標），不遮擋 BEV 網格

    座標系:
      body = 車體座標（前方為上），world = 世界座標（北為上）
    """

    def __init__(self, raw_env, max_range: float = 20.0, frame: str = "body",
                 lidar_r_min: float = 0.1, trail_length: int = 500):
        import matplotlib
        import numpy as np
        from collections import deque
        from isaaclab.managers import SceneEntityCfg
        from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_skrl.mdp.observations.obs_functions import (
            wd_like_sweep_72,
        )

        # Isaac Lab 內建 OpenCV 的 GUI 通常被停用，改用 Matplotlib/Tk 顯示即時視窗
        if "agg" in matplotlib.get_backend().lower():
            matplotlib.use("TkAgg", force=True)
        import matplotlib.pyplot as plt

        self.plt = plt
        self.np = np
        self.raw_env = raw_env
        self.max_range = float(max_range)
        self.frame = frame
        self.lidar_r_min = float(lidar_r_min)
        self.wd_like_sweep_72 = wd_like_sweep_72

        # LiDAR 感測器配置（與訓練相同的 sweep 參數）
        self.sensor_cfg = SceneEntityCfg("lidar")
        self.sensor_cfg.resolve(raw_env.scene)

        # 地板回波距離 = VLP16 高度(1.6m) / tan(15°) - body_radius(0.3m) ≈ 5.6m
        self.ground_echo_dist = 1.6 / np.tan(np.radians(15.0)) - 0.3

        # 目標判定參數（與 goal_reached termination 一致）
        self.goal_body_radius = 0.35   # 機器人體積半徑
        self.goal_threshold = 0.35     # 到達目標距離門檻
        self.goal_success_center_dist = self.goal_body_radius + self.goal_threshold

        # 建立 matplotlib 互動視窗。
        # 版面原則：上方 text_ax 是純文字資料層；下方 ax 是完全不遮擋的 BEV 視覺層。
        self.plt.ion()
        self.fig = self.plt.figure(figsize=(7.5, 7.2), constrained_layout=True)
        gs = self.fig.add_gridspec(nrows=2, ncols=1, height_ratios=[1.8, 5.2], hspace=0.03)
        self.text_ax = self.fig.add_subplot(gs[0])
        self.ax = self.fig.add_subplot(gs[1])
        self.fig.patch.set_facecolor("#141414")
        # 歷史軌跡 buffer：儲存世界座標 (x, y)，每幀轉換為相對座標繪製
        self._trail_enabled = trail_length > 0
        self._trail: deque[tuple[float, float]] = deque(maxlen=max(trail_length, 1))

        try:
            self.fig.canvas.manager.set_window_title("Charge RL BEV — 72-bin LiDAR")
        except Exception:
            pass

    def _sensor_yaw(self) -> float:
        """取得 env 0 感測器在世界座標的航向角 (yaw)。Isaac Lab 四元數格式為 wxyz。"""
        q = self.raw_env.scene.sensors["lidar"].data.quat_w[0].detach().cpu().numpy()
        w, x, y, z = [float(v) for v in q]
        return float(self.np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))

    def _body_to_world(self, x_forward, y_left, yaw: float):
        """車體座標 (前方, 左方) → 世界座標 (x, y) 旋轉變換。"""
        cos_y = self.np.cos(yaw)
        sin_y = self.np.sin(yaw)
        x_world = cos_y * x_forward - sin_y * y_left
        y_world = sin_y * x_forward + cos_y * y_left
        return x_world, y_world

    def _world_rel_to_plot(self, rel_w, yaw: float):
        """世界座標相對向量 → 繪圖座標。body 模式下旋轉到車體座標。"""
        if self.frame == "world":
            return rel_w[..., 0], rel_w[..., 1]
        # 世界座標 → 車體座標（逆旋轉）
        cos_y = self.np.cos(yaw)
        sin_y = self.np.sin(yaw)
        x_forward = cos_y * rel_w[..., 0] + sin_y * rel_w[..., 1]
        y_left = -sin_y * rel_w[..., 0] + cos_y * rel_w[..., 1]
        return y_left, x_forward  # 繪圖：x 軸=左右, y 軸=前後

    def update(self, step: int, obs_tensor: torch.Tensor, actions: torch.Tensor,
               rvo2_filter=None):
        """更新 BEV 圖。回傳 False 表示使用者已關閉視窗。

        Args:
            rvo2_filter: 可選，RVO2SafetyFilter 實例。若提供，在 LiDAR 圖上疊加
                         v_pref（藍）、v_safe（綠）、偏差虛線（紅）箭頭。
        """
        np = self.np

        # --- 重算 72-bin sweep（與訓練觀測完全相同的處理流程）---
        sweep = self.wd_like_sweep_72(
            self.raw_env, self.sensor_cfg,
            num_bins=72, r_max=self.max_range, r_robot=0.3, r_min=self.lidar_r_min, z_filter=0.5,
        )
        real_dist = (sweep[0] * self.max_range).detach().cpu().numpy()  # 反正規化為公尺

        # 72 個 bin 的角度: -180° ~ +175°，每 bin 5°
        angles_deg = -180.0 + np.arange(real_dist.shape[0]) * 5.0
        angles = np.radians(angles_deg)

        # 極座標 → 車體笛卡爾座標
        x_forward = real_dist * np.cos(angles)  # 前方為正
        y_left = real_dist * np.sin(angles)     # 左方為正
        yaw = self._sensor_yaw()  # 目前航向角

        # 依 frame 設定選擇顯示座標系
        if self.frame == "world":
            plot_x, plot_y = self._body_to_world(x_forward, y_left, yaw)
            xlabel = "World X rel. robot (m)"
            ylabel = "World Y rel. robot (m)"
            title_suffix = "World"
        else:
            # 車體座標：x 軸=左右, y 軸=前後。0° 永遠是機器人正前方
            plot_x, plot_y = y_left, x_forward
            xlabel = "Left-Right y (m)"
            ylabel = "Forward x (m)"
            title_suffix = "Body (fwd=up)"

        # 檢查視窗是否仍存在
        if not self.plt.fignum_exists(self.fig.number):
            return False

        # --- 繪製 BEV 圖 ---
        ax = self.ax
        ax.cla()  # 清除上一幀
        ax.set_facecolor("#141414")
        self.fig.patch.set_facecolor("#141414")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-self.max_range, self.max_range)
        ax.set_ylim(-self.max_range, self.max_range)
        ax.set_xlabel(xlabel, color="white")
        ax.set_ylabel(ylabel, color="white")
        ax.tick_params(colors="white")
        ax.grid(True, color="#383838", linewidth=0.7)
        # 標題與狀態文字移到上方 text_ax，BEV 視覺層不放任何資訊框。

        # 距離環（2/5/10/15/20m 同心圓）
        for r_m in [2, 5, 10, 15, 20]:
            circle = self.plt.Circle((0, 0), r_m, fill=False, color="#4a4a4a", linewidth=0.8)
            ax.add_patch(circle)

        # LiDAR 點雲折線
        ax.plot(plot_x, plot_y, color="#7fbf7f", linewidth=1.2, alpha=0.8)

        # 點雲著色：綠=安全(>5m) / 橙=警告(2~5m) / 紅=危險(<2m) / 灰=max-range
        colors = np.full(real_dist.shape, "#50dc50", dtype=object)
        colors[real_dist < 5.0] = "#ffa500"
        colors[real_dist < 2.0] = "#ff3030"
        colors[real_dist >= self.max_range - 0.5] = "#808080"
        sizes = np.full(real_dist.shape, 22.0)
        sizes[real_dist < 5.0] = 36.0
        sizes[real_dist < 2.0] = 54.0
        ax.scatter(plot_x, plot_y, c=colors.tolist(), s=sizes, zorder=3)

        # 機器人：白色圓圈(r=0.3m) + 前方箭頭
        ax.add_patch(self.plt.Circle((0, 0), 0.3, fill=False, color="white", linewidth=2.0, zorder=4))
        if self.frame == "world":
            head_x = 1.2 * np.cos(yaw)
            head_y = 1.2 * np.sin(yaw)
        else:
            head_x, head_y = 0.0, 1.2  # 車體座標下前方固定朝上
        ax.arrow(0, 0, head_x, head_y, color="white", width=0.04, head_width=0.35, length_includes_head=True, zorder=5)
        ax.scatter([0], [0], c="white", s=28, marker="o", edgecolors="black", linewidths=0.6, zorder=7)

        # --- 歷史軌跡 ---
        if self._trail_enabled:
            robot_w = self.raw_env.scene["robot"].data.root_pos_w[0, :2].detach().cpu().numpy()
            # Reset 偵測：位置跳躍 > 2m → 清空軌跡（episode 結束時 robot 瞬移到新 spawn 點）
            if len(self._trail) > 0:
                last = self._trail[-1]
                dx = float(robot_w[0]) - last[0]
                dy = float(robot_w[1]) - last[1]
                if dx * dx + dy * dy > 4.0:  # > 2m
                    self._trail.clear()
            self._trail.append((float(robot_w[0]), float(robot_w[1])))
            if len(self._trail) >= 2:
                trail_arr = np.array(self._trail)  # (N, 2)
                rel_trail = trail_arr - robot_w[None, :]  # 相對當前位置
                tx, ty = self._world_rel_to_plot(rel_trail, yaw)
                n = len(tx)
                # 漸變色：舊=暗橙, 新=亮青，alpha 從 0.15 到 0.9
                from matplotlib.collections import LineCollection
                segments = np.column_stack([tx[:-1], ty[:-1], tx[1:], ty[1:]]).reshape(-1, 2, 2)
                alphas = np.linspace(0.15, 0.9, n - 1)
                # 使用 RGBA：從暗色 (0.6,0.3,0.1) 到亮色 (0.3,0.9,1.0)
                colors = np.zeros((n - 1, 4))
                t = np.linspace(0, 1, n - 1)
                colors[:, 0] = 0.6 * (1 - t) + 0.3 * t   # R
                colors[:, 1] = 0.3 * (1 - t) + 0.9 * t   # G
                colors[:, 2] = 0.1 * (1 - t) + 1.0 * t   # B
                colors[:, 3] = alphas                       # A
                lc = LineCollection(segments, colors=colors, linewidths=1.5, zorder=2)
                ax.add_collection(lc)

        # 繪製所有目標點（cyan X）+ 終止判定目標（洋紅 P）
        self._draw_goals(ax, yaw)
        goal_info = self._draw_termination_goal(ax, yaw)

        # 從觀測中讀取目標向量（obs[4:6] = 車體座標下的目標方向）
        if obs_tensor is not None and obs_tensor.shape[-1] >= 6:
            goal_xy = obs_tensor[0, 4:6].detach().cpu().numpy()
            gx = float(np.clip(goal_xy[0], -self.max_range, self.max_range))
            gy = float(np.clip(goal_xy[1], -self.max_range, self.max_range))
            if self.frame == "world":
                goal_x, goal_y = self._body_to_world(gx, gy, yaw)
            else:
                goal_x, goal_y = gy, gx
            # 黃色箭頭 = 觀測中的目標向量
            ax.arrow(0, 0, goal_x, goal_y, color="#ffd040", width=0.035, head_width=0.45, length_includes_head=True, zorder=4)
            ax.scatter([goal_x], [goal_y], c=["#ffd040"], s=80, zorder=5)

        # --- 上方資料面板內容（不畫在 BEV 網格上）---
        near_idx = int(real_dist.argmin())
        near_d = float(real_dist[near_idx])
        near_angle = -180.0 + near_idx * 5.0
        echo_bins = int(((real_dist >= self.ground_echo_dist - 0.3) & (real_dist <= self.ground_echo_dist + 0.3)).sum())
        action_text = actions[0].detach().cpu().tolist() if actions is not None else ["?", "?"]

        goal_status = "N/A"
        goal_center = "N/A"
        goal_edge = "N/A"
        goal_threshold = f"{self.goal_threshold:.2f}m"
        goal_idx = "N/A"
        if goal_info is not None:
            goal_status = "reached" if goal_info["success"] else "not yet"
            goal_center = f"{goal_info['center_dist']:.2f}m"
            goal_edge = f"{goal_info['effective_dist']:.2f}m"
            goal_idx = str(goal_info.get("idx")) if goal_info.get("idx") is not None else "N/A"
            goal_status = f"{goal_info['source']} / {goal_status}"

        rl_info = {
            "title": f"Charge RL BEV — 72-bin LiDAR ({title_suffix})",
            "step": str(step),
            "action": str(action_text),
            "cmd": self._agent_command_info(actions, rvo2_filter),
            "frame": self.frame,
            "yaw": f"{np.degrees(yaw):+.1f}°",
            "term": goal_status,
            "term_idx": goal_idx,
            "center": goal_center,
            "edge": goal_edge,
            "threshold": goal_threshold,
            "nearest": f"{near_d:.2f}m @ bin {near_idx} ({near_angle:+.0f}°)",
            "lidar_mean": f"{real_dist.mean():.2f}m",
            "lt2": f"{int((real_dist < 2.0).sum())}/72",
            "mid2_5": f"{int(((real_dist >= 2.0) & (real_dist < 5.0)).sum())}/72",
            "ground_echo": f"{echo_bins}/72",
            "max_range": f"{int((real_dist >= self.max_range - 0.5).sum())}/72",
            "legend": "White=robot  Yellow=nav goal  Magenta=term. goal  Gradient=trail",
        }

        # RVO2 向量疊加（藍=v_pref, 綠=v_safe, 紅虛線=偏差）。文字狀態改由上方面板顯示。
        if rvo2_filter is not None:
            rvo2_filter.draw_on_bev(ax, yaw, self.frame, show_overlay_text=False)

        self._draw_top_panel(rl_info, self._rvo2_panel_info(rvo2_filter))

        # 刷新畫面
        self.fig.canvas.draw_idle()
        self.plt.pause(0.001)
        return self.plt.fignum_exists(self.fig.number)

    def _agent_command_info(self, actions: torch.Tensor | None, rvo2_filter=None) -> dict:
        """讀取 env[0] 目前 action term 的物理指令，讓 BEV 明確顯示 agent 輸出。

        MultiDiscrete 動作本身是 [accel_idx, omega_idx]；真正送進車體的是
        DiscreteDifferentialDriveAction.processed_actions = [v_next, omega]。
        若 ORCA/RVO2 已介入，processed_actions 可能是過濾後實際套用值；面板會用 label 說明。
        """
        np = self.np
        info = {
            "v": "N/A",
            "omega": "N/A",
            "omega_deg": "N/A",
            "accel": "N/A",
            "raw_idx": "N/A",
            "label": "AGENT CMD",
        }
        if actions is not None:
            try:
                info["raw_idx"] = str(actions[0].detach().cpu().tolist())
            except Exception:
                pass

        try:
            action_term = list(self.raw_env.action_manager._terms.values())[0]
            processed = action_term.processed_actions[0].detach().cpu().numpy()
            applied = action_term.applied_accelerations[0].detach().cpu().numpy()
            if rvo2_filter is not None and hasattr(rvo2_filter, "last_agent_linear"):
                # ORCA 開啟時，processed_actions 可能是 post-filter；這裡明確顯示 ORCA 前 agent 原始輸出。
                v_cmd = float(rvo2_filter.last_agent_linear)
                omega_cmd = float(rvo2_filter.last_agent_omega)
            else:
                v_cmd = float(processed[0])
                omega_cmd = float(applied[1])
            accel_cmd = float(applied[0])
            info.update({
                "v": f"{v_cmd:+.3f} m/s",
                "omega": f"{omega_cmd:+.3f} rad/s",
                "omega_deg": f"{float(np.degrees(omega_cmd)):+.1f}°/s",
                "accel": f"{accel_cmd:+.3f} m/s²",
            })
        except Exception:
            pass
        return info

    def _rvo2_panel_info(self, rvo2_filter) -> dict:
        """整理 ORCA/RVO2 指標給上方資料面板；不在 BEV grid 疊文字。"""
        np = self.np
        if rvo2_filter is None:
            return {
                "status": "disabled",
                "v_pref": "N/A",
                "v_safe": "N/A",
                "rate": "N/A",
                "mean_distance": "N/A",
                "is_intervening": False,
            }

        v_pref_vec = getattr(rvo2_filter, "v_pref_world", np.zeros(2))
        v_safe_vec = getattr(rvo2_filter, "v_safe_world", np.zeros(2))
        v_pref = float(np.linalg.norm(v_pref_vec))
        v_safe = float(np.linalg.norm(v_safe_vec))
        v_diff = float(np.linalg.norm(v_safe_vec - v_pref_vec))
        total_steps = max(1, int(getattr(rvo2_filter, "total_steps", 0)))
        intervention_steps = int(getattr(rvo2_filter, "intervention_steps", 0))
        rate = intervention_steps / total_steps
        active = bool(getattr(rvo2_filter, "orca_active", False))
        fallback = bool(getattr(rvo2_filter, "fallback_active", False))
        is_intervening = active or fallback or v_diff > 0.01
        status = "ORCA fallback" if fallback else ("RVO active" if is_intervening else "RL pass-through")

        vo_cones = getattr(rvo2_filter, "_vo_cone_data", [])
        if vo_cones:
            distances = [float(np.linalg.norm(rel_xy)) for rel_xy, _ in vo_cones]
            mean_distance = f"{float(np.mean(distances)):.2f}m"
        else:
            mean_distance = "N/A"

        return {
            "status": status,
            "v_pref": f"{v_pref:.2f}",
            "v_safe": f"{v_safe:.2f}",
            "rate": f"{rate:.1%}",
            "mean_distance": mean_distance,
            "is_intervening": is_intervening,
        }

    def _draw_top_panel(self, rl_info: dict, orca_info: dict) -> None:
        """上方固定資料面板：三欄 grid layout + 動態 ORCA/RL 狀態指示燈。

        重要設計：這個 axes 只負責文字資料層。所有文字使用固定欄位 x 座標與
        分層 y 座標，避免在小視窗中把 Step / Action / ORCA 指標擠成同一行。
        """
        ax = self.text_ax
        ax.cla()
        ax.set_facecolor("#101214")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        # 分隔線與欄位底色：讓上方資料層和下方 BEV 視覺層清楚分開。
        ax.axhline(0.02, color="#3c3f44", linewidth=1.0)
        for x in (0.33, 0.66):
            ax.axvline(x, ymin=0.08, ymax=0.82, color="#272b30", linewidth=0.8)

        # ── 動態狀態指示燈（右上角）──
        is_orca_intervening = bool(orca_info.get("is_intervening", False))
        if is_orca_intervening:
            badge_text = " [ ORCA Intervening ] "
            badge_color = "#b3261e"
        else:
            badge_text = " [ RL Control ] "
            badge_color = "#1f8f3a"
        ax.text(
            0.985, 0.91, badge_text,
            transform=ax.transAxes, ha="right", va="center",
            color="white", fontsize=10.5, fontweight="bold",
            bbox={"facecolor": badge_color, "edgecolor": "none", "alpha": 0.88,
                  "boxstyle": "round,pad=0.45"},
        )

        # 固定三欄 grid。每欄只放短行，避免文字水平重疊。
        x_left, x_mid, x_right = 0.02, 0.36, 0.69
        y_header = 0.78
        y_rows = [0.64, 0.53, 0.42, 0.31, 0.20, 0.09]
        mono = "DejaVu Sans Mono"

        ax.text(0.02, 0.91, rl_info["title"], transform=ax.transAxes,
                color="white", fontsize=10.3, fontweight="bold", ha="left", va="center")

        # Column 1: agent/action command
        cmd = rl_info["cmd"]
        ax.text(x_left, y_header, "AGENT / ACTION", transform=ax.transAxes,
                color="#fff4a8", fontsize=9.2, fontweight="bold", ha="left", va="center")
        left_lines = [
            f"Step   : {rl_info['step']}",
            f"Raw idx: {cmd['raw_idx']}",
            f"v      : {cmd['v']}",
            f"omega  : {cmd['omega']}",
            f"omega° : {cmd['omega_deg']}",
            f"accel  : {cmd['accel']}",
        ]
        for y, line in zip(y_rows, left_lines):
            ax.text(x_left, y, line, transform=ax.transAxes,
                    color="#ffe9a8", fontsize=7.6, fontfamily=mono, ha="left", va="center")

        # Column 2: LiDAR / goal termination
        ax.text(x_mid, y_header, "LIDAR / GOAL", transform=ax.transAxes,
                color="#c9f7c9", fontsize=9.2, fontweight="bold", ha="left", va="center")
        mid_lines = [
            f"Nearest: {rl_info['nearest']}",
            f"Mean   : {rl_info['lidar_mean']}",
            f"<2m/2~5: {rl_info['lt2']} / {rl_info['mid2_5']}",
            f"Center : {rl_info['center']}",
            f"Edge   : {rl_info['edge']}",
            f"Term   : {rl_info['term']}",
        ]
        for y, line in zip(y_rows, mid_lines):
            ax.text(x_mid, y, line, transform=ax.transAxes,
                    color="#d8f7d8", fontsize=7.4, fontfamily=mono, ha="left", va="center")

        # Column 3: ORCA/RVO2 metrics
        ax.text(x_right, y_header, "ORCA / RVO2", transform=ax.transAxes,
                color="#88ccff", fontsize=9.2, fontweight="bold", ha="left", va="center")
        right_lines = [
            f"Status : {orca_info['status']}",
            f"|v_pref|: {orca_info['v_pref']}",
            f"|v_safe|: {orca_info['v_safe']}",
            f"Rate   : {orca_info['rate']}",
            f"Mean d : {orca_info['mean_distance']}",
            f"Frame  : {rl_info['frame']} {rl_info['yaw']}",
        ]
        for y, line in zip(y_rows, right_lines):
            ax.text(x_right, y, line, transform=ax.transAxes,
                    color="#d8e6ff", fontsize=7.6, fontfamily=mono, ha="left", va="center")

    def close(self):
        """關閉 BEV 視窗。"""
        if self.plt.fignum_exists(self.fig.number):
            self.plt.close(self.fig)

    def _selected_goal_index(self):
        """找出距離機器人最近的目標點索引（用於高亮顯示）。"""
        try:
            cmd = self.raw_env.command_manager.get_term("goal_command")
            if hasattr(cmd, "all_goals_pos_w"):
                goals_w = cmd.all_goals_pos_w[0, :int(cmd.cfg.num_goals), :2]
                robot_w = self.raw_env.scene["robot"].data.root_pos_w[0, :2]
                return int(torch.norm(goals_w - robot_w.unsqueeze(0), dim=1).argmin().item())
        except Exception:
            return None
        return None

    def _draw_goals(self, ax, yaw: float):
        """繪製所有 MultiGoalCommand 目標點（cyan X）+ 最近目標星號。"""
        try:
            cmd = self.raw_env.command_manager.get_term("goal_command")
            if not hasattr(cmd, "all_goals_pos_w"):
                return
            ng = int(cmd.cfg.num_goals)
            goals_w = cmd.all_goals_pos_w[0, :ng, :2].detach().cpu().numpy()
            robot_w = self.raw_env.scene["robot"].data.root_pos_w[0, :2].detach().cpu().numpy()
        except Exception:
            return

        # 計算相對位置並轉換到繪圖座標
        rel_w = goals_w - robot_w[None, :]
        goal_plot_x, goal_plot_y = self._world_rel_to_plot(rel_w, yaw)

        # 所有目標用 cyan X 標示；不疊文字索引，避免遮擋 BEV 視覺資料。
        selected_idx = self._selected_goal_index()
        ax.scatter(goal_plot_x, goal_plot_y, c="#40d8ff", s=38, marker="x", linewidths=1.2, zorder=4)

        # 最近目標用黃色星號標示
        if selected_idx is not None and 0 <= selected_idx < len(goal_plot_x):
            sx = goal_plot_x[selected_idx]
            sy = goal_plot_y[selected_idx]
            ax.scatter([sx], [sy], c="#ffd040", s=120, marker="*", edgecolors="black", linewidths=0.8, zorder=6)

    def _draw_termination_goal(self, ax, yaw: float):
        """繪製 goal_reached() 使用的精確終止判定目標，並回傳距離統計。

        顯示:
          - 洋紅 P 標記 = 尚未到達
          - 綠色 P 標記 = 已進入到達範圍
          - 虛線圓 = 到達判定半徑（body_radius + threshold）
        """
        try:
            robot_w = self.raw_env.scene["robot"].data.root_pos_w[0, :2].detach().cpu().numpy()
            idx = None
            if hasattr(self.raw_env, "_local_goal_world") and self.raw_env._local_goal_world is not None:
                target_w = self.raw_env._local_goal_world[0, :2].detach().cpu().numpy()
                source = "_local_goal_world"
            else:
                cmd = self.raw_env.command_manager.get_term("goal_command")
                if hasattr(cmd, "all_goals_pos_w"):
                    idx = self._selected_goal_index()
                    target_w = cmd.all_goals_pos_w[0, idx, :2].detach().cpu().numpy()
                    source = "multi_goal_command"
                else:
                    target_w = self.raw_env.command_manager.get_command("goal_command")[0, :2].detach().cpu().numpy()
                    source = "goal_command"
        except Exception:
            return None

        rel_w = target_w - robot_w
        tx, ty = self._world_rel_to_plot(rel_w, yaw)
        center_dist = float(self.np.linalg.norm(rel_w))
        effective_dist = max(center_dist - self.goal_body_radius, 0.0)
        success = effective_dist < self.goal_threshold
        color = "#ff40ff" if not success else "#40ff80"
        ax.scatter([tx], [ty], c=color, s=170, marker="P", edgecolors="black", linewidths=0.9, zorder=8)
        ax.add_patch(self.plt.Circle((tx, ty), self.goal_success_center_dist, fill=False, color=color, linestyle="--", linewidth=1.4, zorder=6))
        ax.plot([0, tx], [0, ty], color=color, linewidth=1.0, alpha=0.75, zorder=4)
        return {
            "source": source,
            "idx": idx,
            "center_dist": center_dist,
            "effective_dist": effective_dist,
            "success": success,
        }


def main():
    """主函式 — 載入 checkpoint、建立環境、執行 play 迴圈。"""

    # ================================================================
    # 1. Checkpoint 路徑解析
    # ================================================================
    # LiDAR sanity test 模式不需要 checkpoint
    if args_cli.lidar_sanity:
        ckpt_path = None
        _ckpt_args = {}
    else:
        ckpt_path = os.path.abspath(args_cli.checkpoint) if args_cli.checkpoint else find_latest_rnn_checkpoint()
        if ckpt_path is None or not os.path.exists(ckpt_path):
            raise FileNotFoundError("找不到 RNN checkpoint（logs/rnn_car 下無檔案且未指定 --checkpoint）")

        # 預讀 checkpoint 中的訓練參數，用於對齊環境設定
        _ckpt_meta = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        _ckpt_args = _ckpt_meta.get("args", {})
        del _ckpt_meta  # 釋放記憶體，完整載入會在指定 device 上重做

    # ================================================================
    # 2. 自動從 checkpoint 繼承關鍵設定
    # ================================================================
    # 自動繼承 curriculum_version（確保 play 使用與訓練相同的 stage 定義）
    if args_cli.curriculum_version is None:
        saved_cv = _ckpt_args.get("curriculum_version")
        if saved_cv:
            print(f"[PLAY] 自動設定 --curriculum_version={saved_cv}（從 checkpoint 讀取）")
            args_cli.curriculum_version = saved_cv

    # 自動繼承 lidar_no_noise（避免 play 時加入訓練未見過的雜訊）
    if _ckpt_args.get("lidar_no_noise", False) and not args_cli.lidar_no_noise:
        print("[PLAY] 自動啟用 --lidar_no_noise（從 checkpoint 讀取）")
        args_cli.lidar_no_noise = True

    # ================================================================
    # 3. 環境配置建立與覆寫
    # ================================================================
    env_cfg = resolve_env_cfg(args_cli.task)
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    # 套用消融實驗 CLI 覆寫（reward 權重、gate 模式等）
    apply_charge_env_overrides(env_cfg, args_cli)

    # 套用 curriculum stage 配置；--stage_parameter 控制是否把訓練時該 stage 的場景參數帶入 play。
    resolved_curriculum, stage_cfg = apply_stage_to_env_cfg(
        env_cfg, args_cli.curriculum_version, args_cli.stage,
        apply_scene_params=args_cli.stage_parameter,
    )

    # 套用場景參數：stage_parameter=True 時保留 stage 預設，只有明確 CLI 才覆寫；False 時用手動設定區/CLI。
    scene_final = configure_play_scene(env_cfg, stage_cfg, args_cli)

    # USD 場景切換（在場景參數套用之後，覆蓋 terrain + 停用牆壁 + 擴展 LiDAR）
    if args_cli.usd_scene:
        apply_usd_scene(env_cfg, args_cli.usd_scene)

    # 套用攝影機視角
    configure_camera(env_cfg, args_cli.camera)

    # 啟用 LiDAR 射線視覺化（play 專用）
    lidar_vis = not args_cli.no_lidar_vis
    if hasattr(env_cfg.scene, "lidar"):
        env_cfg.scene.lidar.debug_vis = lidar_vis

    # 非 headless 模式降低渲染頻率（每 4 步渲染一次）
    if not args_cli.headless:
        env_cfg.sim.render_interval = 4

    # USD 場景效能覆寫（必須在 lidar_vis / render_interval 之後，避免被蓋回去）
    if args_cli.usd_scene:
        # 複雜 USD 場景 + 5760 rays debug_vis → 卡頓，強制關閉
        if hasattr(env_cfg.scene, "lidar"):
            env_cfg.scene.lidar.debug_vis = False
        # render_interval=4 (每 0.04s) 在 USD 場景太頻繁，改為每 env step 一次
        env_cfg.sim.render_interval = env_cfg.decimation
        # BEV matplotlib 在 USD 場景更慢（CPU 要處理更多 render data），自動降頻
        if args_cli.bev_update_interval <= 5:
            args_cli.bev_update_interval = 10
        print(f"[PLAY] USD 效能優化: LiDAR debug_vis=OFF, "
              f"render_interval={env_cfg.sim.render_interval}, "
              f"bev_update_interval={args_cli.bev_update_interval}")

    # CLI 碰撞距離覆寫 — 修改所有 termination term 的碰撞距離參數
    if args_cli.collision_dist != COLLISION_DIST:
        try:
            terms = env_cfg.terminations.__dict__
            for tname, tterm in terms.items():
                params = getattr(tterm, 'params', {}) or {}
                if 'threshold' in params:
                    tterm.params["threshold"] = args_cli.collision_dist
                    print(f"[PLAY] 覆寫 termination '{tname}' threshold → {args_cli.collision_dist}m")
                if 'collision_distance' in params:
                    tterm.params["collision_distance"] = args_cli.collision_dist
                    print(f"[PLAY] 覆寫 termination '{tname}' collision_distance → {args_cli.collision_dist}m")
        except Exception as e:
            print(f"[PLAY] 無法覆寫 collision_dist: {e}")

    # 印出配置摘要
    print(f"[PLAY] checkpoint: {ckpt_path}")
    print(f"[PLAY] task: {args_cli.task}")
    if stage_cfg is not None:
        print(
            f"[PLAY] curriculum={resolved_curriculum} stage={args_cli.stage} "
            f"name={stage_cfg['name']} 目標={stage_cfg['num_goals']} "
            f"障礙={stage_cfg['num_obstacles_static']}S+{stage_cfg['num_obstacles_dynamic']}D "
            f"牆壁={stage_cfg['min_walls']}~{stage_cfg['max_walls']}"
        )

    # ================================================================
    # 4. 建立環境 + 初始化
    # ================================================================
    # Override env seed — default to random if not specified
    import random as _random
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
    else:
        env_cfg.seed = _random.randint(0, 99999)
    print(f"[PLAY] 環境 seed = {env_cfg.seed}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    raw_env = env.unwrapped
    obs, _ = env.reset()
    device = raw_env.device

    # LiDAR sanity test 模式：跑完測試即退出
    if args_cli.lidar_sanity:
        run_lidar_sanity_test(env, raw_env)
        env.close()
        return

    def policy_obs(x):
        """從觀測 dict 或 tensor 中取出 policy 用的觀測。"""
        if isinstance(x, dict):
            return x["policy"]
        return x

    obs_tensor = policy_obs(obs)

    # ================================================================
    # 5. 載入模型權重
    # ================================================================
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})

    # 從 checkpoint 讀取網路結構超參數
    hidden_dim = int(ckpt_args.get("hidden_dim", 30))           # RNN 隱藏層維度
    preprocess_dim = int(ckpt_args.get("preprocess_dim", 12))   # RNN 輸出特徵維度
    fc_dim = int(ckpt_args.get("fc_dim", 48))                   # RNN 前全連接層維度
    rnn_type = ckpt_args.get("rnn_type", "RNN")                 # RNN 類型（RNN/GRU/LSTM）
    encoder_mode = ckpt_args.get("charge_encoder_mode", "extractor_rnn")  # 編碼器模式
    zero_preprocess = ckpt_args.get("zero_preprocess_feature_for_rl", False)  # RNN 特徵是否歸零
    max_active_obstacles = int(ckpt_args.get("max_active_obstacles", 10))  # aux 用的最大障礙物數
    if zero_preprocess:
        print("[PLAY] --zero_preprocess_feature_for_rl 啟用: RL head 的 RNN 特徵被歸零")

    # 依 encoder_mode 決定模型結構
    #   wd_exact_rnn: 139D IsaacLab obs → 113D WD car layout，RNN 直接吃 113D
    #   extractor_rnn: 用 LidarStateExtractor 提取 96D 特徵餵給 RNN
    wd_exact_mode = (encoder_mode == "wd_exact_rnn")
    use_extractor = (encoder_mode == "extractor_rnn")
    policy_obs_dim = 113 if wd_exact_mode else len(POLICY_OBS_INDICES)
    middle_dim = int(ckpt_args.get("wd_middle_dim", 32)) if wd_exact_mode else None

    if use_extractor:
        extractor = LidarStateExtractor().to(device)
        rnn_input_dim = extractor.output_dim  # 96D
    else:
        extractor = None
        rnn_input_dim = policy_obs_dim  # legacy 79D 或 wd_exact 113D

    # 建立 RNN + Policy/Value heads
    preprocess_rnn = PreprocessRNN(
        input_dim=rnn_input_dim,
        hidden_dim=hidden_dim,
        preprocess_dim=preprocess_dim,
        fc_dim=fc_dim,
        rnn_type=rnn_type,
        middle_dim=middle_dim,
    ).to(device)
    policy_head = PolicyHead(input_dim=policy_obs_dim + preprocess_dim).to(device)
    value_head = ValueHead(input_dim=policy_obs_dim + preprocess_dim).to(device)

    # 載入訓練權重
    if use_extractor:
        extractor.load_state_dict(ckpt["extractor"])
        extractor.eval()
    preprocess_rnn.load_state_dict(ckpt["preprocess_rnn"])
    policy_head.load_state_dict(ckpt["policy_head"])
    value_head.load_state_dict(ckpt["value_head"])
    preprocess_rnn.eval()
    policy_head.eval()
    value_head.eval()
    print(
        f"[PLAY] encoder={encoder_mode} rnn_input={rnn_input_dim} "
        f"policy_obs={policy_obs_dim} hidden={hidden_dim} preprocess={preprocess_dim} "
        f"middle={middle_dim}"
    )

    # ================================================================
    # 6. 觀測正規化器（從 checkpoint 還原統計量）
    # ================================================================
    obs_norm = ckpt.get("obs_normalizer", {})
    mean = obs_norm.get("mean", torch.zeros(obs_tensor.shape[-1], device=device))
    var = obs_norm.get("var", torch.ones(obs_tensor.shape[-1], device=device))

    # RNN 隱藏狀態管理器（每個 env 獨立 hidden state）
    rnn_state = RNNStateManager(raw_env.num_envs, hidden_dim, device)

    # ================================================================
    # 7. 障礙物運動控制
    # ================================================================
    # 預設關閉 scripted 運動（與訓練行為一致）
    # --scripted_obstacles 時啟用 interval events 讓障礙物動起來（VO 測試用）
    raw_env._obstacle_policy_active = not args_cli.scripted_obstacles

    # 建立 BehaviorScheduler（與訓練一致）
    # 訓練時由 curriculum 建立，play 時手動建立
    _play_behavior_scheduler = None
    _n_static_play = scene_final["num_static"]
    _n_dynamic_play = scene_final["num_dynamic"]
    # num_dynamic=0 → 全部強制靜態，跳過 BehaviorScheduler
    _effective_behavior = args_cli.obstacle_behavior
    if _n_dynamic_play == 0 and _effective_behavior and _effective_behavior != "static":
        print(f"[PLAY] num_dynamic_obs=0 → obstacle_behavior 強制改為 'static'（原設定: '{_effective_behavior}'）")
        _effective_behavior = "static"
    if _effective_behavior and _effective_behavior != "static":
        try:
            from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_skrl.mdp.events.behavior_scheduler import BehaviorScheduler
            # behavior_mix: 根據用戶選擇的 obstacle_behavior + static/dynamic 數量
            _n_obs = _n_static_play + _n_dynamic_play
            # 動態 slot 使用用戶選擇的行為（而非硬編碼 patrol/random_walk）
            _dynamic_behavior = _effective_behavior  # e.g., "path_crossing", "patrol", etc.
            if _n_obs > 0 and _n_static_play > 0:
                _static_frac = _n_static_play / _n_obs
                _dynamic_frac = 1.0 - _static_frac
                _play_stage_config = {
                    "behavior_mix": {
                        "static": _static_frac,
                        _dynamic_behavior: _dynamic_frac,
                    },
                    "obs_near_goal_count": args_cli.obs_near_goal_count,
                    "obs_near_goal_radius": args_cli.obs_near_goal_radius,
                }
            else:
                # num_static=0 → 全部使用用戶選擇的動態行為
                _play_stage_config = {
                    "behavior_mix": {
                        _dynamic_behavior: 1.0,
                    },
                    "obs_near_goal_count": args_cli.obs_near_goal_count,
                    "obs_near_goal_radius": args_cli.obs_near_goal_radius,
                }
            # 從 event params 讀取 boundary 和 spawn_zones（T 走廊等非矩形場景）
            _obs_evt = getattr(env_cfg.events, "randomize_obstacles", None)
            _spawn_zones = None
            _bnd = getattr(raw_env, '_room_boundary', 7.0)
            if _obs_evt is not None:
                _spawn_zones = _obs_evt.params.get("spawn_zones", None)
                _evt_bnd = _obs_evt.params.get("boundary", None)
                if _evt_bnd is not None:
                    _bnd = _evt_bnd  # 可以是 tuple (28.0, 9.0)
            _play_behavior_scheduler = BehaviorScheduler(
                stage_config=_play_stage_config,
                num_envs=raw_env.num_envs,
                max_obstacles=_n_obs,
                device=str(device),
                boundary=_bnd,
                spawn_zones=_spawn_zones,
            )
            # Play loop 直接呼叫 step()，不需 interval event 重複呼叫
            raw_env._obstacle_policy_active = True  # 讓 interval event skip
            raw_env._behavior_scheduler = _play_behavior_scheduler
            # 初始 reset
            _all_ids = torch.arange(raw_env.num_envs, device=device)
            _play_behavior_scheduler.reset(_all_ids, raw_env)
            print(f"[PLAY] BehaviorScheduler 啟用: {_n_obs} slots (static={_n_static_play}, dynamic={_n_dynamic_play}), "
                  f"mix={_play_stage_config['behavior_mix']}")
        except Exception as e:
            print(f"[PLAY] BehaviorScheduler 建立失敗: {e}")
            _play_behavior_scheduler = None
    else:
        print("[PLAY] 障礙物行為: static（無 BehaviorScheduler）")

    print(
        "[PLAY] 障礙物運動: "
        + ("BehaviorScheduler 控制 (play loop)" if _play_behavior_scheduler else
           "scripted interval events 啟用" if args_cli.scripted_obstacles else
           "全部靜止")
    )

    # Goal movement（與訓練一致，--no_goal_movement 可強制關閉）
    _play_goal_mover = None
    if hasattr(args_cli, 'stage') and args_cli.stage is not None and not args_cli.no_goal_movement:
        try:
            from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_skrl.mdp.events.goal_movement import move_goal_positions
            from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_skrl.curriculum.phases.wd_single_agent_v1 import STAGES, _flatten_phase
            _stage_idx = args_cli.stage - 1  # 0-indexed
            if 0 <= _stage_idx < len(STAGES):
                _phase_cfg = _flatten_phase(STAGES[_stage_idx])
                _gm_speed = _phase_cfg.get("goal_move_speed", 0.0)
                _gm_behavior = _phase_cfg.get("goal_move_behavior", "random_walk")
                if _gm_speed > 0:
                    _gm_params = {
                        "goal_move_speed": _gm_speed,
                        "goal_move_max_radius": _phase_cfg.get("goal_move_max_radius", 3.0),
                        "goal_move_behavior": _gm_behavior,
                        "goal_move_angular_speed": _phase_cfg.get("goal_move_angular_speed", 0.5),
                        "goal_move_dt": 0.2,
                        "goal_move_wall_margin": 0.5,
                        "goal_move_obs_margin": 0.8,
                        "goal_move_dir_steps_min": 15,
                        "goal_move_dir_steps_max": 40,
                    }
                    _all_env_ids = torch.arange(raw_env.num_envs, device=device)
                    _gm_call_count = [0]
                    def _goal_mover_fn(env_ref, params=_gm_params, eids=_all_env_ids, cc=_gm_call_count):
                        move_goal_positions(env_ref, env_ids=eids, **params)
                        cc[0] += 1
                        if cc[0] <= 3 or cc[0] % 100 == 0:
                            try:
                                g = env_ref.command_manager.get_term("goal_command")
                                gp = g.goal_pos_w[0, :2].tolist()
                                print(f"[GOAL_MOVE] call#{cc[0]} goal[0]=({gp[0]:.1f},{gp[1]:.1f})", flush=True)
                            except Exception:
                                pass
                    _play_goal_mover = _goal_mover_fn
                    print(f"[PLAY] Goal movement 啟用: behavior={_gm_behavior}, speed={_gm_speed}")
                else:
                    print("[PLAY] Goal movement: speed=0，關閉")
        except Exception as e:
            print(f"[PLAY] Goal movement 建立失敗: {e}")

    # ================================================================
    # 8. Play 迴圈前置變數
    # ================================================================
    step_dt = env.step_dt if hasattr(env, "step_dt") else raw_env.step_dt
    episode_reward = torch.zeros(raw_env.num_envs, device=device)     # 累積回合獎勵
    episode_step = torch.zeros(raw_env.num_envs, dtype=torch.long, device=device)  # 回合步數

    # --- 回合統計計數器 ---
    stats_goal = 0       # 到達目標次數
    stats_wall = 0       # 撞牆次數
    stats_obs = 0        # 撞障礙物次數
    stats_timeout = 0    # 超時次數
    stats_other = 0      # 其他終止（翻倒、飛出等）
    stats_total = 0      # 總回合數
    stats_steps_list = []  # 每回合步數（用於計算平均）

    # --- Per-episode step-by-step velocity/position log (env 0) ---
    _ep_step_log: list[dict] = []    # current episode buffer
    _all_ep_logs: list[dict] = []    # finished episodes: {episode, cause, steps, log:[...]}

    # --- play_diag 累積診斷變數 ---
    diag_heading_sum = 0.0              # 累積航向誤差（度）
    diag_velocity_to_goal_sum = 0.0     # 累積朝目標速度
    diag_goal_distance_sum = 0.0        # 累積目標距離
    diag_action_linear_sum = 0.0        # 累積線性動作 index
    diag_action_angular_sum = 0.0       # 累積角度動作 index
    diag_samples = 0                    # 診斷樣本數

    # --- RVO2 (ORCA) Safety Filter 初始化 ---
    rvo2_filter = None
    if args_cli.use_rvo2_filter:
        try:
            from rvo2_safety_filter import RVO2SafetyFilter
            action_term = list(raw_env.action_manager._terms.values())[0]
            rvo2_filter = RVO2SafetyFilter(
                raw_env,
                num_envs=raw_env.num_envs,
                time_horizon=args_cli.rvo2_time_horizon,
                time_horizon_obst=args_cli.rvo2_time_horizon_static,
                angle_threshold_deg=args_cli.rvo2_angle_threshold,
                safety_margin=args_cli.rvo2_safety_margin,
                culling_radius=args_cli.rvo2_culling_radius,
                neighbor_dist=args_cli.rvo2_neighbor_dist,
                max_neighbors=args_cli.rvo2_max_neighbors,
                obs_inflation=args_cli.rvo2_obs_inflation,
                tactical_retreat=args_cli.rvo2_tactical_retreat,
                disp_threshold=args_cli.rvo2_disp_threshold,
                disp_window=args_cli.rvo2_disp_window,
            )
            rvo2_filter.install(action_term)
        except ImportError:
            print("[RVO2] 警告: pyrvo2 未安裝，RVO2 filter 跳過。請執行: pip install pyrvo2")
        except Exception as e:
            print(f"[RVO2] 初始化失敗: {e}")

    # --- RSGS-Lite 初始化 ---
    rsgs_filter = None
    if args_cli.use_rsgs:
        from rsgs_lite import RSGSConfig, RSGSLite
        rsgs_cfg = RSGSConfig(
            stuck_window=args_cli.rsgs_stuck_window,
            stuck_threshold=args_cli.rsgs_stuck_threshold,
            gap_min_width=args_cli.rsgs_gap_min_width,
            gap_clear_threshold=args_cli.rsgs_gap_clear_threshold,
            recovery_distance=args_cli.rsgs_recovery_distance,
            max_recovery_steps=args_cli.rsgs_max_recovery_steps,
            exit_displacement=args_cli.rsgs_exit_displacement,
            goal_bias_weight=args_cli.rsgs_goal_bias,
        )
        rsgs_filter = RSGSLite(raw_env, raw_env.num_envs, device, cfg=rsgs_cfg)
        print(f"[RSGS] RSGS-Lite 已啟用 "
              f"(window={rsgs_cfg.stuck_window}, threshold={rsgs_cfg.stuck_threshold}m, "
              f"recovery_dist={rsgs_cfg.recovery_distance}m, "
              f"max_steps={rsgs_cfg.max_recovery_steps})")

    # --- BEV 俯視圖初始化 ---
    bev_visualizer = None
    if args_cli.bev_vis:
        if args_cli.num_envs != 1:
            print("[PLAY] --bev_vis 只顯示 env 0；num_envs > 1 可用但可讀性較低")
        try:
            bev_visualizer = LiveBEVVisualizer(
                raw_env, max_range=args_cli.bev_max_range, frame=args_cli.bev_frame,
                lidar_r_min=args_cli.lidar_r_min,
                trail_length=args_cli.bev_trail_length,
            )
            print(f"[PLAY] BEV 視窗已啟用（座標系={args_cli.bev_frame}，軌跡={args_cli.bev_trail_length}點）。關閉視窗即停止 play。")
        except Exception as exc:
            print(f"[PLAY] 警告: BEV 初始化失敗: {exc}")
            bev_visualizer = None

    # ================================================================
    # 9. 推論用的觀測處理函式
    # ================================================================

    def normalize(x):
        """用訓練時的 running mean/var 正規化觀測，clamp 到 [-5, 5]。"""
        return torch.clamp((x - mean) / (var.sqrt() + 1e-8), -5.0, 5.0)

    def _wd_like_obs(obs_normed: torch.Tensor) -> torch.Tensor:
        """將 IsaacLab 139D 觀測投射到 WD car 113D 格式（wd_exact_rnn 專用）。

        113D = base(17) + obstacles(60) + lidar36(36)
        base 17D 中的映射:
          [0] = speed_x    ← IL obs[1]
          [2:4] = goal x/y ← IL obs[4:6]
          [15] = time       ← IL obs[138]
          [16] = in_game    ← 固定 1.0
        lidar36 = 72-bin 每兩個 bin 取平均，降至 36-bin
        """
        leading = obs_normed.shape[:-1]
        base = torch.zeros(*leading, 17, dtype=obs_normed.dtype, device=obs_normed.device)
        base[..., 0] = obs_normed[..., 1]       # speed_x: 前進速度
        base[..., 2:4] = obs_normed[..., 4:6]   # 車體座標目標 x/y
        base[..., 15] = obs_normed[..., 138]    # 時間步
        base[..., 16] = 1.0                     # in_game 旗標
        obstacles = obs_normed[..., 78:138]     # Top10 障礙物 6D × 10 = 60D
        lidar72 = obs_normed[..., 6:78]         # 72-bin LiDAR
        lidar36 = lidar72.reshape(*leading, 36, 2).mean(dim=-1)  # 降採樣至 36-bin
        return torch.cat([base, obstacles, lidar36], dim=-1)  # 17+60+36 = 113D

    def charge_obs_for_rl(obs_normed: torch.Tensor) -> torch.Tensor:
        """依 encoder_mode 取出 policy 需要的觀測切片。"""
        return _wd_like_obs(obs_normed) if wd_exact_mode else obs_normed[:, POLICY_OBS_INDICES]

    def charge_features_for_rnn(obs_normed: torch.Tensor, p_obs: torch.Tensor) -> torch.Tensor:
        """取出 RNN 的輸入特徵。extractor 模式用 LidarStateExtractor，否則直接用 p_obs。"""
        if use_extractor:
            return extractor(obs_normed)
        return p_obs

    def detect_termination_cause(env_unwrapped, terminated_flat, truncated_flat):
        """從 termination_manager 的 buffer 判定每個 env 的終止原因。

        原因代碼: 0=執行中, 1=到達目標, 2=撞牆, 3=撞障礙物, 4=超時, 5=其他
        """
        N = terminated_flat.shape[0]
        device = terminated_flat.device
        cause = torch.zeros(N, dtype=torch.long, device=device)
        try:
            tm = env_unwrapped.termination_manager
            # 遍歷所有 termination term，依名稱分類
            for name in tm._term_names:
                buf = tm.get_term(name)
                if buf is None:
                    continue
                fired = buf.bool()  # 這一步觸發了哪些 env
                if "goal_reached" in name or "reaching_goal" in name:
                    cause[fired & (cause == 0)] = 1  # 到達目標
                elif "wall_collision" in name:
                    cause[fired & (cause == 0)] = 2  # 撞牆
                elif "obstacle_collision" in name:
                    cause[fired & (cause == 0)] = 3  # 撞障礙物
                elif "collision" in name:
                    cause[fired & (cause == 0)] = 3  # 其他碰撞也歸為障礙物
                elif "tipped" in name or "explosion" in name or "flying" in name:
                    cause[fired & (cause == 0)] = 5  # 翻倒/爆炸/飛出
        except (AttributeError, RuntimeError):
            pass
        # 超時（truncated 但未被 terminated 的 env）
        trunc = truncated_flat.bool() if truncated_flat.dim() == 1 else truncated_flat.squeeze(-1).bool()
        cause[trunc & (cause == 0)] = 4
        return cause

    # 終止原因名稱映射
    CAUSE_NAMES = {0: "執行中", 1: "到達", 2: "撞牆", 3: "撞障礙", 4: "超時", 5: "其他"}

    # ================================================================
    # 9b. 目標附近障礙物放置
    # ================================================================
    def place_obstacles_near_goal(env_ids_to_place=None):
        """將前 N 個障礙物強制放置在 goal 附近。

        每次 episode reset 後呼叫，把指定數量的障礙物移到 goal 周圍
        obs_near_goal_radius 範圍內。
        """
        count = args_cli.obs_near_goal_count
        radius = args_cli.obs_near_goal_radius
        if count <= 0:
            return

        if env_ids_to_place is None:
            env_ids_to_place = torch.arange(raw_env.num_envs, device=device)
        elif not isinstance(env_ids_to_place, torch.Tensor):
            env_ids_to_place = torch.tensor(env_ids_to_place, device=device, dtype=torch.long)

        N = len(env_ids_to_place)
        if N == 0:
            return

        # 取得 goal 世界座標
        try:
            goal_cmd = raw_env.command_manager.get_command("goal_command")
            goal_xy = goal_cmd[env_ids_to_place, :2]  # [N, 2]
        except (AttributeError, KeyError, IndexError):
            return

        min_dist = 0.8  # 不要太貼 goal 中心

        for i in range(count):
            obs_name = f"obstacle_{i}"
            try:
                obstacle = raw_env.scene[obs_name]
            except KeyError:
                break

            # 在 goal 附近 [min_dist, radius] 環形區域隨機生成
            angle = torch.rand(N, device=device) * 2.0 * 3.14159265
            dist = torch.rand(N, device=device) * (radius - min_dist) + min_dist
            offset_x = dist * torch.cos(angle)
            offset_y = dist * torch.sin(angle)

            # pose [N, 7] = [x, y, z, qw, qx, qy, qz]
            pose = torch.zeros(N, 7, device=device)
            pose[:, 0] = goal_xy[:, 0] + offset_x
            pose[:, 1] = goal_xy[:, 1] + offset_y
            pose[:, 2] = 0.9  # 可見高度
            pose[:, 3] = 1.0  # quat w

            obstacle.write_root_pose_to_sim(pose, env_ids=env_ids_to_place)
            # 歸零速度
            vel = torch.zeros(N, 6, device=device)
            obstacle.write_root_velocity_to_sim(vel, env_ids=env_ids_to_place)

        if not hasattr(place_obstacles_near_goal, "_printed"):
            place_obstacles_near_goal._printed = True
            print(f"[PLAY] obs_near_goal: {count} 個障礙物強制放置在 goal {min_dist:.1f}~{radius:.1f}m 範圍內")

    # 初始 reset 後立即放置
    place_obstacles_near_goal()

    # ================================================================
    # 10. 主 Play 迴圈
    # ================================================================
    # LiDAR distance bias — 轉換到 z-score 空間（normalize 後加）
    # 用 BIAS_SCALE (2.0m) 而非 max_range (20.0m) 作為基準，
    # 否則 0.2m / 20.0 = 0.01 → /std ≈ 0.067 std，policy 完全感知不到。
    # 用 2.0m 基準：0.2m / 2.0 = 0.1 → /std ≈ 0.67 std，policy 可明確感知。
    _LIDAR_BIAS_SCALE = 2.0  # 有效避障範圍 (m)，決定 bias 對 policy 的感知強度
    _lidar_bias_zscore = None
    if args_cli.lidar_dist_bias > 0:
        lidar_var = var[6:78]
        var_from_ckpt = "obs_normalizer" in ckpt
        lidar_std = lidar_var.sqrt() + 1e-8  # [72] per-bin std
        _lidar_bias_zscore = (args_cli.lidar_dist_bias / _LIDAR_BIAS_SCALE) / lidar_std  # [72]
        avg_effect = _lidar_bias_zscore.mean().item()
        print(f"[PLAY] LiDAR distance bias: +{args_cli.lidar_dist_bias:.2f}m "
              f"(scale={_LIDAR_BIAS_SCALE}m, z-score shift: avg {avg_effect:.2f} std, "
              f"range {_lidar_bias_zscore.min().item():.2f}~{_lidar_bias_zscore.max().item():.2f})")
        if not var_from_ckpt:
            print(f"[PLAY] ⚠ obs_normalizer 不在 checkpoint 中，var=1.0 fallback → "
                  f"bias 效果可能偏弱 (avg {avg_effect:.2f} std)")
        print(f"[PLAY]   lidar_std 範圍: {lidar_std.min().item():.4f} ~ {lidar_std.max().item():.4f}")

    # --- Per-step 效能分析器 ---
    _perf_accum = {"inference": 0.0, "env_step": 0.0, "orca": 0.0, "bev": 0.0, "other": 0.0, "total": 0.0}
    _perf_count = 0
    _PERF_INTERVAL = 100  # 每 N 步印一次效能摘要

    step = 0
    while simulation_app.is_running() and step < args_cli.steps:
        start = time.time()

        with torch.inference_mode():
            # --- 觀測處理 → 模型推論 → 動作選擇 ---
            obs_tensor = policy_obs(obs)
            obs_normed = normalize(obs_tensor)               # 正規化觀測
            # LiDAR distance bias: 在 z-score 空間加偏移，讓 agent 覺得障礙物更遠
            if _lidar_bias_zscore is not None:
                if step % 500 == 0:
                    _before_min = obs_normed[0, 6:78].min().item()
                obs_normed[:, 6:78] = obs_normed[:, 6:78] + _lidar_bias_zscore
                if step % 500 == 0:
                    _after_min = obs_normed[0, 6:78].min().item()
                    print(f"[BIAS] step {step}: env0 lidar_min z-score "
                          f"before={_before_min:.3f} after={_after_min:.3f} "
                          f"delta={_after_min - _before_min:.3f}")
            p_obs = charge_obs_for_rl(obs_normed)             # 取出 policy 觀測切片
            features = charge_features_for_rnn(obs_normed, p_obs)  # RNN 輸入特徵
            hidden = rnn_state.get()                          # 取得目前 RNN 隱藏狀態
            rnn_feat, aux_pred, new_hidden = preprocess_rnn(  # RNN 前向傳播
                features, hidden, training=args_cli.aux_debug  # aux_debug 時保留 aux 輸出
            )
            # zero_preprocess 模式：RNN 特徵歸零，policy 只靠當前觀測決策
            rnn_for_rl = torch.zeros_like(rnn_feat) if zero_preprocess else rnn_feat
            rl_in = torch.cat([p_obs, rnn_for_rl], dim=-1)    # 拼接觀測 + RNN 特徵
            logits = policy_head(rl_in)                        # Policy head 輸出 logits
            actions = sample_action(logits, args_cli.deterministic)  # 取樣或 argmax

            # --- play_diag: 累積導航對齊診斷 ---
            if args_cli.play_diag:
                # obs[4:6] = 車體座標下的目標向量 (x_forward, y_left)
                goal_xy = obs_tensor[:, 4:6]
                goal_dist = torch.linalg.norm(goal_xy, dim=-1)  # 目標距離
                # 航向誤差 = 目標方向與前方的夾角
                heading_abs = torch.atan2(goal_xy[:, 1], goal_xy[:, 0]).abs() * 180.0 / torch.pi
                # 從 simulator 讀取世界座標速度，轉換到車體座標
                robot_data = raw_env.scene["robot"].data
                vel_w = robot_data.root_lin_vel_w[:, :2]  # 世界座標速度 [vx, vy]
                quat_w = robot_data.root_quat_w           # 世界座標四元數
                qw, qx, qy, qz = quat_w[:, 0], quat_w[:, 1], quat_w[:, 2], quat_w[:, 3]
                yaw = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
                cos_yaw = torch.cos(yaw)
                sin_yaw = torch.sin(yaw)
                # 世界速度 → 車體速度
                vx_body = cos_yaw * vel_w[:, 0] + sin_yaw * vel_w[:, 1]
                vy_body = -sin_yaw * vel_w[:, 0] + cos_yaw * vel_w[:, 1]
                vel_body = torch.stack([vx_body, vy_body], dim=-1)
                # 朝目標速度 = 車體速度在目標方向上的投影
                goal_dir = goal_xy / torch.clamp(goal_dist.unsqueeze(-1), min=1e-6)
                velocity_to_goal = (vel_body * goal_dir).sum(dim=-1)
                # 累積統計
                diag_heading_sum += float(heading_abs.mean().item())
                diag_velocity_to_goal_sum += float(velocity_to_goal.mean().item())
                diag_goal_distance_sum += float(goal_dist.mean().item())
                diag_action_linear_sum += float(actions[:, 0].float().mean().item())
                diag_action_angular_sum += float(actions[:, 1].float().mean().item())
                diag_samples += 1

            # --- aux_debug: 定期印出 RNN 7D 預測 vs 真實 ---
            if args_cli.aux_debug and aux_pred is not None and step % max(1, args_cli.aux_debug_interval) == 0:
                print_aux_debug(raw_env, step, obs_tensor, aux_pred, max_active_obstacles)

        _t_inference = time.time() - start

        # inference_mode 產生的 tensor 不允許 in-place 修改，
        # 但 per-env hidden reset 需要寫入，所以要 clone
        new_hidden = new_hidden.clone()

        # --- 環境步進 ---
        _t0_env = time.time()
        next_obs, reward, terminated, truncated, info = env.step(actions.float())
        # BehaviorScheduler 每步移動障礙物（reset 後前 3 步暫停，防止動態 obs 衝入）
        if _play_behavior_scheduler is not None:
            if int(episode_step[0].item()) > 3:
                _play_behavior_scheduler.step(raw_env, dt=step_dt)
        # Goal movement 每步移動 goal（與訓練一致）
        if _play_goal_mover is not None:
            _play_goal_mover(raw_env)
        # RSGS-Lite: stuck detection + recovery goal injection
        if rsgs_filter is not None:
            rsgs_filter.step(obs_tensor)
        # 合併 terminated + truncated 為 done 旗標
        done = (terminated.squeeze(-1) | truncated.squeeze(-1)) if terminated.ndim > 1 else (terminated | truncated)
        _t_env = time.time() - _t0_env

        # 累積回合獎勵與步數
        episode_reward += reward.squeeze(-1) if reward.ndim > 1 else reward
        episode_step += 1
        rnn_state.update(new_hidden)  # 更新 RNN 隱藏狀態

        # --- Per-step velocity/position log (env 0) for stuck diagnosis ---
        _t0_orca = time.time()
        if rvo2_filter is not None:
            _robot = raw_env.scene["robot"]
            _r_pos = _robot.data.root_pos_w[0, :2].detach().cpu().tolist()
            _r_vel = _robot.data.root_lin_vel_w[0, :2].detach().cpu().tolist()
            _r_speed = ((_r_vel[0]**2 + _r_vel[1]**2)**0.5)
            # goal in body frame from obs
            _obs_flat = obs_tensor.reshape(raw_env.num_envs, -1)
            _goal_body = _obs_flat[0, 4:6].detach().cpu().tolist()
            # nearest obstacle distance
            _nearest_obs_dist = 999.0
            if _play_behavior_scheduler is not None:
                _rp = _robot.data.root_pos_w[0:1, :2]
                _eo = raw_env.scene.env_origins[0:1, :2]
                _rl = _rp - _eo
                _op = _play_behavior_scheduler.positions[0:1, :, :2]
                _dd = (_op - _rl.unsqueeze(1)).norm(dim=2)
                _act = _play_behavior_scheduler.behavior_type[0:1] != 0
                _dd[~_act] = 999.0
                _nearest_obs_dist = float(_dd.min().item())
            _step_rec = {
                "step": int(episode_step[0].item()),
                "agent_pos": _r_pos,
                "agent_vel": _r_vel,
                "speed": round(_r_speed, 4),
                "v_pref": rvo2_filter.v_pref_world.tolist(),
                "v_safe": rvo2_filter.v_safe_world.tolist(),
                "v_body_pref": round(rvo2_filter.last_agent_linear, 4),
                "orca_active": rvo2_filter.orca_active,
                "recovery": rvo2_filter.recovery_active,
                "goal_body": _goal_body,
                "nearest_obs": round(_nearest_obs_dist, 3),
            }
            _ep_step_log.append(_step_rec)

            # Live stuck detection: if speed < 0.1 for 5+ consecutive steps, print each step
            _consec_low = 0
            for _s in reversed(_ep_step_log):
                if _s["speed"] < 0.1:
                    _consec_low += 1
                else:
                    break
            if _consec_low >= 5:
                s = _step_rec
                print(
                    f"[STUCK LIVE] ep_step={s['step']:3d} speed={s['speed']:.3f} "
                    f"v_body={s['v_body_pref']:+.3f} orca={'Y' if s['orca_active'] else 'N'} "
                    f"near_obs={s['nearest_obs']:.2f}m "
                    f"pos=({s['agent_pos'][0]:+.2f},{s['agent_pos'][1]:+.2f}) "
                    f"goal=({s['goal_body'][0]:+.2f},{s['goal_body'][1]:+.2f}) "
                    f"vpref=({s['v_pref'][0]:+.3f},{s['v_pref'][1]:+.3f}) "
                    f"vsafe=({s['v_safe'][0]:+.3f},{s['v_safe'][1]:+.3f})"
                )

        # --- Per-step 行為診斷 logging ---
        if hasattr(args_cli, 'play_diag') and args_cli.play_diag and step < args_cli.steps:
            _obs_flat = obs_tensor.reshape(raw_env.num_envs, -1)
            _lidar = _obs_flat[:, 6:78]  # 72 rays (normalized by max_range)
            _lidar_min = _lidar.min(dim=1).values
            _speed = _obs_flat[:, 1]  # ego linear velocity (normalized by v_max=1.0)
            _omega = _obs_flat[:, 2]  # ego angular velocity (normalized by omega_max=1.5)
            # 最近障礙物距離（從 BehaviorScheduler）
            if _play_behavior_scheduler is not None:
                _robot_pos = raw_env.scene["robot"].data.root_pos_w[:, :2]
                _env_origins = raw_env.scene.env_origins[:, :2]
                _robot_local = _robot_pos - _env_origins
                _obs_pos = _play_behavior_scheduler.positions[:, :, :2]
                _diff = _obs_pos - _robot_local.unsqueeze(1)
                _dists = _diff.norm(dim=2)
                _active = _play_behavior_scheduler.behavior_type != 0
                _dists[~_active] = 999.0
                _obs_min_dist = _dists.min(dim=1).values
            else:
                _obs_min_dist = torch.full((raw_env.num_envs,), 999.0, device=device)
            # 寫入全局 list（在結束時輸出統計）
            if not hasattr(main, '_diag_data'):
                main._diag_data = {'lidar_min': [], 'speed': [], 'omega': [], 'obs_dist': []}
            main._diag_data['lidar_min'].append(_lidar_min.cpu().numpy())
            main._diag_data['speed'].append(_speed.cpu().numpy())
            main._diag_data['omega'].append(_omega.cpu().numpy())
            main._diag_data['obs_dist'].append(_obs_min_dist.cpu().numpy())

        _t_orca = time.time() - _t0_orca

        # --- 回合結束處理 ---
        if done.any():
            done_ids = done.nonzero(as_tuple=False).reshape(-1)  # 結束的 env ID
            terminated_flat = terminated.squeeze(-1) if terminated.ndim > 1 else terminated
            truncated_flat = truncated.squeeze(-1) if truncated.ndim > 1 else truncated
            cause = detect_termination_cause(raw_env, terminated_flat, truncated_flat)

            # 逐 env 印出回合結果 + 更新統計
            for env_id in done_ids.tolist():
                c = cause[env_id].item()
                cause_name = CAUSE_NAMES.get(c, "?")
                ep_steps = episode_step[env_id].item()
                ep_rew = episode_reward[env_id].item()
                stats_total += 1
                stats_steps_list.append(ep_steps)
                if c == 1:
                    stats_goal += 1
                elif c == 2:
                    stats_wall += 1
                elif c == 3:
                    stats_obs += 1
                elif c == 4:
                    stats_timeout += 1
                else:
                    stats_other += 1
                print(
                    f"[回合 {stats_total:3d}] 環境={env_id} 原因={cause_name:7s} "
                    f"步數={ep_steps:4d} 獎勵={ep_rew:.1f}"
                )

                # Save per-step log for env 0
                if env_id == 0 and rvo2_filter is not None and _ep_step_log:
                    speeds = [s["speed"] for s in _ep_step_log]
                    avg_speed = sum(speeds) / max(len(speeds), 1)
                    low_speed_steps = sum(1 for s in speeds if s < 0.1)
                    _all_ep_logs.append({
                        "episode": stats_total,
                        "cause": cause_name,
                        "steps": ep_steps,
                        "reward": ep_rew,
                        "avg_speed": round(avg_speed, 4),
                        "low_speed_steps": low_speed_steps,
                        "low_speed_ratio": round(low_speed_steps / max(len(speeds), 1), 3),
                        "log": list(_ep_step_log),
                    })
                    # Per-episode velocity summary
                    print(
                        f"  └─ 速度: avg={avg_speed:.3f} low(<0.1)={low_speed_steps}/{len(speeds)} "
                        f"({low_speed_steps/max(len(speeds),1):.0%})"
                    )
                    _ep_step_log.clear()

            # 重置結束 env 的 RNN 狀態和累積器
            rnn_state.reset(done_ids)
            episode_reward[done_ids] = 0.0
            episode_step[done_ids] = 0
            if rsgs_filter is not None:
                rsgs_filter.reset(done_ids)

            # 新 episode 開始:
            # BehaviorScheduler.reset() 已由 env 內部事件系統觸發（randomize_obstacles event）
            # 不可再次呼叫，否則會在 robot 已 spawn 後覆寫 obstacle 位置導致重疊
            if _play_behavior_scheduler is None:
                place_obstacles_near_goal(done_ids)

        # --- 每 200 步印出進度摘要 ---
        if step % 200 == 0:
            env0_goal = obs_tensor[0, 4:6].tolist()
            print(f"  [步驟 {step}] 目標=({env0_goal[0]:.2f}, {env0_goal[1]:.2f}) 動作={actions[0].tolist()}")
            if args_cli.play_diag and diag_samples > 0:
                print(
                    f"    [診斷均值] 航向誤差={diag_heading_sum/diag_samples:.1f}° "
                    f"朝目標速度={diag_velocity_to_goal_sum/diag_samples:+.3f} "
                    f"目標距離={diag_goal_distance_sum/diag_samples:.2f} "
                    f"動作=({diag_action_linear_sum/diag_samples:.2f}, {diag_action_angular_sum/diag_samples:.2f})"
                )

        # --- LiDAR 可觀察性診斷（前 10 步 + 每 500 步）---
        if args_cli.diagnostic and (step < 10 or step % 500 == 0):
            print_lidar_diagnostic(raw_env, step)

        # --- BEV 視窗更新 ---
        _t0_bev = time.time()
        if bev_visualizer is not None and step % max(1, args_cli.bev_update_interval) == 0:
            if not bev_visualizer.update(step, obs_tensor, actions, rvo2_filter=rvo2_filter):
                print("[PLAY] BEV 視窗已關閉，停止 play。")
                break
        _t_bev = time.time() - _t0_bev

        obs = next_obs  # 推進觀測
        step += 1

        # --- 效能摘要累積 & 定期印出 ---
        _t_total = time.time() - start
        _t_other = max(0.0, _t_total - _t_inference - _t_env - _t_orca - _t_bev)
        _perf_accum["inference"] += _t_inference
        _perf_accum["env_step"] += _t_env
        _perf_accum["orca"] += _t_orca
        _perf_accum["bev"] += _t_bev
        _perf_accum["other"] += _t_other
        _perf_accum["total"] += _t_total
        _perf_count += 1
        if _perf_count % _PERF_INTERVAL == 0:
            n = _PERF_INTERVAL
            print(
                f"[PERF] avg over {n} steps: "
                f"inference={_perf_accum['inference']/n*1000:.1f}ms "
                f"env_step={_perf_accum['env_step']/n*1000:.1f}ms "
                f"orca={_perf_accum['orca']/n*1000:.1f}ms "
                f"bev={_perf_accum['bev']/n*1000:.1f}ms "
                f"other={_perf_accum['other']/n*1000:.1f}ms "
                f"total={_perf_accum['total']/n*1000:.1f}ms "
                f"({n/_perf_accum['total']:.1f} fps)"
            )
            _perf_accum = {k: 0.0 for k in _perf_accum}

        # 真實時間模式：插入 sleep 以模擬 dt
        if args_cli.real_time:
            elapsed = time.time() - start
            if elapsed < step_dt:
                time.sleep(step_dt - elapsed)

    # ================================================================
    # 11. Play 結束 — 印出統計摘要
    # ================================================================
    print("\n" + "=" * 60)
    print("PLAY 統計摘要")
    print("=" * 60)
    if stats_total > 0:
        sr = stats_goal / stats_total * 100        # 成功率 (Success Rate)
        cr = (stats_wall + stats_obs) / stats_total * 100  # 碰撞率 (Collision Rate)
        to = stats_timeout / stats_total * 100     # 超時率
        avg_steps = sum(stats_steps_list) / len(stats_steps_list)
        print(f"  總回合數: {stats_total}")
        print(f"  成功率 (到達目標):   {stats_goal:4d} ({sr:.1f}%)")
        print(f"  碰撞率 (撞牆):      {stats_wall:4d} ({stats_wall/stats_total*100:.1f}%)")
        print(f"  碰撞率 (撞障礙物):  {stats_obs:4d} ({stats_obs/stats_total*100:.1f}%)")
        print(f"  碰撞率 (總計):      {stats_wall+stats_obs:4d} ({cr:.1f}%)")
        print(f"  超時率:             {stats_timeout:4d} ({to:.1f}%)")
        print(f"  其他:               {stats_other:4d}")
        print(f"  平均步數/回合:      {avg_steps:.1f}")
    else:
        print("  未完成任何回合。")
    if args_cli.play_diag and diag_samples > 0:
        print("  導航診斷:")
        print(f"    航向誤差均值 (度):     {diag_heading_sum/diag_samples:.2f}")
        print(f"    朝目標速度均值:       {diag_velocity_to_goal_sum/diag_samples:+.4f}")
        print(f"    目標距離均值:         {diag_goal_distance_sum/diag_samples:.4f}")
        print(f"    線性動作 idx 均值:    {diag_action_linear_sum/diag_samples:.4f}")
        print(f"    角度動作 idx 均值:    {diag_action_angular_sum/diag_samples:.4f}")
    # --- RVO2 Safety Filter 統計 ---
    if rvo2_filter is not None:
        rvo2_filter.print_stats()
    # --- RSGS-Lite 統計 ---
    if rsgs_filter is not None:
        rsgs_filter.print_stats()

    # --- Stuck episode analysis ---
    if _all_ep_logs:
        print("\n" + "=" * 60)
        print("[VELOCITY LOG] 各回合速度摘要:")
        print("=" * 60)
        for ep in _all_ep_logs:
            flag = " ◀◀ STUCK" if ep["low_speed_ratio"] > 0.5 else ""
            print(
                f"  回合{ep['episode']:3d} [{ep['cause']:7s}] "
                f"步數={ep['steps']:3d} avg_speed={ep['avg_speed']:.3f} "
                f"low_ratio={ep['low_speed_ratio']:.0%}{flag}"
            )

        # Find worst stuck episode
        stuck_eps = [ep for ep in _all_ep_logs if ep["low_speed_ratio"] > 0.3 and ep["steps"] > 3]
        if stuck_eps:
            worst = max(stuck_eps, key=lambda e: e["low_speed_ratio"])
            print(f"\n{'='*60}")
            print(f"[STUCK DETAIL] 回合 {worst['episode']} — {worst['cause']}, "
                  f"{worst['steps']} 步, low_ratio={worst['low_speed_ratio']:.0%}")
            print(f"{'='*60}")
            print(f"{'step':>4s} {'speed':>6s} {'v_body':>6s} {'orca':>4s} {'rec':>3s} "
                  f"{'near_obs':>8s} {'agent_x':>8s} {'agent_y':>8s} "
                  f"{'goal_bx':>7s} {'goal_by':>7s} "
                  f"{'vpref_x':>7s} {'vpref_y':>7s} {'vsafe_x':>7s} {'vsafe_y':>7s}")
            for s in worst["log"]:
                print(
                    f"{s['step']:4d} {s['speed']:6.3f} {s['v_body_pref']:+6.3f} "
                    f"{'Y' if s['orca_active'] else 'N':>4s} "
                    f"{'Y' if s['recovery'] else 'N':>3s} "
                    f"{s['nearest_obs']:8.3f} "
                    f"{s['agent_pos'][0]:8.3f} {s['agent_pos'][1]:8.3f} "
                    f"{s['goal_body'][0]:7.3f} {s['goal_body'][1]:7.3f} "
                    f"{s['v_pref'][0]:+7.3f} {s['v_pref'][1]:+7.3f} "
                    f"{s['v_safe'][0]:+7.3f} {s['v_safe'][1]:+7.3f}"
                )

            # Also dump ALL stuck episodes if more than one
            other_stuck = [ep for ep in stuck_eps if ep["episode"] != worst["episode"]]
            for ep in other_stuck[:2]:
                print(f"\n--- 回合 {ep['episode']} ({ep['cause']}, {ep['steps']} 步, low={ep['low_speed_ratio']:.0%}) ---")
                for s in ep["log"]:
                    print(
                        f"{s['step']:4d} {s['speed']:6.3f} {s['v_body_pref']:+6.3f} "
                        f"{'Y' if s['orca_active'] else 'N':>4s} "
                        f"{s['nearest_obs']:8.3f} "
                        f"{s['agent_pos'][0]:8.3f} {s['agent_pos'][1]:8.3f} "
                        f"{s['goal_body'][0]:7.3f} {s['goal_body'][1]:7.3f}"
                    )

    print("=" * 60)

    # --- Per-step 行為診斷統計輸出 ---
    if hasattr(main, '_diag_data') and main._diag_data['lidar_min']:
        import numpy as np
        lidar_arr = np.concatenate(main._diag_data['lidar_min'])
        speed_arr = np.concatenate(main._diag_data['speed'])
        omega_arr = np.concatenate(main._diag_data['omega'])
        obs_dist_arr = np.concatenate(main._diag_data['obs_dist'])

        print("\n" + "=" * 60)
        print(f"[DIAG] Per-step 行為分析 ({len(lidar_arr)} samples)")
        print("=" * 60)

        # 反 normalize: speed_arr 是 normalized (÷v_max=1.0)，lidar 是 normalized (÷max_range≈18m)
        # obs_dist_arr 已是 actual meters (from BehaviorScheduler positions)
        speed_actual = speed_arr  # v_max=1.0, so normalized = actual m/s
        omega_actual = omega_arr * 1.5  # omega_max=1.5 rad/s
        lidar_actual = lidar_arr * 18.0  # max_range ≈ 18m (VLP16 config)

        print("\n[DIAG] === LiDAR 最近距離(m) vs 速度(m/s) ===")
        bins = [0, 1.0, 2.0, 3.0, 4.0, 6.0, 9.0, 18.0]
        for i in range(len(bins)-1):
            mask = (lidar_actual >= bins[i]) & (lidar_actual < bins[i+1])
            if mask.sum() > 0:
                print(f"  [{bins[i]:.0f}, {bins[i+1]:.0f})m: "
                      f"speed={speed_actual[mask].mean():.3f}±{speed_actual[mask].std():.3f} "
                      f"|ω|={omega_actual[mask].mean():.3f} "
                      f"(n={mask.sum()}, {mask.sum()/len(lidar_actual)*100:.1f}%)")

        print("\n[DIAG] === 最近障礙物距離(m) vs 速度(m/s) ===")
        obs_bins = [0, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0]
        for i in range(len(obs_bins)-1):
            mask = (obs_dist_arr >= obs_bins[i]) & (obs_dist_arr < obs_bins[i+1])
            if mask.sum() > 0:
                print(f"  [{obs_bins[i]:.1f}, {obs_bins[i+1]:.1f})m: "
                      f"speed={speed_actual[mask].mean():.3f}±{speed_actual[mask].std():.3f} "
                      f"|ω|={omega_actual[mask].mean():.3f} "
                      f"(n={mask.sum()}, {mask.sum()/len(obs_dist_arr)*100:.1f}%)")

        print("\n[DIAG] === 全局統計 ===")
        print(f"  Speed (m/s): mean={speed_actual.mean():.3f} std={speed_actual.std():.3f} max={speed_actual.max():.3f}")
        print(f"  |Omega| (rad/s): mean={np.abs(omega_actual).mean():.3f} std={np.abs(omega_actual).std():.3f}")
        print(f"  LiDAR min (m): mean={lidar_actual.mean():.2f} std={lidar_actual.std():.2f}")
        print(f"  Obs dist (m): mean={obs_dist_arr[obs_dist_arr<100].mean():.2f} std={obs_dist_arr[obs_dist_arr<100].std():.2f}")

    # 清理資源
    if bev_visualizer is not None:
        bev_visualizer.close()
    env.close()


if __name__ == "__main__":
    main()
