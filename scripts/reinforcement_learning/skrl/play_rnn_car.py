#!/usr/bin/env python3
"""
Play a trained modular-RNN charge checkpoint in Isaac Lab GUI.

Defaults:
  - loads the newest checkpoint under logs/rnn_car
  - uses 1 environment
  - fixes the environment to a single curriculum stage for inspection
"""

import argparse
import glob
import os
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Play trained modular RNN charge checkpoint.")
parser.add_argument("--task", type=str, default="Isaac-Navigation-Charge-VLP16-Curriculum-NavRL")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint_{steps}.pt")
parser.add_argument("--stage", type=int, default=1, help="Fixed curriculum stage to inspect")
parser.add_argument("--curriculum_version", type=str, default=None,
                    help="Override curriculum version. Default: task config default.")
parser.add_argument("--steps", type=int, default=3000, help="Maximum play steps")
parser.add_argument("--camera", type=str, default="top", choices=["top", "follow", "side"])
parser.add_argument("--deterministic", action="store_true", default=False,
                    help="Use argmax instead of sampling")
parser.add_argument("--real_time", action="store_true", default=False,
                    help="Sleep to approximate environment step dt")
parser.add_argument("--reward_mode", type=str, default="current",
                    choices=["current", "navrl_ground_v1", "navrl_ground_v2", "navrl_ground_v3",
                             "navrl_ground_v4", "navrl_ground_v5", "navrl_ground_v6",
                             "navrl_ground_v7", "navrl_ground_v8"])
parser.add_argument("--v_gate_mode", type=str, default="baseline",
                    choices=["baseline", "floor", "softer"])
parser.add_argument("--progress_gate_mode", type=str, default="baseline",
                    choices=["baseline", "delayed_negative", "weaken_negative"])
parser.add_argument("--use_gap_reward", action="store_true", default=False)
parser.add_argument("--gap_reward_type", type=str, default="heading",
                    choices=["heading", "clearance", "both"])
parser.add_argument("--gap_reward_weight", type=float, default=5.0)
parser.add_argument("--directional_gate", action="store_true", default=False)
parser.add_argument("--gate_cone_half_bins", type=int, default=6)
parser.add_argument("--gate_cone_bottom_k", type=int, default=3)
parser.add_argument("--gate_omni_blend", type=float, default=0.2)
parser.add_argument("--use_safety_shield", action="store_true", default=False)
parser.add_argument("--use_vo_shield", action="store_true", default=False,
                    help="Enable VO predictive shield after discrete action decoding")
parser.add_argument("--scripted_obstacles", action="store_true", default=False,
                    help="Allow scripted dynamic obstacle motion during play instead of freezing it")
parser.add_argument("--shield_mode", type=str, default="soft", choices=["soft", "hard"])
parser.add_argument("--vo_horizon", type=float, default=1.0,
                    help="VO prediction horizon in seconds")
parser.add_argument("--vo_safety_radius", type=float, default=0.45,
                    help="VO collision/safety radius in meters")
parser.add_argument("--vo_evade_gain", type=float, default=1.0,
                    help="VO angular evasion gain")
parser.add_argument("--dynamic_safety_mode", type=str, default="log_distance",
                    choices=["log_distance", "closing_risk"])
parser.add_argument("--goal_vel_gate_beta", type=float, default=0.2)
parser.add_argument("--progress_scale_gamma", type=float, default=0.3)
parser.add_argument("--w_goal", type=float, default=500.0)
parser.add_argument("--w_vel", type=float, default=10.0)
parser.add_argument("--w_prog", type=float, default=12.0)
parser.add_argument("--w_ss", type=float, default=3.0)
parser.add_argument("--w_ds", type=float, default=4.0)
parser.add_argument("--w_smooth", type=float, default=-0.05)
parser.add_argument("--w_time", type=float, default=-0.1)
parser.add_argument("--w_collision", type=float, default=-100.0)
parser.add_argument("--w_alive", type=float, default=0.2)
parser.add_argument("--goal_vel_use_soft_gate", action="store_true", default=False)
parser.add_argument("--no_walls", action="store_true", default=False)
parser.add_argument("--lidar_no_noise", action="store_true", default=False)
parser.add_argument("--lidar_vis", action="store_true", default=True,
                    help="Enable LiDAR raycaster debug visualization (default: on)")
parser.add_argument("--no_lidar_vis", action="store_true", default=False,
                    help="Disable LiDAR raycaster debug visualization")
parser.add_argument("--lidar_sanity", action="store_true", default=False,
                    help="Run LiDAR geometry sanity test after env init, then exit")
parser.add_argument("--diagnostic", action="store_true", default=False,
                    help="Print LiDAR observability diagnostics for first 10 steps")
parser.add_argument("--aux_debug", action="store_true", default=False,
                    help="Print RNN aux prediction vs simulator-derived nearest-obstacle target")
parser.add_argument("--aux_debug_interval", type=int, default=25,
                    help="Print aux debug every N play steps")
parser.add_argument("--bev_vis", action="store_true", default=False,
                    help="Show a live top-down BEV window of the 72-bin LiDAR sweep during play")
parser.add_argument("--bev_update_interval", type=int, default=2,
                    help="Update BEV visualization every N play steps")
parser.add_argument("--bev_max_range", type=float, default=20.0,
                    help="Maximum range in meters for the BEV visualization")
parser.add_argument("--bev_frame", type=str, default="body", choices=["body", "world"],
                    help="BEV frame: body keeps 0 deg as robot front; world keeps world axes fixed")
parser.add_argument("--no_domain_randomization", action="store_true", default=False)
parser.add_argument("--reward_speed_v05", action="store_true", default=False)
parser.add_argument("--ds_weight_boost", type=float, default=1.0)
parser.add_argument("--risk_sigma", type=float, default=None)
parser.add_argument("--b_risk", type=float, default=None)
parser.add_argument("--ss_lower_mode", type=str, default=None, choices=["aggressive", "moderate"])
parser.add_argument("--ss_raise_mode", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if not getattr(args_cli, "headless", False):
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import torch
from torch.distributions import Categorical

from isaaclab.envs import ViewerCfg

import isaaclab_tasks  # noqa: F401

sys.path.insert(0, str(Path(__file__).parent))
from charge_env_overrides import apply_charge_env_overrides
from modular_rnn_models import LidarStateExtractor, PolicyHead, PreprocessRNN, RNNStateManager, ValueHead
from wd_aux_targets import build_wd_preprocess_targets


POLICY_OBS_INDICES = list(range(0, 78)) + [138]


def sample_action(logits: torch.Tensor, deterministic: bool) -> torch.Tensor:
    logits_a, logits_w = logits[:, :19], logits[:, 19:]
    if deterministic:
        act_a = logits_a.argmax(dim=-1)
        act_w = logits_w.argmax(dim=-1)
    else:
        act_a = Categorical(logits=logits_a).sample()
        act_w = Categorical(logits=logits_w).sample()
    return torch.stack([act_a, act_w], dim=-1)


def find_latest_rnn_checkpoint() -> str | None:
    candidates = glob.glob("logs/rnn_car/*/checkpoint_*.pt")
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return os.path.abspath(candidates[0])


def resolve_env_cfg(task: str):
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
    raise ValueError(f"Unsupported task for play_rnn_car.py: {task}")


def apply_stage_to_env_cfg(env_cfg, curriculum_version: str | None, stage: int):
    cur = getattr(env_cfg, "curriculum", None)
    term = getattr(cur, "goal_obstacle_curriculum", None) if cur is not None else None
    if term is None:
        return None, None

    if curriculum_version is not None:
        term.params["curriculum_version"] = curriculum_version
    resolved_version = term.params.get("curriculum_version", "baseline_v1")

    from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_skrl.curriculum.goal_obstacle_curriculum import (
        _load_stages,
    )

    stages, max_stage, _ = _load_stages(resolved_version)
    stage = max(1, min(stage, max_stage))
    cfg = stages[stage]

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

    wall_evt = getattr(env_cfg.events, "randomize_wall_positions", None)
    if wall_evt is not None:
        wall_evt.params["min_walls"] = cfg["min_walls"]
        wall_evt.params["max_walls"] = cfg["max_walls"]

    env_cfg.commands.goal_command.num_goals = cfg["num_goals"]
    env_cfg.commands.goal_command.num_obstacles = cfg["num_obstacles_static"] + cfg["num_obstacles_dynamic"]
    env_cfg.commands.goal_command.ranges.distance = cfg["goal_distance"]
    env_cfg.episode_length_s = cfg["episode_length_s"]
    env_cfg.curriculum = None
    return resolved_version, cfg


def configure_camera(env_cfg, camera: str):
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
    else:
        env_cfg.viewer = ViewerCfg(
            eye=(15.0, -15.0, 15.0), lookat=(0.0, 0.0, 0.0),
            origin_type="env", env_index=0, resolution=(1920, 1080),
        )


def run_lidar_sanity_test(env, raw_env):
    """LiDAR Ray Origin Bug verification — confirms z_filter works with per-ray origins."""
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
        r_min=0.5, z_filter=0.5,
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
    """Lightweight per-step LiDAR diagnostic using per-ray origin."""
    import math as _math
    from isaaclab.managers import SceneEntityCfg
    from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_skrl.mdp.observations.obs_functions import (
        wd_like_sweep_72,
    )
    sensor = raw_env.scene.sensors["lidar"]
    hit_points = sensor.data.ray_hits_w  # [N, R, 3]

    # Per-ray origin (includes OffsetCfg z=1.6)
    if hasattr(sensor, "_ray_starts_w") and sensor._ray_starts_w is not None:
        ray_starts = sensor._ray_starts_w  # [N, R, 3]
        ray_z = ray_starts[0, 0, 2].item()
    else:
        ray_starts = sensor.data.pos_w.unsqueeze(1).expand_as(hit_points)
        ray_z = sensor.data.pos_w[0, 2].item()

    rel_hits = hit_points[0] - ray_starts[0]  # [R, 3]
    dist_2d = torch.linalg.norm(rel_hits[:, :2], dim=-1)
    rel_z = rel_hits[:, 2]
    finite = torch.isfinite(dist_2d) & torch.isfinite(rel_z)
    valid = finite & (dist_2d < 19.5)
    n_ground = (finite & (rel_z < -1.0)).sum().item()
    n_zf_pass = (finite & (rel_z.abs() <= 0.5)).sum().item()

    sensor_cfg = SceneEntityCfg("lidar")
    sensor_cfg.resolve(raw_env.scene)
    sweep = wd_like_sweep_72(raw_env, sensor_cfg, num_bins=72, r_max=20.0, r_robot=0.3, r_min=0.5, z_filter=0.5)
    real_dist = sweep[0] * 20.0

    ground_echo_dist = 1.6 / _math.tan(_math.radians(15.0)) - 0.3
    n_echo = ((real_dist >= ground_echo_dist - 0.3) & (real_dist <= ground_echo_dist + 0.3)).sum().item()

    near_wall = real_dist.min().item()
    near_bin = real_dist.argmin().item()
    near_angle = -180.0 + near_bin * 5.0

    print(f"  [DIAG step={step}] ray_z={ray_z:.2f}m | "
          f"valid={valid.sum().item()}/{hit_points.shape[1]} "
          f"gnd={n_ground} zf_pass={n_zf_pass} | "
          f"72bin: min={near_wall:.2f}m @{near_bin}({near_angle:+.0f}°) "
          f"mean={real_dist.mean().item():.2f} "
          f"echo5.6={n_echo} <2m={int((real_dist < 2.0).sum())} "
          f"max={int((real_dist >= 19.5).sum())}")


def print_aux_debug(raw_env, step: int, obs_tensor: torch.Tensor, aux_pred: torch.Tensor, max_obstacles: int):
    """Print RNN aux prediction against the simulator target for env 0."""
    from isaaclab.managers import SceneEntityCfg
    from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_skrl.mdp.observations.obs_functions import (
        wd_like_sweep_72,
    )

    target = build_wd_preprocess_targets(
        raw_env,
        max_obstacles=max_obstacles,
        device=aux_pred.device,
    )

    pred0 = aux_pred[0].detach()
    tgt0 = target[0].detach()
    err0 = pred0 - tgt0

    def _fmt_triplet(v):
        return f"x={v[0].item():+6.2f} y={v[1].item():+6.2f} d={v[2].item():5.2f}"

    sensor_cfg = SceneEntityCfg("lidar")
    sensor_cfg.resolve(raw_env.scene)
    sweep = wd_like_sweep_72(
        raw_env,
        sensor_cfg,
        num_bins=72,
        r_max=20.0,
        r_robot=0.3,
        r_min=0.5,
        z_filter=0.5,
    )
    lidar_m = sweep[0] * 20.0
    near_bin = int(lidar_m.argmin().item())
    near_angle = -180.0 + near_bin * 5.0

    obs_lidar = obs_tensor[0, 6:78].detach()
    obs_near_bin = int(obs_lidar.argmin().item())
    obs_near_angle = -180.0 + obs_near_bin * 5.0
    # obs_lidar is normalized by env observation config: distance / 20.0.
    obs_near_m = obs_lidar[obs_near_bin].item() * 20.0

    print(f"\n[AUX DEBUG step={step}] env0 RNN prediction vs simulator target")
    print("  near1 pred:  " + _fmt_triplet(pred0[0:3]))
    print("  near1 real:  " + _fmt_triplet(tgt0[0:3]))
    print("  near1 error: " + _fmt_triplet(err0[0:3]))
    print("  near2 pred:  " + _fmt_triplet(pred0[3:6]))
    print("  near2 real:  " + _fmt_triplet(tgt0[3:6]))
    print("  near2 error: " + _fmt_triplet(err0[3:6]))
    print(f"  timestep pred={pred0[6].item():.2f} real={tgt0[6].item():.0f} "
          f"(loss weight is 0 in training)")
    print(f"  lidar from obs[6:78]: min={obs_near_m:.2f}m "
          f"@local_bin={obs_near_bin} global_bin={obs_near_bin + 6} ({obs_near_angle:+.0f}deg)")
    print(f"  lidar recomputed:     min={lidar_m[near_bin].item():.2f}m "
          f"@bin={near_bin} ({near_angle:+.0f}deg)\n")


class LiveBEVVisualizer:
    """Small Matplotlib BEV window for inspecting the exact 72-bin sweep used by RL."""

    def __init__(self, raw_env, max_range: float = 20.0, frame: str = "body"):
        import matplotlib
        import numpy as np
        from isaaclab.managers import SceneEntityCfg
        from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_skrl.mdp.observations.obs_functions import (
            wd_like_sweep_72,
        )

        # The env OpenCV build has GUI disabled, so use Matplotlib/Tk for the live window.
        if "agg" in matplotlib.get_backend().lower():
            matplotlib.use("TkAgg", force=True)
        import matplotlib.pyplot as plt

        self.plt = plt
        self.np = np
        self.raw_env = raw_env
        self.max_range = float(max_range)
        self.frame = frame
        self.wd_like_sweep_72 = wd_like_sweep_72
        self.sensor_cfg = SceneEntityCfg("lidar")
        self.sensor_cfg.resolve(raw_env.scene)
        self.ground_echo_dist = 1.6 / np.tan(np.radians(15.0)) - 0.3
        self.goal_body_radius = 0.35
        self.goal_threshold = 0.35
        self.goal_success_center_dist = self.goal_body_radius + self.goal_threshold
        self.plt.ion()
        self.fig, self.ax = self.plt.subplots(figsize=(7, 7))
        try:
            self.fig.canvas.manager.set_window_title("Charge RL BEV - 72-bin LiDAR")
        except Exception:
            pass

    def _sensor_yaw(self) -> float:
        """Return env-0 sensor yaw in world frame. Isaac Lab quaternions are wxyz."""
        q = self.raw_env.scene.sensors["lidar"].data.quat_w[0].detach().cpu().numpy()
        w, x, y, z = [float(v) for v in q]
        return float(self.np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))

    def _body_to_world(self, x_forward, y_left, yaw: float):
        cos_y = self.np.cos(yaw)
        sin_y = self.np.sin(yaw)
        x_world = cos_y * x_forward - sin_y * y_left
        y_world = sin_y * x_forward + cos_y * y_left
        return x_world, y_world

    def _world_rel_to_plot(self, rel_w, yaw: float):
        if self.frame == "world":
            return rel_w[..., 0], rel_w[..., 1]
        cos_y = self.np.cos(yaw)
        sin_y = self.np.sin(yaw)
        x_forward = cos_y * rel_w[..., 0] + sin_y * rel_w[..., 1]
        y_left = -sin_y * rel_w[..., 0] + cos_y * rel_w[..., 1]
        return y_left, x_forward

    def update(self, step: int, obs_tensor: torch.Tensor, actions: torch.Tensor):
        np = self.np
        sweep = self.wd_like_sweep_72(
            self.raw_env,
            self.sensor_cfg,
            num_bins=72,
            r_max=self.max_range,
            r_robot=0.3,
            r_min=0.5,
            z_filter=0.5,
        )
        real_dist = (sweep[0] * self.max_range).detach().cpu().numpy()
        angles_deg = -180.0 + np.arange(real_dist.shape[0]) * 5.0
        angles = np.radians(angles_deg)
        x_forward = real_dist * np.cos(angles)
        y_left = real_dist * np.sin(angles)
        yaw = self._sensor_yaw()

        if self.frame == "world":
            plot_x, plot_y = self._body_to_world(x_forward, y_left, yaw)
            xlabel = "world X relative to robot (m)"
            ylabel = "world Y relative to robot (m)"
            title_suffix = "world frame"
        else:
            # Body-frame display: x-axis is left/right, y-axis is forward. 0 deg is always robot front.
            plot_x, plot_y = y_left, x_forward
            xlabel = "left/right y (m)"
            ylabel = "forward x (m)"
            title_suffix = "body frame: 0 deg = robot front"

        if not self.plt.fignum_exists(self.fig.number):
            return False

        ax = self.ax
        ax.cla()
        ax.set_facecolor("#141414")
        self.fig.patch.set_facecolor("#141414")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-self.max_range, self.max_range)
        ax.set_ylim(-self.max_range, self.max_range)
        ax.set_xlabel(xlabel, color="white")
        ax.set_ylabel(ylabel, color="white")
        ax.tick_params(colors="white")
        ax.grid(True, color="#383838", linewidth=0.7)
        ax.set_title(f"Charge RL BEV - 72-bin LiDAR ({title_suffix})", color="white")

        for r_m in [2, 5, 10, 15, 20]:
            circle = self.plt.Circle((0, 0), r_m, fill=False, color="#4a4a4a", linewidth=0.8)
            ax.add_patch(circle)
            ax.text(0.2, r_m, f"{r_m}m", color="#888888", fontsize=8)

        ax.plot(plot_x, plot_y, color="#7fbf7f", linewidth=1.2, alpha=0.8)

        colors = np.full(real_dist.shape, "#50dc50", dtype=object)
        colors[real_dist < 5.0] = "#ffa500"
        colors[real_dist < 2.0] = "#ff3030"
        colors[real_dist >= self.max_range - 0.5] = "#808080"
        sizes = np.full(real_dist.shape, 22.0)
        sizes[real_dist < 5.0] = 36.0
        sizes[real_dist < 2.0] = 54.0
        ax.scatter(plot_x, plot_y, c=colors.tolist(), s=sizes, zorder=3)

        # Robot footprint and forward direction.
        ax.add_patch(self.plt.Circle((0, 0), 0.3, fill=False, color="white", linewidth=2.0, zorder=4))
        if self.frame == "world":
            head_x = 1.2 * np.cos(yaw)
            head_y = 1.2 * np.sin(yaw)
        else:
            head_x, head_y = 0.0, 1.2
        ax.arrow(0, 0, head_x, head_y, color="white", width=0.04, head_width=0.35, length_includes_head=True, zorder=5)
        ax.scatter([0], [0], c="white", s=28, marker="o", edgecolors="black", linewidths=0.6, zorder=7)

        self._draw_goals(ax, yaw)
        goal_info = self._draw_termination_goal(ax, yaw)

        # Goal vector from the same observation slice already printed by play.py.
        if obs_tensor is not None and obs_tensor.shape[-1] >= 6:
            goal_xy = obs_tensor[0, 4:6].detach().cpu().numpy()
            gx = float(np.clip(goal_xy[0], -self.max_range, self.max_range))
            gy = float(np.clip(goal_xy[1], -self.max_range, self.max_range))
            if self.frame == "world":
                goal_x, goal_y = self._body_to_world(gx, gy, yaw)
            else:
                goal_x, goal_y = gy, gx
            ax.arrow(0, 0, goal_x, goal_y, color="#ffd040", width=0.035, head_width=0.45, length_includes_head=True, zorder=4)
            ax.scatter([goal_x], [goal_y], c=["#ffd040"], s=80, zorder=5)

        near_idx = int(real_dist.argmin())
        near_d = float(real_dist[near_idx])
        near_angle = -180.0 + near_idx * 5.0
        echo_bins = int(((real_dist >= self.ground_echo_dist - 0.3) & (real_dist <= self.ground_echo_dist + 0.3)).sum())
        action_text = actions[0].detach().cpu().tolist() if actions is not None else ["?", "?"]
        goal_line = "termination goal: unavailable"
        if goal_info is not None:
            status = "SUCCESS" if goal_info["success"] else "not-yet"
            goal_line = (
                f"term_goal[{goal_info['source']}] center={goal_info['center_dist']:.2f}m "
                f"edge={goal_info['effective_dist']:.2f}m<{self.goal_threshold:.2f}? {status}"
            )
            if goal_info.get("idx") is not None:
                goal_line += f" idx={goal_info['idx']}"
        text_lines = [
            f"step={step} action={action_text}",
            f"frame={self.frame}  yaw={np.degrees(yaw):+.1f} deg",
            goal_line,
            f"nearest: {near_d:.2f}m @ bin {near_idx} ({near_angle:+.0f} deg)",
            f"mean={real_dist.mean():.2f}m  <2m={(real_dist < 2.0).sum()}/72  2-5m={((real_dist >= 2.0) & (real_dist < 5.0)).sum()}/72",
            f"near floor echo 5.6m={echo_bins}/72  max-range={(real_dist >= self.max_range - 0.5).sum()}/72",
            "white=robot/front+center  yellow=command goal  magenta=termination target",
        ]
        ax.text(
            0.02, 0.98, "\n".join(text_lines),
            transform=ax.transAxes,
            va="top",
            ha="left",
            color="white",
            fontsize=9,
            bbox={"facecolor": "#202020", "edgecolor": "#606060", "alpha": 0.85},
        )
        self.fig.canvas.draw_idle()
        self.plt.pause(0.001)
        return self.plt.fignum_exists(self.fig.number)

    def close(self):
        if self.plt.fignum_exists(self.fig.number):
            self.plt.close(self.fig)

    def _selected_goal_index(self):
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
        """Draw all MultiGoalCommand goals and the goal-reach success disk."""
        try:
            cmd = self.raw_env.command_manager.get_term("goal_command")
            if not hasattr(cmd, "all_goals_pos_w"):
                return
            ng = int(cmd.cfg.num_goals)
            goals_w = cmd.all_goals_pos_w[0, :ng, :2].detach().cpu().numpy()
            robot_w = self.raw_env.scene["robot"].data.root_pos_w[0, :2].detach().cpu().numpy()
        except Exception:
            return

        rel_w = goals_w - robot_w[None, :]
        goal_plot_x, goal_plot_y = self._world_rel_to_plot(rel_w, yaw)

        selected_idx = self._selected_goal_index()
        ax.scatter(goal_plot_x, goal_plot_y, c="#40d8ff", s=38, marker="x", linewidths=1.2, zorder=4)
        for i, (gx, gy) in enumerate(zip(goal_plot_x, goal_plot_y)):
            ax.text(gx + 0.12, gy + 0.12, str(i), color="#40d8ff", fontsize=7, zorder=4)

        if selected_idx is not None and 0 <= selected_idx < len(goal_plot_x):
            sx = goal_plot_x[selected_idx]
            sy = goal_plot_y[selected_idx]
            ax.scatter([sx], [sy], c="#ffd040", s=120, marker="*", edgecolors="black", linewidths=0.8, zorder=6)

    def _draw_termination_goal(self, ax, yaw: float):
        """Draw the exact target used by goal_reached() and return its distance stats."""
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
    # LiDAR sanity test mode doesn't need a checkpoint
    if args_cli.lidar_sanity:
        ckpt_path = None
        _ckpt_args = {}
    else:
        ckpt_path = os.path.abspath(args_cli.checkpoint) if args_cli.checkpoint else find_latest_rnn_checkpoint()
        if ckpt_path is None or not os.path.exists(ckpt_path):
            raise FileNotFoundError("No RNN checkpoint found under logs/rnn_car and no --checkpoint provided")

        # Pre-read checkpoint args for env config alignment
        _ckpt_meta = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        _ckpt_args = _ckpt_meta.get("args", {})
        del _ckpt_meta  # free memory, full load happens later on correct device

    # Auto-read curriculum_version from checkpoint if not explicitly set
    if args_cli.curriculum_version is None:
        saved_cv = _ckpt_args.get("curriculum_version")
        if saved_cv:
            print(f"[PLAY] auto-setting --curriculum_version={saved_cv} (from checkpoint)")
            args_cli.curriculum_version = saved_cv

    # Auto-read lidar_no_noise from checkpoint
    if _ckpt_args.get("lidar_no_noise", False) and not args_cli.lidar_no_noise:
        print("[PLAY] auto-enabling --lidar_no_noise (from checkpoint)")
        args_cli.lidar_no_noise = True

    env_cfg = resolve_env_cfg(args_cli.task)
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    apply_charge_env_overrides(env_cfg, args_cli)
    resolved_curriculum, stage_cfg = apply_stage_to_env_cfg(env_cfg, args_cli.curriculum_version, args_cli.stage)
    configure_camera(env_cfg, args_cli.camera)

    # Enable LiDAR raycaster visualization for play mode
    lidar_vis = not args_cli.no_lidar_vis
    if hasattr(env_cfg.scene, "lidar"):
        env_cfg.scene.lidar.debug_vis = lidar_vis

    if not args_cli.headless:
        env_cfg.sim.render_interval = 4

    print(f"[PLAY] checkpoint: {ckpt_path}")
    print(f"[PLAY] task: {args_cli.task}")
    if stage_cfg is not None:
        print(
            f"[PLAY] curriculum={resolved_curriculum} stage={args_cli.stage} "
            f"name={stage_cfg['name']} goals={stage_cfg['num_goals']} "
            f"obs={stage_cfg['num_obstacles_static']}S+{stage_cfg['num_obstacles_dynamic']}D "
            f"walls={stage_cfg['min_walls']}-{stage_cfg['max_walls']}"
        )

    env = gym.make(args_cli.task, cfg=env_cfg)
    raw_env = env.unwrapped
    obs, _ = env.reset()
    device = raw_env.device

    # Run LiDAR sanity test if requested
    if args_cli.lidar_sanity:
        run_lidar_sanity_test(env, raw_env)
        env.close()
        return

    def policy_obs(x):
        if isinstance(x, dict):
            return x["policy"]
        return x

    obs_tensor = policy_obs(obs)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})
    hidden_dim = int(ckpt_args.get("hidden_dim", 30))
    preprocess_dim = int(ckpt_args.get("preprocess_dim", 12))
    fc_dim = int(ckpt_args.get("fc_dim", 48))
    rnn_type = ckpt_args.get("rnn_type", "RNN")
    encoder_mode = ckpt_args.get("charge_encoder_mode", "extractor_rnn")
    zero_preprocess = ckpt_args.get("zero_preprocess_feature_for_rl", False)
    max_active_obstacles = int(ckpt_args.get("max_active_obstacles", 10))
    if zero_preprocess:
        print("[PLAY] --zero_preprocess_feature_for_rl active: RNN features zeroed for RL heads")

    # Build models based on encoder mode
    use_extractor = (encoder_mode != "raw_fc_rnn")
    if use_extractor:
        extractor = LidarStateExtractor().to(device)
        rnn_input_dim = 96  # extractor output
    else:
        extractor = None
        rnn_input_dim = 79  # raw obs directly

    preprocess_rnn = PreprocessRNN(
        input_dim=rnn_input_dim,
        hidden_dim=hidden_dim,
        preprocess_dim=preprocess_dim,
        fc_dim=fc_dim,
        rnn_type=rnn_type,
    ).to(device)
    policy_head = PolicyHead(input_dim=79 + preprocess_dim).to(device)
    value_head = ValueHead(input_dim=79 + preprocess_dim).to(device)

    if use_extractor:
        extractor.load_state_dict(ckpt["extractor"])
        extractor.eval()
    preprocess_rnn.load_state_dict(ckpt["preprocess_rnn"])
    policy_head.load_state_dict(ckpt["policy_head"])
    value_head.load_state_dict(ckpt["value_head"])
    preprocess_rnn.eval()
    policy_head.eval()
    value_head.eval()
    print(f"[PLAY] encoder_mode={encoder_mode} rnn_input_dim={rnn_input_dim} hidden={hidden_dim} preprocess={preprocess_dim}")

    obs_norm = ckpt.get("obs_normalizer", {})
    mean = obs_norm.get("mean", torch.zeros(obs_tensor.shape[-1], device=device))
    var = obs_norm.get("var", torch.ones(obs_tensor.shape[-1], device=device))
    rnn_state = RNNStateManager(raw_env.num_envs, hidden_dim, device)

    # Align obstacle behavior with training by default: disable scripted motion.
    # For VO inspection, --scripted_obstacles lets interval events move dynamic obstacles.
    raw_env._obstacle_policy_active = not args_cli.scripted_obstacles
    print(
        "[PLAY] obstacle motion: "
        + ("scripted interval events enabled" if args_cli.scripted_obstacles else "scripted interval events disabled")
    )

    step_dt = env.step_dt if hasattr(env, "step_dt") else raw_env.step_dt
    episode_reward = torch.zeros(raw_env.num_envs, device=device)
    episode_step = torch.zeros(raw_env.num_envs, dtype=torch.long, device=device)

    # Episode statistics
    stats_goal = 0
    stats_wall = 0
    stats_obs = 0
    stats_timeout = 0
    stats_other = 0
    stats_total = 0
    stats_steps_list = []
    bev_visualizer = None
    if args_cli.bev_vis:
        if args_cli.num_envs != 1:
            print("[PLAY] --bev_vis shows env 0 only; using num_envs > 1 is allowed but less readable.")
        try:
            bev_visualizer = LiveBEVVisualizer(raw_env, max_range=args_cli.bev_max_range, frame=args_cli.bev_frame)
            print(f"[PLAY] BEV visualization enabled (frame={args_cli.bev_frame}). Close the BEV window to stop play.")
        except Exception as exc:
            print(f"[PLAY] WARNING: failed to initialize BEV visualization: {exc}")
            bev_visualizer = None

    def normalize(x):
        return torch.clamp((x - mean) / (var.sqrt() + 1e-8), -5.0, 5.0)

    def detect_termination_cause(env_unwrapped, terminated_flat, truncated_flat):
        """Detect per-env termination cause from termination_manager buffers."""
        N = terminated_flat.shape[0]
        device = terminated_flat.device
        # 0=running, 1=goal, 2=wall, 3=obstacle, 4=timeout, 5=other
        cause = torch.zeros(N, dtype=torch.long, device=device)
        try:
            tm = env_unwrapped.termination_manager
            for name in tm._term_names:
                buf = tm.get_term(name)
                if buf is None:
                    continue
                fired = buf.bool()
                if "goal_reached" in name or "reaching_goal" in name:
                    cause[fired & (cause == 0)] = 1
                elif "wall_collision" in name:
                    cause[fired & (cause == 0)] = 2
                elif "obstacle_collision" in name:
                    cause[fired & (cause == 0)] = 3
                elif "collision" in name:
                    cause[fired & (cause == 0)] = 3
                elif "tipped" in name or "explosion" in name or "flying" in name:
                    cause[fired & (cause == 0)] = 5
        except (AttributeError, RuntimeError):
            pass
        # Timeout
        trunc = truncated_flat.bool() if truncated_flat.dim() == 1 else truncated_flat.squeeze(-1).bool()
        cause[trunc & (cause == 0)] = 4
        return cause

    CAUSE_NAMES = {0: "running", 1: "GOAL", 2: "WALL", 3: "OBS_HIT", 4: "TIMEOUT", 5: "OTHER"}

    step = 0
    while simulation_app.is_running() and step < args_cli.steps:
        start = time.time()
        with torch.inference_mode():
            obs_tensor = policy_obs(obs)
            obs_normed = normalize(obs_tensor)
            p_obs = obs_normed[:, POLICY_OBS_INDICES]  # 79D slice
            if use_extractor:
                features = extractor(obs_normed)
            else:
                features = p_obs  # raw_fc_rnn: 79D directly to RNN
            hidden = rnn_state.get()
            rnn_feat, aux_pred, new_hidden = preprocess_rnn(features, hidden, training=args_cli.aux_debug)
            rnn_for_rl = torch.zeros_like(rnn_feat) if zero_preprocess else rnn_feat
            rl_in = torch.cat([p_obs, rnn_for_rl], dim=-1)
            logits = policy_head(rl_in)
            actions = sample_action(logits, args_cli.deterministic)

            if args_cli.aux_debug and aux_pred is not None and step % max(1, args_cli.aux_debug_interval) == 0:
                print_aux_debug(raw_env, step, obs_tensor, aux_pred, max_active_obstacles)

        # `inference_mode()` creates inference tensors that cannot be modified
        # in-place later during per-env hidden-state resets.
        new_hidden = new_hidden.clone()

        next_obs, reward, terminated, truncated, info = env.step(actions.float())
        done = (terminated.squeeze(-1) | truncated.squeeze(-1)) if terminated.ndim > 1 else (terminated | truncated)

        episode_reward += reward.squeeze(-1) if reward.ndim > 1 else reward
        episode_step += 1
        rnn_state.update(new_hidden)

        if done.any():
            done_ids = done.nonzero(as_tuple=False).reshape(-1)
            terminated_flat = terminated.squeeze(-1) if terminated.ndim > 1 else terminated
            truncated_flat = truncated.squeeze(-1) if truncated.ndim > 1 else truncated
            cause = detect_termination_cause(raw_env, terminated_flat, truncated_flat)

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

            rnn_state.reset(done_ids)
            episode_reward[done_ids] = 0.0
            episode_step[done_ids] = 0

        if step % 200 == 0:
            env0_goal = obs_tensor[0, 4:6].tolist()
            print(f"  [步驟 {step}] 目標=({env0_goal[0]:.2f}, {env0_goal[1]:.2f}) 動作={actions[0].tolist()}")

        # LiDAR observability diagnostic (first 10 steps or every 500)
        if args_cli.diagnostic and (step < 10 or step % 500 == 0):
            print_lidar_diagnostic(raw_env, step)

        if bev_visualizer is not None and step % max(1, args_cli.bev_update_interval) == 0:
            if not bev_visualizer.update(step, obs_tensor, actions):
                print("[PLAY] BEV window requested stop.")
                break

        obs = next_obs
        step += 1

        if args_cli.real_time:
            elapsed = time.time() - start
            if elapsed < step_dt:
                time.sleep(step_dt - elapsed)

    # Final statistics
    print("\n" + "=" * 60)
    print("PLAY SUMMARY")
    print("=" * 60)
    if stats_total > 0:
        sr = stats_goal / stats_total * 100
        cr = (stats_wall + stats_obs) / stats_total * 100
        to = stats_timeout / stats_total * 100
        avg_steps = sum(stats_steps_list) / len(stats_steps_list)
        print(f"  Episodes: {stats_total}")
        print(f"  SR (goal):     {stats_goal:4d} ({sr:.1f}%)")
        print(f"  CR (wall):     {stats_wall:4d} ({stats_wall/stats_total*100:.1f}%)")
        print(f"  CR (obstacle): {stats_obs:4d} ({stats_obs/stats_total*100:.1f}%)")
        print(f"  CR (total):    {stats_wall+stats_obs:4d} ({cr:.1f}%)")
        print(f"  TO (timeout):  {stats_timeout:4d} ({to:.1f}%)")
        print(f"  Other:         {stats_other:4d}")
        print(f"  Avg steps/ep:  {avg_steps:.1f}")
    else:
        print("  No episodes completed.")
    print("=" * 60)

    if bev_visualizer is not None:
        bev_visualizer.close()
    env.close()


if __name__ == "__main__":
    main()
