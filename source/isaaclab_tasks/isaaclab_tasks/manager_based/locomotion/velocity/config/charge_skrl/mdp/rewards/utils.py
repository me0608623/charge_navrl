"""獎勵工具函數 — 診斷與異常偵測

函數:
    _print_diagnostics: 印出機器人狀態、感測器數值的詳細診斷
    _check_reward_term: 檢查獎勵值是否含 NaN/Inf

使用場景:
    在 debug 模式下，獎勵函數可在開頭呼叫 _check_reward_term
    來偵測異常值並自動印出完整環境狀態。

注意: 這些是 debug 工具，不影響訓練邏輯。
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _print_diagnostics(env: ManagerBasedRLEnv, env_id: int, reward_value: Optional[float] = None):
    """打印環境的詳細診斷信息
    
    Args:
        env: 環境實例
        env_id: 環境 ID
        reward_value: 異常的 reward 值（可選）
    """
    print(f"\n  [Diagnostics for env {env_id}]:")
    if reward_value is not None:
        print(f"    Reward value: {reward_value:.6e}")
    
    try:
        # 獲取機器人狀態
        asset = env.scene["robot"]
        robot_pos = asset.data.root_pos_w[env_id, :3].cpu()
        robot_vel = asset.data.root_lin_vel_w[env_id, :2].cpu()
        robot_ang_vel = asset.data.root_ang_vel_w[env_id, :].cpu()  # 全部角速度
        
        # 獲取目標位置
        goal_pos = env.command_manager.get_command("goal_command")[env_id, :3].cpu()
        goal_dist = torch.norm(goal_pos[:2] - robot_pos[:2]).item()
        
        # 獲取雷達數據
        sensor = env.scene.sensors["lidar"]
        sensor_pos = sensor.data.pos_w[env_id, :3].cpu()
        hit_points = sensor.data.ray_hits_w[env_id, :, :3].cpu()
        distances = torch.norm(hit_points - sensor_pos.unsqueeze(0), dim=-1)
        min_lidar_dist = distances[torch.isfinite(distances)].min().item() if torch.isfinite(distances).any() else float('inf')
        max_lidar_dist = distances[torch.isfinite(distances)].max().item() if torch.isfinite(distances).any() else float('inf')
        
        # 獲取動作
        if hasattr(env.action_manager, 'action'):
            action = env.action_manager.action[env_id, :].cpu()
        else:
            action = torch.zeros(2)
        
        # 檢查狀態中的 NaN/Inf
        pos_has_nan = torch.isnan(robot_pos).any().item() or torch.isinf(robot_pos).any().item()
        vel_has_nan = torch.isnan(robot_vel).any().item() or torch.isinf(robot_vel).any().item()
        goal_has_nan = torch.isnan(goal_pos).any().item() or torch.isinf(goal_pos).any().item()
        
        print(f"    Robot position: {robot_pos.tolist()} {'[HAS NaN/Inf!]' if pos_has_nan else ''}")
        print(f"    Robot velocity (XY): {robot_vel.tolist()} (|v|={torch.norm(robot_vel).item():.6f} m/s) {'[HAS NaN/Inf!]' if vel_has_nan else ''}")
        print(f"    Robot angular vel: {robot_ang_vel.tolist()} rad/s")
        print(f"    Goal position: {goal_pos.tolist()} {'[HAS NaN/Inf!]' if goal_has_nan else ''}")
        print(f"    Goal distance: {goal_dist:.6f} m")
        print(f"    Min lidar distance: {min_lidar_dist:.6f} m")
        print(f"    Max lidar distance: {max_lidar_dist:.6f} m")
        print(f"    Action: {action.tolist()}")
        print(f"    Episode length: {env.episode_length_buf[env_id].item()}")
        
        # 檢查是否有異常大的狀態值
        vel_mag = torch.norm(robot_vel).item()
        if vel_mag > 100.0:
            print(f"    ⚠️  WARNING: Robot velocity magnitude is very large: {vel_mag:.6f} m/s")
        if abs(goal_dist) > 100.0:
            print(f"    ⚠️  WARNING: Goal distance is very large: {goal_dist:.6f} m")
        if min_lidar_dist < 0.0 or not torch.isfinite(torch.tensor(min_lidar_dist)):
            print(f"    ⚠️  WARNING: Min lidar distance is invalid: {min_lidar_dist}")
    except Exception as e:
        print(f"    ⚠️  Error getting diagnostics: {e}")


def _check_reward_term(name: str, reward: torch.Tensor, env: ManagerBasedRLEnv, raise_on_error: bool = True) -> torch.Tensor:
    """檢查 reward term 是否包含 NaN/Inf 或異常值（用於定位 PPO std>=0 錯誤）
    
    Args:
        name: reward term 名稱（用於錯誤訊息）
        reward: reward 張量
        env: 環境實例（用於獲取診斷信息）
        raise_on_error: 如果發現異常是否立即拋出異常
    
    Returns:
        清理後的 reward 張量
    
    這個函數會：
    1. 檢查 NaN/Inf
    2. 檢查異常大的值（> 1e6 或 min/max 突然變大）
    3. 如果發現問題，打印詳細的診斷信息（機器人狀態、距離、速度、角速度、min lidar range、action 等）
    4. 嘗試清理異常值
    """
    # 計算統計值（用於檢查 min/max 是否異常）
    reward_finite = reward[torch.isfinite(reward)]
    if reward_finite.numel() > 0:
        min_val = reward_finite.min().item()
        max_val = reward_finite.max().item()
        mean_val = reward_finite.mean().item()
        std_val = reward_finite.std().item()
    else:
        min_val = float('nan')
        max_val = float('nan')
        mean_val = float('nan')
        std_val = float('nan')
    
    # 檢查 NaN/Inf
    if not torch.isfinite(reward).all():
        bad_mask = ~torch.isfinite(reward)
        bad_env_ids = bad_mask.nonzero(as_tuple=False).squeeze(-1).cpu().tolist()
        num_bad = bad_mask.sum().item()
        num_total = reward.numel()
        nan_count = torch.isnan(reward).sum().item()
        inf_count = torch.isinf(reward).sum().item()
        
        print(f"\n{'='*80}")
        print(f"[REWARD ERROR] {name}: Non-finite values detected!")
        print(f"{'='*80}")
        print(f"  Bad count: {num_bad}/{num_total}")
        print(f"  NaN count: {nan_count}")
        print(f"  Inf count: {inf_count}")
        print(f"  Bad env IDs: {bad_env_ids[:10]}..." if len(bad_env_ids) > 10 else f"  Bad env IDs: {bad_env_ids}")
        
        # 打印 reward 統計
        print(f"\n  [Reward statistics (finite values only)]:")
        print(f"    Min: {min_val:.6e}")
        print(f"    Max: {max_val:.6e}")
        print(f"    Mean: {mean_val:.6e}")
        print(f"    Std: {std_val:.6e}")
        
        # 打印診斷信息（只對第一個有問題的環境）
        if len(bad_env_ids) > 0:
            env_id = bad_env_ids[0]
            bad_value = reward[env_id].item()
            _print_diagnostics(env, env_id, reward_value=bad_value)
        
        if raise_on_error:
            raise RuntimeError(
                f"[{name}] Reward term contains non-finite values! "
                f"nan={nan_count} inf={inf_count} bad_envs={bad_env_ids[:5]}"
            )
        
        # 嘗試清理（如果允許）
        reward_cleaned = torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
        reward_cleaned = torch.clamp(reward_cleaned, -1000.0, 1000.0)
        return reward_cleaned
    
    # 檢查異常大的值（> 1e6 或 > 1e10）
    abs_reward = torch.abs(reward)
    
    # 檢查是否有 > 1e10 的值（極端異常）
    extreme_mask = abs_reward > 1e10
    if extreme_mask.any():
        extreme_env_ids = extreme_mask.nonzero(as_tuple=False).squeeze(-1).cpu().tolist()
        extreme_max_val = abs_reward.max().item()
        
        print(f"\n{'='*80}")
        print(f"[REWARD CRITICAL ERROR] {name}: EXTREMELY large values detected (> 1e10)!")
        print(f"{'='*80}")
        print(f"  Max absolute value: {extreme_max_val:.2e}")
        print(f"  Extreme env IDs: {extreme_env_ids[:10]}..." if len(extreme_env_ids) > 10 else f"  Extreme env IDs: {extreme_env_ids}")
        
        print(f"\n  [Reward statistics]:")
        print(f"    Min: {min_val:.6e}")
        print(f"    Max: {max_val:.6e}")
        print(f"    Mean: {mean_val:.6e}")
        print(f"    Std: {std_val:.6e}")
        
        # 打印詳細診斷信息
        if len(extreme_env_ids) > 0:
            env_id = extreme_env_ids[0]
            extreme_value = reward[env_id].item()
            _print_diagnostics(env, env_id, reward_value=extreme_value)
        
        if raise_on_error:
            raise RuntimeError(
                f"[{name}] Reward term contains EXTREMELY large values! "
                f"max={extreme_max_val:.2e} extreme_envs={extreme_env_ids[:5]}"
            )
        
        # 嘗試清理
        reward_cleaned = torch.clamp(reward, -1000.0, 1000.0)
        return reward_cleaned
    
    # 檢查是否有 > 1e6 的值（較大異常）
    large_mask = abs_reward > 1e6
    if large_mask.any():
        large_env_ids = large_mask.nonzero(as_tuple=False).squeeze(-1).cpu().tolist()
        large_max_val = abs_reward.max().item()
        
        print(f"\n{'='*80}")
        print(f"[REWARD WARNING] {name}: Very large values detected (> 1e6)!")
        print(f"{'='*80}")
        print(f"  Max absolute value: {large_max_val:.2e}")
        print(f"  Large env IDs: {large_env_ids[:10]}..." if len(large_env_ids) > 10 else f"  Large env IDs: {large_env_ids}")
        
        print(f"\n  [Reward statistics]:")
        print(f"    Min: {min_val:.6e}")
        print(f"    Max: {max_val:.6e}")
        print(f"    Mean: {mean_val:.6e}")
        print(f"    Std: {std_val:.6e}")
        
        # 打印詳細診斷信息
        if len(large_env_ids) > 0:
            env_id = large_env_ids[0]
            large_value = reward[env_id].item()
            _print_diagnostics(env, env_id, reward_value=large_value)
        
        if raise_on_error:
            raise RuntimeError(
                f"[{name}] Reward term contains very large values! "
                f"max={large_max_val:.2e} large_envs={large_env_ids[:5]}"
            )
        
        # 嘗試清理
        reward_cleaned = torch.clamp(reward, -1000.0, 1000.0)
        return reward_cleaned
    
    # 檢查 min/max 是否突然變得很大（即使沒有單個值 > 1e6，但 min/max 差距很大也可能有問題）
    if abs(min_val) > 1e4 or abs(max_val) > 1e4:
        print(f"\n{'='*80}")
        print(f"[REWARD WARNING] {name}: Min/Max values are suspiciously large!")
        print(f"{'='*80}")
        print(f"  Min: {min_val:.6e}")
        print(f"  Max: {max_val:.6e}")
        print(f"  Mean: {mean_val:.6e}")
        print(f"  Std: {std_val:.6e}")
        
        # 找出 min/max 對應的環境
        if abs(min_val) > 1e4:
            min_env_id = (reward == min_val).nonzero(as_tuple=False)[0, 0].item()
            _print_diagnostics(env, min_env_id, reward_value=min_val)
        if abs(max_val) > 1e4:
            max_env_id = (reward == max_val).nonzero(as_tuple=False)[0, 0].item()
            _print_diagnostics(env, max_env_id, reward_value=max_val)
        
        if raise_on_error:
            raise RuntimeError(
                f"[{name}] Reward term has suspiciously large min/max values! "
                f"min={min_val:.2e} max={max_val:.2e}"
            )
    
    return reward
