"""PPO Internal Diagnostics — 真實 rollout 內部診斷

在 PPO _update 完成後，用更新後的 policy 重新 forward pass，
計算真實的 ratio / KL / clip_fraction / per-action advantage。

不修改 SKRL 內部程式碼，只在 _update 後攔截 memory 數據。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Optional


class PPOInternalDiagnostics:
    """攔截 PPO update 後的 memory，計算真實診斷指標。"""

    # Obs layout constants
    _LIDAR_START = 6
    _LIDAR_END = 78
    _OBS_START = 78
    _OBS_END = 138

    def __init__(self, ratio_clip: float = 0.2):
        self._ratio_clip = ratio_clip
        self._last_metrics: dict = {}
        self._call_count = 0

    def compute(
        self,
        agent,
        *,
        base_env=None,
        max_samples: int = 16384,
    ) -> dict:
        """PPO update 完成後呼叫。用更新後的 policy 重算 new_log_prob。

        Args:
            agent: SKRL PPO agent（已完成 _update，weights 已更新）
            base_env: unwrapped ManagerBasedRLEnv（可選，用於 env-side coverage 診斷）
            max_samples: 最多取多少樣本（避免 OOM）
        """
        try:
            mem = agent.memory
            old_log_prob = mem.get_tensor_by_name("log_prob")     # [T, 1]
            states = mem.get_tensor_by_name("states")             # [T, obs_dim]
            actions = mem.get_tensor_by_name("actions")           # [T, act_dim]
            advantages = mem.get_tensor_by_name("advantages")     # [T, 1]
            returns = mem.get_tensor_by_name("returns")           # [T, 1]
            values = mem.get_tensor_by_name("values")             # [T, 1]
            rewards = mem.get_tensor_by_name("rewards")           # [T, 1]

            if any(x is None for x in [old_log_prob, states, actions, advantages]):
                return {}

            # SKRL memory stores [rollouts, envs, dim] — flatten to [rollouts*envs, dim]
            if states.dim() == 3:
                R, E, D = states.shape
                states = states.reshape(R * E, D)
                old_log_prob = old_log_prob.reshape(R * E, -1)
                actions = actions.reshape(R * E, -1)
                advantages = advantages.reshape(R * E, -1)
                if returns is not None:
                    returns = returns.reshape(R * E, -1)
                if values is not None:
                    values = values.reshape(R * E, -1)
                if rewards is not None:
                    rewards = rewards.reshape(R * E, -1)

            T = states.shape[0]
            n = min(T, max_samples)
            # Random subsample if needed
            if n < T:
                idx = torch.randperm(T, device=states.device)[:n]
                old_lp = old_log_prob[idx]
                st = states[idx]
                act = actions[idx]
                adv = advantages[idx]
                ret = returns[idx] if returns is not None else None
                val = values[idx] if values is not None else None
                rew = rewards[idx] if rewards is not None else None
            else:
                old_lp = old_log_prob
                st = states
                act = actions
                adv = advantages
                ret = returns
                val = values
                rew = rewards

            m: dict = {}

            # ── Step 1: Real ratio / KL / clip_fraction ──
            with torch.no_grad():
                # Memory 中的 states 已經被 SKRL preprocessor 處理過
                # 不需要再做一次 CADN preprocessing
                st_proc = st

                _, new_lp, _ = agent.policy.act(
                    {"states": st_proc, "taken_actions": act}, role="policy"
                )

                log_ratio = (new_lp - old_lp).view(-1)
                ratio = torch.exp(log_ratio)

                # Approx KL (k3 estimator)
                kl = ((ratio - 1) - log_ratio).mean().clamp(0.0, 10.0)
                clip_frac = ((ratio - 1.0).abs() > self._ratio_clip).float().mean()

                m["diag/real_approx_kl"] = kl.item()
                m["diag/real_clip_fraction"] = clip_frac.item()
                m["diag/real_ratio_mean"] = ratio.mean().item()
                m["diag/real_ratio_std"] = ratio.std().item()
                m["diag/real_ratio_min"] = ratio.min().item()
                m["diag/real_ratio_max"] = ratio.max().item()
                m["diag/real_log_ratio_abs_mean"] = log_ratio.abs().mean().item()

            # ── Step 2: Per-action advantage / reward ──
            adv_flat = adv.view(-1)   # [n]
            accel_idx = act[:, 0].long()  # [n]

            fwd_mask = accel_idx > 10
            stop_mask = (accel_idx >= 8) & (accel_idx <= 10)
            ret_mask = accel_idx < 8

            for label, mask in [("fwd", fwd_mask), ("stop", stop_mask), ("retreat", ret_mask)]:
                cnt = mask.sum().item()
                m[f"diag/action_{label}_count"] = cnt
                if cnt > 0:
                    a = adv_flat[mask]
                    m[f"diag/adv_{label}_mean"] = a.mean().item()
                    m[f"diag/adv_{label}_std"] = a.std().item()
                    m[f"diag/adv_{label}_pos_ratio"] = (a > 0).float().mean().item()

                    if rew is not None:
                        r = rew.view(-1)[mask]
                        m[f"diag/rew_{label}_mean"] = r.mean().item()
                        m[f"diag/rew_{label}_std"] = r.std().item()

            # Global advantage stats
            m["diag/advantage_mean"] = adv_flat.mean().item()
            m["diag/advantage_std"] = adv_flat.std().item()
            m["diag/advantage_pos_ratio"] = (adv_flat > 0).float().mean().item()

            # ── Step 3: Context-conditioned analysis ──
            with torch.no_grad():
                # 用 raw states（CADN 前）取得 obs context
                lidar = st[:, self._LIDAR_START:self._LIDAR_END]    # [n, 72]
                obs_block = st[:, self._OBS_START:self._OBS_END]    # [n, 60]

                lidar_min = lidar.min(dim=-1).values                # [n]

                # Near obstacle: lidar_min < 0.1 (= ~2.35m raw)
                # Far obstacle: lidar_min > 0.3 (= ~6.35m raw)
                near_obs = lidar_min < 0.1
                far_obs = lidar_min > 0.3

                # Goal direction: goal in front (goal_x > 0) vs lateral (|goal_y| > |goal_x|)
                goal_x = st[:, 4]
                goal_y = st[:, 5]
                goal_front = goal_x > goal_y.abs()
                goal_lateral = ~goal_front

                # Obstacle density: count non-zero obstacle slots
                obs_reshaped = obs_block.view(-1, 10, 6)
                obs_norms = obs_reshaped[:, :, :2].norm(dim=-1)  # [n, 10]
                obs_active = (obs_norms > 0.01).sum(dim=-1)      # [n]
                dense = obs_active >= 7
                sparse = obs_active <= 3

                contexts = {
                    "near_obs": near_obs,
                    "far_obs": far_obs,
                    "goal_front": goal_front,
                    "goal_lateral": goal_lateral,
                    "dense": dense,
                    "sparse": sparse,
                }

                for ctx_name, ctx_mask in contexts.items():
                    ctx_n = ctx_mask.sum().item()
                    m[f"diag/ctx_{ctx_name}_count"] = ctx_n
                    if ctx_n < 50:
                        continue

                    ctx_adv = adv_flat[ctx_mask]
                    ctx_accel = accel_idx[ctx_mask]

                    for label, lo, hi in [("fwd", 11, 19), ("stop", 8, 11), ("retreat", 0, 8)]:
                        act_mask = (ctx_accel >= lo) & (ctx_accel < hi) if label != "fwd" else ctx_accel > 10
                        act_n = act_mask.sum().item()
                        if act_n > 10:
                            a = ctx_adv[act_mask]
                            m[f"diag/ctx_{ctx_name}_{label}_adv_mean"] = a.mean().item()
                            m[f"diag/ctx_{ctx_name}_{label}_adv_pos_ratio"] = (a > 0).float().mean().item()

                # ── Advantage sign flip quantification ──
                # For each context, check if forward advantage is positive in some
                # contexts but negative in others
                for label in ["fwd", "retreat"]:
                    means = []
                    for ctx_name in ["near_obs", "far_obs", "dense", "sparse"]:
                        key = f"diag/ctx_{ctx_name}_{label}_adv_mean"
                        if key in m:
                            means.append(m[key])
                    if len(means) >= 2:
                        has_pos = any(v > 0.05 for v in means)
                        has_neg = any(v < -0.05 for v in means)
                        m[f"diag/{label}_adv_sign_flip"] = 1.0 if (has_pos and has_neg) else 0.0

            # ── Step 4: Observation sufficiency ──
            with torch.no_grad():
                # LiDAR occupied ratio: bins with value < 0.5 (= ~10.35m raw)
                lidar_occupied = (lidar < 0.5).float().mean(dim=-1)  # [n]
                m["diag/lidar_occupied_ratio"] = lidar_occupied.mean().item()

                # LiDAR saturation: bins at 0 (= object at body radius)
                lidar_saturated = (lidar < 0.01).float().mean(dim=-1)
                m["diag/lidar_saturated_ratio"] = lidar_saturated.mean().item()

                # ── C2/C3: Top-10 Coverage 診斷 ──
                # obs_active = 從 observation 中偵測到的非零 slots（top-k 輸出）
                # 注意：top_k=10 是 sim-to-real 一致性約束（真機 MOT 上限 10）
                filled_slots = obs_active.float()
                m["diag/filled_slots_mean"] = filled_slots.mean().item()
                m["diag/filled_slots_std"] = filled_slots.std().item()
                m["diag/top10_fill_ratio"] = filled_slots.mean().item() / 10.0

                # 判斷 top-k 截斷是否發生：filled_slots == 10 代表可能有更多可見障礙物被截斷
                m["diag/topk_saturated_ratio"] = (filled_slots >= 10).float().mean().item()

                # ── C3: Env-side visible obstacle count (pre-topk) ──
                # 從 env 取得實際可見障礙物數量，區分「本來就少」vs「top-k 截斷」
                if base_env is not None:
                    try:
                        num_visible = getattr(base_env, '_env_num_visible_obstacles', None)
                        num_total_scene = getattr(base_env, '_num_obstacles', None)
                        if num_visible is not None:
                            vis = num_visible.float()
                            m["diag/visible_obstacle_count_mean"] = vis.mean().item()
                            m["diag/visible_obstacle_count_std"] = vis.std().item()
                            # C3 核心指標：可見障礙物 > 10 的 env 比例 = top-k 確實截斷
                            m["diag/visible_gt_topk_ratio"] = (vis > 10).float().mean().item()
                        if num_total_scene is not None:
                            m["diag/num_total_obstacles_in_scene"] = float(num_total_scene)
                    except Exception:
                        pass

            # ── Step 5: GAE / gamma diagnostics ──
            if ret is not None and val is not None:
                ret_flat = ret.view(-1)
                val_flat = val.view(-1)
                m["diag/return_mean"] = ret_flat.mean().item()
                m["diag/return_std"] = ret_flat.std().item()
                m["diag/value_mean"] = val_flat.mean().item()
                m["diag/value_std"] = val_flat.std().item()

                # Advantage before normalization ≈ returns - values
                adv_raw = ret_flat - val_flat
                adv_raw_std = adv_raw.std().item()
                m["diag/advantage_raw_mean"] = adv_raw.mean().item()
                m["diag/advantage_raw_std"] = adv_raw_std

                # ── Std floor diagnostics ──
                _ADV_MIN_STD = 0.1
                adv_std_after = max(adv_raw_std, _ADV_MIN_STD)
                m["diag/adv_std_before_clamp"] = adv_raw_std
                m["diag/adv_std_after_clamp"] = adv_std_after
                m["diag/adv_clamp_triggered"] = 1.0 if adv_raw_std < _ADV_MIN_STD else 0.0

                # Normalized advantage stats (what the actor actually sees)
                adv_norm = adv.view(-1)  # already normalized from memory
                m["diag/adv_norm_mean"] = adv_norm.mean().item()
                m["diag/adv_norm_std"] = adv_norm.std().item()

                # fwd - retreat advantage gap
                fwd_adv_m = m.get("diag/adv_fwd_mean", 0)
                ret_adv_m = m.get("diag/adv_retreat_mean", 0)
                m["diag/fwd_minus_retreat_adv"] = fwd_adv_m - ret_adv_m

                # Effective horizon from gamma
                # gamma is stored on agent
                gamma = getattr(agent, '_discount_factor', 0.998)
                lam = getattr(agent, '_lambda', 0.95)
                m["diag/gamma"] = gamma
                m["diag/lambda"] = lam
                m["diag/effective_horizon_gamma"] = 1.0 / (1.0 - gamma) if gamma < 1 else float('inf')
                m["diag/effective_horizon_gae"] = 1.0 / (1.0 - gamma * lam) if (gamma * lam) < 1 else float('inf')

            # ── Logit statistics ──
            with torch.no_grad():
                features = agent.policy.extractor(st_proc)
                logits_raw = agent.policy.head(features)
                lin_logits = logits_raw[:, :19]
                ang_logits = logits_raw[:, 19:]
                m["diag/logit_linear_std"] = lin_logits.std().item()
                m["diag/logit_angular_std"] = ang_logits.std().item()
                m["diag/logit_linear_range"] = (lin_logits.max() - lin_logits.min()).item()
                m["diag/logit_angular_range"] = (ang_logits.max() - ang_logits.min()).item()

            self._last_metrics = m
            self._call_count += 1

            # Console print on first 5 calls and every 20th
            if self._call_count <= 5 or self._call_count % 20 == 0:
                print(f"\n[PPODiag] ── Rollout #{self._call_count} ──", flush=True)
                print(f"  Real KL={m.get('diag/real_approx_kl',0):.6f}  "
                      f"Clip={m.get('diag/real_clip_fraction',0):.4f}  "
                      f"Ratio={m.get('diag/real_ratio_mean',0):.4f}±{m.get('diag/real_ratio_std',0):.4f} "
                      f"[{m.get('diag/real_ratio_min',0):.4f}, {m.get('diag/real_ratio_max',0):.4f}]",
                      flush=True)
                print(f"  Advantage: mean={m.get('diag/advantage_mean',0):.4f} "
                      f"std={m.get('diag/advantage_std',0):.4f} "
                      f"pos_ratio={m.get('diag/advantage_pos_ratio',0):.3f}",
                      flush=True)
                print(f"  Per-action adv:  "
                      f"Fwd={m.get('diag/adv_fwd_mean',0):+.4f}  "
                      f"Stop={m.get('diag/adv_stop_mean',0):+.4f}  "
                      f"Retreat={m.get('diag/adv_retreat_mean',0):+.4f}",
                      flush=True)
                print(f"  Per-action rew:  "
                      f"Fwd={m.get('diag/rew_fwd_mean',0):+.4f}  "
                      f"Stop={m.get('diag/rew_stop_mean',0):+.4f}  "
                      f"Retreat={m.get('diag/rew_retreat_mean',0):+.4f}",
                      flush=True)
                # Context-conditioned
                for ctx in ["near_obs", "far_obs", "dense", "sparse"]:
                    fwd_a = m.get(f"diag/ctx_{ctx}_fwd_adv_mean")
                    ret_a = m.get(f"diag/ctx_{ctx}_retreat_adv_mean")
                    if fwd_a is not None or ret_a is not None:
                        print(f"  {ctx:>10s}: fwd_adv={fwd_a:+.4f}" if fwd_a is not None else "", end="", flush=True)
                        print(f"  ret_adv={ret_a:+.4f}" if ret_a is not None else "", flush=True)
                # Sign flip
                fwd_flip = m.get("diag/fwd_adv_sign_flip")
                ret_flip = m.get("diag/retreat_adv_sign_flip")
                if fwd_flip is not None:
                    print(f"  Sign flip: fwd={fwd_flip:.0f} retreat={ret_flip:.0f}", flush=True)
                # Obs sufficiency (C2/C3)
                vis_mean = m.get('diag/visible_obstacle_count_mean', -1)
                vis_gt10 = m.get('diag/visible_gt_topk_ratio', -1)
                total_scene = m.get('diag/num_total_obstacles_in_scene', '?')
                print(f"  TopK fill: {m.get('diag/filled_slots_mean',0):.1f}/10  "
                      f"saturated={m.get('diag/topk_saturated_ratio',0):.3f}  "
                      f"visible_pre_topk={vis_mean:.1f}  "
                      f"visible>10={vis_gt10:.3f}  "
                      f"scene_total={total_scene}",
                      flush=True)
                print(f"  LiDAR occupied={m.get('diag/lidar_occupied_ratio',0):.3f}  "
                      f"saturated={m.get('diag/lidar_saturated_ratio',0):.4f}",
                      flush=True)
                # GAE
                print(f"  GAE: γ={m.get('diag/gamma',0):.4f} λ={m.get('diag/lambda',0):.3f} "
                      f"horizon_γ={m.get('diag/effective_horizon_gamma',0):.0f} "
                      f"horizon_GAE={m.get('diag/effective_horizon_gae',0):.0f}",
                      flush=True)
                print(f"  Advantage raw: mean={m.get('diag/advantage_raw_mean',0):.4f} "
                      f"std={m.get('diag/advantage_raw_std',0):.4f} "
                      f"clamp={'YES' if m.get('diag/adv_clamp_triggered',0) > 0 else 'no'} "
                      f"(before={m.get('diag/adv_std_before_clamp',0):.4f} "
                      f"after={m.get('diag/adv_std_after_clamp',0):.4f})",
                      flush=True)
                print(f"  fwd-retreat gap: {m.get('diag/fwd_minus_retreat_adv',0):+.4f}",
                      flush=True)
                print(f"  Logit std: lin={m.get('diag/logit_linear_std',0):.4f} "
                      f"ang={m.get('diag/logit_angular_std',0):.4f}",
                      flush=True)

            return m

        except Exception as e:
            import traceback
            print(f"[PPODiag] Error: {e}", flush=True)
            traceback.print_exc()
            return {}
