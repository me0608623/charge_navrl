"""swept_arc.py — action-conditioned「預測行駛軌跡安全度」幾何（r_arc reward 核心）。

動機（2026-07-14 用戶提案）：global 距離稅在 SA4 幾乎一直啟動＝背景稅，無法教「往哪邊避」。
改用 action-conditioned swept-arc：拿 policy 看到的 72-beam LiDAR + 實際動作 (v,ω) 預測未來 ~2.7s
圓弧軌跡，算車體沿弧與障礙的最小間距 c_arc，penalty = -w·[max(0, c_safe-c_arc)/c_safe]²。
提供「直走會撞/左轉仍撞/右轉能清」的方向性梯度——這是 gap/膨脹/多幀/TTC速度稅都缺的那一味。

★兩條紅線（用戶定，務必守）：
  1. 只用 policy 實際看到的 72-beam LiDAR（body frame），不可偷用 world obstacle position。
  2. 用實際套到車上的 (v,ω)。action[0]=accel → v_next=clamp(v+a·dt)（積分）；action[1]=ω 直接命令。
     → 呼叫端負責把 accel 積成 v 再傳進來（本模組收 resulting v,ω）。

幾何約定（對齊 train_rnn_car_wdclip.py:3802）：
  beam i 角度 θᵢ = (i·5° − 180°)  → bin36=0°(正前)、bin54=+90°(左)、bin18=−90°(右)
  障礙點 Pⱼ = (dⱼ cosθⱼ, dⱼ sinθⱼ)   body frame，+x 前、+y 左
  ω>0 = 左轉(CCW)；差速輪弧：R=v/ω，中心 (0,R)，pose(t)=(R sin ωt, R(1−cos ωt)), φ=ωt
"""
from __future__ import annotations

import math
import torch

NUM_BEAMS = 72
_DEG = math.pi / 180.0


def policy_lidar_to_sensor_range(
    lidar_obs: torch.Tensor,
    *,
    max_range: float = 20.0,
    body_radius: float = 0.35,
) -> torch.Tensor:
    """Undo the policy LiDAR normalization and recover sensor-to-hit range in meters.

    ``wd_like_sweep_72`` exposes ``(sensor_range - body_radius) / max_range``.
    Swept-arc geometry needs the original sensor-to-hit range because it subtracts
    ``body_radius`` after measuring distance from the predicted robot center.
    """
    return lidar_obs * max_range + body_radius


def beam_angles(num_beams: int = NUM_BEAMS, device=None, dtype=torch.float32) -> torch.Tensor:
    """回傳 [num_beams] 每 beam 角度(rad)。對齊訓練:θᵢ=(i·5°−180°)。bin36=正前。"""
    i = torch.arange(num_beams, device=device, dtype=dtype)
    return (i * 5.0 - 180.0) * _DEG


def _arc_poses(v: torch.Tensor, omega: torch.Tensor, t: torch.Tensor,
               omega_eps: float = 1e-3) -> tuple[torch.Tensor, torch.Tensor]:
    """差速輪弧軌跡。v,omega:[B]  t:[T] → 回傳 x,y 各 [B,T]（body frame，起點原點朝 +x）。

    直線(|ω|<eps): x=v·t, y=0。曲線: R=v/ω, x=R sin(ωt), y=R(1−cos ωt)。
    """
    v = v.unsqueeze(-1)          # [B,1]
    omega = omega.unsqueeze(-1)  # [B,1]
    t = t.unsqueeze(0)           # [1,T]
    straight = omega.abs() < omega_eps
    # 曲線公式（用 safe omega 避免除零，直線 case 後面覆蓋）
    omega_safe = torch.where(straight, torch.ones_like(omega), omega)
    R = v / omega_safe
    ang = omega_safe * t                                  # [B,T]
    x_curve = R * torch.sin(ang)
    y_curve = R * (1.0 - torch.cos(ang))
    x_straight = v * t
    y_straight = torch.zeros_like(x_straight)
    x = torch.where(straight, x_straight, x_curve)
    y = torch.where(straight, y_straight, y_curve)
    return x, y


def arc_clearance(
    lidar_m: torch.Tensor,      # [B, num_beams] LiDAR sensor-to-hit range (m), not policy clearance
    v: torch.Tensor,            # [B] resulting forward speed (m/s, 已由 accel 積分)
    omega: torch.Tensor,        # [B] angular vel (rad/s)
    *,
    angles: torch.Tensor | None = None,   # [num_beams] beam 角度; None→beam_angles()
    horizon: float = 2.7,       # 預測秒數（v≈0.75 下 2m/0.75≈2.67s → 2m 開始反應）
    pred_dt: float = 0.1,       # 軌跡離散步長
    body_radius: float = 0.35,  # 車體半徑
    max_range: float = 20.0,    # LiDAR 上限(m)；≥此視為無回波
    hole_thresh_m: float = 0.4, # <此(m)視為破洞/自反射，忽略（對齊 metrics 的 <0.02 norm≈0.4m）
    chunk: int = 0,             # >0 時對 B 分塊算（省記憶體）
) -> torch.Tensor:
    """回傳 [B] 沿預測弧的最小間距 c_arc(m)。無有效障礙→+inf。

    c_arc = min_j min_t ( ||pose(t) − Pⱼ|| ) − body_radius
    """
    B, K = lidar_m.shape
    device = lidar_m.device
    if angles is None:
        angles = beam_angles(K, device=device, dtype=lidar_m.dtype)
    # 障礙點（body frame）
    px = lidar_m * torch.cos(angles).unsqueeze(0)   # [B,K]
    py = lidar_m * torch.sin(angles).unsqueeze(0)
    # 有效 beam mask：非破洞、非超範圍
    valid = (lidar_m > hole_thresh_m) & (lidar_m < max_range)  # [B,K]

    T = max(1, int(round(horizon / pred_dt)))
    t = torch.arange(1, T + 1, device=device, dtype=lidar_m.dtype) * pred_dt  # [T]（t=0 不含,起點自身）

    def _one(sl):
        _px, _py, _v, _w, _valid = px[sl], py[sl], v[sl], omega[sl], valid[sl]
        rx, ry = _arc_poses(_v, _w, t)                 # [b,T]
        # dist: [b,T,K]
        dx = rx.unsqueeze(-1) - _px.unsqueeze(1)       # [b,T,K]
        dy = ry.unsqueeze(-1) - _py.unsqueeze(1)
        d = torch.sqrt(dx * dx + dy * dy)              # [b,T,K]
        # 無效 beam → +inf 不參與 min
        d = torch.where(_valid.unsqueeze(1), d, torch.full_like(d, float("inf")))
        cmin = d.amin(dim=(1, 2))                      # [b] min over t & beam
        return cmin - body_radius

    if chunk and chunk < B:
        outs = []
        for s in range(0, B, chunk):
            outs.append(_one(slice(s, min(s + chunk, B))))
        c = torch.cat(outs, dim=0)
    else:
        c = _one(slice(0, B))
    return c


def r_arc_from_clearance(
    c_arc: torch.Tensor,        # [B] 最小間距(m)，可含 +inf
    *,
    c_safe: float = 0.5,        # 車體邊緣安全間距門檻(m)：c_arc<c_safe 才開始扣
    w_arc: float = 0.08,        # 危險一步的量級（-0.05~-0.10）
    cap: float = 0.10,          # penalty 上限（絕對值）
) -> torch.Tensor:
    """r_arc = -w·[max(0, c_safe−c_arc)/c_safe]²，clamp 到 [-cap, 0]。無障礙(inf)→0。"""
    c = torch.nan_to_num(c_arc, posinf=1e6)
    deficit = (c_safe - c).clamp(min=0.0) / max(c_safe, 1e-6)   # [0,1+]
    pen = w_arc * deficit.pow(2)
    return (-pen).clamp(min=-cap, max=0.0)


def integrate_speed(v_now: torch.Tensor, accel: torch.Tensor, dt: float,
                    v_max: float = 1.0, reverse_scale: float = 1.0) -> torch.Tensor:
    """紅線②：把 accel 積成 resulting speed。v_next=clamp(v+a·dt, -v_max·rev, +v_max)。"""
    return (v_now + accel * dt).clamp(-v_max * reverse_scale, v_max)
