"""方向性長方形（OBB）碰撞的純幾何函式。

背景：真車完整 footprint 是長方形（含輪寬 0.60m × 長 0.70m），但既有碰撞判定把車當各向同性圓
（半徑 0.35m + buffer 0.10 = 0.45m 各向同性），導致 < 0.90m 的窄縫被誤判會撞而拒走。
本模組提供方向性判定，讓車頭對齊時可通過 0.85m 窄縫、真正碰撞仍會終止。

★純 torch、無 Isaac Sim 依賴 → 可離線單元測試（見 test_obb_collision.py）。
termination wrapper（robot_state.py）負責從 env 取出 robot 位置/yaw、牆/障礙資料後呼叫本模組。

座標約定
- yaw：車體前向（+車長方向）相對世界 +x 的角度 [rad]。
- 車體半尺寸：half_len（沿前向）、half_wid（沿側向）；buffer 為安全裕量（膨脹車體）。
- offset_x：碰撞盒中心沿車頭方向相對車體原點（root_pos）的偏移 [m]，負=盒中心偏後。
  OBB 幾何中心 = robot_xy + offset_x · (cos yaw, sin yaw)。
"""
from __future__ import annotations

import torch
from torch import Tensor


def relative_points_in_obb_frame(
    delta_xy: Tensor,
    cos_yaw: Tensor,
    sin_yaw: Tensor,
    offset_x: float,
) -> tuple[Tensor, Tensor]:
    """Convert root-relative points to the offset OBB frame.

    ``delta_xy`` is [N, M, 2]; trigonometric inputs are [N, 1]. Keeping this
    primitive separate lets the runtime wrapper reuse precomputed sin/cos.
    """
    dx = delta_xy[..., 0] - offset_x * cos_yaw
    dy = delta_xy[..., 1] - offset_x * sin_yaw
    lx = dx * cos_yaw + dy * sin_yaw
    ly = -dx * sin_yaw + dy * cos_yaw
    return lx, ly


def obb_aabb_collision(
    robot_xy: Tensor,       # [N, 2] 車中心（與 box_centers 同座標系）
    yaw: Tensor,            # [N] 車頭朝向 [rad]
    box_centers: Tensor,    # [N, W, 2] 各牆 AABB 中心
    box_sizes: Tensor,      # [N, W, 2] 各牆 AABB 全尺寸 (size_x, size_y)
    box_mask: Tensor,       # [N, W] bool，True = 該牆啟用
    half_len: float,
    half_wid: float,
    buffer: float,
    offset_x: float = 0.0,  # 碰撞盒中心沿車頭方向的偏移 [m]
) -> Tensor:
    """方向性長方形（車，膨脹 buffer）對軸對齊盒（牆）的碰撞。

    用分離軸定理（SAT）：2D 下檢查 4 條軸（世界 x/y + 車體前向/側向）。
    任一軸投影不重疊 → 分離（不撞）；4 軸全重疊 → 碰撞。

    Returns: [N] bool，True = 與任一啟用牆碰撞。
    """
    c = torch.cos(yaw)                       # [N]
    s = torch.sin(yaw)
    hl = half_len + buffer                   # 車體半尺寸（含 buffer）
    hw = half_wid + buffer

    # OBB 幾何中心（把 root 沿車頭方向偏移 offset_x）
    obb_c = robot_xy + offset_x * torch.stack([c, s], dim=-1)  # [N, 2]
    d = box_centers - obb_c.unsqueeze(1)     # [N, W, 2]
    dx = d[..., 0]                           # [N, W]
    dy = d[..., 1]
    ex = box_sizes[..., 0] * 0.5             # 牆半尺寸 [N, W]
    ey = box_sizes[..., 1] * 0.5

    cc = c.unsqueeze(1)                      # [N, 1]
    ss = s.unsqueeze(1)
    ac = cc.abs()
    as_ = ss.abs()

    # 軸 = 世界 x：車投影半徑 = hl|c| + hw|s|，牆 = ex，中心投影 = |dx|
    sep_x = dx.abs() > (hl * ac + hw * as_ + ex)
    # 軸 = 世界 y：車 = hl|s| + hw|c|，牆 = ey，中心投影 = |dy|
    sep_y = dy.abs() > (hl * as_ + hw * ac + ey)
    # 軸 = 車體前向 u=(c,s)：車 = hl，牆 = ex|c| + ey|s|，中心投影 = |dx*c + dy*s|
    sep_u = (dx * cc + dy * ss).abs() > (hl + ex * ac + ey * as_)
    # 軸 = 車體側向 v=(-s,c)：車 = hw，牆 = ex|s| + ey|c|，中心投影 = |-dx*s + dy*c|
    sep_v = (-dx * ss + dy * cc).abs() > (hw + ex * as_ + ey * ac)

    separated = sep_x | sep_y | sep_u | sep_v      # [N, W]
    collide = (~separated) & box_mask              # 只計啟用牆
    return collide.any(dim=1)                      # [N]


def obb_circle_collision(
    robot_xy: Tensor,       # [N, 2] 車中心
    yaw: Tensor,            # [N] 車頭朝向 [rad]
    circ_centers: Tensor,   # [N, M, 2] 各障礙圓心
    circ_radii: Tensor,     # [N, M] 各障礙物理半徑
    valid_mask: Tensor,     # [N, M] bool，True = 該障礙有效（可見）
    half_len: float,
    half_wid: float,
    buffer: float,
    offset_x: float = 0.0,  # 碰撞盒中心沿車頭方向的偏移 [m]
) -> Tensor:
    """方向性長方形（車）對圓（障礙物理半徑）的碰撞。

    把障礙圓心轉進車體座標系，夾到長方形邊界求最近點，
    最近點到圓心距離 < 障礙半徑 + buffer → 碰撞。

    Returns: [N] bool，True = 與任一有效障礙碰撞。
    """
    c = torch.cos(yaw).unsqueeze(1)          # [N, 1]
    s = torch.sin(yaw).unsqueeze(1)

    # Convert root-relative obstacle centers into the offset OBB frame.
    d = circ_centers - robot_xy.unsqueeze(1)
    lx, ly = relative_points_in_obb_frame(d, c, s, offset_x)

    # 長方形 [±half_len, ±half_wid] 上離圓心最近點
    cx = lx.clamp(-half_len, half_len)
    cy = ly.clamp(-half_wid, half_wid)

    dist = torch.sqrt((lx - cx) ** 2 + (ly - cy) ** 2)   # [N, M]
    collide = (dist < (circ_radii + buffer)) & valid_mask
    return collide.any(dim=1)                            # [N]
