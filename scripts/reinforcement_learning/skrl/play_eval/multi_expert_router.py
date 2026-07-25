"""Multi-expert per-env router for charge navigation eval.

此模組提供一個「每步、每 env」的專家路由器：根據觀測（原始公尺 LiDAR）
決定每個 env 該用「走廊專家（corridor）」還是「窄縫專家（narrow）」的動作。

設計要點
--------
* **輸入用原始公尺 LiDAR**（非正規化）：router 的閾值以物理距離（公尺）表達，
  可讀、可調、跨 checkpoint 一致。呼叫端須傳入 *未經 obs_normalizer 正規化* 的
  LiDAR 前錐距離（obs_tensor[:, 6:78]，單位公尺）。
* **無狀態、純函式**：router 不維護跨步狀態；每步依當前觀測獨立決策。
  （若未來要加 hysteresis，可在呼叫端持有 last_choice 傳入。）
* **專家索引約定**：0 = corridor（走廊/開闊），1 = narrow（窄縫）。

啟發式邏輯（rule 模式）
-----------------------
窄縫的幾何特徵 = 車體兩側都貼近牆、但前方（正前錐）相對可通行，形成一條通道。
一般走廊/開闊場 = 側向淨空較大。因此：

    is_narrow = (left_clear < side_thresh) AND (right_clear < side_thresh)
                AND (front_clear > front_min)

其中 left/right/front clear 由 LiDAR 前/側錐的「最小距離」估計。
可用閾值調整靈敏度。always_corridor / always_narrow 模式供 parity 對照。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


# 專家索引約定（與載入順序一致）
EXPERT_CORRIDOR = 0
EXPERT_NARROW = 1

# LiDAR 幾何約定：72 bin 環繞一圈（360°），index 0 對齊車體正前方，順時針/逆時針
# 由訓練端決定。此處以「相對 bin 偏移」定義前/左/右錐，避免硬編死絕對方向，
# 呼叫端若已知確切對應可覆寫 front/left/right bin 集合。
LIDAR_BINS = 72
DEG_PER_BIN = 360.0 / LIDAR_BINS  # 5.0°/bin


@dataclass(frozen=True)
class RouterConfig:
    """Router 閾值設定（單位：公尺 / bin）。"""

    # 側向淨空門檻：左右兩側最近牆 < side_clear_thresh → 視為「被牆夾住」
    side_clear_thresh: float = 1.10
    # 前方最小可通行門檻：正前錐最近障礙 > front_min_clear → 前方有路
    front_min_clear: float = 0.55
    # 前錐半角（度）：正前方 ±front_cone_deg 內的 bin 當作「前方」
    front_cone_deg: float = 30.0
    # 側錐中心角（度）：左 = +side_center_deg，右 = -side_center_deg（或 360-）
    side_center_deg: float = 90.0
    # 側錐半角（度）
    side_cone_deg: float = 40.0
    # 前方最遠飽和：超過此距離的 bin 視為「無回波/開闊」，避免 inf 汙染
    max_range: float = 10.0


def _cone_mask(num_bins: int, center_deg: float, half_deg: float,
               device: torch.device) -> torch.Tensor:
    """回傳 [num_bins] bool mask：與 center_deg 的角度差 <= half_deg 的 bin 為 True。

    採環繞角度差（0..360 折疊到 0..180），對前/左/右錐皆正確。
    """
    idx = torch.arange(num_bins, device=device, dtype=torch.float32)
    bin_deg = idx * (360.0 / num_bins)
    diff = (bin_deg - center_deg).abs()
    diff = torch.minimum(diff, 360.0 - diff)  # 環繞折疊
    return diff <= half_deg


def _min_over_cone(lidar_m: torch.Tensor, mask: torch.Tensor,
                   max_range: float) -> torch.Tensor:
    """對每個 env，回傳 cone 內 LiDAR 最小距離（公尺）。

    Args:
        lidar_m: [B, num_bins] 原始公尺 LiDAR（前錐距離）。
        mask: [num_bins] bool cone mask。
        max_range: 無效/無回波值的上限（用來取代 <=0 或 nan）。
    """
    B = lidar_m.shape[0]
    # 清理：nan/inf/<=0 → max_range（視為無障礙）
    clean = lidar_m.clone()
    clean = torch.nan_to_num(clean, nan=max_range, posinf=max_range, neginf=max_range)
    clean = torch.where(clean <= 0.0, torch.full_like(clean, max_range), clean)
    clean = torch.clamp(clean, max=max_range)
    cone = clean[:, mask]  # [B, k]
    if cone.shape[1] == 0:
        return torch.full((B,), max_range, device=lidar_m.device, dtype=lidar_m.dtype)
    return cone.min(dim=1).values


def compute_clearances(lidar_m: torch.Tensor, cfg: RouterConfig):
    """從原始公尺 LiDAR 計算 front/left/right 最小淨空。

    Returns: (front_clear, left_clear, right_clear) 各為 [B] tensor（公尺）。
    """
    device = lidar_m.device
    num_bins = lidar_m.shape[-1]
    front_mask = _cone_mask(num_bins, 0.0, cfg.front_cone_deg, device)
    left_mask = _cone_mask(num_bins, cfg.side_center_deg, cfg.side_cone_deg, device)
    right_mask = _cone_mask(num_bins, 360.0 - cfg.side_center_deg, cfg.side_cone_deg, device)
    front = _min_over_cone(lidar_m, front_mask, cfg.max_range)
    left = _min_over_cone(lidar_m, left_mask, cfg.max_range)
    right = _min_over_cone(lidar_m, right_mask, cfg.max_range)
    return front, left, right


def route(lidar_m: torch.Tensor, mode: str = "rule",
          cfg: RouterConfig | None = None) -> torch.Tensor:
    """決定每個 env 的專家索引。

    Args:
        lidar_m: [B, num_bins] 原始公尺 LiDAR（未正規化前錐距離）。
        mode: "rule" | "always_corridor" | "always_narrow"。
        cfg: RouterConfig（rule 模式閾值）。

    Returns:
        [B] long tensor，值 ∈ {EXPERT_CORRIDOR(0), EXPERT_NARROW(1)}。
    """
    B = lidar_m.shape[0]
    device = lidar_m.device
    if mode == "always_corridor":
        return torch.full((B,), EXPERT_CORRIDOR, dtype=torch.long, device=device)
    if mode == "always_narrow":
        return torch.full((B,), EXPERT_NARROW, dtype=torch.long, device=device)
    if mode != "rule":
        raise ValueError(f"unknown router_mode: {mode!r}")

    cfg = cfg or RouterConfig()
    front, left, right = compute_clearances(lidar_m, cfg)
    pinched = (left < cfg.side_clear_thresh) & (right < cfg.side_clear_thresh)
    front_open = front > cfg.front_min_clear
    is_narrow = pinched & front_open
    choice = torch.where(
        is_narrow,
        torch.full((B,), EXPERT_NARROW, dtype=torch.long, device=device),
        torch.full((B,), EXPERT_CORRIDOR, dtype=torch.long, device=device),
    )
    return choice


def routing_stats(choice: torch.Tensor) -> dict:
    """回傳路由分布統計（供每步列印）。"""
    total = int(choice.numel())
    n_narrow = int((choice == EXPERT_NARROW).sum().item())
    n_corridor = total - n_narrow
    return {
        "total": total,
        "corridor": n_corridor,
        "narrow": n_narrow,
        "narrow_frac": (n_narrow / total) if total else 0.0,
    }
