"""VLP-16 經驗雜訊注入 — 感測端量測 → Isaac Lab domain randomization。

資料來源：真實 VLP-16（rover 從機）對白牆 0.5–3.0m 實測，
σ 以「點到平面殘差」量得（非中位數 std，避免 √N 低估）。
用法見同資料夾 README.md。
"""
from __future__ import annotations
import importlib.util
import torch


def load_vlp16_params(py_path: str, device: str) -> dict:
    """讀 isaac_lab_noise_params.py → tensor 參數字典。"""
    spec = importlib.util.spec_from_file_location("vlp16_params", py_path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    per_ring = torch.zeros(16, dtype=torch.float32, device=device)
    for r, v in getattr(m, "LIDAR_PER_RING_BIAS", {}).items():
        per_ring[int(r)] = float(v)

    return {
        "sigma": float(m.LIDAR_NOISE_STD_MEAN),                 # 固定 σ（建議）
        "sigma_slope": float(getattr(m, "LIDAR_NOISE_SLOPE", 0.0)),
        "sigma_intercept": float(getattr(m, "LIDAR_NOISE_INTERCEPT", m.LIDAR_NOISE_STD_MEAN)),
        "forced_constant": bool(getattr(m, "LIDAR_NOISE_MODEL_FORCED_CONSTANT", True)),
        "global_bias": float(m.LIDAR_BIAS_MEAN),
        "bias_type": str(getattr(m, "LIDAR_BIAS_TYPE", "fixed")),
        "bias_slope": float(getattr(m, "LIDAR_BIAS_SLOPE", 0.0)),
        "bias_intercept": float(getattr(m, "LIDAR_BIAS_INTERCEPT", m.LIDAR_BIAS_MEAN)),
        "per_ring_bias": per_ring,                              # [16]
        "dropout_rate": float(m.LIDAR_DROPOUT_RATE),
        "mixed_pixel_rate": float(getattr(m, "LIDAR_MIXED_PIXEL_RATE", 0.0)),
    }


def apply_vlp16_noise(ranges: torch.Tensor, ring_idx: torch.Tensor, p: dict,
                      max_range: float, mode: str = "full") -> torch.Tensor:
    """對光達 ranges 注入經驗雜訊。

    ranges  : [N, B] 每條 ray 的距離（公尺）
    ring_idx: [B] long，每條 ray 對應的 ring(0..15)（見 README §RayCaster 對應）
    mode    : ablation 開關 — 'ideal'|'sigma'|'bias'|'dropout'|'full'
    回傳注入後的 [N, B]。
    """
    out = ranges
    add_sigma = mode in ("sigma", "full")
    add_bias = mode in ("bias", "full")
    add_drop = mode in ("dropout", "full")

    # 1) 隨機測距雜訊 N(0, σ)。σ 可距離相關；預設固定（實測 R² 低）。
    if add_sigma:
        if p["forced_constant"]:
            sigma = p["sigma"]
        else:
            sigma = (p["sigma_slope"] * out + p["sigma_intercept"]).clamp_(min=1e-4)
        out = out + torch.randn_like(out) * sigma

    # 2) 系統偏差：per-ring + 全域（或距離相關）
    if add_bias:
        if p["bias_type"] == "distance_dependent":
            gbias = p["bias_slope"] * out + p["bias_intercept"]
        else:
            gbias = p["global_bias"]
        out = out + p["per_ring_bias"][ring_idx].unsqueeze(0) + gbias

    # 3) dropout：以 Bernoulli 機率把 ray 設為 max_range（等同「無回波」）
    if add_drop and p["dropout_rate"] > 0.0:
        keep = torch.rand_like(out) > p["dropout_rate"]
        out = torch.where(keep, out, torch.full_like(out, max_range))

    return out.clamp_(min=0.0)
