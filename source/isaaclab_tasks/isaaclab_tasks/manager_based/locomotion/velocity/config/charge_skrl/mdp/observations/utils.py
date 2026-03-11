"""觀測工具函數 — NaN/Inf 偵測

函數:
    check_finite(name, tensor, raise_on_error=True):
        檢查觀測張量是否含 NaN/Inf。

背景: 2026-03-07 訓練中發現物理爆炸產生的 NaN 速度會
通過觀測管線進入 RunningStandardScaler，永久污染統計量。
此函數用於在觀測層攔截異常值。
"""

from __future__ import annotations

import torch


def check_finite(name: str, x: torch.Tensor, raise_on_error: bool = True) -> bool:
    """檢查張量是否包含 NaN/Inf 值(用於定位 PPO std>=0 錯誤)
    
    Args:
        name: 觀測項名稱(用於錯誤訊息)
        x: 要檢查的張量
        raise_on_error: 如果發現 NaN/Inf 是否立即拋出異常
    
    Returns:
        True 如果所有值都是有限的,False 如果有 NaN/Inf
    
    這個函數是修復 PPO std>=0 錯誤的關鍵工具。
    當 policy 的 std 變成 NaN/Inf 時,通常是因為觀測或獎勵中出現了異常值。
    """
    if not torch.isfinite(x).all():
        bad = ~torch.isfinite(x)
        num_bad = bad.sum().item()
        num_total = x.numel()
        nan_count = torch.isnan(x).sum().item()
        inf_count = torch.isinf(x).sum().item()
        
        # 計算統計值(用 NaN/Inf 安全的方式)
        x_safe = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        min_val = x_safe.min().item()
        max_val = x_safe.max().item()
        
        error_msg = (
            f"[NaN/Inf Check] {name}: "
            f"shape={tuple(x.shape)} "
            f"bad_count={num_bad}/{num_total} "
            f"nan={nan_count} inf={inf_count} "
            f"min={min_val:.6f} max={max_val:.6f}"
        )
        
        if raise_on_error:
            raise RuntimeError(error_msg)
        else:
            print(f"WARNING: {error_msg}")
        
        return False
    return True