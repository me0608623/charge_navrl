# 自動生成的 VLP-16 噪聲參數 — 供 Isaac Lab Domain Randomization 使用
# 生成時間：2026-07-01T18:23:52.640426
# 資料來源：6 個量測檔案，材質：white_wall
# 偏差模式：fixed

# ===========================================================================
# 1. 距離偏移 (Bias) — 系統性偏差，加在每個測距值上
# ===========================================================================
LIDAR_BIAS_MEAN      = 0.014813    # m，全局平均偏差
LIDAR_BIAS_RANGE     = (-0.000476, 0.023619)  # m，per-ring 偏差範圍

# 距離相關偏差（若 type == "distance_dependent"，用線性公式）
LIDAR_BIAS_TYPE      = "fixed"
LIDAR_BIAS_SLOPE     = 0.002153    # bias = slope * distance + intercept
LIDAR_BIAS_INTERCEPT = 0.011045

# ===========================================================================
# 2. 高斯噪聲 (Noise) — 隨機誤差，每個測距值加 N(0, std)
# ===========================================================================
LIDAR_NOISE_STD_MEAN  = 0.008672    # m，全局平均標準差
LIDAR_NOISE_STD_RANGE = (0.008196, 0.009311)  # m，實測範圍

# 距離相關噪聲（std 隨距離變化的線性模型）
LIDAR_NOISE_SLOPE     = 0.000000   # std = slope * distance + intercept
LIDAR_NOISE_INTERCEPT = 0.008672
LIDAR_NOISE_R2        = 0.0780

# ===========================================================================
# 3. 點雲丟失 (Dropout) — 從真實硬體數據量測
# ===========================================================================
LIDAR_DROPOUT_RATE         = 0.194859       # 低反射率丟點率 (intensity < threshold)
LIDAR_MIXED_PIXEL_RATE     = 0.002515   # 混合像素率 (邊緣 outlier)
LIDAR_TOTAL_INVALID_RATE   = 0.197374  # 總無效點率

# ===========================================================================
# 4. Per-Ring 偏差（各光束通道的獨立偏差）
# ===========================================================================
LIDAR_PER_RING_BIAS = {
    0: 0.004389,  # ring 0, std=0.006032, avg 51 pts/frame
    1: 0.009268,  # ring 1, std=0.006501, avg 51 pts/frame
    2: 0.018511,  # ring 2, std=0.005804, avg 51 pts/frame
    3: 0.023619,  # ring 3, std=0.005440, avg 51 pts/frame
    4: 0.004456,  # ring 4, std=0.006266, avg 51 pts/frame
    5: 0.013587,  # ring 5, std=0.006336, avg 51 pts/frame
    6: 0.014694,  # ring 6, std=0.006245, avg 51 pts/frame
    7: 0.001912,  # ring 7, std=0.006114, avg 50 pts/frame
    8: 0.014077,  # ring 8, std=0.006121, avg 51 pts/frame
    9: 0.023182,  # ring 9, std=0.006233, avg 50 pts/frame
    10: 0.019564,  # ring 10, std=0.006134, avg 51 pts/frame
    11: 0.019096,  # ring 11, std=0.007418, avg 50 pts/frame
    12: 0.019010,  # ring 12, std=0.005612, avg 51 pts/frame
    13: 0.016875,  # ring 13, std=0.007243, avg 51 pts/frame
    14: -0.000476,  # ring 14, std=0.006361, avg 51 pts/frame
    15: 0.005355,  # ring 15, std=0.006411, avg 51 pts/frame
}
