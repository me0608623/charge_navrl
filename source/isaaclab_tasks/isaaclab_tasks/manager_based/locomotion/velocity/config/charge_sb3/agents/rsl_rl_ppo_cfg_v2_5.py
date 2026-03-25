"""Phase 2.5 PPO 配置：從 Phase 2 checkpoint 繼續訓練（自適應課程學習）

訓練命令（使用最新的 checkpoint）：
   ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Navigation-Charge-v2.5 \
    --num_envs 128 \
    --resume \
    --load_run "/home/aa/IsaacLab/logs/rsl_rl/charge_navigation_phase2/charge_phase2"
    # 注意：不指定 --checkpoint 參數，系統會自動使用最新的 checkpoint（model_*.pt）

播放命令（Phase 2.5 訓練結果）：
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
        --task Isaac-Navigation-Charge-Play-v2.5 \
        --num_envs 3

注意：
    - 訓練時必須加 --resume 參數！否則即使配置文件設定 resume=True 也會被覆蓋
    - --load_run 支持絕對路徑，可以跨目錄載入（如從 Phase 2 載入到 Phase 2.5）
    - Phase 2.5 使用自適應課程學習，根據成功率和碰撞率動態調整難度
    - 障礙物數量會根據訓練表現動態調整（3-10 個）
"""

from isaaclab.utils import configclass

from .rsl_rl_ppo_cfg import ChargeNavigationPPORunnerCfg


@configclass
class ChargeNavigationPPORunnerCfgPhase2_5(ChargeNavigationPPORunnerCfg):
    """Phase 2.5 訓練配置：載入 Phase 2 checkpoint 繼續訓練（自適應課程學習）"""

    # 使用新的實驗名稱，避免覆蓋 Phase 2
    experiment_name = "charge_navigation_phase2_5"

    # resume 在配置文件中設為 True，但命令行必須也加 --resume
    # （因為 cli_args.py 的 default=False 會覆蓋配置）
    resume = True
    
    # load_run 通過命令行參數指定，支持絕對路徑
    # load_run = None
    
    # 只匹配 model_*.pt 檔案，避免匹配到 events.out.tfevents.*
    load_checkpoint = "model_.*\\.pt$"
