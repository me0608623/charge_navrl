"""WD Sparse Reward 環境配置 — 專為 train_rnn_car_wdclip.py 設計

與 NavRL 版差異:
  - 不定義任何 reward (訓練腳本用自己的 compute_wd_charge_reward)
  - 保留 termination (goal_reached / collision)，訓練腳本讀取這些來計算 reward
  - 場景/觀測/動作/課程 全部繼承自 VLP16Curriculum

用法:
  gym.register 時指向此 cfg，或 train_rnn_car_wdclip.py 直接 import。
"""

from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg
from isaaclab.utils import configclass

from .charge_env_cfg_vlp16_curriculum import (
    ChargeNavigationEnvCfgVLP16Curriculum,
    RewardsCfgVLP16,
    COLLISION_THRESHOLD,
)
from ..mdp.rewards.potential_based_rewards import (
    collision_terminal_penalty,
    per_step_time_penalty,
)


@configclass
class RewardsCfgWDSparse(RewardsCfgVLP16):
    """WD Sparse — 所有 reward weight=0，只保留 termination 偵測用的殼。

    train_rnn_car_wdclip.py 自己計算 reward:
      - goal_reached: +40 (from termination_manager)
      - collision: -5 ~ -200 (from termination_manager)
      - action_cost: per-step (Phase 1 only)

    此 cfg 的 reward 對 policy gradient 無影響。
    """

    # 全部歸零 — 不產生任何 env.step() reward
    collision_terminal = RewTerm(
        func=collision_terminal_penalty,
        params={"sensor_cfg": SceneEntityCfg("lidar"), "threshold": COLLISION_THRESHOLD},
        weight=0.0,
    )
    time_penalty = RewTerm(
        func=per_step_time_penalty,
        params={},
        weight=0.0,
    )


@configclass
class ChargeNavigationEnvCfgVLP16CurriculumWD(ChargeNavigationEnvCfgVLP16Curriculum):
    """VLP-16 Curriculum + WD Sparse Reward

    繼承 VLP16Curriculum 的 scene/commands/events/curriculum，
    rewards 歸零（訓練腳本自行計算）。
    """

    rewards: RewardsCfgWDSparse = RewardsCfgWDSparse()
