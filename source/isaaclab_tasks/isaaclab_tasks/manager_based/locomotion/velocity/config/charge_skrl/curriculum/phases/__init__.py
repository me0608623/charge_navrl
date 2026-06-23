"""Phase configs — 每個 curriculum version 一個檔案，所有 phase 設定集中在此。

用法：
    from .phases import PHASE_REGISTRY

    # 取得某版本的完整 config
    config = PHASE_REGISTRY["warp_drive_single_agent_v1"]
"""

from .wd_single_agent_v1 import CONFIG as _wd_sa_v1
from .wd_single_agent_v3 import CONFIG as _wd_sa_v3
from .wd_single_agent_v3d import CONFIG as _wd_sa_v3d
from .wd_single_agent_v3e import CONFIG as _wd_sa_v3e
from .wd_single_agent_v3g import CONFIG as _wd_sa_v3g
from .wd_single_agent_v3f_react import CONFIG as _wd_sa_v3f_react
from .wd_goal_first import CONFIG as _wd_gf
from .wd_v1 import CONFIG as _wd_v1
from .baseline_v1 import CONFIG as _baseline_v1
from .rule_based_v1 import CONFIG as _rule_based_v1
from .wd_tcorridor_v1 import CONFIG as _wd_tc_v1

# 所有 phase configs 註冊表 — goal_obstacle_curriculum.py 從這裡讀取
PHASE_REGISTRY: dict[str, dict] = {
    "warp_drive_single_agent_v1": _wd_sa_v1,
    "warp_drive_single_agent_v3": _wd_sa_v3,  # v3: r_min=0.25 + penalty_smoothness=0.005
    "warp_drive_single_agent_v3d": _wd_sa_v3d,  # v3d: smoothness=0.015 + ent_ang floor=0.02 (抗 sin 波抽動)
    "warp_drive_single_agent_v3e": _wd_sa_v3e,  # v3e: smoothness=0 (移除 sin 波元兇) + ent_ang floor=0.02
    "warp_drive_single_agent_v3g": _wd_sa_v3g,  # v3g: smoothness=0.003 (抗 bang-bang 塌縮) + ent_ang floor=0.02
    "warp_drive_single_agent_v3f_react": _wd_sa_v3f_react,  # v3f-react: SA5 clearance-gated 減速 + 壓低動態速度 (抗晚反應碰撞)
    "warp_drive_goal_first": _wd_gf,
    "warp_drive_v1": _wd_v1,
    "baseline_v1": _baseline_v1,
    "rule_based_v1": _rule_based_v1,
    "wd_tcorridor_v1": _wd_tc_v1,
}
