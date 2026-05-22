"""Phase configs — 每個 curriculum version 一個檔案，所有 phase 設定集中在此。

用法：
    from .phases import PHASE_REGISTRY

    # 取得某版本的完整 config
    config = PHASE_REGISTRY["warp_drive_single_agent_v1"]
"""

from .wd_single_agent_v1 import CONFIG as _wd_sa_v1
from .wd_goal_first import CONFIG as _wd_gf
from .wd_v1 import CONFIG as _wd_v1
from .baseline_v1 import CONFIG as _baseline_v1
from .rule_based_v1 import CONFIG as _rule_based_v1
from .wd_tcorridor_v1 import CONFIG as _wd_tc_v1

# 所有 phase configs 註冊表 — goal_obstacle_curriculum.py 從這裡讀取
PHASE_REGISTRY: dict[str, dict] = {
    "warp_drive_single_agent_v1": _wd_sa_v1,
    "warp_drive_goal_first": _wd_gf,
    "warp_drive_v1": _wd_v1,
    "baseline_v1": _baseline_v1,
    "rule_based_v1": _rule_based_v1,
    "wd_tcorridor_v1": _wd_tc_v1,
}
