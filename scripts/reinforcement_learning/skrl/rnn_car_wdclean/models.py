"""Re-export model classes from modular_rnn_models.py (no duplication)."""

import sys
from pathlib import Path

_parent = str(Path(__file__).resolve().parent.parent)
_models_dir = str(Path(__file__).resolve().parent.parent / "models")
for _p in [_parent, _models_dir]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from modular_rnn_models import (  # noqa: F401
    # Constants
    EGO_START, EGO_END,
    GOAL_START, GOAL_END,
    LIDAR_START, LIDAR_END,
    TIME_START, TIME_END,
    STATE_DIM, LIDAR_DIM, USED_OBS_DIM, RAW_OBS_DIM,
    NUM_BINS, TOTAL_LOGITS,
    OBS_POLICY_OBS_DIM, OBS_POLICY_ACT_DIM,
    # Charge models
    LidarStateExtractor,
    PreprocessRNN,
    PreprocessGRU,
    RNNStateManager,
    PolicyHead,
    ValueHead,
    # Obstacle models
    ObstaclePolicyFC,
    ObstacleValueFC,
)
