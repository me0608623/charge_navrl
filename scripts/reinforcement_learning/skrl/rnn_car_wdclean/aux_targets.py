"""Re-export WD auxiliary target functions from wd_aux_targets.py (no duplication)."""

import sys
from pathlib import Path

_parent = str(Path(__file__).resolve().parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from wd_aux_targets import (  # noqa: F401
    build_wd_preprocess_targets,
    compute_wd_module_loss,
    WD_DEFAULT_WEIGHT,
)
