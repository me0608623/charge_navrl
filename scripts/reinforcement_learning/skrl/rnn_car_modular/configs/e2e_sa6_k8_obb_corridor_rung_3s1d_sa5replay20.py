"""One-update 3S+1D corridor bridge with stronger SA5 scene replay."""

from dataclasses import replace

from rnn_car_modular.configs.e2e_sa6_k8_obb_corridor_rung_3s1d import (
    CONFIG as _RUNG_3S1D,
)


CONFIG = replace(
    _RUNG_3S1D,
    name="e2e_sa6_k8_obb_corridor_rung_3s1d_sa5replay20",
    description=(
        "One production-size 3S+1D corridor update from c128 with SA5 "
        "general-scene replay increased from 10% to 20%."
    ),
    previous_stage_replay_fraction=0.20,
    tags=_RUNG_3S1D.tags + (
        "sa5_general_replay_20pct",
        "retention_balance_probe",
    ),
    notes=(
        "The 10% SA5 replay branch passed 3S+1D but regressed Gate2 to "
        "10.23% CR. Its realized rollout mix was only 7.3% SA5-general "
        "versus 18.1% corridor. Re-run from the original c128/Adam state "
        "with only the configured SA5 replay fraction changed to 20%. "
        "Reward, c20 beta, corridor fraction/rung, LR, and all other "
        "variables remain fixed."
    ),
)
