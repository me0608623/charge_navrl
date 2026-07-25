"""K8-aware corridor residual adapter from the joint c12 policy."""

from dataclasses import replace

from rnn_car_modular.configs.e2e_sa6_k8_obb_c12_corridor_adapter_2s1d import (
    CONFIG as _CURRENT_OBS_ADAPTER,
)


CONFIG = replace(
    _CURRENT_OBS_ADAPTER,
    name="e2e_sa6_k8_obb_c12_corridor_adapter_k8_2s1d",
    description=(
        "One SA6 update with a frozen c12 base, an 83D observation gate, "
        "and a residual branch that consumes the full 179D deployable K8 "
        "policy features at the 2S+1D corridor rung."
    ),
    corridor_adapter_residual_features="policy_features",
    tags=_CURRENT_OBS_ADAPTER.tags + (
        "k8_policy_feature_residual",
        "dynamic_obstacle_temporal_context",
    ),
    notes=(
        "The current-observation residual changed low-margin argmax decisions "
        "but did not improve the 2S+1D deterministic corridor gate; 72% of "
        "its collisions were dynamic. Keep the proven 83D scene gate, but "
        "feed the residual branch the frozen base policy's 179D deployable "
        "feature vector (83D current obs + 96D K8 CNN embedding). This adds "
        "no privileged input and preserves exact zero-init policy identity."
    ),
)
