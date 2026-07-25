"""Observation-gated corridor adapter from the joint Gate2/Gate5 c12 policy."""

from dataclasses import replace

from rnn_car_modular.configs.e2e_sa6_k8_obb_c12_corridor_rung_2s1d import (
    CONFIG as _C12_RUNG,
)


_GATE_CHECKPOINT = (
    "/home/aa/IsaacLab/logs/probes/"
    "corridor_scene_observability_report_current_83d_mlp.pt"
)


CONFIG = replace(
    _C12_RUNG,
    name="e2e_sa6_k8_obb_c12_corridor_adapter_2s1d",
    description=(
        "One SA6 update with a frozen c12 base and a deployable "
        "observation-gated corridor residual adapter at the 2S+1D rung."
    ),
    no_resume_optimizer=True,
    teacher_retention_weight=0.02,
    corridor_adapter_enabled=True,
    corridor_adapter_hidden_dim=64,
    corridor_adapter_gate_loss_weight=0.05,
    corridor_adapter_gate_init_probability=0.01,
    corridor_adapter_max_logit_delta=2.0,
    corridor_adapter_freeze_base=True,
    corridor_adapter_gate_checkpoint=_GATE_CHECKPOINT,
    tags=_C12_RUNG.tags + (
        "corridor_residual_adapter",
        "observation_gate_83d",
        "frozen_c12_base",
        "c20_narrow_retention_beta0p02",
    ),
    notes=(
        "The c12 base passes Gate2 (CR 8.37%) and Gate5 (96.71% crossing) "
        "but first fails the exact corridor frontier at 2S+1D. Freeze its "
        "policy head and K8 CNN, then train only a zero-init residual adapter "
        "plus the critic. The gate consumes current normalized 83D policy obs "
        "and is initialized from the held-out scene probe (95.43% balanced "
        "accuracy); privileged corridor identity supplies BCE labels only. "
        "Keep reward unchanged. Narrow-only c20 forward KL is reduced to 0.02 "
        "because structural freezing is the primary retention mechanism."
    ),
)
