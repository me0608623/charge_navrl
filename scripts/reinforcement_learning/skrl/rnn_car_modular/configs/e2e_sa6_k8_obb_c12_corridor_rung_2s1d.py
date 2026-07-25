"""First SA6 corridor rung from the joint Gate2/Gate5 c12 checkpoint."""

from dataclasses import replace

from rnn_car_modular.configs.e2e_sa6_k8_obb import CONFIG as _SA6


_C12 = (
    "/home/aa/IsaacLab/logs/rnn_car/"
    "sa5_e2e_k8_obb_corridor_teacher_distill_c12_stepnorm_s42/"
    "checkpoint_128.pt"
)


CONFIG = replace(
    _SA6,
    name="e2e_sa6_k8_obb_c12_corridor_rung_2s1d",
    description=(
        "One production-size SA6 update from the joint Gate2/Gate5 c12 "
        "checkpoint at its first failing corridor rung: 2S+1D."
    ),
    checkpoint=_C12,
    no_resume_optimizer=False,
    timesteps=128,
    save_interval=1,
    previous_stage_replay_fraction=0.20,
    long_corridor_static_obstacles=2,
    long_corridor_dynamic_obstacles=1,
    tags=_SA6.tags + (
        "c12_joint_warm_start",
        "corridor_frontier_rung",
        "corridor_2s1d",
        "sa5_general_replay_20pct",
        "one_update_probe",
    ),
    notes=(
        "c12 passes Gate2 at 8.37% CR and Gate5 at 96.71% crossing. Its "
        "exact corridor frontier is 100% SR at 2S+0D and 84.12% at "
        "2S+1D. Start SA6 from c12/Adam and train only 2S+1D. Keep "
        "20% SA5-general replay for Gate2 margin, 12% narrow replay with "
        "the frozen c20 beta=0.30 teacher, and 10% corridor replay. "
        "Reward and all optimizer hyperparameters remain unchanged."
    ),
)
