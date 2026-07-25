"""One-update SA6 bridge at the first failing corridor frontier rung."""

from dataclasses import replace

from rnn_car_modular.configs.e2e_sa6_k8_obb import CONFIG as _SA6


_C128 = (
    "/home/aa/IsaacLab/logs/rnn_car/"
    "sa6_k8_obb_sa5replay10_recovery_c10_2u_s42/checkpoint_128.pt"
)


CONFIG = replace(
    _SA6,
    name="e2e_sa6_k8_obb_corridor_rung_3s1d",
    description=(
        "One production-size SA6 update from replay-c128 at the first "
        "failing 4m x 10m corridor frontier rung: 3 static + 1 dynamic."
    ),
    checkpoint=_C128,
    no_resume_optimizer=False,
    timesteps=128,
    save_interval=1,
    long_corridor_static_obstacles=3,
    long_corridor_dynamic_obstacles=1,
    tags=_SA6.tags + (
        "c128_warm_start",
        "corridor_frontier_rung",
        "corridor_3s1d",
        "one_update_probe",
    ),
    notes=(
        "The fixed c128 frontier was 100% SR at 2S+0D, 91.11% at "
        "2S+1D, and 89.67% at 3S+1D. Train only the first failing rung. "
        "Keep the 68/10/12/10 scene mix, reward, optimizer state, c20 "
        "narrow beta=0.30, learning rate, and all other variables fixed."
    ),
)
