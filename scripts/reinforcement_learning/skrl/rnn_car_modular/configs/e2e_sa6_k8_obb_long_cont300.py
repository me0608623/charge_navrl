"""Long fixed-recipe SA6 continuation from the current Pareto checkpoint."""

from dataclasses import replace

from rnn_car_modular.configs.e2e_sa6_k8_obb import CONFIG as _SA6


_PARETO_C128 = (
    "/home/aa/IsaacLab/logs/rnn_car/"
    "sa6_k8_obb_sa5replay10_recovery_c10_2u_s42/checkpoint_128.pt"
)


CONFIG = replace(
    _SA6,
    name="e2e_sa6_k8_obb_long_cont300",
    description=(
        "Three-hundred-iteration fixed-recipe SA6 continuation testing "
        "whether corridor learning is training-volume limited."
    ),
    checkpoint=_PARETO_C128,
    no_resume_optimizer=False,
    timesteps=300 * _SA6.rollout_length,
    save_interval=10,
    tags=_SA6.tags + (
        "c128_warm_start",
        "long_continuation",
        "fixed_recipe_300_iterations",
    ),
    notes=(
        "Resume c128 model and Adam state. Do not change reward, learning "
        "rate, horizon, network, or replay mix. Keep 68% native SA6, 10% "
        "SA5-general, 12% fixed narrow with c20 forward-KL beta=0.30, and "
        "10% exact 4m x 10m 4S+2D corridor. Save every 10 iterations. "
        "Use 2S+1D as the capability-growth diagnostic and retain 4S+2D as "
        "the final deployment gate."
    ),
)
