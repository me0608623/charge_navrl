"""Keep deployment replay and narrow retention frozen in SA6-SA8."""

from rnn_car_modular.configs.e2e_sa6_k8_obb import CONFIG as SA6
from rnn_car_modular.configs.e2e_sa7_k8_obb import CONFIG as SA7
from rnn_car_modular.configs.e2e_sa8_k8_obb import CONFIG as SA8


def test_sa6_sa8_preserve_joint_replay_recipe() -> None:
    for config in (SA6, SA7, SA8):
        assert config.narrow_passage_fraction == 0.12
        assert config.narrow_passage_fixed_width_range == (1.2, 1.4)
        assert config.narrow_passage_fixed_yaw_limit_deg == 4.0
        assert config.narrow_passage_final_stress_ratio == 0.0
        assert config.long_corridor_fraction == 0.10
        assert config.long_corridor_free_width == 4.0
        assert config.long_corridor_length == 10.0
        assert config.long_corridor_static_obstacles == 4
        assert config.long_corridor_dynamic_obstacles == 2
        assert config.teacher_retention_weight == 0.10
        assert config.teacher_retention_checkpoint
