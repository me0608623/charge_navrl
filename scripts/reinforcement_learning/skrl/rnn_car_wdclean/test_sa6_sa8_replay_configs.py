"""Keep deployment replay and narrow retention frozen in SA6-SA8."""

from rnn_car_modular.configs.e2e_sa6_k8_obb import CONFIG as SA6
from rnn_car_modular.configs.e2e_sa7_k8_obb import CONFIG as SA7
from rnn_car_modular.configs.e2e_sa8_k8_obb import CONFIG as SA8
from rnn_car_modular.configs.e2e_sa6_k8_obb_postmargin_smoke import (
    CONFIG as SA6_POSTMARGIN,
)
from rnn_car_modular.configs.e2e_sa6_k8_obb_beta0p3_cont import (
    CONFIG as SA6_BETA03_CONT,
)
from rnn_car_modular.configs.e2e_sa6_k8_obb_sa5_replay_recovery import (
    CONFIG as SA6_SA5_REPLAY_RECOVERY,
)
from rnn_car_modular.configs.e2e_sa6_k8_obb_long_cont300 import (
    CONFIG as SA6_LONG_CONT300,
)
from rnn_car_modular.configs.e2e_sa6_k8_obb_long_c60_to_c100 import (
    CONFIG as SA6_LONG_C60_TO_C100,
)
from rnn_car_modular.configs.e2e_sa6_k8_obb_long_c100_to_c110 import (
    CONFIG as SA6_LONG_C100_TO_C110,
)
from rnn_car_modular.configs.e2e_sa6_k8_obb_mixed_corridor_probe_from_c19200 import (
    CONFIG as SA6_MIXED_CORRIDOR_PROBE,
)
from rnn_car_modular.configs.e2e_sa6_k8_obb_c50_narrow_postkl_recovery import (
    CONFIG as SA6_C50_NARROW_POSTKL_RECOVERY,
)
from rnn_car_modular.configs.e2e_sa6_k8_obb_c50_narrow_fullactor_projection import (
    CONFIG as SA6_C50_NARROW_FULLACTOR_PROJECTION,
)
from rnn_car_modular.configs.e2e_sa6_k8_obb_c50_narrow_fullactor_projection8 import (
    CONFIG as SA6_C50_NARROW_FULLACTOR_PROJECTION8,
)
from rnn_car_modular.configs.e2e_sa6_k8_obb_c50_dual_projection import (
    CONFIG as SA6_C50_DUAL_PROJECTION,
)
from rnn_car_modular.configs.e2e_sa6_k8_obb_c50_dual_projection_lambda1 import (
    CONFIG as SA6_C50_DUAL_PROJECTION_LAMBDA1,
)
from rnn_car_modular.configs.e2e_sa6_k8_obb_c50_corridor_projection import (
    CONFIG as SA6_C50_CORRIDOR_PROJECTION,
)
from rnn_car_modular.configs.e2e_sa6_k8_obb_c100_dual_projection import (
    CONFIG as SA6_C100_DUAL_PROJECTION,
)
from rnn_car_modular.configs.e2e_sa6_k8_obb_c100_dual_projection_lambda0p5 import (
    CONFIG as SA6_C100_DUAL_PROJECTION_LAMBDA05,
)
from rnn_car_modular.configs.e2e_sa6_k8_obb_c100_dual_projection_cycle2 import (
    CONFIG as SA6_C100_DUAL_PROJECTION_CYCLE2,
)
from rnn_car_modular.configs.e2e_sa6_k8_obb_c100_narrow_projection import (
    CONFIG as SA6_C100_NARROW_PROJECTION,
)
from rnn_car_modular.configs.e2e_sa6_k8_obb_c100_teacher_forced_projection import (
    CONFIG as SA6_C100_TEACHER_FORCED_PROJECTION,
)
from rnn_car_modular.configs.e2e_sa6_k8_obb_dual_retention_beta0p02 import (
    CONFIG as SA6_DUAL_RETENTION,
)
from rnn_car_modular.configs.e2e_sa6_k8_obb_dual_retention_beta0p05 import (
    CONFIG as SA6_DUAL_RETENTION_BETA05,
)
from rnn_car_modular.configs.e2e_sa6_k8_obb_dual_retention_beta0p10 import (
    CONFIG as SA6_DUAL_RETENTION_BETA10,
)
from rnn_car_modular.configs.e2e_sa6_k8_obb_corridor_rung_3s1d import (
    CONFIG as SA6_CORRIDOR_RUNG_3S1D,
)
from rnn_car_modular.configs.e2e_sa6_k8_obb_corridor_rung_3s1d_sa5replay20 import (
    CONFIG as SA6_CORRIDOR_RUNG_3S1D_SA5_20,
)
from rnn_car_modular.configs.e2e_sa6_k8_obb_c12_corridor_rung_2s1d import (
    CONFIG as SA6_C12_CORRIDOR_RUNG_2S1D,
)
from rnn_car_modular.configs.e2e_sa6_k8_obb_c12_corridor_rung_2s1d_cont2 import (
    CONFIG as SA6_C12_CORRIDOR_RUNG_2S1D_CONT2,
)
from rnn_car_modular.configs.e2e_sa6_k8_obb_corridor_rung_3s1d_sa5recovery40 import (
    CONFIG as SA6_CORRIDOR_RUNG_3S1D_SA5_RECOVERY40,
)
from rnn_car_modular.configs.e2e_sa5_k8_obb_teacher_corridor_recovery import (
    CONFIG as SA5_RETENTION,
)
from rnn_car_modular.configs.e2e_sa5_k8_obb_corridor_bridge import (
    CONFIG as SA5_CORRIDOR_BRIDGE,
)
from rnn_car_modular.configs.e2e_sa5_k8_obb_corridor_bridge_easy import (
    CONFIG as SA5_CORRIDOR_EASY,
)
from rnn_car_modular.configs.e2e_sa5_k8_obb_corridor_bridge_medium import (
    CONFIG as SA5_CORRIDOR_MEDIUM,
)
from rnn_car_modular.configs.e2e_sa5_k8_obb_corridor_bridge_medium_cont import (
    CONFIG as SA5_CORRIDOR_MEDIUM_CONT,
)
from rnn_car_modular.configs.e2e_sa5_k8_obb_corridor_bridge_dynamic import (
    CONFIG as SA5_CORRIDOR_DYNAMIC,
)
from rnn_car_modular.configs.e2e_sa5_k8_obb_corridor_bridge_dynamic_ce5 import (
    CONFIG as SA5_CORRIDOR_DYNAMIC_CE5,
)
from rnn_car_modular.configs.e2e_sa5_k8_obb_corridor_bridge_dynamic_postkl import (
    CONFIG as SA5_CORRIDOR_DYNAMIC_POSTKL,
)
from rnn_car_modular.configs.e2e_sa5_k8_obb_corridor_bridge_dynamic_postkl_cont import (
    CONFIG as SA5_CORRIDOR_DYNAMIC_POSTKL_CONT,
)
from rnn_car_modular.configs.e2e_sa5_k8_obb_corridor_teacher_distill import (
    CONFIG as SA5_CORRIDOR_TEACHER_DISTILL,
)
from rnn_car_modular.configs.e2e_sa5_k8_obb_corridor_teacher_distill_cont import (
    CONFIG as SA5_CORRIDOR_TEACHER_DISTILL_CONT,
)
from rnn_car_modular.configs.e2e_sa5_k8_obb_corridor_teacher_intervention import (
    CONFIG as SA5_CORRIDOR_TEACHER_INTERVENTION,
)
from rnn_car_modular.configs.e2e_sa5_k8_obb_corridor_teacher_intervention_projection_only import (
    CONFIG as SA5_CORRIDOR_TEACHER_INTERVENTION_PROJECTION_ONLY,
)
from rnn_car_modular.configs.e2e_sa5_k8_obb_corridor_bridge_dynamic_future05 import (
    CONFIG as SA5_CORRIDOR_DYNAMIC_FUTURE05,
)


def test_sa5_retention_replay_contains_exact_deployment_boundary() -> None:
    assert SA5_RETENTION.narrow_passage_fixed_width_range == (1.2, 1.4)
    assert SA5_RETENTION.narrow_passage_exact_width == 1.2
    assert SA5_RETENTION.narrow_passage_exact_width_ratio == 0.50


def test_sa5_corridor_bridge_is_temporary_twenty_percent_replay() -> None:
    assert SA5_CORRIDOR_BRIDGE.long_corridor_fraction == 0.20
    assert SA5_CORRIDOR_BRIDGE.narrow_passage_fraction == 0.12
    assert SA5_CORRIDOR_BRIDGE.teacher_retention_action_ce_weight == 1.0
    assert SA5_CORRIDOR_BRIDGE.critic_detach_encoder is True
    assert SA5_CORRIDOR_BRIDGE.no_resume_optimizer is True


def test_first_corridor_rung_keeps_deployment_geometry() -> None:
    assert SA5_CORRIDOR_EASY.long_corridor_free_width == 4.0
    assert SA5_CORRIDOR_EASY.long_corridor_length == 10.0
    assert SA5_CORRIDOR_EASY.long_corridor_static_obstacles == 2
    assert SA5_CORRIDOR_EASY.long_corridor_dynamic_obstacles == 0


def test_second_corridor_rung_advances_only_obstacle_count() -> None:
    assert SA5_CORRIDOR_MEDIUM.long_corridor_free_width == 4.0
    assert SA5_CORRIDOR_MEDIUM.long_corridor_length == 10.0
    assert SA5_CORRIDOR_MEDIUM.long_corridor_static_obstacles == 3
    assert SA5_CORRIDOR_MEDIUM.long_corridor_dynamic_obstacles == 1
    assert SA5_CORRIDOR_MEDIUM.checkpoint.endswith("checkpoint_256.pt")
    assert (
        SA5_CORRIDOR_MEDIUM.teacher_retention_checkpoint
        == SA5_CORRIDOR_MEDIUM.checkpoint
    )


def test_medium_continuation_preserves_optimizer_and_teacher() -> None:
    assert SA5_CORRIDOR_MEDIUM_CONT.no_resume_optimizer is False
    assert (
        SA5_CORRIDOR_MEDIUM_CONT.teacher_retention_checkpoint
        == SA5_CORRIDOR_MEDIUM_CONT.checkpoint
    )
    assert SA5_CORRIDOR_MEDIUM_CONT.long_corridor_static_obstacles == 3
    assert SA5_CORRIDOR_MEDIUM_CONT.long_corridor_dynamic_obstacles == 1


def test_dynamic_rung_changes_only_obstacle_mix_and_save_interval() -> None:
    assert SA5_CORRIDOR_DYNAMIC.no_resume_optimizer is False
    assert SA5_CORRIDOR_DYNAMIC.save_interval == 1
    assert SA5_CORRIDOR_DYNAMIC.long_corridor_static_obstacles == 2
    assert SA5_CORRIDOR_DYNAMIC.long_corridor_dynamic_obstacles == 1
    assert SA5_CORRIDOR_DYNAMIC.lr == SA5_CORRIDOR_MEDIUM_CONT.lr
    assert (
        SA5_CORRIDOR_DYNAMIC.teacher_retention_checkpoint
        == SA5_CORRIDOR_MEDIUM_CONT.teacher_retention_checkpoint
    )


def test_dynamic_ce5_branch_is_one_update_and_uses_its_own_teacher() -> None:
    assert SA5_CORRIDOR_DYNAMIC_CE5.timesteps == 128
    assert SA5_CORRIDOR_DYNAMIC_CE5.save_interval == 1
    assert SA5_CORRIDOR_DYNAMIC_CE5.no_resume_optimizer is False
    assert SA5_CORRIDOR_DYNAMIC_CE5.teacher_retention_action_ce_weight == 5.0
    assert (
        SA5_CORRIDOR_DYNAMIC_CE5.teacher_retention_checkpoint
        == SA5_CORRIDOR_DYNAMIC_CE5.checkpoint
    )


def test_dynamic_postkl_branch_has_no_pre_update_retention_loss() -> None:
    assert SA5_CORRIDOR_DYNAMIC_POSTKL.timesteps == 128
    assert SA5_CORRIDOR_DYNAMIC_POSTKL.teacher_retention_weight == 0.0
    assert SA5_CORRIDOR_DYNAMIC_POSTKL.teacher_retention_margin_weight == 0.0
    assert SA5_CORRIDOR_DYNAMIC_POSTKL.teacher_retention_action_ce_weight == 0.0
    assert SA5_CORRIDOR_DYNAMIC_POSTKL.teacher_retention_post_kl_epochs == 4
    assert SA5_CORRIDOR_DYNAMIC_POSTKL.teacher_retention_post_kl_lr == 1e-3
    assert (
        SA5_CORRIDOR_DYNAMIC_POSTKL.teacher_retention_checkpoint
        == SA5_CORRIDOR_DYNAMIC_POSTKL.checkpoint
    )


def test_dynamic_postkl_continuation_keeps_one_process_alive() -> None:
    assert SA5_CORRIDOR_DYNAMIC_POSTKL_CONT.timesteps == 640
    assert SA5_CORRIDOR_DYNAMIC_POSTKL_CONT.rollout_length == 128
    assert SA5_CORRIDOR_DYNAMIC_POSTKL_CONT.save_interval == 1
    assert SA5_CORRIDOR_DYNAMIC_POSTKL_CONT.no_resume_optimizer is False
    assert SA5_CORRIDOR_DYNAMIC_POSTKL_CONT.long_corridor_static_obstacles == 2
    assert SA5_CORRIDOR_DYNAMIC_POSTKL_CONT.long_corridor_dynamic_obstacles == 1
    assert SA5_CORRIDOR_DYNAMIC_POSTKL_CONT.teacher_retention_post_kl_epochs == 4
    assert SA5_CORRIDOR_DYNAMIC_POSTKL_CONT.teacher_retention_weight == 0.0
    assert (
        SA5_CORRIDOR_DYNAMIC_POSTKL_CONT.teacher_retention_action_ce_weight
        == 0.0
    )


def test_corridor_teacher_distillation_is_one_update_and_keeps_narrow_guard() -> None:
    assert SA5_CORRIDOR_TEACHER_DISTILL.timesteps == 128
    assert SA5_CORRIDOR_TEACHER_DISTILL.save_interval == 1
    assert SA5_CORRIDOR_TEACHER_DISTILL.no_resume_optimizer is False
    assert SA5_CORRIDOR_TEACHER_DISTILL.long_corridor_fraction == 0.20
    assert SA5_CORRIDOR_TEACHER_DISTILL.long_corridor_static_obstacles == 2
    assert SA5_CORRIDOR_TEACHER_DISTILL.long_corridor_dynamic_obstacles == 1
    assert SA5_CORRIDOR_TEACHER_DISTILL.corridor_teacher_distill_epochs == 4
    assert SA5_CORRIDOR_TEACHER_DISTILL.corridor_teacher_distill_lr == 0.1
    assert (
        SA5_CORRIDOR_TEACHER_DISTILL.corridor_teacher_distill_neighbor_mass
        == 0.20
    )
    assert SA5_CORRIDOR_TEACHER_DISTILL.teacher_retention_post_kl_epochs == 4
    assert (
        SA5_CORRIDOR_TEACHER_DISTILL.teacher_retention_checkpoint
        == SA5_CORRIDOR_TEACHER_DISTILL.checkpoint
    )


def test_corridor_teacher_continuation_keeps_c10_as_narrow_teacher() -> None:
    assert SA5_CORRIDOR_TEACHER_DISTILL_CONT.timesteps == 128
    assert SA5_CORRIDOR_TEACHER_DISTILL_CONT.save_interval == 1
    assert SA5_CORRIDOR_TEACHER_DISTILL_CONT.no_resume_optimizer is False
    assert SA5_CORRIDOR_TEACHER_DISTILL_CONT.checkpoint.endswith(
        "corridor_teacher_distill_c12_stepnorm_s42/checkpoint_128.pt"
    )
    assert (
        SA5_CORRIDOR_TEACHER_DISTILL_CONT.teacher_retention_checkpoint
        == SA5_CORRIDOR_TEACHER_DISTILL.teacher_retention_checkpoint
    )
    assert (
        SA5_CORRIDOR_TEACHER_DISTILL_CONT.teacher_retention_checkpoint
        != SA5_CORRIDOR_TEACHER_DISTILL_CONT.checkpoint
    )
    assert (
        SA5_CORRIDOR_TEACHER_DISTILL_CONT.corridor_teacher_distill_batch_size
        == 65536
    )


def test_corridor_teacher_intervention_is_conservative_and_c12_based() -> None:
    assert SA5_CORRIDOR_TEACHER_INTERVENTION.timesteps == 128
    assert SA5_CORRIDOR_TEACHER_INTERVENTION.checkpoint.endswith(
        "corridor_teacher_distill_c12_stepnorm_s42/checkpoint_128.pt"
    )
    assert SA5_CORRIDOR_TEACHER_INTERVENTION.no_resume_optimizer is False
    assert (
        SA5_CORRIDOR_TEACHER_INTERVENTION.corridor_teacher_intervention_only
        is True
    )
    assert (
        SA5_CORRIDOR_TEACHER_INTERVENTION.corridor_teacher_intervention_clearance_m
        == 0.20
    )
    assert SA5_CORRIDOR_TEACHER_INTERVENTION.corridor_teacher_distill_lr == 0.02
    assert (
        SA5_CORRIDOR_TEACHER_INTERVENTION.corridor_teacher_distill_batch_size
        == 65536
    )
    assert (
        SA5_CORRIDOR_TEACHER_INTERVENTION.teacher_retention_checkpoint
        == SA5_CORRIDOR_TEACHER_DISTILL.teacher_retention_checkpoint
    )


def test_intervention_projection_only_skips_ppo_without_changing_teacher() -> None:
    assert SA5_CORRIDOR_TEACHER_INTERVENTION_PROJECTION_ONLY.ppo_epochs == 0
    assert SA5_CORRIDOR_TEACHER_INTERVENTION_PROJECTION_ONLY.timesteps == 128
    assert (
        SA5_CORRIDOR_TEACHER_INTERVENTION_PROJECTION_ONLY.checkpoint
        == SA5_CORRIDOR_TEACHER_INTERVENTION.checkpoint
    )
    assert (
        SA5_CORRIDOR_TEACHER_INTERVENTION_PROJECTION_ONLY.teacher_retention_checkpoint
        == SA5_CORRIDOR_TEACHER_INTERVENTION.teacher_retention_checkpoint
    )
    assert (
        SA5_CORRIDOR_TEACHER_INTERVENTION_PROJECTION_ONLY.corridor_teacher_intervention_only
        is True
    )


def test_dynamic_future05_changes_only_weight_and_window() -> None:
    assert SA5_CORRIDOR_DYNAMIC_FUTURE05.future_occupancy_weight == 0.50
    assert SA5_CORRIDOR_DYNAMIC_FUTURE05.timesteps == 384
    assert SA5_CORRIDOR_DYNAMIC_FUTURE05.save_interval == 1
    assert SA5_CORRIDOR_DYNAMIC_FUTURE05.no_resume_optimizer is False
    assert SA5_CORRIDOR_DYNAMIC_FUTURE05.long_corridor_static_obstacles == 2
    assert SA5_CORRIDOR_DYNAMIC_FUTURE05.long_corridor_dynamic_obstacles == 1
    assert SA5_CORRIDOR_DYNAMIC_FUTURE05.teacher_retention_post_kl_epochs == 4


def test_sa6_sa8_preserve_joint_replay_recipe() -> None:
    for config in (SA6, SA7, SA8):
        assert config.previous_stage_replay_fraction == 0.10
        assert config.previous_stage_replay_static_obstacles == 10
        assert config.previous_stage_replay_dynamic_obstacles == 3
        assert config.previous_stage_replay_min_walls == 2
        assert config.previous_stage_replay_max_walls == 3
        assert config.previous_stage_replay_wall_length == 4.0
        assert config.previous_stage_replay_obstacle_boundary == 5.5
        assert config.narrow_passage_fraction == 0.12
        assert config.narrow_passage_fixed_width_range == (1.2, 1.4)
        assert config.narrow_passage_fixed_yaw_limit_deg == 4.0
        assert config.narrow_passage_final_stress_ratio == 0.0
        assert config.long_corridor_fraction == 0.10
        assert config.long_corridor_free_width == 4.0
        assert config.long_corridor_length == 10.0
        assert config.long_corridor_static_obstacles == 4
        assert config.long_corridor_dynamic_obstacles == 2
        assert config.teacher_retention_weight == 0.30
        assert config.teacher_retention_checkpoint
        assert (
            config.previous_stage_replay_fraction
            + config.narrow_passage_fraction
            + config.long_corridor_fraction
        ) == 0.32


def test_c50_narrow_recovery_is_projection_only_and_head_scoped() -> None:
    config = SA6_C50_NARROW_POSTKL_RECOVERY
    assert config.checkpoint.endswith(
        "sa6_k8_obb_long_cont300_c128_s42/checkpoint_6400.pt"
    )
    assert config.ppo_epochs == 0
    assert config.timesteps == config.rollout_length == 128
    assert config.save_interval == 1
    assert config.teacher_retention_weight == 0.0
    assert config.teacher_retention_post_kl_epochs == 4
    assert config.teacher_retention_post_kl_batch_size == 65536
    assert config.teacher_retention_post_policy_head_only is True
    assert config.teacher_retention_post_margin_weight == 1.0
    assert config.previous_stage_replay_fraction == 0.10
    assert config.narrow_passage_fraction == 0.12
    assert config.long_corridor_fraction == 0.10
    assert config.long_corridor_static_obstacles == 4
    assert config.long_corridor_dynamic_obstacles == 2


def test_c50_fullactor_projection_changes_only_projection_scope() -> None:
    head = SA6_C50_NARROW_POSTKL_RECOVERY
    full = SA6_C50_NARROW_FULLACTOR_PROJECTION
    assert full.checkpoint == head.checkpoint
    assert full.ppo_epochs == head.ppo_epochs == 0
    assert full.timesteps == head.timesteps == head.rollout_length
    assert full.teacher_retention_post_kl_epochs == (
        head.teacher_retention_post_kl_epochs
    )
    assert full.teacher_retention_post_kl_lr == (
        head.teacher_retention_post_kl_lr
    )
    assert full.teacher_retention_post_margin_weight == (
        head.teacher_retention_post_margin_weight
    )
    assert head.teacher_retention_post_policy_head_only is True
    assert full.teacher_retention_post_policy_head_only is False
    assert full.previous_stage_replay_fraction == (
        head.previous_stage_replay_fraction
    )
    assert full.narrow_passage_fraction == head.narrow_passage_fraction
    assert full.long_corridor_fraction == head.long_corridor_fraction


def test_c50_fullactor_projection8_changes_only_projection_steps() -> None:
    four = SA6_C50_NARROW_FULLACTOR_PROJECTION
    eight = SA6_C50_NARROW_FULLACTOR_PROJECTION8
    assert eight.checkpoint == four.checkpoint
    assert eight.ppo_epochs == four.ppo_epochs == 0
    assert eight.teacher_retention_post_policy_head_only is False
    assert four.teacher_retention_post_kl_epochs == 4
    assert eight.teacher_retention_post_kl_epochs == 8
    assert eight.teacher_retention_post_kl_lr == (
        four.teacher_retention_post_kl_lr
    )
    assert eight.teacher_retention_post_margin_weight == (
        four.teacher_retention_post_margin_weight
    )
    assert eight.previous_stage_replay_fraction == (
        four.previous_stage_replay_fraction
    )
    assert eight.narrow_passage_fraction == four.narrow_passage_fraction
    assert eight.long_corridor_fraction == four.long_corridor_fraction


def test_c50_dual_projection_adds_only_disjoint_c50_anchor() -> None:
    narrow_only = SA6_C50_NARROW_FULLACTOR_PROJECTION8
    dual = SA6_C50_DUAL_PROJECTION
    assert dual.checkpoint == narrow_only.checkpoint
    assert dual.ppo_epochs == narrow_only.ppo_epochs == 0
    assert dual.timesteps == narrow_only.timesteps
    assert dual.teacher_retention_checkpoint == (
        narrow_only.teacher_retention_checkpoint
    )
    assert dual.teacher_retention_post_kl_epochs == 8
    assert dual.teacher_retention_post_policy_head_only is False
    assert dual.previous_stage_teacher_checkpoint == dual.checkpoint
    assert dual.previous_stage_teacher_retention_weight == 0.0
    assert dual.previous_stage_teacher_scope == "non_narrow"
    assert dual.teacher_retention_post_anchor_weight == 0.5
    assert dual.previous_stage_replay_fraction == (
        narrow_only.previous_stage_replay_fraction
    )
    assert dual.narrow_passage_fraction == (
        narrow_only.narrow_passage_fraction
    )
    assert dual.long_corridor_fraction == (
        narrow_only.long_corridor_fraction
    )


def test_c50_dual_projection_lambda1_changes_only_anchor_strength() -> None:
    half = SA6_C50_DUAL_PROJECTION
    full = SA6_C50_DUAL_PROJECTION_LAMBDA1
    assert full.checkpoint == half.checkpoint
    assert full.teacher_retention_checkpoint == (
        half.teacher_retention_checkpoint
    )
    assert full.previous_stage_teacher_checkpoint == (
        half.previous_stage_teacher_checkpoint
    )
    assert full.previous_stage_teacher_scope == half.previous_stage_teacher_scope
    assert full.teacher_retention_post_kl_epochs == (
        half.teacher_retention_post_kl_epochs
    )
    assert full.teacher_retention_post_policy_head_only is False
    assert half.teacher_retention_post_anchor_weight == 0.5
    assert full.teacher_retention_post_anchor_weight == 1.0
    assert full.previous_stage_replay_fraction == (
        half.previous_stage_replay_fraction
    )
    assert full.narrow_passage_fraction == half.narrow_passage_fraction
    assert full.long_corridor_fraction == half.long_corridor_fraction


def test_c50_corridor_projection_changes_only_anchor_scope() -> None:
    broad = SA6_C50_DUAL_PROJECTION
    corridor = SA6_C50_CORRIDOR_PROJECTION
    assert corridor.checkpoint == broad.checkpoint
    assert corridor.teacher_retention_checkpoint == (
        broad.teacher_retention_checkpoint
    )
    assert corridor.previous_stage_teacher_checkpoint == (
        broad.previous_stage_teacher_checkpoint
    )
    assert corridor.teacher_retention_post_anchor_weight == (
        broad.teacher_retention_post_anchor_weight
    )
    assert corridor.teacher_retention_post_anchor_weight == 0.5
    assert broad.previous_stage_teacher_scope == "non_narrow"
    assert corridor.previous_stage_teacher_scope == "corridor"
    assert corridor.teacher_retention_post_kl_epochs == (
        broad.teacher_retention_post_kl_epochs
    )
    assert corridor.teacher_retention_post_policy_head_only is False
    assert corridor.previous_stage_replay_fraction == (
        broad.previous_stage_replay_fraction
    )
    assert corridor.narrow_passage_fraction == broad.narrow_passage_fraction
    assert corridor.long_corridor_fraction == broad.long_corridor_fraction


def test_c100_dual_projection_uses_same_checkpoint_as_c100_anchor() -> None:
    config = SA6_C100_DUAL_PROJECTION
    assert config.checkpoint.endswith(
        "sa6_k8_obb_long_c60_to_c100_s42/checkpoint_5120.pt"
    )
    assert config.previous_stage_teacher_checkpoint == config.checkpoint
    assert config.previous_stage_teacher_scope == "non_narrow"
    assert config.teacher_retention_checkpoint.endswith(
        "sa5_e2e_k8_obb_narrow_recovery_seg2_c1280_s42/checkpoint_1280.pt"
    )
    assert config.teacher_retention_post_anchor_weight == 0.75
    assert config.teacher_retention_post_kl_epochs == 8
    assert config.teacher_retention_post_policy_head_only is False
    assert config.ppo_epochs == 0
    assert config.timesteps == config.rollout_length == 128
    assert config.previous_stage_replay_fraction == 0.10
    assert config.narrow_passage_fraction == 0.12
    assert config.long_corridor_fraction == 0.10
    assert config.long_corridor_static_obstacles == 4
    assert config.long_corridor_dynamic_obstacles == 2


def test_c100_dual_lambda05_changes_only_anchor_strength() -> None:
    lambda075 = SA6_C100_DUAL_PROJECTION
    lambda05 = SA6_C100_DUAL_PROJECTION_LAMBDA05
    assert lambda05.checkpoint == lambda075.checkpoint
    assert (
        lambda05.teacher_retention_checkpoint
        == lambda075.teacher_retention_checkpoint
    )
    assert (
        lambda05.previous_stage_teacher_checkpoint
        == lambda075.previous_stage_teacher_checkpoint
    )
    assert (
        lambda05.previous_stage_teacher_scope
        == lambda075.previous_stage_teacher_scope
        == "non_narrow"
    )
    assert (
        lambda05.teacher_retention_post_kl_epochs
        == lambda075.teacher_retention_post_kl_epochs
        == 8
    )
    assert lambda075.teacher_retention_post_anchor_weight == 0.75
    assert lambda05.teacher_retention_post_anchor_weight == 0.5
    assert lambda05.teacher_retention_rollout_override is False


def test_c100_projection_cycle2_refreshes_only_student_and_step_count() -> None:
    cycle1 = SA6_C100_DUAL_PROJECTION
    cycle2 = SA6_C100_DUAL_PROJECTION_CYCLE2
    assert cycle2.checkpoint.endswith(
        "sa6_c100_dual_projection_lambda0p75_1u_s42/checkpoint_128.pt"
    )
    assert cycle2.previous_stage_teacher_checkpoint == (
        cycle1.previous_stage_teacher_checkpoint
    )
    assert cycle2.teacher_retention_checkpoint == (
        cycle1.teacher_retention_checkpoint
    )
    assert cycle2.teacher_retention_post_anchor_weight == (
        cycle1.teacher_retention_post_anchor_weight
    )
    assert cycle2.teacher_retention_post_anchor_weight == 0.75
    assert cycle1.teacher_retention_post_kl_epochs == 8
    assert cycle2.teacher_retention_post_kl_epochs == 4
    assert cycle2.ppo_epochs == 0
    assert cycle2.timesteps == cycle2.rollout_length == 128
    assert cycle2.previous_stage_teacher_scope == "non_narrow"


def test_c100_narrow_projection_is_proven_c20_only_recipe() -> None:
    config = SA6_C100_NARROW_PROJECTION
    assert config.checkpoint.endswith(
        "sa6_k8_obb_long_c60_to_c100_s42/checkpoint_5120.pt"
    )
    assert config.teacher_retention_checkpoint.endswith(
        "sa5_e2e_k8_obb_narrow_recovery_seg2_c1280_s42/checkpoint_1280.pt"
    )
    assert config.previous_stage_teacher_checkpoint is None
    assert config.previous_stage_teacher_retention_weight == 0.0
    assert config.teacher_retention_post_anchor_weight == 0.0
    assert config.teacher_retention_post_kl_epochs == 8
    assert config.teacher_retention_post_policy_head_only is False
    assert config.ppo_epochs == 0
    assert config.timesteps == config.rollout_length == 128


def test_c100_teacher_forced_projection_changes_only_rollout_coverage() -> None:
    student_visited = SA6_C100_NARROW_PROJECTION
    teacher_forced = SA6_C100_TEACHER_FORCED_PROJECTION
    assert teacher_forced.checkpoint == student_visited.checkpoint
    assert (
        teacher_forced.teacher_retention_checkpoint
        == student_visited.teacher_retention_checkpoint
    )
    assert teacher_forced.ppo_epochs == student_visited.ppo_epochs == 0
    assert teacher_forced.timesteps == student_visited.timesteps
    assert teacher_forced.teacher_retention_post_kl_epochs == (
        student_visited.teacher_retention_post_kl_epochs
    )
    assert teacher_forced.teacher_retention_post_kl_lr == (
        student_visited.teacher_retention_post_kl_lr
    )
    assert teacher_forced.teacher_retention_post_margin_weight == (
        student_visited.teacher_retention_post_margin_weight
    )
    assert student_visited.teacher_retention_rollout_override is False
    assert teacher_forced.teacher_retention_rollout_override is True
    assert teacher_forced.previous_stage_teacher_checkpoint is None
    assert teacher_forced.teacher_retention_post_anchor_weight == 0.0


def test_sa6_postmargin_calibration_is_post_update_and_head_only() -> None:
    assert SA6_POSTMARGIN.teacher_retention_weight == 0.0
    assert SA6_POSTMARGIN.teacher_retention_post_kl_epochs == 4
    assert SA6_POSTMARGIN.teacher_retention_post_kl_batch_size == 65536
    assert SA6_POSTMARGIN.teacher_retention_post_margin_weight == 1.0
    assert SA6_POSTMARGIN.teacher_retention_post_action_ce_weight == 0.0
    assert SA6_POSTMARGIN.teacher_retention_post_policy_head_only is True
    assert SA6_POSTMARGIN.teacher_retention_argmax_margin == 0.2


def test_sa6_beta03_continuation_preserves_calibrated_recipe() -> None:
    assert SA6_BETA03_CONT.checkpoint.endswith(
        "sa6_c12_beta0p3_formal1update_ne1024_s42/checkpoint_128.pt"
    )
    assert SA6_BETA03_CONT.no_resume_optimizer is False
    assert SA6_BETA03_CONT.teacher_retention_weight == 0.30
    assert SA6_BETA03_CONT.timesteps == 1280
    assert SA6_BETA03_CONT.save_interval == 2
    assert SA6_BETA03_CONT.narrow_passage_fraction == 0.12
    assert SA6_BETA03_CONT.long_corridor_fraction == 0.10


def test_sa6_sa5_replay_recovery_resumes_c10_optimizer() -> None:
    assert SA6_SA5_REPLAY_RECOVERY.checkpoint.endswith(
        "sa6_k8_obb_beta0p3_joint_replay_cont10_s42/checkpoint_1280.pt"
    )
    assert SA6_SA5_REPLAY_RECOVERY.no_resume_optimizer is False
    assert SA6_SA5_REPLAY_RECOVERY.timesteps == 256
    assert SA6_SA5_REPLAY_RECOVERY.save_interval == 1
    assert SA6_SA5_REPLAY_RECOVERY.previous_stage_replay_fraction == 0.10
    assert SA6_SA5_REPLAY_RECOVERY.teacher_retention_weight == 0.30


def test_sa6_long_continuation_changes_only_training_budget() -> None:
    assert SA6_LONG_CONT300.checkpoint.endswith(
        "sa6_k8_obb_sa5replay10_recovery_c10_2u_s42/checkpoint_128.pt"
    )
    assert SA6_LONG_CONT300.no_resume_optimizer is False
    assert SA6_LONG_CONT300.timesteps == 300 * 128
    assert SA6_LONG_CONT300.save_interval == 10
    assert SA6_LONG_CONT300.previous_stage_replay_fraction == 0.10
    assert SA6_LONG_CONT300.narrow_passage_fraction == 0.12
    assert SA6_LONG_CONT300.long_corridor_fraction == 0.10
    assert SA6_LONG_CONT300.long_corridor_static_obstacles == 4
    assert SA6_LONG_CONT300.long_corridor_dynamic_obstacles == 2
    assert SA6_LONG_CONT300.teacher_retention_weight == 0.30
    assert SA6_LONG_CONT300.previous_stage_teacher_checkpoint is None
    assert SA6_LONG_CONT300.corridor_adapter_enabled is False


def test_sa6_long_c60_to_c100_preserves_fixed_recipe() -> None:
    assert SA6_LONG_C60_TO_C100.checkpoint.endswith(
        "sa6_k8_obb_long_cont300_c128_s42/checkpoint_7680.pt"
    )
    assert SA6_LONG_C60_TO_C100.no_resume_optimizer is False
    assert SA6_LONG_C60_TO_C100.timesteps == 40 * 128
    assert SA6_LONG_C60_TO_C100.save_interval == 10
    assert SA6_LONG_C60_TO_C100.lr == SA6_LONG_CONT300.lr
    assert SA6_LONG_C60_TO_C100.teacher_retention_weight == 0.30
    assert SA6_LONG_C60_TO_C100.previous_stage_replay_fraction == 0.10
    assert SA6_LONG_C60_TO_C100.narrow_passage_fraction == 0.12
    assert SA6_LONG_C60_TO_C100.long_corridor_fraction == 0.10
    assert SA6_LONG_C60_TO_C100.long_corridor_static_obstacles == 4
    assert SA6_LONG_C60_TO_C100.long_corridor_dynamic_obstacles == 2


def test_sa6_long_c100_to_c110_preserves_recipe_and_optimizer() -> None:
    assert SA6_LONG_C100_TO_C110.checkpoint.endswith(
        "sa6_k8_obb_long_c60_to_c100_s42/checkpoint_5120.pt"
    )
    assert SA6_LONG_C100_TO_C110.no_resume_optimizer is False
    assert SA6_LONG_C100_TO_C110.timesteps == 10 * 128
    assert SA6_LONG_C100_TO_C110.save_interval == 5
    assert SA6_LONG_C100_TO_C110.lr == SA6_LONG_CONT300.lr
    assert SA6_LONG_C100_TO_C110.teacher_retention_weight == 0.30
    assert SA6_LONG_C100_TO_C110.previous_stage_replay_fraction == 0.10
    assert SA6_LONG_C100_TO_C110.narrow_passage_fraction == 0.12
    assert SA6_LONG_C100_TO_C110.long_corridor_fraction == 0.10
    assert SA6_LONG_C100_TO_C110.long_corridor_static_obstacles == 4
    assert SA6_LONG_C100_TO_C110.long_corridor_dynamic_obstacles == 2


def test_sa6_mixed_corridor_probe_changes_only_motion_and_budget() -> None:
    assert SA6_MIXED_CORRIDOR_PROBE.checkpoint.endswith(
        "sa6_k8_obb_corridor_deepen_from_iter160_s42/checkpoint_19200.pt"
    )
    assert SA6_MIXED_CORRIDOR_PROBE.no_resume_optimizer is False
    assert SA6_MIXED_CORRIDOR_PROBE.timesteps == 30 * 128
    assert SA6_MIXED_CORRIDOR_PROBE.save_interval == 5
    assert SA6_MIXED_CORRIDOR_PROBE.long_corridor_fraction == 0.10
    assert SA6_MIXED_CORRIDOR_PROBE.long_corridor_static_obstacles == 4
    assert SA6_MIXED_CORRIDOR_PROBE.long_corridor_dynamic_obstacles == 2
    assert SA6_MIXED_CORRIDOR_PROBE.long_corridor_dynamic_motion_mode == "mixed"
    assert SA6_MIXED_CORRIDOR_PROBE.previous_stage_replay_fraction == 0.10
    assert SA6_MIXED_CORRIDOR_PROBE.narrow_passage_fraction == 0.12
    assert SA6_MIXED_CORRIDOR_PROBE.teacher_retention_weight == 0.30


def test_sa6_dual_retention_uses_disjoint_c20_and_c12_teachers() -> None:
    assert SA6_DUAL_RETENTION.checkpoint.endswith(
        "sa6_k8_obb_sa5replay10_recovery_c10_2u_s42/checkpoint_128.pt"
    )
    assert SA6_DUAL_RETENTION.no_resume_optimizer is False
    assert SA6_DUAL_RETENTION.timesteps == 128
    assert SA6_DUAL_RETENTION.save_interval == 1
    assert SA6_DUAL_RETENTION.teacher_retention_weight == 0.30
    assert SA6_DUAL_RETENTION.teacher_retention_checkpoint.endswith(
        "sa5_e2e_k8_obb_narrow_recovery_seg2_c1280_s42/checkpoint_1280.pt"
    )
    assert SA6_DUAL_RETENTION.previous_stage_teacher_retention_weight == 0.02
    assert SA6_DUAL_RETENTION.previous_stage_teacher_checkpoint.endswith(
        "sa5_e2e_k8_obb_corridor_teacher_distill_c12_stepnorm_s42/"
        "checkpoint_128.pt"
    )


def test_sa6_dual_retention_beta05_changes_only_general_anchor_strength() -> None:
    assert (
        SA6_DUAL_RETENTION_BETA05.checkpoint
        == SA6_DUAL_RETENTION.checkpoint
    )
    assert (
        SA6_DUAL_RETENTION_BETA05.teacher_retention_checkpoint
        == SA6_DUAL_RETENTION.teacher_retention_checkpoint
    )
    assert SA6_DUAL_RETENTION_BETA05.teacher_retention_weight == 0.30
    assert (
        SA6_DUAL_RETENTION_BETA05.previous_stage_teacher_checkpoint
        == SA6_DUAL_RETENTION.previous_stage_teacher_checkpoint
    )
    assert (
        SA6_DUAL_RETENTION_BETA05.previous_stage_teacher_retention_weight
        == 0.05
    )


def test_sa6_dual_retention_beta10_changes_only_general_anchor_strength() -> None:
    assert (
        SA6_DUAL_RETENTION_BETA10.checkpoint
        == SA6_DUAL_RETENTION.checkpoint
    )
    assert (
        SA6_DUAL_RETENTION_BETA10.teacher_retention_checkpoint
        == SA6_DUAL_RETENTION.teacher_retention_checkpoint
    )
    assert SA6_DUAL_RETENTION_BETA10.teacher_retention_weight == 0.30
    assert (
        SA6_DUAL_RETENTION_BETA10.previous_stage_teacher_checkpoint
        == SA6_DUAL_RETENTION.previous_stage_teacher_checkpoint
    )
    assert (
        SA6_DUAL_RETENTION_BETA10.previous_stage_teacher_retention_weight
        == 0.10
    )


def test_sa6_corridor_rung_changes_only_frontier_obstacle_mix() -> None:
    assert SA6_CORRIDOR_RUNG_3S1D.checkpoint.endswith(
        "sa6_k8_obb_sa5replay10_recovery_c10_2u_s42/checkpoint_128.pt"
    )
    assert SA6_CORRIDOR_RUNG_3S1D.no_resume_optimizer is False
    assert SA6_CORRIDOR_RUNG_3S1D.timesteps == 128
    assert SA6_CORRIDOR_RUNG_3S1D.save_interval == 1
    assert SA6_CORRIDOR_RUNG_3S1D.long_corridor_fraction == 0.10
    assert SA6_CORRIDOR_RUNG_3S1D.long_corridor_static_obstacles == 3
    assert SA6_CORRIDOR_RUNG_3S1D.long_corridor_dynamic_obstacles == 1
    assert SA6_CORRIDOR_RUNG_3S1D.teacher_retention_weight == 0.30
    assert SA6_CORRIDOR_RUNG_3S1D.previous_stage_replay_fraction == 0.10
    assert (
        SA6_CORRIDOR_RUNG_3S1D.previous_stage_teacher_retention_weight
        == 0.0
    )


def test_sa6_corridor_rung_sa5replay20_changes_only_replay_balance() -> None:
    assert (
        SA6_CORRIDOR_RUNG_3S1D_SA5_20.checkpoint
        == SA6_CORRIDOR_RUNG_3S1D.checkpoint
    )
    assert SA6_CORRIDOR_RUNG_3S1D_SA5_20.no_resume_optimizer is False
    assert SA6_CORRIDOR_RUNG_3S1D_SA5_20.timesteps == 128
    assert SA6_CORRIDOR_RUNG_3S1D_SA5_20.previous_stage_replay_fraction == 0.20
    assert SA6_CORRIDOR_RUNG_3S1D_SA5_20.narrow_passage_fraction == 0.12
    assert SA6_CORRIDOR_RUNG_3S1D_SA5_20.long_corridor_fraction == 0.10
    assert SA6_CORRIDOR_RUNG_3S1D_SA5_20.long_corridor_static_obstacles == 3
    assert SA6_CORRIDOR_RUNG_3S1D_SA5_20.long_corridor_dynamic_obstacles == 1
    assert SA6_CORRIDOR_RUNG_3S1D_SA5_20.teacher_retention_weight == 0.30
    assert (
        SA6_CORRIDOR_RUNG_3S1D_SA5_20.previous_stage_teacher_retention_weight
        == 0.0
    )


def test_sa6_c12_corridor_rung_starts_from_joint_margin_checkpoint() -> None:
    assert SA6_C12_CORRIDOR_RUNG_2S1D.checkpoint.endswith(
        "sa5_e2e_k8_obb_corridor_teacher_distill_c12_stepnorm_s42/"
        "checkpoint_128.pt"
    )
    assert SA6_C12_CORRIDOR_RUNG_2S1D.no_resume_optimizer is False
    assert SA6_C12_CORRIDOR_RUNG_2S1D.timesteps == 128
    assert SA6_C12_CORRIDOR_RUNG_2S1D.save_interval == 1
    assert SA6_C12_CORRIDOR_RUNG_2S1D.previous_stage_replay_fraction == 0.20
    assert SA6_C12_CORRIDOR_RUNG_2S1D.narrow_passage_fraction == 0.12
    assert SA6_C12_CORRIDOR_RUNG_2S1D.long_corridor_fraction == 0.10
    assert SA6_C12_CORRIDOR_RUNG_2S1D.long_corridor_static_obstacles == 2
    assert SA6_C12_CORRIDOR_RUNG_2S1D.long_corridor_dynamic_obstacles == 1
    assert SA6_C12_CORRIDOR_RUNG_2S1D.teacher_retention_weight == 0.30
    assert (
        SA6_C12_CORRIDOR_RUNG_2S1D.previous_stage_teacher_retention_weight
        == 0.0
    )


def test_sa6_c12_corridor_cont2_preserves_rung_and_optimizer() -> None:
    assert SA6_C12_CORRIDOR_RUNG_2S1D_CONT2.checkpoint.endswith(
        "sa6_k8_obb_c12_corridor_rung2s1d_1u_s42/checkpoint_128.pt"
    )
    assert SA6_C12_CORRIDOR_RUNG_2S1D_CONT2.no_resume_optimizer is False
    assert SA6_C12_CORRIDOR_RUNG_2S1D_CONT2.timesteps == 128
    assert SA6_C12_CORRIDOR_RUNG_2S1D_CONT2.save_interval == 1
    assert (
        SA6_C12_CORRIDOR_RUNG_2S1D_CONT2.previous_stage_replay_fraction
        == SA6_C12_CORRIDOR_RUNG_2S1D.previous_stage_replay_fraction
        == 0.20
    )
    assert (
        SA6_C12_CORRIDOR_RUNG_2S1D_CONT2.long_corridor_static_obstacles
        == 2
    )
    assert (
        SA6_C12_CORRIDOR_RUNG_2S1D_CONT2.long_corridor_dynamic_obstacles
        == 1
    )
    assert SA6_C12_CORRIDOR_RUNG_2S1D_CONT2.teacher_retention_weight == 0.30


def test_sa6_gate2_recovery_keeps_corridor_and_narrow_replay() -> None:
    assert SA6_CORRIDOR_RUNG_3S1D_SA5_RECOVERY40.checkpoint.endswith(
        "sa6_k8_obb_corridor_rung3s1d_sa5replay20_c128_1u_s42/"
        "checkpoint_128.pt"
    )
    assert SA6_CORRIDOR_RUNG_3S1D_SA5_RECOVERY40.no_resume_optimizer is False
    assert SA6_CORRIDOR_RUNG_3S1D_SA5_RECOVERY40.timesteps == 128
    assert (
        SA6_CORRIDOR_RUNG_3S1D_SA5_RECOVERY40.previous_stage_replay_fraction
        == 0.40
    )
    assert SA6_CORRIDOR_RUNG_3S1D_SA5_RECOVERY40.narrow_passage_fraction == 0.12
    assert SA6_CORRIDOR_RUNG_3S1D_SA5_RECOVERY40.long_corridor_fraction == 0.10
    assert (
        SA6_CORRIDOR_RUNG_3S1D_SA5_RECOVERY40.long_corridor_static_obstacles
        == 3
    )
    assert (
        SA6_CORRIDOR_RUNG_3S1D_SA5_RECOVERY40.long_corridor_dynamic_obstacles
        == 1
    )
    assert (
        SA6_CORRIDOR_RUNG_3S1D_SA5_RECOVERY40.teacher_retention_weight
        == 0.30
    )
