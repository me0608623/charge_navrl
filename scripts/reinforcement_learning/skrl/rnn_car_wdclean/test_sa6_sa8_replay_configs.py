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
from rnn_car_modular.configs.e2e_sa6_k8_obb_corridor_env_stratified_probe_from_c100 import (
    CONFIG as SA6_ENV_STRATIFIED_CORRIDOR_PROBE,
)
from rnn_car_modular.configs.e2e_sa6_k8_obb_corridor_weighted_stratified_d1_from_d0 import (
    CONFIG as SA6_WEIGHTED_STRATIFIED_D1,
)
from rnn_car_modular.configs.e2e_sa6_w1_wander_from_d0 import (
    CONFIG as SA6_W1_WANDER,
)
from rnn_car_modular.configs.e2e_sa6_w2_wander_lrdecay_from_d0 import (
    CONFIG as SA6_W2_WANDER_LRDECAY,
)
from rnn_car_modular.configs.e2e_sa6_n1_direct_imitation_from_d0 import (
    CONFIG as SA6_N1_DIRECT_IMITATION,
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


def test_sa6_env_stratified_probe_changes_only_motion_and_budget() -> None:
    assert SA6_ENV_STRATIFIED_CORRIDOR_PROBE.checkpoint.endswith(
        "sa6_k8_obb_corridor_priority_c30_s42/checkpoint_12800.pt"
    )
    assert SA6_ENV_STRATIFIED_CORRIDOR_PROBE.no_resume_optimizer is False
    assert SA6_ENV_STRATIFIED_CORRIDOR_PROBE.timesteps == 30 * 128
    assert SA6_ENV_STRATIFIED_CORRIDOR_PROBE.save_interval == 5
    assert SA6_ENV_STRATIFIED_CORRIDOR_PROBE.long_corridor_fraction == 0.10
    assert (
        SA6_ENV_STRATIFIED_CORRIDOR_PROBE.long_corridor_static_obstacles == 4
    )
    assert (
        SA6_ENV_STRATIFIED_CORRIDOR_PROBE.long_corridor_dynamic_obstacles == 2
    )
    assert (
        SA6_ENV_STRATIFIED_CORRIDOR_PROBE.long_corridor_dynamic_motion_mode
        == "env_stratified"
    )
    assert (
        SA6_ENV_STRATIFIED_CORRIDOR_PROBE.previous_stage_replay_fraction
        == 0.10
    )
    assert SA6_ENV_STRATIFIED_CORRIDOR_PROBE.narrow_passage_fraction == 0.12
    assert SA6_ENV_STRATIFIED_CORRIDOR_PROBE.teacher_retention_weight == 0.30


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


# ---------------------------------------------------------------------------
# D1 weighted env-stratified corridor replay
# ---------------------------------------------------------------------------


def test_d1_changes_only_the_motion_weights_versus_d0():
    """D1 must be a single-variable change on top of D0.

    Anything beyond metadata, the warm-start checkpoint and the new weights
    would break the causal attribution of the D1 result.
    """
    d0 = SA6_ENV_STRATIFIED_CORRIDOR_PROBE
    d1 = SA6_WEIGHTED_STRATIFIED_D1
    allowed = {
        "name",
        "description",
        "notes",
        "tags",
        "checkpoint",
        "long_corridor_dynamic_motion_weights",
    }
    changed = {
        field
        for field in d0.__dataclass_fields__
        if getattr(d0, field) != getattr(d1, field)
    }
    assert changed == allowed, f"unexpected D1 field changes: {changed - allowed}"


def test_d1_run_shape_and_resume():
    d1 = SA6_WEIGHTED_STRATIFIED_D1
    assert d1.timesteps // d1.rollout_length == 30
    assert d1.save_interval == 5
    assert d1.no_resume_optimizer is False
    assert d1.ppo_epochs > 0
    assert d1.checkpoint.endswith(
        "sa6_k8_obb_corridor_env_stratified_probe_c100_s42/checkpoint_3840.pt"
    )


def test_d1_weights_and_corridor_share():
    d0 = SA6_ENV_STRATIFIED_CORRIDOR_PROBE
    d1 = SA6_WEIGHTED_STRATIFIED_D1
    assert d1.long_corridor_dynamic_motion_mode == "env_stratified"
    assert d1.long_corridor_dynamic_motion_weights == (0.30, 0.10, 0.60)
    # the corridor share itself must not move; only its internal mix does
    assert d1.long_corridor_fraction == d0.long_corridor_fraction


def test_baseline_configs_keep_weights_unset():
    for config in (SA6, SA6_MIXED_CORRIDOR_PROBE, SA6_ENV_STRATIFIED_CORRIDOR_PROBE):
        assert config.long_corridor_dynamic_motion_weights is None


# ---------------------------------------------------------------------------
# W1 wander corridor kinematics arm
# ---------------------------------------------------------------------------


def test_w1_changes_only_the_random_2d_kinematics_versus_d0():
    """W1 must be a single-variable change on top of D0 — permanently.

    D1 taught that reweighting the corridor mix costs the down-weighted
    families without helping the target one; W1 therefore keeps the balanced
    1:1:1 stratification and changes only the random_2d kinematics. Any field
    beyond metadata, the warm-start checkpoint and the kinematics switch
    breaks the causal attribution of the W1 result and must fail here.
    """
    d0 = SA6_ENV_STRATIFIED_CORRIDOR_PROBE
    w1 = SA6_W1_WANDER
    allowed = {
        "name",
        "description",
        "notes",
        "tags",
        "checkpoint",
        "long_corridor_random_2d_kinematics",
    }
    changed = {
        field
        for field in d0.__dataclass_fields__
        if getattr(d0, field) != getattr(w1, field)
    }
    assert changed == allowed, f"unexpected W1 field changes: {changed ^ allowed}"


def test_w1_run_shape_resume_and_kinematics():
    w1 = SA6_W1_WANDER
    assert w1.long_corridor_random_2d_kinematics == "wander"
    assert w1.timesteps // w1.rollout_length == 30
    assert w1.save_interval == 5
    assert w1.no_resume_optimizer is False
    assert w1.checkpoint.endswith(
        "sa6_k8_obb_corridor_env_stratified_probe_c100_s42/checkpoint_3840.pt"
    )
    # Sampling stays balanced: no weights, env_stratified mode, 10% share.
    assert w1.long_corridor_dynamic_motion_weights is None
    assert w1.long_corridor_dynamic_motion_mode == "env_stratified"
    assert w1.long_corridor_fraction == SA6_ENV_STRATIFIED_CORRIDOR_PROBE.long_corridor_fraction


# ---------------------------------------------------------------------------
# W2 learning-rate-decay arm
# ---------------------------------------------------------------------------


def test_w2_changes_only_the_optimizer_schedule_versus_w1():
    """W2 must differ from W1 only in the optimizer schedule — permanently.

    W1 and D1 both peaked around iter 10 under a constant LR from a mature
    checkpoint; the ruled follow-up is a single-variable LR-decay arm, NOT a
    recipe change. Any diff beyond metadata, the shorter 20-iteration budget,
    the denser save cadence and lr_decay itself would break the causal
    attribution of the W2 result and must fail here. In particular the 12%
    narrow-gap replay, teacher KL and the 10% corridor share are inherited
    from W1 untouched.
    """
    w1 = SA6_W1_WANDER
    w2 = SA6_W2_WANDER_LRDECAY
    allowed = {
        "name",
        "description",
        "notes",
        "tags",
        "timesteps",
        "save_interval",
        "lr_decay",
    }
    changed = {
        field
        for field in w1.__dataclass_fields__
        if getattr(w1, field) != getattr(w2, field)
    }
    assert changed == allowed, f"unexpected W2 field changes: {changed ^ allowed}"


def test_w2_run_shape_decay_and_inheritance():
    w2 = SA6_W2_WANDER_LRDECAY
    # Codex-ruled run shape: D0 warm start, seed 42, 20 iters, ckpt every 2.
    assert w2.checkpoint == SA6_W1_WANDER.checkpoint
    assert w2.checkpoint.endswith(
        "sa6_k8_obb_corridor_env_stratified_probe_c100_s42/checkpoint_3840.pt"
    )
    assert w2.seed == 42
    assert w2.timesteps // w2.rollout_length == 20
    assert w2.save_interval == 2
    assert w2.no_resume_optimizer is False
    # The one new training variable: linear LR decay 0.05/iter
    # (2e-4 -> 1e-4 @ iter10 -> ~0 @ iter20, floor 1% via trainer clamp).
    assert w2.lr_decay == 0.05
    # Inherited W1 recipe: wander kinematics, balanced 1:1:1, 10% corridor.
    assert w2.long_corridor_random_2d_kinematics == "wander"
    assert w2.long_corridor_dynamic_motion_weights is None
    assert w2.long_corridor_dynamic_motion_mode == "env_stratified"
    assert w2.long_corridor_fraction == SA6_W1_WANDER.long_corridor_fraction


# ---------------------------------------------------------------------------
# N1 direct-crossing imitation arm
# ---------------------------------------------------------------------------


def test_n1_changes_only_the_narrow_supervision_versus_w1():
    """N1 must swap the narrow supervision and nothing else — permanently.

    W2 established that LR decay is not reproducible across seeds and was
    archived as a diagnostic. N1 goes back to the W1 recipe and changes one
    thing: the SA5 narrow teacher KL (that teacher detours — 3 seeds
    n=2094, crossing 95.9% but direct 0.000) is removed and replaced by a
    cross-entropy against the scripted direct-crossing teacher, applied only
    on narrow-replay frames. Reward, network, PPO, replay mix, corridor
    share and LR schedule must all stay exactly as W1.
    """
    w1 = SA6_W1_WANDER
    n1 = SA6_N1_DIRECT_IMITATION
    allowed = {
        "name",
        "description",
        "notes",
        "tags",
        "timesteps",
        "save_interval",
        "teacher_retention_weight",
        "narrow_imitation_weight",
        "narrow_passage_goal_distance_range",
        "narrow_passage_goal_lateral_offset_range",
    }
    changed = {
        field
        for field in w1.__dataclass_fields__
        if getattr(w1, field) != getattr(n1, field)
    }
    assert changed == allowed, f"unexpected N1 field changes: {changed ^ allowed}"


def test_n1_removes_the_detouring_teacher_and_keeps_rollouts_clean():
    n1 = SA6_N1_DIRECT_IMITATION
    # The detouring SA5 teacher KL must be gone, not merely down-weighted.
    assert n1.teacher_retention_weight == 0.0
    # Imitation is on, and shadow mode is off (shadow is for lambda calibration).
    assert n1.narrow_imitation_weight > 0.0
    assert n1.narrow_imitation_shadow is False
    # The teacher must never drive: no rollout override anywhere.
    assert n1.teacher_retention_rollout_override is False
    assert n1.corridor_teacher_intervention_only is False
    assert n1.corridor_teacher_distill_epochs == 0


def test_n1_run_shape_and_inherited_recipe():
    n1 = SA6_N1_DIRECT_IMITATION
    assert n1.checkpoint == SA6_W1_WANDER.checkpoint
    assert n1.checkpoint.endswith(
        "sa6_k8_obb_corridor_env_stratified_probe_c100_s42/checkpoint_3840.pt"
    )
    assert n1.seed == 42
    assert n1.timesteps // n1.rollout_length == 10
    assert n1.save_interval == 2
    assert n1.no_resume_optimizer is False
    # W2's LR decay is archived — N1 runs at the constant W1 learning rate.
    assert n1.lr_decay == 0.0
    # Inherited W1 recipe: wander kinematics, balanced 1:1:1, 10% corridor,
    # and the 12% narrow replay whose frames the imitation term acts on.
    assert n1.long_corridor_random_2d_kinematics == "wander"
    assert n1.long_corridor_dynamic_motion_weights is None
    assert n1.long_corridor_dynamic_motion_mode == "env_stratified"
    assert n1.long_corridor_fraction == SA6_W1_WANDER.long_corridor_fraction
    assert n1.narrow_passage_fraction == SA6_W1_WANDER.narrow_passage_fraction
    assert n1.narrow_passage_goal_distance_range == (2.0, 4.0)
    assert n1.narrow_passage_goal_lateral_offset_range == (-1.5, 1.5)
    assert SA6_W1_WANDER.narrow_passage_goal_distance_range is None
    assert SA6_W1_WANDER.narrow_passage_goal_lateral_offset_range is None


def test_sa7_keeps_narrow_and_corridor_replay_on():
    """SA7 must keep replaying narrow gaps and corridors — permanently.

    The 2026-07-27 roadmap is explicit: these cannot be a one-off remedial
    block, because the denser SA7/SA8 scenes wash the capability out
    otherwise. The 68/10/12/10 mix is therefore inherited untouched from
    e2e_sa7_k8_obb, and this test fails if a future edit trims either share.
    """
    from rnn_car_modular.configs.e2e_sa7_k8_obb import CONFIG as sa7_base
    from rnn_car_modular.configs.e2e_sa7_wander_from_w1c10 import (
        CONFIG as sa7_wander,
    )

    assert sa7_wander.narrow_passage_fraction == 0.12
    assert sa7_wander.long_corridor_fraction == 0.10
    assert sa7_wander.previous_stage_replay_fraction == 0.10
    # Inherited, not re-specified — a drift in the base must surface here too.
    assert sa7_wander.narrow_passage_fraction == sa7_base.narrow_passage_fraction
    assert sa7_wander.long_corridor_fraction == sa7_base.long_corridor_fraction


def test_sa7_changes_only_warm_start_corridor_kinematics_and_density():
    """SA7 must differ from its historical base in exactly three ways.

    第三項是 2026-07-27 裁決的高密度混合分佈（逐 env 抽 (S, D)，上限
    5S+5D）。清單是**鎖**：任何沒被裁決核准的欄位變動都必須讓這個測試紅掉。
    """
    from rnn_car_modular.configs.e2e_sa7_k8_obb import CONFIG as sa7_base
    from rnn_car_modular.configs.e2e_sa7_wander_from_w1c10 import (
        CONFIG as sa7_wander,
    )

    allowed = {
        "name",
        "description",
        "notes",
        "tags",
        "checkpoint",
        "long_corridor_random_2d_kinematics",
        "long_corridor_dynamic_motion_mode",
        "long_corridor_obstacle_count_mix",
    }
    changed = {
        field
        for field in sa7_base.__dataclass_fields__
        if getattr(sa7_base, field) != getattr(sa7_wander, field)
    }
    assert changed == allowed, f"unexpected SA7 field changes: {changed ^ allowed}"


def test_sa7_runs_at_the_thirteen_metre_arena():
    from rnn_car_modular.configs.e2e_sa7_wander_from_w1c10 import (
        CONFIG as sa7_wander,
    )

    assert sa7_wander.initial_stage == 7
    # room_size is the half extent: 6.5 -> a 13 x 13 m arena.
    assert sa7_wander.room_size == 6.5
    assert sa7_wander.long_corridor_random_2d_kinematics == "wander"
    assert sa7_wander.checkpoint.endswith(
        "sa6_k8_obb_corridor_w1_wander_s42/checkpoint_1280.pt"
    )


def test_n1_is_blocked_because_its_premise_was_refuted():
    """N1 must stay unlaunchable until the SA8/12m question is answered.

    The premise ("the policy cannot cross a narrow gap straight") was an
    evaluation artefact: `NarrowGapSpec.arena_half_extent` was hard-coded to
    5.0 and `--arena_size` was never passed in, so the barrier only sealed at
    a 10 m arena. With it genuinely sealed at the training arena size, D0
    scores direct 1.000 over 2304 episodes, 0% collisions, 0.052 m median
    pre-cross lateral. Unsetting either flag below silently re-enables an arm
    that teaches a behaviour the policy already performs perfectly.
    """
    from rnn_car_modular.configs import e2e_sa6_n1_direct_imitation_from_d0 as n1_mod

    assert n1_mod._ARM_IS_BLOCKED, (
        "N1's premise was refuted on 2026-07-27; unblock only together with a "
        "fresh 12 m shadow calibration"
    )
    assert n1_mod._BLOCKED_REASON, "a blocked arm must state why"
    # The SA6/14m measurement is void for the SA8/12m setting it was deferred to.
    assert not n1_mod._LAMBDA_IS_CALIBRATED, (
        "0.067 belongs to SA6/14m; re-derive it against the 12 m recipe"
    )
    # Kept on record so the re-derivation has a documented baseline.
    assert n1_mod._LAMBDA == 0.067
    assert n1_mod.CONFIG.narrow_imitation_weight == n1_mod._LAMBDA


def test_n1_does_not_add_a_d0_anchor_or_actuator_delay():
    """Both were explicitly deferred until direct crossing is solved."""
    n1 = SA6_N1_DIRECT_IMITATION
    assert n1.previous_stage_teacher_checkpoint is None
    assert n1.previous_stage_teacher_retention_weight == 0.0


def test_pre_n1_configs_keep_imitation_off():
    for config in (
        SA6,
        SA6_ENV_STRATIFIED_CORRIDOR_PROBE,
        SA6_WEIGHTED_STRATIFIED_D1,
        SA6_W1_WANDER,
        SA6_W2_WANDER_LRDECAY,
    ):
        assert config.narrow_imitation_weight == 0.0, config.name
        assert config.narrow_imitation_shadow is False, config.name
        assert config.narrow_passage_goal_distance_range is None, config.name
        assert config.narrow_passage_goal_lateral_offset_range is None, config.name


def test_every_pre_w1_config_keeps_patrol_kinematics():
    """Historical configs must stay on the frozen patrol kinematics."""
    for config in (
        SA6,
        SA6_MIXED_CORRIDOR_PROBE,
        SA6_ENV_STRATIFIED_CORRIDOR_PROBE,
        SA6_WEIGHTED_STRATIFIED_D1,
    ):
        assert config.long_corridor_random_2d_kinematics == "patrol", config.name


# ---------------------------------------------------------------------------
# Suite profiles: the deployment gate must be explicit, defaults must stay
# legacy patrol.
# ---------------------------------------------------------------------------


def test_suite_profiles_lock_deployment_and_legacy_definitions():
    import importlib.util
    from pathlib import Path

    suite_path = Path(__file__).resolve().parent / "run_corridor_motion_suite.py"
    spec = importlib.util.spec_from_file_location("_suite_for_test", suite_path)
    suite = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(suite)

    deploy = suite.PROFILES["deployment_wander_v1"]
    assert deploy["modes"] == ("lateral", "longitudinal", "random_2d", "mixed_iid")
    assert deploy["random_2d_kinematics"] == "wander"
    legacy = suite.PROFILES["legacy_patrol_v1"]
    assert legacy["modes"] == suite.MODES
    assert legacy["random_2d_kinematics"] == "patrol"

    # No profile + no overrides -> historical behaviour exactly.
    modes, kin, complete = suite.resolve_profile(None, None, None)
    assert modes is None and kin is None and complete is None
    # Full profile run.
    modes, kin, complete = suite.resolve_profile(
        "deployment_wander_v1", None, None
    )
    assert modes == deploy["modes"] and kin == "wander" and complete is True
    # Subset is allowed for sentinels but flagged incomplete.
    modes, kin, complete = suite.resolve_profile(
        "deployment_wander_v1", ("random_2d",), None
    )
    assert modes == ("random_2d",) and kin == "wander" and complete is False
    # The profile is a frozen contract: a contradicting kinematics flag and a
    # mode outside the profile must both fail loudly, never silently mislabel.
    import pytest as _pytest
    with _pytest.raises(ValueError):
        suite.resolve_profile("deployment_wander_v1", None, "patrol")
    with _pytest.raises(ValueError):
        suite.resolve_profile("deployment_wander_v1", ("mixed",), None)
    # Matching explicit kinematics is redundant but not contradictory.
    _, kin, _ = suite.resolve_profile("deployment_wander_v1", None, "wander")
    assert kin == "wander"


def test_suite_verdict_semantics_lock_subset_vs_full_profile():
    """PASS semantics must not outrun coverage (Codex second-review finding).

    Three locked cases: full profile, subset PASS, subset FAIL. A subset run
    must never emit the ALL= token nor set all_modes_pass, so a supervisor
    grepping ALL=PASS cannot mistake a sentinel for a deployment JOINT PASS.
    """
    import importlib.util
    from pathlib import Path

    suite_path = Path(__file__).resolve().parent / "run_corridor_motion_suite.py"
    spec = importlib.util.spec_from_file_location("_suite_verdict_test", suite_path)
    suite = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(suite)

    # Full profile, all modes pass -> genuine JOINT PASS.
    v = suite.summarize_verdict(True, True)
    assert v["all_modes_pass"] is True
    assert v["profile_joint_pass"] is True
    assert "ALL=PASS" in v["banner"] and "JOINT=PASS" in v["banner"]

    # Full profile, a mode failed.
    v = suite.summarize_verdict(False, True)
    assert v["all_modes_pass"] is False
    assert v["profile_joint_pass"] is False
    assert "ALL=FAIL" in v["banner"] and "JOINT=FAIL" in v["banner"]

    # Subset sentinel PASS: exit-0 semantics allowed, but never ALL=PASS,
    # never all_modes_pass, JOINT is N/A.
    v = suite.summarize_verdict(True, False)
    assert v["all_selected_modes_pass"] is True
    assert v["all_modes_pass"] is False
    assert v["profile_joint_pass"] is False
    assert "ALL=" not in v["banner"]
    assert "SELECTED=PASS" in v["banner"] and "JOINT=N/A" in v["banner"]

    # Subset sentinel FAIL.
    v = suite.summarize_verdict(False, False)
    assert v["all_selected_modes_pass"] is False
    assert v["all_modes_pass"] is False
    assert "SELECTED=FAIL" in v["banner"] and "ALL=" not in v["banner"]

    # No profile -> historical banner exactly.
    v = suite.summarize_verdict(True, None)
    assert v["all_modes_pass"] is True
    assert v["profile_joint_pass"] is None
    assert v["banner"] == "ALL=PASS"
    v = suite.summarize_verdict(False, None)
    assert v["banner"] == "ALL=FAIL"


def test_sa7_density_mix_matches_the_approved_distribution():
    """高密度分佈必須逐項等於裁決數字，且權重合計為 1。"""
    from rnn_car_modular.configs.e2e_sa7_wander_from_w1c10 import (
        CONFIG as sa7_wander,
    )

    # ── CANONICAL LOCK ──────────────────────────────────────────────────
    # 這是 SA7 密度分佈的**唯一**字面值來源，逐字對齊
    # experiment_config.py:163 的凍結規格：
    #   25% 3S1D / 35% 4S2D / 20% 4S3D / 15% 5S3D / 5% 5S5D
    # 其他任何測試都必須 import SA7_DENSITY_MIX，不得自行重打一張表。
    assert sa7_wander.long_corridor_obstacle_count_mix == (
        ((3, 1), 0.25),
        ((4, 2), 0.35),
        ((4, 3), 0.20),
        ((5, 3), 0.15),
        ((5, 5), 0.05),
    )
    # 平均動態數 2.25 —— 這是 retention replay 的量級。高於它就變成高密壓測，
    # policy 會學成遇到人就停車。
    mean_dynamic = sum(d * w for (_, d), w in sa7_wander.long_corridor_obstacle_count_mix)
    assert abs(mean_dynamic - 2.25) < 1e-9, mean_dynamic
    # D1/D2 必須存在：移除低密度等於讓 policy 沒看過稀疏走廊。
    present = {d for (_, d), _ in sa7_wander.long_corridor_obstacle_count_mix}
    assert 1 in present and 2 in present, present
    weights = [w for _, w in sa7_wander.long_corridor_obstacle_count_mix]
    assert abs(sum(weights) - 1.0) < 1e-9
    # 上限 5S+5D —— 超過就代表有人偷偷放寬了走廊容量。
    for (static, dynamic), _ in sa7_wander.long_corridor_obstacle_count_mix:
        assert 0 <= static <= 5 and 0 <= dynamic <= 5


def test_sa71_changes_only_the_corridor_split_and_nothing_else():
    """SA7.1 相對 SA7 只准差一件事：走廊 10% -> 20% + 一半換成 Gate 題型。

    reward / network / LR / seed / 窄縫比例 / 前階段 replay 全部必須逐項相同。
    這條測試的價值在於：SA7.1 若沒過閘，我們要能斷言「差別只有題型分佈」，
    否則就沒有可歸因的實驗。
    """
    from rnn_car_modular.configs.e2e_sa7_wander_from_w1c10 import (
        CONFIG as sa7,
        SA7_DENSITY_MIX,
    )
    from rnn_car_modular.configs.e2e_sa71_gate_aligned_from_w1c10 import CONFIG as sa71

    # 允許差異：走廊總量與 gate 分流。
    assert sa7.long_corridor_fraction == 0.10
    assert sa71.long_corridor_fraction == 0.20
    assert sa7.long_corridor_gate_aligned_share == 0.0
    assert sa71.long_corridor_gate_aligned_share == 0.5

    # 全域曝光：真實混合 10%、Gate 題型 10%、native 58%。
    gate_global = sa71.long_corridor_fraction * sa71.long_corridor_gate_aligned_share
    mixed_global = sa71.long_corridor_fraction - gate_global
    native = 1.0 - (
        sa71.previous_stage_replay_fraction
        + sa71.narrow_passage_fraction
        + sa71.long_corridor_fraction
    )
    assert abs(gate_global - 0.10) < 1e-9, gate_global
    assert abs(mixed_global - 0.10) < 1e-9, mixed_global
    assert abs(native - 0.58) < 1e-9, native

    # 密度分佈仍是同一個凍結表物件 —— 不得為 SA7.1 另開一張。
    assert sa71.long_corridor_obstacle_count_mix is SA7_DENSITY_MIX

    # 其餘所有欄位逐項相同。
    from dataclasses import fields

    allowed = {
        "name",
        "description",
        "notes",
        "tags",
        "long_corridor_fraction",
        "long_corridor_gate_aligned_share",
    }
    for f in fields(sa7):
        if f.name in allowed:
            continue
        assert getattr(sa7, f.name) == getattr(sa71, f.name), f.name

    # 暖啟動點必須跟 SA7 同一顆 W1-c10（診斷結論：SA7 沒產出更好的 ckpt）。
    assert sa71.checkpoint == sa7.checkpoint
    assert sa71.checkpoint.endswith("sa6_k8_obb_corridor_w1_wander_s42/checkpoint_1280.pt")
    assert sa71.no_resume_optimizer is True


def test_actuator_delay_bridge_diff_vs_w1_parent_is_exactly_the_allowed_set():
    """Actuator-delay bridge 相對 W1 parent 的差異必須**全欄位**受控。

    這條用 `dataclasses.fields` 逐欄比對，而不是手挑幾個欄位斷言 ——
    手挑只能證明「我檢查的那幾項沒變」，證明不了「其他欄位也沒變」。
    bridge 的整個價值就在於 actuator DR bundle 是唯一新增的行為因子；
    這個 bundle 由 delay、velocity scale、motor lag 組成，不是 delay-only
    消融。只要另有第二個 config 因子悄悄跟著改，橋接失敗時就無法歸因
    （SA7.1 的 `long_corridor_fraction`
    一個欄位承載兩個因子，正是這種錯誤的代價）。
    """
    from dataclasses import fields

    from rnn_car_modular.configs.e2e_sa6_w1_wander_from_d0 import CONFIG as w1
    from rnn_car_modular.configs.e2e_deploy_bridge_actuator_delay_from_w1c10 import (
        CONFIG as bridge,
        W1_C10_CHECKPOINT,
        BRIDGE_TIMESTEPS,
        BRIDGE_SAVE_INTERVAL,
        ACTUATOR_DELAY_RANGE,
        ACTUATOR_VELOCITY_SCALE,
        ACTUATOR_MOTOR_LAG,
    )

    metadata = {"name", "description", "tags", "notes"}
    #: 允許差異的**行為**欄位。actuator_* 與 obs_delay_steps 雖然在 config 裡
    #: 明寫，實際與 parent 同值（見下方逐項斷言），列在這裡是允許而非要求。
    allowed_behavioural = {
        "checkpoint",
        "timesteps",
        "save_interval",
        "no_resume_optimizer",
        "enable_actuator_dr",
        "actuator_delay_range",
        "actuator_velocity_scale",
        "actuator_motor_lag",
        "obs_delay_steps",
    }
    allowed = metadata | allowed_behavioural

    differing = {
        f.name
        for f in fields(w1)
        if getattr(w1, f.name) != getattr(bridge, f.name)
    }
    unexpected = differing - allowed
    assert not unexpected, f"bridge 動到了未核准的欄位: {sorted(unexpected)}"

    # 唯一新增的行為因子必須是致動器 DR bundle 開關 + 暖啟動/預算 metadata。
    assert differing - metadata == {
        "checkpoint",
        "timesteps",
        "save_interval",
        "enable_actuator_dr",
    }, sorted(differing - metadata)

    # ── 致動器規格（歷史規格，凍結）────────────────────────────────
    assert bridge.enable_actuator_dr is True
    assert bridge.actuator_delay_range == (0, 2) == ACTUATOR_DELAY_RANGE
    # U{0,1,2}：兩端 inclusive，control_dt 0.2 s -> 0/200/400 ms。
    lo, hi = bridge.actuator_delay_range
    assert [round(d * 0.2 * 1000) for d in range(lo, hi + 1)] == [0, 200, 400]
    # 不得塌縮成固定延遲。
    assert lo != hi, "delay 被改成固定值；歷史規格是 U{0,1,2}"
    assert bridge.actuator_velocity_scale == (0.9, 1.1) == ACTUATOR_VELOCITY_SCALE
    assert bridge.actuator_motor_lag == 0.3 == ACTUATOR_MOTOR_LAG

    # ── 觀測延遲永遠關 ─────────────────────────────────────────────
    # 觀測延遲曾被誤植為「馬達延遲」，並被單變因隔離證實是 stage3
    # deterministic 滿舵塌縮的兇手。致動延遲保留，觀測延遲永遠關。
    assert bridge.obs_delay_steps == (0, 0)
    assert w1.obs_delay_steps == (0, 0), "parent 前提改變"

    # ── OBB lineage 的 action history 必須原封不動 ─────────────────
    assert bridge.use_action_history is True
    assert bridge.use_action_history == w1.use_action_history

    # ── 暖啟動與預算 ───────────────────────────────────────────────
    assert bridge.checkpoint == W1_C10_CHECKPOINT
    assert bridge.checkpoint.endswith(
        "sa6_k8_obb_corridor_w1_wander_s42/checkpoint_1280.pt"
    )
    # 不得誤用未過閘的 SA7 / SA7.1 checkpoint。
    assert "sa7" not in bridge.checkpoint.lower()
    assert bridge.timesteps == BRIDGE_TIMESTEPS == 1280      # 10 iter × 128 步
    assert bridge.save_interval == BRIDGE_SAVE_INTERVAL == 2
    assert bridge.timesteps // 128 == 10

    # model + Adam 都續用：10 個 iteration 的預算付不起動量重置。
    assert bridge.no_resume_optimizer is False

    # ── 其餘關鍵欄位逐項等同 parent ────────────────────────────────
    for name in ("num_envs", "seed", "no_domain_randomization"):
        assert getattr(bridge, name) == getattr(w1, name), name
    assert bridge.num_envs == 1024
    assert bridge.seed == 42

    # ── 標成診斷/預部署，不得被誤讀為 final accepted policy ────────
    assert "pre_deployment" in bridge.tags
    assert "diagnostic" in bridge.tags


def test_actuator_bridge_continuation_only_changes_resume_metadata_and_budget():
    """c10 continuation must not introduce a second behavioural factor."""
    from dataclasses import fields

    from rnn_car_modular.configs.e2e_deploy_bridge_actuator_delay_from_w1c10 import (
        CONFIG as bridge,
    )
    from rnn_car_modular.configs.e2e_deploy_bridge_actuator_delay_continue_c10 import (
        BRIDGE_C10_CHECKPOINT,
        CONFIG as continuation,
        CONTINUATION_SAVE_INTERVAL,
        CONTINUATION_TIMESTEPS,
    )

    allowed = {
        "name",
        "description",
        "tags",
        "notes",
        "checkpoint",
        "timesteps",
        "save_interval",
    }
    differing = {
        field.name
        for field in fields(bridge)
        if getattr(bridge, field.name) != getattr(continuation, field.name)
    }
    assert differing <= allowed, sorted(differing - allowed)
    assert differing == allowed

    assert continuation.checkpoint == BRIDGE_C10_CHECKPOINT
    assert continuation.checkpoint.endswith(
        "bridge_actdelay_w1c10_s42_r2_cmdqueue/checkpoint_1280.pt"
    )
    assert continuation.timesteps == CONTINUATION_TIMESTEPS == 640
    assert continuation.timesteps // 128 == 5
    assert continuation.save_interval == CONTINUATION_SAVE_INTERVAL == 1
    assert continuation.no_resume_optimizer is False

    for field in fields(bridge):
        if field.name in allowed:
            continue
        assert getattr(bridge, field.name) == getattr(
            continuation, field.name
        ), field.name
