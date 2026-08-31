import json

import numpy as np
import pytest

from rnn_car_wdclean.analyze_feasibility_mask_decomposition import (
    DIAGNOSTIC_SCHEMA,
    FUNNEL_MASK_NAMES,
    _load_archive,
    analyze_cell,
    analyze_suite,
    classify_candidates_cleared,
    render_markdown,
)


def _masks(*, shape=(1,), **overrides):
    if overrides:
        shape = next(iter(overrides.values())).shape
    base = {name: np.zeros(shape, dtype=np.int64) for name in FUNNEL_MASK_NAMES}
    base.update(overrides)
    return {name: base[name] for name in FUNNEL_MASK_NAMES}


def test_has_candidates_when_final_is_nonzero():
    code = classify_candidates_cleared(
        n_final=np.array([3]),
        n_without=_masks(),
        pairwise_recovering_count=np.array([0]),
    )
    assert code.tolist() == [0]


def test_unique_binder_when_exactly_one_leave_one_out_recovers():
    for name in FUNNEL_MASK_NAMES:
        code = classify_candidates_cleared(
            n_final=np.array([0]),
            n_without=_masks(**{name: np.array([1])}),
            pairwise_recovering_count=np.array([0]),
        )
        assert code.tolist() == [1], name


def test_ambiguous_when_two_leave_one_outs_recover():
    code = classify_candidates_cleared(
        n_final=np.array([0]),
        n_without=_masks(
            static_obstacle=np.array([1]), wall=np.array([2])
        ),
        pairwise_recovering_count=np.array([2]),
    )
    assert code.tolist() == [2]


def test_multi_filter_when_no_single_but_a_pair_recovers():
    code = classify_candidates_cleared(
        n_final=np.array([0]),
        n_without=_masks(),
        pairwise_recovering_count=np.array([1]),
    )
    assert code.tolist() == [3]


def test_no_recovery_with_up_to_two_relaxations_when_neither_single_nor_pair_recovers():
    code = classify_candidates_cleared(
        n_final=np.array([0]),
        n_without=_masks(),
        pairwise_recovering_count=np.array([0]),
    )
    assert code.tolist() == [4]


def test_classification_is_vectorized_across_all_five_buckets_at_once():
    n_final = np.array([5, 0, 0, 0, 0])
    n_without = _masks(
        static_obstacle=np.array([0, 1, 1, 0, 0]),
        wall=np.array([0, 0, 1, 0, 0]),
    )
    pairwise = np.array([0, 0, 2, 1, 0])

    code = classify_candidates_cleared(n_final, n_without, pairwise)

    assert code.tolist() == [0, 1, 2, 3, 4]


SHAPE = (6, 1)


def _write_v3_archive(path, **field_overrides):
    def zeros(dtype):
        return np.zeros(SHAPE, dtype=dtype)

    def ones(dtype):
        return np.ones(SHAPE, dtype=dtype)

    fields = {
        "lateral_m": zeros(np.float32),
        "longitudinal_m": zeros(np.float32),
        "state": np.full(SHAPE, 1, dtype=np.int8),
        "committed_side": np.ones(SHAPE, dtype=np.int8),
        "committed_valid": ones(np.bool_),
        "left_valid": ones(np.bool_),
        "right_valid": ones(np.bool_),
        "used_wait": zeros(np.bool_),
        "emergency_brake": zeros(np.bool_),
        "interaction_active": ones(np.bool_),
        "applied_v_mps": zeros(np.float32),
        "applied_omega_rps": zeros(np.float32),
        "episode_step": np.arange(SHAPE[0], dtype=np.int16)[:, None],
        "pred_err_1step_m": np.zeros((*SHAPE, 2), dtype=np.float32),
        "pred_err_5step_m": np.zeros((*SHAPE, 2), dtype=np.float32),
        "robot_yaw_rad": zeros(np.float32),
        "nearest_dyn_slot": zeros(np.int8),
        "nearest_dyn_bearing_rad": zeros(np.float32),
        "nearest_dyn_distance_m": ones(np.float32),
        "n_raw_left": ones(np.int16),
        "n_raw_right": ones(np.int16),
        "n_final_left": ones(np.int16),
        "n_final_right": ones(np.int16),
        "reachable_linear_mps": ones(np.float32),
        "reachable_progress_m": ones(np.float32),
        "reachable_lateral_m": ones(np.float32),
        "launch_threshold_mps": ones(np.float32) * 0.15,
        "progress_threshold_m": ones(np.float32) * 0.2,
        "side_threshold_m": ones(np.float32) * 0.15,
        "pairwise_recovering_count_left": zeros(np.int16),
        "pairwise_recovering_count_right": zeros(np.int16),
        "pairwise_best_count_left": zeros(np.int16),
        "pairwise_best_count_right": zeros(np.int16),
        "pairwise_best_bitmask_left": zeros(np.int16),
        "pairwise_best_bitmask_right": zeros(np.int16),
    }
    for name in FUNNEL_MASK_NAMES:
        fields[f"n_without_{name}_left"] = zeros(np.int16)
        fields[f"n_without_{name}_right"] = zeros(np.int16)
    fields.update(field_overrides)

    metadata = {"schema": DIAGNOSTIC_SCHEMA, "record_only": True}
    np.savez_compressed(
        path,
        **fields,
        episode_end=np.asarray([[5, 0, 3, 6]], dtype=np.int32),
        metadata_json=np.asarray(json.dumps(metadata)),
    )


def test_v1_schema_is_rejected_not_silently_misread(tmp_path):
    path = tmp_path / "old.npz"
    _write_v3_archive(path)
    with np.load(path) as archive:
        data = {k: archive[k] for k in archive.files}
    data["metadata_json"] = np.asarray(
        json.dumps({"schema": "stateful_teacher_step_diagnostic/v1"})
    )
    np.savez_compressed(path, **data)

    with pytest.raises(ValueError, match="v3"):
        _load_archive(path)


def test_v2_schema_is_rejected_not_silently_misread(tmp_path):
    path = tmp_path / "old.npz"
    _write_v3_archive(path)
    with np.load(path) as archive:
        data = {k: archive[k] for k in archive.files}
    data["metadata_json"] = np.asarray(
        json.dumps({"schema": "stateful_teacher_step_diagnostic/v2"})
    )
    np.savez_compressed(path, **data)

    with pytest.raises(ValueError, match="v3"):
        _load_archive(path)


def test_v2_style_archive_missing_v3_funnel_fields_is_rejected(tmp_path):
    """A v2 archive lacks the fine-grained fields even if mislabeled v3."""
    path = tmp_path / "mislabeled.npz"
    _write_v3_archive(path)
    with np.load(path) as archive:
        data = {k: archive[k] for k in archive.files}
    del data["n_without_heading_left"]

    metadata = {"schema": DIAGNOSTIC_SCHEMA}
    data["metadata_json"] = np.asarray(json.dumps(metadata))
    np.savez_compressed(path, **data)

    with pytest.raises(ValueError, match="missing"):
        _load_archive(path)


def test_analyze_cell_reports_denominators_and_binder_breakdown(tmp_path):
    path = tmp_path / "diag.npz"
    # Frame 1: both-sides-blocked, right cleared by wall alone (unique).
    # Frame 2: left cleared by static+dynamic together (ambiguous: both
    #          single leave-one-outs recover it independently).
    n_final_right = np.ones(SHAPE, dtype=np.int16)
    n_final_right[1, 0] = 0
    n_without_wall_right = np.zeros(SHAPE, dtype=np.int16)
    n_without_wall_right[1, 0] = 1

    n_final_left = np.ones(SHAPE, dtype=np.int16)
    n_final_left[2, 0] = 0
    n_without_static_left = np.zeros(SHAPE, dtype=np.int16)
    n_without_static_left[2, 0] = 1
    n_without_dynamic_left = np.zeros(SHAPE, dtype=np.int16)
    n_without_dynamic_left[2, 0] = 1

    committed_valid = np.ones(SHAPE, dtype=np.bool_)
    committed_valid[1, 0] = False
    left_valid = np.ones(SHAPE, dtype=np.bool_)
    left_valid[1, 0] = False
    right_valid = np.ones(SHAPE, dtype=np.bool_)
    right_valid[1, 0] = False

    _write_v3_archive(
        path,
        n_final_right=n_final_right,
        n_without_wall_right=n_without_wall_right,
        n_final_left=n_final_left,
        n_without_static_obstacle_left=n_without_static_left,
        n_without_dynamic_obstacle_left=n_without_dynamic_left,
        committed_valid=committed_valid,
        left_valid=left_valid,
        right_valid=right_valid,
    )

    report = analyze_cell(path)

    denom = report["denominators"]
    assert denom["all_frames"] == 6
    assert denom["interaction_active_frames"] == 6
    assert denom["committed_invalid_frames"] == 1
    assert denom["both_sides_blocked_frames"] == 1
    assert denom["terminal_frames_by_cause"]["obstacle"] == 1

    right_binders = report["binder_breakdown"]["right"]["all_frames"]
    assert right_binders["unique"]["count"] == 1
    assert right_binders["unique"]["by_mask"]["wall"] == 1

    left_binders = report["binder_breakdown"]["left"]["all_frames"]
    assert left_binders["ambiguous"]["count"] == 1
    assert left_binders["ambiguous"]["involved_mask_counts"][
        "static_obstacle"
    ] == 1
    assert left_binders["ambiguous"]["involved_mask_counts"][
        "dynamic_obstacle"
    ] == 1

    # Denominator-scoped breakdown must not conflate frame share with
    # collision share: fractions are reported against their own labeled
    # denominator, not against total collisions.
    assert (
        right_binders["unique"]["fraction_of_all_frames"]
        == pytest.approx(1 / 6)
    )

    # "grid_limited" overclaims -- it reads as "the grid cannot do this",
    # when it only means neither a single nor a two-gate relaxation was
    # tried and found sufficient (three or more gates may jointly bind).
    assert "no_recovery_with_up_to_two_relaxations" in right_binders
    assert "grid_limited" not in right_binders


def test_analyze_suite_verifies_r5_r6_byte_identical_equivalence(tmp_path):
    r5 = tmp_path / "r5"
    r6 = tmp_path / "r6"
    r5.mkdir()
    r6.mkdir()
    for scenario in ("lateral_lateral", "mixed"):
        _write_v3_archive(r5 / f"{scenario}_stateful_diag.npz")
        _write_v3_archive(r6 / f"{scenario}_stateful_diag.npz")
        for suffix in ("teacher.json", "corridor.json", "cell.json"):
            content = json.dumps({"scenario": scenario, "suffix": suffix})
            (r5 / f"{scenario}_{suffix}").write_text(content)
            (r6 / f"{scenario}_{suffix}").write_text(content)
    fingerprint = json.dumps({"a": "b"})
    (r6 / "source_fingerprint_before.json").write_text(fingerprint)
    (r6 / "source_fingerprint_after.json").write_text(fingerprint)
    manifest = {
        "status": "COMPLETE_VALID_DIAGNOSTIC_EVIDENCE",
        "protocol_sha256": "abc123",
        "checkpoint_sha256": "def456",
        "decision": {
            "teacher_pass": False,
            "training_authorized": False,
            "distillation_authorized": False,
            "sa6_authorized": False,
            "cells": {
                "lateral_lateral": {"episodes": 1},
                "mixed": {"episodes": 1},
            },
        },
    }
    (r6 / "suite_manifest.json").write_text(json.dumps(manifest))

    report = analyze_suite(r6, r5)

    assert report["r5_r6_behavior_equivalence"]["lateral_lateral"][
        "teacher.json"
    ]["byte_identical"] is True
    assert report["protocol_sha256"] == "abc123"


def test_analyze_suite_raises_on_byte_mismatch(tmp_path):
    r5 = tmp_path / "r5"
    r6 = tmp_path / "r6"
    r5.mkdir()
    r6.mkdir()
    for scenario in ("lateral_lateral", "mixed"):
        _write_v3_archive(r5 / f"{scenario}_stateful_diag.npz")
        _write_v3_archive(r6 / f"{scenario}_stateful_diag.npz")
        for suffix in ("teacher.json", "corridor.json", "cell.json"):
            (r5 / f"{scenario}_{suffix}").write_text("same")
            (r6 / f"{scenario}_{suffix}").write_text("same")
    (r6 / f"lateral_lateral_teacher.json").write_text("different")
    fingerprint = json.dumps({"a": "b"})
    (r6 / "source_fingerprint_before.json").write_text(fingerprint)
    (r6 / "source_fingerprint_after.json").write_text(fingerprint)
    manifest = {
        "status": "COMPLETE_VALID_DIAGNOSTIC_EVIDENCE",
        "protocol_sha256": "abc123",
        "checkpoint_sha256": "def456",
        "decision": {
            "teacher_pass": False,
            "training_authorized": False,
            "distillation_authorized": False,
            "sa6_authorized": False,
            "cells": {
                "lateral_lateral": {"episodes": 1},
                "mixed": {"episodes": 1},
            },
        },
    }
    (r6 / "suite_manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="behavior-equivalence"):
        analyze_suite(r6, r5)


def test_render_markdown_produces_a_string_for_a_real_two_scenario_report(
    tmp_path,
):
    """Regression: a stray trailing comma once turned an f-string into a
    1-tuple, which join() only rejects when handed a real multi-scenario
    report -- a single-line smoke check like this would have caught it.
    """
    r5 = tmp_path / "r5"
    r6 = tmp_path / "r6"
    r5.mkdir()
    r6.mkdir()
    for scenario in ("lateral_lateral", "mixed"):
        _write_v3_archive(r5 / f"{scenario}_stateful_diag.npz")
        _write_v3_archive(r6 / f"{scenario}_stateful_diag.npz")
        for suffix in ("teacher.json", "corridor.json", "cell.json"):
            content = json.dumps({"scenario": scenario, "suffix": suffix})
            (r5 / f"{scenario}_{suffix}").write_text(content)
            (r6 / f"{scenario}_{suffix}").write_text(content)
    fingerprint = json.dumps({"a": "b"})
    (r6 / "source_fingerprint_before.json").write_text(fingerprint)
    (r6 / "source_fingerprint_after.json").write_text(fingerprint)
    manifest = {
        "status": "COMPLETE_VALID_DIAGNOSTIC_EVIDENCE",
        "protocol_sha256": "abc123",
        "checkpoint_sha256": "def456",
        "decision": {
            "teacher_pass": False,
            "training_authorized": False,
            "distillation_authorized": False,
            "sa6_authorized": False,
            "cells": {
                "lateral_lateral": {"episodes": 1},
                "mixed": {"episodes": 1},
            },
        },
    }
    (r6 / "suite_manifest.json").write_text(json.dumps(manifest))

    report = analyze_suite(r6, r5)
    rendered = render_markdown(report)

    assert isinstance(rendered, str)
    assert "lateral_lateral" in rendered
    assert "mixed" in rendered


def test_render_markdown_does_not_falsely_claim_byte_identical(tmp_path):
    """A report built outside analyze_suite() can carry a known, explained
    byte diff (analyze_suite() itself always raises on any diff, so this
    exercises the reporting path used when someone has verified a diff is
    benign and wants it stated, not hidden behind a blanket claim).
    """
    path = tmp_path / "diag.npz"
    _write_v3_archive(path)
    report = {
        "status": "COMPLETE_VALID_DIAGNOSTIC_EVIDENCE",
        "protocol_sha256": "abc123",
        "checkpoint_sha256": "def456",
        "teacher_pass": False,
        "training_authorized": False,
        "distillation_authorized": False,
        "sa6_authorized": False,
        "source_fingerprint_stable_within_r6_run": True,
        "r5_r6_behavior_equivalence": {
            "lateral_lateral": {
                "teacher.json": {"byte_identical": False},
                "corridor.json": {"byte_identical": True},
                "cell.json": {"byte_identical": False},
            },
        },
        "r5_r6_equivalence_note": "two added default-valued fields",
        "cells": {"lateral_lateral": analyze_cell(path)},
    }

    rendered = render_markdown(report)

    assert "NOT byte-identical" in rendered
    assert "lateral_lateral_teacher.json" in rendered
    assert "lateral_lateral_cell.json" in rendered
    assert "two added default-valued fields" in rendered
    assert "all teacher/corridor/cell JSON byte-identical" not in rendered
