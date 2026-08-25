import json

import numpy as np
import pytest

from rnn_car_wdclean.analyze_stateful_teacher_step_diagnostic import (
    _failure_masks,
    _load_archive,
    _validate_archive,
    analyze_cell,
)


def _write_archive(path, *, bad_terminal_alignment=False):
    shape = (8, 1)
    state = np.full(shape, 2, dtype=np.int8)
    side = np.ones(shape, dtype=np.int8)
    left_valid = np.ones(shape, dtype=np.bool_)
    right_valid = np.ones(shape, dtype=np.bool_)
    left_valid[1, 0] = False
    left_valid[2, 0] = False
    right_valid[2, 0] = False
    left_valid[6, 0] = False
    right_valid[6, 0] = False
    committed_valid = left_valid.copy()
    used_wait = ~committed_valid
    episode_step = np.asarray([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int16)[:, None]
    ends = np.asarray(
        [[3, 0, 1, 5 if bad_terminal_alignment else 4], [7, 0, 3, 4]],
        dtype=np.int32,
    )
    metadata = {
        "schema": "stateful_teacher_step_diagnostic/v1",
        "record_only": True,
    }
    np.savez_compressed(
        path,
        applied_omega_rps=np.zeros(shape, dtype=np.float32),
        applied_v_mps=np.zeros(shape, dtype=np.float32),
        committed_side=side,
        committed_valid=committed_valid,
        emergency_brake=np.zeros(shape, dtype=np.bool_),
        episode_end=ends,
        episode_step=episode_step,
        interaction_active=np.ones(shape, dtype=np.bool_),
        lateral_m=np.zeros(shape, dtype=np.float32),
        left_valid=left_valid,
        longitudinal_m=np.zeros(shape, dtype=np.float32),
        metadata_json=np.asarray(json.dumps(metadata)),
        pred_err_1step_m=np.zeros((8, 1, 2), dtype=np.float32),
        pred_err_5step_m=np.zeros((8, 1, 2), dtype=np.float32),
        right_valid=right_valid,
        state=state,
        used_wait=used_wait,
    )


def test_archive_masks_separate_no_switch_from_both_blocked(tmp_path):
    path = tmp_path / "diag.npz"
    _write_archive(path)
    arrays, _ = _load_archive(path)

    masks = _failure_masks(arrays)

    assert masks["committed_invalid"].sum() == 3
    assert masks["no_switch_other_open"].sum() == 1
    assert masks["both_sides_blocked"].sum() == 2


def test_episode_end_must_join_to_terminal_action_row(tmp_path):
    path = tmp_path / "bad.npz"
    _write_archive(path, bad_terminal_alignment=True)
    arrays, _ = _load_archive(path)

    with pytest.raises(ValueError, match="terminal action row"):
        _validate_archive(arrays)


def test_cell_analysis_reports_episode_and_frame_reconciliation(tmp_path):
    path = tmp_path / "diag.npz"
    _write_archive(path)

    report = analyze_cell(path)

    assert report["validation"]["completed_episodes"] == 2
    assert report["validation"]["terminal_alignment_ok"] is True
    assert report["frame_accounting"]["reconciliation_ok"] is True
    assert report["episode_outcomes"]["goal"]["episodes"] == 1
    assert report["episode_outcomes"]["obstacle"]["episodes"] == 1
