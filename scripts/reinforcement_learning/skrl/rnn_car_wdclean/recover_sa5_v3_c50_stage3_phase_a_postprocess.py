"""Recover the completed Phase-A result after a presentation-only KeyError."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path

import sa5_v3_c50_stage3_phase_a_screen as protocol


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
ROOT = protocol.SCREEN_ROOT


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    before_path = ROOT / "SOURCE_FINGERPRINT_BEFORE.json"
    after_path = ROOT / "SOURCE_FINGERPRINT_AFTER.json"
    summary_path = ROOT / "SUMMARY.json"
    incomplete_path = ROOT / "INCOMPLETE.json"
    targets = (before_path, after_path, summary_path, incomplete_path)
    missing = [str(path) for path in targets if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Phase-A recovery inputs missing: {missing}")

    before = load_json(before_path)
    after = load_json(after_path)
    incomplete = load_json(incomplete_path)
    stored_summary = load_json(summary_path)
    if before != after:
        raise RuntimeError("Phase-A source fingerprints differ before/after")
    if (
        incomplete.get("status") != "INCOMPLETE_FAIL_CLOSED"
        or incomplete.get("error") != "KeyError: 'selected_extension_checkpoint'"
        or len(incomplete.get("completed_cells", [])) != 8
        or int(incomplete.get("expected_cells", -1)) != 8
    ):
        raise RuntimeError("Phase-A failure was not the known postprocess KeyError")

    for relative, expected in before.get("files", {}).items():
        path = REPO / relative
        if not path.is_file() or sha256_of(path) != expected:
            raise RuntimeError(f"recorded Phase-A source changed: {relative}")

    payloads = []
    for candidate in protocol.CANDIDATE_SPECS:
        for scenario in protocol.ALL_SCENARIOS:
            matches = list((ROOT / candidate["name"] / scenario).glob("*_cell.json"))
            if len(matches) != 1:
                raise RuntimeError(
                    f"expected one Phase-A cell for {candidate['name']}/{scenario}"
                )
            payload = load_json(matches[0])
            protocol.validate_cell(payload)
            payloads.append(payload)

    recomputed = protocol.final_verdict(payloads)
    mismatches = {
        key: {"stored": stored_summary.get(key), "recomputed": value}
        for key, value in recomputed.items()
        if stored_summary.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"stored Phase-A summary mismatch: {mismatches}")
    if (
        stored_summary.get("source_fingerprint_stable") is not True
        or stored_summary.get("source_fingerprint_sha256") != before["sha256"]
        or stored_summary.get("protocol_sha256") != protocol.screen_protocol()["sha256"]
    ):
        raise RuntimeError("stored Phase-A metadata mismatch")

    recovery = {
        "schema": "sa5_v3_c50_stage3_phase_a_postprocess_recovery/v1",
        "status": "COMPLETE_VALID_POSTPROCESS_RECOVERED",
        "recovered_at": datetime.now().astimezone().isoformat(),
        "reason": (
            "all eight cells, fingerprints, and SUMMARY were complete before "
            "the generic queue attempted to print an absent legacy field"
        ),
        "original_incomplete_retained": str(incomplete_path.relative_to(REPO)),
        "original_error": incomplete["error"],
        "validated_cells": len(payloads),
        "expected_cells": 8,
        "protocol_sha256": protocol.screen_protocol()["sha256"],
        "source_fingerprint_sha256": before["sha256"],
        "source_fingerprint_before_after_identical": True,
        "recorded_sources_still_match": True,
        "recomputed_summary_matches_stored": True,
        "summary": str(summary_path.relative_to(REPO)),
        "summary_sha256": sha256_of(summary_path),
        "phase_a_pass": bool(stored_summary["phase_a_pass"]),
        "next_action": stored_summary["next_action"],
        "selected_parent": None,
        "extension_authorized": False,
        "graduation_authorized": False,
        "sa6_started": False,
        "interpretation_limit": stored_summary["interpretation_limit"],
    }
    output = ROOT / "POSTPROCESS_RECOVERY.json"
    output.write_text(
        json.dumps(recovery, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(recovery, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
