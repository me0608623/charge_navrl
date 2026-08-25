"""Run one frozen 8-cell direction-decomposition cell (c50 anchor vs pilot it50)."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_sa3_v3_retention_screen_cell as retention_helpers  # noqa: E402
import run_sa5_r2_checkpoint_screen_cell as implementation  # noqa: E402
import sa5_v3_c50_it50_direction_ab as protocol  # noqa: E402


_MOTION_MODE = {row["name"]: row["motion_mode"] for row in protocol.SCENARIO_SPECS}
_DENSITY = {
    row["name"]: (row["static_obstacles"], row["dynamic_obstacles"])
    for row in protocol.SCENARIO_SPECS
}

_base_common_args = implementation._common_args
_base_build_scene_args = implementation.build_scene_args
_base_verify_corridor_runtime = implementation.verify_corridor_runtime


def stage_values() -> dict:
    values = dict(protocol._stage_scene())
    values["corridor_speed_range_m_s"] = list(protocol.DYNAMIC_SPEED_RANGE)
    return values


def _common_args(values: dict) -> list[str]:
    return [
        *_base_common_args(values),
        "--speed_rate",
        f"{protocol.SPEED_RATE:g}",
        "--speed_rate_obs",
        protocol.SPEED_RATE_OBS,
        "--deployment_speed_scale",
        f"{protocol.DEPLOYMENT_SPEED_SCALE:g}",
    ]


def build_scene_args(scenario: str, corridor_json: Path, **_: object) -> list[str]:
    spec = protocol.scenario_by_name(scenario)
    with _implementation_context():
        return _base_build_scene_args(
            scenario,
            corridor_json,
            dynamic_speed_range=protocol.DYNAMIC_SPEED_RANGE,
            corridor_motion_mode=spec["motion_mode"],
        )


def verify_corridor_runtime(
    report: dict,
    scenario: str,
    values: dict,
    **_: object,
) -> dict:
    expected = dict(values)
    expected["corridor_speed_range_m_s"] = list(protocol.DYNAMIC_SPEED_RANGE)
    spec = protocol.scenario_by_name(scenario)
    with _implementation_context():
        return _base_verify_corridor_runtime(
            report,
            scenario,
            expected,
            corridor_motion_mode=spec["motion_mode"],
        )


@contextmanager
def _implementation_context(*, install_callbacks: bool = False):
    names = ["protocol", "MOTION_MODE", "DENSITY", "stage_values", "_common_args"]
    if install_callbacks:
        names.extend(["build_scene_args", "verify_corridor_runtime"])
    saved = {name: getattr(implementation, name) for name in names}
    implementation.protocol = protocol
    implementation.MOTION_MODE = dict(_MOTION_MODE)
    implementation.DENSITY = dict(_DENSITY)
    implementation.stage_values = stage_values
    implementation._common_args = _common_args
    if install_callbacks:
        implementation.build_scene_args = build_scene_args
        implementation.verify_corridor_runtime = verify_corridor_runtime
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(implementation, name, value)


def _output_dir(argv: list[str]) -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("checkpoint")
    parser.add_argument("--output-dir", type=Path, required=True)
    args, _ = parser.parse_known_args(argv)
    return args.output_dir.expanduser().resolve()


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    output = _output_dir(raw_argv)
    with _implementation_context(install_callbacks=True):
        result = implementation.main(raw_argv)

    matches = list(output.glob("*_cell.json"))
    if len(matches) != 1:
        raise RuntimeError(
            f"direction-AB runner produced {len(matches)} cell JSON files"
        )
    path = matches[0]
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidate = protocol.candidate_by_name(str(payload["checkpoint_name"]))
    payload["schema"] = protocol.CELL_SCHEMA
    payload["candidate_status"] = candidate["status"]
    payload["diagnostic_spec"] = protocol.scenario_by_name(payload["scenario"])
    payload["speed_rate"] = protocol.SPEED_RATE
    payload["speed_rate_obs"] = protocol.SPEED_RATE_OBS
    payload["deployment_speed_scale"] = protocol.DEPLOYMENT_SPEED_SCALE
    payload["speed_rate_runtime"] = retention_helpers.verify_speed_rate_log(
        Path(payload["log"])
    )
    protocol.validate_cell(payload)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return result


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
