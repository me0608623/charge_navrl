"""Run one frozen c500 SA5-v3 parent-control screen cell."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_sa3_v3_retention_screen_cell as retention_helpers  # noqa: E402
import run_sa5_r2_checkpoint_screen_cell as implementation  # noqa: E402
import sa5_v3_c500_parent_control_screen as protocol  # noqa: E402


implementation.protocol = protocol
implementation.MOTION_MODE = dict(implementation.MOTION_MODE)
for _scenario in protocol.LOW_DENSITY_SCENARIOS:
    implementation.MOTION_MODE[_scenario] = "lateral"

_base_common_args = implementation._common_args
_base_build_scene_args = implementation.build_scene_args
_base_verify_corridor_runtime = implementation.verify_corridor_runtime


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
    if scenario in protocol.LOW_DENSITY_SCENARIOS:
        return _base_build_scene_args(
            scenario,
            corridor_json,
            dynamic_speed_range=protocol.P060,
            corridor_motion_mode="lateral",
        )
    return _base_build_scene_args(
        scenario,
        corridor_json,
        dynamic_speed_range=(
            protocol.P035 if scenario == protocol.TARGET_SCENARIO else None
        ),
        corridor_motion_mode=(
            "mixed" if scenario == protocol.TARGET_SCENARIO else None
        ),
    )


def verify_corridor_runtime(
    report: dict,
    scenario: str,
    values: dict,
    **_: object,
) -> dict:
    expected = dict(values)
    if scenario in protocol.LOW_DENSITY_SCENARIOS:
        expected["corridor_speed_range_m_s"] = list(protocol.P060)
        mode = "lateral"
    else:
        expected["corridor_speed_range_m_s"] = list(protocol.P035)
        mode = "mixed"
    return _base_verify_corridor_runtime(
        report,
        scenario,
        expected,
        corridor_motion_mode=mode,
    )


implementation._common_args = _common_args
implementation.build_scene_args = build_scene_args
implementation.verify_corridor_runtime = verify_corridor_runtime


def _output_dir(argv: list[str]) -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("checkpoint")
    parser.add_argument("--output-dir", type=Path, required=True)
    args, _ = parser.parse_known_args(argv)
    return args.output_dir.expanduser().resolve()


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    output = _output_dir(raw_argv)
    result = implementation.main(raw_argv)

    matches = list(output.glob("*_cell.json"))
    if len(matches) != 1:
        raise RuntimeError(
            f"c500 control runner produced {len(matches)} cell JSON files"
        )
    path = matches[0]
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema"] = protocol.CELL_SCHEMA
    payload["speed_rate"] = protocol.SPEED_RATE
    payload["speed_rate_obs"] = protocol.SPEED_RATE_OBS
    payload["deployment_speed_scale"] = protocol.DEPLOYMENT_SPEED_SCALE
    payload["speed_rate_runtime"] = retention_helpers.verify_speed_rate_log(
        Path(payload["log"])
    )
    protocol.validate_cell(payload)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
