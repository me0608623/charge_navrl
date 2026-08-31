"""Run one frozen c500/c550/c600 P060/P080 comparison cell."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_sa3_v3_retention_screen_cell as retention_helpers  # noqa: E402
import run_sa5_v3_provisional_pilot_screen_cell as implementation  # noqa: E402
import sa5_v3_c50_stage3_b3_c500_c600_screen as protocol  # noqa: E402


implementation.protocol = protocol
implementation.implementation.protocol = protocol
implementation.implementation.MOTION_MODE = dict(
    implementation.implementation.MOTION_MODE
)
implementation.implementation.DENSITY = dict(
    implementation.implementation.DENSITY
)
for _scenario in protocol.ALL_SCENARIOS:
    implementation.implementation.MOTION_MODE[_scenario] = "lateral"
    implementation.implementation.DENSITY[_scenario] = protocol.density_for(
        _scenario
    )

_base_build_scene_args = implementation._base_build_scene_args
_base_verify_corridor_runtime = implementation._base_verify_corridor_runtime


def build_scene_args(scenario: str, corridor_json: Path, **_: object) -> list[str]:
    return _base_build_scene_args(
        scenario,
        corridor_json,
        dynamic_speed_range=protocol.speed_range_for(scenario),
        corridor_motion_mode="lateral",
    )


def verify_corridor_runtime(
    report: dict, scenario: str, values: dict, **_: object
) -> dict:
    expected = dict(values)
    expected["corridor_speed_range_m_s"] = list(
        protocol.speed_range_for(scenario)
    )
    return _base_verify_corridor_runtime(
        report,
        scenario,
        expected,
        corridor_motion_mode="lateral",
    )


implementation.build_scene_args = build_scene_args
implementation.verify_corridor_runtime = verify_corridor_runtime
implementation.implementation.build_scene_args = build_scene_args
implementation.implementation.verify_corridor_runtime = verify_corridor_runtime


def _output_dir(argv: list[str]) -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("checkpoint")
    parser.add_argument("--output-dir", type=Path, required=True)
    args, _ = parser.parse_known_args(argv)
    return args.output_dir.expanduser().resolve()


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    output = _output_dir(raw_argv)
    result = implementation.implementation.main(raw_argv)
    matches = list(output.glob("*_cell.json"))
    if len(matches) != 1:
        raise RuntimeError(
            f"c500-c600 runner produced {len(matches)} cell JSON files"
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
