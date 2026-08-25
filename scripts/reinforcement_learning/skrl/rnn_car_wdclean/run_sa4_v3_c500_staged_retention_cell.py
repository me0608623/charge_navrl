"""Run one frozen staged SA4-v3 c500 retention cell."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_sa4_v3_c550_c600_retention_cell as implementation  # noqa: E402
import sa4_v3_c500_staged_retention as protocol  # noqa: E402


# Reuse the completed retention evaluator without changing its frozen source.
implementation.protocol = protocol
implementation.retention_helpers.protocol = protocol
implementation.corridor_helpers.protocol = protocol


def _output_dir(argv: list[str]) -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("checkpoint")
    parser.add_argument("--output-dir", type=Path, required=True)
    args, _ = parser.parse_known_args(argv)
    return args.output_dir.expanduser().resolve()


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    output = _output_dir(raw_argv)
    strict_validate = protocol.validate_cell

    # The reused implementation validates before writing and contains only a
    # historical schema literal. Adapt that literal while retaining every
    # substantive c500 protocol check.
    def validate_historical_payload(payload: dict) -> None:
        adapted = dict(payload)
        adapted["schema"] = protocol.CELL_SCHEMA
        strict_validate(adapted)

    protocol.validate_cell = validate_historical_payload
    try:
        result = implementation.main(raw_argv)
    finally:
        protocol.validate_cell = strict_validate

    matches = list(output.glob("*_cell.json"))
    if len(matches) != 1:
        raise RuntimeError(
            f"c500 runner produced {len(matches)} cell JSON files"
        )
    path = matches[0]
    payload = json.loads(path.read_text(encoding="utf-8"))
    is_smoke = not bool(payload.get("formal_evidence"))
    payload["schema"] = protocol.SMOKE_SCHEMA if is_smoke else protocol.CELL_SCHEMA
    if not is_smoke:
        strict_validate(payload)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
