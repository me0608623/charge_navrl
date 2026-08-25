"""Run one frozen SA5-v3 stage-2 paired-retention cell."""

from __future__ import annotations

import sys

from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_sa5_v3_provisional_pilot_screen_cell as implementation  # noqa: E402
import sa5_v3_c50_stage2_retention_screen as protocol  # noqa: E402


implementation.protocol = protocol
implementation.implementation.protocol = protocol


def main(argv: list[str] | None = None) -> int:
    return implementation.main(argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
