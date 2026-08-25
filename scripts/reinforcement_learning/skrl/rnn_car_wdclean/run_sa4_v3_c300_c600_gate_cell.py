"""Run one c300-c600 Gate cell using the frozen SA4-v3 evaluator."""

from __future__ import annotations

import run_sa4_v3_checkpoint_screen_cell as implementation
import sa4_v3_c300_c600_gate as protocol


implementation.protocol = protocol


def main(argv: list[str] | None = None) -> int:
    return implementation.main(argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

