"""Run one frozen SA5-v3 stage-2 paired fixed-screen cell."""

from __future__ import annotations

import run_sa5_v3_c50_random2d_weighted_screen_cell as base_runner
import sa5_v3_c50_stage2_paired_screen as protocol


def main(argv: list[str] | None = None) -> int:
    saved = base_runner.protocol
    base_runner.protocol = protocol
    try:
        return base_runner.main(argv)
    finally:
        base_runner.protocol = saved


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

