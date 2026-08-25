"""Run the frozen stateful delay-aware 2S2D teacher diagnostic."""

from __future__ import annotations

import run_teacher_2s2d_d1_screen as shared_runner
import teacher_2s2d_stateful_d1_screen as stateful_protocol


def main(argv: list[str] | None = None) -> int:
    shared_runner.protocol = stateful_protocol
    return shared_runner.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
