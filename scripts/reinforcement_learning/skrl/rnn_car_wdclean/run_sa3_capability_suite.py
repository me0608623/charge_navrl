"""Run one SA3 capability cell: the scene SA3 actually trained on.

Why a new runner rather than a flag on the SA2 one
--------------------------------------------------
``run_sa2_capability_suite.py`` is hard-locked to stage 2 by design --
``SA2_STAGE = 2``, module-level spec guards, ``--stage 2`` in the argument
list, and a runtime marker check for ``stage=2``. Those locks exist so an SA2
result can never be produced from a non-SA2 scene; reusing that module for SA3
would mean loosening exactly the guard that makes its recorded results
trustworthy.

Every scene value here is read from ``STAGE_SCENE_CURRICULUM[3]`` and every
threshold from ``sa3_sa8_acceptance_contract``, so neither the geometry nor the
bars can drift by editing one side.

Cell granularity
----------------
One invocation = one (checkpoint, scenario, delay, seed). ``--seed`` takes a
single value so there is no code path that pools results across seeds or delay
conditions.

Scenario coverage
-----------------
``narrow``, ``lateral`` and ``longitudinal`` are implemented. The remaining
stage-3 scenarios in ``SCENARIOS_BY_STAGE[3]`` (``nav_clean``, ``nav_native``,
``native_crossing``) are Phase B work and are deliberately absent rather than
stubbed, so asking for one fails loudly instead of silently grading the wrong
scene.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys


sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fixed_actuator_eval import (  # noqa: E402
    fixed_actuator_cli_args,
    fixed_actuator_metadata,
    verify_fixed_actuator_runtime,
)

from rnn_car_modular.configs.sim2real_stage_curriculum_v2 import (  # noqa: E402
    STAGE_SCENE_CURRICULUM,
    make_sim2real_curriculum_config,
)

import sa3_sa8_acceptance_contract as contract  # noqa: E402


SA3_STAGE = 3
SPEC = STAGE_SCENE_CURRICULUM[SA3_STAGE]
# Built only to read the two geometry values that live on the config rather
# than on the spec dataclass. The checkpoint string is never loaded.
_SA3_CFG = make_sim2real_curriculum_config(
    SA3_STAGE, checkpoint="__SCENE_CONTRACT_PROBE_NOT_LOADED__.pt"
)

ARENA_SIZE_M = 2.0 * SPEC.room_half_extent
NARROW_WIDTH_RANGE = SPEC.narrow_width_range
NARROW_YAW_LIMIT_DEG = SPEC.narrow_yaw_limit_deg
NARROW_SEGMENT_LENGTH_M = _SA3_CFG.narrow_passage_segment_length
CORRIDOR_FREE_WIDTH_M = SPEC.corridor_free_width       # width, NOT length
CORRIDOR_LENGTH_M = _SA3_CFG.long_corridor_length      # fixed for SA1-SA8
CORRIDOR_STATIC = SPEC.corridor_static_obstacles
CORRIDOR_DYNAMIC = SPEC.corridor_dynamic_obstacles
CORRIDOR_SPEED_RANGE = SPEC.corridor_speed_range

#: SA3 corridor motion is ``env_stratified`` with weights
#: ``(lateral, longitudinal, random_2d)`` and a zero random_2d weight, so these
#: two families are the complete SA3 corridor scenario set, not a sample.
CORRIDOR_FAMILIES = ("lateral", "longitudinal")
SCENARIOS = ("narrow",) + CORRIDOR_FAMILIES

if SPEC.corridor_motion_weights is None:
    raise RuntimeError("SA3 spec lost its corridor motion weights")
if SPEC.corridor_motion_weights[2] != 0.0:
    raise RuntimeError(
        "SA3 corridor now samples random_2d "
        f"(weights={SPEC.corridor_motion_weights}); the lateral+longitudinal "
        "pair is no longer the complete scenario set and this suite would "
        "silently under-test SA3"
    )
if SPEC.corridor_density_mix is not None:
    raise RuntimeError(
        "SA3 spec gained a corridor density mix; fixed static/dynamic counts no "
        "longer describe the training distribution"
    )
if SPEC.narrow_exact_width is not None:
    raise RuntimeError(
        "SA3 spec gained an exact-width narrow stress rung; the plain width "
        "range no longer describes the training distribution"
    )

#: Minimum completed episodes for a cell to be *measurable*. A validity
#: requirement, not a performance bar: below it the rates are noise, so the cell
#: is void rather than failed, and it stays out of ``worst_margin`` -- an
#: episode count and a rate cannot share a units-free margin.
MIN_EPISODES = contract.CORRIDOR_THRESHOLDS.episodes_min

#: Blocking bars, taken from the frozen SA3-SA8 contract rather than restated.
THRESHOLDS: dict[str, dict[str, float]] = {
    "narrow": {
        "sr_min": contract.NARROW_THRESHOLDS.sr_min,
        "cr_max": contract.NARROW_THRESHOLDS.cr_max,
        "to_max": contract.NARROW_THRESHOLDS.to_max,
        "crossing_min": contract.NARROW_THRESHOLDS.crossing_min,
        "direct_crossing_min": contract.NARROW_THRESHOLDS.direct_crossing_min,
    },
    "lateral": {
        "sr_min": contract.CORRIDOR_THRESHOLDS.sr_min,
        "cr_max": contract.CORRIDOR_THRESHOLDS.cr_max,
        "to_max": contract.CORRIDOR_THRESHOLDS.to_max,
    },
    "longitudinal": {
        "sr_min": contract.CORRIDOR_THRESHOLDS.sr_min,
        "cr_max": contract.CORRIDOR_THRESHOLDS.cr_max,
        "to_max": contract.CORRIDOR_THRESHOLDS.to_max,
    },
}

DELAY_MS = {0: 0, 1: 200, 2: 400}
ACTUATOR_PROFILE = contract.ACTUATOR_PROFILE


def thresholds_for(scenario: str) -> dict[str, float]:
    try:
        return dict(THRESHOLDS[scenario])
    except KeyError as exc:
        raise ValueError(f"unknown SA3 scenario {scenario!r}") from exc


def make_cell_id(
    checkpoint: Path, scenario: str, delay_steps: int, seed: int
) -> str:
    """Canonical identity shared by the cell runner and the queue."""
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown SA3 scenario {scenario!r}")
    if delay_steps not in DELAY_MS:
        raise ValueError(f"delay_steps must be one of {tuple(DELAY_MS)}")
    return (
        f"{Path(checkpoint).stem}__{scenario}__"
        f"d{int(delay_steps)}__seed{int(seed)}"
    )


def build_scene_args(
    scenario: str,
    *,
    seed: int,
    actuator_args: list[str],
    corridor_json: Path,
) -> list[str]:
    """CLI arguments that build the SA3 scene for one scenario.

    ``--stage`` and ``--vlp16_noise_mode`` stay explicit: play auto-inherits
    both from the checkpoint, but only ``if "--stage" not in sys.argv``. Pinning
    them is what keeps a comparison across checkpoints from different lineages
    meaningful rather than a property of whatever each one happened to record.
    """
    common = [
        "--stage", str(SA3_STAGE),
        "--vlp16_noise_mode", "full",
        "--arena_size", f"{ARENA_SIZE_M:g}",
        "--obs_near_goal_count", "0",
        "--seed", str(seed),
    ]
    if scenario == "narrow":
        return [
            *common,
            # The narrow barrier is a dedicated pair of assets, explicitly not
            # one of the random wall slots, so zeroing the slots removes
            # confounding interior walls without disabling the barrier.
            "--num_static_obs", "0",
            "--num_dynamic_obs", "0",
            "--num_walls", "0",
            "--narrow_replay_eval",
            "--narrow_replay_width_range",
            f"{NARROW_WIDTH_RANGE[0]:g}", f"{NARROW_WIDTH_RANGE[1]:g}",
            "--narrow_replay_yaw_limit_deg", f"{NARROW_YAW_LIMIT_DEG:g}",
            "--narrow_replay_segment_length", f"{NARROW_SEGMENT_LENGTH_M:g}",
            *actuator_args,
        ]
    if scenario in CORRIDOR_FAMILIES:
        return [
            *common,
            "--long_corridor_eval",
            "--long_corridor_free_width", f"{CORRIDOR_FREE_WIDTH_M:g}",
            "--long_corridor_dynamic_speed_range",
            f"{CORRIDOR_SPEED_RANGE[0]:g}", f"{CORRIDOR_SPEED_RANGE[1]:g}",
            "--long_corridor_static_obstacles", str(CORRIDOR_STATIC),
            "--long_corridor_dynamic_obstacles", str(CORRIDOR_DYNAMIC),
            "--long_corridor_motion_mode", scenario,
            "--long_corridor_output", str(corridor_json),
            "--long_corridor_pause_mode", "default",
            *actuator_args,
        ]
    raise ValueError(f"unknown SA3 scenario {scenario!r}")


def parse_narrow_replay_metrics(log_path: Path) -> dict:
    """Read the ``[NARROW-REPLAY-METRICS]`` banner.

    The replay path emits ``NARROW-REPLAY-METRICS``; ``parse_narrow_gap`` in
    ``validate_gates`` matches ``NARROW-GAP-METRICS`` and would silently return
    None here, so the banner is parsed directly.
    """
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    line = re.search(r"\[NARROW-REPLAY-METRICS\](.*)", text)
    if line is None:
        raise RuntimeError(f"no narrow replay metrics in {log_path}")
    values = {
        key: float(value)
        for key, value in re.findall(
            r"([A-Za-z0-9_.]+)=([+-]?(?:nan|\d+(?:\.\d+)?))", line.group(1)
        )
    }
    for required in ("episodes", "crossing_rate", "direct_crossing_rate"):
        if required not in values:
            raise RuntimeError(
                f"narrow replay metrics missing {required!r} in {log_path}"
            )
    return values


#: Parsed numerically rather than string-matched: ``resolve_replay_layout`` may
#: clamp a request, and a clamped run must fail this check instead of passing
#: because the log happens to contain the substring that was asked for.
_NARROW_LAYOUT_RE = re.compile(
    r"\[PLAY\] 隨機窄縫 replay 評測: "
    r"width∈\(([-\d.]+), ([-\d.]+)\) m.*?"
    r"segment=([\d.]+)m, "
    r"yaw±([\d.]+)deg.*?"
    r"room_half_extent=([\d.]+)m"
)


def verify_narrow_runtime(log_path: Path) -> dict:
    """Fail closed unless play built SA3's narrow replay geometry."""
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    for marker in (f"stage={SA3_STAGE} ", "[SIM2REAL][VLP16-ablation] mode=full"):
        if marker not in text:
            raise RuntimeError(f"{marker!r} absent in {log_path}")

    match = _NARROW_LAYOUT_RE.search(text)
    if match is None:
        raise RuntimeError(
            f"no narrow replay layout line in {log_path}; the scene cannot be "
            "confirmed, so the cell is void"
        )
    width_lo, width_hi, segment, yaw, half_extent = (
        float(value) for value in match.groups()
    )
    expected = {
        "width_lo": (width_lo, NARROW_WIDTH_RANGE[0]),
        "width_hi": (width_hi, NARROW_WIDTH_RANGE[1]),
        "segment_length": (segment, NARROW_SEGMENT_LENGTH_M),
        "yaw_limit_deg": (yaw, NARROW_YAW_LIMIT_DEG),
        "room_half_extent": (half_extent, SPEC.room_half_extent),
    }
    problems = [
        f"{name}: built {built} != SA3 {want}"
        for name, (built, want) in expected.items()
        if abs(built - want) > 1e-6
    ]
    if "[NARROW-REPLAY] ⚠" in text:
        warnings = re.findall(r"\[NARROW-REPLAY\] ⚠ (.*)", text)
        problems.append(f"layout was clamped: {warnings}")
    if problems:
        raise RuntimeError(
            f"SA3 narrow runtime contract failed for {log_path}: {problems}"
        )
    return {
        "scenario": "narrow",
        "arena_size_m": ARENA_SIZE_M,
        "width_range_m_built": [width_lo, width_hi],
        "width_range_m_sa3": list(NARROW_WIDTH_RANGE),
        "yaw_limit_deg_built": yaw,
        "segment_length_m_built": segment,
        "room_half_extent_m_built": half_extent,
        "mechanism": "narrow_replay_eval (same EventTerm as training)",
    }


def verify_corridor_runtime(log_path: Path, report: dict, family: str) -> dict:
    """Fail closed unless play built SA3's corridor for ``family``.

    The checks read measured quantities from the corridor report, not the
    requested config: ``geometry_pass`` compares built wall centres against the
    requested free width, and the motion fraction is counted from the installed
    motion types.
    """
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    if "[SIM2REAL][VLP16-ablation] mode=full" not in text:
        raise RuntimeError(f"VLP16 full-noise marker absent in {log_path}")
    if f"stage={SA3_STAGE} " not in text:
        raise RuntimeError(f"stage={SA3_STAGE} marker absent in {log_path}")

    problems: list[str] = []

    requested_width = report.get("requested_free_width_m")
    if requested_width is None:
        raise RuntimeError(
            f"{log_path} predates the parameterised corridor width; rebuild "
            "against the current play_rnn_car.py"
        )
    if abs(float(requested_width) - CORRIDOR_FREE_WIDTH_M) > 1e-6:
        problems.append(
            f"requested free width {requested_width} != {CORRIDOR_FREE_WIDTH_M}"
        )
    if abs(float(report["requested_length_m"]) - CORRIDOR_LENGTH_M) > 1e-6:
        problems.append(
            f"requested length {report['requested_length_m']} != {CORRIDOR_LENGTH_M}"
        )
    requested_speed = [
        float(v) for v in report["requested_dynamic_speed_range_m_s"]
    ]
    if requested_speed != [float(v) for v in CORRIDOR_SPEED_RANGE]:
        problems.append(
            f"requested speed {requested_speed} != {list(CORRIDOR_SPEED_RANGE)}"
        )

    for flag in (
        "geometry_pass",
        "movement_pass",
        "motion_mode_pass",
        "obstacle_mix_pass",
        "goal_alignment_pass",
        "penetration_pass",
        "speed_upper_bound_pass",
    ):
        if not bool(report.get(flag)):
            problems.append(f"{flag}=false")
    unsolvable = int(report.get("constructive_unsolvable_count", -1))
    if unsolvable != 0:
        problems.append(f"constructive_unsolvable_count={unsolvable}")

    static_mean = float(report["static_obstacles_per_env"])
    dynamic_mean = float(report["dynamic_obstacles_per_env"])
    if abs(static_mean - CORRIDOR_STATIC) > 1e-6:
        problems.append(f"static per env {static_mean} != {CORRIDOR_STATIC}")
    if abs(dynamic_mean - CORRIDOR_DYNAMIC) > 1e-6:
        problems.append(f"dynamic per env {dynamic_mean} != {CORRIDOR_DYNAMIC}")

    fractions = report.get("dynamic_motion_type_fractions")
    if not isinstance(fractions, dict) or family not in fractions:
        problems.append("dynamic_motion_type_fractions missing requested family")
        fractions = {}
    family_fraction = float(fractions.get(family, 0.0))
    if abs(family_fraction - 1.0) > 1e-6:
        problems.append(
            f"{family} motion fraction {family_fraction} != 1.0 (all: {fractions})"
        )

    if problems:
        raise RuntimeError(
            f"SA3 corridor runtime contract failed for {log_path}: {problems}"
        )

    return {
        "scenario": family,
        "arena_size_m": ARENA_SIZE_M,
        "free_width_m_requested": CORRIDOR_FREE_WIDTH_M,
        "free_width_m_measured": float(report["free_width_m_mean"]),
        "length_m_requested": CORRIDOR_LENGTH_M,
        "length_m_measured": float(report["length_m_mean"]),
        "static_obstacles": CORRIDOR_STATIC,
        "dynamic_obstacles": CORRIDOR_DYNAMIC,
        "speed_range_m_s_requested": list(CORRIDOR_SPEED_RANGE),
        "observed_max_dynamic_speed_m_s": float(
            report["observed_max_dynamic_speed_m_s"]
        ),
        "motion_family": family,
        "motion_type_fractions": fractions,
        "mechanism": "long_corridor_eval (same replay installer as training)",
    }


def reconcile_corridor_metrics(summary: dict, report: dict) -> dict:
    """Use precise JSON rates after checking the rounded console ledger."""
    precise = {
        "n": int(report["episodes"]),
        "sr": float(report["success_rate"]),
        "cr": float(report["collision_rate"]),
        "to": float(report["timeout_rate"]),
    }
    if precise["n"] != int(summary["n"]):
        raise RuntimeError(
            "corridor episode ledgers disagree: "
            f"play={summary['n']} corridor={precise['n']}"
        )
    for key in ("sr", "cr", "to"):
        displayed = summary.get(key)
        if displayed is None or not math.isfinite(float(displayed)):
            raise RuntimeError(f"corridor console ledger missing {key}")
        # Console percentages are rounded for display; JSON retains exact counts.
        if abs(float(displayed) - precise[key]) > 0.00101:
            raise RuntimeError(
                f"corridor {key} ledgers disagree: "
                f"play={float(displayed):.6f} corridor={precise[key]:.6f}"
            )
    return precise


def evaluate(scenario: str, metrics: dict) -> dict:
    """Compare one cell's metrics against its scenario thresholds.

    Margins are in rate units: ``value - threshold`` for a lower bound and
    ``threshold - value`` for an upper bound, so a negative margin is exactly a
    failed check and ``min`` over the checks is the cell's worst margin.
    """
    thresholds = thresholds_for(scenario)
    checks: dict[str, dict] = {}

    def lower(name: str, value: float | None, bound: float) -> None:
        if value is None or not math.isfinite(float(value)):
            raise RuntimeError(f"{scenario}: {name} was not measured")
        value = float(value)
        checks[name] = {
            "value": value,
            "bound": bound,
            "direction": ">=",
            "margin": value - bound,
            "pass": value >= bound,
        }

    def upper(name: str, value: float | None, bound: float) -> None:
        if value is None or not math.isfinite(float(value)):
            raise RuntimeError(f"{scenario}: {name} was not measured")
        value = float(value)
        checks[name] = {
            "value": value,
            "bound": bound,
            "direction": "<=",
            "margin": bound - value,
            "pass": value <= bound,
        }

    episodes = metrics.get("n")
    if episodes is None:
        raise RuntimeError(f"{scenario}: episode count was not measured")
    if int(episodes) < MIN_EPISODES:
        raise RuntimeError(
            f"{scenario}: only {int(episodes)} completed episodes "
            f"(< {MIN_EPISODES}); the rates are not measurable, so this cell "
            "is void rather than failed"
        )

    lower("sr", metrics.get("sr"), thresholds["sr_min"])
    upper("cr", metrics.get("cr"), thresholds["cr_max"])
    upper("to", metrics.get("to"), thresholds["to_max"])
    if scenario == "narrow":
        lower("crossing_rate", metrics.get("crossing_rate"),
              thresholds["crossing_min"])
        lower("direct_crossing_rate", metrics.get("direct_crossing_rate"),
              thresholds["direct_crossing_min"])

    worst = min(check["margin"] for check in checks.values())
    return {
        "thresholds": thresholds,
        "episodes_min": MIN_EPISODES,
        "episodes": int(episodes),
        "checks": checks,
        "worst_margin": worst,
        "worst_check": min(checks, key=lambda k: checks[k]["margin"]),
        "threshold_pass": all(check["pass"] for check in checks.values()),
    }


def main(argv: list[str] | None = None) -> int:
    # Keep simulator-heavy imports out of module import so contract tests can
    # run with plain Python and no Isaac/torch process.
    from run_sa5_joint_retention_gates import _run_play
    from validate_gates import parse_play_summary

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    parser.add_argument(
        "--seed", type=int, required=True,
        help="exactly one seed; cells are never pooled across seeds",
    )
    parser.add_argument(
        "--actuator-delay-steps", type=int, choices=(0, 1, 2), required=True,
        help="0/1/2 steps = 0/200/400 ms of decoded-command dead time",
    )
    parser.add_argument(
        "--actuator-profile", choices=(ACTUATOR_PROFILE,),
        default=ACTUATOR_PROFILE,
        help="SA3 cells are pinned to the delay-only training profile",
    )
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=1200)
    args = parser.parse_args(argv)

    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    actuator_args = fixed_actuator_cli_args(
        args.actuator_delay_steps, args.actuator_profile
    )
    stem = f"{args.scenario}_d{args.actuator_delay_steps}_s{args.seed}"
    log_path = output / f"{stem}.log"
    corridor_json = output / f"{stem}_corridor.json"

    print(
        f"[SA3-SUITE] scenario={args.scenario} seed={args.seed} "
        f"delay={args.actuator_delay_steps} "
        f"({DELAY_MS[args.actuator_delay_steps]} ms) arena={ARENA_SIZE_M:g}m",
        flush=True,
    )
    _run_play(
        checkpoint,
        log_path,
        num_envs=args.num_envs,
        steps=args.steps,
        extra=build_scene_args(
            args.scenario,
            seed=args.seed,
            actuator_args=actuator_args,
            corridor_json=corridor_json,
        ),
    )
    verify_fixed_actuator_runtime(
        log_path, args.actuator_delay_steps, args.actuator_profile
    )

    summary = parse_play_summary(str(log_path))
    if not summary or summary.get("sr") is None:
        raise RuntimeError(f"no play outcome summary in {log_path}")
    metrics = dict(summary)

    if args.scenario == "narrow":
        narrow = parse_narrow_replay_metrics(log_path)
        if int(narrow["episodes"]) != int(metrics["n"]):
            raise RuntimeError(
                "narrow episode ledgers disagree: "
                f"play={metrics['n']} replay={int(narrow['episodes'])}"
            )
        metrics.update(narrow)
        scene_contract = verify_narrow_runtime(log_path)
        corridor_report = None
    else:
        corridor_report = json.loads(corridor_json.read_text(encoding="utf-8"))
        scene_contract = verify_corridor_runtime(
            log_path, corridor_report, args.scenario
        )
        metrics = reconcile_corridor_metrics(metrics, corridor_report)

    verdict = evaluate(args.scenario, metrics)
    cell = {
        "schema": "sa3_capability/v1",
        "cell_id": make_cell_id(
            checkpoint, args.scenario, args.actuator_delay_steps, args.seed
        ),
        "checkpoint": str(checkpoint),
        "stage": SA3_STAGE,
        "scenario": args.scenario,
        "seed": args.seed,
        "delay_steps": args.actuator_delay_steps,
        "delay_ms": DELAY_MS[args.actuator_delay_steps],
        "num_envs": args.num_envs,
        "steps": args.steps,
        "scene_contract": scene_contract,
        "actuator_eval": fixed_actuator_metadata(
            args.actuator_delay_steps, args.actuator_profile
        ),
        "metrics": metrics,
        "corridor_report": corridor_report,
        "log": str(log_path),
        **verdict,
    }
    cell_path = output / f"{stem}_cell.json"
    cell_path.write_text(
        json.dumps(cell, indent=2, sort_keys=True), encoding="utf-8"
    )

    detail = " ".join(
        f"{name}={check['value']:.4f}{check['direction']}{check['bound']:g}"
        f"({check['margin']:+.4f})"
        for name, check in verdict["checks"].items()
    )
    print(
        f"[SA3-SUITE] {stem}={'PASS' if verdict['threshold_pass'] else 'FAIL'} "
        f"n={metrics['n']} {detail} "
        f"worst={verdict['worst_check']}({verdict['worst_margin']:+.4f}) "
        f"report={cell_path}",
        flush=True,
    )
    return 0 if verdict["threshold_pass"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
