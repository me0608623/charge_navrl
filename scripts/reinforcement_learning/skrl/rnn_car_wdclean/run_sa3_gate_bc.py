"""Run one SA3 Phase B / Phase C graduation-gate cell.

Independent of ``run_sa3_capability_suite.py`` and ``sa3_phase_a_queue.py``:
neither is imported or modified, so Phase A's recorded evidence keeps whatever
guarantees it had.

Where the scene comes from
--------------------------
Nothing here restates a geometry constant.

* ``nav_native`` / ``native_crossing`` pass **only** ``--stage`` and let play
  build the stage's own scene. ``play_rnn_car.py`` runs with
  ``STAGE_PARAMETER = True``, so its scene defaults come from the trained stage
  config and only flags explicitly present in ``argv`` override them. Passing
  obstacle counts would therefore *replace* the trained scene with a restated
  one -- the exact failure this module is built to avoid.
* ``nav_clean`` deliberately overrides the obstacle/wall counts to zero. That is
  a defined empty-room probe, not a restatement of stage geometry.
* ``narrow_range`` and the corridor families read every value from
  ``STAGE_SCENE_CURRICULUM[geometry_stage]``.
* Every threshold comes from ``sa3_sa8_acceptance_contract``.

Phase C cross-geometry
----------------------
``--geometry-stage`` is separate from the checkpoint. A cell is identified by
both, so a retention comparison can never silently confuse "which policy" with
"which stage's scene".
"""

from __future__ import annotations

import argparse
import hashlib
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


SUPPORTED_GEOMETRY_STAGES = (2, 3)
NATIVE_SCENARIOS = ("nav_clean", "nav_native", "native_crossing")
CORRIDOR_SCENARIOS = ("corridor_lateral", "corridor_longitudinal")
SCENARIOS = NATIVE_SCENARIOS + ("narrow_range",) + CORRIDOR_SCENARIOS

DELAY_MS = {0: 0, 1: 200, 2: 400}
ACTUATOR_PROFILE = contract.ACTUATOR_PROFILE
MIN_EPISODES = contract.CORRIDOR_THRESHOLDS.episodes_min


def spec_for(geometry_stage: int):
    if int(geometry_stage) not in SUPPORTED_GEOMETRY_STAGES:
        raise ValueError(
            f"geometry_stage must be one of {SUPPORTED_GEOMETRY_STAGES}, "
            f"got {geometry_stage}"
        )
    return STAGE_SCENE_CURRICULUM[int(geometry_stage)]


def scene_values(geometry_stage: int) -> dict:
    """Every geometry value this runner may pass, read from the training source."""
    spec = spec_for(geometry_stage)
    cfg = make_sim2real_curriculum_config(
        int(geometry_stage), checkpoint="__SCENE_CONTRACT_PROBE_NOT_LOADED__.pt"
    )
    if spec.corridor_motion_weights is None:
        raise RuntimeError(f"stage {geometry_stage} lost its corridor motion weights")
    if spec.corridor_motion_weights[2] != 0.0:
        raise RuntimeError(
            f"stage {geometry_stage} corridor now samples random_2d "
            f"(weights={spec.corridor_motion_weights}); lateral+longitudinal is "
            "no longer the complete family set"
        )
    if spec.corridor_density_mix is not None:
        raise RuntimeError(
            f"stage {geometry_stage} gained a corridor density mix; fixed counts "
            "no longer describe the training distribution"
        )
    if spec.narrow_exact_width is not None:
        raise RuntimeError(
            f"stage {geometry_stage} gained an exact-width narrow stress rung"
        )
    return {
        "stage": int(geometry_stage),
        "arena_size_m": 2.0 * spec.room_half_extent,
        "room_half_extent_m": spec.room_half_extent,
        "narrow_width_range_m": tuple(spec.narrow_width_range),
        "narrow_yaw_limit_deg": spec.narrow_yaw_limit_deg,
        "narrow_segment_length_m": cfg.narrow_passage_segment_length,
        "corridor_free_width_m": spec.corridor_free_width,
        "corridor_length_m": cfg.long_corridor_length,
        "corridor_static": spec.corridor_static_obstacles,
        "corridor_dynamic": spec.corridor_dynamic_obstacles,
        "corridor_speed_range_m_s": tuple(spec.corridor_speed_range),
    }


def thresholds_for(scenario: str, geometry_stage: int) -> dict[str, float]:
    """Blocking bars, taken from the frozen contract."""
    if scenario == "nav_clean":
        bars = contract.NAV_CLEAN_THRESHOLDS
    elif scenario in ("nav_native", "native_crossing"):
        try:
            bars = contract.NATIVE_THRESHOLDS[int(geometry_stage)]
        except KeyError as exc:
            raise ValueError(
                f"no native thresholds for stage {geometry_stage}; native "
                "scenarios are only defined for the stages in the contract"
            ) from exc
    elif scenario == "narrow_range":
        bars = contract.NARROW_THRESHOLDS
    elif scenario in CORRIDOR_SCENARIOS:
        bars = contract.CORRIDOR_THRESHOLDS
    else:
        raise ValueError(f"unknown scenario {scenario!r}")
    out = {"sr_min": bars.sr_min, "cr_max": bars.cr_max, "to_max": bars.to_max}
    if bars.crossing_min is not None:
        out["crossing_min"] = bars.crossing_min
    if bars.direct_crossing_min is not None:
        out["direct_crossing_min"] = bars.direct_crossing_min
    return out


def sha256_of(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_cell_id(
    checkpoint, scenario: str, geometry_stage: int, delay_steps: int, seed: int
) -> str:
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}")
    if delay_steps not in DELAY_MS:
        raise ValueError(f"delay_steps must be one of {tuple(DELAY_MS)}")
    if int(geometry_stage) not in SUPPORTED_GEOMETRY_STAGES:
        raise ValueError(f"unsupported geometry stage {geometry_stage}")
    lineage = Path(checkpoint).parent.name.split("_")[0]
    return (
        f"{lineage}__{Path(checkpoint).stem}__{scenario}__"
        f"g{int(geometry_stage)}__d{int(delay_steps)}__seed{int(seed)}"
    )


def build_scene_args(
    scenario: str,
    *,
    geometry_stage: int,
    seed: int,
    actuator_args: list[str],
    corridor_json: Path,

) -> list[str]:
    values = scene_values(geometry_stage)
    common = [
        "--stage", str(values["stage"]),
        "--vlp16_noise_mode", "full",
        "--arena_size", f"{values['arena_size_m']:g}",
        "--obs_near_goal_count", "0",
        "--seed", str(seed),
    ]

    if scenario == "nav_clean":
        # Defined empty-room probe: the only place obstacle counts are set.
        return [
            *common,
            "--num_static_obs", "0",
            "--num_dynamic_obs", "0",
            "--num_walls", "0",
            *actuator_args,
        ]
    if scenario == "nav_native":
        # No scene overrides at all -- play's STAGE_PARAMETER=True path builds
        # the trained stage scene. Adding counts here would replace it.
        return [*common, *actuator_args]
    if scenario == "native_crossing":
        # The stage's own scene with every dynamic obstacle set to the crossing
        # behaviour the SA3 curriculum injects (``corridor_crossing_fraction``).
        #
        # Deliberately NOT play's ``--near_wall_crossing_eval`` probe: that path
        # sets every ``*collision*`` termination term to None so contact stays
        # observable (near_wall_crossing_eval.py:49-53). With collisions unable
        # to end an episode, ``CR`` is pinned at 0 by construction and ``TO``
        # absorbs the runs that should have been collisions -- the contract's
        # SR/CR/TO bars then measure something they were never written for.
        return [
            *common,
            "--obstacle_behavior", "corridor_crossing",
            *actuator_args,
        ]
    if scenario == "narrow_range":
        return [
            *common,
            "--num_static_obs", "0",
            "--num_dynamic_obs", "0",
            "--num_walls", "0",
            "--narrow_replay_eval",
            "--narrow_replay_width_range",
            f"{values['narrow_width_range_m'][0]:g}",
            f"{values['narrow_width_range_m'][1]:g}",
            "--narrow_replay_yaw_limit_deg", f"{values['narrow_yaw_limit_deg']:g}",
            "--narrow_replay_segment_length",
            f"{values['narrow_segment_length_m']:g}",
            *actuator_args,
        ]
    if scenario in CORRIDOR_SCENARIOS:
        family = scenario.split("_", 1)[1]
        return [
            *common,
            "--long_corridor_eval",
            "--long_corridor_free_width", f"{values['corridor_free_width_m']:g}",
            "--long_corridor_dynamic_speed_range",
            f"{values['corridor_speed_range_m_s'][0]:g}",
            f"{values['corridor_speed_range_m_s'][1]:g}",
            "--long_corridor_static_obstacles", str(values["corridor_static"]),
            "--long_corridor_dynamic_obstacles", str(values["corridor_dynamic"]),
            "--long_corridor_motion_mode", family,
            "--long_corridor_output", str(corridor_json),
            "--long_corridor_pause_mode", "default",
            *actuator_args,
        ]
    raise ValueError(f"unknown scenario {scenario!r}")


_NARROW_LAYOUT_RE = re.compile(
    r"\[PLAY\] 隨機窄縫 replay 評測: "
    r"width∈\(([-\d.]+), ([-\d.]+)\) m.*?"
    r"segment=([\d.]+)m, "
    r"yaw±([\d.]+)deg.*?"
    r"room_half_extent=([\d.]+)m"
)
#: play prints the built room as ``外牆 ±8.5m`` (the training log's
#: ``room_size=... → scene WxH`` line is a *trainer* banner and never appears
#: here). Parsed numerically so a clamped or defaulted arena fails the cell.
_ROOM_RE = re.compile(r"外牆 ±([\d.]+)m")
#: The stage banner play emits once the curriculum is resolved. It names the
#: stage and reports the obstacle load actually installed, which is the only
#: runtime confirmation that ``nav_native`` got the trained scene rather than
#: play's own module defaults.
_STAGE_BANNER_RE = re.compile(
    r"\[PLAY\] curriculum=(\S+) stage=(\d+) name=(\S+) .*?障礙=(\d+)S\+(\d+)D"
)


#: play's outcome block prints both an exact count and a rounded percentage.
#: ``parse_play_summary`` keeps only the percentage, which is fatal on the
#: boundary: a cell reading ``5.0%`` against a ``0.05`` bar could be anywhere in
#: [0.0495, 0.0505], i.e. either side of the threshold. These read the counts.
_EXACT_TOTAL_RE = re.compile(r"總回合數[:：]\s*(\d+)")
_EXACT_SUCCESS_RE = re.compile(r"成功率\s*\(到達目標\)[:：]\s*(\d+)\s*\(")
_EXACT_COLLISION_RE = re.compile(r"碰撞率\s*\(總計\)[:：]\s*(\d+)\s*\(")
_EXACT_TIMEOUT_RE = re.compile(r"超時率[:：]\s*(\d+)\s*\(")


def parse_exact_outcome_counts(log_path: Path) -> dict:
    """Exact SR/CR/TO from the printed counts, not the rounded percentages."""
    text = Path(log_path).read_text(encoding="utf-8", errors="ignore")
    found = {}
    for key, pattern in (
        ("n", _EXACT_TOTAL_RE), ("success", _EXACT_SUCCESS_RE),
        ("collision", _EXACT_COLLISION_RE), ("timeout", _EXACT_TIMEOUT_RE),
    ):
        match = pattern.search(text)
        if match is None:
            raise RuntimeError(
                f"no exact {key} count in {log_path}; the cell cannot be graded "
                "from rounded percentages"
            )
        found[key] = int(match.group(1))
    total = found["n"]
    if total <= 0:
        raise RuntimeError(f"episode total is {total} in {log_path}")
    return {
        "n": total,
        "sr": found["success"] / total,
        "cr": found["collision"] / total,
        "to": found["timeout"] / total,
        "counts": {k: found[k] for k in ("success", "collision", "timeout")},
    }


def reconcile_console_metrics(summary: dict, exact: dict) -> dict:
    """Use exact counts after checking they agree with the rounded ledger."""
    if int(summary["n"]) != exact["n"]:
        raise RuntimeError(
            f"episode ledgers disagree: play={summary['n']} exact={exact['n']}"
        )
    for key in ("sr", "cr", "to"):
        displayed = summary.get(key)
        if displayed is None or not math.isfinite(float(displayed)):
            raise RuntimeError(f"console ledger missing {key}")
        # The console rounds to one tenth of a percentage point = 0.001.
        if abs(float(displayed) - exact[key]) > 0.00101:
            raise RuntimeError(
                f"{key} ledgers disagree: console={float(displayed):.6f} "
                f"exact={exact[key]:.6f}"
            )
    return {k: exact[k] for k in ("n", "sr", "cr", "to")} | {
        "outcome_counts": exact["counts"]
    }


def parse_narrow_replay_metrics(log_path: Path) -> dict:
    text = Path(log_path).read_text(encoding="utf-8", errors="ignore")
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
            raise RuntimeError(f"narrow metrics missing {required!r} in {log_path}")
    return values


def verify_common_runtime(log_path: Path, values: dict) -> dict:
    """Markers every cell must show, whatever the scenario.

    Returns the observed stage banner so each cell records which stage scene was
    actually built, not merely which one was requested.
    """
    text = Path(log_path).read_text(encoding="utf-8", errors="ignore")
    if "[SIM2REAL][VLP16-ablation] mode=full" not in text:
        raise RuntimeError(f"VLP16 full-noise marker absent in {log_path}")

    problems = []
    room = _ROOM_RE.search(text)
    if room is None:
        raise RuntimeError(f"no room geometry line in {log_path}")
    half = float(room.group(1))
    if abs(half - values["room_half_extent_m"]) > 1e-6:
        problems.append(f"room half extent {half} != {values['room_half_extent_m']}")

    banner = _STAGE_BANNER_RE.search(text)
    if banner is None:
        raise RuntimeError(
            f"no stage banner in {log_path}; the built stage cannot be confirmed"
        )
    curriculum, built_stage, stage_name, n_static, n_dynamic = banner.groups()
    if int(built_stage) != int(values["stage"]):
        problems.append(f"built stage {built_stage} != requested {values['stage']}")
    if problems:
        raise RuntimeError(
            f"runtime geometry contract failed for {log_path}: {problems}"
        )
    return {
        "curriculum": curriculum,
        "stage_built": int(built_stage),
        "stage_name": stage_name,
        "native_static_obstacles": int(n_static),
        "native_dynamic_obstacles": int(n_dynamic),
        "room_half_extent_m_built": half,
    }


def verify_narrow_runtime(log_path: Path, values: dict) -> dict:
    text = Path(log_path).read_text(encoding="utf-8", errors="ignore")
    match = _NARROW_LAYOUT_RE.search(text)
    if match is None:
        raise RuntimeError(
            f"no narrow replay layout line in {log_path}; scene unconfirmed, cell void"
        )
    width_lo, width_hi, segment, yaw, half_extent = (
        float(value) for value in match.groups()
    )
    expected = {
        "width_lo": (width_lo, values["narrow_width_range_m"][0]),
        "width_hi": (width_hi, values["narrow_width_range_m"][1]),
        "segment_length": (segment, values["narrow_segment_length_m"]),
        "yaw_limit_deg": (yaw, values["narrow_yaw_limit_deg"]),
        "room_half_extent": (half_extent, values["room_half_extent_m"]),
    }
    problems = [
        f"{name}: built {built} != stage {want}"
        for name, (built, want) in expected.items()
        if abs(built - want) > 1e-6
    ]
    if "[NARROW-REPLAY] ⚠" in text:
        warnings = re.findall(r"\[NARROW-REPLAY\] ⚠ (.*)", text)
        problems.append(f"layout was clamped: {warnings}")
    if problems:
        raise RuntimeError(f"narrow runtime contract failed for {log_path}: {problems}")
    return {
        "scenario": "narrow_range",
        "width_range_m_built": [width_lo, width_hi],
        "yaw_limit_deg_built": yaw,
        "segment_length_m_built": segment,
        "room_half_extent_m_built": half_extent,
    }


def verify_corridor_runtime(log_path: Path, report: dict, family: str, values: dict) -> dict:
    problems: list[str] = []
    requested_width = report.get("requested_free_width_m")
    if requested_width is None:
        raise RuntimeError(
            f"{log_path} predates the parameterised corridor width; rebuild play"
        )
    if abs(float(requested_width) - values["corridor_free_width_m"]) > 1e-6:
        problems.append(
            f"requested width {requested_width} != {values['corridor_free_width_m']}"
        )
    if abs(float(report["requested_length_m"]) - values["corridor_length_m"]) > 1e-6:
        problems.append(
            f"requested length {report['requested_length_m']} != {values['corridor_length_m']}"
        )
    if [float(v) for v in report["requested_dynamic_speed_range_m_s"]] != [
        float(v) for v in values["corridor_speed_range_m_s"]
    ]:
        problems.append("requested speed range mismatch")
    for flag in (
        "geometry_pass", "movement_pass", "motion_mode_pass", "obstacle_mix_pass",
        "goal_alignment_pass", "penetration_pass", "speed_upper_bound_pass",
    ):
        if not bool(report.get(flag)):
            problems.append(f"{flag}=false")
    if int(report.get("constructive_unsolvable_count", -1)) != 0:
        problems.append("constructive_unsolvable_count != 0")
    if abs(float(report["static_obstacles_per_env"]) - values["corridor_static"]) > 1e-6:
        problems.append("static per env mismatch")
    if abs(float(report["dynamic_obstacles_per_env"]) - values["corridor_dynamic"]) > 1e-6:
        problems.append("dynamic per env mismatch")
    fractions = report.get("dynamic_motion_type_fractions") or {}
    if abs(float(fractions.get(family, 0.0)) - 1.0) > 1e-6:
        problems.append(f"{family} motion fraction != 1.0 (all: {fractions})")
    if problems:
        raise RuntimeError(f"corridor runtime contract failed for {log_path}: {problems}")
    return {
        "scenario": f"corridor_{family}",
        "free_width_m_measured": float(report["free_width_m_mean"]),
        "length_m_measured": float(report["length_m_mean"]),
        "observed_max_dynamic_speed_m_s": float(report["observed_max_dynamic_speed_m_s"]),
        "motion_type_fractions": fractions,
    }


def reconcile_corridor_metrics(summary: dict, report: dict) -> dict:
    precise = {
        "n": int(report["episodes"]),
        "sr": float(report["success_rate"]),
        "cr": float(report["collision_rate"]),
        "to": float(report["timeout_rate"]),
    }
    if precise["n"] != int(summary["n"]):
        raise RuntimeError(
            f"corridor ledgers disagree: play={summary['n']} corridor={precise['n']}"
        )
    for key in ("sr", "cr", "to"):
        displayed = summary.get(key)
        if displayed is None or not math.isfinite(float(displayed)):
            raise RuntimeError(f"corridor console ledger missing {key}")
        if abs(float(displayed) - precise[key]) > 0.00101:
            raise RuntimeError(f"corridor {key} ledgers disagree")
    return precise


def evaluate(scenario: str, geometry_stage: int, metrics: dict) -> dict:
    thresholds = thresholds_for(scenario, geometry_stage)
    checks: dict[str, dict] = {}

    def bound(name, value, limit, direction):
        if value is None or not math.isfinite(float(value)):
            raise RuntimeError(f"{scenario}: {name} was not measured")
        value = float(value)
        margin = value - limit if direction == ">=" else limit - value
        checks[name] = {
            "value": value, "bound": limit, "direction": direction,
            "margin": margin, "pass": margin >= 0.0,
        }

    episodes = metrics.get("n")
    if episodes is None:
        raise RuntimeError(f"{scenario}: episode count was not measured")
    if int(episodes) < MIN_EPISODES:
        raise RuntimeError(
            f"{scenario}: only {int(episodes)} completed episodes (< {MIN_EPISODES}); "
            "the rates are not measurable, so this cell is void rather than failed"
        )

    bound("sr", metrics.get("sr"), thresholds["sr_min"], ">=")
    bound("cr", metrics.get("cr"), thresholds["cr_max"], "<=")
    bound("to", metrics.get("to"), thresholds["to_max"], "<=")
    if "crossing_min" in thresholds:
        bound("crossing_rate", metrics.get("crossing_rate"),
              thresholds["crossing_min"], ">=")
    if "direct_crossing_min" in thresholds:
        bound("direct_crossing_rate", metrics.get("direct_crossing_rate"),
              thresholds["direct_crossing_min"], ">=")

    return {
        "thresholds": thresholds,
        "episodes_min": MIN_EPISODES,
        "episodes": int(episodes),
        "checks": checks,
        "worst_margin": min(c["margin"] for c in checks.values()),
        "worst_check": min(checks, key=lambda k: checks[k]["margin"]),
        "threshold_pass": all(c["pass"] for c in checks.values()),
    }


def main(argv: list[str] | None = None) -> int:
    from run_sa5_joint_retention_gates import _run_play
    from validate_gates import parse_play_summary

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    parser.add_argument("--geometry-stage", type=int,
                        choices=SUPPORTED_GEOMETRY_STAGES, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--actuator-delay-steps", type=int, choices=(0, 1, 2),
                        required=True)
    parser.add_argument("--actuator-profile", choices=(ACTUATOR_PROFILE,),
                        default=ACTUATOR_PROFILE)
    parser.add_argument("--expect-checkpoint-sha256", default=None,
                        help="fail closed if the checkpoint is not this exact file")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=1200)
    args = parser.parse_args(argv)

    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    digest = sha256_of(checkpoint)
    if args.expect_checkpoint_sha256 and digest != args.expect_checkpoint_sha256:
        raise RuntimeError(
            f"checkpoint hash mismatch for {checkpoint}: "
            f"expected {args.expect_checkpoint_sha256}, got {digest}"
        )

    values = scene_values(args.geometry_stage)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    actuator_args = fixed_actuator_cli_args(
        args.actuator_delay_steps, args.actuator_profile
    )
    stem = (f"{args.scenario}_g{args.geometry_stage}"
            f"_d{args.actuator_delay_steps}_s{args.seed}")
    log_path = output / f"{stem}.log"
    corridor_json = output / f"{stem}_corridor.json"

    print(f"[SA3-BC] scenario={args.scenario} geometry_stage={args.geometry_stage} "
          f"seed={args.seed} delay={args.actuator_delay_steps} "
          f"({DELAY_MS[args.actuator_delay_steps]} ms) "
          f"arena={values['arena_size_m']:g}m ckpt={digest[:12]}", flush=True)

    _run_play(
        checkpoint, log_path, num_envs=args.num_envs, steps=args.steps,
        extra=build_scene_args(
            args.scenario, geometry_stage=args.geometry_stage, seed=args.seed,
            actuator_args=actuator_args, corridor_json=corridor_json,

        ),
    )
    verify_fixed_actuator_runtime(
        log_path, args.actuator_delay_steps, args.actuator_profile
    )
    stage_banner = verify_common_runtime(log_path, values)

    summary = parse_play_summary(str(log_path))
    if not summary or summary.get("sr") is None:
        raise RuntimeError(f"no play outcome summary in {log_path}")
    metrics = dict(summary)
    corridor_report = None

    if args.scenario == "narrow_range":
        narrow = parse_narrow_replay_metrics(log_path)
        if int(narrow["episodes"]) != int(metrics["n"]):
            raise RuntimeError("narrow episode ledgers disagree")
        exact = parse_exact_outcome_counts(log_path)
        metrics = reconcile_console_metrics(metrics, exact)
        metrics.update({k: v for k, v in narrow.items() if k != "episodes"})
        scene_contract = verify_narrow_runtime(log_path, values)
    elif args.scenario in CORRIDOR_SCENARIOS:
        corridor_report = json.loads(corridor_json.read_text(encoding="utf-8"))
        family = args.scenario.split("_", 1)[1]
        scene_contract = verify_corridor_runtime(
            log_path, corridor_report, family, values
        )
        metrics = reconcile_corridor_metrics(metrics, corridor_report)
    else:
        metrics = reconcile_console_metrics(
            metrics, parse_exact_outcome_counts(log_path)
        )
        scene_contract = {"scenario": args.scenario,
                          "mechanism": "stage scene via STAGE_PARAMETER"}

    verdict = evaluate(args.scenario, args.geometry_stage, metrics)
    cell = {
        "schema": "sa3_gate_bc/v1",
        "cell_id": make_cell_id(checkpoint, args.scenario, args.geometry_stage,
                                args.actuator_delay_steps, args.seed),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest,
        "geometry_stage": args.geometry_stage,
        "scene_values": {k: list(v) if isinstance(v, tuple) else v
                         for k, v in values.items()},
        "scenario": args.scenario,
        "seed": args.seed,
        "delay_steps": args.actuator_delay_steps,
        "delay_ms": DELAY_MS[args.actuator_delay_steps],
        "num_envs": args.num_envs,
        "steps": args.steps,
        "scene_contract": scene_contract,
        "stage_banner": stage_banner,
        "actuator_eval": fixed_actuator_metadata(
            args.actuator_delay_steps, args.actuator_profile
        ),
        "metrics": metrics,
        "corridor_report": corridor_report,
        "log": str(log_path),
        **verdict,
    }
    cell_path = output / f"{stem}_cell.json"
    cell_path.write_text(json.dumps(cell, indent=2, sort_keys=True), encoding="utf-8")

    detail = " ".join(
        f"{n}={c['value']:.4f}{c['direction']}{c['bound']:g}({c['margin']:+.4f})"
        for n, c in verdict["checks"].items()
    )
    print(f"[SA3-BC] {stem}={'PASS' if verdict['threshold_pass'] else 'FAIL'} "
          f"n={metrics['n']} {detail} "
          f"worst={verdict['worst_check']}({verdict['worst_margin']:+.4f}) "
          f"report={cell_path}", flush=True)
    return 0 if verdict["threshold_pass"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
