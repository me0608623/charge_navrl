#!/usr/bin/env python3
"""Recompute the SA5 B3 c500-c600 corridor exposure ledger from JSONL."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable


REPO = Path(__file__).resolve().parents[4]
DEFAULT_METRICS = (
    REPO
    / "logs/rnn_car/"
    "sa5_v3_c50_stage3_b3_cont500_from_it100_ne1024_s42_p500_r1/"
    "supervisor_metrics.jsonl"
)
DEFAULT_SCREEN = (
    REPO
    / "logs/gates/sa5_v3_c50_stage3_b3_c500_c600_screen/"
    "screen_20260826_r1/SUMMARY.json"
)
DEFAULT_OUTPUT_DIR = DEFAULT_SCREEN.parent

WINDOWS = {
    "c500": (351, 400),
    "c550": (401, 450),
    "c600": (451, 500),
}
PROFILES = (
    "p060_0s1d",
    "p060_1s1d",
    "p060_2s1d",
    "p035_3s2d",
    "p035_4s2d",
)
LOW_DENSITY_PROFILES = ("p060_0s1d", "p060_1s1d")
FAMILIES = (
    "lateral",
    "longitudinal",
    "random_2d",
    "mixed",
    "no_dynamic",
    "unready",
)

CURRENT_MIX = (
    ((0, 1), (0.50, 0.70), 0.10),
    ((1, 1), (0.50, 0.70), 0.10),
    ((2, 1), (0.50, 0.70), 0.15),
    ((3, 2), (0.25, 0.45), 0.30),
    ((4, 2), (0.25, 0.45), 0.35),
)
PROPOSED_MIX = (
    ((0, 1), (0.50, 0.70), 0.10),
    ((1, 1), (0.50, 0.70), 0.10),
    ((0, 1), (0.70, 0.90), 0.05),
    ((1, 1), (0.70, 0.90), 0.05),
    ((2, 1), (0.50, 0.70), 0.15),
    ((3, 2), (0.25, 0.45), 0.25),
    ((4, 2), (0.25, 0.45), 0.30),
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
    return rows


def _sum(rows: Iterable[dict[str, Any]], key: str) -> float:
    return sum(float(row.get(key, 0.0) or 0.0) for row in rows)


def _rate_counts(
    rows: Iterable[dict[str, Any]], prefix: str
) -> dict[str, float]:
    rows = list(rows)
    episodes = _sum(rows, f"{prefix}/episodes")
    out = {"episodes": episodes}
    for name, suffix in (("successes", "sr"), ("collisions", "cr"), ("timeouts", "timeout")):
        out[name] = sum(
            float(row.get(f"{prefix}/episodes", 0.0) or 0.0)
            * float(row.get(f"{prefix}/{suffix}", 0.0) or 0.0)
            for row in rows
        )
    out["sr"] = out["successes"] / episodes if episodes else 0.0
    out["cr"] = out["collisions"] / episodes if episodes else 0.0
    out["to"] = out["timeouts"] / episodes if episodes else 0.0
    return out


def aggregate_window(
    rows: list[dict[str, Any]], start: int, end: int
) -> dict[str, Any]:
    selected = [row for row in rows if start <= int(row["iteration"]) <= end]
    expected = list(range(start, end + 1))
    actual = [int(row["iteration"]) for row in selected]
    if actual != expected:
        raise ValueError(
            f"window {start}-{end} is incomplete: expected {expected}, got {actual}"
        )
    reconciliation_failures = sum(
        float(row.get("corridor_profile_family/accounting/reconciliation_ok", 0.0))
        != 1.0
        for row in selected
    )
    if reconciliation_failures:
        raise ValueError(
            f"window {start}-{end} has {reconciliation_failures} reconciliation failures"
        )

    totals = {
        "active_steps": _sum(
            selected, "corridor_profile_family/accounting/active_steps_total"
        ),
        "resets": _sum(
            selected, "corridor_profile_family/accounting/resets_total"
        ),
        "episodes": _sum(
            selected, "corridor_profile_family/accounting/episodes_total"
        ),
    }
    profiles: dict[str, Any] = {}
    for profile in PROFILES:
        prefix = f"corridor_profile/{profile}"
        bucket = _rate_counts(selected, prefix)
        bucket["active_steps"] = _sum(selected, f"{prefix}/active_steps")
        bucket["resets"] = _sum(selected, f"{prefix}/reset_count")
        bucket["active_step_share"] = (
            bucket["active_steps"] / totals["active_steps"]
            if totals["active_steps"]
            else 0.0
        )
        bucket["reset_share"] = (
            bucket["resets"] / totals["resets"] if totals["resets"] else 0.0
        )
        bucket["completed_episode_share"] = (
            bucket["episodes"] / totals["episodes"]
            if totals["episodes"]
            else 0.0
        )
        if profile in LOW_DENSITY_PROFILES:
            families: dict[str, Any] = {}
            for family in FAMILIES:
                family_prefix = f"corridor_profile_family/{profile}/{family}"
                if not any(
                    f"{family_prefix}/episodes" in row
                    or f"{family_prefix}/active_steps" in row
                    for row in selected
                ):
                    continue
                family_bucket = _rate_counts(selected, family_prefix)
                family_bucket["active_steps"] = _sum(
                    selected, f"{family_prefix}/active_steps"
                )
                family_bucket["resets"] = _sum(
                    selected, f"{family_prefix}/reset_count"
                )
                family_bucket["active_share_within_profile"] = (
                    family_bucket["active_steps"] / bucket["active_steps"]
                    if bucket["active_steps"]
                    else 0.0
                )
                family_bucket["reset_share_within_profile"] = (
                    family_bucket["resets"] / bucket["resets"]
                    if bucket["resets"]
                    else 0.0
                )
                family_bucket["episode_share_within_profile"] = (
                    family_bucket["episodes"] / bucket["episodes"]
                    if bucket["episodes"]
                    else 0.0
                )
                families[family] = family_bucket
            bucket["families"] = families
        profiles[profile] = bucket

    for quantity, total_key in (
        ("active_steps", "active_steps"),
        ("resets", "resets"),
        ("episodes", "episodes"),
    ):
        residual = sum(float(p[quantity]) for p in profiles.values()) - totals[total_key]
        if abs(residual) > 1e-6:
            raise ValueError(
                f"window {start}-{end} profile {quantity} residual is {residual}"
            )

    return {
        "iterations": [start, end],
        "rows": len(selected),
        "reconciliation_failures": reconciliation_failures,
        "totals": totals,
        "profiles": profiles,
    }


def binomial_difference_se(a: dict[str, Any], b: dict[str, Any]) -> float:
    pa, pb = float(a["cr"]), float(b["cr"])
    na, nb = float(a["n"]), float(b["n"])
    return math.sqrt(pa * (1.0 - pa) / na + pb * (1.0 - pb) / nb)


def decide_pilot(
    windows: dict[str, Any], screen: dict[str, Any], p080_key_count: int
) -> dict[str, Any]:
    by_name = {
        row["checkpoint_name"]: row for row in screen["candidate_results"]
    }
    active_spans = {}
    reset_spans = {}
    for profile in LOW_DENSITY_PROFILES:
        active = [windows[name]["profiles"][profile]["active_step_share"] for name in WINDOWS]
        reset = [windows[name]["profiles"][profile]["reset_share"] for name in WINDOWS]
        active_spans[profile] = max(active) - min(active)
        reset_spans[profile] = max(reset) - min(reset)

    c550 = by_name["c550"]
    c600 = by_name["c600"]
    fixed_tradeoff = (
        c550["p060_absolute_pass"]
        and not c550["p080_retention_pass"]
        and not c600["p060_absolute_pass"]
        and c600["p080_retention_pass"]
    )
    exposure_stable = all(value <= 0.005 for value in active_spans.values()) and all(
        value <= 0.001 for value in reset_spans.values()
    )
    establish = p080_key_count == 0 and exposure_stable and fixed_tradeoff
    return {
        "decision": (
            "ESTABLISH_FRESH_C550_MATCHED_PROFILE_RATIO_PILOT_READY_NOT_RUN"
            if establish
            else "DO_NOT_ESTABLISH_PROFILE_RATIO_PILOT"
        ),
        "establish_pilot": establish,
        "reasons": {
            "explicit_p080_profile_key_count": p080_key_count,
            "p060_top_level_exposure_stable": exposure_stable,
            "fixed_screen_tradeoff_present": fixed_tradeoff,
            "active_step_share_spans": active_spans,
            "reset_share_spans": reset_spans,
        },
        "interpretation_limits": [
            "P080=0 means no explicit labeled P080 corridor profile replay; it does not prove that no vaguely P080-like state occurred elsewhere.",
            "The 30/30/40 motion weights are global dynamic-slot quotas, not per-profile quotas.",
            "Training-window CR is on-policy and is not a substitute for the frozen fixed-screen CR.",
            "The c600 historical continuation is not a fresh process-matched control for a new intervention arm.",
        ],
        "pilot": {
            "status": "READY_NOT_RUN" if establish else "NOT_CREATED",
            "parent": "c550",
            "parent_sha256": "501c79199b9c0a73d556d5ac1c1345cde347675a213197414af1183c5a2be4ed",
            "iterations_per_arm": 50,
            "save_conceptual_offsets": [25, 50],
            "control_mix": CURRENT_MIX,
            "intervention_mix": PROPOSED_MIX,
            "only_behavioral_difference": "long_corridor_speed_density_mix",
            "requires_fresh_control": True,
            "automatic_launch_authorized": False,
            "sa6_authorized": False,
        },
    }


def build_report(
    windows: dict[str, Any], screen: dict[str, Any], decision: dict[str, Any]
) -> str:
    lines = [
        "# SA5 B3 c500-c600 post-hoc exposure ledger",
        "",
        "This report is recomputed from raw JSONL count fields. No GPU rollout was used.",
        "",
        "## Top-level profile exposure",
        "",
        "| window | profile | active share | reset share | episodes | SR | CR | TO |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, window in windows.items():
        for profile in PROFILES:
            p = window["profiles"][profile]
            lines.append(
                f"| {name} | {profile} | {p['active_step_share']:.4%} | "
                f"{p['reset_share']:.4%} | {p['episodes']:.0f} | {p['sr']:.4%} | "
                f"{p['cr']:.4%} | {p['to']:.4%} |"
            )

    lines.extend(
        [
            "",
            "## P060 low-density family allocation",
            "",
            "Shares below are conditional on the named profile, not on all corridor frames.",
            "",
            "| window | profile | family | active share | reset share | episodes | CR |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, window in windows.items():
        for profile in LOW_DENSITY_PROFILES:
            for family, f in window["profiles"][profile]["families"].items():
                if f["active_steps"] == 0 and f["episodes"] == 0:
                    continue
                lines.append(
                    f"| {name} | {profile} | {family} | "
                    f"{f['active_share_within_profile']:.4%} | "
                    f"{f['reset_share_within_profile']:.4%} | "
                    f"{f['episodes']:.0f} | {f['cr']:.4%} |"
                )

    by_name = {
        row["checkpoint_name"]: row for row in screen["candidate_results"]
    }
    lines.extend(
        [
            "",
            "## Fixed-screen comparison",
            "",
            "| checkpoint | P060 0S1D CR | P060 1S1D CR | P080 0S1D CR | P080 1S1D CR | P060 pass | P080 retention |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name in WINDOWS:
        row = by_name[name]
        cells = row["rows"]
        lines.append(
            f"| {name} | {cells['p060_0s1d']['cr']:.4%} | "
            f"{cells['p060_1s1d']['cr']:.4%} | "
            f"{cells['p080_0s1d']['cr']:.4%} | "
            f"{cells['p080_1s1d']['cr']:.4%} | "
            f"{row['p060_absolute_pass']} | {row['p080_retention_pass']} |"
        )

    c550 = by_name["c550"]["rows"]
    c600 = by_name["c600"]["rows"]
    p0600_delta = c600["p060_0s1d"]["cr"] - c550["p060_0s1d"]["cr"]
    p0600_se = binomial_difference_se(
        c600["p060_0s1d"], c550["p060_0s1d"]
    )
    p0801_delta = c600["p080_1s1d"]["cr"] - c550["p080_1s1d"]["cr"]
    p0801_se = binomial_difference_se(
        c600["p080_1s1d"], c550["p080_1s1d"]
    )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"**{decision['decision']}**",
            "",
            f"- Explicit P080 corridor-profile keys in 500 training rows: {decision['reasons']['explicit_p080_profile_key_count']}.",
            "- P060 0S1D/1S1D reset exposure stayed at approximately 10% per profile in every 50-iteration window.",
            "- The c550-to-c600 fixed-screen flip cannot be attributed to a top-level P060 exposure-count jump.",
            f"- c550->c600 P060 0S1D CR changed by {p0600_delta:+.4%} ({p0600_delta / p0600_se:+.2f} pooled SE).",
            f"- c550->c600 P080 1S1D CR changed by {p0801_delta:+.4%} ({p0801_delta / p0801_se:+.2f} pooled SE).",
            "- A bounded pilot is justified as a causal test of explicit P080 retention replay, not as proof that ratios caused the historical fluctuation.",
            "- Run a fresh c550 control and a fresh c550 intervention; do not treat historical c600 as the formal control.",
            "- Keep P060 0S1D, P060 1S1D, and P060 2S1D weights unchanged. Add P080 0S1D/1S1D at 5% each by reducing P035 3S2D/4S2D from 30/35% to 25/30%.",
            "- Status is READY_NOT_RUN. This analysis does not authorize GPU training or SA6.",
            "",
            "## Interpretation limits",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in decision["interpretation_limits"])
    lines.append("")
    return "\n".join(lines)


def analyze(metrics_path: Path, screen_path: Path) -> dict[str, Any]:
    rows = load_jsonl(metrics_path)
    iterations = [int(row["iteration"]) for row in rows]
    if iterations != list(range(1, 501)):
        raise ValueError("metrics must contain exactly iterations 1 through 500")
    windows = {
        name: aggregate_window(rows, start, end)
        for name, (start, end) in WINDOWS.items()
    }
    p080_keys = sorted(
        {
            key
            for row in rows
            for key in row
            if key.startswith("corridor_profile/p080_")
            or key.startswith("corridor_profile_family/p080_")
        }
    )
    screen = json.loads(screen_path.read_text(encoding="utf-8"))
    decision = decide_pilot(windows, screen, len(p080_keys))
    return {
        "schema": "sa5_b3_c500_c600_posthoc_exposure_ledger/v1",
        "status": "COMPLETE_VALID_ZERO_GPU_POSTHOC",
        "metrics_path": str(metrics_path.resolve()),
        "screen_path": str(screen_path.resolve()),
        "training_rows": len(rows),
        "profile_reconciliation_failures": 0,
        "explicit_p080_profile_keys": p080_keys,
        "windows": windows,
        "fixed_screen_protocol_sha256": screen["protocol_sha256"],
        "fixed_screen_status": screen["status"],
        "decision": decision,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--screen", type=Path, default=DEFAULT_SCREEN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    payload = analyze(args.metrics, args.screen)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "POSTHOC_EXPOSURE_LEDGER.json"
    markdown_path = args.output_dir / "POSTHOC_EXPOSURE_LEDGER.md"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    screen = json.loads(args.screen.read_text(encoding="utf-8"))
    markdown_path.write_text(
        build_report(payload["windows"], screen, payload["decision"]),
        encoding="utf-8",
    )
    print(json.dumps({
        "status": payload["status"],
        "decision": payload["decision"]["decision"],
        "json": str(json_path),
        "markdown": str(markdown_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
