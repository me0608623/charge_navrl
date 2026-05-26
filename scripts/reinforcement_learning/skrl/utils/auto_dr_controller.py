"""DORAEMON-style auto DR controller.

Automatically expands domain randomization parameter ranges based on policy
success rate, replacing manual SA1→SA8 DR scheduling.

Reference: Tiboni et al., "DORAEMON: Domain Randomization via Entropy
Maximization", ICLR 2024 (arXiv 2311.01885).

Simplified version: instead of full entropy maximization, we use a greedy
expansion strategy — when SR >= threshold, each DR param range expands by
a fixed fraction toward its maximum bounds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class DRParamSpec:
    """Specification for a single DR parameter's expansion range."""
    name: str
    initial: tuple[float, float]     # starting range (narrow)
    bounds: tuple[float, float]      # maximum range (widest)
    current: list[float] = field(default_factory=list)
    is_bool: bool = False            # toggle params (e.g. enable_actuator_dr)

    def __post_init__(self):
        if not self.current:
            self.current = list(self.initial)


# Default DR parameter specs matching SA1→SA8 progression
DEFAULT_DR_SPECS: list[DRParamSpec] = [
    # LiDAR noise (L1)
    DRParamSpec("lidar_displacement_std",  (0.0, 0.0),    (0.001, 0.005)),
    DRParamSpec("lidar_hole_rate",         (0.0, 0.0),    (0.15, 0.30)),
    DRParamSpec("lidar_block_dropout_prob",(0.0, 0.0),    (0.0, 0.15)),
    # Physics DR
    DRParamSpec("physics_mass_dr",         (0.95, 1.05),  (0.80, 1.20)),
    DRParamSpec("physics_friction_dr",     (0.85, 1.15),  (0.60, 1.40)),
    DRParamSpec("physics_com_offset",      (0.0, 0.02),   (0.0, 0.08)),
    # Disturbance DR
    DRParamSpec("disturbance_wind_force",  (0.0, 1.0),    (0.0, 6.0)),
    DRParamSpec("disturbance_push_force",  (5.0, 10.0),   (5.0, 30.0)),
    DRParamSpec("disturbance_push_ratio",  (0.05, 0.05),  (0.05, 0.12)),
]


class AutoDRController:
    """Greedy DORAEMON-style DR expansion controller.

    Every `check_interval` iterations, evaluates recent SR:
      - SR >= threshold → expand all DR param ranges by `expansion_rate`
      - SR < threshold  → hold current ranges (let policy catch up)

    Usage:
        controller = AutoDRController(threshold=0.85, check_interval=50)
        # In training loop:
        if controller.maybe_expand(iteration, recent_sr):
            controller.apply_to_env(env)  # patch running env's DR params
    """

    def __init__(
        self,
        threshold: float = 0.85,
        check_interval: int = 50,
        expansion_rate: float = 0.1,
        specs: list[DRParamSpec] | None = None,
    ):
        self.threshold = threshold
        self.check_interval = check_interval
        self.expansion_rate = expansion_rate
        self.specs = [DRParamSpec(
            s.name, s.initial, s.bounds, list(s.initial), s.is_bool
        ) for s in (specs or DEFAULT_DR_SPECS)]
        self.expansion_count = 0
        self.history: list[dict[str, Any]] = []
        self._fully_expanded = False

    @property
    def fully_expanded(self) -> bool:
        return self._fully_expanded

    def maybe_expand(self, iteration: int, recent_sr: float) -> bool:
        """Check if DR ranges should expand. Returns True if expanded."""
        if self._fully_expanded:
            return False
        if iteration % self.check_interval != 0 or iteration == 0:
            return False

        self.history.append({
            "iteration": iteration,
            "sr": recent_sr,
            "expanded": recent_sr >= self.threshold,
        })

        if recent_sr < self.threshold:
            return False

        any_expanded = False
        for spec in self.specs:
            if spec.is_bool:
                continue
            lo_cur, hi_cur = spec.current
            lo_bound, hi_bound = spec.bounds

            lo_new = lo_cur + self.expansion_rate * (lo_bound - lo_cur)
            hi_new = hi_cur + self.expansion_rate * (hi_bound - hi_cur)

            if abs(lo_new - lo_cur) > 1e-8 or abs(hi_new - hi_cur) > 1e-8:
                spec.current = [lo_new, hi_new]
                any_expanded = True

        if any_expanded:
            self.expansion_count += 1

        self._fully_expanded = all(
            abs(s.current[0] - s.bounds[0]) < 1e-6
            and abs(s.current[1] - s.bounds[1]) < 1e-6
            for s in self.specs if not s.is_bool
        )

        return any_expanded

    def apply_to_env(self, env_unwrapped) -> None:
        """Patch running env's DR EventTerm params with current ranges."""
        em = getattr(env_unwrapped, 'event_manager', None)
        if em is None:
            return

        dr_term = None
        for term_name in dir(em.cfg):
            if term_name.startswith("_"):
                continue
            term = getattr(em.cfg, term_name, None)
            if term is not None and hasattr(term, 'params'):
                if 'mass_scale' in getattr(term, 'params', {}):
                    dr_term = term
                    break

        if dr_term is None:
            return

        params = dr_term.params
        mapping = {
            "physics_mass_dr": "mass_scale",
            "physics_friction_dr": "friction_scale",
            "physics_com_offset": "com_offset",
            "disturbance_wind_force": "wind_force_range",
            "disturbance_push_force": "push_force_range",
            "disturbance_push_ratio": "push_env_ratio",
        }

        for spec in self.specs:
            param_key = mapping.get(spec.name)
            if param_key is None:
                continue
            if param_key == "com_offset" or param_key == "push_env_ratio":
                params[param_key] = spec.current[1]
            else:
                params[param_key] = tuple(spec.current)

    def apply_to_obs_terms(self, env_cfg) -> None:
        """Patch LiDAR obs terms (works at reset-time for per-episode DR)."""
        from charge_env_overrides import _find_lidar_obs_terms

        lidar_mapping = {
            "lidar_displacement_std": "displacement_std",
            "lidar_hole_rate": "hole_rate",
            "lidar_block_dropout_prob": "block_dropout_prob",
        }

        for spec in self.specs:
            obs_key = lidar_mapping.get(spec.name)
            if obs_key is None:
                continue
            for _, _, term in _find_lidar_obs_terms(env_cfg):
                if obs_key in term.params:
                    term.params[obs_key] = spec.current[1]

    def summary(self) -> str:
        """Human-readable summary of current DR ranges."""
        lines = [f"[DORAEMON] expansion #{self.expansion_count} "
                 f"(τ={self.threshold}, rate={self.expansion_rate})"]
        for spec in self.specs:
            if spec.is_bool:
                continue
            lo, hi = spec.current
            blo, bhi = spec.bounds
            pct = 0.0
            denom = abs(bhi - spec.initial[1]) + abs(blo - spec.initial[0])
            if denom > 1e-8:
                numer = abs(hi - spec.initial[1]) + abs(lo - spec.initial[0])
                pct = numer / denom * 100
            lines.append(f"  {spec.name}: [{lo:.4f}, {hi:.4f}] "
                         f"(→ [{blo:.4f}, {bhi:.4f}], {pct:.0f}%)")
        if self._fully_expanded:
            lines.append("  ** FULLY EXPANDED **")
        return "\n".join(lines)

    def to_wandb_dict(self) -> dict[str, float]:
        """Export current ranges for WandB logging."""
        d: dict[str, float] = {
            "doraemon/expansion_count": float(self.expansion_count),
            "doraemon/fully_expanded": float(self._fully_expanded),
        }
        for spec in self.specs:
            if spec.is_bool:
                continue
            d[f"doraemon/{spec.name}_lo"] = spec.current[0]
            d[f"doraemon/{spec.name}_hi"] = spec.current[1]
        return d
