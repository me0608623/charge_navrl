"""Pure geometry for the deployment-corridor replay scene."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


MOTION_LATERAL = 0
MOTION_LONGITUDINAL = 1
MOTION_RANDOM_2D = 2
DYNAMIC_MOTION_MODES = (
    "lateral",
    "longitudinal",
    "random_2d",
    "mixed",
    "mixed_iid",
    "env_stratified",
)


@dataclass(frozen=True)
class LongCorridorSpec:
    """Frozen interaction geometry with an independently sealable wall span."""

    free_width: float = 4.0
    length: float = 10.0
    wall_span_length: float | None = None
    wall_thickness: float = 1.0
    wall_height: float = 3.0
    robot_start_y: float = -4.1
    goal_y: float = 4.1
    static_x: float = 1.25
    static_y: tuple[float, ...] = (-3.0, -0.6, 0.6, 3.0)
    dynamic_y: tuple[float, ...] = (-1.8, 1.8)
    dynamic_x_limit: float = 1.30
    obstacle_radius: float = 0.35
    wall_clearance: float = 0.10
    robot_half_length: float = 0.35
    robot_half_width: float = 0.30
    robot_buffer: float = 0.10
    static_xy_jitter: float = 0.10
    dynamic_y_jitter: float = 0.08

    @property
    def wall_center_offset(self) -> float:
        return 0.5 * (self.free_width + self.wall_thickness)

    @property
    def inner_half_width(self) -> float:
        return 0.5 * self.free_width

    @property
    def physical_wall_span(self) -> float:
        """Physical side-wall length; legacy callers retain the 10 m span."""
        if self.wall_span_length is None:
            return self.length
        return self.wall_span_length

    @property
    def robot_conservative_radius(self) -> float:
        return math.hypot(
            self.robot_half_length + self.robot_buffer,
            self.robot_half_width + self.robot_buffer,
        )

    @property
    def centerline_static_clearance(self) -> float:
        return (
            self.static_x
            - self.static_xy_jitter
            - self.obstacle_radius
            - self.robot_conservative_radius
        )


def validate_spec(spec: LongCorridorSpec) -> None:
    """Reject geometry that cannot guarantee the intended constructive route."""
    if spec.free_width <= 0.0 or spec.length <= 0.0:
        raise ValueError("corridor width and length must be positive")
    if spec.wall_thickness <= 0.0:
        raise ValueError("wall thickness must be positive")
    if spec.physical_wall_span < spec.length:
        raise ValueError(
            "physical corridor wall span cannot be shorter than the "
            "interaction length"
        )
    if len(spec.static_y) != 4 or len(spec.dynamic_y) != 2:
        raise ValueError("deployment corridor requires exactly 4 static and 2 dynamic obstacles")
    if abs(spec.robot_start_y) >= 0.5 * spec.length:
        raise ValueError("robot start must stay inside the corridor")
    if abs(spec.goal_y) >= 0.5 * spec.length:
        raise ValueError("goal must stay inside the corridor")
    max_center_x = (
        spec.inner_half_width - spec.obstacle_radius - spec.wall_clearance
    )
    if spec.static_x + spec.static_xy_jitter > max_center_x:
        raise ValueError("static obstacle can overlap a corridor wall")
    if spec.dynamic_x_limit > max_center_x:
        raise ValueError("dynamic patrol can overlap a corridor wall")
    if spec.centerline_static_clearance <= 0.0:
        raise ValueError("static obstacles do not leave a conservative centerline route")

    max_center_y = 0.5 * spec.length - spec.obstacle_radius - spec.wall_clearance
    all_y = (*spec.static_y, *spec.dynamic_y)
    if max(abs(y) for y in all_y) + spec.static_xy_jitter > max_center_y:
        raise ValueError("obstacle template can leave the corridor ends")

    min_crossing_separation = min(
        abs(dynamic_y - static_y)
        for dynamic_y in spec.dynamic_y
        for static_y in spec.static_y
    )
    required = 2.0 * spec.obstacle_radius + (
        spec.static_xy_jitter + spec.dynamic_y_jitter
    )
    if min_crossing_separation <= required:
        raise ValueError("dynamic patrol lane can overlap a static obstacle")


#: 歷史模板的上限。`long_corridor_obstacle_count_mix=None` 時一律沿用這組，
#: 既有 gate 與 SA1–SA6 血緣因此完全不變、不需重訓。
LEGACY_MAX_STATIC = 4
LEGACY_MAX_DYNAMIC = 2

#: 2026-07-27 裁決把高密度混合場的上限提到 5S+5D。動態能放到 5 的前提是
#: motion family 配額把橫向壓在 2 以內（橫向需獨占橫排，走廊放不下 5 條）；
#: 縱向與隨機走中央帶 x=±0.40，與靜態 x=±1.25 左右錯開，不佔排。
MIXED_MAX_STATIC = 5
MIXED_MAX_DYNAMIC = 5


def validate_obstacle_counts(
    static_obstacles: int,
    dynamic_obstacles: int,
    *,
    max_static: int = LEGACY_MAX_STATIC,
    max_dynamic: int = LEGACY_MAX_DYNAMIC,
) -> None:
    """Validate a curriculum subset of the corridor template.

    預設維持歷史 4S+2D 上限 —— 呼叫端必須**明確**傳入更高的上限才會放行，
    避免舊路徑因為一次改動就默默接受高密度場景。
    """
    if not 0 <= int(static_obstacles) <= int(max_static):
        raise ValueError(
            f"corridor static obstacle count must be in [0, {int(max_static)}]"
        )
    if not 0 <= int(dynamic_obstacles) <= int(max_dynamic):
        raise ValueError(
            f"corridor dynamic obstacle count must be in [0, {int(max_dynamic)}]"
        )


def validate_dynamic_motion_mode(mode: str) -> str:
    """Return a normalized deployment-corridor dynamic motion mode."""
    normalized = str(mode).strip().lower()
    if normalized not in DYNAMIC_MOTION_MODES:
        raise ValueError(
            f"unsupported corridor motion mode {mode!r}; "
            f"expected one of {DYNAMIC_MOTION_MODES}"
        )
    return normalized


def normalize_dynamic_motion_weights(
    weights, mode: str
) -> tuple[float, float, float] | None:
    """Validate optional per-family env weights for ``env_stratified``.

    Weights are ``(lateral, longitudinal, random_2d)`` and are renormalized to
    sum to one. ``None`` keeps the balanced 1:1:1 stratification. Supplying
    weights for any other motion mode raises ``ValueError`` so that a
    misconfigured experiment fails loudly instead of silently ignoring them.
    """
    if weights is None:
        return None
    normalized_mode = validate_dynamic_motion_mode(mode)
    if normalized_mode != "env_stratified":
        raise ValueError(
            "corridor motion weights are only supported for "
            f"mode='env_stratified', got mode={normalized_mode!r}"
        )
    values = tuple(float(w) for w in weights)
    if len(values) != 3:
        raise ValueError(
            f"expected three corridor motion weights, got {len(values)}"
        )
    if any(not math.isfinite(w) or w < 0.0 for w in values):
        raise ValueError(
            f"corridor motion weights must be finite and non-negative, got {values}"
        )
    total = sum(values)
    if total <= 0.0:
        raise ValueError("corridor motion weights must sum to a positive value")
    return tuple(w / total for w in values)


def normalize_pause_steps_range(value) -> tuple[int, int] | None:
    """Validate an eval-only patrol pause override.

    ``None`` means "leave the scheduler alone". Anything else must be an
    ordered, non-negative ``(low, high)`` step range so a typo fails loudly
    instead of silently reshaping the obstacle motion the gate measures.
    """
    if value is None:
        return None
    values = tuple(int(v) for v in value)
    if len(values) != 2:
        raise ValueError(
            f"pause_steps_range must have two entries, got {len(values)}"
        )
    low, high = values
    if low < 0 or high < 0:
        raise ValueError(
            f"pause_steps_range must be non-negative, got {values}"
        )
    if low > high:
        raise ValueError(f"pause_steps_range must be ordered, got {values}")
    return (low, high)


def apply_pause_override(env, value) -> tuple[int, int] | None:
    """Apply the eval-only pause override and report the range now in force.

    Returns the range actually in effect so callers can record it in the
    evaluation JSON, which is what makes a pause A/B auditable after the fact.
    A ``None`` override never writes to the scheduler, so training and every
    historical gate keep the stock 0-5 step behaviour.
    """
    normalized = normalize_pause_steps_range(value)
    scheduler = getattr(getattr(env, "unwrapped", env), "_behavior_scheduler", None)
    patrol = getattr(getattr(scheduler, "cfg", None), "patrol", None)
    if patrol is None:
        return normalized
    if normalized is not None:
        patrol.pause_steps_range = normalized
    return tuple(int(v) for v in patrol.pause_steps_range)


_RANDOM_2D_KINEMATICS = ("patrol", "wander")


def validate_random_2d_kinematics(value: str) -> str:
    """Validate the random_2d motion implementation.

    ``patrol`` is the frozen two-point ping-pong used by every historical gate.
    ``wander`` swaps in a bounded random walk with no fixed turnaround point.
    """
    normalized = str(value).strip().lower()
    if normalized not in _RANDOM_2D_KINEMATICS:
        raise ValueError(
            f"unsupported random_2d kinematics {value!r}; "
            f"expected one of {_RANDOM_2D_KINEMATICS}"
        )
    return normalized


def corridor_wander_bounds(spec: LongCorridorSpec) -> tuple[float, float, float, float]:
    """Axis-aligned free area a wandering obstacle may occupy.

    Inset by the obstacle radius plus the wall clearance so a reflecting
    obstacle never overlaps a corridor wall.
    """
    x_limit = spec.inner_half_width - spec.obstacle_radius - spec.wall_clearance
    y_limit = 0.5 * spec.length - spec.obstacle_radius - spec.wall_clearance
    return (-x_limit, x_limit, -y_limit, y_limit)


def sample_motion_families(
    count: int,
    weights: tuple[float, float, float],
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Random-phase systematic stratification of ``count`` envs across families.

    Resets arrive in small, variable batches, so a deterministic largest-
    remainder quota would bias every batch the same way and systematically
    starve low-weight families (with weights 0.30/0.10/0.60 a single-env reset
    always produced random_2d, and longitudinal never appeared below seven
    envs). Random-phase systematic sampling keeps the per-call allocation within
    one env of ``count * weight`` while staying unbiased in expectation for any
    batch size, including ``count == 1``. It is fully determined by the torch
    RNG, so runs stay reproducible.
    """
    if count <= 0:
        return torch.zeros(0, dtype=torch.long, device=device)
    weights_tensor = torch.tensor(weights, dtype=torch.float64, device=device)
    boundaries = torch.cumsum(weights_tensor, dim=0)[:-1]
    phase = torch.rand((), dtype=torch.float64, device=device)
    points = (
        torch.arange(count, device=device, dtype=torch.float64) + phase
    ) / count
    families = torch.bucketize(points, boundaries, right=True)
    return families[torch.randperm(count, device=device)].to(torch.long)


def wall_geometry(
    count: int,
    spec: LongCorridorSpec,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return local wall centers/sizes for `count` mirrored-identical corridors."""
    validate_spec(spec)
    centers = torch.zeros(count, 2, 2, device=device)
    centers[:, 0, 0] = -spec.wall_center_offset
    centers[:, 1, 0] = spec.wall_center_offset
    sizes = torch.zeros(count, 2, 2, device=device)
    sizes[:, :, 0] = spec.wall_thickness
    sizes[:, :, 1] = spec.physical_wall_span
    return centers, sizes


def wall_boundary_overlap(
    spec: LongCorridorSpec,
    *,
    room_half_extent: float,
    boundary_wall_width: float,
) -> float:
    """Return side-wall overlap with each north/south boundary wall in metres.

    A negative value is a traversable geometric opening before robot size is
    considered. Zero means face contact; a positive value is deliberate
    overlap and is preferred for robust collision geometry.
    """
    room_half_extent = float(room_half_extent)
    boundary_wall_width = float(boundary_wall_width)
    if room_half_extent <= 0.0:
        raise ValueError("room_half_extent must be positive")
    if boundary_wall_width <= 0.0:
        raise ValueError("boundary_wall_width must be positive")
    boundary_inner_face = room_half_extent - 0.5 * boundary_wall_width
    return 0.5 * spec.physical_wall_span - boundary_inner_face


def sample_obstacle_layout(
    count: int,
    spec: LongCorridorSpec,
    device: torch.device | str,
    *,
    max_static: int = LEGACY_MAX_STATIC,
    max_dynamic: int = LEGACY_MAX_DYNAMIC,
    permute_slots: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample safe static positions, dynamic starts and two-point patrol paths.

    Slot 數量可變（`max_static` / `max_dynamic`），預設仍是歷史的 4S+2D ——
    高密度混合場必須明確傳入更大的值。

    模板只有 ``len(spec.static_y)`` 列與 ``len(spec.dynamic_y)`` 條橫排；
    超出的 slot 與既有列配對（靜態補上對側形成「門口」，中央通道
    ``2*static_x - 2*radius`` ≈ 1.8 m）或重用橫排（動態，由上層的 motion
    family 配額保證橫向不超過橫排數）。

    ``permute_slots`` 逐 env 打散 slot→列/橫排 的綁定。高密度混合場用**前綴**
    active mask 關掉多餘 slot，若不打散，S=3 就永遠只留下前三列（最靠目標的
    y=3.0 那列一次都不會出現）。預設 False，既有血緣逐位元不變。
    """
    validate_spec(spec)
    max_static = int(max_static)
    max_dynamic = int(max_dynamic)
    mirror = torch.where(
        torch.rand(count, device=device) < 0.5,
        torch.full((count,), -1.0, device=device),
        torch.ones(count, device=device),
    )

    rows = len(spec.static_y)
    # 模板列的 x 交替 ±static_x；超出的 slot 回頭與既有列配對（補對側）。
    slot_rows = [i % rows for i in range(max_static)]
    slot_signs = [
        (-1.0 if (i % 2 == 0) else 1.0) if i < rows
        # 配對 slot 取該列的反側，形成左右各一的「門口」。
        else (1.0 if ((i % rows) % 2 == 0) else -1.0)
        for i in range(max_static)
    ]
    base_static_x = torch.tensor(
        [sign * spec.static_x for sign in slot_signs], device=device
    )
    base_static_y = torch.tensor(
        [spec.static_y[r] for r in slot_rows], device=device
    )
    static = torch.zeros(count, max_static, 2, device=device)
    static[:, :, 0] = mirror[:, None] * base_static_x[None, :]
    static[:, :, 1] = base_static_y[None, :]
    static += torch.empty_like(static).uniform_(
        -spec.static_xy_jitter, spec.static_xy_jitter
    )

    lanes = len(spec.dynamic_y)
    lane_values = [spec.dynamic_y[i % lanes] for i in range(max_dynamic)]
    dynamic_y = torch.tensor(lane_values, device=device)[None, :].expand(count, -1)
    dynamic_y = dynamic_y + torch.empty_like(dynamic_y).uniform_(
        -spec.dynamic_y_jitter, spec.dynamic_y_jitter
    )
    dynamic = torch.zeros(count, max_dynamic, 2, device=device)
    dynamic[:, :, 0] = torch.empty(count, max_dynamic, device=device).uniform_(
        -spec.dynamic_x_limit, spec.dynamic_x_limit
    )
    dynamic[:, :, 1] = dynamic_y

    waypoints = torch.zeros(count, max_dynamic, 2, 2, device=device)
    waypoints[:, :, 0, 0] = -spec.dynamic_x_limit
    waypoints[:, :, 1, 0] = spec.dynamic_x_limit
    waypoints[:, :, :, 1] = dynamic_y[:, :, None]

    if permute_slots:
        static = _permute_slots(static, count, max_static, device)
        perm_d = torch.argsort(torch.rand(count, max_dynamic, device=device), dim=1)
        dynamic = torch.gather(dynamic, 1, perm_d[:, :, None].expand(-1, -1, 2))
        waypoints = torch.gather(
            waypoints, 1, perm_d[:, :, None, None].expand(-1, -1, 2, 2)
        )
    return static, dynamic, waypoints


def _permute_slots(
    tensor: torch.Tensor, count: int, slots: int, device
) -> torch.Tensor:
    """Independently shuffle the slot axis of ``[count, slots, 2]`` per env."""
    perm = torch.argsort(torch.rand(count, slots, device=device), dim=1)
    return torch.gather(tensor, 1, perm[:, :, None].expand(-1, -1, 2))


def sample_dynamic_trajectories(
    dynamic: torch.Tensor,
    lateral_waypoints: torch.Tensor,
    spec: LongCorridorSpec,
    mode: str,
    motion_weights: tuple[float, float, float] | None = None,
    motion_types: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build controlled dynamic paths for one corridor motion family.

    Returns starts, two-point patrol paths, per-obstacle motion type IDs, and
    the first target waypoint index.

    ``mixed`` deals both obstacles from a *balanced* pool of ``2 * count``
    slots rather than drawing each independently. Under auto-reset the common
    batch size is ``count == 1``, where the pool holds two consecutive family
    ids and the two obstacles are therefore always *different* -- measured
    homogeneous pairs are 11.2% against the 33.3% an IID mixture implies. This
    behaviour is retained unchanged as the legacy regression gate so historical
    numbers stay comparable; ``mixed_iid`` is the unbiased per-obstacle
    variant. ``env_stratified`` assigns one family to both dynamic obstacles in an
    environment while balancing families across the batch. This matches the
    pure-family regression gates without removing heterogeneous mixed scenes.
    """
    normalized = validate_dynamic_motion_mode(mode)
    if motion_types is not None:
        # Fixed-family resampling: redraw positions only. Spawn-conflict
        # rejection must NOT redraw the family assignment, otherwise pairs
        # whose geometry conflicts more often (measured: 54% of lat+long
        # spawns overlap the partner lane) get silently under-represented and
        # the measured pair distribution stops matching the sampler's.
        if motion_types.shape != (dynamic.shape[0], 2):
            raise ValueError(
                f"motion_types must be [{dynamic.shape[0]}, 2], got "
                f"{tuple(motion_types.shape)}"
            )
    weights = normalize_dynamic_motion_weights(motion_weights, normalized)
    # Slot 數量可變（2026-07-27 高密度混合場）；只檢查維度與最後一軸，
    # 不再寫死 2 —— 寫死會讓 5 個動態的場景在此被誤判成形狀錯誤。
    if dynamic.ndim != 3 or dynamic.shape[-1] != 2:
        raise ValueError(f"expected dynamic [N,2,2], got {tuple(dynamic.shape)}")
    if lateral_waypoints.shape != (dynamic.shape[0], dynamic.shape[1], 2, 2):
        raise ValueError(
            "expected lateral_waypoints [N,2,2,2], got "
            f"{tuple(lateral_waypoints.shape)}"
        )

    count = dynamic.shape[0]
    device = dynamic.device
    lateral_starts = dynamic.clone()
    lateral_targets = (
        torch.rand(count, 2, device=device) < 0.5
    ).long()

    # One lane approaches the robot while the other initially travels with it.
    longitudinal_starts = torch.zeros_like(dynamic)
    longitudinal_starts[:, 0, 0] = -0.40
    longitudinal_starts[:, 0, 1] = abs(float(spec.dynamic_y[1]))
    longitudinal_starts[:, 1, 0] = 0.40
    longitudinal_starts[:, 1, 1] = -abs(float(spec.dynamic_y[0]))
    longitudinal_waypoints = torch.zeros_like(lateral_waypoints)
    longitudinal_y_limit = min(
        3.4,
        0.5 * spec.length - spec.obstacle_radius - spec.wall_clearance,
    )
    longitudinal_waypoints[:, :, 0, 1] = -longitudinal_y_limit
    longitudinal_waypoints[:, :, 1, 1] = longitudinal_y_limit
    longitudinal_waypoints[:, 0, :, 0] = -0.40
    longitudinal_waypoints[:, 1, :, 0] = 0.40
    longitudinal_targets = torch.tensor(
        [0, 1], dtype=torch.long, device=device
    ).expand(count, -1).clone()

    # Randomized diagonal patrols stay in separate longitudinal bands and in
    # the center strip, so they cannot tunnel through the static side obstacles.
    random_waypoints = torch.zeros_like(lateral_waypoints)
    x_left = torch.empty(count, 2, device=device).uniform_(-0.35, -0.15)
    x_right = torch.empty(count, 2, device=device).uniform_(0.15, 0.35)
    swap_x = torch.rand(count, 2, device=device) < 0.5
    random_waypoints[:, :, 0, 0] = torch.where(
        swap_x, x_right, x_left
    )
    random_waypoints[:, :, 1, 0] = torch.where(
        swap_x, x_left, x_right
    )
    random_outer_y = min(
        3.3,
        0.5 * spec.length - spec.obstacle_radius - spec.wall_clearance,
    )
    random_waypoints[:, 0, 0, 1] = torch.empty(
        count, device=device
    ).uniform_(-random_outer_y, -2.1)
    random_waypoints[:, 0, 1, 1] = torch.empty(
        count, device=device
    ).uniform_(-1.8, -0.8)
    random_waypoints[:, 1, 0, 1] = torch.empty(
        count, device=device
    ).uniform_(0.8, 1.8)
    random_waypoints[:, 1, 1, 1] = torch.empty(
        count, device=device
    ).uniform_(2.1, random_outer_y)
    random_starts = random_waypoints[:, :, 0].clone()
    random_targets = torch.ones(count, 2, dtype=torch.long, device=device)

    if motion_types is not None:
        pass  # families fixed by the caller; only positions are redrawn
    elif normalized == "lateral":
        motion_types = torch.full(
            (count, 2), MOTION_LATERAL, dtype=torch.long, device=device
        )
    elif normalized == "longitudinal":
        motion_types = torch.full(
            (count, 2), MOTION_LONGITUDINAL, dtype=torch.long, device=device
        )
    elif normalized == "random_2d":
        motion_types = torch.full(
            (count, 2), MOTION_RANDOM_2D, dtype=torch.long, device=device
        )
    elif normalized == "mixed":
        flat_count = count * 2
        offset = int(torch.randint(0, 3, (1,), device=device).item())
        balanced = (
            torch.arange(flat_count, device=device) + offset
        ) % (MOTION_RANDOM_2D + 1)
        motion_types = balanced[
            torch.randperm(flat_count, device=device)
        ].reshape(count, 2)
    elif normalized == "mixed_iid":
        # Per-obstacle independent uniform draw. ``mixed`` deals from a
        # perfectly balanced pool, so a single-env reset (count == 1, the
        # common case under auto-reset) always yields two *different*
        # families and homogeneous pairs are under-sampled (measured 11.2%
        # against the 33.3% an IID mixture implies). ``mixed_iid`` is the
        # unbiased variant; ``mixed`` is retained untouched as the legacy
        # regression gate so historical numbers stay comparable.
        motion_types = torch.randint(
            0,
            MOTION_RANDOM_2D + 1,
            (count, 2),
            dtype=torch.long,
            device=device,
        )
    elif weights is None:
        offset = int(torch.randint(0, 3, (1,), device=device).item())
        balanced = (
            torch.arange(count, device=device) + offset
        ) % (MOTION_RANDOM_2D + 1)
        env_motion_types = balanced[torch.randperm(count, device=device)]
        motion_types = env_motion_types[:, None].expand(-1, 2).clone()
    else:
        # Weighted env-level stratification. Random-phase systematic sampling
        # keeps small reset batches unbiased; both obstacles of an env share the
        # drawn family.
        env_motion_types = sample_motion_families(count, weights, device)
        motion_types = env_motion_types[:, None].expand(-1, 2).clone()

    starts = lateral_starts
    waypoints = lateral_waypoints.clone()
    target_indices = lateral_targets
    longitudinal = motion_types == MOTION_LONGITUDINAL
    random_2d = motion_types == MOTION_RANDOM_2D
    starts = torch.where(
        longitudinal[..., None], longitudinal_starts, starts
    )
    starts = torch.where(random_2d[..., None], random_starts, starts)
    waypoints = torch.where(
        longitudinal[..., None, None], longitudinal_waypoints, waypoints
    )
    waypoints = torch.where(
        random_2d[..., None, None], random_waypoints, waypoints
    )
    target_indices = torch.where(
        longitudinal, longitudinal_targets, target_indices
    )
    target_indices = torch.where(
        random_2d, random_targets, target_indices
    )
    return starts, waypoints, motion_types, target_indices


def layout_is_constructively_solvable(
    static: torch.Tensor,
    dynamic: torch.Tensor,
    waypoints: torch.Tensor,
    spec: LongCorridorSpec,
) -> torch.Tensor:
    """Check wall bounds and the static centerline route for each sampled env."""
    validate_spec(spec)
    # 同上：靜態 slot 數可變（4 或 5），只鎖維度與最後一軸。
    if static.ndim != 3 or static.shape[-1] != 2:
        raise ValueError(f"expected static [N,4,2], got {tuple(static.shape)}")
    if dynamic.ndim != 3 or dynamic.shape[0] != static.shape[0] or dynamic.shape[-1] != 2:
        raise ValueError(f"expected dynamic [N,2,2], got {tuple(dynamic.shape)}")
    if waypoints.shape != (static.shape[0], dynamic.shape[1], 2, 2):
        raise ValueError(f"expected waypoints [N,2,2,2], got {tuple(waypoints.shape)}")

    max_x = spec.inner_half_width - spec.obstacle_radius - spec.wall_clearance
    max_y = 0.5 * spec.length - spec.obstacle_radius - spec.wall_clearance
    all_points = torch.cat(
        [static, dynamic, waypoints.reshape(static.shape[0], -1, 2)], dim=1
    )
    inside = (
        (all_points[..., 0].abs() <= max_x)
        & (all_points[..., 1].abs() <= max_y)
    ).all(dim=1)

    centerline_clearance = (
        static[..., 0].abs()
        - spec.obstacle_radius
        - spec.robot_conservative_radius
    )
    centerline_open = (centerline_clearance > 0.0).all(dim=1)

    # Controlled patrol segments must not intersect the fixed side obstacles.
    point = static[:, :, None, :]
    segment_start = waypoints[:, None, :, 0, :]
    segment_delta = (
        waypoints[:, None, :, 1, :]
        - segment_start
    )
    projection = (
        ((point - segment_start) * segment_delta).sum(dim=-1)
        / segment_delta.square().sum(dim=-1).clamp_min(1e-9)
    ).clamp(0.0, 1.0)
    closest = segment_start + projection[..., None] * segment_delta
    dynamic_static_clear = (
        torch.linalg.vector_norm(point - closest, dim=-1)
        > (2.0 * spec.obstacle_radius)
    ).all(dim=(1, 2))
    return inside & centerline_open & dynamic_static_clear


def corridor_penetration_masks(
    dynamic_pos: torch.Tensor,
    static_pos: torch.Tensor,
    obstacle_radius: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-slot physical-overlap masks for corridor dynamic obstacles.

    The constructive-solvability check runs at *install* time, before any
    motion; the wander walk only reflects off the corridor walls and knows
    nothing about the other obstacles, so overlap with a static obstacle or
    the other dynamic obstacle is possible in principle and must be measured
    rather than assumed away. Two same-radius discs overlap when their
    centres are closer than ``2 * obstacle_radius``.

    Args:
        dynamic_pos: ``[E, S, 2]`` dynamic obstacle centres (local frame).
        static_pos:  ``[E, K, 2]`` static obstacle centres (local frame).
        obstacle_radius: shared disc radius in metres.

    Returns:
        ``(dyn_static, dyn_dyn)`` boolean masks, both ``[E, S]``: whether each
        dynamic slot currently penetrates any static obstacle / any *other*
        dynamic slot.
    """
    if obstacle_radius <= 0.0:
        raise ValueError(f"obstacle_radius must be positive, got {obstacle_radius}")
    threshold = 2.0 * float(obstacle_radius)

    delta_static = dynamic_pos[:, :, None, :] - static_pos[:, None, :, :]
    dyn_static = (
        delta_static.norm(dim=-1) < threshold
    ).any(dim=-1) if static_pos.shape[1] > 0 else torch.zeros(
        dynamic_pos.shape[:2], dtype=torch.bool, device=dynamic_pos.device
    )

    delta_dyn = dynamic_pos[:, :, None, :] - dynamic_pos[:, None, :, :]
    pair_overlap = delta_dyn.norm(dim=-1) < threshold
    slots = dynamic_pos.shape[1]
    eye = torch.eye(slots, dtype=torch.bool, device=dynamic_pos.device)
    dyn_dyn = (pair_overlap & ~eye).any(dim=-1)
    return dyn_static, dyn_dyn


def corridor_penetration_pass(
    audited_slot_frames: int,
    dynamic_static_count: int,
) -> bool:
    """Hard gate on dynamic-vs-static physical penetration.

    Dynamic-dynamic overlap is *not* part of this gate by ruling: scripted
    pedestrians move independently and may cross; the eval audit records the
    overlap fraction for the report but does not fail on it. What must never
    happen is a dynamic obstacle passing through a static one (or a wall).

    Zero counts alone are not enough: an audit that never ran (zero audited
    frames) would trivially report zero penetrations, so the gate also
    requires that frames were actually audited. This is what stops a wiring
    failure from being graded as a clean run.
    """
    return (
        int(audited_slot_frames) > 0
        and int(dynamic_static_count) == 0
    )


def sample_conflict_free_layout(
    count: int,
    spec: LongCorridorSpec,
    device,
    mode: str,
    motion_weights: tuple[float, float, float] | None = None,
    static_obstacles: int = 4,
    dynamic_obstacles: int = 2,
    max_tries: int = 20,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample a corridor layout whose *spawn* is free of physical overlap.

    Dynamic obstacles are allowed to cross each other while moving (scripted
    pedestrians are independent), but two obstacles spawning inside each other
    merge into one — measured 54% of lat+long draws put the lateral start
    inside the partner's lane. Rejected environments are redrawn with their
    **motion families fixed**, so feasibility-biased families (lat+long) are
    not silently under-represented. A postcondition raises after
    ``max_tries`` instead of shipping an overlapped scene.

    Returns ``(static, dynamic, waypoints, motion_types, target_indices)``.
    """
    static, dynamic, waypoints = sample_obstacle_layout(count, spec, device)
    dynamic, waypoints, motion_types, target_indices = (
        sample_dynamic_trajectories(
            dynamic, waypoints, spec, mode, motion_weights
        )
    )
    # max_tries + 1 checks: the final resample must be re-checked too, or a
    # last-attempt success would be misreported as a failure.
    clean = None
    for attempt in range(max_tries + 1):
        solvable = layout_is_constructively_solvable(
            static, dynamic, waypoints, spec
        )
        dyn_active = dynamic[:, :dynamic_obstacles]
        static_active = static[:, :static_obstacles]
        ds, dd = corridor_penetration_masks(
            dyn_active, static_active, spec.obstacle_radius
        )
        clean = solvable & ~ds.any(dim=1) & ~dd.any(dim=1)
        if bool(clean.all()):
            return static, dynamic, waypoints, motion_types, target_indices
        if attempt == max_tries:
            break
        bad = ~clean
        n_bad = int(bad.sum().item())
        s2, d2, w2 = sample_obstacle_layout(n_bad, spec, device)
        d2, w2, _, g2 = sample_dynamic_trajectories(
            d2, w2, spec, mode, motion_weights,
            motion_types=motion_types[bad],
        )
        static[bad] = s2
        dynamic[bad] = d2
        waypoints[bad] = w2
        target_indices[bad] = g2
    raise RuntimeError(
        f"corridor spawn resampling failed to clear overlaps within "
        f"{max_tries} tries for {int((~clean).sum().item())} envs "
        f"(mode={mode!r}); refusing to install an overlapped scene"
    )


def static_layout_id(
    static_xy: torch.Tensor,
    active_counts: torch.Tensor | None = None,
) -> torch.Tensor:
    """Encode the discrete static skeleton as ``active * 100 + sign_bitmask``.

    ``static_xy`` is ``[count, slots, 2]``. Bit ``i`` of the mask is set when
    slot ``i`` sits at ``+x``; inactive slots are parked at ``x = 0`` and so
    read as bit 0. The number of *active* static obstacles is folded into the id
    because the bitmask alone cannot separate densities — an all-negative
    2-obstacle layout and an all-negative 3-obstacle layout are both mask 0.

    ``active_counts`` is required whenever ``static_xy`` carries inactive slots
    (the mixed-density path passes all slots); when omitted the caller must have
    already sliced to the active prefix, and ``slots`` is used as the count.

    Purely diagnostic: the id never feeds an observation, reward or action.
    """

    if static_xy.ndim != 3 or static_xy.shape[-1] != 2:
        raise ValueError("static_xy must have shape [count, slots, 2]")
    count, slots = static_xy.shape[0], static_xy.shape[1]
    if slots > 6:
        raise ValueError("static layout id supports at most 6 static slots")
    device = static_xy.device
    if slots == 0:
        return torch.zeros(count, dtype=torch.long, device=device)
    bits = (static_xy[:, :, 0] > 0).long()
    weights = 2 ** torch.arange(slots, device=device, dtype=torch.long)
    mask = (bits * weights).sum(dim=-1)
    if active_counts is None:
        counts = torch.full((count,), slots, dtype=torch.long, device=device)
    else:
        counts = active_counts.reshape(-1).long()
        if counts.numel() != count:
            raise ValueError("active_counts must have one entry per env")
    return counts * 100 + mask

