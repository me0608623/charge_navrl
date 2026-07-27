"""Sampling tests for the narrow-passage reset injector's layout controls.

These cover the play-side manual overrides (pinned gap/wall positions, one-way
direction, split start/goal distances, off-axis goal) and, just as importantly,
that omitting them leaves the training distribution byte-for-byte unchanged.
"""

import importlib.util
from pathlib import Path
import sys
import types

import pytest

torch = pytest.importorskip("torch")


_REPO = Path(__file__).resolve().parents[4]
_EVENTS_DIR = (
    _REPO
    / "source" / "isaaclab_tasks" / "isaaclab_tasks" / "manager_based"
    / "locomotion" / "velocity" / "config" / "charge_skrl" / "mdp" / "events"
)


def _load_events_package():
    """Load the two event modules under a synthetic package.

    ``narrow_passage_bridge`` uses a relative import, so it needs a parent
    package; importing the real one would drag in Isaac Sim.
    """
    pkg_name = "_npb_test_pkg"
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(_EVENTS_DIR)]
        sys.modules[pkg_name] = pkg
    loaded = {}
    for name in ("narrow_passage_bridge_geometry", "narrow_passage_bridge"):
        full = f"{pkg_name}.{name}"
        if full in sys.modules:
            loaded[name] = sys.modules[full]
            continue
        spec = importlib.util.spec_from_file_location(full, _EVENTS_DIR / f"{name}.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[full] = module
        spec.loader.exec_module(module)
        loaded[name] = module
    return loaded


_MODULES = _load_events_package()
setup_narrow_passage_bridge = _MODULES["narrow_passage_bridge"].setup_narrow_passage_bridge
TRAINING_REPLAY_LAYOUT = _MODULES["narrow_passage_bridge_geometry"].TRAINING_REPLAY_LAYOUT


class _FakeAsset:
    def __init__(self, num_envs: int) -> None:
        self.poses = torch.zeros(num_envs, 7)
        self.velocities = torch.zeros(num_envs, 6)
        self.data = types.SimpleNamespace(default_root_state=torch.zeros(num_envs, 13))

    def write_root_pose_to_sim(self, pose, env_ids=None):
        self.poses[env_ids] = pose

    def write_root_velocity_to_sim(self, velocity, env_ids=None):
        self.velocities[env_ids] = velocity


class _FakeScene:
    def __init__(self, num_envs: int) -> None:
        self.env_origins = torch.zeros(num_envs, 3)
        self._assets = {
            "robot": _FakeAsset(num_envs),
            "narrow_bridge_wall_0": _FakeAsset(num_envs),
            "narrow_bridge_wall_1": _FakeAsset(num_envs),
        }

    def __getitem__(self, key):
        return self._assets[key]

    def keys(self):
        return self._assets.keys()


class _FakeEnv:
    """The minimum surface ``setup_narrow_passage_bridge`` touches."""

    def __init__(self, num_envs: int = 256) -> None:
        self.num_envs = num_envs
        self.device = "cpu"
        self.common_step_counter = 0
        self.scene = _FakeScene(num_envs)
        self.unwrapped = self
        goal_term = types.SimpleNamespace(goal_pos_w=torch.zeros(num_envs, 3))
        self.command_manager = types.SimpleNamespace(get_term=lambda _name: goal_term)
        self._goal_term = goal_term


def _inject(env=None, *, seed: int = 0, **kwargs):
    """Run the injector on every env with a fixed seed; return the env."""
    torch.manual_seed(seed)
    env = env or _FakeEnv()
    defaults = dict(
        fraction=1.0,
        schedule_steps=1,
        room_half_extent=7.0,
        final_stress_ratio=0.0,
        fixed_width_range=(1.2, 1.4),
        fixed_yaw_limit_deg=4.0,
    )
    defaults.update(kwargs)
    setup_narrow_passage_bridge(env, None, **defaults)
    return env


def _scene_of(env):
    """Recover the per-env sampled geometry from what the injector wrote."""
    centers = env._narrow_bridge_wall_centers
    barrier_x = centers[:, 0, 0]
    gap_center = 0.5 * (centers[:, 0, 1] + centers[:, 1, 1])
    start_x = env.scene["robot"].poses[:, 0]
    goal_x = env._goal_term.goal_pos_w[:, 0]
    goal_y = env._goal_term.goal_pos_w[:, 1]
    return barrier_x, gap_center, start_x, goal_x, goal_y


# ---------------------------------------------------------------------------
# Historical pre-N1 defaults must not move
# ---------------------------------------------------------------------------


def test_default_kwargs_reproduce_the_training_ranges():
    env = _inject()
    barrier_x, gap_center, start_x, goal_x, _ = _scene_of(env)

    assert bool(env._narrow_bridge_active.all())
    assert float(gap_center.min()) >= -1.0 and float(gap_center.max()) <= 1.0
    assert float(barrier_x.min()) >= -0.5 and float(barrier_x.max()) <= 0.5
    # Wide enough sampling that both training limits are actually exercised.
    assert float(gap_center.max()) > 0.8 and float(gap_center.min()) < -0.8
    assert float(barrier_x.max()) > 0.4 and float(barrier_x.min()) < -0.4

    # Both crossing directions appear, mirrored around the barrier at 3.0 m.
    direction = torch.sign(goal_x - barrier_x)
    assert bool((direction > 0).any()) and bool((direction < 0).any())
    torch.testing.assert_close(goal_x - barrier_x, direction * 3.0)
    torch.testing.assert_close(start_x - barrier_x, -direction * 3.0)


def test_defaults_match_the_shared_training_layout_constant():
    d = TRAINING_REPLAY_LAYOUT
    explicit = _inject(
        seed=7,
        gap_center_range=d["gap_center_range"],
        barrier_x_range=d["barrier_x_range"],
        direction_mode=d["direction_mode"],
        start_goal_distance=d["start_distance"],
        goal_distance=d["goal_distance"],
        goal_lateral_offset=d["goal_lateral_offset"],
    )
    implicit = _inject(seed=7)
    for a, b in zip(_scene_of(explicit), _scene_of(implicit)):
        torch.testing.assert_close(a, b)


# ---------------------------------------------------------------------------
# Manual overrides
# ---------------------------------------------------------------------------


def test_pinned_ranges_produce_one_exact_scene():
    env = _inject(
        gap_center_range=(0.75, 0.75),
        barrier_x_range=(-0.4, -0.4),
        direction_mode="forward",
        start_goal_distance=2.0,
        goal_distance=4.0,
    )
    barrier_x, gap_center, start_x, goal_x, goal_y = _scene_of(env)

    torch.testing.assert_close(gap_center, torch.full_like(gap_center, 0.75))
    torch.testing.assert_close(barrier_x, torch.full_like(barrier_x, -0.4))
    torch.testing.assert_close(start_x, torch.full_like(start_x, -2.4))   # -0.4 - 2.0
    torch.testing.assert_close(goal_x, torch.full_like(goal_x, 3.6))      # -0.4 + 4.0
    torch.testing.assert_close(goal_y, torch.full_like(goal_y, 0.75))


def test_backward_direction_mirrors_every_env():
    env = _inject(barrier_x_range=(0.0, 0.0), direction_mode="backward")
    barrier_x, _, start_x, goal_x, _ = _scene_of(env)
    assert bool((goal_x < barrier_x).all())
    assert bool((start_x > barrier_x).all())


def test_forward_direction_never_mirrors():
    env = _inject(barrier_x_range=(0.0, 0.0), direction_mode="forward")
    barrier_x, _, start_x, goal_x, _ = _scene_of(env)
    assert bool((goal_x > barrier_x).all())
    assert bool((start_x < barrier_x).all())


def test_split_start_and_goal_distances_are_independent():
    env = _inject(
        barrier_x_range=(0.0, 0.0),
        direction_mode="forward",
        start_goal_distance=1.5,
        goal_distance=5.0,
    )
    _, _, start_x, goal_x, _ = _scene_of(env)
    torch.testing.assert_close(start_x, torch.full_like(start_x, -1.5))
    torch.testing.assert_close(goal_x, torch.full_like(goal_x, 5.0))


def test_goal_lateral_offset_moves_the_goal_but_not_the_gap():
    offset = -1.25
    plain = _inject(seed=3)
    shifted = _inject(seed=3, goal_lateral_offset=offset)

    _, gap_plain, _, goal_x_plain, goal_y_plain = _scene_of(plain)
    _, gap_shifted, _, goal_x_shifted, goal_y_shifted = _scene_of(shifted)

    torch.testing.assert_close(gap_plain, gap_shifted)
    torch.testing.assert_close(goal_x_plain, goal_x_shifted)
    torch.testing.assert_close(goal_y_shifted, goal_y_plain + offset)


def test_goal_lateral_offset_defaults_to_the_gap_center():
    env = _inject()
    _, gap_center, _, _, goal_y = _scene_of(env)
    torch.testing.assert_close(goal_y, gap_center)


def test_n1_goal_ranges_randomize_distance_and_lateral_offset():
    env = _inject(
        seed=11,
        goal_distance_range=(2.0, 4.0),
        goal_lateral_offset_range=(-1.5, 1.5),
    )
    barrier_x, gap_center, _, goal_x, goal_y = _scene_of(env)
    distance = (goal_x - barrier_x).abs()
    offset = goal_y - gap_center

    assert float(distance.min()) >= 2.0
    assert float(distance.max()) <= 4.0
    assert float(offset.min()) >= -1.5
    assert float(offset.max()) <= 1.5
    # The 256-env smoke must exercise both ends rather than silently pinning
    # the midpoint through a broken config/wiring path.
    assert float(distance.min()) < 2.1 and float(distance.max()) > 3.9
    assert float(offset.min()) < -1.4 and float(offset.max()) > 1.4
    torch.testing.assert_close(env._narrow_bridge_goal_distance_m, distance)
    torch.testing.assert_close(env._narrow_bridge_goal_lateral_offset_m, offset)


def test_n1_goal_ranges_fail_fast_when_reversed():
    with pytest.raises(ValueError, match="ordered"):
        _inject(goal_distance_range=(4.0, 2.0))
    with pytest.raises(ValueError, match="ordered"):
        _inject(goal_lateral_offset_range=(1.5, -1.5))


def test_start_stays_inside_the_gap_corridor_when_pinned():
    # The lateral start jitter is bounded by the straight-OBB clearance, so a
    # pinned gap must still spawn the robot inside the opening.
    env = _inject(gap_center_range=(1.0, 1.0), fixed_width_range=(1.4, 1.4))
    _, gap_center, _, _, _ = _scene_of(env)
    start_y = env.scene["robot"].poses[:, 1]
    clearance = 0.5 * 1.4 - 0.40
    assert bool(((start_y - gap_center).abs() <= min(0.30, 0.65 * clearance) + 1e-6).all())


def test_unknown_direction_mode_is_rejected():
    with pytest.raises(ValueError, match="direction_mode"):
        _inject(direction_mode="sideways")


def test_unsolvable_manual_override_still_fails_fast():
    # The injector keeps its hard assert; only the play-side resolver clamps.
    with pytest.raises(RuntimeError, match="unsolvable"):
        _inject(barrier_x_range=(0.0, 0.0), start_goal_distance=20.0)
