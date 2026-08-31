"""Contracts for the event-level domain-randomization on/off switch."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import types


REPO = Path(__file__).resolve().parents[4]
SKRL = REPO / "scripts/reinforcement_learning/skrl"
OVERRIDES = SKRL / "utils/charge_env_overrides.py"
TRAINER = SKRL / "train/train_rnn_car_wdclip.py"


def _load_by_path(alias: str, path: Path):
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _calls_in_main(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    main = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "main"
    )
    calls = []
    for node in sorted(
        (node for node in ast.walk(main) if isinstance(node, ast.Call)),
        key=lambda node: (node.lineno, node.col_offset),
    ):
        if isinstance(node.func, ast.Name):
            calls.append(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            calls.append(node.func.attr)
    return calls


def _env_cfg():
    params = {
        "enable_physics": True,
        "enable_sensor_noise": True,
        "enable_external_force": True,
        "sentinel": "unchanged",
    }
    term = types.SimpleNamespace(params=params)
    return types.SimpleNamespace(
        events=types.SimpleNamespace(domain_randomization=term)
    )


def test_no_domain_randomization_disables_every_event_level_dr_family():
    overrides = _load_by_path("_domain_switch_overrides", OVERRIDES)
    env_cfg = _env_cfg()

    changed = overrides._apply_domain_randomization_switch(
        env_cfg, types.SimpleNamespace(no_domain_randomization=True)
    )

    assert changed is True
    assert env_cfg.events.domain_randomization.params == {
        "enable_physics": False,
        "enable_sensor_noise": False,
        "enable_external_force": False,
        "sentinel": "unchanged",
    }


def test_enabled_domain_randomization_leaves_event_defaults_untouched():
    overrides = _load_by_path("_domain_switch_enabled_overrides", OVERRIDES)
    env_cfg = _env_cfg()

    changed = overrides._apply_domain_randomization_switch(
        env_cfg, types.SimpleNamespace(no_domain_randomization=False)
    )

    assert changed is False
    assert env_cfg.events.domain_randomization.params["enable_physics"] is True
    assert env_cfg.events.domain_randomization.params["enable_sensor_noise"] is True
    assert env_cfg.events.domain_randomization.params["enable_external_force"] is True


def test_trainer_calls_switch_before_dr_ranges_are_applied():
    calls = _calls_in_main(TRAINER)
    assert "_apply_domain_randomization_switch" in calls
    assert calls.index("_apply_actuator_dr_config") < calls.index(
        "_apply_domain_randomization_switch"
    )
    assert calls.index("_apply_domain_randomization_switch") < calls.index(
        "_apply_dr_param_overrides"
    )


def test_shared_play_wrapper_uses_the_same_switch():
    source = OVERRIDES.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(OVERRIDES))
    wrapper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "apply_charge_env_overrides"
    )
    calls = {
        node.func.id
        for node in ast.walk(wrapper)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "_apply_domain_randomization_switch" in calls
