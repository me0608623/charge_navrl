"""Actuator-delay bridge：接線與延遲語意的行為測試（純 CPU，不需 Isaac Sim）。

分兩層，各自守不同的失效模式：

**接線層**（`ActuatorWiringTest`）
    ExperimentConfig -> args_cli -> `env_cfg.actions.diff_drive` 的值必須**逐項相等**。
    只斷言 config 的字面值不夠 —— 那證明不了值真的送達 action term。
    另外守一個具體的靜默失效：`no_domain_randomization=True`
    **不得**把 action-term 的致動器 DR 一起關掉（它關的是 events 那條 DR，
    兩者是不同機制；W1 parent 正是 `no_domain_randomization=True`，
    若這兩者被綁在一起，bridge 會在沒有任何錯誤訊息的情況下退化成無延遲訓練）。

**語意層**（`ActionDelaySemanticsTest`）
    直接驅動 `apply_action_delay`，斷言 d=0 執行當步、d=1 執行前一步、
    d=2 執行前兩步；取樣上下界 inclusive；per-env 在 episode 內固定；
    reset 才重抽。這些都是**行為**，不是設定值。

裁決規格（不得更動）：delay = U{0,1,2} 步，control_dt 0.2 s → 0/200/400 ms；
`actuator_motor_lag=0.3`；`actuator_velocity_scale=(0.9,1.1)`；
`obs_delay_steps=(0,0)` 永遠關。
"""

import ast
import importlib.util
import pathlib
import sys
import tempfile
import types
import unittest

import torch

_REPO = pathlib.Path(__file__).resolve().parents[4]
_SKRL = _REPO / "scripts" / "reinforcement_learning" / "skrl"
if str(_SKRL) not in sys.path:
    sys.path.insert(0, str(_SKRL))

_ACTUATOR_DR = (
    _REPO
    / "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity"
    / "config/charge_skrl/domain_randomization/actuator_dr.py"
)

_ACTION_TERM = (
    _REPO
    / "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity"
    / "config/charge_skrl/mdp/actions/discrete_differential_drive.py"
)

_OBS_FUNCTIONS = (
    _REPO
    / "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity"
    / "config/charge_skrl/mdp/observations/obs_functions.py"
)


_OVERRIDES = _SKRL / "utils" / "charge_env_overrides.py"
_FIXED_EVAL = (
    _SKRL / "rnn_car_wdclean" / "fixed_actuator_eval.py"
)
_GATE2_SUITE = (
    _SKRL / "rnn_car_wdclean" / "run_gate2_suite.py"
)


def _load_by_path(alias: str, path: pathlib.Path):
    """以檔案路徑載入模組。

    `utils` 這個套件名在本 repo 會與其他已載入的 `utils` 撞名
    （ROS 的 site-packages 也提供一個），直接 `from utils... import` 會拿到
    錯的那個。用路徑載入把歧義完全消掉。
    """
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


adr = _load_by_path("_actuator_dr", _ACTUATOR_DR)
_overrides = _load_by_path("_charge_env_overrides", _OVERRIDES)
_apply_actuator_dr_config = _overrides._apply_actuator_dr_config
_fixed_eval = _load_by_path("_fixed_actuator_eval", _FIXED_EVAL)
_gate2_suite = _load_by_path("_run_gate2_suite", _GATE2_SUITE)

from rnn_car_modular.configs.e2e_deploy_bridge_actuator_delay_from_w1c10 import (  # noqa: E402
    CONFIG as BRIDGE,
)
from rnn_car_modular.experiment_config import apply_experiment_config  # noqa: E402


#: control_dt。delay 步數乘上它就是毫秒規格。
CONTROL_DT_S = 0.2


class _StubDiffDrive:
    """模擬 `env_cfg.actions.diff_drive` —— 只要有這幾個欄位就會被接線。"""

    def __init__(self):
        self.enable_actuator_dr = False
        self.actuator_delay_range = (0, 0)
        self.actuator_velocity_scale = (1.0, 1.0)
        self.actuator_motor_lag = 1.0


class _StubEnvCfg:
    def __init__(self):
        self.actions = types.SimpleNamespace(diff_drive=_StubDiffDrive())


class _StubEnv:
    """`apply_action_delay` 只用到 num_envs / device / episode_length_buf。"""

    def __init__(self, num_envs=8, device="cpu"):
        self.num_envs = num_envs
        self.device = device
        self.episode_length_buf = torch.ones(num_envs, dtype=torch.long, device=device)

    def mark_reset(self, env_ids):
        self.episode_length_buf[:] = 1
        self.episode_length_buf[env_ids] = 0


# ---------------------------------------------------------------------------
# 接線層
# ---------------------------------------------------------------------------


class ActuatorWiringTest(unittest.TestCase):
    def _args_cli_from_bridge(self):
        args = types.SimpleNamespace()
        # argv 不含任何 flag -> config 的值不會被「使用者 CLI 覆寫」邏輯跳過。
        apply_experiment_config(args, BRIDGE, argv=["prog"])
        return args

    def test_bridge_config_reaches_args_cli(self):
        args = self._args_cli_from_bridge()
        self.assertIs(args.enable_actuator_dr, True)
        self.assertEqual(tuple(args.actuator_delay_range), (0, 2))
        self.assertEqual(tuple(args.actuator_velocity_scale), (0.9, 1.1))
        self.assertEqual(args.actuator_motor_lag, 0.3)
        self.assertEqual(tuple(args.obs_delay_steps), (0, 0))
        self.assertIs(args.use_action_history, True)

    def test_args_cli_reaches_action_term_field_by_field(self):
        """args_cli -> env_cfg.actions.diff_drive 必須逐項相等。"""
        args = self._args_cli_from_bridge()
        env_cfg = _StubEnvCfg()
        _apply_actuator_dr_config(env_cfg, args)

        diff = env_cfg.actions.diff_drive
        self.assertIs(diff.enable_actuator_dr, True)
        self.assertEqual(diff.actuator_delay_range, tuple(args.actuator_delay_range))
        self.assertEqual(
            diff.actuator_velocity_scale, tuple(args.actuator_velocity_scale)
        )
        self.assertEqual(diff.actuator_motor_lag, float(args.actuator_motor_lag))
        # 並且等於裁決的字面規格。
        self.assertEqual(diff.actuator_delay_range, (0, 2))
        self.assertEqual(diff.actuator_velocity_scale, (0.9, 1.1))
        self.assertEqual(diff.actuator_motor_lag, 0.3)

    def test_no_domain_randomization_does_not_silently_disable_actuator_dr(self):
        """W1 parent 是 `no_domain_randomization=True`。

        那個旗標關的是 `events.domain_randomization`（physics / sensor noise /
        external force），**不是** action term 的致動器 DR。若哪天有人把兩者綁在
        一起，bridge 會安靜地退化成無延遲訓練 —— 沒有例外、沒有警告，
        只有一份看起來正常卻沒學到延遲的權重。
        """
        self.assertIs(
            BRIDGE.no_domain_randomization,
            True,
            "前提改變：本測試假設 bridge 繼承 parent 的 no_domain_randomization=True",
        )
        args = self._args_cli_from_bridge()
        self.assertIs(args.no_domain_randomization, True)

        env_cfg = _StubEnvCfg()
        _apply_actuator_dr_config(env_cfg, args)
        self.assertIs(
            env_cfg.actions.diff_drive.enable_actuator_dr,
            True,
            "no_domain_randomization 把 action-term 致動器 DR 一起關掉了",
        )
        self.assertEqual(env_cfg.actions.diff_drive.actuator_delay_range, (0, 2))

    def test_actuator_dr_off_leaves_action_term_untouched(self):
        """預設關閉時必須是完全的 no-op，既有 config 不受影響。"""
        args = self._args_cli_from_bridge()
        args.enable_actuator_dr = False
        env_cfg = _StubEnvCfg()
        _apply_actuator_dr_config(env_cfg, args)

        diff = env_cfg.actions.diff_drive
        self.assertIs(diff.enable_actuator_dr, False)
        self.assertEqual(diff.actuator_delay_range, (0, 0))
        self.assertEqual(diff.actuator_motor_lag, 1.0)

    def test_actuator_dr_on_requires_a_supported_action_term(self):
        """明確要求 DR 時，不得因 action term 不支援而靜默退化。"""
        args = self._args_cli_from_bridge()

        for env_cfg in (
            types.SimpleNamespace(),
            types.SimpleNamespace(actions=types.SimpleNamespace()),
            types.SimpleNamespace(
                actions=types.SimpleNamespace(diff_drive=types.SimpleNamespace())
            ),
        ):
            with self.subTest(env_cfg=env_cfg):
                with self.assertRaisesRegex(
                    RuntimeError, "enable_actuator_dr=True"
                ):
                    _apply_actuator_dr_config(env_cfg, args)

    def test_delay_range_maps_to_the_ruled_millisecond_spec(self):
        lo, hi = BRIDGE.actuator_delay_range
        self.assertEqual((lo, hi), (0, 2))
        millis = [round(d * CONTROL_DT_S * 1000) for d in range(lo, hi + 1)]
        self.assertEqual(millis, [0, 200, 400])


class FixedDelayGateWiringTest(unittest.TestCase):
    """Deployment gates must fix delay without dropping scale/lag or provenance."""

    def test_fixed_delay_cli_is_the_frozen_actuator_bundle(self):
        for delay in (0, 1, 2):
            with self.subTest(delay=delay):
                self.assertEqual(
                    _fixed_eval.fixed_actuator_cli_args(delay),
                    [
                        "--enable_actuator_dr",
                        "--actuator_delay_range",
                        str(delay),
                        str(delay),
                        "--actuator_velocity_scale",
                        "0.9",
                        "1.1",
                        "--actuator_motor_lag",
                        "0.3",
                    ],
                )

    def test_fixed_delay_metadata_is_self_describing(self):
        metadata = _fixed_eval.fixed_actuator_metadata(1)
        self.assertIs(metadata["enabled"], True)
        self.assertEqual(metadata["delay_steps"], 1)
        self.assertEqual(metadata["delay_ms"], 200)
        self.assertEqual(metadata["velocity_scale_range"], [0.9, 1.1])
        self.assertEqual(metadata["motor_lag_alpha"], 0.3)
        self.assertEqual(metadata["fixed_component"], "delay_only")
        self.assertEqual(
            metadata["pipeline"], "decode->delay->scale->lag"
        )
        self.assertEqual(metadata["history"], "issued_command_queue")

    def test_invalid_fixed_delay_fails_closed(self):
        for delay in (-1, 3, True):
            with self.subTest(delay=delay):
                with self.assertRaises(ValueError):
                    _fixed_eval.fixed_actuator_cli_args(delay)

    def test_runtime_marker_is_verified_field_by_field(self):
        complete = (
            "[SIM2REAL] Actuator DR: delay=(1, 1) steps, "
            "vel_scale=(0.9, 1.1), motor_lag alpha=0.3, "
            "pipeline=decode->delay->scale->lag, "
            "history=issued_command_queue\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            log = pathlib.Path(tmp) / "play.log"
            log.write_text(complete, encoding="utf-8")
            _fixed_eval.verify_fixed_actuator_runtime(log, 1)

            log.write_text(
                complete.replace("delay=(1, 1)", "delay=(0, 2)"),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                RuntimeError, "actuator gate wiring incomplete"
            ):
                _fixed_eval.verify_fixed_actuator_runtime(log, 1)

    def test_all_deployment_gate_runners_apply_and_verify_the_bundle(self):
        runners = (
            "run_gate2_suite.py",
            "run_corridor_motion_suite.py",
            "run_narrow_path_suite.py",
        )
        root = _SKRL / "rnn_car_wdclean"
        for runner in runners:
            with self.subTest(runner=runner):
                tree = ast.parse(
                    (root / runner).read_text(encoding="utf-8"),
                    filename=runner,
                )
                main = next(
                    node
                    for node in tree.body
                    if isinstance(node, ast.FunctionDef)
                    and node.name == "main"
                )
                calls = {
                    node.func.id
                    if isinstance(node.func, ast.Name)
                    else node.func.attr
                    if isinstance(node.func, ast.Attribute)
                    else ""
                    for node in ast.walk(main)
                    if isinstance(node, ast.Call)
                }
                self.assertIn("fixed_actuator_cli_args", calls)
                self.assertIn("fixed_actuator_metadata", calls)
                self.assertIn("verify_fixed_actuator_runtime", calls)

    def test_gate2_pools_action_stability_without_averaging_rms(self):
        reports = [
            {
                "samples": 100,
                "omega_squared_mean_rad2_s2": 0.25,
                "full_steer_fraction": 0.10,
                "ratio_flip_rate_mean": 0.20,
                "ratio_flip_rate_p95": 0.30,
                "omega_abs_std_p95_rad_s": 0.40,
                "dominant_frequency_p95_hz": 0.50,
                "max_same_sign_turn_p95_s": 1.2,
            },
            {
                "samples": 300,
                "omega_squared_mean_rad2_s2": 1.00,
                "full_steer_fraction": 0.30,
                "ratio_flip_rate_mean": 0.10,
                "ratio_flip_rate_p95": 0.25,
                "omega_abs_std_p95_rad_s": 0.60,
                "dominant_frequency_p95_hz": 0.75,
                "max_same_sign_turn_p95_s": 0.8,
            },
        ]
        aggregate = _gate2_suite._aggregate_jitter(reports)
        self.assertEqual(aggregate["samples"], 400)
        self.assertAlmostEqual(
            aggregate["omega_rms_rad_s"],
            ((100 * 0.25 + 300 * 1.0) / 400) ** 0.5,
        )
        self.assertAlmostEqual(aggregate["full_steer_fraction"], 0.25)
        self.assertEqual(
            aggregate["worst_seed_ratio_flip_rate_p95"], 0.30
        )
        self.assertEqual(
            aggregate["worst_seed_dominant_frequency_p95_hz"], 0.75
        )

    def test_gate2_requests_machine_readable_jitter_audit(self):
        source = _GATE2_SUITE.read_text(encoding="utf-8")
        self.assertIn('"--jitter_eval_output"', source)
        self.assertIn('"action_stability"', source)


# ---------------------------------------------------------------------------
# 語意層 —— 直接驅動 apply_action_delay
# ---------------------------------------------------------------------------


class ActionDelaySemanticsTest(unittest.TestCase):
    """d 步延遲必須真的執行 d 步前的動作。"""

    def setUp(self):
        torch.manual_seed(0)
        self.n = 6
        self.env = _StubEnv(num_envs=self.n)

    def _drive(self, per_env_delay, n_steps=6, delay_range=(0, 2)):
        """餵入可辨識的動作序列，回傳每一步的 delayed 輸出。

        第 t 步的動作值 = t（每個 env 都一樣），所以輸出值直接就是
        「這一步執行的是第幾步的動作」，不需要再反推。
        """
        outputs = []
        for t in range(n_steps):
            actions = torch.full((self.n, 2), float(t))
            out = adr.apply_action_delay(self.env, actions, delay_range)
            if t == 0:
                # 緩衝區在第一次呼叫時建立，之後才能覆寫 per-env 延遲。
                self.env._action_delay_per_env = per_env_delay.clone()
                out = adr.apply_action_delay(self.env, actions, delay_range)
            outputs.append(out.clone())
        return outputs

    def test_delay_zero_executes_current_step(self):
        delays = torch.zeros(self.n, dtype=torch.long)
        outs = self._drive(delays)
        for t, out in enumerate(outs):
            self.assertTrue(
                torch.all(out == float(t)), f"step {t}: d=0 應執行當步，得到 {out[0,0]}"
            )

    def test_delay_one_executes_previous_step(self):
        delays = torch.ones(self.n, dtype=torch.long)
        outs = self._drive(delays)
        # 前幾步緩衝區還是暖機值，從 t>=2 起語意穩定。
        for t in range(2, len(outs)):
            self.assertTrue(
                torch.all(outs[t] == float(t - 1)),
                f"step {t}: d=1 應執行 t-1 的動作，得到 {outs[t][0,0]}",
            )

    def test_delay_two_executes_two_steps_ago(self):
        delays = torch.full((self.n,), 2, dtype=torch.long)
        outs = self._drive(delays)
        for t in range(3, len(outs)):
            self.assertTrue(
                torch.all(outs[t] == float(t - 2)),
                f"step {t}: d=2 應執行 t-2 的動作，得到 {outs[t][0,0]}",
            )

    def test_mixed_delays_are_per_env(self):
        """同一批 env 可以各自帶不同延遲，互不干擾。"""
        delays = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.long)
        outs = self._drive(delays)
        t = 5
        expected = torch.tensor([t - int(d) for d in delays], dtype=torch.float)
        self.assertTrue(
            torch.all(outs[t][:, 0] == expected),
            f"per-env 延遲錯亂：期望 {expected.tolist()}，得到 {outs[t][:,0].tolist()}",
        )

    def test_sampling_bounds_are_inclusive(self):
        """U{0,1,2} 的兩端都必須抽得到，且不得超出。"""
        env = _StubEnv(num_envs=4096)
        actions = torch.zeros(env.num_envs, 2)
        adr.apply_action_delay(env, actions, (0, 2))
        sampled = env._action_delay_per_env
        self.assertEqual(int(sampled.min()), 0, "下界 0 沒被抽到")
        self.assertEqual(int(sampled.max()), 2, "上界 2 沒被抽到（inclusive 失效）")
        self.assertEqual(
            sorted(set(sampled.tolist())), [0, 1, 2], "抽樣值域不是 {0,1,2}"
        )

    def test_delay_is_fixed_within_an_episode(self):
        """episode 內不得每步重抽 —— 那會變成隨機延遲，不是固定延遲。"""
        env = _StubEnv(num_envs=64)
        actions = torch.zeros(env.num_envs, 2)
        adr.apply_action_delay(env, actions, (0, 2))
        first = env._action_delay_per_env.clone()
        for _ in range(10):
            adr.apply_action_delay(env, actions, (0, 2))
        self.assertTrue(
            torch.equal(first, env._action_delay_per_env),
            "episode 內延遲被重抽了",
        )

    def test_delay_is_resampled_only_on_reset(self):
        """只有 episode_length_buf == 0 的 env 會重抽，其餘保持不變。"""
        env = _StubEnv(num_envs=512)
        actions = torch.zeros(env.num_envs, 2)
        adr.apply_action_delay(env, actions, (0, 2))
        before = env._action_delay_per_env.clone()

        reset_ids = torch.arange(0, 256)
        env.mark_reset(reset_ids)
        adr.apply_action_delay(env, actions, (0, 2))
        after = env._action_delay_per_env

        keep = torch.arange(256, 512)
        self.assertTrue(
            torch.equal(before[keep], after[keep]),
            "未 reset 的 env 延遲被動到了",
        )
        # 重抽過的那半邊值域仍須合法（不保證數值改變 —— 重抽可能抽到同值）。
        self.assertTrue(bool((after[reset_ids] >= 0).all()))
        self.assertTrue(bool((after[reset_ids] <= 2).all()))

    def test_reset_clears_the_action_history(self):
        """reset 後不得把上一個 episode 的動作當成自己的歷史執行。"""
        env = _StubEnv(num_envs=4)
        for t in range(4):
            adr.apply_action_delay(env, torch.full((4, 2), float(t + 10)), (0, 2))
        env._action_delay_per_env = torch.full((4,), 2, dtype=torch.long)

        env.mark_reset(torch.arange(4))
        out = adr.apply_action_delay(env, torch.zeros(4, 2), (0, 2))
        self.assertTrue(
            torch.all(out == 0.0),
            f"reset 後仍讀到上個 episode 的動作：{out[0,0]}",
        )

    def test_actuator_pipeline_delays_decoded_velocity_commands(self):
        """DR pipeline 應延遲解碼後的 (v, omega)，而非離散 action index。"""
        env = _StubEnv(num_envs=3)
        first_target = torch.tensor(
            [[0.1, 0.6], [0.2, -0.6], [-0.1, 0.0]], dtype=torch.float
        )
        second_target = torch.tensor(
            [[0.3, 1.0], [0.4, -1.0], [0.0, 0.2]], dtype=torch.float
        )

        first_applied = adr.apply_actuator_dynamics(
            env,
            first_target,
            delay_steps=(1, 1),
            scale_range=(1.0, 1.0),
            response_lag_alpha=1.0,
        )
        second_applied = adr.apply_actuator_dynamics(
            env,
            second_target,
            delay_steps=(1, 1),
            scale_range=(1.0, 1.0),
            response_lag_alpha=1.0,
        )

        self.assertTrue(torch.equal(first_applied, torch.zeros_like(first_target)))
        self.assertTrue(
            torch.equal(second_applied, first_target),
            f"d=1 應執行前一筆 decoded command，得到 {second_applied}",
        )


# ---------------------------------------------------------------------------
# Trainer 呼叫鏈 —— 守「函式存在但沒人叫它」這個靜默失效
# ---------------------------------------------------------------------------


class TrainerCallsActuatorDRTest(unittest.TestCase):
    """`_apply_actuator_dr_config` 必須真的被訓練入口呼叫。

    2026-07-28 的 wiring smoke 抓到：bridge config 正確地把
    `enable_actuator_dr=True` 送進 args_cli，`_apply_actuator_dr_config`
    本身也正確，但 `train_rnn_car_wdclip.py` **只 import 了另外三個 helper**
    （obb / lidar / dr_param），從未呼叫 actuator 那個 —— 於是
    `env_cfg.actions.diff_drive.enable_actuator_dr` 永遠是 False，
    整個 bridge 會在**沒有任何錯誤訊息**的情況下退化成無延遲訓練，
    log 裡連 `[SIM2REAL] Actuator DR` 那行都不會出現。

    上面的接線測試抓不到這個 —— 它直接呼叫 `_apply_actuator_dr_config`，
    等於預設了「有人會呼叫它」。這條測試補的就是那個預設。

    對照組：`play_rnn_car.py` 走 `apply_charge_env_overrides`（內含 actuator），
    所以 play 有延遲、train 沒有 —— 訓練與評測的致動器行為不一致。
    """

    def _source(self, relative: str) -> str:
        return (_SKRL / relative).read_text(encoding="utf-8")

    def _calls_in_function(self, relative: str, function_name: str) -> list[str]:
        """以 AST 讀取函式內的真實呼叫，忽略 import、註解與字串。"""
        tree = ast.parse(self._source(relative), filename=relative)
        function = next(
            (
                node
                for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == function_name
            ),
            None,
        )
        self.assertIsNotNone(
            function, f"{relative} 找不到函式 {function_name}()"
        )

        calls = []
        call_nodes = sorted(
            (node for node in ast.walk(function) if isinstance(node, ast.Call)),
            key=lambda node: (node.lineno, node.col_offset),
        )
        for node in call_nodes:
            if isinstance(node.func, ast.Name):
                calls.append(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.append(node.func.attr)
        return calls

    def test_trainer_applies_actuator_dr_config(self):
        calls = self._calls_in_function(
            "train/train_rnn_car_wdclip.py", "main"
        )
        self.assertIn(
            "_apply_actuator_dr_config",
            calls,
            "train_rnn_car_wdclip.main() 沒有呼叫 _apply_actuator_dr_config；"
            "enable_actuator_dr=True 的 config 會靜默地在無延遲下訓練",
        )
        self.assertLess(
            calls.index("_apply_lidar_noise_config"),
            calls.index("_apply_actuator_dr_config"),
            "actuator DR 應在 LiDAR config 後套用",
        )
        self.assertLess(
            calls.index("_apply_actuator_dr_config"),
            calls.index("_apply_dr_param_overrides"),
            "actuator DR 必須先於會受 no_domain_randomization 提早返回的 DR event helper",
        )

    def test_trainer_and_play_apply_the_same_actuator_wiring(self):
        """train 與 play 的致動器 DR 接線必須一致，否則訓練/評測行為不同。"""
        train_calls = self._calls_in_function(
            "train/train_rnn_car_wdclip.py", "main"
        )
        play_calls = self._calls_in_function("play_eval/play_rnn_car.py", "main")
        wrapper_calls = self._calls_in_function(
            "utils/charge_env_overrides.py", "apply_charge_env_overrides"
        )
        self.assertIn("_apply_actuator_dr_config", train_calls)
        self.assertIn(
            "apply_charge_env_overrides",
            play_calls,
            "play_rnn_car.main() 沒有呼叫共用 env override",
        )
        self.assertIn(
            "_apply_actuator_dr_config",
            wrapper_calls,
            "play 使用的共用 env override 沒有套用 actuator DR",
        )

    def test_delay_is_downstream_of_decode_and_history_uses_issued_commands(self):
        """延遲位置與 83D history 語意必須對齊車端 policy_node。"""
        action_tree = ast.parse(
            _ACTION_TERM.read_text(encoding="utf-8"), filename=str(_ACTION_TERM)
        )
        process = next(
            node
            for node in action_tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "DiscreteDifferentialDriveAction"
            for node in node.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "process_actions"
        )
        calls = [
            node
            for node in ast.walk(process)
            if isinstance(node, ast.Call)
        ]
        call_names = [
            node.func.id
            if isinstance(node.func, ast.Name)
            else node.func.attr
            if isinstance(node.func, ast.Attribute)
            else ""
            for node in calls
        ]
        self.assertNotIn(
            "apply_action_delay",
            call_names,
            "離散 index 仍在 decode 前被 delay",
        )
        pipeline_call = next(
            node
            for node in calls
            if (
                isinstance(node.func, ast.Name)
                and node.func.id == "apply_actuator_dynamics"
            )
        )
        self.assertGreaterEqual(len(pipeline_call.args), 2)
        self.assertIsInstance(pipeline_call.args[1], ast.Name)
        self.assertEqual(pipeline_call.args[1].id, "target_vel")

        target_build_lines = [
            node.lineno
            for node in ast.walk(process)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "target_vel"
                for target in node.targets
            )
            and isinstance(node.value, ast.Call)
            and (
                isinstance(node.value.func, ast.Attribute)
                and node.value.func.attr == "stack"
            )
        ]
        self.assertTrue(target_build_lines, "找不到 decoded target_vel 建立點")
        self.assertLess(min(target_build_lines), pipeline_call.lineno)

        obs_tree = ast.parse(
            _OBS_FUNCTIONS.read_text(encoding="utf-8"),
            filename=str(_OBS_FUNCTIONS),
        )
        history_fn = next(
            node
            for node in obs_tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "discrete_applied_action_history"
        )
        referenced_attrs = {
            node.attr
            for node in ast.walk(history_fn)
            if isinstance(node, ast.Attribute)
        }
        self.assertIn(
            "commanded_accelerations",
            referenced_attrs,
            "83D history 沒有讀 issued-command queue",
        )


if __name__ == "__main__":
    unittest.main()
