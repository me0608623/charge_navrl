"""致動器域隨機化 (Actuator Domain Randomization)

模擬真實馬達和傳動系統的非理想行為，提高策略對控制不確定性的魯棒性。

功能：
    1. 動作延遲 (Action Delay): 模擬通訊/處理延遲 1-2 步
    2. 速度縮放 (Velocity Scaling): 模擬馬達效率差異 ±10%
    3. 回應滯後 (Motor Response Lag): 一階低通濾波模擬馬達響應慣性

參數:
    action_delay_steps: tuple = (0, 2)     — 延遲步數範圍
    velocity_scale: tuple = (0.9, 1.1)     — 速度縮放範圍
    response_lag_alpha: float = 0.1        — 滯後濾波係數 (0=全滯後, 1=無滯後)

訓練流程整合:
    - action_delay: 在 env.step() 中插入，作為 pre_physics_step callback
    - velocity_scaling: 在動作執行前縮放 applied velocity
    - response_lag: 在動作執行後平滑實際輸出

Isaac Lab 整合:
    使用 EventTerm(mode="interval") 或在 wrapper 層實現。
    動作延遲需要在 wrapper (aac_wrapper.py) 中實現 action buffer。

參考:
    - Andrychowicz et al. (2020) "What Matters in On-Policy RL" — 動作延遲分析
    - Tan et al. (2018) "Sim-to-Real: Learning Agile Locomotion" — 致動器 DR
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from collections import deque

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def apply_action_delay(
    env: ManagerBasedRLEnv,
    actions: torch.Tensor,
    delay_steps: tuple[int, int] = (0, 2),
) -> torch.Tensor:
    """動作延遲 — 模擬通訊/處理延遲。

    物理意義: 真實系統中，從感測器讀取到動作執行之間存在
    通訊延遲 (CAN bus ~2ms)、計算延遲 (推論 ~5ms)、
    馬達響應延遲 (~10ms)。在 5Hz 控制頻率下，
    1-2 步延遲 (0.2-0.4s) 是保守估計。

    實現: 維護一個 per-env 的動作歷史緩衝區。
    每次 reset 時重新採樣延遲步數。

    Args:
        env: 環境實例
        actions: [num_envs, action_dim] 當前動作
        delay_steps: (min_delay, max_delay) 延遲步數範圍

    Returns:
        [num_envs, action_dim] 延遲後的動作
    """
    if not hasattr(env, "_action_delay_buffer"):
        max_delay = delay_steps[1]
        env._action_delay_buffer = deque(maxlen=max_delay + 1)
        # 初始化為零動作
        for _ in range(max_delay + 1):
            env._action_delay_buffer.append(
                torch.zeros_like(actions)
            )
        # 為每個 env 採樣延遲步數
        env._action_delay_per_env = torch.randint(
            delay_steps[0], delay_steps[1] + 1,
            (env.num_envs,), device=env.device
        )

    # 偵測 reset 的 env 並重新採樣延遲
    just_reset = env.episode_length_buf == 0
    if just_reset.any():
        env._action_delay_per_env[just_reset] = torch.randint(
            delay_steps[0], delay_steps[1] + 1,
            (just_reset.sum().item(),), device=env.device
        )
        # 清空 reset env 的歷史
        for buf in env._action_delay_buffer:
            buf[just_reset] = 0.0

    # 推入當前動作
    env._action_delay_buffer.append(actions.clone())

    # 根據每個 env 的延遲步數取對應歷史動作
    delayed_actions = actions.clone()
    buffer_list = list(env._action_delay_buffer)
    for delay in range(delay_steps[0], delay_steps[1] + 1):
        mask = env._action_delay_per_env == delay
        if mask.any() and delay < len(buffer_list):
            idx = len(buffer_list) - 1 - delay
            delayed_actions[mask] = buffer_list[idx][mask]

    return delayed_actions


def apply_velocity_scaling(
    env: ManagerBasedRLEnv,
    velocities: torch.Tensor,
    scale_range: tuple[float, float] = (0.9, 1.1),
) -> torch.Tensor:
    """速度縮放 — 模擬馬達效率差異。

    物理意義: 真實馬達的實際輸出與指令之間存在差異：
    - 電池電壓下降 → 輸出力矩降低
    - 傳動齒輪磨損 → 效率下降
    - 溫度升高 → 馬達性能變化
    ±10% 覆蓋正常工作範圍。

    實現: 每次 reset 時為每個 env 採樣縮放因子，整個 episode 保持不變。

    Args:
        env: 環境實例
        velocities: [num_envs, 2] (v_linear, omega) 速度指令
        scale_range: 縮放範圍 (0.9, 1.1) = ±10%

    Returns:
        [num_envs, 2] 縮放後的速度指令
    """
    if not hasattr(env, "_velocity_scale_factors"):
        env._velocity_scale_factors = torch.empty(
            env.num_envs, 2, device=env.device
        ).uniform_(scale_range[0], scale_range[1])

    # 偵測 reset 並重新採樣
    just_reset = env.episode_length_buf == 0
    if just_reset.any():
        env._velocity_scale_factors[just_reset] = torch.empty(
            just_reset.sum().item(), 2, device=env.device
        ).uniform_(scale_range[0], scale_range[1])

    return velocities * env._velocity_scale_factors


def apply_motor_response_lag(
    env: ManagerBasedRLEnv,
    target_velocity: torch.Tensor,
    alpha: float | tuple[float, float] = 0.3,
) -> torch.Tensor:
    """馬達回應滯後 — 一階低通濾波。

    物理意義: 真實馬達不能瞬間達到目標速度。
    機械慣性和電氣時間常數導致實際速度漸進追蹤指令。
    一階濾波: y_t = alpha * u_t + (1 - alpha) * y_{t-1}
    在 control_dt=0.2 s 時，alpha=0.3 的等效時間常數為
    tau=-dt/log(1-alpha)=0.56 s，90% 上升時間約 2.303*tau=1.29 s。
    這個數值必須由實車 step response 校準，不能只由通訊 dead time 推定。
    ``alpha`` 可為 legacy scalar，或 ``(alpha_v, alpha_omega)``。線速度與
    角速度的底盤響應通常不同，新的 sim-to-real 血緣應使用分通道校準值。

    Args:
        env: 環境實例
        target_velocity: [num_envs, 2] 目標速度 (v, omega)
        alpha: scalar 或 (alpha_v, alpha_omega)，各值皆須落在 [0, 1]

    Returns:
        [num_envs, 2] 濾波後的實際速度
    """
    if target_velocity.ndim != 2 or target_velocity.shape[1] != 2:
        raise ValueError(
            "target_velocity must have shape [num_envs, 2] for (v, omega), "
            f"got {tuple(target_velocity.shape)}"
        )

    if isinstance(alpha, (int, float)):
        alpha_values = (float(alpha), float(alpha))
    else:
        try:
            alpha_values = tuple(float(value) for value in alpha)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "alpha must be a scalar or (alpha_v, alpha_omega)"
            ) from exc
        if len(alpha_values) != 2:
            raise ValueError(
                "channel-specific alpha must contain exactly "
                f"(alpha_v, alpha_omega), got {alpha_values}"
            )

    if any(value < 0.0 or value > 1.0 for value in alpha_values):
        raise ValueError(
            f"motor lag alpha values must stay in [0, 1], got {alpha_values}"
        )

    if not hasattr(env, "_prev_actual_velocity"):
        env._prev_actual_velocity = torch.zeros_like(target_velocity)

    # Reset env 的前一速度清零
    just_reset = env.episode_length_buf == 0
    if just_reset.any():
        env._prev_actual_velocity[just_reset] = 0.0

    # 一階低通濾波
    alpha_tensor = target_velocity.new_tensor(alpha_values).view(1, 2)
    actual = (
        alpha_tensor * target_velocity
        + (1.0 - alpha_tensor) * env._prev_actual_velocity
    )
    env._prev_actual_velocity = actual.clone()

    return actual


def apply_actuator_dynamics(
    env: ManagerBasedRLEnv,
    target_velocity: torch.Tensor,
    delay_steps: tuple[int, int] = (0, 2),
    scale_range: tuple[float, float] = (0.9, 1.1),
    response_lag_alpha: float | tuple[float, float] = 0.3,
) -> torch.Tensor:
    """Apply the frozen actuator-DR pipeline to decoded ``(v, omega)`` commands.

    The real vehicle decodes policy logits before publishing ``cmd_vel``; the
    communication/actuator delay therefore acts on the decoded velocity command,
    not on a categorical action index. Keeping this order also lets the policy
    observation retain the issued-command queue needed to make a delayed MDP
    Markov.
    """
    delayed = apply_action_delay(env, target_velocity, delay_steps)
    scaled = apply_velocity_scaling(env, delayed, scale_range)
    return apply_motor_response_lag(env, scaled, response_lag_alpha)


__all__ = [
    "apply_action_delay",
    "apply_velocity_scaling",
    "apply_motor_response_lag",
    "apply_actuator_dynamics",
]
