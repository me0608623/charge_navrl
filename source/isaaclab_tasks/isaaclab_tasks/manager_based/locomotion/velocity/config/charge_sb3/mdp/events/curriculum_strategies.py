"""課程學習策略模組

此模組實現多種課程學習策略，用於系統性比較不同策略的樣本效率。

學術依據：
- Bengio et al. (2009) "Curriculum Learning" ICML
- Narvekar et al. (2020) "Curriculum Learning for RL: A Framework and Survey" JMLR
- Florensa et al. (2017) "Reverse Curriculum Generation" CoRL

實驗設計：
- 比較 5 種課程策略：No Curriculum, Linear, Exponential, Logarithmic, Adaptive
- 評估指標：樣本效率、最終性能、穩定性、泛化能力
- 控制變量：PPO 超參數、獎勵結構、總訓練步數

使用方式：
    from .curriculum_strategies import get_curriculum_strategy, CurriculumStrategy

    # 選擇策略
    strategy = get_curriculum_strategy("adaptive")

    # 在訓練循環中使用
    difficulty = strategy.get_difficulty(
        current_step=10000,
        total_steps=500000,
        metrics={"success_rate": 0.75, "collision_rate": 0.15}
    )
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional, List
import torch


# ============================================================================
# 課程策略基類
# ============================================================================

@dataclass
class CurriculumMetrics:
    """課程學習的評估指標

    用於記錄訓練過程中的關鍵指標，供自適應策略使用。
    """
    success_rate: float = 0.0       # 成功率（到達目標）
    collision_rate: float = 0.0     # 碰撞率
    timeout_rate: float = 0.0       # 超時率
    episode_length: float = 0.0     # 平均 episode 長度
    episode_reward: float = 0.0     # 平均 episode 獎勵

    # 歷史記錄（用於計算趨勢）
    success_history: List[float] = field(default_factory=list)
    collision_history: List[float] = field(default_factory=list)


class CurriculumStrategy(ABC):
    """課程策略基類

    所有課程策略都需要繼承此類並實現 get_difficulty 方法。

    Attributes:
        min_difficulty: 最小難度（障礙物數量）
        max_difficulty: 最大難度（障礙物數量）
        name: 策略名稱（用於日誌和比較）
    """

    def __init__(
        self,
        min_difficulty: int = 3,
        max_difficulty: int = 10,
        name: str = "base",
    ):
        self.min_difficulty = min_difficulty
        self.max_difficulty = max_difficulty
        self.name = name
        self._current_difficulty = min_difficulty
        self._step_history: List[int] = []
        self._difficulty_history: List[int] = []

    @abstractmethod
    def get_difficulty(
        self,
        current_step: int,
        total_steps: int,
        metrics: Optional[CurriculumMetrics] = None,
    ) -> int:
        """計算當前應該使用的難度

        Args:
            current_step: 當前訓練步數
            total_steps: 總訓練步數
            metrics: 當前的訓練指標（自適應策略需要）

        Returns:
            難度值（障礙物數量），範圍 [min_difficulty, max_difficulty]
        """
        raise NotImplementedError

    def record_step(self, step: int, difficulty: int):
        """記錄每一步的難度（用於分析）"""
        self._step_history.append(step)
        self._difficulty_history.append(difficulty)

    def get_history(self) -> Dict[str, List]:
        """獲取歷史記錄"""
        return {
            "steps": self._step_history,
            "difficulties": self._difficulty_history,
        }

    def reset(self):
        """重置策略狀態"""
        self._current_difficulty = self.min_difficulty
        self._step_history = []
        self._difficulty_history = []


# ============================================================================
# 具體課程策略實現
# ============================================================================

class NoCurriculum(CurriculumStrategy):
    """無課程策略：直接在最高難度訓練

    這是一個 baseline，用於比較課程學習是否有效。

    學術意義：
    - 如果其他策略顯著優於此策略，說明課程學習有效
    - 如果此策略表現相當，說明任務不需要課程學習
    """

    def __init__(self, min_difficulty: int = 3, max_difficulty: int = 10):
        super().__init__(min_difficulty, max_difficulty, name="no_curriculum")

    def get_difficulty(
        self,
        current_step: int,
        total_steps: int,
        metrics: Optional[CurriculumMetrics] = None,
    ) -> int:
        """始終返回最高難度"""
        return self.max_difficulty


class LinearCurriculum(CurriculumStrategy):
    """線性課程策略：難度隨訓練進度線性增加

    D(t) = min_difficulty + (max_difficulty - min_difficulty) × (t / T)

    特點：
    - 難度增加速度恆定
    - 簡單直觀
    - 常用的 baseline 策略

    學術參考：
    - Bengio et al. (2009) 的原始 Curriculum Learning 論文使用類似方法
    """

    def __init__(self, min_difficulty: int = 3, max_difficulty: int = 10):
        super().__init__(min_difficulty, max_difficulty, name="linear")

    def get_difficulty(
        self,
        current_step: int,
        total_steps: int,
        metrics: Optional[CurriculumMetrics] = None,
    ) -> int:
        """線性增加難度"""
        if total_steps <= 0:
            return self.min_difficulty

        progress = min(current_step / total_steps, 1.0)
        difficulty_range = self.max_difficulty - self.min_difficulty
        difficulty = self.min_difficulty + difficulty_range * progress

        return int(round(difficulty))


class ExponentialCurriculum(CurriculumStrategy):
    """指數課程策略：難度先慢後快增加

    D(t) = min_difficulty + (max_difficulty - min_difficulty) × (t / T)^α

    其中 α > 1（預設 α = 2）

    特點：
    - 訓練早期難度增加緩慢，給 Agent 更多時間學習基礎
    - 訓練後期難度快速增加，挑戰 Agent 的極限

    適用場景：
    - 任務有明顯的「基礎技能」需要先學會
    - 高難度需要基礎技能作為支撐
    """

    def __init__(
        self,
        min_difficulty: int = 3,
        max_difficulty: int = 10,
        exponent: float = 2.0,
    ):
        super().__init__(min_difficulty, max_difficulty, name="exponential")
        self.exponent = exponent

    def get_difficulty(
        self,
        current_step: int,
        total_steps: int,
        metrics: Optional[CurriculumMetrics] = None,
    ) -> int:
        """指數增加難度（先慢後快）"""
        if total_steps <= 0:
            return self.min_difficulty

        progress = min(current_step / total_steps, 1.0)
        # 使用指數函數：progress^α，當 α > 1 時先慢後快
        scaled_progress = progress ** self.exponent
        difficulty_range = self.max_difficulty - self.min_difficulty
        difficulty = self.min_difficulty + difficulty_range * scaled_progress

        return int(round(difficulty))


class LogarithmicCurriculum(CurriculumStrategy):
    """對數課程策略：難度先快後慢增加

    D(t) = min_difficulty + (max_difficulty - min_difficulty) × log(1 + t/T × (e-1)) / log(e)

    簡化為：D(t) = min_difficulty + (max_difficulty - min_difficulty) × log(1 + t×k) / log(1 + T×k)

    特點：
    - 訓練早期難度快速增加
    - 訓練後期難度增加緩慢，讓 Agent 在高難度下穩定

    適用場景：
    - Agent 學習速度快，可以快速適應更高難度
    - 需要在高難度下花更多時間訓練
    """

    def __init__(
        self,
        min_difficulty: int = 3,
        max_difficulty: int = 10,
        steepness: float = 10.0,  # 控制曲線陡峭程度
    ):
        super().__init__(min_difficulty, max_difficulty, name="logarithmic")
        self.steepness = steepness

    def get_difficulty(
        self,
        current_step: int,
        total_steps: int,
        metrics: Optional[CurriculumMetrics] = None,
    ) -> int:
        """對數增加難度（先快後慢）"""
        if total_steps <= 0:
            return self.min_difficulty

        progress = min(current_step / total_steps, 1.0)
        # 使用對數函數：log(1 + progress × k) / log(1 + k)
        # 當 k > 0 時，曲線呈現先快後慢的特性
        k = self.steepness
        scaled_progress = math.log(1 + progress * k) / math.log(1 + k)
        difficulty_range = self.max_difficulty - self.min_difficulty
        difficulty = self.min_difficulty + difficulty_range * scaled_progress

        return int(round(difficulty))


class AdaptiveCurriculum(CurriculumStrategy):
    """自適應課程策略：根據 Agent 表現動態調整難度

    這是你目前使用的策略，基於 success_rate 和 collision_rate 調整。

    調整規則：
    - success_rate > increase_threshold 且 collision_rate < safe_collision_rate → 增加難度
    - success_rate < decrease_threshold 或 collision_rate > danger_collision_rate → 降低難度
    - 否則 → 保持當前難度

    學術參考：
    - OpenAI (2019) "Solving Rubik's Cube" 使用類似的 Automatic Domain Randomization
    - Florensa et al. (2017) 的 Reverse Curriculum 也是自適應的

    優點：
    - 自動適應 Agent 的學習速度
    - 避免過早進入高難度導致崩潰

    缺點：
    - 需要調整多個超參數
    - 難度可能頻繁波動
    """

    def __init__(
        self,
        min_difficulty: int = 3,
        max_difficulty: int = 10,
        # 難度增加條件
        increase_threshold: float = 0.80,     # success_rate > 80% 時增加難度
        safe_collision_rate: float = 0.20,    # collision_rate < 20% 時才增加難度
        # 難度降低條件
        decrease_threshold: float = 0.55,     # success_rate < 55% 時降低難度
        danger_collision_rate: float = 0.35,  # collision_rate > 35% 時降低難度
        # 調整參數
        adjustment_step: int = 1,             # 每次調整的難度變化量
        cooldown_steps: int = 1000,           # 調整後的冷卻期（步數）
    ):
        super().__init__(min_difficulty, max_difficulty, name="adaptive")

        # 閾值參數
        self.increase_threshold = increase_threshold
        self.safe_collision_rate = safe_collision_rate
        self.decrease_threshold = decrease_threshold
        self.danger_collision_rate = danger_collision_rate

        # 調整參數
        self.adjustment_step = adjustment_step
        self.cooldown_steps = cooldown_steps

        # 狀態
        self._last_adjustment_step = 0

    def get_difficulty(
        self,
        current_step: int,
        total_steps: int,
        metrics: Optional[CurriculumMetrics] = None,
    ) -> int:
        """根據表現自適應調整難度"""
        if metrics is None:
            return self._current_difficulty

        # 冷卻期檢查：避免頻繁調整
        if current_step - self._last_adjustment_step < self.cooldown_steps:
            return self._current_difficulty

        success_rate = metrics.success_rate
        collision_rate = metrics.collision_rate

        # 判斷是否需要調整
        should_increase = (
            success_rate > self.increase_threshold and
            collision_rate < self.safe_collision_rate
        )
        should_decrease = (
            success_rate < self.decrease_threshold or
            collision_rate > self.danger_collision_rate
        )

        # 執行調整
        if should_increase and self._current_difficulty < self.max_difficulty:
            self._current_difficulty += self.adjustment_step
            self._last_adjustment_step = current_step
        elif should_decrease and self._current_difficulty > self.min_difficulty:
            self._current_difficulty -= self.adjustment_step
            self._last_adjustment_step = current_step

        # 確保在範圍內
        self._current_difficulty = max(
            self.min_difficulty,
            min(self.max_difficulty, self._current_difficulty)
        )

        return self._current_difficulty

    def reset(self):
        """重置策略狀態"""
        super().reset()
        self._last_adjustment_step = 0


class SigmoidCurriculum(CurriculumStrategy):
    """Sigmoid 課程策略：S 形曲線，中期快速增加

    D(t) = min + (max - min) × sigmoid(k × (t/T - 0.5))

    特點：
    - 早期和晚期變化緩慢
    - 中期快速增加
    - 比線性更平滑的過渡

    適用場景：
    - 需要在中期快速提升難度
    - 早期和晚期需要穩定
    """

    def __init__(
        self,
        min_difficulty: int = 3,
        max_difficulty: int = 10,
        steepness: float = 10.0,  # 控制 S 形的陡峭程度
    ):
        super().__init__(min_difficulty, max_difficulty, name="sigmoid")
        self.steepness = steepness

    def get_difficulty(
        self,
        current_step: int,
        total_steps: int,
        metrics: Optional[CurriculumMetrics] = None,
    ) -> int:
        """Sigmoid 增加難度"""
        if total_steps <= 0:
            return self.min_difficulty

        progress = min(current_step / total_steps, 1.0)
        # Sigmoid: 1 / (1 + exp(-k × (x - 0.5)))
        # 將 [0, 1] 映射到 [-0.5, 0.5]，再通過 sigmoid
        x = self.steepness * (progress - 0.5)
        sigmoid_value = 1 / (1 + math.exp(-x))

        difficulty_range = self.max_difficulty - self.min_difficulty
        difficulty = self.min_difficulty + difficulty_range * sigmoid_value

        return int(round(difficulty))


# ============================================================================
# 工廠函數
# ============================================================================

CURRICULUM_STRATEGIES = {
    "no_curriculum": NoCurriculum,
    "linear": LinearCurriculum,
    "exponential": ExponentialCurriculum,
    "logarithmic": LogarithmicCurriculum,
    "adaptive": AdaptiveCurriculum,
    "sigmoid": SigmoidCurriculum,
}


def get_curriculum_strategy(
    name: str,
    min_difficulty: int = 3,
    max_difficulty: int = 10,
    **kwargs,
) -> CurriculumStrategy:
    """獲取指定的課程策略

    Args:
        name: 策略名稱，可選：
            - "no_curriculum": 無課程，固定最高難度
            - "linear": 線性增加
            - "exponential": 指數增加（先慢後快）
            - "logarithmic": 對數增加（先快後慢）
            - "adaptive": 自適應調整
            - "sigmoid": S 形曲線
        min_difficulty: 最小難度
        max_difficulty: 最大難度
        **kwargs: 策略特定的參數

    Returns:
        CurriculumStrategy 實例

    Example:
        >>> strategy = get_curriculum_strategy("adaptive", min_difficulty=3, max_difficulty=10)
        >>> difficulty = strategy.get_difficulty(10000, 500000, metrics)
    """
    if name not in CURRICULUM_STRATEGIES:
        available = list(CURRICULUM_STRATEGIES.keys())
        raise ValueError(f"Unknown curriculum strategy: {name}. Available: {available}")

    strategy_class = CURRICULUM_STRATEGIES[name]
    return strategy_class(min_difficulty=min_difficulty, max_difficulty=max_difficulty, **kwargs)


def list_available_strategies() -> List[str]:
    """列出所有可用的課程策略"""
    return list(CURRICULUM_STRATEGIES.keys())


# ============================================================================
# 比較工具
# ============================================================================

def compare_strategies_theoretical(
    total_steps: int = 500000,
    sample_points: int = 100,
    min_difficulty: int = 3,
    max_difficulty: int = 10,
) -> Dict[str, List[int]]:
    """理論比較：繪製不同策略的難度曲線（不需要實際訓練）

    Args:
        total_steps: 總訓練步數
        sample_points: 採樣點數
        min_difficulty: 最小難度
        max_difficulty: 最大難度

    Returns:
        字典，key 為策略名稱，value 為難度列表

    Example:
        >>> curves = compare_strategies_theoretical(500000, 100)
        >>> import matplotlib.pyplot as plt
        >>> for name, difficulties in curves.items():
        >>>     plt.plot(difficulties, label=name)
        >>> plt.legend()
        >>> plt.show()
    """
    results = {}
    steps = [int(i * total_steps / sample_points) for i in range(sample_points + 1)]

    for name in CURRICULUM_STRATEGIES:
        if name == "adaptive":
            # Adaptive 需要 metrics，這裡跳過或使用模擬
            continue

        strategy = get_curriculum_strategy(name, min_difficulty, max_difficulty)
        difficulties = [
            strategy.get_difficulty(step, total_steps)
            for step in steps
        ]
        results[name] = difficulties

    results["steps"] = steps
    return results


# ============================================================================
# 實驗配置
# ============================================================================

@dataclass
class CurriculumExperimentConfig:
    """課程學習實驗配置

    用於系統性比較實驗的配置類。
    """
    # 策略列表
    strategies: List[str] = field(default_factory=lambda: [
        "no_curriculum", "linear", "exponential", "logarithmic", "adaptive"
    ])

    # 難度範圍
    min_difficulty: int = 3
    max_difficulty: int = 10

    # 訓練參數
    total_steps: int = 500_000
    num_seeds: int = 5

    # 評估參數
    eval_difficulties: List[int] = field(default_factory=lambda: [3, 6, 10])
    eval_episodes_per_difficulty: int = 100

    # 指標閾值（用於計算樣本效率）
    success_thresholds: List[float] = field(default_factory=lambda: [0.5, 0.7, 0.8])

    def get_experiment_matrix(self) -> List[Dict]:
        """生成實驗矩陣

        Returns:
            實驗配置列表，每個配置包含策略名稱和隨機種子
        """
        experiments = []
        for strategy in self.strategies:
            for seed in range(self.num_seeds):
                experiments.append({
                    "strategy": strategy,
                    "seed": seed,
                    "min_difficulty": self.min_difficulty,
                    "max_difficulty": self.max_difficulty,
                    "total_steps": self.total_steps,
                })
        return experiments


# ============================================================================
# 測試代碼
# ============================================================================

if __name__ == "__main__":
    # 測試各種策略
    print("=== 課程策略測試 ===\n")

    total_steps = 100000
    test_points = [0, 25000, 50000, 75000, 100000]

    for name in list_available_strategies():
        if name == "adaptive":
            continue  # Adaptive 需要 metrics

        strategy = get_curriculum_strategy(name)
        print(f"{name}:")
        for step in test_points:
            difficulty = strategy.get_difficulty(step, total_steps)
            progress = step / total_steps * 100
            print(f"  Step {step:6d} ({progress:5.1f}%): difficulty = {difficulty}")
        print()

    # 測試 Adaptive 策略
    print("adaptive (模擬):")
    strategy = get_curriculum_strategy("adaptive")

    # 模擬訓練過程
    metrics = CurriculumMetrics()
    for step in test_points:
        # 模擬：隨著訓練進行，成功率提升
        metrics.success_rate = min(0.5 + step / total_steps * 0.4, 0.9)
        metrics.collision_rate = max(0.3 - step / total_steps * 0.2, 0.1)

        difficulty = strategy.get_difficulty(step, total_steps, metrics)
        print(f"  Step {step:6d}: success={metrics.success_rate:.2f}, "
              f"collision={metrics.collision_rate:.2f}, difficulty={difficulty}")

    print("\n=== 理論曲線比較 ===")
    curves = compare_strategies_theoretical(100000, 10)
    for name, difficulties in curves.items():
        if name != "steps":
            print(f"{name}: {difficulties}")
