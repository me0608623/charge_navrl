"""走廊 replay 的密度、slot、motion 配額與互動型態。

2026-07-27 裁決。放在環境端（`mdp/events/`）而非 `scripts/`：環境不能反向
import scripts，否則會重現先前 play/eval 的 sys.path 回歸。
**SA7/SA8 的比例表留在 config**，經 ExperimentConfig → trainer → event.params
傳入，這裡只提供不含階段語意的機制。

四個關注點：

1. **密度混合** —— 逐 env 抽 (static, dynamic)，含 random-phase systematic 配額，
   小批次也保證抽得到稀有的 5S+5D。
2. **slot 配置** —— per-env active mask，取代原本硬寫的 `slice(4, 4 + D)`
   （5 個靜態時 slot 4 會被動態覆蓋，且不會報錯）。
3. **motion family 配額** —— 橫向（lateral）需獨占整條橫排，走廊放不下 5 條，
   故上限 2；縱向與隨機走中央帶不佔排。
4. **互動型態** —— independent / crossing / side_by_side。
   side_by_side 需要「同 family 的兩人」，因此只在 family 配額真的出現重複時
   才可用；D=3 的 (1,1,1) 沒有重複，機率重新正規化為 75/25。
"""

from __future__ import annotations

import torch


# ── 容量 ────────────────────────────────────────────────────────────────
MAX_CORRIDOR_STATIC = 5
MAX_CORRIDOR_DYNAMIC = 5

# 動態 slot 起點 = 靜態上限，確保任何 (S, D) 都不互相覆蓋。
DYNAMIC_SLOT_BASE = MAX_CORRIDOR_STATIC

#: 縱向與 random_2d 走的中央帶 x 偏移。靜態在 x=±1.25，中央帶在 ±0.40，
#: 左右錯開 -> 這兩個 family 不需要獨占橫排，5 個動態才放得下。
CENTER_STRIP_X = 0.40

#: 中央帶的 y 格位與抖動。格距 0.90 m，抖動 ±0.08 m ->
#: 同側兩人最近 0.74 m > 2 * 0.35 m；離最壞情況的橫排（1.8 - 0.08）亦有 0.74 m。
CENTER_STRIP_BANDS = (-0.90, 0.0, 0.90)
CENTER_STRIP_BAND_JITTER = 0.08

#: random_2d 巡邏段兩端 y 的死區半寬 -> 最短段長 sqrt(0.4^2 + 1.0^2) ~ 1.08 m。
_RANDOM_2D_Y_DEADZONE = 0.5


def required_scheduler_slots() -> int:
    return MAX_CORRIDOR_STATIC + MAX_CORRIDOR_DYNAMIC


# ── motion family ───────────────────────────────────────────────────────
MOTION_LATERAL = 0
MOTION_LONGITUDINAL = 1
MOTION_RANDOM_2D = 2
_FAMILY_COUNT = 3

# 橫向排數限制推出的硬上限（見模組 docstring）。
MAX_PER_FAMILY = 2
INACTIVE_FAMILY = -1

_PURE_LATERAL_MAX_DYNAMIC = 2


# ── 互動型態 ────────────────────────────────────────────────────────────
INTERACTION_INDEPENDENT = 0
INTERACTION_CROSSING = 1
INTERACTION_SIDE_BY_SIDE = 2

INTERACTION_NAMES: dict[int, str] = {
    INTERACTION_INDEPENDENT: "independent",
    INTERACTION_CROSSING: "crossing",
    INTERACTION_SIDE_BY_SIDE: "side_by_side",
}

# 標稱權重。實際會依該 env 能不能做交叉／並排重新正規化。
INTERACTION_WEIGHTS: dict[int, float] = {
    INTERACTION_INDEPENDENT: 0.60,
    INTERACTION_CROSSING: 0.20,
    INTERACTION_SIDE_BY_SIDE: 0.20,
}

SIDE_BY_SIDE_SPACING_RANGE: tuple[float, float] = (0.8, 1.0)
SIDE_BY_SIDE_SPEED_TOLERANCE: float = 0.05

_MIN_DYNAMIC_FOR_CROSSING = 2


# ── 密度混合 ────────────────────────────────────────────────────────────
def validate_density_mix(mix) -> None:
    if not mix:
        raise ValueError("density mix must not be empty")
    total = sum(weight for _, weight in mix)
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"density mix weights must sum to 1, got {total}")
    for (static, dynamic), weight in mix:
        if weight < 0.0:
            raise ValueError("density mix weights must be non-negative")
        if not 0 <= static <= MAX_CORRIDOR_STATIC:
            raise ValueError(f"static count {static} exceeds {MAX_CORRIDOR_STATIC}")
        if not 0 <= dynamic <= MAX_CORRIDOR_DYNAMIC:
            raise ValueError(f"dynamic count {dynamic} exceeds {MAX_CORRIDOR_DYNAMIC}")


def sample_density_counts_systematic(
    count: int, *, mix, device, carry: dict | None = None
) -> torch.Tensor:
    """Random-phase systematic 取樣，回傳 [E,2]。

    純 multinomial 在 64 env 批次下，5% 的類別約有 4% 機率整批掛零 ——
    SA7 的 5S+5D 正是這種尾端類別，長期抽不到等於沒訓練到。

    ``carry`` 給定時改用**跨批次 residual 配額**：記住至今各組合已發出的
    次數，每批補上累積落後最多的組合。

    注意定位：random-phase systematic 本身**期望值已經無偏**（隔離實測
    batch=8、n=32000 誤差 0.18pp）。carry 的作用是**降低變異**，讓前 500 筆
    就能落在 ±2pp 內，不是修正偏差 —— 走廊注入每批只有數個 env，
    無記憶的話短期波動會蓋過門檻。
    """
    validate_density_mix(mix)
    count = int(count)
    if count <= 0:
        return torch.zeros(0, 2, dtype=torch.long, device=device)

    if carry is not None:
        return _sample_density_with_carry(count, mix=mix, device=device, carry=carry)

    combos = torch.tensor(
        [list(combo) for combo, _ in mix], dtype=torch.long, device=device
    )
    weights = torch.tensor(
        [weight for _, weight in mix], dtype=torch.float64, device=device
    )
    edges = torch.cumsum(weights, dim=0)
    edges[-1] = 1.0

    phase = torch.rand(1, dtype=torch.float64, device=device)
    u = (torch.arange(count, dtype=torch.float64, device=device) + phase) / count
    picks = torch.searchsorted(edges, u, right=True).clamp_(max=len(mix) - 1)
    return combos[picks[torch.randperm(count, device=device)]]


def _sample_density_with_carry(count: int, *, mix, device, carry: dict) -> torch.Tensor:
    """Largest-remainder allocation with memory across reset batches."""
    k = len(mix)
    emitted = carry.get("emitted")
    if emitted is None or emitted.numel() != k:
        emitted = torch.zeros(k, dtype=torch.float64, device=device)
        carry["emitted"] = emitted
    total = float(carry.get("total", 0.0))

    weights = torch.tensor(
        [w for _, w in mix], dtype=torch.float64, device=device
    )
    target = (total + count) * weights
    deficit = target - emitted
    # 先按整數部分配，再把餘額給落後最多的組合。
    base = deficit.clamp_min(0.0).floor()
    if float(base.sum()) > count:
        base = torch.zeros_like(base)
    remainder = int(count - int(base.sum()))
    if remainder > 0:
        order = torch.argsort(deficit - base, descending=True)
        base[order[:remainder]] += 1.0
    picks = torch.repeat_interleave(
        torch.arange(k, device=device), base.long()
    )[:count]
    if picks.numel() < count:                      # 保險：補滿
        extra = torch.multinomial(weights.float(), count - picks.numel(), True)
        picks = torch.cat([picks, extra])
    picks = picks[torch.randperm(count, device=device)]

    carry["emitted"] = emitted + torch.bincount(picks, minlength=k).to(emitted.dtype)
    carry["total"] = total + count

    combos = torch.tensor(
        [list(combo) for combo, _ in mix], dtype=torch.long, device=device
    )
    return combos[picks]


# ── slot mask ───────────────────────────────────────────────────────────
def active_masks_from_counts(counts: torch.Tensor):
    """[E,2] 數量 → (static_mask [E,5], dynamic_mask [E,5])，皆為前綴形式。"""
    if counts.ndim != 2 or counts.shape[-1] != 2:
        raise ValueError(f"expected counts [E,2], got {tuple(counts.shape)}")
    if counts.numel():
        if bool((counts < 0).any()):
            raise ValueError("obstacle counts must be non-negative")
        if int(counts[:, 0].max()) > MAX_CORRIDOR_STATIC:
            raise ValueError(f"static count exceeds cap {MAX_CORRIDOR_STATIC}")
        if int(counts[:, 1].max()) > MAX_CORRIDOR_DYNAMIC:
            raise ValueError(f"dynamic count exceeds cap {MAX_CORRIDOR_DYNAMIC}")
    device = counts.device
    static_idx = torch.arange(MAX_CORRIDOR_STATIC, device=device)[None, :]
    dynamic_idx = torch.arange(MAX_CORRIDOR_DYNAMIC, device=device)[None, :]
    return static_idx < counts[:, 0:1], dynamic_idx < counts[:, 1:2]


def validate_lateral_dynamic_count(mode: str, dynamic_count: int) -> None:
    """`lateral` 配 D>2 直接報錯，不得靜默改成 mixed。"""
    if str(mode).strip().lower() != "lateral":
        return
    if int(dynamic_count) > _PURE_LATERAL_MAX_DYNAMIC:
        raise ValueError(
            f"pure-lateral corridors support at most {_PURE_LATERAL_MAX_DYNAMIC} "
            f"dynamic obstacles, got {int(dynamic_count)}: five wall-to-wall "
            "lateral lanes do not fit in the 8.9 m usable corridor length. "
            "Use the deployment_mixed mode instead of silently promoting a "
            "pure-lateral regression scene."
        )


def static_layout_for_counts(
    static_counts: torch.Tensor,
    *,
    rows: tuple[float, ...],
    static_x: float,
    device,
) -> torch.Tensor:
    """逐 env 產生靜態障礙位置，回傳 ``[E, MAX_CORRIDOR_STATIC, 2]``。

    模板只有四列（rows），但上限是 5 個靜態。**第 5 個放不進新的一列**：
    同 x 側的兩個靜態需相距 ≥0.9 m，而剩下的 y 位置不是撞到動態橫排（±1.8），
    就是撞到機器人起點（−4.1）或目標（+4.1）。

    因此第 5 個與**隨機一列配對** —— 該列變成左右各一（x = ±static_x），
    中間留 ``2*static_x - 2*radius`` ≈ 1.8 m 通道，遠大於機器人所需，
    可解性有保障。這也是真實走廊常見的「兩側都有東西的門口」結構。

    低數量（≤4）時**隨機挑列、隨機挑左右**，不再固定取模板的前幾個位置。
    未啟用的 slot 填 0。
    """
    counts = static_counts.reshape(-1).to(device=device, dtype=torch.long)
    envs = counts.shape[0]
    if envs and int(counts.max()) > MAX_CORRIDOR_STATIC:
        raise ValueError(f"static count exceeds cap {MAX_CORRIDOR_STATIC}")
    layout = torch.zeros(envs, MAX_CORRIDOR_STATIC, 2, device=device)
    if envs == 0:
        return layout

    row_values = torch.tensor(rows, dtype=torch.float32, device=device)
    n_rows = row_values.numel()

    # 每個 env 獨立打散列順序 -> 低數量不會永遠落在同幾列。
    row_order = torch.argsort(torch.rand(envs, n_rows, device=device), dim=1)
    # 每列的左右側也隨機。
    side = torch.where(
        torch.rand(envs, n_rows, device=device) < 0.5, -1.0, 1.0
    )

    for env_idx in range(envs):
        n = int(counts[env_idx])
        if n == 0:
            continue
        picked = row_order[env_idx, : min(n, n_rows)]
        for slot, row_idx in enumerate(picked):
            layout[env_idx, slot, 0] = side[env_idx, row_idx] * static_x
            layout[env_idx, slot, 1] = row_values[row_idx]
        if n > n_rows:
            # 第 5 個：與已選的其中一列配對，補上對側。
            pair_at = int(picked[int(torch.randint(n_rows, (1,), device=device))])
            layout[env_idx, n_rows, 0] = -side[env_idx, pair_at] * static_x
            layout[env_idx, n_rows, 1] = row_values[pair_at]
    return layout


# ── 互動型態 ────────────────────────────────────────────────────────────
def sample_interaction_types(*, dynamic_counts: torch.Tensor, device) -> torch.Tensor:
    """逐 env 抽互動型態 —— **只看動態數量，不看 family**。

    順序很重要：互動型態是契約，family 是為了實現它而指派的。
    先抽 family 再問「這組 family 做得出什麼互動」，會讓實際分佈被 family
    的偶然組合決定，永遠湊不出承諾的 60/20/20，而且只能默默降級。

    * ``D < 2`` —— 只能 independent（互動至少要兩個人）。
    * ``D == 3`` —— **明定政策 75/25/0**（independent / crossing），不做並排。
      這是政策決定，不是舊 (1,1,1) 逐 env 配額的推論結果 —— 逐 env 配額已由
      整批累積平衡取代，不能再拿它當這條規則的理由。
    * 其餘（D=2、D>=4）—— **60/20/20**。
    """
    counts = dynamic_counts.reshape(-1).to(device=device, dtype=torch.long)
    envs = counts.shape[0]
    types = torch.full(
        (envs,), INTERACTION_INDEPENDENT, dtype=torch.long, device=device
    )
    if envs == 0:
        return types

    weights = torch.zeros(envs, 3, device=device)
    full = torch.tensor(
        [
            INTERACTION_WEIGHTS[INTERACTION_INDEPENDENT],
            INTERACTION_WEIGHTS[INTERACTION_CROSSING],
            INTERACTION_WEIGHTS[INTERACTION_SIDE_BY_SIDE],
        ],
        device=device,
    )
    interactive = counts >= _MIN_DYNAMIC_FOR_CROSSING
    weights[interactive] = full
    # D=3 的配額是三個 family 各一，沒有兩個同伴可以並排 -> 重新正規化為 75/25。
    three = counts == 3
    if bool(three.any()):
        renorm = full.clone()
        renorm[INTERACTION_SIDE_BY_SIDE] = 0.0
        weights[three] = renorm / renorm.sum()

    active = interactive.nonzero(as_tuple=False).flatten()
    if active.numel():
        types[active] = torch.multinomial(weights[active], 1).squeeze(1)
    return types


def interaction_fractions(
    types: torch.Tensor, dynamic_counts: torch.Tensor
) -> dict[str, float]:
    """分開回報整體比例與「有資格做三種型態」(D>=4) 的比例。

    包含 D=1／D=3 之後整體比例本來就不會是 60/20/20 —— 兩者分開記錄，
    才不會把正常的分佈誤報成配置失效。
    """
    counts = dynamic_counts.reshape(-1)
    eligible = counts >= 4
    out: dict[str, float] = {}
    total = max(int(types.numel()), 1)
    total_eligible = max(int(eligible.sum()), 1)
    for kind, name in INTERACTION_NAMES.items():
        out[f"interaction_fraction_overall/{name}"] = float(
            (types == kind).sum()
        ) / total
        out[f"interaction_fraction_eligible_d_ge_4/{name}"] = float(
            ((types == kind) & eligible).sum()
        ) / total_eligible
    out["interaction_eligible_d_ge_4_env_count"] = float(eligible.sum())
    return out


# ── 分項指標 ────────────────────────────────────────────────────────────
class CorridorDensityStats:
    """按 (S,D) 組合與互動型態分別累計走廊 replay 的結果。

    裁決要求記錄「每種 count pair 的實際比例、SR、CR、TO、停止率與障礙清空後
    恢復前進率」，以及 independent/crossing/side_by_side 各自的 SR/CR/TO。

    停讓與恢復是 5S+5D 的**正確目標**：有安全通道就通過、暫時被堵住就停下等待、
    通道重新打開就恢復前進，任何情況都不得為了 progress 硬撞。因此
    「停下」本身不是失敗，「停下後不再前進」才是。
    """

    def __init__(self) -> None:
        self._rows: dict[tuple, dict[str, float]] = {}

    @staticmethod
    def _blank() -> dict[str, float]:
        return {
            "episodes": 0.0, "success": 0.0, "collision": 0.0, "timeout": 0.0,
            "stopped": 0.0, "resumed_after_clear": 0.0, "clear_events": 0.0,
        }

    def add(
        self,
        *,
        static_counts: torch.Tensor,
        dynamic_counts: torch.Tensor,
        interaction_types: torch.Tensor,
        success: torch.Tensor,
        collision: torch.Tensor,
        timeout: torch.Tensor,
        stopped: torch.Tensor,
        resumed_after_clear: torch.Tensor,
        had_clear_event: torch.Tensor,
    ) -> None:
        """累計一批已結束的回合。所有張量皆為 [N]，一個元素一個回合。"""
        s = static_counts.reshape(-1).tolist()
        d = dynamic_counts.reshape(-1).tolist()
        it = interaction_types.reshape(-1).tolist()
        fields = {
            "success": success, "collision": collision, "timeout": timeout,
            "stopped": stopped, "resumed_after_clear": resumed_after_clear,
            "clear_events": had_clear_event,
        }
        values = {k: v.reshape(-1).float().tolist() for k, v in fields.items()}
        for i in range(len(s)):
            key = (int(s[i]), int(d[i]), int(it[i]))
            row = self._rows.setdefault(key, self._blank())
            row["episodes"] += 1.0
            for name, seq in values.items():
                row[name] += float(seq[i])

    def to_dict(self) -> dict[str, float]:
        """展平成 wandb 可吃的扁平鍵值。"""
        total = sum(r["episodes"] for r in self._rows.values()) or 1.0
        out: dict[str, float] = {}

        def emit(prefix: str, rows: list[dict[str, float]]) -> None:
            n = sum(r["episodes"] for r in rows)
            if n <= 0:
                return
            out[f"{prefix}/episodes"] = n
            out[f"{prefix}/fraction"] = n / total
            for name in ("success", "collision", "timeout", "stopped"):
                out[f"{prefix}/{name}_rate"] = sum(r[name] for r in rows) / n
            clears = sum(r["clear_events"] for r in rows)
            # 恢復率的分母是「真的發生過通道清空」的回合，否則沒有可恢復的機會。
            out[f"{prefix}/resume_after_clear_rate"] = (
                sum(r["resumed_after_clear"] for r in rows) / clears
                if clears > 0 else float("nan")
            )

        by_counts: dict[tuple, list] = {}
        by_interaction: dict[int, list] = {}
        for (s, d, it), row in self._rows.items():
            by_counts.setdefault((s, d), []).append(row)
            by_interaction.setdefault(it, []).append(row)
        for (s, d), rows in sorted(by_counts.items()):
            emit(f"corridor_density/{s}S{d}D", rows)
        for it, rows in sorted(by_interaction.items()):
            emit(f"corridor_interaction/{INTERACTION_NAMES[it]}", rows)
        return out


#: 最後一次 allocator 的輸入。幾何層在後面才拋出，那時已拿不到這些輸入，
#: 但重現必須要有它們。
_LAST_ALLOCATION: dict = {}


def dump_corridor_allocation_state(
    families,
    *,
    reason: str,
    env_idx: int | None = None,
    counts=None,
    interaction_types=None,
    pairs=None,
    family_debt=None,
    global_env_ids=None,
    extra: dict | None = None,
) -> None:
    """Persist the full allocator state so a failure can be replayed exactly.

    只 dump `families` 是不夠的 —— 重現需要輸入（counts、interaction_types）、
    配對（pairs）、跨批次狀態（family_debt）與 **global env ID**（否則無法
    對回是哪些環境）。
    """
    import json
    import os

    path = os.environ.get(
        "CORRIDOR_OVERFLOW_DUMP", "/tmp/corridor_lateral_overflow.json"
    )

    def as_list(x):
        return None if x is None else (
            x.tolist() if hasattr(x, "tolist") else x
        )

    try:
        payload = {
            "reason": reason,
            "env_idx": None if env_idx is None else int(env_idx),
            "families": as_list(families),
            "lateral_per_env": (families == MOTION_LATERAL).sum(dim=1).tolist(),
            "counts": as_list(counts),
            "interaction_types": as_list(interaction_types),
            "pairs": as_list(pairs),
            "global_env_ids": as_list(global_env_ids),
            "family_debt": None if family_debt is None else {
                "emitted": as_list(family_debt.get("emitted")),
                "total": float(family_debt.get("total", 0.0)),
            },
        }
        if extra:
            payload.update(extra)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"[CORRIDOR-OVERFLOW] state dumped to {path}", flush=True)
    except Exception as exc:                     # dump 失敗不得掩蓋原始錯誤
        print(f"[CORRIDOR-OVERFLOW] dump failed: {exc}", flush=True)


def dynamic_layout_for_families(
    families: torch.Tensor,
    *,
    lateral_lanes: tuple[float, ...],
    center_strip_x: float,
    lateral_x_limit: float,
    device,
) -> torch.Tensor:
    """依 family 指派產生動態障礙的起始位置，回傳 ``[E, MAX_CORRIDOR_DYNAMIC, 2]``。

    三種 family 佔用互不衝突的空間，這正是橫向上限 2 之後 5 個動態放得下的原因：

    * **lateral** —— 需獨占一條橫排（``lateral_lanes``，最多 2 條），
      起始 x 在 ``±lateral_x_limit`` 內隨機。
    * **longitudinal** —— 走中央帶 ``x = ±center_strip_x``，沿 y 縱貫，不佔排。
    * **random_2d** —— 同樣以中央帶為基準，不佔排。

    靜態在 ``x = ±1.25``，與中央帶（±0.40）左右錯開，所以縱向與隨機不需要
    專屬橫排。未啟用的 slot 填 0。
    """
    envs = families.shape[0]
    layout = torch.zeros(envs, MAX_CORRIDOR_DYNAMIC, 2, device=device)
    if envs == 0:
        return layout

    lanes = torch.tensor(lateral_lanes, dtype=torch.float32, device=device)
    n_lanes = lanes.numel()
    # 每個 env 都獨立打散橫排。否則 crossing 固定使用 slot 0/1 時，
    # 第一個 lateral 永遠落在 lateral_lanes[0]，交點 y 每回合都相同。
    lane_order = torch.argsort(torch.rand(envs, n_lanes, device=device), dim=1)

    # 中央帶共有 2 側 x 3 個 y band。逐 env 打散後再依序取用，既保留
    # 格位的最小間距保證，也避免第一個 longitudinal 永遠出現在
    # (-center_strip_x, CENTER_STRIP_BANDS[0])。
    center_cell_count = 2 * len(CENTER_STRIP_BANDS)
    center_cell_order = torch.argsort(
        torch.rand(envs, center_cell_count, device=device), dim=1
    )

    for env_idx in range(envs):
        used_lane = 0
        # 中央帶的格位計數必須**跨 family 共用**：兩個 family 各自從 0 起算的話
        # 會同時認領同一格（縱向第一人與 random_2d 第一人都落在 x=-0.40）。
        used_center_cells = 0
        for slot in range(MAX_CORRIDOR_DYNAMIC):
            family = int(families[env_idx, slot])
            if family < 0:
                continue
            if family == MOTION_LATERAL:
                if used_lane >= n_lanes:
                    # 2026-07-27 SA7 曾在 iter 10 之後因此炸掉整個訓練，而事後
                    # 用四種方式都重現不了觸發條件。把當下的完整狀態 dump 出來，
                    # 萬一再發生就有重現素材，不必再靠猜。
                    dump_corridor_allocation_state(
                        families,
                        reason="lateral_lane_overflow_at_geometry",
                        env_idx=env_idx,
                        counts=_LAST_ALLOCATION.get("counts"),
                        interaction_types=_LAST_ALLOCATION.get(
                            "interaction_types"
                        ),
                        pairs=_LAST_ALLOCATION.get("pairs"),
                        family_debt=_LAST_ALLOCATION.get("family_debt"),
                        global_env_ids=_LAST_ALLOCATION.get("global_env_ids"),
                        extra={"n_lanes": int(n_lanes)},
                    )
                    raise ValueError(
                        f"lateral obstacles exceed the {n_lanes} available lanes; "
                        f"env {env_idx} families={families[env_idx].tolist()}; "
                        "state dumped for reproduction"
                    )
                layout[env_idx, slot, 0] = (
                    torch.rand(1, device=device) * 2.0 - 1.0
                ) * lateral_x_limit
                lane_idx = int(lane_order[env_idx, used_lane])
                layout[env_idx, slot, 1] = lanes[lane_idx]
                used_lane += 1
            else:
                # 中央帶用**固定格位**，不是自由亂數。自由亂數會讓中央帶的人
                # 落在橫排 y=±1.8 附近，而橫向的人 x 掃遍整條走廊 ——
                # 實測 53% 的 draw 一出生就疊在一起。格位保證任兩人
                # 中心距 > 2 * 0.35 m，且離最壞情況的橫排仍有 0.74 m。
                cell = int(center_cell_order[env_idx, used_center_cells])
                layout[env_idx, slot, 0] = (
                    -1.0 if cell % 2 == 0 else 1.0
                ) * center_strip_x
                band = CENTER_STRIP_BANDS[(cell // 2) % len(CENTER_STRIP_BANDS)]
                jitter = (
                    torch.rand(1, device=device) * 2.0 - 1.0
                ) * CENTER_STRIP_BAND_JITTER
                layout[env_idx, slot, 1] = band + jitter
                used_center_cells += 1
    return layout


def dynamic_waypoints_for_families(
    families: torch.Tensor,
    starts: torch.Tensor,
    *,
    lateral_x_limit: float,
    longitudinal_y_limit: float,
    center_strip_x: float,
    device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """依 family 產生兩點巡邏路徑，回傳 ``([E,D,2,2], [E,D])``。

    幾何模組的 `sample_dynamic_trajectories` **body 寫死 slot 0/1**
    （`longitudinal_starts[:, 0, ...]`、`lateral_targets = rand(count, 2)`），
    只泛化形狀檢查並不能讓它吃 5 個動態 —— 多出來的 slot 會靜默不被設定，
    變成速度為 0 的隱形障礙。所以高密度路徑改由這裡產生路徑。

    每個 family 的路徑軸與 `dynamic_layout_for_families` 的佔位軸一致：

    * **lateral** —— 沿 x 橫掃，y 固定在該人的橫排（不換排）。
    * **longitudinal** —— 沿 y 縱貫，x 固定在該人的中央帶側（不越側）。
    * **random_2d** —— 中央帶內的對角線，兩端 x 分居 0 兩側、y 各自隨機。

    未啟用的 slot（family < 0）填 0，target index 填 0。
    """
    if families.shape != starts.shape[:2]:
        raise ValueError(
            f"families {tuple(families.shape)} must match starts "
            f"{tuple(starts.shape[:2])}"
        )
    envs, slots = families.shape
    waypoints = torch.zeros(envs, slots, 2, 2, device=device)
    targets = torch.zeros(envs, slots, dtype=torch.long, device=device)
    if envs == 0:
        return waypoints, targets

    active = families >= 0
    lateral = families == MOTION_LATERAL
    longitudinal = families == MOTION_LONGITUDINAL
    random_2d = families == MOTION_RANDOM_2D

    start_x = starts[..., 0]
    start_y = starts[..., 1]

    # lateral：x 兩端固定，y 保持在自己的橫排。
    lat = torch.zeros_like(waypoints)
    lat[..., 0, 0] = -float(lateral_x_limit)
    lat[..., 1, 0] = float(lateral_x_limit)
    lat[..., 0, 1] = start_y
    lat[..., 1, 1] = start_y

    # longitudinal：y 兩端固定，x 保持在自己的中央帶側。
    lon = torch.zeros_like(waypoints)
    lon[..., 0, 1] = -float(longitudinal_y_limit)
    lon[..., 1, 1] = float(longitudinal_y_limit)
    lon[..., 0, 0] = start_x
    lon[..., 1, 0] = start_x

    # random_2d：中央帶內的對角線，兩端分居 x=0 兩側，避免退化成純縱走。
    strip = abs(float(center_strip_x))
    rnd = torch.zeros_like(waypoints)
    magnitude = torch.empty(envs, slots, 2, device=device).uniform_(
        0.5 * strip, strip
    )
    rnd[..., 0, 0] = -magnitude[..., 0]
    rnd[..., 1, 0] = magnitude[..., 1]
    flip = torch.rand(envs, slots, device=device) < 0.5
    rnd[..., 0, 0] = torch.where(flip, -rnd[..., 0, 0], rnd[..., 0, 0])
    rnd[..., 1, 0] = torch.where(flip, -rnd[..., 1, 0], rnd[..., 1, 0])
    # 兩端的 y 必須**夾住自己的起點**，各留至少 ``_RANDOM_2D_Y_DEADZONE``。
    # 原本以 0 為中心抽兩端，起點卻在中央帶的 ±0.98 —— 起點會落在自己的巡邏
    # 段之外（實測有 y 起點 -0.835、段卻是 [-0.807, 3.175] 的案例），
    # 那個人一出生就在段外，且永遠到不了走廊另一側。
    dead = _RANDOM_2D_Y_DEADZONE
    limit = float(longitudinal_y_limit)
    base_y = start_y.clamp(-(limit - dead - 0.1), limit - dead - 0.1)
    low = base_y - dead - torch.rand(envs, slots, device=device) * (
        (base_y - dead + limit).clamp_min(0.0)
    )
    high = base_y + dead + torch.rand(envs, slots, device=device) * (
        (limit - base_y - dead).clamp_min(0.0)
    )
    swap_y = torch.rand(envs, slots, device=device) < 0.5
    rnd[..., 0, 1] = torch.where(swap_y, high, low)
    rnd[..., 1, 1] = torch.where(swap_y, low, high)

    pick = lateral[..., None, None]
    waypoints = torch.where(pick, lat, waypoints)
    waypoints = torch.where(longitudinal[..., None, None], lon, waypoints)
    waypoints = torch.where(random_2d[..., None, None], rnd, waypoints)
    waypoints = torch.where(active[..., None, None], waypoints, 0.0)

    targets = torch.where(
        active,
        (torch.rand(envs, slots, device=device) < 0.5).long(),
        torch.zeros_like(targets),
    )
    return waypoints, targets
SIDE_BY_SIDE_MAX_SPEED_DELTA = 0.05


def center_strip_member_count(families: torch.Tensor) -> torch.Tensor:
    """每個 env 的中央帶成員數（縱向 + random_2d）。

    並排需要**兩個中央帶成員**，不要求同 family —— 並排本來就會把兩人的
    路徑改寫成共用的直線同行，family 標籤只決定原本的路徑。橫向的兩人各自
    獨占一條橫排（y = ±1.8），相距 3.6 m，那不是並排。
    """
    return (
        (families == MOTION_LONGITUDINAL) | (families == MOTION_RANDOM_2D)
    ).sum(dim=1)


def _endpoint_towards(
    endpoints: torch.Tensor, position: torch.Tensor, target: torch.Tensor
) -> int:
    """Pick the patrol endpoint that lies on the way to ``target``."""
    direction = float(target) - float(position)
    if abs(direction) < 1e-6:
        # 已經在交點上：挑離自己較遠的端點，先離開再折返仍會通過交點。
        return int(torch.argmax((endpoints - float(position)).abs()))
    signed = (endpoints - float(position)) * direction
    if bool((signed > 0).any()):
        return int(torch.argmax(signed))
    return int(torch.argmax((endpoints - float(position)).abs()))


def apply_interaction_geometry(
    interaction_types: torch.Tensor,
    pairs: torch.Tensor,
    families: torch.Tensor,
    dynamic: torch.Tensor,
    waypoints: torch.Tensor,
    targets: torch.Tensor,
    speeds: torch.Tensor,
    *,
    longitudinal_y_limit: float,
    center_strip_x: float,
    device,
):
    """依**固定的配對**擺出互動幾何，回傳改寫後的佈局與 realized 遮罩。

    配對 slot 由 `assign_families_and_pairs` 事先決定且 family 已正確，
    所以這裡不再去猜「哪兩個人可以配」—— 猜的版本會挑到 random_2d，
    而 random_2d 之後會被 wander 重抽 heading，把幾何蓋掉。
    """
    envs = interaction_types.shape[0]
    dynamic = dynamic.clone()
    waypoints = waypoints.clone()
    targets = targets.clone()
    speeds = speeds.clone()
    realized = interaction_types == INTERACTION_INDEPENDENT

    limit = float(longitudinal_y_limit)
    strip_x = abs(float(center_strip_x))
    low, high = SIDE_BY_SIDE_SPACING_RANGE

    for env_idx in range(envs):
        kind = int(interaction_types[env_idx])
        if kind == INTERACTION_INDEPENDENT:
            continue
        a = int(pairs[env_idx, 0])
        b = int(pairs[env_idx, 1])
        if a < 0 or b < 0:
            continue

        if kind == INTERACTION_CROSSING:
            # a = lateral（沿 x 掃）、b = longitudinal（沿 y 走）。
            # 交點 = (b.x, a.y)；兩人的第一個目標都必須指向它。
            lane_y = float(dynamic[env_idx, a, 1])
            strip_pos_x = float(dynamic[env_idx, b, 0])
            pos_y = float(dynamic[env_idx, b, 1])
            step = 1.0 if lane_y >= pos_y else -1.0
            # longitudinal 的巡邏段必須真的涵蓋橫排的 y，否則永遠走不到交點。
            waypoints[env_idx, b, 0, 0] = strip_pos_x
            waypoints[env_idx, b, 1, 0] = strip_pos_x
            waypoints[env_idx, b, 0, 1] = max(min(pos_y - step * 1.0, limit), -limit)
            waypoints[env_idx, b, 1, 1] = max(min(lane_y + step * 0.6, limit), -limit)
            targets[env_idx, b] = 1
            targets[env_idx, a] = _endpoint_towards(
                waypoints[env_idx, a, :, 0], dynamic[env_idx, a, 0], strip_pos_x
            )
            realized[env_idx] = True
            continue

        # side_by_side：兩個 longitudinal，同一個 y 帶、x 分居 ±strip_x。
        band = float(dynamic[env_idx, a, 1])
        band = max(min(band, limit - 1.0), -(limit - 1.0))
        spacing = 0.5 * (low + high)
        half = 0.5 * spacing
        dynamic[env_idx, a, 0] = -half
        dynamic[env_idx, b, 0] = half
        dynamic[env_idx, a, 1] = band
        dynamic[env_idx, b, 1] = band
        for slot, x in ((a, -half), (b, half)):
            waypoints[env_idx, slot, 0, 0] = x
            waypoints[env_idx, slot, 1, 0] = x
            waypoints[env_idx, slot, 0, 1] = -limit
            waypoints[env_idx, slot, 1, 1] = limit
        # 完全同步：同一個 waypoint index、同一條路徑長度、**完全相同的速度**。
        # 契約門檻是速差 <= 0.05 m/s；取 0 仍然滿足契約，而且是唯一能讓兩人
        # 永久維持隊形的選擇 —— 只要有速差，端點折返時必然失步（實測
        # opposite_phase 278 次、間距超標 242 次）。暫停由 patrol_no_pause 關掉。
        targets[env_idx, b] = targets[env_idx, a]
        speeds[env_idx, b] = speeds[env_idx, a]
        realized[env_idx] = True

    return dynamic, waypoints, targets, speeds, realized


def crossing_axis_and_side(
    interaction_types: torch.Tensor,
    pairs: torch.Tensor,
    families: torch.Tensor,
    dynamic: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """回傳 crossing 的固定交點 ``[E,2]`` 與兩人的起始側別符號 ``[E,2]``。

    交叉是**事件**不是狀態：兩人各自穿過交點一次就算完成，之後不該再要求
    他們朝交點走（穿過去、或在 waypoint 暫停，都會自然不符合）。
    因此保存固定交點與起始側別，之後只看側別有沒有翻轉。

    側別定義在各自的行進軸上：lateral 沿 x、longitudinal 沿 y。
    未配對或非 crossing 的 env 填 0。
    """
    envs = interaction_types.shape[0]
    point = torch.zeros(envs, 2, device=dynamic.device)
    side = torch.zeros(envs, 2, device=dynamic.device)
    for env_idx in range(envs):
        if int(interaction_types[env_idx]) != INTERACTION_CROSSING:
            continue
        a, b = int(pairs[env_idx, 0]), int(pairs[env_idx, 1])
        if a < 0 or b < 0:
            continue
        fa = int(families[env_idx, a])
        lat, strip = (a, b) if fa == MOTION_LATERAL else (b, a)
        cross_x = float(dynamic[env_idx, strip, 0])
        cross_y = float(dynamic[env_idx, lat, 1])
        point[env_idx, 0] = cross_x
        point[env_idx, 1] = cross_y
        # index 0 一律記 lateral（沿 x），index 1 記 longitudinal（沿 y）。
        side[env_idx, 0] = torch.sign(
            cross_x - dynamic[env_idx, lat, 0]
        )
        side[env_idx, 1] = torch.sign(
            cross_y - dynamic[env_idx, strip, 1]
        )
    return point, side


def crossing_progress(
    interaction_types: torch.Tensor,
    pairs: torch.Tensor,
    families: torch.Tensor,
    positions: torch.Tensor,
    cross_point: torch.Tensor,
    start_side: torch.Tensor,
    already: torch.Tensor,
) -> torch.Tensor:
    """更新「各自穿過交點」的旗標 ``[E,2]``，回傳新的旗標。

    側別由起始符號翻轉（或歸零）判定 —— 一旦翻轉就永久記為已穿越，
    不會因為之後走遠又被清掉。
    """
    updated = already.clone()
    for env_idx in range(interaction_types.shape[0]):
        if int(interaction_types[env_idx]) != INTERACTION_CROSSING:
            continue
        a, b = int(pairs[env_idx, 0]), int(pairs[env_idx, 1])
        if a < 0 or b < 0:
            continue
        fa = int(families[env_idx, a])
        lat, strip = (a, b) if fa == MOTION_LATERAL else (b, a)
        now_lat = torch.sign(cross_point[env_idx, 0] - positions[env_idx, lat, 0])
        now_strip = torch.sign(cross_point[env_idx, 1] - positions[env_idx, strip, 1])
        if float(start_side[env_idx, 0]) != 0.0 and float(now_lat) != float(
            start_side[env_idx, 0]
        ):
            updated[env_idx, 0] = True
        if float(start_side[env_idx, 1]) != 0.0 and float(now_strip) != float(
            start_side[env_idx, 1]
        ):
            updated[env_idx, 1] = True
    return updated


def validate_pair_families(
    interaction_types: torch.Tensor,
    pairs: torch.Tensor,
    families: torch.Tensor,
) -> list[str]:
    """硬閘：配對 slot 的 family 必須完全符合契約。回傳違規描述清單。"""
    problems: list[str] = []
    for env_idx in range(interaction_types.shape[0]):
        kind = int(interaction_types[env_idx])
        if kind == INTERACTION_INDEPENDENT:
            continue
        a, b = int(pairs[env_idx, 0]), int(pairs[env_idx, 1])
        if a < 0 or b < 0:
            problems.append(f"env {env_idx}: {INTERACTION_NAMES[kind]} 沒有配對 slot")
            continue
        fa, fb = int(families[env_idx, a]), int(families[env_idx, b])
        if kind == INTERACTION_CROSSING:
            if {fa, fb} != {MOTION_LATERAL, MOTION_LONGITUDINAL}:
                problems.append(
                    f"env {env_idx}: crossing 配對必須是 lateral+longitudinal，得到 "
                    f"({fa},{fb})"
                )
        elif (fa, fb) != (MOTION_LONGITUDINAL, MOTION_LONGITUDINAL):
            problems.append(
                f"env {env_idx}: side_by_side 配對必須是 longitudinal+longitudinal，"
                f"得到 ({fa},{fb})"
            )
    return problems


def audit_installed_pairs(
    *,
    behavior_type: torch.Tensor,
    positions: torch.Tensor,
    velocities: torch.Tensor,
    interaction_types: torch.Tensor,
    pairs: torch.Tensor,
    families: torch.Tensor,
    random_walk_id: int,
) -> dict[str, int]:
    """從**安裝後的 scheduler 狀態**複驗互動，而不是安裝前的暫存張量。

    量安裝前的幾何是假陽性的來源：wander 在幾何之後才覆寫 heading，
    安裝前看起來完美的 crossing，安裝後可能已經變成獨立亂走。

    所有輸入都是「已切到走廊動態 slot」的視圖，形狀 ``[E, D, ...]``。
    """
    out = {
        "crossing_ok": 0, "crossing_n": 0,
        "side_ok": 0, "side_n": 0,
        "paired_is_random_walk": 0,
        # 逐子條件拆解 —— 低通過率若集中在「已通過交點」或「折返相位差」，
        # 那是度量的結構性產物；若集中在 family/行為型態，才是真的被改寫。
        "x_wrong_family": 0, "x_zero_speed": 0, "x_past_crossing": 0,
        "s_wrong_family": 0, "s_spacing": 0, "s_speed_delta": 0,
        "s_opposite_phase": 0,
    }
    low, high = SIDE_BY_SIDE_SPACING_RANGE

    for env_idx in range(interaction_types.shape[0]):
        kind = int(interaction_types[env_idx])
        if kind == INTERACTION_INDEPENDENT:
            continue
        a, b = int(pairs[env_idx, 0]), int(pairs[env_idx, 1])
        if a < 0 or b < 0:
            continue
        # 硬閘：配對 slot 不得是 RANDOM_WALK。
        for slot in (a, b):
            if int(behavior_type[env_idx, slot]) == random_walk_id:
                out["paired_is_random_walk"] += 1

        pa, pb = positions[env_idx, a], positions[env_idx, b]
        va, vb = velocities[env_idx, a], velocities[env_idx, b]

        if kind == INTERACTION_CROSSING:
            out["crossing_n"] += 1
            fa, fb = int(families[env_idx, a]), int(families[env_idx, b])
            if {fa, fb} != {MOTION_LATERAL, MOTION_LONGITUDINAL}:
                out["x_wrong_family"] += 1
                continue
            lat, strip = (a, b) if fa == MOTION_LATERAL else (b, a)
            cross = torch.stack(
                [positions[env_idx, strip, 0], positions[env_idx, lat, 1]]
            )
            ok = True
            for slot in (lat, strip):
                to_cross = cross - positions[env_idx, slot]
                heading_ok = bool(
                    (to_cross * velocities[env_idx, slot]).sum() > 0
                )
                moving = bool(
                    torch.linalg.vector_norm(velocities[env_idx, slot]) > 1e-6
                )
                if not moving:
                    out["x_zero_speed"] += 1
                elif not heading_ok:
                    # 速度非零但不指向交點 = 已經越過交點（互動發生過）。
                    out["x_past_crossing"] += 1
                ok &= heading_ok and moving
            out["crossing_ok"] += int(ok)
        else:
            out["side_n"] += 1
            if (
                int(families[env_idx, a]) != MOTION_LONGITUDINAL
                or int(families[env_idx, b]) != MOTION_LONGITUDINAL
            ):
                out["s_wrong_family"] += 1
                continue
            spacing = float(torch.linalg.vector_norm(pa - pb))
            speed_a = float(torch.linalg.vector_norm(va))
            speed_b = float(torch.linalg.vector_norm(vb))
            same_way = bool((va * vb).sum() > 0)
            spacing_ok = low - 1e-6 <= spacing <= high + 1e-6
            delta_ok = (
                abs(speed_a - speed_b) <= SIDE_BY_SIDE_MAX_SPEED_DELTA + 1e-6
            )
            if not spacing_ok:
                out["s_spacing"] += 1
            if not delta_ok:
                out["s_speed_delta"] += 1
            if not same_way:
                # 允許的 <=0.05 m/s 速差會讓兩人逐漸相位偏移，端點折返時
                # 必然有一段反向 —— 這是契約本身的結果，不是被改寫。
                out["s_opposite_phase"] += 1
            out["side_ok"] += int(
                spacing_ok and delta_ok and same_way and speed_a > 1e-6
            )
    return out


# ── 分項指標 ────────────────────────────────────────────────────────────


#: 並排的橫向間距上下限（裁決：0.8–1.0 m）與速度差上限（≤0.05 m/s）。


#: 未配對的哨兵值。
NO_PAIR = -1


def assign_families_and_pairs(
    dynamic_counts: torch.Tensor,
    interaction_types: torch.Tensor,
    *,
    device,
    family_debt: dict | None = None,
    global_env_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """指派 family 並固定互動配對，回傳 ``(families [E,5], pairs [E,2])``。

    2026-07-27 裁決的第三種做法 —— 不在「保留錯標」與「改標籤破壞配額」之間
    二選一，而是讓**被配對的 slot 一開始就是正確的 family**：

    * ``crossing`` -> 一個 ``lateral`` + 一個 ``longitudinal``，兩者都維持 PATROL。
    * ``side_by_side`` -> 兩個 ``longitudinal``，平行同行。
    * ``random_2d`` **永遠不會被配對** —— wander 會重抽獨立 heading，把已經
      擺好的互動幾何整個蓋掉，讓 audit 量到「安裝前的幾何」而報出假陽性。

    因此 family 欄位永遠等於該 slot 的實際運動，不需要事後改標籤。

    配額改成**整批累積平衡**：互動強制吃掉的 longitudinal 額度，由其他未配對
    的 slot 補回 lateral / random_2d，所以整批三種 family 的總數仍然接近均等，
    而不是靠逐 env 的固定配額（那與互動需求會直接衝突）。
    """
    counts = dynamic_counts.reshape(-1).to(device=device, dtype=torch.long)
    envs = counts.shape[0]
    families = torch.full(
        (envs, MAX_CORRIDOR_DYNAMIC), INACTIVE_FAMILY, dtype=torch.long, device=device
    )
    pairs = torch.full((envs, 2), NO_PAIR, dtype=torch.long, device=device)
    if envs == 0:
        return families, pairs

    # ── 1. 互動強制的配對 slot ────────────────────────────────────────
    crossing = interaction_types == INTERACTION_CROSSING
    side = interaction_types == INTERACTION_SIDE_BY_SIDE
    paired = (crossing | side) & (counts >= _MIN_DYNAMIC_FOR_CROSSING)

    families[crossing & paired, 0] = MOTION_LATERAL
    families[crossing & paired, 1] = MOTION_LONGITUDINAL
    families[side & paired, 0] = MOTION_LONGITUDINAL
    families[side & paired, 1] = MOTION_LONGITUDINAL
    pairs[paired, 0] = 0
    pairs[paired, 1] = 1

    # ── 2. 其餘 slot：整批累積平衡 ────────────────────────────────────
    slot_index = torch.arange(MAX_CORRIDOR_DYNAMIC, device=device)[None, :]
    active = slot_index < counts[:, None]
    free = active & (families == INACTIVE_FAMILY)
    free_idx = free.nonzero(as_tuple=False)
    n_free = free_idx.shape[0]

    # 帳本狀態必須在**任何** return 之前就備妥。
    total = int(active.sum())
    emitted = torch.zeros(_FAMILY_COUNT, dtype=torch.float64, device=device)
    seen = 0.0
    if family_debt is not None:
        emitted = family_debt.get(
            "emitted",
            torch.zeros(_FAMILY_COUNT, dtype=torch.float64, device=device),
        )
        seen = float(family_debt.get("total", 0.0))

    if n_free == 0:
        # 沒有自由 slot 也**必須**記帳。這一批全是互動強制的 slot（多為
        # longitudinal），提早 return 會讓它們永遠不進帳本 —— 帳本因此顯示
        # 三類均衡，補償永遠不啟動，實際卻持續偏向 longitudinal。
        # 實測 500 批有 74 批屬於此類，正是 +4.6pp 的來源。
        _record_family_debt(family_debt, emitted, seen, families, active, total)
        return families, pairs

    # 跨批次 family debt：配對強制吃掉 longitudinal 額度，只在當批補償的話
    # 補不回來（實測 longitudinal 長期偏高 5pp）。把至今的累積量帶進來，
    # 讓後續的自由 slot 優先補 lateral / random_2d。
    forced = torch.bincount(families[~free & active], minlength=_FAMILY_COUNT)
    cumulative_target = (seen + total) / _FAMILY_COUNT
    target = torch.full(
        (_FAMILY_COUNT,), cumulative_target, dtype=torch.float64, device=device
    )
    deficit = (
        target - emitted - forced.to(torch.float64)
    ).clamp_min(0.0).to(torch.float32)
    if float(deficit.sum()) <= 0.0:
        deficit = torch.ones(_FAMILY_COUNT, device=device)
    # Largest-remainder 配額，不是比例四捨五入。實際注入幾乎是**一次一個 env**
    # （free slot 常常只有 1-2 個），比例乘完再 round 會整個歸零，殘差又被固定
    # 丟給最後一個 family —— 實測 batch=1 時偏差 4.8pp，正好複現 sim 的 4.6pp。
    # 改成先配整數部分、餘額給缺口最大者，n_free=1 也能正確落在最缺的那一類。
    deficit64 = deficit.to(torch.float64)
    base = deficit64.clamp_min(0.0).floor()
    if float(base.sum()) > n_free:
        base = torch.zeros_like(base)
    remainder = int(n_free - int(base.sum()))
    while remainder > 0:
        # 逐一發給「當下缺口最大」的 family。一次只發一個，發完就把該類的
        # 缺口減 1 —— 用 argsort 一次發完的話，remainder > 3 會發不完
        # （只有三個 family），share 加總小於 n_free 就會索引越界。
        j = int(torch.argmax(deficit64 - base))
        base[j] += 1.0
        remainder -= 1
    share = base.long()

    # ── 3. 在分配時就守住「每個 env 最多兩個橫向」──────────────────
    # 走廊只有兩條橫排，第三個橫向沒有排可站，`dynamic_layout_for_families`
    # 會直接 raise。先前用**事後修補**（把超額的橫向與別的 env 交換），
    # 但找不到接收方時會帶著未解的超額退出 —— 64 env 的 smoke 從未觸發，
    # 1024 env 一跑就在 iter 10 之後炸掉整個訓練。
    # 現在改成分配前先算出容量並把超額名額改配給別的 family，
    # 超額在結構上無法產生。
    _LAST_ALLOCATION.clear()
    _LAST_ALLOCATION.update(
        counts=counts, interaction_types=interaction_types, pairs=pairs,
        family_debt=(
            None if family_debt is None
            else {"emitted": family_debt.get("emitted"),
                  "total": family_debt.get("total", 0.0)}
        ),
        global_env_ids=global_env_ids,
    )
    families = _assign_free_slots_within_lateral_cap(
        families, free, free_idx, active, share, deficit64, device,
        counts=counts, interaction_types=interaction_types, pairs=pairs,
        family_debt=family_debt, global_env_ids=global_env_ids,
    )
    _record_family_debt(family_debt, emitted, seen, families, active, total)
    return families, pairs


def _assign_free_slots_within_lateral_cap(
    families: torch.Tensor,
    free: torch.Tensor,
    free_idx: torch.Tensor,
    active: torch.Tensor,
    share: torch.Tensor,
    deficit: torch.Tensor,
    device,
    *,
    counts: torch.Tensor | None = None,
    interaction_types: torch.Tensor | None = None,
    pairs: torch.Tensor | None = None,
    family_debt: dict | None = None,
    global_env_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fill the free slots, never letting an env exceed the lateral lane count.

    橫向名額若超過整批的容納量，多出來的改配給缺口較大的另一個 family ——
    寧可讓整批 family 比例稍微偏離，也不能產出一個蓋不出來的場景。
    """
    share = share.clone()
    free_per_env = free.sum(dim=1)
    forced_lateral = ((families == MOTION_LATERAL) & ~free & active).sum(dim=1)
    capacity = torch.minimum(
        (MAX_PER_FAMILY - forced_lateral).clamp_min(0), free_per_env
    )
    total_capacity = int(capacity.sum())

    excess = int(share[MOTION_LATERAL]) - total_capacity
    if excess > 0:
        share[MOTION_LATERAL] = total_capacity
        others = [MOTION_LONGITUDINAL, MOTION_RANDOM_2D]
        for _ in range(excess):
            # 給當下缺口較大的那一個，維持整批平衡的意圖。
            pick = max(others, key=lambda f: float(deficit[f]) - float(share[f]))
            share[pick] += 1

    lateral_pool = int(share[MOTION_LATERAL])
    other_pool: list[int] = []
    for family in (MOTION_LONGITUDINAL, MOTION_RANDOM_2D):
        other_pool.extend([family] * int(share[family]))
    perm = torch.randperm(len(other_pool), device=device).tolist()
    other_pool = [other_pool[i] for i in perm]

    slots_by_env: dict[int, list[int]] = {}
    for row in range(free_idx.shape[0]):
        env_id = int(free_idx[row, 0])
        slots_by_env.setdefault(env_id, []).append(int(free_idx[row, 1]))

    # env 與 slot 的走訪順序都必須隨機化。照 row 順序填的話，橫向名額會被
    # 前面的 env 先吃光 —— 實測 1000 個 D=5 env：env 0-499 各拿 2 個橫向、
    # env 900-999 各拿 0 個，**整批比例卻完美**，所以 aggregate 審計看不出來。
    env_ids = list(slots_by_env.keys())
    order = torch.randperm(len(env_ids), device=device).tolist()
    env_ids = [env_ids[i] for i in order]

    def fail(reason: str, message: str):
        dump_corridor_allocation_state(
            families, reason=reason, counts=counts,
            interaction_types=interaction_types, pairs=pairs,
            family_debt=family_debt, global_env_ids=global_env_ids,
        )
        raise RuntimeError(message)

    cursor = 0
    for env_id in env_ids:
        slots = slots_by_env[env_id]
        if len(slots) > 1:
            shuffle = torch.randperm(len(slots), device=device).tolist()
            slots = [slots[i] for i in shuffle]
        take = min(int(capacity[env_id]), lateral_pool, len(slots))
        for k, slot in enumerate(slots):
            if k < take:
                families[env_id, slot] = MOTION_LATERAL
            else:
                if cursor >= len(other_pool):
                    # 名額不足時直接索引會 IndexError，那會**繞過**下面的
                    # postcondition，真的失配時反而拿不到 dump。
                    fail(
                        "label_pool_exhausted",
                        f"corridor allocation ran out of labels at env {env_id} "
                        f"slot {slot} (pool={len(other_pool)}, "
                        f"lateral_pool={lateral_pool})",
                    )
                families[env_id, slot] = other_pool[cursor]
                cursor += 1
        lateral_pool -= take

    # Postcondition：分配器出口就驗，不要等到幾何層才炸。
    if lateral_pool != 0 or cursor != len(other_pool):
        fail(
            "label_pool_mismatch",
            f"corridor free-slot allocation left {lateral_pool} lateral and "
            f"{len(other_pool) - cursor} other labels unplaced",
        )
    if bool((families[active] < 0).any()):
        fail(
            "unassigned_active_slot",
            "corridor allocation left an active slot unassigned",
        )
    over = ((families == MOTION_LATERAL) & active).sum(dim=1)
    if int(over.max()) > MAX_PER_FAMILY:
        fail("lateral_cap_breached_at_allocator", (
            f"corridor allocation produced {int(over.max())} lateral obstacles "
            f"in one env (cap {MAX_PER_FAMILY})"
        ))
    return families


def _record_family_debt(family_debt, emitted, seen, families, active, total) -> None:
    """Append this batch's realized family counts to the running ledger.

    帳本必須記錄**實際裝上去的**，不是「有經過分配器的」。任何提早 return
    的路徑都要呼叫這裡，否則帳本與現實脫節，補償就會失效。
    """
    if family_debt is None:
        return
    family_debt["emitted"] = emitted + torch.bincount(
        families[active], minlength=_FAMILY_COUNT
    ).to(torch.float64)
    family_debt["total"] = seen + total
