#!/usr/bin/env python3
"""Train Launcher GUI — 互動式設定 train_rnn_car_wdclip.py 參數並開始訓練

Usage (推薦，中文字型正常):
    python3 scripts/reinforcement_learning/skrl/train/train_launcher.py

Note:
    不要用 ./isaaclab.sh -p 啟動本 GUI，因為 conda Tk 沒有 Xft
    無法渲染 CJK 字型。直接用系統 python3 即可。
"""
from __future__ import annotations

import sys
import os

_SYSTEM_PYTHON = "/usr/bin/python3"
_is_conda = "conda" in sys.executable.lower() or "miniconda" in sys.executable.lower()
if (
    _is_conda
    and os.path.exists(_SYSTEM_PYTHON)
    and os.path.realpath(sys.executable) != os.path.realpath(_SYSTEM_PYTHON)
    and not os.environ.get("_TRAIN_LAUNCHER_SYSPY")
):
    os.environ["_TRAIN_LAUNCHER_SYSPY"] = "1"
    os.execv(_SYSTEM_PYTHON, [_SYSTEM_PYTHON] + sys.argv)

import json
import subprocess
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

# ============================================================================
# Constants
# ============================================================================

REPO_ROOT = Path(__file__).resolve().parents[4]  # /home/aa/IsaacLab
LOGS_DIR = REPO_ROOT / "logs" / "rnn_car"
CONFIGS_DIR = REPO_ROOT / "scripts" / "reinforcement_learning" / "skrl" / "rnn_car_modular" / "configs"
PHASES_DIR = (
    REPO_ROOT / "source" / "isaaclab_tasks" / "isaaclab_tasks"
    / "manager_based" / "locomotion" / "velocity" / "config"
    / "charge_skrl" / "curriculum" / "phases"
)
TRAIN_SCRIPT = (
    REPO_ROOT / "scripts" / "reinforcement_learning" / "skrl" / "train" / "train_rnn_car_wdclip.py"
)

TASKS = [
    "Isaac-Navigation-Charge-VLP16-Curriculum-WD",
    "Isaac-Navigation-Charge-VLP16-Curriculum-WD-TCorridor",
]

ALGORITHMS = ["a2c", "ppo"]
ENCODER_MODES = ["extractor_rnn", "wd_exact_rnn", "flat_rnn"]
AUX_PROFILES = ["wd_7d_geometry", "none"]
CRITIC_PROFILES = ["symmetric", "asymmetric"]
REWARD_PROFILES = ["wd_sparse"]
RNN_TYPES = ["RNN", "GRU"]
OBSTACLE_MODES = ["rule_based", "learned"]
ADV_NORM_MODES = ["mean_only", "partial", "full"]

# SA stage descriptions for quick reference
SA_DESCRIPTIONS = {
    1: "SA1: 基礎導航 (no obstacles, no walls)",
    2: "SA2: 靜態障礙物 + walls",
    3: "SA3: 動態障礙物 (patrol)",
    4: "SA4: 靜態+動態混合 (path_crossing)",
    5: "SA5: T-Corridor 窄通道",
    6: "SA6: Dense dynamic (near_miss 20%)",
    7: "SA7: Dense + 強 DR (Per-Episode DR)",
    8: "SA8: Full complexity",
}


def _load_curriculum_stages() -> list[dict]:
    """Load STAGES from wd_single_agent_v1.py without importing Isaac Lab."""
    phases_py = PHASES_DIR / "wd_single_agent_v1.py"
    if not phases_py.exists():
        return []
    try:
        ns: dict = {}
        exec(compile(phases_py.read_text(encoding="utf-8"), str(phases_py), "exec"), ns)
        return ns.get("STAGES", [])
    except Exception:
        return []


CURRICULUM_STAGES: list[dict] = _load_curriculum_stages()


def format_stage_info(stage_idx: int, lang: int = 1) -> str:
    """Format curriculum stage info as human-readable text.

    lang: 0=EN, 1=ZH
    """
    if not CURRICULUM_STAGES or stage_idx < 1 or stage_idx > len(CURRICULUM_STAGES):
        return ""
    stage = CURRICULUM_STAGES[stage_idx - 1]
    scene = stage.get("scene", {})
    reward = stage.get("reward", {})
    behavior = stage.get("behavior", {})

    goals = scene.get("goals", "?")
    static_obs = scene.get("static_obstacles", 0)
    dynamic_obs = scene.get("dynamic_obstacles", 0)
    dynamic_min = scene.get("dynamic_obstacles_min", 0)
    walls_min = scene.get("walls_min", 0)
    walls_max = scene.get("walls_max", 0)
    wall_len = scene.get("wall_length", 0)
    episode_s = scene.get("episode_s", 60)
    gamma = scene.get("gamma", 0.99)
    goal_dist = scene.get("goal_distance", (0, 0))
    goal_mv_speed = scene.get("goal_move_speed", 0)

    penalty = reward.get("penalty_hit", 0)
    goal_reward = reward.get("reward_get_goal", 40)
    cost_op = reward.get("cost_operate", 0)
    penalty_to = reward.get("penalty_timeout", 0)

    obs_speed = behavior.get("obstacle_speed", 0.8)
    mix = behavior.get("behavior_mix", {})

    if lang == 1:
        lines = [
            f"--- SA{stage_idx} 場景資訊 ({stage.get('name', '')}) ---",
            f"  Goals: {goals}   目標距離: {goal_dist[0]}~{goal_dist[1]}m"
            + (f"   移動速度: {goal_mv_speed} m/s" if goal_mv_speed > 0 else "   (靜態)"),
            f"  靜態障礙: {static_obs}   動態障礙: {dynamic_min}~{dynamic_obs}"
            f"   牆壁: {walls_min}~{walls_max} (長 {wall_len}m)",
            f"  Episode: {episode_s}s   γ: {gamma}",
            "",
            f"--- Reward ---",
            f"  Goal 獎勵: +{goal_reward}   碰撞懲罰: {penalty}"
            + (f"   超時懲罰: {penalty_to}" if penalty_to else "")
            + f"   操作成本: {cost_op}/step",
            "",
            f"--- Behavior (障礙物速度: {obs_speed} m/s) ---",
        ]
        for btype, ratio in sorted(mix.items(), key=lambda x: -x[1]):
            lines.append(f"  {btype}: {ratio*100:.0f}%")
    else:
        lines = [
            f"--- SA{stage_idx} Scene Info ({stage.get('name', '')}) ---",
            f"  Goals: {goals}   Distance: {goal_dist[0]}~{goal_dist[1]}m"
            + (f"   Move speed: {goal_mv_speed} m/s" if goal_mv_speed > 0 else "   (static)"),
            f"  Static obs: {static_obs}   Dynamic obs: {dynamic_min}~{dynamic_obs}"
            f"   Walls: {walls_min}~{walls_max} (len {wall_len}m)",
            f"  Episode: {episode_s}s   gamma: {gamma}",
            "",
            f"--- Reward ---",
            f"  Goal: +{goal_reward}   Collision: {penalty}"
            + (f"   Timeout: {penalty_to}" if penalty_to else "")
            + f"   Operate cost: {cost_op}/step",
            "",
            f"--- Behavior (obs speed: {obs_speed} m/s) ---",
        ]
        for btype, ratio in sorted(mix.items(), key=lambda x: -x[1]):
            lines.append(f"  {btype}: {ratio*100:.0f}%")

    return "\n".join(lines)


# ============================================================================
# i18n — (English, 繁體中文)
# ============================================================================

_STRINGS: dict[str, tuple[str, str]] = {
    "title": ("Charge Train Launcher", "Charge 訓練啟動器"),

    # Section headers
    "sec_config": ("Experiment Config & SA Stage", "實驗配置 & SA 階段"),
    "sec_resume": ("Checkpoint Resume", "續訓 Checkpoint"),
    "sec_budget": ("Training Budget & Run", "訓練預算 & 執行名稱"),
    "sec_algo": ("Algorithm & RL Hyperparams", "演算法 & RL 超參數"),
    "sec_model": ("Model / Encoder / Aux", "模型 / 編碼器 / 輔助任務"),
    "sec_critic": ("Critic Profile", "Critic 配置"),
    "sec_lidar_dr": ("Sim-to-Real DR: LiDAR (TLNI)", "Sim-to-Real DR: LiDAR (TLNI)"),
    "sec_physics_dr": ("Sim-to-Real DR: Physics & Disturbance", "Sim-to-Real DR: 物理 & 擾動"),
    "sec_advanced_dr": ("Sim-to-Real DR: Advanced", "Sim-to-Real DR: 進階功能"),
    "sec_logging": ("Logging & Runtime", "日誌 & 執行設定"),
    "sec_preview": ("Command Preview", "指令預覽"),

    # Config section
    "experiment_config": ("Experiment Config:", "實驗配置:"),
    "config_desc": ("Description:", "描述:"),
    "sa_stage": ("SA Stage:", "SA 階段:"),
    "fixed_stage": ("Fixed stage", "鎖定階段"),
    "load_yaml": ("Load from YAML", "從 YAML 載入"),
    "sa_desc": ("Stage info:", "階段說明:"),

    # Resume section
    "run": ("Previous Run:", "過去訓練:"),
    "checkpoint": ("Checkpoint:", "模型檔:"),
    "browse": ("Browse...", "瀏覽..."),
    "no_resume_opt": ("Don't resume optimizer", "不載入 optimizer"),

    # Budget section
    "task": ("Task:", "任務:"),
    "run_name": ("Run name:", "執行名稱:"),
    "num_envs": ("num_envs:", "環境數:"),
    "timesteps": ("Timesteps:", "訓練步數:"),
    "rollout_length": ("Rollout length:", "Rollout 長度:"),
    "seed": ("Seed:", "隨機種子:"),
    "headless": ("Headless", "無畫面"),

    # Algorithm section
    "algorithm": ("Algorithm:", "演算法:"),
    "lr": ("LR (policy):", "學習率 (policy):"),
    "rnn_lr": ("LR (RNN/aux):", "學習率 (RNN/aux):"),
    "gamma": ("Gamma:", "折扣因子:"),
    "gae_lambda": ("GAE Lambda:", "GAE Lambda:"),
    "vf_coeff": ("VF coeff:", "Value 係數:"),
    "max_grad_norm": ("Grad norm:", "梯度裁剪:"),
    "normalize_return": ("Normalize return", "正規化 return"),
    "adv_norm_mode": ("Adv norm:", "Advantage 正規化:"),
    "ent_coeff_linear": ("Ent (linear):", "熵 (線速度):"),
    "ent_coeff_angular": ("Ent (angular):", "熵 (角速度):"),
    "lr_decay": ("LR decay:", "學習率衰減:"),
    # PPO
    "ppo_epochs": ("PPO epochs:", "PPO 迭代:"),
    "mini_batches": ("Mini-batches:", "Mini-batches:"),
    "clip_eps": ("Clip eps:", "PPO clip:"),
    # WD caps
    "wd_update_clip": ("WD update clip", "WD 分離裁剪"),
    "wd_actor_clip": ("Actor clip:", "Actor 裁剪:"),
    "wd_critic_clip": ("Critic clip:", "Critic 裁剪:"),
    "policy_loss_clamp": ("Policy clamp:", "Policy clamp:"),
    "vf_term_clamp": ("VF clamp:", "VF clamp:"),

    # Model section
    "encoder_profile": ("Encoder:", "編碼器:"),
    "hidden_dim": ("RNN hidden:", "RNN 隱藏:"),
    "preprocess_dim": ("Preprocess:", "前處理:"),
    "fc_dim": ("FC dim:", "FC 維度:"),
    "rnn_type": ("RNN type:", "RNN 類型:"),
    "aux_profile": ("Aux profile:", "輔助任務:"),
    "aux_seq_len": ("Aux seq len:", "Aux 序列:"),
    "aux_seq_batch": ("Aux batch:", "Aux 批次:"),
    "aux_grad_clip": ("Aux grad clip:", "Aux 裁剪:"),
    "disable_aux": ("Disable aux training", "關閉輔助訓練"),

    # Critic section
    "critic_profile": ("Critic profile:", "Critic 配置:"),
    "critic_desc_sym": ("Policy & Value share same obs (91D)", "Policy 與 Value 看相同 obs (91D)"),
    "critic_desc_asym": ("Value gets +50D privileged obs (141D total)", "Value 額外獲得 50D 特權 obs (141D)"),

    # LiDAR DR section
    "lidar_no_noise": ("No LiDAR noise (clean)", "無 LiDAR 噪聲 (乾淨)"),
    "no_domain_rand": ("Disable ALL DR", "關閉所有域隨機"),
    "lidar_l1_header": ("Layer 1: Per-Ray", "層 1: 逐射線"),
    "displacement_std": ("Displacement σ:", "位移 σ:"),
    "hole_rate": ("Hole rate:", "空洞率:"),
    "distractor_rate": ("Distractor rate:", "干擾率:"),
    "distance_bias": ("Distance bias", "距離偏差"),
    "per_ring_bias": ("Per-ring bias", "逐環偏差"),
    "lidar_l2_header": ("Layer 2: Per-Bin", "層 2: 逐 Bin"),
    "obs_noise_std": ("Obs noise σ:", "觀測噪聲 σ:"),
    "block_dropout": ("Block dropout:", "區塊丟棄:"),
    "block_width": ("Block width:", "區塊寬度:"),
    "lidar_l3_header": ("Layer 3: Per-Episode DR", "層 3: 逐回合 DR"),
    "disp_std_dr": ("Disp σ DR:", "位移 σ DR:"),
    "hole_rate_dr": ("Hole rate DR:", "空洞率 DR:"),

    # Physics DR section
    "physics_header": ("Physics DR", "物理 DR"),
    "mass_dr": ("Mass scale:", "質量縮放:"),
    "friction_dr": ("Friction scale:", "摩擦縮放:"),
    "com_offset": ("CoM offset (m):", "質心偏移 (m):"),
    "disturbance_header": ("Disturbance DR", "外部擾動 DR"),
    "wind_force": ("Wind force (N):", "風力 (N):"),
    "push_force": ("Push force (N):", "推力 (N):"),
    "push_ratio": ("Push ratio:", "推力比例:"),
    "actuator_header": ("Actuator DR", "致動器 DR"),
    "enable_actuator": ("Enable actuator DR", "啟用致動器 DR"),
    "actuator_delay": ("Action delay:", "動作延遲:"),
    "velocity_scale": ("Velocity scale:", "速度縮放:"),
    "motor_lag": ("Motor lag α:", "馬達滯後 α:"),

    # Advanced DR section
    "obs_delay": ("Obs delay steps:", "觀測延遲步數:"),
    "heading_weight": ("Heading stability:", "方向穩定性:"),
    "rgdr_enabled": ("RGDR (reward-guided)", "RGDR (獎勵引導)"),
    "rgdr_clamp": ("RGDR clamp:", "RGDR 夾範圍:"),
    "doraemon_enabled": ("DORAEMON auto DR", "DORAEMON 自動 DR"),
    "doraemon_sr_th": ("SR threshold:", "SR 門檻:"),
    "doraemon_interval": ("Check interval:", "檢查間隔:"),
    "doraemon_rate": ("Expansion rate:", "擴張比例:"),

    # Logging section
    "log_interval": ("Log interval:", "日誌間隔:"),
    "save_interval": ("Save interval:", "存檔間隔:"),
    "wandb_notes": ("WandB notes:", "WandB 備註:"),
    "extra_args": ("Extra args:", "額外參數:"),

    # Buttons
    "preview_cmd": ("Preview Command", "預覽指令"),
    "copy_clipboard": ("Copy to Clipboard", "複製到剪貼簿"),
    "launch_train": ("  LAUNCH TRAIN  ", "  開始訓練  "),
    "kill_train": ("Kill All Train", "清除所有訓練"),
    "save_settings": ("Save Settings", "儲存設定"),
    "reset_from_yaml": ("Reset from YAML", "從 YAML 重置"),

    # Dialogs
    "copied_title": ("Copied", "已複製"),
    "copied_msg": ("Command copied to clipboard!", "指令已複製到剪貼簿！"),
    "kill_title": ("Kill Train", "清除訓練"),
    "kill_confirm": ("Kill all train_rnn_car processes?", "清除所有 train_rnn_car 進程？"),
    "kill_done": ("Killed {n} process(es).", "已清除 {n} 個進程。"),
    "kill_none": ("No train processes found.", "沒有找到訓練進程。"),
    "launch_title": ("Launch Train", "開始訓練"),
    "launch_confirm": ("Launch training?", "確定開始訓練？"),
    "launched_title": ("Launched", "已開始"),
    "launched_msg": ("Training process started!", "訓練程序已開始！"),
    "error_title": ("Error", "錯誤"),
    "error_launch": ("Failed to launch:", "開始失敗："),
    "select_ckpt": ("Select Checkpoint", "選擇模型檔"),
    "saved_title": ("Saved", "已儲存"),
    "saved_msg": ("Settings saved!", "設定已儲存！"),
    "loaded_msg": ("Settings loaded from last session.", "已載入上次設定。"),
    "no_yaml": ("(none — use GUI values)", "(無 — 使用 GUI 值)"),
    "yaml_loaded": ("YAML loaded: {name}", "已載入 YAML: {name}"),
    "range_sep": (" ~ ", " ~ "),
}


def scan_config_yamls() -> list[str]:
    """Scan configs/ for YAML files, return sorted names (without .yaml)."""
    if not CONFIGS_DIR.exists():
        return []
    return sorted(
        p.stem for p in CONFIGS_DIR.glob("*.yaml")
    )


def scan_run_dirs() -> list[str]:
    """Scan logs/rnn_car/ for run directories with checkpoints."""
    dirs = []
    if not LOGS_DIR.exists():
        return dirs
    for run_dir in sorted(LOGS_DIR.iterdir(), reverse=True):
        if run_dir.is_dir() and any(run_dir.glob("checkpoint_*.pt")):
            dirs.append(run_dir.name)
    return dirs


def load_yaml_config(name: str) -> dict | None:
    """Load a YAML config by name, return as dict or None."""
    path = CONFIGS_DIR / f"{name}.yaml"
    if not path.exists():
        return None
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f)
    except ImportError:
        with open(path) as f:
            content = f.read()
        data = {}
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                key, _, val = line.partition(":")
                key = key.strip()
                val = val.strip()
                if val.startswith("#"):
                    val = ""
                comment_idx = val.find("  #")
                if comment_idx > 0:
                    val = val[:comment_idx].strip()
                if val.startswith("[") and val.endswith("]"):
                    inner = val[1:-1]
                    parts = [p.strip() for p in inner.split(",")]
                    try:
                        data[key] = [float(p) for p in parts]
                    except ValueError:
                        data[key] = val
                elif val in ("true", "True"):
                    data[key] = True
                elif val in ("false", "False"):
                    data[key] = False
                elif val in ("null", "None", "~", ""):
                    data[key] = None
                else:
                    try:
                        data[key] = int(val)
                    except ValueError:
                        try:
                            data[key] = float(val)
                        except ValueError:
                            data[key] = val.strip('"').strip("'")
        return data
    except Exception:
        return None


# ============================================================================
# GUI
# ============================================================================


class TrainLauncherApp:
    _EN_HEADER_FONT = ("Helvetica", 14, "bold")
    _EN_BODY_FONT = ("Helvetica", 12)
    _ZH_HEADER_FONT_FALLBACK = ("TkDefaultFont", 14)
    _ZH_BODY_FONT_FALLBACK = ("TkDefaultFont", 12)
    _ZH_FONT_CANDIDATES = [
        "Noto Sans CJK TC",
        "Source Han Sans TC",
        "WenQuanYi Micro Hei",
        "AR PL UMing TW",
        "Microsoft JhengHei",
    ]

    def __init__(self, root: tk.Tk):
        self.root = root
        self.lang = 1  # default ZH; 0=EN, 1=ZH
        self._i18n_widgets: list[tuple[tk.Widget, str]] = []

        available = {f.lower() for f in tkfont.families()}
        zh_family = None
        for candidate in self._ZH_FONT_CANDIDATES:
            if candidate.lower() in available:
                zh_family = candidate
                break
        self._zh_family = zh_family
        if zh_family:
            self._zh_header_font = (zh_family, 14)
            self._zh_body_font = (zh_family, 12)
        else:
            self._zh_header_font = self._ZH_HEADER_FONT_FALLBACK
            self._zh_body_font = self._ZH_BODY_FONT_FALLBACK

        self.root.title(self._t("title"))
        self.root.geometry("1100x980")
        self.root.resizable(True, True)

        # Force CJK font on ALL default fonts so every widget renders Chinese
        _base_family = zh_family or "TkDefaultFont"
        for fname in ("TkDefaultFont", "TkTextFont", "TkMenuFont",
                      "TkHeadingFont", "TkCaptionFont", "TkTooltipFont"):
            try:
                tkfont.nametofont(fname).configure(family=_base_family, size=12)
            except Exception:
                pass

        style = ttk.Style()
        _body = self._zh_body_font if self.lang == 1 else self._EN_BODY_FONT
        _header = self._zh_header_font if self.lang == 1 else self._EN_HEADER_FONT
        style.configure("Header.TLabel", font=_header)
        style.configure("Desc.TLabel", foreground="gray", font=_body)
        for s in ("TLabel", "TCheckbutton", "TButton", "TLabelframe.Label",
                  "TSpinbox", "TEntry", "TCombobox", "TRadiobutton"):
            style.configure(s, font=_body)

        # Scrollable canvas
        canvas = tk.Canvas(root)
        scrollbar = ttk.Scrollbar(root, orient="vertical", command=canvas.yview)
        self.main_frame = ttk.Frame(canvas)
        self.main_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self.main_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        canvas.bind_all("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))
        canvas.bind_all("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))

        self._build_ui()

    # ------------------------------------------------------------------
    # i18n
    # ------------------------------------------------------------------

    def _t(self, key: str) -> str:
        return _STRINGS.get(key, (key, key))[self.lang]

    def _reg(self, widget: tk.Widget, key: str) -> tk.Widget:
        self._i18n_widgets.append((widget, key))
        return widget

    def _switch_lang(self):
        self.lang = 1 - self.lang
        self.root.title(self._t("title"))
        self.lang_btn.configure(text="EN" if self.lang == 1 else "中文")
        style = ttk.Style()
        if self.lang == 1:
            style.configure("Header.TLabel", font=self._zh_header_font)
            for s in ("TLabel", "TCheckbutton", "TButton", "TLabelframe.Label",
                      "TSpinbox", "TEntry", "TCombobox", "TRadiobutton"):
                style.configure(s, font=self._zh_body_font)
        else:
            style.configure("Header.TLabel", font=self._EN_HEADER_FONT)
            for s in ("TLabel", "TCheckbutton", "TButton", "TLabelframe.Label",
                      "TSpinbox", "TEntry", "TCombobox", "TRadiobutton"):
                style.configure(s, font=self._EN_BODY_FONT)
        for widget, key in self._i18n_widgets:
            try:
                widget.configure(text=self._t(key))
            except Exception:
                pass
        self._update_stage_info()

    # ------------------------------------------------------------------
    # UI layout
    # ------------------------------------------------------------------

    def _build_ui(self):
        f = self.main_frame
        row = 0

        # Language toggle
        lang_frame = ttk.Frame(f)
        lang_frame.grid(row=row, column=0, columnspan=6, sticky="e", padx=5, pady=(5, 0))
        self.lang_btn = ttk.Button(lang_frame, text="EN", width=6, command=self._switch_lang)
        self.lang_btn.pack(side="right")
        row += 1

        # ── 1. Experiment Config & SA Stage ──
        row = self._separator(f, row, "sec_config")

        self._reg(ttk.Label(f, text=self._t("experiment_config")), "experiment_config").grid(
            row=row, column=0, sticky="w", padx=5)
        config_names = scan_config_yamls()
        self.config_var = tk.StringVar(value=config_names[0] if config_names else "")
        config_cb = ttk.Combobox(f, textvariable=self.config_var, values=config_names, width=40)
        config_cb.grid(row=row, column=1, columnspan=2, sticky="w", padx=5)
        config_cb.bind("<<ComboboxSelected>>", self._on_config_selected)
        self._reg(ttk.Button(f, text=self._t("load_yaml"), command=self._load_yaml_btn),
                  "load_yaml").grid(row=row, column=3, padx=5)
        row += 1

        self._reg(ttk.Label(f, text=self._t("config_desc")), "config_desc").grid(
            row=row, column=0, sticky="w", padx=5)
        self.config_desc_var = tk.StringVar(value="")
        ttk.Label(f, textvariable=self.config_desc_var, style="Desc.TLabel",
                  wraplength=600).grid(row=row, column=1, columnspan=5, sticky="w", padx=5)
        row += 1

        self._reg(ttk.Label(f, text=self._t("sa_stage")), "sa_stage").grid(
            row=row, column=0, sticky="w", padx=5)
        self.sa_stage_var = tk.IntVar(value=1)
        ttk.Spinbox(f, from_=1, to=8, textvariable=self.sa_stage_var, width=5).grid(
            row=row, column=1, sticky="w", padx=5)
        self.fixed_stage_var = tk.BooleanVar(value=True)
        self._reg(ttk.Checkbutton(f, text=self._t("fixed_stage"),
                                   variable=self.fixed_stage_var), "fixed_stage").grid(
            row=row, column=2, sticky="w", padx=5)
        row += 1

        self.sa_desc_var = tk.StringVar(value=SA_DESCRIPTIONS.get(1, ""))
        ttk.Label(f, textvariable=self.sa_desc_var, style="Desc.TLabel").grid(
            row=row, column=0, columnspan=6, sticky="w", padx=20)
        self.sa_stage_var.trace_add("write", self._on_sa_changed)
        row += 1

        # Stage info panel (scene + reward + behavior)
        self.stage_info_text = tk.Text(
            f, height=12, width=72, wrap="none", relief="groove", borderwidth=1,
            state="disabled", background="#f5f5f0",
        )
        if self._zh_family:
            self.stage_info_text.configure(font=(self._zh_family, 11))
        else:
            self.stage_info_text.configure(font=("monospace", 11))
        self.stage_info_text.grid(
            row=row, column=0, columnspan=6, sticky="w", padx=20, pady=(2, 5))
        row += 1
        self._update_stage_info()

        # ── 2. Checkpoint Resume ──
        row = self._separator(f, row, "sec_resume")

        self._reg(ttk.Label(f, text=self._t("run")), "run").grid(
            row=row, column=0, sticky="w", padx=5)
        run_dirs = scan_run_dirs()
        self.run_var = tk.StringVar(value=run_dirs[0] if run_dirs else "")
        run_cb = ttk.Combobox(f, textvariable=self.run_var, values=run_dirs, width=45)
        run_cb.grid(row=row, column=1, columnspan=3, sticky="w", padx=5)
        run_cb.bind("<<ComboboxSelected>>", self._on_run_selected)
        row += 1

        self._reg(ttk.Label(f, text=self._t("checkpoint")), "checkpoint").grid(
            row=row, column=0, sticky="w", padx=5)
        self.ckpt_var = tk.StringVar()
        self.ckpt_cb = ttk.Combobox(f, textvariable=self.ckpt_var, values=[], width=55)
        self.ckpt_cb.grid(row=row, column=1, columnspan=2, sticky="w", padx=5)
        self._reg(ttk.Button(f, text=self._t("browse"), command=self._browse_ckpt),
                  "browse").grid(row=row, column=3, padx=5)
        row += 1
        self._on_run_selected(None)

        self.no_resume_opt_var = tk.BooleanVar(value=True)
        self._reg(ttk.Checkbutton(f, text=self._t("no_resume_opt"),
                                   variable=self.no_resume_opt_var), "no_resume_opt").grid(
            row=row, column=0, columnspan=3, sticky="w", padx=5)
        row += 1

        # ── 3. Training Budget ──
        row = self._separator(f, row, "sec_budget")

        self._reg(ttk.Label(f, text=self._t("task")), "task").grid(
            row=row, column=0, sticky="w", padx=5)
        self.task_var = tk.StringVar(value=TASKS[0])
        ttk.Combobox(f, textvariable=self.task_var, values=TASKS, width=55).grid(
            row=row, column=1, columnspan=3, sticky="w", padx=5)
        row += 1

        self._reg(ttk.Label(f, text=self._t("run_name")), "run_name").grid(
            row=row, column=0, sticky="w", padx=5)
        self.run_name_var = tk.StringVar(value="")
        ttk.Entry(f, textvariable=self.run_name_var, width=45).grid(
            row=row, column=1, columnspan=3, sticky="w", padx=5)
        row += 1

        # num_envs, timesteps
        self._reg(ttk.Label(f, text=self._t("num_envs")), "num_envs").grid(
            row=row, column=0, sticky="w", padx=5)
        self.num_envs_var = tk.IntVar(value=1024)
        ttk.Spinbox(f, from_=4, to=8192, increment=256,
                     textvariable=self.num_envs_var, width=8).grid(
            row=row, column=1, sticky="w", padx=5)
        self._reg(ttk.Label(f, text=self._t("timesteps")), "timesteps").grid(
            row=row, column=2, sticky="w", padx=5)
        self.timesteps_var = tk.IntVar(value=270000)
        ttk.Spinbox(f, from_=1000, to=2000000, increment=10000,
                     textvariable=self.timesteps_var, width=10).grid(
            row=row, column=3, sticky="w", padx=5)
        row += 1

        # rollout_length, seed
        self._reg(ttk.Label(f, text=self._t("rollout_length")), "rollout_length").grid(
            row=row, column=0, sticky="w", padx=5)
        self.rollout_var = tk.IntVar(value=300)
        ttk.Spinbox(f, from_=32, to=2048, increment=32,
                     textvariable=self.rollout_var, width=8).grid(
            row=row, column=1, sticky="w", padx=5)
        self._reg(ttk.Label(f, text=self._t("seed")), "seed").grid(
            row=row, column=2, sticky="w", padx=5)
        self.seed_var = tk.IntVar(value=42)
        ttk.Spinbox(f, from_=1, to=99999, textvariable=self.seed_var, width=8).grid(
            row=row, column=3, sticky="w", padx=5)
        row += 1

        self.headless_var = tk.BooleanVar(value=True)
        self._reg(ttk.Checkbutton(f, text=self._t("headless"),
                                   variable=self.headless_var), "headless").grid(
            row=row, column=0, columnspan=2, sticky="w", padx=5)
        row += 1

        # ── 4. Algorithm & RL ──
        row = self._separator(f, row, "sec_algo")

        self._reg(ttk.Label(f, text=self._t("algorithm")), "algorithm").grid(
            row=row, column=0, sticky="w", padx=5)
        self.algo_var = tk.StringVar(value="a2c")
        ttk.Combobox(f, textvariable=self.algo_var, values=ALGORITHMS,
                      width=8, state="readonly").grid(row=row, column=1, sticky="w", padx=5)
        self.algo_var.trace_add("write", self._on_algo_changed)
        row += 1

        # LR row
        self._reg(ttk.Label(f, text=self._t("lr")), "lr").grid(
            row=row, column=0, sticky="w", padx=5)
        self.lr_var = tk.DoubleVar(value=0.0005)
        ttk.Entry(f, textvariable=self.lr_var, width=10).grid(
            row=row, column=1, sticky="w", padx=5)
        self._reg(ttk.Label(f, text=self._t("rnn_lr")), "rnn_lr").grid(
            row=row, column=2, sticky="w", padx=5)
        self.rnn_lr_var = tk.DoubleVar(value=0.0005)
        ttk.Entry(f, textvariable=self.rnn_lr_var, width=10).grid(
            row=row, column=3, sticky="w", padx=5)
        row += 1

        # gamma, gae_lambda
        self._reg(ttk.Label(f, text=self._t("gamma")), "gamma").grid(
            row=row, column=0, sticky="w", padx=5)
        self.gamma_var = tk.DoubleVar(value=0.991)
        ttk.Entry(f, textvariable=self.gamma_var, width=10).grid(
            row=row, column=1, sticky="w", padx=5)
        self._reg(ttk.Label(f, text=self._t("gae_lambda")), "gae_lambda").grid(
            row=row, column=2, sticky="w", padx=5)
        self.gae_lambda_var = tk.DoubleVar(value=0.95)
        ttk.Entry(f, textvariable=self.gae_lambda_var, width=10).grid(
            row=row, column=3, sticky="w", padx=5)
        row += 1

        # vf_coeff, max_grad_norm
        self._reg(ttk.Label(f, text=self._t("vf_coeff")), "vf_coeff").grid(
            row=row, column=0, sticky="w", padx=5)
        self.vf_coeff_var = tk.DoubleVar(value=0.5)
        ttk.Entry(f, textvariable=self.vf_coeff_var, width=10).grid(
            row=row, column=1, sticky="w", padx=5)
        self._reg(ttk.Label(f, text=self._t("max_grad_norm")), "max_grad_norm").grid(
            row=row, column=2, sticky="w", padx=5)
        self.max_grad_norm_var = tk.DoubleVar(value=1.0)
        ttk.Entry(f, textvariable=self.max_grad_norm_var, width=10).grid(
            row=row, column=3, sticky="w", padx=5)
        row += 1

        # normalize_return, adv_norm_mode
        self.normalize_return_var = tk.BooleanVar(value=True)
        self._reg(ttk.Checkbutton(f, text=self._t("normalize_return"),
                                   variable=self.normalize_return_var), "normalize_return").grid(
            row=row, column=0, columnspan=2, sticky="w", padx=5)
        self._reg(ttk.Label(f, text=self._t("adv_norm_mode")), "adv_norm_mode").grid(
            row=row, column=2, sticky="w", padx=5)
        self.adv_norm_var = tk.StringVar(value="mean_only")
        ttk.Combobox(f, textvariable=self.adv_norm_var, values=ADV_NORM_MODES,
                      width=10, state="readonly").grid(row=row, column=3, sticky="w", padx=5)
        row += 1

        # Entropy coefficients
        self._reg(ttk.Label(f, text=self._t("ent_coeff_linear")), "ent_coeff_linear").grid(
            row=row, column=0, sticky="w", padx=5)
        self.ent_linear_var = tk.DoubleVar(value=0.0)
        ttk.Entry(f, textvariable=self.ent_linear_var, width=10).grid(
            row=row, column=1, sticky="w", padx=5)
        self._reg(ttk.Label(f, text=self._t("ent_coeff_angular")), "ent_coeff_angular").grid(
            row=row, column=2, sticky="w", padx=5)
        self.ent_angular_var = tk.DoubleVar(value=0.0)
        ttk.Entry(f, textvariable=self.ent_angular_var, width=10).grid(
            row=row, column=3, sticky="w", padx=5)
        row += 1

        # lr_decay
        self._reg(ttk.Label(f, text=self._t("lr_decay")), "lr_decay").grid(
            row=row, column=0, sticky="w", padx=5)
        self.lr_decay_var = tk.DoubleVar(value=0.0)
        ttk.Entry(f, textvariable=self.lr_decay_var, width=10).grid(
            row=row, column=1, sticky="w", padx=5)
        row += 1

        # PPO-specific frame
        self.ppo_frame = ttk.LabelFrame(f, text="PPO")
        self.ppo_frame.grid(row=row, column=0, columnspan=4, sticky="ew", padx=5, pady=3)
        pf = self.ppo_frame

        self._reg(ttk.Label(pf, text=self._t("ppo_epochs")), "ppo_epochs").grid(
            row=0, column=0, sticky="w", padx=5)
        self.ppo_epochs_var = tk.IntVar(value=1)
        ttk.Spinbox(pf, from_=1, to=20, textvariable=self.ppo_epochs_var, width=5).grid(
            row=0, column=1, sticky="w", padx=5)
        self._reg(ttk.Label(pf, text=self._t("mini_batches")), "mini_batches").grid(
            row=0, column=2, sticky="w", padx=5)
        self.mini_batches_var = tk.IntVar(value=1)
        ttk.Spinbox(pf, from_=1, to=64, textvariable=self.mini_batches_var, width=5).grid(
            row=0, column=3, sticky="w", padx=5)
        self._reg(ttk.Label(pf, text=self._t("clip_eps")), "clip_eps").grid(
            row=1, column=0, sticky="w", padx=5)
        self.clip_eps_var = tk.DoubleVar(value=0.2)
        ttk.Entry(pf, textvariable=self.clip_eps_var, width=8).grid(
            row=1, column=1, sticky="w", padx=5)
        row += 1

        # WD caps frame
        self.wd_frame = ttk.LabelFrame(f, text="WD Gradient Caps")
        self.wd_frame.grid(row=row, column=0, columnspan=4, sticky="ew", padx=5, pady=3)
        wf = self.wd_frame
        self.wd_clip_var = tk.BooleanVar(value=True)
        self._reg(ttk.Checkbutton(wf, text=self._t("wd_update_clip"),
                                   variable=self.wd_clip_var), "wd_update_clip").grid(
            row=0, column=0, columnspan=2, sticky="w", padx=5)

        self._reg(ttk.Label(wf, text=self._t("wd_actor_clip")), "wd_actor_clip").grid(
            row=1, column=0, sticky="w", padx=5)
        self.wd_actor_clip_var = tk.DoubleVar(value=8.0)
        ttk.Entry(wf, textvariable=self.wd_actor_clip_var, width=8).grid(
            row=1, column=1, sticky="w", padx=5)
        self._reg(ttk.Label(wf, text=self._t("wd_critic_clip")), "wd_critic_clip").grid(
            row=1, column=2, sticky="w", padx=5)
        self.wd_critic_clip_var = tk.DoubleVar(value=30.0)
        ttk.Entry(wf, textvariable=self.wd_critic_clip_var, width=8).grid(
            row=1, column=3, sticky="w", padx=5)

        self._reg(ttk.Label(wf, text=self._t("policy_loss_clamp")), "policy_loss_clamp").grid(
            row=2, column=0, sticky="w", padx=5)
        self.policy_clamp_var = tk.DoubleVar(value=20.0)
        ttk.Entry(wf, textvariable=self.policy_clamp_var, width=8).grid(
            row=2, column=1, sticky="w", padx=5)
        self._reg(ttk.Label(wf, text=self._t("vf_term_clamp")), "vf_term_clamp").grid(
            row=2, column=2, sticky="w", padx=5)
        self.vf_clamp_var = tk.DoubleVar(value=8.0)
        ttk.Entry(wf, textvariable=self.vf_clamp_var, width=8).grid(
            row=2, column=3, sticky="w", padx=5)
        row += 1

        self._on_algo_changed()

        # ── 5. Model / Encoder / Aux ──
        row = self._separator(f, row, "sec_model")

        self._reg(ttk.Label(f, text=self._t("encoder_profile")), "encoder_profile").grid(
            row=row, column=0, sticky="w", padx=5)
        self.encoder_var = tk.StringVar(value="extractor_rnn")
        ttk.Combobox(f, textvariable=self.encoder_var, values=ENCODER_MODES,
                      width=18, state="readonly").grid(row=row, column=1, sticky="w", padx=5)
        self._reg(ttk.Label(f, text=self._t("rnn_type")), "rnn_type").grid(
            row=row, column=2, sticky="w", padx=5)
        self.rnn_type_var = tk.StringVar(value="RNN")
        ttk.Combobox(f, textvariable=self.rnn_type_var, values=RNN_TYPES,
                      width=6, state="readonly").grid(row=row, column=3, sticky="w", padx=5)
        row += 1

        self._reg(ttk.Label(f, text=self._t("hidden_dim")), "hidden_dim").grid(
            row=row, column=0, sticky="w", padx=5)
        self.hidden_dim_var = tk.IntVar(value=30)
        ttk.Spinbox(f, from_=8, to=256, textvariable=self.hidden_dim_var, width=6).grid(
            row=row, column=1, sticky="w", padx=5)
        self._reg(ttk.Label(f, text=self._t("preprocess_dim")), "preprocess_dim").grid(
            row=row, column=2, sticky="w", padx=5)
        self.preprocess_dim_var = tk.IntVar(value=12)
        ttk.Spinbox(f, from_=4, to=64, textvariable=self.preprocess_dim_var, width=6).grid(
            row=row, column=3, sticky="w", padx=5)
        row += 1

        self._reg(ttk.Label(f, text=self._t("fc_dim")), "fc_dim").grid(
            row=row, column=0, sticky="w", padx=5)
        self.fc_dim_var = tk.IntVar(value=48)
        ttk.Spinbox(f, from_=16, to=256, textvariable=self.fc_dim_var, width=6).grid(
            row=row, column=1, sticky="w", padx=5)
        row += 1

        # Aux
        self._reg(ttk.Label(f, text=self._t("aux_profile")), "aux_profile").grid(
            row=row, column=0, sticky="w", padx=5)
        self.aux_profile_var = tk.StringVar(value="wd_7d_geometry")
        ttk.Combobox(f, textvariable=self.aux_profile_var, values=AUX_PROFILES,
                      width=18, state="readonly").grid(row=row, column=1, sticky="w", padx=5)
        self.disable_aux_var = tk.BooleanVar(value=False)
        self._reg(ttk.Checkbutton(f, text=self._t("disable_aux"),
                                   variable=self.disable_aux_var), "disable_aux").grid(
            row=row, column=2, columnspan=2, sticky="w", padx=5)
        row += 1

        self._reg(ttk.Label(f, text=self._t("aux_seq_len")), "aux_seq_len").grid(
            row=row, column=0, sticky="w", padx=5)
        self.aux_seq_len_var = tk.IntVar(value=15)
        ttk.Spinbox(f, from_=5, to=100, textvariable=self.aux_seq_len_var, width=6).grid(
            row=row, column=1, sticky="w", padx=5)
        self._reg(ttk.Label(f, text=self._t("aux_seq_batch")), "aux_seq_batch").grid(
            row=row, column=2, sticky="w", padx=5)
        self.aux_batch_var = tk.IntVar(value=256)
        ttk.Spinbox(f, from_=32, to=1024, increment=32,
                     textvariable=self.aux_batch_var, width=6).grid(
            row=row, column=3, sticky="w", padx=5)
        row += 1

        self._reg(ttk.Label(f, text=self._t("aux_grad_clip")), "aux_grad_clip").grid(
            row=row, column=0, sticky="w", padx=5)
        self.aux_grad_clip_var = tk.DoubleVar(value=0.5)
        ttk.Entry(f, textvariable=self.aux_grad_clip_var, width=8).grid(
            row=row, column=1, sticky="w", padx=5)
        row += 1

        # ── 6. Critic Profile ──
        row = self._separator(f, row, "sec_critic")

        self._reg(ttk.Label(f, text=self._t("critic_profile")), "critic_profile").grid(
            row=row, column=0, sticky="w", padx=5)
        self.critic_var = tk.StringVar(value="symmetric")
        ttk.Combobox(f, textvariable=self.critic_var, values=CRITIC_PROFILES,
                      width=15, state="readonly").grid(row=row, column=1, sticky="w", padx=5)
        self.critic_desc_label = ttk.Label(f, text=self._t("critic_desc_sym"), style="Desc.TLabel")
        self.critic_desc_label.grid(row=row, column=2, columnspan=2, sticky="w", padx=5)
        self.critic_var.trace_add("write", self._on_critic_changed)
        row += 1

        # ── 7. Sim-to-Real DR: LiDAR ──
        row = self._separator(f, row, "sec_lidar_dr")

        self.lidar_no_noise_var = tk.BooleanVar(value=False)
        self._reg(ttk.Checkbutton(f, text=self._t("lidar_no_noise"),
                                   variable=self.lidar_no_noise_var), "lidar_no_noise").grid(
            row=row, column=0, columnspan=2, sticky="w", padx=5)
        self.no_dr_var = tk.BooleanVar(value=False)
        self._reg(ttk.Checkbutton(f, text=self._t("no_domain_rand"),
                                   variable=self.no_dr_var), "no_domain_rand").grid(
            row=row, column=2, columnspan=2, sticky="w", padx=5)
        row += 1

        # L1: Per-Ray
        self._reg(ttk.Label(f, text=self._t("lidar_l1_header"), style="Header.TLabel"),
                  "lidar_l1_header").grid(row=row, column=0, columnspan=4, sticky="w", padx=15)
        row += 1

        self._reg(ttk.Label(f, text=self._t("displacement_std")), "displacement_std").grid(
            row=row, column=0, sticky="w", padx=15)
        self.disp_std_var = tk.DoubleVar(value=0.002)
        ttk.Entry(f, textvariable=self.disp_std_var, width=10).grid(
            row=row, column=1, sticky="w", padx=5)
        self._reg(ttk.Label(f, text=self._t("hole_rate")), "hole_rate").grid(
            row=row, column=2, sticky="w", padx=5)
        self.hole_rate_var = tk.DoubleVar(value=0.20)
        ttk.Entry(f, textvariable=self.hole_rate_var, width=10).grid(
            row=row, column=3, sticky="w", padx=5)
        row += 1

        self._reg(ttk.Label(f, text=self._t("distractor_rate")), "distractor_rate").grid(
            row=row, column=0, sticky="w", padx=15)
        self.distractor_var = tk.DoubleVar(value=0.002)
        ttk.Entry(f, textvariable=self.distractor_var, width=10).grid(
            row=row, column=1, sticky="w", padx=5)
        self.dist_bias_var = tk.BooleanVar(value=False)
        self._reg(ttk.Checkbutton(f, text=self._t("distance_bias"),
                                   variable=self.dist_bias_var), "distance_bias").grid(
            row=row, column=2, sticky="w", padx=5)
        self.ring_bias_var = tk.BooleanVar(value=False)
        self._reg(ttk.Checkbutton(f, text=self._t("per_ring_bias"),
                                   variable=self.ring_bias_var), "per_ring_bias").grid(
            row=row, column=3, sticky="w", padx=5)
        row += 1

        # L2: Per-Bin
        self._reg(ttk.Label(f, text=self._t("lidar_l2_header"), style="Header.TLabel"),
                  "lidar_l2_header").grid(row=row, column=0, columnspan=4, sticky="w", padx=15)
        row += 1

        self._reg(ttk.Label(f, text=self._t("obs_noise_std")), "obs_noise_std").grid(
            row=row, column=0, sticky="w", padx=15)
        self.obs_noise_var = tk.DoubleVar(value=0.005)
        ttk.Entry(f, textvariable=self.obs_noise_var, width=10).grid(
            row=row, column=1, sticky="w", padx=5)
        self._reg(ttk.Label(f, text=self._t("block_dropout")), "block_dropout").grid(
            row=row, column=2, sticky="w", padx=5)
        self.block_dropout_var = tk.DoubleVar(value=0.0)
        ttk.Entry(f, textvariable=self.block_dropout_var, width=10).grid(
            row=row, column=3, sticky="w", padx=5)
        row += 1

        self._reg(ttk.Label(f, text=self._t("block_width")), "block_width").grid(
            row=row, column=0, sticky="w", padx=15)
        bw_frame = ttk.Frame(f)
        bw_frame.grid(row=row, column=1, columnspan=2, sticky="w", padx=5)
        self.block_w_lo_var = tk.IntVar(value=3)
        self.block_w_hi_var = tk.IntVar(value=8)
        ttk.Spinbox(bw_frame, from_=1, to=36, textvariable=self.block_w_lo_var, width=4).pack(side="left")
        ttk.Label(bw_frame, text=" ~ ").pack(side="left")
        ttk.Spinbox(bw_frame, from_=1, to=36, textvariable=self.block_w_hi_var, width=4).pack(side="left")
        row += 1

        # L3: Per-Episode DR
        self._reg(ttk.Label(f, text=self._t("lidar_l3_header"), style="Header.TLabel"),
                  "lidar_l3_header").grid(row=row, column=0, columnspan=4, sticky="w", padx=15)
        row += 1

        self._reg(ttk.Label(f, text=self._t("disp_std_dr")), "disp_std_dr").grid(
            row=row, column=0, sticky="w", padx=15)
        dr_disp_frame = ttk.Frame(f)
        dr_disp_frame.grid(row=row, column=1, columnspan=2, sticky="w", padx=5)
        self.disp_dr_lo_var = tk.StringVar(value="")
        self.disp_dr_hi_var = tk.StringVar(value="")
        ttk.Entry(dr_disp_frame, textvariable=self.disp_dr_lo_var, width=8).pack(side="left")
        ttk.Label(dr_disp_frame, text=" ~ ").pack(side="left")
        ttk.Entry(dr_disp_frame, textvariable=self.disp_dr_hi_var, width=8).pack(side="left")
        ttk.Label(dr_disp_frame, text="  (empty=off)", foreground="gray").pack(side="left", padx=3)
        row += 1

        self._reg(ttk.Label(f, text=self._t("hole_rate_dr")), "hole_rate_dr").grid(
            row=row, column=0, sticky="w", padx=15)
        dr_hole_frame = ttk.Frame(f)
        dr_hole_frame.grid(row=row, column=1, columnspan=2, sticky="w", padx=5)
        self.hole_dr_lo_var = tk.StringVar(value="")
        self.hole_dr_hi_var = tk.StringVar(value="")
        ttk.Entry(dr_hole_frame, textvariable=self.hole_dr_lo_var, width=8).pack(side="left")
        ttk.Label(dr_hole_frame, text=" ~ ").pack(side="left")
        ttk.Entry(dr_hole_frame, textvariable=self.hole_dr_hi_var, width=8).pack(side="left")
        ttk.Label(dr_hole_frame, text="  (empty=off)", foreground="gray").pack(side="left", padx=3)
        row += 1

        # ── 8. Physics & Disturbance DR ──
        row = self._separator(f, row, "sec_physics_dr")

        # Physics
        self._reg(ttk.Label(f, text=self._t("mass_dr")), "mass_dr").grid(
            row=row, column=0, sticky="w", padx=5)
        mass_frame = ttk.Frame(f)
        mass_frame.grid(row=row, column=1, sticky="w", padx=5)
        self.mass_lo_var = tk.DoubleVar(value=0.85)
        self.mass_hi_var = tk.DoubleVar(value=1.15)
        ttk.Entry(mass_frame, textvariable=self.mass_lo_var, width=6).pack(side="left")
        ttk.Label(mass_frame, text=" ~ ").pack(side="left")
        ttk.Entry(mass_frame, textvariable=self.mass_hi_var, width=6).pack(side="left")

        self._reg(ttk.Label(f, text=self._t("friction_dr")), "friction_dr").grid(
            row=row, column=2, sticky="w", padx=5)
        fric_frame = ttk.Frame(f)
        fric_frame.grid(row=row, column=3, sticky="w", padx=5)
        self.fric_lo_var = tk.DoubleVar(value=0.7)
        self.fric_hi_var = tk.DoubleVar(value=1.3)
        ttk.Entry(fric_frame, textvariable=self.fric_lo_var, width=6).pack(side="left")
        ttk.Label(fric_frame, text=" ~ ").pack(side="left")
        ttk.Entry(fric_frame, textvariable=self.fric_hi_var, width=6).pack(side="left")
        row += 1

        self._reg(ttk.Label(f, text=self._t("com_offset")), "com_offset").grid(
            row=row, column=0, sticky="w", padx=5)
        self.com_offset_var = tk.DoubleVar(value=0.05)
        ttk.Entry(f, textvariable=self.com_offset_var, width=8).grid(
            row=row, column=1, sticky="w", padx=5)
        row += 1

        # Disturbance
        self._reg(ttk.Label(f, text=self._t("wind_force")), "wind_force").grid(
            row=row, column=0, sticky="w", padx=5)
        wind_frame = ttk.Frame(f)
        wind_frame.grid(row=row, column=1, sticky="w", padx=5)
        self.wind_lo_var = tk.DoubleVar(value=0.0)
        self.wind_hi_var = tk.DoubleVar(value=3.0)
        ttk.Entry(wind_frame, textvariable=self.wind_lo_var, width=6).pack(side="left")
        ttk.Label(wind_frame, text=" ~ ").pack(side="left")
        ttk.Entry(wind_frame, textvariable=self.wind_hi_var, width=6).pack(side="left")

        self._reg(ttk.Label(f, text=self._t("push_force")), "push_force").grid(
            row=row, column=2, sticky="w", padx=5)
        push_frame = ttk.Frame(f)
        push_frame.grid(row=row, column=3, sticky="w", padx=5)
        self.push_lo_var = tk.DoubleVar(value=5.0)
        self.push_hi_var = tk.DoubleVar(value=20.0)
        ttk.Entry(push_frame, textvariable=self.push_lo_var, width=6).pack(side="left")
        ttk.Label(push_frame, text=" ~ ").pack(side="left")
        ttk.Entry(push_frame, textvariable=self.push_hi_var, width=6).pack(side="left")
        row += 1

        self._reg(ttk.Label(f, text=self._t("push_ratio")), "push_ratio").grid(
            row=row, column=0, sticky="w", padx=5)
        self.push_ratio_var = tk.DoubleVar(value=0.10)
        ttk.Entry(f, textvariable=self.push_ratio_var, width=8).grid(
            row=row, column=1, sticky="w", padx=5)
        row += 1

        # Actuator DR
        self.actuator_var = tk.BooleanVar(value=False)
        self._reg(ttk.Checkbutton(f, text=self._t("enable_actuator"),
                                   variable=self.actuator_var, command=self._toggle_actuator),
                  "enable_actuator").grid(row=row, column=0, columnspan=2, sticky="w", padx=5)
        row += 1

        self.actuator_frame = ttk.LabelFrame(f, text="Actuator DR")
        self.actuator_frame.grid(row=row, column=0, columnspan=4, sticky="ew", padx=5, pady=2)
        af = self.actuator_frame

        self._reg(ttk.Label(af, text=self._t("actuator_delay")), "actuator_delay").grid(
            row=0, column=0, sticky="w", padx=5)
        delay_frame = ttk.Frame(af)
        delay_frame.grid(row=0, column=1, sticky="w", padx=5)
        self.act_delay_lo_var = tk.IntVar(value=0)
        self.act_delay_hi_var = tk.IntVar(value=2)
        ttk.Spinbox(delay_frame, from_=0, to=10, textvariable=self.act_delay_lo_var, width=3).pack(side="left")
        ttk.Label(delay_frame, text=" ~ ").pack(side="left")
        ttk.Spinbox(delay_frame, from_=0, to=10, textvariable=self.act_delay_hi_var, width=3).pack(side="left")

        self._reg(ttk.Label(af, text=self._t("velocity_scale")), "velocity_scale").grid(
            row=0, column=2, sticky="w", padx=5)
        vscale_frame = ttk.Frame(af)
        vscale_frame.grid(row=0, column=3, sticky="w", padx=5)
        self.vscale_lo_var = tk.DoubleVar(value=0.9)
        self.vscale_hi_var = tk.DoubleVar(value=1.1)
        ttk.Entry(vscale_frame, textvariable=self.vscale_lo_var, width=5).pack(side="left")
        ttk.Label(vscale_frame, text=" ~ ").pack(side="left")
        ttk.Entry(vscale_frame, textvariable=self.vscale_hi_var, width=5).pack(side="left")

        self._reg(ttk.Label(af, text=self._t("motor_lag")), "motor_lag").grid(
            row=1, column=0, sticky="w", padx=5)
        self.motor_lag_var = tk.DoubleVar(value=0.3)
        ttk.Entry(af, textvariable=self.motor_lag_var, width=6).grid(
            row=1, column=1, sticky="w", padx=5)

        self._toggle_actuator()
        row += 1

        # ── 9. Advanced DR ──
        row = self._separator(f, row, "sec_advanced_dr")

        self._reg(ttk.Label(f, text=self._t("obs_delay")), "obs_delay").grid(
            row=row, column=0, sticky="w", padx=5)
        od_frame = ttk.Frame(f)
        od_frame.grid(row=row, column=1, sticky="w", padx=5)
        self.obs_delay_lo_var = tk.IntVar(value=0)
        self.obs_delay_hi_var = tk.IntVar(value=0)
        ttk.Spinbox(od_frame, from_=0, to=5, textvariable=self.obs_delay_lo_var, width=3).pack(side="left")
        ttk.Label(od_frame, text=" ~ ").pack(side="left")
        ttk.Spinbox(od_frame, from_=0, to=5, textvariable=self.obs_delay_hi_var, width=3).pack(side="left")

        self._reg(ttk.Label(f, text=self._t("heading_weight")), "heading_weight").grid(
            row=row, column=2, sticky="w", padx=5)
        self.heading_wt_var = tk.DoubleVar(value=0.0)
        ttk.Entry(f, textvariable=self.heading_wt_var, width=8).grid(
            row=row, column=3, sticky="w", padx=5)
        row += 1

        # RGDR
        self.rgdr_var = tk.BooleanVar(value=False)
        self._reg(ttk.Checkbutton(f, text=self._t("rgdr_enabled"),
                                   variable=self.rgdr_var), "rgdr_enabled").grid(
            row=row, column=0, columnspan=2, sticky="w", padx=5)
        self._reg(ttk.Label(f, text=self._t("rgdr_clamp")), "rgdr_clamp").grid(
            row=row, column=2, sticky="w", padx=5)
        rgdr_frame = ttk.Frame(f)
        rgdr_frame.grid(row=row, column=3, sticky="w", padx=5)
        self.rgdr_lo_var = tk.DoubleVar(value=0.5)
        self.rgdr_hi_var = tk.DoubleVar(value=2.0)
        ttk.Entry(rgdr_frame, textvariable=self.rgdr_lo_var, width=5).pack(side="left")
        ttk.Label(rgdr_frame, text=" ~ ").pack(side="left")
        ttk.Entry(rgdr_frame, textvariable=self.rgdr_hi_var, width=5).pack(side="left")
        row += 1

        # DORAEMON
        self.doraemon_var = tk.BooleanVar(value=False)
        self._reg(ttk.Checkbutton(f, text=self._t("doraemon_enabled"),
                                   variable=self.doraemon_var, command=self._toggle_doraemon),
                  "doraemon_enabled").grid(row=row, column=0, columnspan=2, sticky="w", padx=5)
        row += 1

        self.doraemon_frame = ttk.LabelFrame(f, text="DORAEMON")
        self.doraemon_frame.grid(row=row, column=0, columnspan=4, sticky="ew", padx=5, pady=2)
        df = self.doraemon_frame
        self._reg(ttk.Label(df, text=self._t("doraemon_sr_th")), "doraemon_sr_th").grid(
            row=0, column=0, sticky="w", padx=5)
        self.dora_sr_var = tk.DoubleVar(value=0.85)
        ttk.Entry(df, textvariable=self.dora_sr_var, width=6).grid(
            row=0, column=1, sticky="w", padx=5)
        self._reg(ttk.Label(df, text=self._t("doraemon_interval")), "doraemon_interval").grid(
            row=0, column=2, sticky="w", padx=5)
        self.dora_interval_var = tk.IntVar(value=50)
        ttk.Spinbox(df, from_=10, to=500, textvariable=self.dora_interval_var, width=6).grid(
            row=0, column=3, sticky="w", padx=5)
        self._reg(ttk.Label(df, text=self._t("doraemon_rate")), "doraemon_rate").grid(
            row=1, column=0, sticky="w", padx=5)
        self.dora_rate_var = tk.DoubleVar(value=0.1)
        ttk.Entry(df, textvariable=self.dora_rate_var, width=6).grid(
            row=1, column=1, sticky="w", padx=5)

        self._toggle_doraemon()
        row += 1

        # ── 10. Logging ──
        row = self._separator(f, row, "sec_logging")

        self._reg(ttk.Label(f, text=self._t("log_interval")), "log_interval").grid(
            row=row, column=0, sticky="w", padx=5)
        self.log_interval_var = tk.IntVar(value=10)
        ttk.Spinbox(f, from_=1, to=100, textvariable=self.log_interval_var, width=6).grid(
            row=row, column=1, sticky="w", padx=5)
        self._reg(ttk.Label(f, text=self._t("save_interval")), "save_interval").grid(
            row=row, column=2, sticky="w", padx=5)
        self.save_interval_var = tk.IntVar(value=100)
        ttk.Spinbox(f, from_=10, to=1000, increment=10,
                     textvariable=self.save_interval_var, width=6).grid(
            row=row, column=3, sticky="w", padx=5)
        row += 1

        self._reg(ttk.Label(f, text=self._t("wandb_notes")), "wandb_notes").grid(
            row=row, column=0, sticky="w", padx=5)
        self.wandb_notes_var = tk.StringVar(value="")
        ttk.Entry(f, textvariable=self.wandb_notes_var, width=60).grid(
            row=row, column=1, columnspan=3, sticky="w", padx=5)
        row += 1

        self._reg(ttk.Label(f, text=self._t("extra_args")), "extra_args").grid(
            row=row, column=0, sticky="w", padx=5)
        self.extra_args_var = tk.StringVar(value="")
        ttk.Entry(f, textvariable=self.extra_args_var, width=60).grid(
            row=row, column=1, columnspan=3, sticky="w", padx=5)
        row += 1

        # ── 11. Command Preview ──
        row = self._separator(f, row, "sec_preview")

        self.cmd_text = tk.Text(f, height=10, width=110, font=("Courier", 11), wrap="word")
        self.cmd_text.grid(row=row, column=0, columnspan=4, sticky="ew", padx=5)
        row += 1

        # Buttons
        btn_frame = ttk.Frame(f)
        btn_frame.grid(row=row, column=0, columnspan=4, pady=15)

        self._reg(ttk.Button(btn_frame, text=self._t("preview_cmd"),
                             command=self._preview_cmd), "preview_cmd").pack(side="left", padx=8)
        self._reg(ttk.Button(btn_frame, text=self._t("copy_clipboard"),
                             command=self._copy_cmd), "copy_clipboard").pack(side="left", padx=8)
        self._reg(ttk.Button(btn_frame, text=self._t("launch_train"),
                             command=self._launch), "launch_train").pack(side="left", padx=8)
        self._reg(ttk.Button(btn_frame, text=self._t("kill_train"),
                             command=self._kill_train), "kill_train").pack(side="left", padx=8)
        self._reg(ttk.Button(btn_frame, text=self._t("save_settings"),
                             command=self._save_settings), "save_settings").pack(side="left", padx=8)
        self._reg(ttk.Button(btn_frame, text=self._t("reset_from_yaml"),
                             command=self._load_yaml_btn), "reset_from_yaml").pack(side="left", padx=8)

        self._load_settings()
        self._preview_cmd()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _separator(self, parent, row: int, key: str) -> int:
        ttk.Separator(parent, orient="horizontal").grid(
            row=row, column=0, columnspan=4, sticky="ew", pady=10)
        row += 1
        self._reg(ttk.Label(parent, text=self._t(key), style="Header.TLabel"), key).grid(
            row=row, column=0, columnspan=4, sticky="w", padx=5, pady=(0, 5))
        return row + 1

    def _on_sa_changed(self, *_args):
        try:
            stage = self.sa_stage_var.get()
        except (tk.TclError, ValueError):
            return
        self.sa_desc_var.set(SA_DESCRIPTIONS.get(stage, ""))
        self._update_stage_info()

    def _update_stage_info(self):
        try:
            stage = self.sa_stage_var.get()
        except (tk.TclError, ValueError):
            return
        info = format_stage_info(stage, self.lang)
        self.stage_info_text.configure(state="normal")
        self.stage_info_text.delete("1.0", "end")
        if info:
            self.stage_info_text.insert("1.0", info)
        else:
            no_data = "（無課程資料）" if self.lang == 1 else "(no curriculum data)"
            self.stage_info_text.insert("1.0", no_data)
        self.stage_info_text.configure(state="disabled")

    def _on_algo_changed(self, *_args):
        is_ppo = self.algo_var.get() == "ppo"
        state = "normal" if is_ppo else "disabled"
        for child in self.ppo_frame.winfo_children():
            if isinstance(child, (ttk.Spinbox, ttk.Entry)):
                child.configure(state=state)

    def _on_critic_changed(self, *_args):
        if self.critic_var.get() == "asymmetric":
            self.critic_desc_label.configure(text=self._t("critic_desc_asym"))
        else:
            self.critic_desc_label.configure(text=self._t("critic_desc_sym"))

    def _toggle_actuator(self):
        state = "normal" if self.actuator_var.get() else "disabled"
        for child in self.actuator_frame.winfo_children():
            if isinstance(child, (ttk.Spinbox, ttk.Entry)):
                child.configure(state=state)
            elif isinstance(child, ttk.Frame):
                for sub in child.winfo_children():
                    if isinstance(sub, (ttk.Spinbox, ttk.Entry)):
                        sub.configure(state=state)

    def _toggle_doraemon(self):
        state = "normal" if self.doraemon_var.get() else "disabled"
        for child in self.doraemon_frame.winfo_children():
            if isinstance(child, (ttk.Spinbox, ttk.Entry)):
                child.configure(state=state)

    def _on_config_selected(self, _event):
        self._load_yaml_btn()

    def _on_run_selected(self, _event):
        run_name = self.run_var.get()
        run_dir = LOGS_DIR / run_name
        if run_dir.exists():
            pts = sorted(run_dir.glob("checkpoint_*.pt"), reverse=True)
            ckpt_list = [str(p.relative_to(REPO_ROOT)) for p in pts]
            self.ckpt_cb["values"] = ckpt_list
            if ckpt_list:
                self.ckpt_var.set(ckpt_list[0])

    def _browse_ckpt(self):
        path = filedialog.askopenfilename(
            initialdir=str(LOGS_DIR),
            title=self._t("select_ckpt"),
            filetypes=[("PyTorch checkpoint", "*.pt")],
        )
        if path:
            try:
                rel = str(Path(path).relative_to(REPO_ROOT))
            except ValueError:
                rel = path
            self.ckpt_var.set(rel)

    def _load_yaml_btn(self):
        name = self.config_var.get()
        if not name:
            return
        data = load_yaml_config(name)
        if data is None:
            messagebox.showerror(self._t("error_title"), f"Failed to load {name}.yaml")
            return
        self._apply_yaml(data)
        desc = data.get("description", "")
        self.config_desc_var.set(desc)

    def _apply_yaml(self, data: dict):
        _map = {
            "initial_stage": self.sa_stage_var,
            "fixed_stage": self.fixed_stage_var,
            "task": self.task_var,
            "num_envs": self.num_envs_var,
            "timesteps": self.timesteps_var,
            "rollout_length": self.rollout_var,
            "seed": self.seed_var,
            "algorithm": self.algo_var,
            "lr": self.lr_var,
            "rnn_lr": self.rnn_lr_var,
            "gamma": self.gamma_var,
            "gae_lambda": self.gae_lambda_var,
            "vf_coeff": self.vf_coeff_var,
            "max_grad_norm": self.max_grad_norm_var,
            "normalize_return": self.normalize_return_var,
            "adv_norm_mode": self.adv_norm_var,
            "ent_coeff_linear": self.ent_linear_var,
            "ent_coeff_angular": self.ent_angular_var,
            "lr_decay": self.lr_decay_var,
            "ppo_epochs": self.ppo_epochs_var,
            "mini_batches": self.mini_batches_var,
            "clip_eps": self.clip_eps_var,
            "wd_update_clip": self.wd_clip_var,
            "wd_actor_update_clip": self.wd_actor_clip_var,
            "wd_critic_update_clip": self.wd_critic_clip_var,
            "policy_loss_clamp": self.policy_clamp_var,
            "vf_term_clamp": self.vf_clamp_var,
            "charge_encoder_mode": self.encoder_var,
            "encoder_profile": self.encoder_var,
            "rnn_type": self.rnn_type_var,
            "hidden_dim": self.hidden_dim_var,
            "preprocess_dim": self.preprocess_dim_var,
            "fc_dim": self.fc_dim_var,
            "aux_profile": self.aux_profile_var,
            "aux_seq_len": self.aux_seq_len_var,
            "aux_seq_batch_size": self.aux_batch_var,
            "aux_grad_clip": self.aux_grad_clip_var,
            "critic_profile": self.critic_var,
            "lidar_no_noise": self.lidar_no_noise_var,
            "no_domain_randomization": self.no_dr_var,
            "lidar_displacement_std": self.disp_std_var,
            "lidar_hole_rate": self.hole_rate_var,
            "lidar_distractor_rate": self.distractor_var,
            "lidar_distance_bias": self.dist_bias_var,
            "lidar_per_ring_bias": self.ring_bias_var,
            "lidar_obs_noise_std": self.obs_noise_var,
            "lidar_block_dropout_prob": self.block_dropout_var,
            "physics_com_offset": self.com_offset_var,
            "disturbance_push_ratio": self.push_ratio_var,
            "enable_actuator_dr": self.actuator_var,
            "actuator_motor_lag": self.motor_lag_var,
            "heading_stability_weight": self.heading_wt_var,
            "rgdr_enabled": self.rgdr_var,
            "doraemon_enabled": self.doraemon_var,
            "doraemon_sr_threshold": self.dora_sr_var,
            "doraemon_check_interval": self.dora_interval_var,
            "doraemon_expansion_rate": self.dora_rate_var,
            "no_resume_optimizer": self.no_resume_opt_var,
            "log_interval": self.log_interval_var,
            "save_interval": self.save_interval_var,
            "checkpoint": self.ckpt_var,
        }

        for key, var in _map.items():
            if key in data and data[key] is not None:
                try:
                    var.set(data[key])
                except Exception:
                    pass

        # Range fields
        def _set_range(key, lo_var, hi_var, is_int=False):
            val = data.get(key)
            if isinstance(val, (list, tuple)) and len(val) == 2:
                lo_var.set(int(val[0]) if is_int else val[0])
                hi_var.set(int(val[1]) if is_int else val[1])

        _set_range("lidar_block_dropout_width", self.block_w_lo_var, self.block_w_hi_var, is_int=True)
        _set_range("physics_mass_dr", self.mass_lo_var, self.mass_hi_var)
        _set_range("physics_friction_dr", self.fric_lo_var, self.fric_hi_var)
        _set_range("disturbance_wind_force", self.wind_lo_var, self.wind_hi_var)
        _set_range("disturbance_push_force", self.push_lo_var, self.push_hi_var)
        _set_range("obs_delay_steps", self.obs_delay_lo_var, self.obs_delay_hi_var, is_int=True)
        _set_range("actuator_delay_range", self.act_delay_lo_var, self.act_delay_hi_var, is_int=True)
        _set_range("actuator_velocity_scale", self.vscale_lo_var, self.vscale_hi_var)
        _set_range("rgdr_weight_clamp", self.rgdr_lo_var, self.rgdr_hi_var)

        # Per-Episode DR (None = off → clear entries)
        disp_dr = data.get("lidar_displacement_std_dr")
        if isinstance(disp_dr, (list, tuple)) and len(disp_dr) == 2:
            self.disp_dr_lo_var.set(str(disp_dr[0]))
            self.disp_dr_hi_var.set(str(disp_dr[1]))
        else:
            self.disp_dr_lo_var.set("")
            self.disp_dr_hi_var.set("")

        hole_dr = data.get("lidar_hole_rate_dr")
        if isinstance(hole_dr, (list, tuple)) and len(hole_dr) == 2:
            self.hole_dr_lo_var.set(str(hole_dr[0]))
            self.hole_dr_hi_var.set(str(hole_dr[1]))
        else:
            self.hole_dr_lo_var.set("")
            self.hole_dr_hi_var.set("")

        self._on_algo_changed()
        self._toggle_actuator()
        self._toggle_doraemon()
        self._on_critic_changed()
        self._on_sa_changed()

    # ------------------------------------------------------------------
    # Command builder
    # ------------------------------------------------------------------

    def _build_gui_yaml(self) -> dict:
        """Collect ALL GUI values into an ExperimentConfig-compatible dict."""
        cfg_name = self.config_var.get() or "gui_launch"
        run_name = self.run_name_var.get().strip()
        d = {
            "name": f"_gui_{cfg_name}",
            "description": f"Generated by Train Launcher from {cfg_name}",
            # Environment / curriculum
            "task": self.task_var.get(),
            "curriculum_version": "warp_drive_single_agent_v1",
            "initial_stage": self.sa_stage_var.get(),
            "fixed_stage": self.fixed_stage_var.get(),
            "scene_profile": "warp_drive_single_agent_v1",
            "obstacle_mode": "rule_based",
            # Profiles
            "reward_profile": "wd_sparse",
            "algorithm": self.algo_var.get(),
            "aux_profile": self.aux_profile_var.get(),
            "encoder_profile": self.encoder_var.get(),
            "critic_profile": self.critic_var.get(),
            # Training budget
            "num_envs": self.num_envs_var.get(),
            "rollout_length": self.rollout_var.get(),
            "timesteps": self.timesteps_var.get(),
            "seed": self.seed_var.get(),
            # RL hyperparams
            "lr": self.lr_var.get(),
            "rnn_lr": self.rnn_lr_var.get(),
            "vf_coeff": self.vf_coeff_var.get(),
            "gamma": self.gamma_var.get(),
            "gae_lambda": self.gae_lambda_var.get(),
            "normalize_return": self.normalize_return_var.get(),
            "adv_norm_mode": self.adv_norm_var.get(),
            "max_grad_norm": self.max_grad_norm_var.get(),
            "lr_decay": self.lr_decay_var.get(),
            "ent_coeff_linear": self.ent_linear_var.get(),
            "ent_coeff_angular": self.ent_angular_var.get(),
            # PPO
            "ppo_epochs": self.ppo_epochs_var.get(),
            "mini_batches": self.mini_batches_var.get(),
            "clip_eps": self.clip_eps_var.get(),
            # WD caps
            "wd_update_clip": self.wd_clip_var.get(),
            "wd_actor_update_clip": self.wd_actor_clip_var.get(),
            "wd_critic_update_clip": self.wd_critic_clip_var.get(),
            "policy_loss_clamp": self.policy_clamp_var.get(),
            "vf_term_clamp": self.vf_clamp_var.get(),
            # Model
            "charge_encoder_mode": self.encoder_var.get(),
            "hidden_dim": self.hidden_dim_var.get(),
            "preprocess_dim": self.preprocess_dim_var.get(),
            "fc_dim": self.fc_dim_var.get(),
            "rnn_type": self.rnn_type_var.get(),
            # Aux
            "aux_seq_len": self.aux_seq_len_var.get(),
            "aux_seq_batch_size": self.aux_batch_var.get(),
            "aux_grad_clip": self.aux_grad_clip_var.get(),
            # LiDAR DR
            "lidar_no_noise": self.lidar_no_noise_var.get(),
            "lidar_displacement_std": self.disp_std_var.get(),
            "lidar_hole_rate": self.hole_rate_var.get(),
            "lidar_distractor_rate": self.distractor_var.get(),
            "lidar_distance_bias": self.dist_bias_var.get(),
            "lidar_per_ring_bias": self.ring_bias_var.get(),
            "lidar_obs_noise_std": self.obs_noise_var.get(),
            "lidar_block_dropout_prob": self.block_dropout_var.get(),
            "lidar_block_dropout_width": [self.block_w_lo_var.get(), self.block_w_hi_var.get()],
            # Physics DR
            "physics_mass_dr": [self.mass_lo_var.get(), self.mass_hi_var.get()],
            "physics_friction_dr": [self.fric_lo_var.get(), self.fric_hi_var.get()],
            "physics_com_offset": self.com_offset_var.get(),
            # Disturbance DR
            "disturbance_wind_force": [self.wind_lo_var.get(), self.wind_hi_var.get()],
            "disturbance_push_force": [self.push_lo_var.get(), self.push_hi_var.get()],
            "disturbance_push_ratio": self.push_ratio_var.get(),
            # Actuator DR
            "enable_actuator_dr": self.actuator_var.get(),
            "actuator_delay_range": [self.act_delay_lo_var.get(), self.act_delay_hi_var.get()],
            "actuator_velocity_scale": [self.vscale_lo_var.get(), self.vscale_hi_var.get()],
            "actuator_motor_lag": self.motor_lag_var.get(),
            # Advanced DR
            "obs_delay_steps": [self.obs_delay_lo_var.get(), self.obs_delay_hi_var.get()],
            "heading_stability_weight": self.heading_wt_var.get(),
            "rgdr_enabled": self.rgdr_var.get(),
            "rgdr_weight_clamp": [self.rgdr_lo_var.get(), self.rgdr_hi_var.get()],
            "doraemon_enabled": self.doraemon_var.get(),
            "doraemon_sr_threshold": self.dora_sr_var.get(),
            "doraemon_check_interval": self.dora_interval_var.get(),
            "doraemon_expansion_rate": self.dora_rate_var.get(),
            # No-DR
            "no_domain_randomization": self.no_dr_var.get(),
            # Checkpoint
            "no_resume_optimizer": self.no_resume_opt_var.get(),
            # Logging
            "log_interval": self.log_interval_var.get(),
            "save_interval": self.save_interval_var.get(),
            # Metadata
            "tags": [self.algo_var.get(), f"sa{self.sa_stage_var.get()}", "gui_launch"],
            "notes": run_name or f"GUI launch: {cfg_name}",
        }

        # Per-Episode DR (None if empty)
        dlo = self.disp_dr_lo_var.get().strip()
        dhi = self.disp_dr_hi_var.get().strip()
        if dlo and dhi:
            d["lidar_displacement_std_dr"] = [float(dlo), float(dhi)]

        hlo = self.hole_dr_lo_var.get().strip()
        hhi = self.hole_dr_hi_var.get().strip()
        if hlo and hhi:
            d["lidar_hole_rate_dr"] = [float(hlo), float(hhi)]

        # Checkpoint path
        ckpt = self.ckpt_var.get().strip()
        if ckpt:
            d["checkpoint"] = ckpt

        return d

    def _write_gui_yaml(self) -> str:
        """Write current GUI state to _gui_launch.yaml, return config name."""
        d = self._build_gui_yaml()
        yaml_path = CONFIGS_DIR / "_gui_launch.yaml"

        lines = [f"# Auto-generated by Train Launcher — do not edit manually"]
        for key, val in d.items():
            if val is None:
                lines.append(f"{key}: null")
            elif isinstance(val, bool):
                lines.append(f"{key}: {'true' if val else 'false'}")
            elif isinstance(val, list):
                lines.append(f"{key}: [{', '.join(str(v) for v in val)}]")
            elif isinstance(val, str):
                if any(c in val for c in ":{}[]#,"):
                    lines.append(f'{key}: "{val}"')
                else:
                    lines.append(f"{key}: {val}")
            else:
                lines.append(f"{key}: {val}")
        yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return "_gui_launch"

    def _build_command(self) -> str:
        gui_cfg_name = self._write_gui_yaml()

        parts = [
            "PYTHONUNBUFFERED=1",
            "./isaaclab.sh -p",
            str(TRAIN_SCRIPT.relative_to(REPO_ROOT)),
        ]

        # The YAML carries ALL params (including DR that has no argparse)
        parts.append(f"--experiment_config {gui_cfg_name}")

        # CLI flags below are for argparse-registered args only (override YAML)
        parts.append(f"--task {self.task_var.get()}")
        parts.append(f"--initial_stage {self.sa_stage_var.get()}")
        if self.fixed_stage_var.get():
            parts.append("--fixed_stage")

        # Checkpoint
        ckpt = self.ckpt_var.get().strip()
        if ckpt:
            parts.append(f"--checkpoint {ckpt}")
        if self.no_resume_opt_var.get():
            parts.append("--no_resume_optimizer")

        # Budget
        parts.append(f"--num_envs {self.num_envs_var.get()}")
        parts.append(f"--timesteps {self.timesteps_var.get()}")
        parts.append(f"--rollout_length {self.rollout_var.get()}")
        parts.append(f"--seed {self.seed_var.get()}")
        if self.headless_var.get():
            parts.append("--headless")

        run_name = self.run_name_var.get().strip()
        if run_name:
            parts.append(f"--run_name {run_name}")

        # Algorithm
        if self.algo_var.get() == "ppo":
            parts.append("--ppo")
            parts.append(f"--ppo_epochs {self.ppo_epochs_var.get()}")
            parts.append(f"--mini_batches {self.mini_batches_var.get()}")
            parts.append(f"--clip_eps {self.clip_eps_var.get()}")
        else:
            parts.append("--a2c")

        # RL hyperparams (all have add_argument)
        parts.append(f"--lr {self.lr_var.get()}")
        parts.append(f"--rnn_lr {self.rnn_lr_var.get()}")
        parts.append(f"--gamma {self.gamma_var.get()}")
        parts.append(f"--gae_lambda {self.gae_lambda_var.get()}")
        parts.append(f"--vf_coeff {self.vf_coeff_var.get()}")
        parts.append(f"--max_grad_norm {self.max_grad_norm_var.get()}")
        parts.append(f"--adv_norm_mode {self.adv_norm_var.get()}")
        if self.normalize_return_var.get():
            parts.append("--normalize_return")

        ent_l = self.ent_linear_var.get()
        ent_a = self.ent_angular_var.get()
        if ent_l != 0.0:
            parts.append(f"--ent_coeff_linear {ent_l}")
        if ent_a != 0.0:
            parts.append(f"--ent_coeff_angular {ent_a}")
        lr_d = self.lr_decay_var.get()
        if lr_d != 0.0:
            parts.append(f"--lr_decay {lr_d}")

        # WD caps
        if self.wd_clip_var.get():
            parts.append("--wd_update_clip")
            parts.append(f"--wd_actor_update_clip {self.wd_actor_clip_var.get()}")
            parts.append(f"--wd_critic_update_clip {self.wd_critic_clip_var.get()}")
        else:
            parts.append("--no_wd_update_clip")
        parts.append(f"--policy_loss_clamp {self.policy_clamp_var.get()}")
        parts.append(f"--vf_term_clamp {self.vf_clamp_var.get()}")

        # Model (all have add_argument)
        parts.append(f"--charge_encoder_mode {self.encoder_var.get()}")
        parts.append(f"--rnn_type {self.rnn_type_var.get()}")
        parts.append(f"--hidden_dim {self.hidden_dim_var.get()}")
        parts.append(f"--preprocess_dim {self.preprocess_dim_var.get()}")
        parts.append(f"--fc_dim {self.fc_dim_var.get()}")

        # Aux
        parts.append(f"--aux_profile {self.aux_profile_var.get()}")
        if self.disable_aux_var.get():
            parts.append("--disable_aux_training")
        parts.append(f"--aux_seq_len {self.aux_seq_len_var.get()}")
        parts.append(f"--aux_seq_batch_size {self.aux_batch_var.get()}")
        agc = self.aux_grad_clip_var.get()
        if agc > 0:
            parts.append(f"--aux_grad_clip {agc}")

        # Critic
        parts.append(f"--critic_profile {self.critic_var.get()}")

        # LiDAR noise toggle (has add_argument)
        if self.lidar_no_noise_var.get():
            parts.append("--lidar_no_noise")
        if self.no_dr_var.get():
            parts.append("--no_domain_randomization")

        # DR params: all via YAML (_gui_launch.yaml) — no CLI flags needed

        # Logging
        parts.append(f"--log_interval {self.log_interval_var.get()}")
        parts.append(f"--save_interval {self.save_interval_var.get()}")
        notes = self.wandb_notes_var.get().strip()
        if notes:
            parts.append(f'--wandb_notes "{notes}"')

        extra = self.extra_args_var.get().strip()
        if extra:
            parts.append(extra)

        return " \\\n  ".join(parts)

    def _preview_cmd(self):
        cmd = self._build_command()
        self.cmd_text.delete("1.0", tk.END)
        self.cmd_text.insert("1.0", cmd)

    def _copy_cmd(self):
        self._preview_cmd()
        cmd = self.cmd_text.get("1.0", tk.END).strip()
        self.root.clipboard_clear()
        self.root.clipboard_append(cmd)
        messagebox.showinfo(self._t("copied_title"), self._t("copied_msg"))

    def _launch(self):
        self._preview_cmd()
        cmd = self._build_command().replace(" \\\n  ", " ")

        result = messagebox.askyesno(
            self._t("launch_title"),
            f"{self._t('launch_confirm')}\n\n{cmd[:400]}...",
        )
        if not result:
            return

        self._save_settings_silent()

        try:
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            exec_cmd = cmd.replace("PYTHONUNBUFFERED=1 ", "")
            subprocess.Popen(exec_cmd, shell=True, cwd=str(REPO_ROOT), env=env)
            messagebox.showinfo(self._t("launched_title"), self._t("launched_msg"))
        except Exception as e:
            messagebox.showerror(self._t("error_title"), f"{self._t('error_launch')}\n{e}")

    def _kill_train(self):
        import signal
        result = messagebox.askyesno(self._t("kill_title"), self._t("kill_confirm"))
        if not result:
            return
        try:
            out = subprocess.check_output(
                ["pgrep", "-f", "train_rnn_car"], text=True).strip()
            pids = [int(p) for p in out.splitlines() if p.strip()]
        except subprocess.CalledProcessError:
            pids = []
        my_pid = os.getpid()
        pids = [p for p in pids if p != my_pid]
        if not pids:
            messagebox.showinfo(self._t("kill_title"), self._t("kill_none"))
            return
        killed = 0
        for pid in pids:
            try:
                os.kill(pid, signal.SIGTERM)
                killed += 1
            except (ProcessLookupError, PermissionError):
                pass
        self.root.after(1500, lambda: self._sigkill_remaining(pids))
        messagebox.showinfo(self._t("kill_title"), self._t("kill_done").format(n=killed))

    def _sigkill_remaining(self, pids: list[int]):
        import signal
        for pid in pids:
            try:
                os.kill(pid, 0)
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass

    # ------------------------------------------------------------------
    # Settings persistence
    # ------------------------------------------------------------------

    _SETTINGS_FILE = REPO_ROOT / ".train_launcher_settings.json"

    _SETTINGS_KEYS: list[tuple[str, str]] = [
        # Config
        ("config_var", "str"),
        ("sa_stage_var", "int"),
        ("fixed_stage_var", "bool"),
        # Resume
        ("run_var", "str"),
        ("ckpt_var", "str"),
        ("no_resume_opt_var", "bool"),
        # Budget
        ("task_var", "str"),
        ("run_name_var", "str"),
        ("num_envs_var", "int"),
        ("timesteps_var", "int"),
        ("rollout_var", "int"),
        ("seed_var", "int"),
        ("headless_var", "bool"),
        # Algorithm
        ("algo_var", "str"),
        ("lr_var", "float"),
        ("rnn_lr_var", "float"),
        ("gamma_var", "float"),
        ("gae_lambda_var", "float"),
        ("vf_coeff_var", "float"),
        ("max_grad_norm_var", "float"),
        ("normalize_return_var", "bool"),
        ("adv_norm_var", "str"),
        ("ent_linear_var", "float"),
        ("ent_angular_var", "float"),
        ("lr_decay_var", "float"),
        ("ppo_epochs_var", "int"),
        ("mini_batches_var", "int"),
        ("clip_eps_var", "float"),
        ("wd_clip_var", "bool"),
        ("wd_actor_clip_var", "float"),
        ("wd_critic_clip_var", "float"),
        ("policy_clamp_var", "float"),
        ("vf_clamp_var", "float"),
        # Model
        ("encoder_var", "str"),
        ("rnn_type_var", "str"),
        ("hidden_dim_var", "int"),
        ("preprocess_dim_var", "int"),
        ("fc_dim_var", "int"),
        ("aux_profile_var", "str"),
        ("disable_aux_var", "bool"),
        ("aux_seq_len_var", "int"),
        ("aux_batch_var", "int"),
        ("aux_grad_clip_var", "float"),
        # Critic
        ("critic_var", "str"),
        # LiDAR DR
        ("lidar_no_noise_var", "bool"),
        ("no_dr_var", "bool"),
        ("disp_std_var", "float"),
        ("hole_rate_var", "float"),
        ("distractor_var", "float"),
        ("dist_bias_var", "bool"),
        ("ring_bias_var", "bool"),
        ("obs_noise_var", "float"),
        ("block_dropout_var", "float"),
        ("block_w_lo_var", "int"),
        ("block_w_hi_var", "int"),
        ("disp_dr_lo_var", "str"),
        ("disp_dr_hi_var", "str"),
        ("hole_dr_lo_var", "str"),
        ("hole_dr_hi_var", "str"),
        # Physics DR
        ("mass_lo_var", "float"),
        ("mass_hi_var", "float"),
        ("fric_lo_var", "float"),
        ("fric_hi_var", "float"),
        ("com_offset_var", "float"),
        ("wind_lo_var", "float"),
        ("wind_hi_var", "float"),
        ("push_lo_var", "float"),
        ("push_hi_var", "float"),
        ("push_ratio_var", "float"),
        # Actuator DR
        ("actuator_var", "bool"),
        ("act_delay_lo_var", "int"),
        ("act_delay_hi_var", "int"),
        ("vscale_lo_var", "float"),
        ("vscale_hi_var", "float"),
        ("motor_lag_var", "float"),
        # Advanced DR
        ("obs_delay_lo_var", "int"),
        ("obs_delay_hi_var", "int"),
        ("heading_wt_var", "float"),
        ("rgdr_var", "bool"),
        ("rgdr_lo_var", "float"),
        ("rgdr_hi_var", "float"),
        ("doraemon_var", "bool"),
        ("dora_sr_var", "float"),
        ("dora_interval_var", "int"),
        ("dora_rate_var", "float"),
        # Logging
        ("log_interval_var", "int"),
        ("save_interval_var", "int"),
        ("wandb_notes_var", "str"),
        ("extra_args_var", "str"),
    ]

    def _collect_settings(self) -> dict:
        data = {}
        for attr, vtype in self._SETTINGS_KEYS:
            var = getattr(self, attr, None)
            if var is None:
                continue
            try:
                data[attr] = var.get()
            except Exception:
                pass
        return data

    def _save_settings_silent(self):
        try:
            data = self._collect_settings()
            self._SETTINGS_FILE.write_text(
                json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    def _save_settings(self):
        data = self._collect_settings()
        try:
            self._SETTINGS_FILE.write_text(
                json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            messagebox.showinfo(self._t("saved_title"), self._t("saved_msg"))
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save settings:\n{e}")

    def _load_settings(self):
        if not self._SETTINGS_FILE.exists():
            return
        try:
            data = json.loads(self._SETTINGS_FILE.read_text(encoding="utf-8"))
        except Exception:
            return
        for attr, vtype in self._SETTINGS_KEYS:
            if attr not in data:
                continue
            var = getattr(self, attr, None)
            if var is None:
                continue
            try:
                val = data[attr]
                if vtype == "int":
                    var.set(int(val))
                elif vtype == "float":
                    var.set(float(val))
                elif vtype == "bool":
                    var.set(bool(val))
                else:
                    var.set(str(val))
            except Exception:
                pass

        run_name = data.get("run_var", "")
        if run_name:
            run_dir = LOGS_DIR / run_name
            if run_dir.exists():
                pts = sorted(run_dir.glob("checkpoint_*.pt"), reverse=True)
                ckpt_list = [str(p.relative_to(REPO_ROOT)) for p in pts]
                self.ckpt_cb["values"] = ckpt_list

        self._on_algo_changed()
        self._toggle_actuator()
        self._toggle_doraemon()
        self._on_critic_changed()
        self._on_sa_changed()


# ============================================================================
# Main
# ============================================================================

def main():
    root = tk.Tk()
    TrainLauncherApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
