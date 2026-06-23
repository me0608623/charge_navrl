#!/usr/bin/env python3
"""Play Launcher GUI — 互動式設定 play_rnn_car.py 參數並開始

Usage (推薦，中文字型正常):
    python3 scripts/reinforcement_learning/skrl/play_eval/play_launcher.py

Note:
    不要用 ./isaaclab.sh -p 啟動本 GUI，因為 conda Tk 沒有 Xft
    無法渲染 CJK 字型。直接用系統 python3 即可。
"""
from __future__ import annotations

import sys
import os

# --- Conda Tk 沒有 Xft，CJK 會亂碼 → 自動用系統 Python 重啟 ---
# 偵測方式：sys.executable 是否在 conda/miniconda 路徑下
_SYSTEM_PYTHON = "/usr/bin/python3"
_is_conda = "conda" in sys.executable.lower() or "miniconda" in sys.executable.lower()
if (
    _is_conda
    and os.path.exists(_SYSTEM_PYTHON)
    and os.path.realpath(sys.executable) != os.path.realpath(_SYSTEM_PYTHON)
    and not os.environ.get("_PLAY_LAUNCHER_SYSPY")
):
    os.environ["_PLAY_LAUNCHER_SYSPY"] = "1"
    os.execv(_SYSTEM_PYTHON, [_SYSTEM_PYTHON] + sys.argv)

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
PLAY_SCRIPT = (
    REPO_ROOT
    / "scripts"
    / "reinforcement_learning"
    / "skrl"
    / "play_eval"
    / "play_rnn_car.py"
)

TASKS = [
    "Isaac-Navigation-Charge-VLP16-Curriculum-WD",
    "Isaac-Navigation-Charge-VLP16-Curriculum-WD-TCorridor",
    "Isaac-Navigation-Charge-VLP16-Curriculum-NavRL",
    "Isaac-Navigation-Charge-VLP16-Curriculum-NavRL-TCorridor",
    "Isaac-Navigation-Charge-VLP16-Curriculum-NavRL-Play",
    "Isaac-Navigation-Charge-VLP16-Curriculum-NavRL-Play-TCorridor",
]

# ============================================================================
# i18n — (English, 繁體中文)
# ============================================================================

_STRINGS: dict[str, tuple[str, str]] = {
    # Window
    "title": ("Charge Play Launcher", "Charge 播放器"),

    # Section headers
    "sec_task": ("Task & Checkpoint", "任務 & 模型檔"),
    "sec_play": ("Play Control", "播放控制"),
    "sec_scene": ("Scene & Environment", "場景 & 環境"),
    "sec_vis": ("Visualization & Flags", "視覺化 & 選項"),
    "sec_orca": ("ORCA / RVO2 Safety Filter", "ORCA / RVO2 安全過濾"),
    "sec_advanced": ("Advanced", "進階"),
    "sec_preview": ("Command Preview", "指令預覽"),

    # Labels
    "task": ("Task:", "任務:"),
    "run": ("Run:", "訓練紀錄:"),
    "checkpoint": ("Checkpoint:", "模型檔:"),
    "browse": ("Browse...", "瀏覽..."),
    "steps": ("Steps:", "步數:"),
    "camera": ("Camera:", "攝影機:"),
    "num_envs": ("num_envs:", "環境數:"),
    "stage": ("Stage:", "階段:"),
    "load_stage_params": ("Load Stage Params", "載入訓練參數"),
    "load_stage_title": ("Stage Params Loaded", "已載入訓練階段參數"),
    "load_stage_free": (
        "Stage=0 means Free Play (no curriculum). Pick 1-8 to load training params.",
        "Stage=0 = 自由模式（不套用課程）。請選 1-8 載入對應訓練參數。",
    ),
    "load_stage_fail": ("Failed to load stage params", "無法載入訓練參數"),
    "goals": ("Goals:", "目標數:"),
    "goal_dist": ("Goal dist (min, max):", "目標距離 (近, 遠):"),
    "static_obs": ("Static obs:", "靜態障礙:"),
    "dynamic_obs": ("Dynamic obs:", "動態障礙:"),
    "obs_behavior": ("Obs behavior:", "障礙行為:"),
    "obs_speed": ("Obs speed:", "障礙速度:"),
    "walls": ("Walls:", "牆壁數:"),
    "episode_sec": ("Episode (sec):", "回合 (秒):"),
    "empty_default": ("(empty=env default)", "(空=環境預設)"),
    "obs_near_goal": ("Obs near goal:", "目標旁障礙:"),
    "near_goal_r": ("Near goal radius:", "目標旁半徑:"),
    "extra_args": ("Extra args:", "額外參數:"),

    # Checkbuttons
    "deterministic": ("Deterministic", "確定性推論"),
    "real_time": ("Real-time", "即時模式"),
    "scripted_obs": ("Scripted obstacles (interval events)", "動態障礙移動 (定時事件)"),
    "no_walls": ("No walls (--no_walls)", "無牆壁 (--no_walls)"),
    "bev_vis": ("BEV Visualization (--bev_vis)", "BEV 視覺化 (--bev_vis)"),
    "headless": ("Headless (--headless)", "無畫面 (--headless)"),
    "lidar_no_noise": ("LiDAR no noise", "LiDAR 無雜訊"),
    "lidar_r_min": ("LiDAR r_min (m):", "LiDAR 盲區 (m):"),
    "collision_dist": ("Collision dist (center-to-center, m):", "碰撞距離 (中心到中心, m):"),
    "max_angular_vel": ("Max ω (rad/s):", "角速度上限 ω (rad/s):"),
    "max_angular_accel": ("Max α (rad/s²):", "角加速度上限 α (rad/s²):"),
    "empty_angular_default": ("(empty=env default, e.g. 1.2 / 3.0)", "(空=環境預設，如 1.2 / 3.0)"),
    "no_goal_movement": ("No goal movement", "目標不移動"),
    "bev_trail": ("BEV Trail", "BEV 軌跡"),
    "trail_length": ("Trail pts:", "軌跡點數:"),
    "aux_debug": ("Aux debug", "輔助除錯"),
    "play_diag": ("Play diagnostics", "播放診斷"),
    "usd_scene": ("USD Scene:", "USD 場景:"),
    "usd_scene_none": ("(Procedural maze)", "(程式化迷宮)"),
    "arena_size": ("Arena size:", "場景邊長:"),
    "arena_size_default": ("(20×20 train default)", "(20×20 訓練預設)"),
    "usd_scene_browse": ("Custom USD...", "自訂 USD..."),
    "lidar_bias_enable": (
        "LiDAR distance bias (narrow gap assist)",
        "LiDAR 距離偏移 (窄通道輔助)",
    ),
    "lidar_bias_val": ("Bias (m):", "偏移量 (m):"),
    "enable_rvo2": (
        "Enable RVO2 ORCA Filter (--use_rvo2_filter)",
        "使用 RVO2 ORCA 過濾 (--use_rvo2_filter)",
    ),

    # RSGS-Lite
    "sec_rsgs": (
        "RSGS-Lite (Stuck Recovery)",
        "RSGS-Lite (卡住自動脫困)",
    ),
    "enable_rsgs": (
        "Enable RSGS-Lite — detect stuck, find safe gap, set intermediate goal (--use_rsgs)",
        "啟用 RSGS-Lite — 偵測卡住後從 LiDAR 找安全通道，設中間目標繞行 (--use_rsgs)",
    ),
    "rsgs_desc": (
        "Stuck detect -> LiDAR gap search -> recovery goal -> resume original goal",
        "卡住偵測 -> LiDAR 通道搜尋 -> 設脫困中間目標 -> 脫困後回到原目標",
    ),
    "rsgs_params": ("RSGS Parameters", "RSGS 脫困參數"),
    "rsgs_stuck_window": ("Detect window (steps):", "偵測窗口 (步) N步沒動=卡:"),
    "rsgs_stuck_thresh": ("Min displacement (m):", "最低位移 (m) 低於=卡住:"),
    "rsgs_gap_width": ("Min gap width (bins):", "最小通道寬 (bins):"),
    "rsgs_gap_clear": ("Gap clear value:", "通道淨空值 (LiDAR):"),
    "rsgs_recovery_dist": ("Subgoal dist (m):", "中間目標距離 (m):"),
    "rsgs_max_steps": ("Max recovery steps:", "最長脫困步數:"),
    "rsgs_exit_disp": ("Success dist (m):", "脫困成功距離 (m):"),
    "rsgs_goal_bias": ("Goal dir bias [0-1]:", "目標方向偏好 [0-1]:"),

    # ORCA parameters
    "orca_params": ("ORCA Parameters", "ORCA 參數"),
    "th_dynamic": ("TH dynamic (s):", "動態 TH (秒):"),
    "th_static": ("TH static (s):", "靜態 TH (秒):"),
    "angle_thresh": ("Angle thresh (deg):", "角度門檻 (度):"),
    "safety_margin": ("Safety margin (m):", "安全邊距 (m):"),
    "culling_radius": ("Culling radius (m):", "篩選半徑 (m):"),
    "neighbor_dist": ("Neighbor dist (m):", "鄰近距離 (m):"),
    "max_neighbors": ("Max neighbors:", "最大鄰居數:"),
    "obs_inflation": ("Obs inflation:", "障礙膨脹倍率:"),
    "tactical_retreat": (
        "Tactical Retreat (forced reverse on lateral threat)",
        "戰術後退 (側向威脅強制倒車)",
    ),
    "disp_thresh": ("Disp thresh (m):", "位移門檻 (m):"),
    "disp_window": ("Disp window (steps):", "位移窗口 (步):"),

    # Buttons
    "preview_cmd": ("Preview Command", "預覽指令"),
    "copy_clipboard": ("Copy to Clipboard", "複製到剪貼簿"),
    "launch_play": ("  LAUNCH PLAY  ", "  開始播放  "),
    "kill_play": ("Kill All Play", "清除所有 Play"),
    "save_settings": ("Save Settings", "儲存設定"),

    # Save/Load dialogs
    "saved_title": ("Saved", "已儲存"),
    "saved_msg": ("Settings saved!", "設定已儲存！"),
    "loaded_msg": ("Settings loaded from last session.", "已載入上次設定。"),

    # Dialogs
    "copied_title": ("Copied", "已複製"),
    "copied_msg": ("Command copied to clipboard!", "指令已複製到剪貼簿！"),
    "kill_title": ("Kill Play", "清除 Play"),
    "kill_confirm": ("Kill all play_rnn_car processes?", "清除所有 play_rnn_car 進程？"),
    "kill_done": ("Killed {n} process(es).", "已清除 {n} 個進程。"),
    "kill_none": ("No play processes found.", "沒有找到 play 進程。"),
    "launch_title": ("Launch Play", "開始播放"),
    "launch_confirm": ("Launch play?", "確定開始播放？"),
    "launched_title": ("Launched", "已開始"),
    "launched_msg": ("Play process started!", "播放程序已開始！"),
    "error_title": ("Error", "錯誤"),
    "error_launch": ("Failed to launch:", "開始失敗："),
    "select_ckpt": ("Select Checkpoint", "選擇模型檔"),
}


def guess_task_from_run(run_name: str) -> str | None:
    """Guess training task from run directory name."""
    name = run_name.lower()
    if "_tc_" in name or "_tcorridor" in name or name.endswith("_tc"):
        return "Isaac-Navigation-Charge-VLP16-Curriculum-WD-TCorridor"
    if name.startswith(("sa", "debug")) or "a2c" in name or "extractor" in name:
        return "Isaac-Navigation-Charge-VLP16-Curriculum-WD"
    if "navrl" in name:
        return "Isaac-Navigation-Charge-VLP16-Curriculum-NavRL"
    if "marl" in name:
        return "Isaac-Navigation-Charge-VLP16-Curriculum-WD"
    return None

OBSTACLE_BEHAVIORS = [
    "patrol",
    "static",
    "random_walk",
    "horizontal_crossing",
    "path_crossing",
    "near_miss",
    "corridor_crossing",
    "occlusion",
    "mixed",
]

CAMERAS = ["top", "follow", "side"]


_STAGES_FILE = (
    REPO_ROOT
    / "source" / "isaaclab_tasks" / "isaaclab_tasks" / "manager_based"
    / "locomotion" / "velocity" / "config" / "charge_skrl"
    / "curriculum" / "phases" / "wd_single_agent_v1.py"
)


def load_training_stage_params(stage: int) -> dict | None:
    """Load STAGES[stage-1] from wd_single_agent_v1.py (1-indexed).

    Returns the nested phase dict (scene/behavior/reward/...) or None.
    Lazy import via importlib to avoid Isaac Lab dependency at launcher startup.
    """
    if stage < 1 or stage > 8:
        return None
    if not _STAGES_FILE.exists():
        return None
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "wd_single_agent_v1_for_launcher", str(_STAGES_FILE),
        )
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        stages = getattr(mod, "STAGES", None)
        if not stages or stage > len(stages):
            return None
        return stages[stage - 1]
    except Exception:
        return None


def scan_run_dirs() -> list[str]:
    """Scan logs/rnn_car/ for run directories with checkpoints."""
    dirs = []
    if not LOGS_DIR.exists():
        return dirs
    for run_dir in sorted(LOGS_DIR.iterdir(), reverse=True):
        if run_dir.is_dir() and any(run_dir.glob("checkpoint_*.pt")):
            dirs.append(run_dir.name)
    return dirs


# ============================================================================
# GUI
# ============================================================================


class PlayLauncherApp:
    # Fonts per language: (header_font, body_font)
    _EN_HEADER_FONT = ("Helvetica", 11, "bold")
    _EN_BODY_FONT = ("TkDefaultFont",)
    # CJK bold is hard to read — use normal weight, slightly larger
    _ZH_HEADER_FONT_FALLBACK = ("TkDefaultFont", 12)
    _ZH_BODY_FONT_FALLBACK = ("TkDefaultFont", 10)

    # Preferred CJK fonts in priority order
    _ZH_FONT_CANDIDATES = [
        "Noto Sans CJK TC",
        "Source Han Sans TC",
        "WenQuanYi Micro Hei",
        "AR PL UMing TW",
        "Microsoft JhengHei",
    ]

    def __init__(self, root: tk.Tk):
        self.root = root
        self.lang = 0  # 0 = EN, 1 = ZH
        self._i18n_widgets: list[tuple[tk.Widget, str]] = []

        # Detect best CJK font
        available = {f.lower() for f in tkfont.families()}
        zh_family = None
        for candidate in self._ZH_FONT_CANDIDATES:
            if candidate.lower() in available:
                zh_family = candidate
                break
        if zh_family:
            self._zh_header_font = (zh_family, 12)
            self._zh_body_font = (zh_family, 10)
        else:
            self._zh_header_font = self._ZH_HEADER_FONT_FALLBACK
            self._zh_body_font = self._ZH_BODY_FONT_FALLBACK

        self.root.title(self._t("title"))
        self.root.geometry("940x850")
        self.root.resizable(True, True)

        # Set TkDefaultFont to CJK-capable font so all widgets render Chinese
        if zh_family:
            default_font = tkfont.nametofont("TkDefaultFont")
            default_font.configure(family=zh_family, size=10)

        style = ttk.Style()
        style.configure("Header.TLabel", font=self._EN_HEADER_FONT)

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

        # Linux mousewheel
        canvas.bind_all("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))
        canvas.bind_all("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))

        self._build_ui()

    # ------------------------------------------------------------------
    # i18n helpers
    # ------------------------------------------------------------------

    def _t(self, key: str) -> str:
        """Return the translated string for the current language."""
        return _STRINGS[key][self.lang]

    def _reg(self, widget: tk.Widget, key: str) -> tk.Widget:
        """Register a widget for language switching and return it."""
        self._i18n_widgets.append((widget, key))
        return widget

    def _switch_lang(self):
        """Toggle between EN and ZH and update all registered widgets + fonts."""
        self.lang = 1 - self.lang
        self.root.title(self._t("title"))
        self.lang_btn.configure(text="EN" if self.lang == 1 else "中文")

        # Switch fonts — CJK bold is too thick, use normal weight
        style = ttk.Style()
        if self.lang == 1:  # ZH
            style.configure("Header.TLabel", font=self._zh_header_font)
            style.configure("TLabel", font=self._zh_body_font)
            style.configure("TCheckbutton", font=self._zh_body_font)
            style.configure("TButton", font=self._zh_body_font)
            style.configure("TLabelframe.Label", font=self._zh_body_font)
            style.configure("TSpinbox", font=self._zh_body_font)
            style.configure("TEntry", font=self._zh_body_font)
            style.configure("TCombobox", font=self._zh_body_font)
            style.configure("TRadiobutton", font=self._zh_body_font)
        else:  # EN
            style.configure("Header.TLabel", font=self._EN_HEADER_FONT)
            style.configure("TLabel", font=self._EN_BODY_FONT)
            style.configure("TCheckbutton", font=self._EN_BODY_FONT)
            style.configure("TButton", font=self._EN_BODY_FONT)
            style.configure("TLabelframe.Label", font=self._EN_BODY_FONT)
            style.configure("TSpinbox", font=self._EN_BODY_FONT)
            style.configure("TEntry", font=self._EN_BODY_FONT)
            style.configure("TCombobox", font=self._EN_BODY_FONT)
            style.configure("TRadiobutton", font=self._EN_BODY_FONT)

        for widget, key in self._i18n_widgets:
            widget.configure(text=self._t(key))

    # ------------------------------------------------------------------
    # UI layout
    # ------------------------------------------------------------------

    def _build_ui(self):
        f = self.main_frame
        row = 0

        # ── Language toggle ──
        lang_frame = ttk.Frame(f)
        lang_frame.grid(row=row, column=0, columnspan=4, sticky="e", padx=5, pady=(5, 0))
        self.lang_btn = ttk.Button(lang_frame, text="中文", width=6,
                                   command=self._switch_lang)
        self.lang_btn.pack(side="right")
        row += 1

        # ── Task & Checkpoint ──
        self._reg(
            ttk.Label(f, text=self._t("sec_task"), style="Header.TLabel"),
            "sec_task",
        ).grid(row=row, column=0, columnspan=4, sticky="w", padx=5, pady=(10, 5))
        row += 1

        self._reg(ttk.Label(f, text=self._t("task")), "task").grid(
            row=row, column=0, sticky="w", padx=5
        )
        self.task_var = tk.StringVar(value=TASKS[0])
        ttk.Combobox(f, textvariable=self.task_var, values=TASKS, width=55).grid(
            row=row, column=1, columnspan=3, sticky="w", padx=5
        )
        row += 1

        self._reg(ttk.Label(f, text=self._t("run")), "run").grid(
            row=row, column=0, sticky="w", padx=5
        )
        run_dirs = scan_run_dirs()
        self.run_var = tk.StringVar(value=run_dirs[0] if run_dirs else "")
        run_cb = ttk.Combobox(f, textvariable=self.run_var, values=run_dirs, width=55)
        run_cb.grid(row=row, column=1, columnspan=3, sticky="w", padx=5)
        run_cb.bind("<<ComboboxSelected>>", self._on_run_selected)
        row += 1

        self._reg(ttk.Label(f, text=self._t("checkpoint")), "checkpoint").grid(
            row=row, column=0, sticky="w", padx=5
        )
        self.ckpt_var = tk.StringVar()
        self.ckpt_cb = ttk.Combobox(f, textvariable=self.ckpt_var, values=[], width=55)
        self.ckpt_cb.grid(row=row, column=1, columnspan=2, sticky="w", padx=5)
        self._reg(
            ttk.Button(f, text=self._t("browse"), command=self._browse_ckpt),
            "browse",
        ).grid(row=row, column=3, padx=5)
        row += 1
        self._on_run_selected(None)

        # ── Play Control ──
        row = self._separator(f, row, "sec_play")

        self._reg(ttk.Label(f, text=self._t("steps")), "steps").grid(
            row=row, column=0, sticky="w", padx=5
        )
        self.steps_var = tk.IntVar(value=3000)
        ttk.Spinbox(f, from_=100, to=50000, increment=500,
                     textvariable=self.steps_var, width=8).grid(
            row=row, column=1, sticky="w", padx=5
        )

        self._reg(ttk.Label(f, text=self._t("camera")), "camera").grid(
            row=row, column=2, sticky="w", padx=5
        )
        self.camera_var = tk.StringVar(value="top")
        ttk.Combobox(f, textvariable=self.camera_var, values=CAMERAS, width=8,
                      state="readonly").grid(
            row=row, column=3, sticky="w", padx=5
        )
        row += 1

        self.deterministic_var = tk.BooleanVar(value=True)
        self._reg(
            ttk.Checkbutton(f, text=self._t("deterministic"),
                            variable=self.deterministic_var),
            "deterministic",
        ).grid(row=row, column=0, columnspan=2, sticky="w", padx=5)

        self.real_time_var = tk.BooleanVar(value=False)
        self._reg(
            ttk.Checkbutton(f, text=self._t("real_time"),
                            variable=self.real_time_var),
            "real_time",
        ).grid(row=row, column=2, columnspan=2, sticky="w", padx=5)
        row += 1

        # ── Scene & Environment ──
        row = self._separator(f, row, "sec_scene")

        # USD Scene selector
        self._reg(ttk.Label(f, text=self._t("usd_scene")), "usd_scene").grid(
            row=row, column=0, sticky="w", padx=5
        )
        _usd_choices = [
            self._t("usd_scene_none"),  # index 0 = procedural
            "warehouse",
            "warehouse_full",
            "hospital",
            "simple_room",
            "grid",
            "grid_black",
            "rough_plane",
            "3floor",
            "3floor_v2",
            self._t("usd_scene_browse"),  # last = custom browse
        ]
        self.usd_scene_var = tk.StringVar(value=_usd_choices[0])
        self._usd_combo = ttk.Combobox(
            f, textvariable=self.usd_scene_var,
            values=_usd_choices, width=28, state="readonly",
        )
        self._usd_combo.grid(row=row, column=1, columnspan=2, sticky="w", padx=5)
        self._usd_custom_path = tk.StringVar(value="")

        def _on_usd_scene_select(_event=None):
            val = self.usd_scene_var.get()
            # USD 場景與 arena 縮放互斥：選了實際 USD 場景就還原 arena 為訓練預設，
            # 避免兩者同時設定造成 --arena_size 被靜默忽略的混淆。
            if val and val not in (self._t("usd_scene_none"), self._t("usd_scene_browse")):
                if hasattr(self, "_arena_default_label"):
                    self.arena_size_var.set(self._arena_default_label)
            if val == self._t("usd_scene_browse"):
                path = filedialog.askopenfilename(
                    title="Select USD Scene",
                    filetypes=[("USD files", "*.usd *.usda *.usdc"), ("All", "*.*")],
                )
                if path:
                    self._usd_custom_path.set(path)
                    # Show filename in combo display
                    short = os.path.basename(path)
                    self._usd_combo["values"] = list(_usd_choices) + [short]
                    self.usd_scene_var.set(short)
                else:
                    self.usd_scene_var.set(_usd_choices[0])
            self._preview_cmd()

        self._usd_combo.bind("<<ComboboxSelected>>", _on_usd_scene_select)
        row += 1

        # Arena size selector (程式化場景等比例縮放；與 USD 互斥)
        self._reg(ttk.Label(f, text=self._t("arena_size")), "arena_size").grid(
            row=row, column=0, sticky="w", padx=5
        )
        self._arena_default_label = self._t("arena_size_default")
        _arena_choices = [self._arena_default_label, "16", "12", "10", "8", "6"]
        self.arena_size_var = tk.StringVar(value=_arena_choices[0])

        def _on_arena_select(_event=None):
            # Arena 縮放與 USD 場景互斥：選了數字邊長就自動取消 USD 場景，
            # 否則 _build_command 會把 --arena_size 靜默丟掉（USD 自帶幾何）。
            raw = self.arena_size_var.get().strip()
            try:
                val = float(raw)
            except (TypeError, ValueError):
                val = None
            if val is not None and val > 0 and self._resolve_usd_scene_arg() is not None:
                self.usd_scene_var.set(_usd_choices[0])  # 還原為程序化場景
                self._usd_custom_path.set("")
            self._preview_cmd()

        self._arena_combo = ttk.Combobox(
            f, textvariable=self.arena_size_var,
            values=_arena_choices, width=18, state="readonly",
        )
        self._arena_combo.grid(row=row, column=1, columnspan=2, sticky="w", padx=5)
        self._arena_combo.bind("<<ComboboxSelected>>", _on_arena_select)
        self._reg(
            ttk.Label(f, text=self._t("arena_size_default"), foreground="gray"),
            "arena_size_default",
        ).grid(row=row, column=3, sticky="w", padx=5)
        row += 1

        self._reg(ttk.Label(f, text=self._t("num_envs")), "num_envs").grid(
            row=row, column=0, sticky="w", padx=5
        )
        self.num_envs_var = tk.IntVar(value=1)
        ttk.Spinbox(f, from_=1, to=64, textvariable=self.num_envs_var, width=8).grid(
            row=row, column=1, sticky="w", padx=5
        )

        self._reg(ttk.Label(f, text=self._t("stage")), "stage").grid(
            row=row, column=2, sticky="w", padx=5
        )
        stage_frame = ttk.Frame(f)
        stage_frame.grid(row=row, column=3, sticky="w", padx=5)
        self.stage_var = tk.IntVar(value=0)
        ttk.Spinbox(
            stage_frame, from_=0, to=8, textvariable=self.stage_var, width=4,
        ).pack(side="left")
        self._reg(
            ttk.Button(
                stage_frame, text=self._t("load_stage_params"),
                command=self._load_stage_params, width=14,
            ),
            "load_stage_params",
        ).pack(side="left", padx=4)
        row += 1

        # Goals
        self._reg(ttk.Label(f, text=self._t("goals")), "goals").grid(
            row=row, column=0, sticky="w", padx=5
        )
        self.goals_var = tk.IntVar(value=1)
        ttk.Spinbox(f, from_=1, to=10, textvariable=self.goals_var, width=8).grid(
            row=row, column=1, sticky="w", padx=5
        )

        self._reg(ttk.Label(f, text=self._t("goal_dist")), "goal_dist").grid(
            row=row, column=2, sticky="w", padx=5
        )
        gd_frame = ttk.Frame(f)
        gd_frame.grid(row=row, column=3, sticky="w", padx=5)
        self.goal_dist_min_var = tk.DoubleVar(value=5.0)
        self.goal_dist_max_var = tk.DoubleVar(value=6.0)
        ttk.Spinbox(gd_frame, from_=1.0, to=20.0, increment=0.5,
                     textvariable=self.goal_dist_min_var, width=5).pack(side="left")
        ttk.Label(gd_frame, text=" ~ ").pack(side="left")
        ttk.Spinbox(gd_frame, from_=1.0, to=20.0, increment=0.5,
                     textvariable=self.goal_dist_max_var, width=5).pack(side="left")
        row += 1

        # Obstacles
        self._reg(ttk.Label(f, text=self._t("static_obs")), "static_obs").grid(
            row=row, column=0, sticky="w", padx=5
        )
        self.static_obs_var = tk.IntVar(value=20)
        ttk.Spinbox(f, from_=0, to=20, textvariable=self.static_obs_var, width=8).grid(
            row=row, column=1, sticky="w", padx=5
        )

        self._reg(ttk.Label(f, text=self._t("dynamic_obs")), "dynamic_obs").grid(
            row=row, column=2, sticky="w", padx=5
        )
        self.dynamic_obs_var = tk.IntVar(value=7)
        ttk.Spinbox(f, from_=0, to=20, textvariable=self.dynamic_obs_var, width=8).grid(
            row=row, column=3, sticky="w", padx=5
        )
        row += 1

        # Obstacle behavior & speed
        self._reg(ttk.Label(f, text=self._t("obs_behavior")), "obs_behavior").grid(
            row=row, column=0, sticky="w", padx=5
        )
        self.obs_behavior_var = tk.StringVar(value="patrol")
        ttk.Combobox(f, textvariable=self.obs_behavior_var,
                      values=OBSTACLE_BEHAVIORS, width=18, state="readonly").grid(
            row=row, column=1, sticky="w", padx=5
        )

        self._reg(ttk.Label(f, text=self._t("obs_speed")), "obs_speed").grid(
            row=row, column=2, sticky="w", padx=5
        )
        self.obs_speed_var = tk.DoubleVar(value=0.6)
        ttk.Spinbox(f, from_=0.0, to=2.0, increment=0.1,
                     textvariable=self.obs_speed_var, width=8).grid(
            row=row, column=3, sticky="w", padx=5
        )
        row += 1

        # Scripted obstacles checkbox
        self.scripted_obs_var = tk.BooleanVar(value=False)
        self._reg(
            ttk.Checkbutton(f, text=self._t("scripted_obs"),
                            variable=self.scripted_obs_var),
            "scripted_obs",
        ).grid(row=row, column=0, columnspan=2, sticky="w", padx=5)
        row += 1

        # Walls
        self._reg(ttk.Label(f, text=self._t("walls")), "walls").grid(
            row=row, column=0, sticky="w", padx=5
        )
        self.walls_var = tk.IntVar(value=1)
        ttk.Spinbox(f, from_=0, to=8, textvariable=self.walls_var, width=8).grid(
            row=row, column=1, sticky="w", padx=5
        )

        self.no_walls_var = tk.BooleanVar(value=True)
        self._reg(
            ttk.Checkbutton(f, text=self._t("no_walls"),
                            variable=self.no_walls_var),
            "no_walls",
        ).grid(row=row, column=2, columnspan=2, sticky="w", padx=5)
        row += 1

        # Episode length
        self._reg(ttk.Label(f, text=self._t("episode_sec")), "episode_sec").grid(
            row=row, column=0, sticky="w", padx=5
        )
        self.episode_s_var = tk.StringVar(value="")  # empty = use env default
        ttk.Entry(f, textvariable=self.episode_s_var, width=10).grid(
            row=row, column=1, sticky="w", padx=5
        )
        self._reg(ttk.Label(f, text=self._t("empty_default")), "empty_default").grid(
            row=row, column=2, sticky="w", padx=5
        )
        row += 1

        # Goal near obstacles
        self._reg(ttk.Label(f, text=self._t("obs_near_goal")), "obs_near_goal").grid(
            row=row, column=0, sticky="w", padx=5
        )
        self.obs_near_goal_var = tk.IntVar(value=2)
        ttk.Spinbox(f, from_=0, to=10, textvariable=self.obs_near_goal_var,
                     width=8).grid(row=row, column=1, sticky="w", padx=5)

        self._reg(ttk.Label(f, text=self._t("near_goal_r")), "near_goal_r").grid(
            row=row, column=2, sticky="w", padx=5
        )
        self.obs_near_goal_r_var = tk.DoubleVar(value=2.0)
        ttk.Spinbox(f, from_=0.5, to=10.0, increment=0.5,
                     textvariable=self.obs_near_goal_r_var, width=8).grid(
            row=row, column=3, sticky="w", padx=5
        )
        row += 1

        # ── Visualization & Flags ──
        row = self._separator(f, row, "sec_vis")

        self.bev_var = tk.BooleanVar(value=True)
        self._reg(
            ttk.Checkbutton(f, text=self._t("bev_vis"), variable=self.bev_var),
            "bev_vis",
        ).grid(row=row, column=0, columnspan=1, sticky="w", padx=5)

        self.bev_trail_var = tk.BooleanVar(value=True)
        self._reg(
            ttk.Checkbutton(f, text=self._t("bev_trail"), variable=self.bev_trail_var),
            "bev_trail",
        ).grid(row=row, column=1, sticky="w", padx=5)

        self._reg(ttk.Label(f, text=self._t("trail_length")), "trail_length").grid(
            row=row, column=2, sticky="e", padx=2
        )
        self.trail_length_var = tk.IntVar(value=500)
        ttk.Spinbox(f, from_=50, to=5000, increment=50,
                     textvariable=self.trail_length_var, width=6).grid(
            row=row, column=3, sticky="w", padx=5
        )
        row += 1

        self.headless_var = tk.BooleanVar(value=False)
        self._reg(
            ttk.Checkbutton(f, text=self._t("headless"), variable=self.headless_var),
            "headless",
        ).grid(row=row, column=0, columnspan=2, sticky="w", padx=5)
        row += 1

        self.lidar_no_noise_var = tk.BooleanVar(value=True)
        self._reg(
            ttk.Checkbutton(f, text=self._t("lidar_no_noise"),
                            variable=self.lidar_no_noise_var),
            "lidar_no_noise",
        ).grid(row=row, column=0, columnspan=2, sticky="w", padx=5)

        self.no_goal_movement_var = tk.BooleanVar(value=True)
        self._reg(
            ttk.Checkbutton(f, text=self._t("no_goal_movement"),
                            variable=self.no_goal_movement_var),
            "no_goal_movement",
        ).grid(row=row, column=2, columnspan=2, sticky="w", padx=5)
        row += 1

        # LiDAR r_min & Collision distance
        self._reg(ttk.Label(f, text=self._t("lidar_r_min")), "lidar_r_min").grid(
            row=row, column=0, sticky="w", padx=5
        )
        self.lidar_r_min_var = tk.DoubleVar(value=0.1)
        ttk.Spinbox(f, from_=0.0, to=2.0, increment=0.1,
                     textvariable=self.lidar_r_min_var, width=8).grid(
            row=row, column=1, sticky="w", padx=5
        )

        self._reg(ttk.Label(f, text=self._t("collision_dist")), "collision_dist").grid(
            row=row, column=2, sticky="w", padx=5
        )
        # ⚠️ collision_dist = obstacle_collision_geometric 的「中心到中心」門檻，
        #    不是 LiDAR 表面距離。表面接觸時中心距 ≈ robot_radius(0.35)+obs_radius(≈0.3)=0.65m，
        #    故預設 0.65 才會在「碰到」當下觸發；設太小（如 0.2）要嚴重重疊才判定。
        self.collision_dist_var = tk.DoubleVar(value=0.65)
        ttk.Spinbox(f, from_=0.1, to=2.0, increment=0.05,
                     textvariable=self.collision_dist_var, width=8).grid(
            row=row, column=3, sticky="w", padx=5
        )
        row += 1

        # 角速度 / 角加速度上限（slew clamp）
        self._reg(ttk.Label(f, text=self._t("max_angular_vel")), "max_angular_vel").grid(
            row=row, column=0, sticky="w", padx=5
        )
        self.max_angular_vel_var = tk.StringVar(value="")  # 空 = 環境預設
        ttk.Entry(f, textvariable=self.max_angular_vel_var, width=8).grid(
            row=row, column=1, sticky="w", padx=5
        )

        self._reg(ttk.Label(f, text=self._t("max_angular_accel")), "max_angular_accel").grid(
            row=row, column=2, sticky="w", padx=5
        )
        self.max_angular_accel_var = tk.StringVar(value="")  # 空 = 環境預設
        ttk.Entry(f, textvariable=self.max_angular_accel_var, width=8).grid(
            row=row, column=3, sticky="w", padx=5
        )
        row += 1

        self._reg(ttk.Label(f, text=self._t("empty_angular_default"), foreground="gray"),
                  "empty_angular_default").grid(
            row=row, column=0, columnspan=4, sticky="w", padx=5
        )
        row += 1

        self.aux_debug_var = tk.BooleanVar(value=True)
        self._reg(
            ttk.Checkbutton(f, text=self._t("aux_debug"),
                            variable=self.aux_debug_var),
            "aux_debug",
        ).grid(row=row, column=0, columnspan=2, sticky="w", padx=5)

        self.play_diag_var = tk.BooleanVar(value=False)
        self._reg(
            ttk.Checkbutton(f, text=self._t("play_diag"),
                            variable=self.play_diag_var),
            "play_diag",
        ).grid(row=row, column=2, columnspan=2, sticky="w", padx=5)
        row += 1

        # ── LiDAR Distance Bias ──
        self.lidar_bias_var = tk.BooleanVar(value=False)
        self._reg(
            ttk.Checkbutton(f, text=self._t("lidar_bias_enable"),
                            variable=self.lidar_bias_var,
                            command=self._toggle_lidar_bias),
            "lidar_bias_enable",
        ).grid(row=row, column=0, columnspan=2, sticky="w", padx=5)

        self._reg(ttk.Label(f, text=self._t("lidar_bias_val")), "lidar_bias_val").grid(
            row=row, column=2, sticky="w", padx=5
        )
        self.lidar_bias_spin_var = tk.DoubleVar(value=0.20)
        self.lidar_bias_spin = ttk.Spinbox(
            f, from_=0.05, to=1.0, increment=0.05,
            textvariable=self.lidar_bias_spin_var, width=6
        )
        self.lidar_bias_spin.grid(row=row, column=3, sticky="w", padx=5)
        self._toggle_lidar_bias()
        row += 1

        # ── ORCA / RVO2 Safety Filter ──
        row = self._separator(f, row, "sec_orca")

        self.rvo2_var = tk.BooleanVar(value=True)
        self._reg(
            ttk.Checkbutton(f, text=self._t("enable_rvo2"),
                            variable=self.rvo2_var, command=self._toggle_rvo2),
            "enable_rvo2",
        ).grid(row=row, column=0, columnspan=4, sticky="w", padx=5)
        row += 1

        self.rvo2_frame = ttk.LabelFrame(f, text=self._t("orca_params"))
        self._i18n_widgets.append((self.rvo2_frame, "orca_params"))
        self.rvo2_frame.grid(row=row, column=0, columnspan=4, sticky="ew", padx=5, pady=3)
        rf = self.rvo2_frame
        rr = 0

        # Row 0: Time horizons
        self._reg(ttk.Label(rf, text=self._t("th_dynamic")), "th_dynamic").grid(
            row=rr, column=0, sticky="w", padx=5
        )
        self.rvo2_th_var = tk.DoubleVar(value=2.5)
        ttk.Spinbox(rf, from_=0.1, to=10.0, increment=0.1,
                     textvariable=self.rvo2_th_var, width=6).grid(
            row=rr, column=1, sticky="w", padx=5
        )

        self._reg(ttk.Label(rf, text=self._t("th_static")), "th_static").grid(
            row=rr, column=2, sticky="w", padx=5
        )
        self.rvo2_th_static_var = tk.DoubleVar(value=0.3)
        ttk.Spinbox(rf, from_=0.1, to=5.0, increment=0.1,
                     textvariable=self.rvo2_th_static_var, width=6).grid(
            row=rr, column=3, sticky="w", padx=5
        )
        rr += 1

        # Row 1: Angle threshold + safety margin
        self._reg(ttk.Label(rf, text=self._t("angle_thresh")), "angle_thresh").grid(
            row=rr, column=0, sticky="w", padx=5
        )
        self.rvo2_angle_var = tk.IntVar(value=60)
        ttk.Spinbox(rf, from_=10, to=180, increment=10,
                     textvariable=self.rvo2_angle_var, width=6).grid(
            row=rr, column=1, sticky="w", padx=5
        )

        self._reg(ttk.Label(rf, text=self._t("safety_margin")), "safety_margin").grid(
            row=rr, column=2, sticky="w", padx=5
        )
        self.rvo2_margin_var = tk.DoubleVar(value=0.10)
        ttk.Spinbox(rf, from_=0.0, to=1.0, increment=0.05,
                     textvariable=self.rvo2_margin_var, width=6).grid(
            row=rr, column=3, sticky="w", padx=5
        )
        rr += 1

        # Row 2: Culling radius + neighbor dist
        self._reg(ttk.Label(rf, text=self._t("culling_radius")), "culling_radius").grid(
            row=rr, column=0, sticky="w", padx=5
        )
        self.rvo2_culling_var = tk.DoubleVar(value=1.5)
        ttk.Spinbox(rf, from_=1.0, to=20.0, increment=0.5,
                     textvariable=self.rvo2_culling_var, width=6).grid(
            row=rr, column=1, sticky="w", padx=5
        )

        self._reg(ttk.Label(rf, text=self._t("neighbor_dist")), "neighbor_dist").grid(
            row=rr, column=2, sticky="w", padx=5
        )
        self.rvo2_neighbor_var = tk.DoubleVar(value=2.0)
        ttk.Spinbox(rf, from_=1.0, to=30.0, increment=1.0,
                     textvariable=self.rvo2_neighbor_var, width=6).grid(
            row=rr, column=3, sticky="w", padx=5
        )
        rr += 1

        # Row 3: Max neighbors + displacement stuck
        self._reg(ttk.Label(rf, text=self._t("max_neighbors")), "max_neighbors").grid(
            row=rr, column=0, sticky="w", padx=5
        )
        self.rvo2_max_nb_var = tk.IntVar(value=2)
        ttk.Spinbox(rf, from_=1, to=50, increment=1,
                     textvariable=self.rvo2_max_nb_var, width=6).grid(
            row=rr, column=1, sticky="w", padx=5
        )

        self._reg(ttk.Label(rf, text=self._t("disp_thresh")), "disp_thresh").grid(
            row=rr, column=2, sticky="w", padx=5
        )
        self.rvo2_disp_th_var = tk.DoubleVar(value=0.3)
        ttk.Spinbox(rf, from_=0.05, to=2.0, increment=0.05,
                     textvariable=self.rvo2_disp_th_var, width=6).grid(
            row=rr, column=3, sticky="w", padx=5
        )
        rr += 1

        # Row 4: Displacement window + Obs inflation
        self._reg(ttk.Label(rf, text=self._t("disp_window")), "disp_window").grid(
            row=rr, column=0, sticky="w", padx=5
        )
        self.rvo2_disp_win_var = tk.IntVar(value=15)
        ttk.Spinbox(rf, from_=5, to=50, increment=1,
                     textvariable=self.rvo2_disp_win_var, width=6).grid(
            row=rr, column=1, sticky="w", padx=5
        )

        self._reg(ttk.Label(rf, text=self._t("obs_inflation")), "obs_inflation").grid(
            row=rr, column=2, sticky="w", padx=5
        )
        self.rvo2_obs_inflation_var = tk.DoubleVar(value=1.8)
        ttk.Spinbox(rf, from_=1.0, to=3.0, increment=0.1,
                     textvariable=self.rvo2_obs_inflation_var, width=6).grid(
            row=rr, column=3, sticky="w", padx=5
        )

        rr += 1

        # Row 5: Tactical retreat toggle
        self.rvo2_tactical_var = tk.BooleanVar(value=True)
        self._reg(
            ttk.Checkbutton(rf, text=self._t("tactical_retreat"),
                            variable=self.rvo2_tactical_var),
            "tactical_retreat",
        ).grid(row=rr, column=0, columnspan=4, sticky="w", padx=5)

        self._toggle_rvo2()
        row += 1

        # ── RSGS-Lite ──
        row = self._separator(f, row, "sec_rsgs")

        self.rsgs_var = tk.BooleanVar(value=False)
        self._reg(
            ttk.Checkbutton(f, text=self._t("enable_rsgs"),
                            variable=self.rsgs_var, command=self._toggle_rsgs),
            "enable_rsgs",
        ).grid(row=row, column=0, columnspan=4, sticky="w", padx=5)
        row += 1

        # 流程說明
        self._reg(
            ttk.Label(f, text=self._t("rsgs_desc"), foreground="gray"),
            "rsgs_desc",
        ).grid(row=row, column=0, columnspan=4, sticky="w", padx=20)
        row += 1

        self.rsgs_frame = ttk.LabelFrame(f, text=self._t("rsgs_params"))
        self._i18n_widgets.append((self.rsgs_frame, "rsgs_params"))
        self.rsgs_frame.grid(row=row, column=0, columnspan=4, sticky="ew", padx=5, pady=3)
        sf = self.rsgs_frame

        rr = 0
        self._reg(ttk.Label(sf, text=self._t("rsgs_stuck_window")), "rsgs_stuck_window").grid(
            row=rr, column=0, sticky="w", padx=5)
        self.rsgs_stuck_window_var = tk.IntVar(value=15)
        ttk.Spinbox(sf, from_=5, to=60, increment=1,
                     textvariable=self.rsgs_stuck_window_var, width=6).grid(
            row=rr, column=1, sticky="w", padx=2)
        self._reg(ttk.Label(sf, text=self._t("rsgs_stuck_thresh")), "rsgs_stuck_thresh").grid(
            row=rr, column=2, sticky="w", padx=5)
        self.rsgs_stuck_thresh_var = tk.DoubleVar(value=0.3)
        ttk.Spinbox(sf, from_=0.05, to=2.0, increment=0.05,
                     textvariable=self.rsgs_stuck_thresh_var, width=6).grid(
            row=rr, column=3, sticky="w", padx=2)
        rr += 1

        self._reg(ttk.Label(sf, text=self._t("rsgs_gap_width")), "rsgs_gap_width").grid(
            row=rr, column=0, sticky="w", padx=5)
        self.rsgs_gap_width_var = tk.IntVar(value=3)
        ttk.Spinbox(sf, from_=1, to=36, increment=1,
                     textvariable=self.rsgs_gap_width_var, width=6).grid(
            row=rr, column=1, sticky="w", padx=2)
        self._reg(ttk.Label(sf, text=self._t("rsgs_gap_clear")), "rsgs_gap_clear").grid(
            row=rr, column=2, sticky="w", padx=5)
        self.rsgs_gap_clear_var = tk.DoubleVar(value=0.10)
        ttk.Spinbox(sf, from_=0.01, to=1.0, increment=0.01,
                     textvariable=self.rsgs_gap_clear_var, width=6).grid(
            row=rr, column=3, sticky="w", padx=2)
        rr += 1

        self._reg(ttk.Label(sf, text=self._t("rsgs_recovery_dist")), "rsgs_recovery_dist").grid(
            row=rr, column=0, sticky="w", padx=5)
        self.rsgs_recovery_dist_var = tk.DoubleVar(value=2.0)
        ttk.Spinbox(sf, from_=0.5, to=10.0, increment=0.5,
                     textvariable=self.rsgs_recovery_dist_var, width=6).grid(
            row=rr, column=1, sticky="w", padx=2)
        self._reg(ttk.Label(sf, text=self._t("rsgs_max_steps")), "rsgs_max_steps").grid(
            row=rr, column=2, sticky="w", padx=5)
        self.rsgs_max_steps_var = tk.IntVar(value=50)
        ttk.Spinbox(sf, from_=10, to=200, increment=5,
                     textvariable=self.rsgs_max_steps_var, width=6).grid(
            row=rr, column=3, sticky="w", padx=2)
        rr += 1

        self._reg(ttk.Label(sf, text=self._t("rsgs_exit_disp")), "rsgs_exit_disp").grid(
            row=rr, column=0, sticky="w", padx=5)
        self.rsgs_exit_disp_var = tk.DoubleVar(value=1.0)
        ttk.Spinbox(sf, from_=0.1, to=5.0, increment=0.1,
                     textvariable=self.rsgs_exit_disp_var, width=6).grid(
            row=rr, column=1, sticky="w", padx=2)
        self._reg(ttk.Label(sf, text=self._t("rsgs_goal_bias")), "rsgs_goal_bias").grid(
            row=rr, column=2, sticky="w", padx=5)
        self.rsgs_goal_bias_var = tk.DoubleVar(value=0.3)
        ttk.Spinbox(sf, from_=0.0, to=1.0, increment=0.05,
                     textvariable=self.rsgs_goal_bias_var, width=6).grid(
            row=rr, column=3, sticky="w", padx=2)

        self._toggle_rsgs()
        row += 1

        # ── Advanced ──
        row = self._separator(f, row, "sec_advanced")

        self._reg(ttk.Label(f, text=self._t("extra_args")), "extra_args").grid(
            row=row, column=0, sticky="w", padx=5
        )
        self.extra_args_var = tk.StringVar(value="")
        ttk.Entry(f, textvariable=self.extra_args_var, width=60).grid(
            row=row, column=1, columnspan=3, sticky="w", padx=5
        )
        row += 1

        # ── Command Preview ──
        row = self._separator(f, row, "sec_preview")

        self.cmd_text = tk.Text(f, height=7, width=105, font=("Courier", 9), wrap="word")
        self.cmd_text.grid(row=row, column=0, columnspan=4, sticky="ew", padx=5)
        row += 1

        # ── Buttons ──
        btn_frame = ttk.Frame(f)
        btn_frame.grid(row=row, column=0, columnspan=4, pady=15)

        self._reg(
            ttk.Button(btn_frame, text=self._t("preview_cmd"),
                       command=self._preview_cmd),
            "preview_cmd",
        ).pack(side="left", padx=10)
        self._reg(
            ttk.Button(btn_frame, text=self._t("copy_clipboard"),
                       command=self._copy_cmd),
            "copy_clipboard",
        ).pack(side="left", padx=10)
        self._reg(
            ttk.Button(btn_frame, text=self._t("launch_play"),
                       command=self._launch),
            "launch_play",
        ).pack(side="left", padx=10)
        self._reg(
            ttk.Button(btn_frame, text=self._t("kill_play"),
                       command=self._kill_play),
            "kill_play",
        ).pack(side="left", padx=10)
        self._reg(
            ttk.Button(btn_frame, text=self._t("save_settings"),
                       command=self._save_settings),
            "save_settings",
        ).pack(side="left", padx=10)

        # 啟動時自動載入上次設定
        self._load_settings()
        self._preview_cmd()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _separator(self, parent, row: int, key: str) -> int:
        ttk.Separator(parent, orient="horizontal").grid(
            row=row, column=0, columnspan=4, sticky="ew", pady=10
        )
        row += 1
        self._reg(
            ttk.Label(parent, text=self._t(key), style="Header.TLabel"),
            key,
        ).grid(row=row, column=0, columnspan=4, sticky="w", padx=5, pady=(0, 5))
        return row + 1

    def _on_run_selected(self, event):
        run_name = self.run_var.get()
        run_dir = LOGS_DIR / run_name
        if run_dir.exists():
            pts = sorted(run_dir.glob("checkpoint_*.pt"), reverse=True)
            ckpt_list = [str(p.relative_to(REPO_ROOT)) for p in pts]
            self.ckpt_cb["values"] = ckpt_list
            if ckpt_list:
                self.ckpt_var.set(ckpt_list[0])
        # Auto-detect task from run name
        guessed = guess_task_from_run(run_name)
        if guessed:
            self.task_var.set(guessed)

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

    def _load_stage_params(self):
        """Load STAGES[stage-1] into all GUI fields to reproduce training scene.

        Maps wd_single_agent_v1.STAGES nested schema → GUI variables:
          scene.goals → goals_var
          scene.goal_distance → goal_dist_min_var / goal_dist_max_var
          scene.static_obstacles / dynamic_obstacles → static_obs_var / dynamic_obs_var
          scene.walls_max → walls_var (+ disable no_walls if walls > 0)
          scene.obs_near_goal_count / radius → obs_near_goal_var / obs_near_goal_r_var
          scene.episode_s → episode_s_var
          behavior.obstacle_speed → obs_speed_var
          behavior.behavior_mix → obs_behavior_var ("mixed" if 2+, else dominant)
        """
        stage = self.stage_var.get()
        if stage < 1:
            messagebox.showinfo(
                self._t("load_stage_title"), self._t("load_stage_free"),
            )
            return

        params = load_training_stage_params(stage)
        if params is None:
            messagebox.showerror(
                self._t("error_title"),
                f"{self._t('load_stage_fail')} (stage={stage})\n"
                f"path: {_STAGES_FILE}",
            )
            return

        scene = params.get("scene", {})
        behavior = params.get("behavior", {})

        # Scene → GUI
        self.goals_var.set(int(scene.get("goals", 1)))
        gd = scene.get("goal_distance", (5.0, 6.0))
        self.goal_dist_min_var.set(float(gd[0]))
        self.goal_dist_max_var.set(float(gd[1]))
        self.static_obs_var.set(int(scene.get("static_obstacles", 0)))
        self.dynamic_obs_var.set(int(scene.get("dynamic_obstacles", 0)))

        walls_max = int(scene.get("walls_max", 0))
        self.walls_var.set(walls_max)
        # Stage has walls → disable "no_walls" so they actually spawn
        if walls_max > 0 and self.no_walls_var.get():
            self.no_walls_var.set(False)

        self.obs_near_goal_var.set(int(scene.get("obs_near_goal_count", 0)))
        self.obs_near_goal_r_var.set(float(scene.get("obs_near_goal_radius", 2.0)))
        self.episode_s_var.set(str(int(scene.get("episode_s", 60))))

        # Behavior → GUI
        self.obs_speed_var.set(float(behavior.get("obstacle_speed", 0.8)))
        bmix = behavior.get("behavior_mix", {}) or {}
        if len(bmix) > 1:
            self.obs_behavior_var.set("mixed")
        elif len(bmix) == 1:
            only = next(iter(bmix.keys()))
            if only in OBSTACLE_BEHAVIORS:
                self.obs_behavior_var.set(only)

        self._preview_cmd()

        # Summary popup
        name = params.get("name", f"SA{stage}")
        beh_summary = (
            "mixed (" + ",".join(f"{k}:{v:.0%}" for k, v in bmix.items()) + ")"
            if len(bmix) > 1 else (next(iter(bmix.keys())) if bmix else "—")
        )
        msg = (
            f"Stage {stage} — {name}\n"
            f"goals={int(scene.get('goals', 0))}  "
            f"dist={gd[0]:.1f}~{gd[1]:.1f}m  "
            f"static={int(scene.get('static_obstacles', 0))}  "
            f"dynamic={int(scene.get('dynamic_obstacles', 0))}\n"
            f"walls={walls_max}({float(scene.get('wall_length', 0)):.1f}m)  "
            f"episode={int(scene.get('episode_s', 60))}s  "
            f"obs_speed={behavior.get('obstacle_speed', 0.8):.2f}\n"
            f"behavior: {beh_summary}\n"
            f"obs_near_goal={int(scene.get('obs_near_goal_count', 0))}"
            f"@r={float(scene.get('obs_near_goal_radius', 0)):.1f}m"
        )
        messagebox.showinfo(self._t("load_stage_title"), msg)

    def _toggle_lidar_bias(self):
        state = "normal" if self.lidar_bias_var.get() else "disabled"
        self.lidar_bias_spin.configure(state=state)

    def _toggle_rvo2(self):
        state = "normal" if self.rvo2_var.get() else "disabled"
        for child in self.rvo2_frame.winfo_children():
            if isinstance(child, (ttk.Spinbox, ttk.Entry)):
                child.configure(state=state)

    def _toggle_rsgs(self):
        state = "normal" if self.rsgs_var.get() else "disabled"
        for child in self.rsgs_frame.winfo_children():
            if isinstance(child, (ttk.Spinbox, ttk.Entry)):
                child.configure(state=state)

    # 已知的內建 USD 場景 alias（與 play_rnn_car.py 的 _USD_SCENE_ALIASES /
    # _USD_LOCAL_ALIASES 對齊）。判斷 USD 是否啟用一律用這個集合，而非本地化標籤，
    # 避免跨語言載入舊存檔時把「程序化場景」誤判為 USD 場景。
    _KNOWN_USD_ALIASES = (
        "warehouse", "warehouse_full", "hospital", "simple_room",
        "grid", "grid_black", "rough_plane", "3floor", "3floor_v1", "3floor_v2",
    )

    def _resolve_usd_scene_arg(self):
        """回傳要送給 --usd_scene 的值（alias 或自訂絕對路徑），若未啟用 USD 則 None。

        Label-agnostic：只認已知 alias 或實際自訂路徑，不依賴本地化的 none/browse 標籤，
        因此跨語言載入舊存檔（標籤字串變動）時也不會誤判。
        """
        val = (self.usd_scene_var.get() or "").strip()
        if not val:
            return None
        if val in self._KNOWN_USD_ALIASES:
            return val
        # 自訂路徑：combobox 顯示 basename，實際路徑存在 _usd_custom_path
        custom = (self._usd_custom_path.get() or "").strip()
        if custom and val not in (self._t("usd_scene_none"), self._t("usd_scene_browse")):
            return custom
        return None

    def _build_command(self) -> str:
        """Build the play command string with correct play_rnn_car.py args."""
        parts = [
            "PYTHONUNBUFFERED=1",
            "./isaaclab.sh -p",
            str(PLAY_SCRIPT.relative_to(REPO_ROOT)),
        ]

        parts.append(f"--task {self.task_var.get()}")
        parts.append(f"--checkpoint {self.ckpt_var.get()}")
        parts.append(f"--num_envs {self.num_envs_var.get()}")
        parts.append(f"--steps {self.steps_var.get()}")
        parts.append(f"--camera {self.camera_var.get()}")

        if self.deterministic_var.get():
            parts.append("--deterministic")
        if self.real_time_var.get():
            parts.append("--real_time")
        if self.headless_var.get():
            parts.append("--headless")
        if self.bev_var.get():
            parts.append("--bev_vis")
            trail_len = self.trail_length_var.get() if self.bev_trail_var.get() else 0
            parts.append(f"--bev_trail_length {trail_len}")
        if self.lidar_no_noise_var.get():
            parts.append("--lidar_no_noise")
        parts.append(f"--lidar_r_min {self.lidar_r_min_var.get()}")
        parts.append(f"--collision_dist {self.collision_dist_var.get()}")
        # 角速度 / 角加速度覆寫（空 = 不覆寫，沿用 env cfg）
        _omega_v = self.max_angular_vel_var.get().strip()
        if _omega_v:
            parts.append(f"--max_angular_vel {_omega_v}")
        _omega_a = self.max_angular_accel_var.get().strip()
        if _omega_a:
            parts.append(f"--max_angular_accel {_omega_a}")
        if self.no_goal_movement_var.get():
            parts.append("--no_goal_movement")
        if self.aux_debug_var.get():
            parts.append("--aux_debug")
        if self.play_diag_var.get():
            parts.append("--play_diag")

        # Stage (0 = use default)
        stage = self.stage_var.get()
        if stage > 0:
            parts.append(f"--stage {stage}")

        # Scene overrides — using correct play_rnn_car.py arg names
        parts.append(f"--num_goals_override {self.goals_var.get()}")
        parts.append(f"--num_static_obs {self.static_obs_var.get()}")
        parts.append(f"--num_dynamic_obs {self.dynamic_obs_var.get()}")
        parts.append(f"--num_walls {self.walls_var.get()}")
        parts.append(f"--goal_distance_min {self.goal_dist_min_var.get()}")
        parts.append(f"--goal_distance_max {self.goal_dist_max_var.get()}")
        parts.append(f"--obstacle_speed {self.obs_speed_var.get()}")
        parts.append(f"--obstacle_behavior {self.obs_behavior_var.get()}")
        parts.append(f"--obs_near_goal_count {self.obs_near_goal_var.get()}")
        parts.append(f"--obs_near_goal_radius {self.obs_near_goal_r_var.get()}")

        # USD scene
        # ⚠️ 不要用「!= 本地化 none/browse 標籤」來判斷 USD 是否啟用：
        #    combobox 把本地化標籤當值存檔，跨語言載入時舊標籤(如英文 '(Procedural maze)')
        #    會 != 當前語言的 none 標籤(中文 '(程式化迷宮)') → 誤判 USD 啟用 → --arena_size
        #    被靜默丟掉、--usd_scene 也因不是已知 alias 而沒送 → 改 arena 完全沒反應。
        #    改為 label-agnostic：USD 啟用 ⇔ 值是已知 alias 或有實際自訂路徑。
        _usd_scene_arg = self._resolve_usd_scene_arg()  # str alias/path 或 None
        if _usd_scene_arg is not None:
            if " " in _usd_scene_arg:
                parts.append(f'--usd_scene "{_usd_scene_arg}"')
            else:
                parts.append(f"--usd_scene {_usd_scene_arg}")

        # Arena size — 只有選了數字邊長且未用 USD 場景時才送出（兩者互斥）
        _arena_raw = self.arena_size_var.get().strip()
        _usd_active = _usd_scene_arg is not None
        try:
            _arena_val = float(_arena_raw)
        except (TypeError, ValueError):
            _arena_val = None
        if _arena_val is not None and _arena_val > 0 and not _usd_active:
            parts.append(f"--arena_size {_arena_val:g}")

        if self.no_walls_var.get():
            parts.append("--no_walls")
        if self.scripted_obs_var.get():
            parts.append("--scripted_obstacles")

        # LiDAR distance bias
        if self.lidar_bias_var.get():
            parts.append(f"--lidar_dist_bias {self.lidar_bias_spin_var.get()}")

        # Episode length (optional)
        ep_s = self.episode_s_var.get().strip()
        if ep_s:
            parts.append(f"--episode_length_s {ep_s}")

        # ORCA / RVO2
        if self.rvo2_var.get():
            parts.append("--use_rvo2_filter")
            parts.append(f"--rvo2_time_horizon {self.rvo2_th_var.get()}")
            parts.append(f"--rvo2_time_horizon_static {self.rvo2_th_static_var.get()}")
            parts.append(f"--rvo2_angle_threshold {self.rvo2_angle_var.get()}")
            parts.append(f"--rvo2_safety_margin {self.rvo2_margin_var.get()}")
            parts.append(f"--rvo2_culling_radius {self.rvo2_culling_var.get()}")
            parts.append(f"--rvo2_neighbor_dist {self.rvo2_neighbor_var.get()}")
            parts.append(f"--rvo2_max_neighbors {self.rvo2_max_nb_var.get()}")
            parts.append(f"--rvo2_obs_inflation {self.rvo2_obs_inflation_var.get()}")
            if not self.rvo2_tactical_var.get():
                parts.append("--no_rvo2_tactical_retreat")
            parts.append(f"--rvo2_disp_threshold {self.rvo2_disp_th_var.get()}")
            parts.append(f"--rvo2_disp_window {self.rvo2_disp_win_var.get()}")

        # RSGS-Lite
        if self.rsgs_var.get():
            parts.append("--use_rsgs")
            parts.append(f"--rsgs_stuck_window {self.rsgs_stuck_window_var.get()}")
            parts.append(f"--rsgs_stuck_threshold {self.rsgs_stuck_thresh_var.get()}")
            parts.append(f"--rsgs_gap_min_width {self.rsgs_gap_width_var.get()}")
            parts.append(f"--rsgs_gap_clear_threshold {self.rsgs_gap_clear_var.get()}")
            parts.append(f"--rsgs_recovery_distance {self.rsgs_recovery_dist_var.get()}")
            parts.append(f"--rsgs_max_recovery_steps {self.rsgs_max_steps_var.get()}")
            parts.append(f"--rsgs_exit_displacement {self.rsgs_exit_disp_var.get()}")
            parts.append(f"--rsgs_goal_bias {self.rsgs_goal_bias_var.get()}")

        # Extra args
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
            f"{self._t('launch_confirm')}\n\n{cmd[:300]}...",
        )
        if not result:
            return

        # 啟動前自動儲存設定（靜默，不彈窗）
        self._save_settings_silent()

        try:
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            exec_cmd = cmd.replace("PYTHONUNBUFFERED=1 ", "")
            subprocess.Popen(
                exec_cmd,
                shell=True,
                cwd=str(REPO_ROOT),
                env=env,
            )
            messagebox.showinfo(self._t("launched_title"), self._t("launched_msg"))
        except Exception as e:
            messagebox.showerror(
                self._t("error_title"), f"{self._t('error_launch')}\n{e}"
            )

    def _kill_play(self):
        """Kill all play_rnn_car related processes."""
        import signal

        result = messagebox.askyesno(
            self._t("kill_title"), self._t("kill_confirm"),
        )
        if not result:
            return

        # Find all play_rnn_car processes (python, bash wrappers, sh -c)
        try:
            out = subprocess.check_output(
                ["pgrep", "-f", "play_rnn_car"],
                text=True,
            ).strip()
            pids = [int(p) for p in out.splitlines() if p.strip()]
        except subprocess.CalledProcessError:
            pids = []

        # Exclude our own launcher PID
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
            except ProcessLookupError:
                pass
            except PermissionError:
                pass

        # SIGKILL stragglers after a short wait
        self.root.after(1500, lambda: self._sigkill_remaining(pids))

        messagebox.showinfo(
            self._t("kill_title"),
            self._t("kill_done").format(n=killed),
        )

    def _sigkill_remaining(self, pids: list[int]):
        """Force-kill any processes that didn't exit after SIGTERM."""
        import signal
        for pid in pids:
            try:
                os.kill(pid, 0)  # check if still alive
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass

    # ------------------------------------------------------------------
    # Settings persistence (JSON)
    # ------------------------------------------------------------------

    _SETTINGS_FILE = REPO_ROOT / ".play_launcher_settings.json"

    # (attr_name, var_type) — 所有需要持久化的 GUI 變數
    _SETTINGS_KEYS: list[tuple[str, str]] = [
        # Task & Checkpoint
        ("task_var", "str"),
        ("run_var", "str"),
        ("ckpt_var", "str"),
        # Play Control
        ("steps_var", "int"),
        ("camera_var", "str"),
        ("deterministic_var", "bool"),
        ("real_time_var", "bool"),
        # Scene
        ("usd_scene_var", "str"),
        ("arena_size_var", "str"),
        ("num_envs_var", "int"),
        ("stage_var", "int"),
        ("goals_var", "int"),
        ("goal_dist_min_var", "float"),
        ("goal_dist_max_var", "float"),
        ("static_obs_var", "int"),
        ("dynamic_obs_var", "int"),
        ("obs_behavior_var", "str"),
        ("obs_speed_var", "float"),
        ("scripted_obs_var", "bool"),
        ("walls_var", "int"),
        ("no_walls_var", "bool"),
        ("episode_s_var", "str"),
        ("obs_near_goal_var", "int"),
        ("obs_near_goal_r_var", "float"),
        # Vis & Flags
        ("bev_var", "bool"),
        ("bev_trail_var", "bool"),
        ("trail_length_var", "int"),
        ("headless_var", "bool"),
        ("lidar_no_noise_var", "bool"),
        ("no_goal_movement_var", "bool"),
        ("lidar_r_min_var", "float"),
        ("collision_dist_var", "float"),
        ("max_angular_vel_var", "str"),
        ("max_angular_accel_var", "str"),
        ("aux_debug_var", "bool"),
        ("play_diag_var", "bool"),
        ("lidar_bias_var", "bool"),
        ("lidar_bias_spin_var", "float"),
        # ORCA
        ("rvo2_var", "bool"),
        ("rvo2_th_var", "float"),
        ("rvo2_th_static_var", "float"),
        ("rvo2_angle_var", "int"),
        ("rvo2_margin_var", "float"),
        ("rvo2_culling_var", "float"),
        ("rvo2_neighbor_var", "float"),
        ("rvo2_max_nb_var", "int"),
        ("rvo2_disp_th_var", "float"),
        ("rvo2_disp_win_var", "int"),
        ("rvo2_obs_inflation_var", "float"),
        ("rvo2_tactical_var", "bool"),
        # RSGS-Lite
        ("rsgs_var", "bool"),
        ("rsgs_stuck_window_var", "int"),
        ("rsgs_stuck_thresh_var", "float"),
        ("rsgs_gap_width_var", "int"),
        ("rsgs_gap_clear_var", "float"),
        ("rsgs_recovery_dist_var", "float"),
        ("rsgs_max_steps_var", "int"),
        ("rsgs_exit_disp_var", "float"),
        ("rsgs_goal_bias_var", "float"),
        # Advanced
        ("extra_args_var", "str"),
    ]

    def _collect_settings(self) -> dict:
        """收集所有 GUI 變數的當前值。"""
        data = {}
        for attr, vtype in self._SETTINGS_KEYS:
            var = getattr(self, attr, None)
            if var is None:
                continue
            try:
                data[attr] = var.get()
            except Exception:
                pass
        if hasattr(self, "_usd_custom_path"):
            data["_usd_custom_path"] = self._usd_custom_path.get()
        return data

    def _save_settings_silent(self):
        """靜默儲存（launch 時自動呼叫，不彈窗）。"""
        import json
        try:
            data = self._collect_settings()
            self._SETTINGS_FILE.write_text(
                json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception:
            pass

    def _save_settings(self):
        """將目前所有 GUI 設定寫入 JSON 檔案（含彈窗確認）。"""
        import json
        data = self._collect_settings()
        try:
            self._SETTINGS_FILE.write_text(
                json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            messagebox.showinfo(self._t("saved_title"), self._t("saved_msg"))
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save settings:\n{e}")

    def _load_settings(self):
        """從 JSON 檔案還原 GUI 設定（啟動時自動呼叫）。"""
        import json

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

        # 還原 custom USD path + 更新 combo 選項
        custom_path = data.get("_usd_custom_path", "")
        if custom_path and hasattr(self, "_usd_custom_path"):
            self._usd_custom_path.set(custom_path)
            usd_val = self.usd_scene_var.get()
            # 如果選中的是自訂路徑的 basename，把它加回 combo values
            _known_aliases = {"warehouse", "warehouse_full", "hospital",
                              "simple_room", "grid", "grid_black", "rough_plane",
                              "3floor", "3floor_v1", "3floor_v2"}
            if usd_val not in _known_aliases | {self._t("usd_scene_none"), self._t("usd_scene_browse")}:
                current_vals = list(self._usd_combo["values"])
                if usd_val not in current_vals:
                    current_vals.append(usd_val)
                    self._usd_combo["values"] = current_vals

        # 正規化跨語言 / 過時的 USD 標籤值：若載入的值既非已知 alias、也沒有自訂路徑，
        # 代表它是某語言版本的「程序化場景」none 標籤（或失效值）。把它還原成「當前語言」
        # 的 none 標籤，否則 readonly combobox 會顯示空白/殘值，且舊邏輯會誤判 USD 啟用。
        _usd_loaded = (self.usd_scene_var.get() or "").strip()
        if (
            _usd_loaded
            and _usd_loaded not in self._KNOWN_USD_ALIASES
            and _usd_loaded != self._t("usd_scene_browse")
            and not (self._usd_custom_path.get() or "").strip()
        ):
            self.usd_scene_var.set(self._t("usd_scene_none"))

        # USD 場景 vs arena 縮放互斥調和：舊存檔可能同時存了兩者。
        # 只有當 USD「實際啟用」(label-agnostic 判定為已知 alias/自訂路徑) 時才需調和；
        # 像 '(Procedural maze)' 這種純標籤值代表程序化場景，USD 並未啟用，arena 應保留。
        if self._resolve_usd_scene_arg() is not None and hasattr(self, "_arena_default_label"):
            _arena_raw = self.arena_size_var.get().strip()
            try:
                _arena_num = float(_arena_raw)
            except (TypeError, ValueError):
                _arena_num = None
            if _arena_num is not None and _arena_num > 0:
                self.arena_size_var.set(self._arena_default_label)

        # 載入 run 後刷新 checkpoint 列表
        run_name = data.get("run_var", "")
        if run_name:
            run_dir = LOGS_DIR / run_name
            if run_dir.exists():
                pts = sorted(run_dir.glob("checkpoint_*.pt"), reverse=True)
                ckpt_list = [str(p.relative_to(REPO_ROOT)) for p in pts]
                self.ckpt_cb["values"] = ckpt_list


# ============================================================================
# Main
# ============================================================================

def main():
    root = tk.Tk()
    PlayLauncherApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
