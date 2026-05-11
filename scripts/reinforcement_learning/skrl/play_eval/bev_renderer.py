"""
bev_renderer.py — Charge RL BEV 輔助顯示模組（中文介面）

功能:
  1. 72-bin LiDAR 極座標圖
  2. 機器人位置/方向 + 目標
  3. RNN aux 7D 預測 vs 真實值對比（最近兩個障礙物）
  4. 障礙物/牆壁/Goal 場景元素

設計: 獨立模組，被 play_charge.py import 使用
"""

import numpy as np
import torch


class ChargeBEVRenderer:
    """中文 BEV 視覺化器 — 顯示 LiDAR + RNN aux 預測 vs 真實"""

    def __init__(self, raw_env, max_range: float = 20.0, frame: str = "world",
                 show_aux: bool = True):
        import matplotlib
        from isaaclab.managers import SceneEntityCfg
        from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_skrl.mdp.observations.obs_functions import (
            wd_like_sweep_72,
        )

        if "agg" in matplotlib.get_backend().lower():
            matplotlib.use("TkAgg", force=True)
        import matplotlib.pyplot as plt
        # 設定中文字型
        plt.rcParams["font.sans-serif"] = ["Noto Sans CJK TC", "WenQuanYi Micro Hei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

        self.plt = plt
        self.raw_env = raw_env
        self.max_range = float(max_range)
        self.frame = frame
        self.show_aux = show_aux
        self.wd_like_sweep_72 = wd_like_sweep_72
        self.sensor_cfg = SceneEntityCfg("lidar")
        self.sensor_cfg.resolve(raw_env.scene)
        self.body_radius = 0.35

        self.plt.ion()
        if show_aux:
            self.fig, (self.ax_bev, self.ax_aux) = self.plt.subplots(
                1, 2, figsize=(14, 7),
                gridspec_kw={"width_ratios": [2, 1]},
            )
        else:
            self.fig, self.ax_bev = self.plt.subplots(figsize=(7, 7))
            self.ax_aux = None

        try:
            self.fig.canvas.manager.set_window_title("Charge RL BEV — 導航診斷")
        except Exception:
            pass

        # aux 歷史（用於繪製趨勢）
        self._aux_history_pred = []
        self._aux_history_gt = []
        self._aux_history_steps = []

    def _sensor_yaw(self) -> float:
        q = self.raw_env.scene.sensors["lidar"].data.quat_w[0].detach().cpu().numpy()
        w, x, y, z = [float(v) for v in q]
        return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))

    def _body_to_world(self, x_fwd, y_left, yaw: float):
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        return cos_y * x_fwd - sin_y * y_left, sin_y * x_fwd + cos_y * y_left

    def _world_rel_to_plot(self, rel_w, yaw: float):
        if self.frame == "world":
            return rel_w[..., 0], rel_w[..., 1]
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        x_fwd = cos_y * rel_w[..., 0] + sin_y * rel_w[..., 1]
        y_left = -sin_y * rel_w[..., 0] + cos_y * rel_w[..., 1]
        return y_left, x_fwd

    def update(self, step: int, obs_tensor: torch.Tensor, actions: torch.Tensor,
               aux_pred: torch.Tensor | None = None,
               aux_gt: torch.Tensor | None = None) -> bool:
        """更新 BEV 顯示。回傳 False 表示視窗已關閉。"""
        if not self.plt.fignum_exists(self.fig.number):
            return False

        self._draw_bev(step, obs_tensor, actions)
        if self.show_aux and self.ax_aux is not None:
            self._draw_aux_panel(step, aux_pred, aux_gt)

        self.fig.canvas.draw_idle()
        self.plt.pause(0.001)
        return self.plt.fignum_exists(self.fig.number)

    def _draw_bev(self, step: int, obs_tensor: torch.Tensor, actions: torch.Tensor):
        """繪製主 BEV LiDAR 圖（中文標籤）"""
        sweep = self.wd_like_sweep_72(
            self.raw_env, self.sensor_cfg,
            num_bins=72, r_max=self.max_range, r_robot=0.3, r_min=0.5, z_filter=0.5,
        )
        real_dist = (sweep[0] * self.max_range).detach().cpu().numpy()
        angles_deg = -180.0 + np.arange(72) * 5.0
        angles = np.radians(angles_deg)
        x_fwd = real_dist * np.cos(angles)
        y_left = real_dist * np.sin(angles)
        yaw = self._sensor_yaw()

        if self.frame == "world":
            plot_x, plot_y = self._body_to_world(x_fwd, y_left, yaw)
        else:
            plot_x, plot_y = y_left, x_fwd

        ax = self.ax_bev
        ax.cla()
        ax.set_facecolor("#141414")
        self.fig.patch.set_facecolor("#141414")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-self.max_range, self.max_range)
        ax.set_ylim(-self.max_range, self.max_range)
        frame_label = "世界座標" if self.frame == "world" else "車體座標（前方為上）"
        ax.set_xlabel("X (m)", color="white")
        ax.set_ylabel("Y (m)", color="white")
        ax.tick_params(colors="white")
        ax.grid(True, color="#383838", linewidth=0.7)
        ax.set_title(f"Charge BEV — 72-bin LiDAR ({frame_label})", color="white", fontsize=11)

        # 距離環
        for r_m in [2, 5, 10, 15, 20]:
            circle = self.plt.Circle((0, 0), r_m, fill=False, color="#4a4a4a", linewidth=0.8)
            ax.add_patch(circle)
            ax.text(0.2, r_m, f"{r_m}m", color="#888888", fontsize=8)

        # LiDAR 點雲
        ax.plot(plot_x, plot_y, color="#7fbf7f", linewidth=1.2, alpha=0.8)
        colors = np.full(real_dist.shape, "#50dc50", dtype=object)
        colors[real_dist < 5.0] = "#ffa500"
        colors[real_dist < 2.0] = "#ff3030"
        colors[real_dist >= self.max_range - 0.5] = "#808080"
        sizes = np.full(real_dist.shape, 22.0)
        sizes[real_dist < 5.0] = 36.0
        sizes[real_dist < 2.0] = 54.0
        ax.scatter(plot_x, plot_y, c=colors.tolist(), s=sizes, zorder=3)

        # 機器人
        ax.add_patch(self.plt.Circle((0, 0), 0.3, fill=False, color="white", linewidth=2.0, zorder=4))
        if self.frame == "world":
            hx, hy = 1.2 * np.cos(yaw), 1.2 * np.sin(yaw)
        else:
            hx, hy = 0.0, 1.2
        ax.arrow(0, 0, hx, hy, color="white", width=0.04, head_width=0.35,
                 length_includes_head=True, zorder=5)

        # 目標向量
        if obs_tensor is not None and obs_tensor.shape[-1] >= 6:
            goal_xy = obs_tensor[0, 4:6].detach().cpu().numpy()
            gx = float(np.clip(goal_xy[0], -self.max_range, self.max_range))
            gy = float(np.clip(goal_xy[1], -self.max_range, self.max_range))
            if self.frame == "world":
                goal_x, goal_y = self._body_to_world(gx, gy, yaw)
            else:
                goal_x, goal_y = gy, gx
            ax.arrow(0, 0, goal_x, goal_y, color="#ffd040", width=0.035,
                     head_width=0.45, length_includes_head=True, zorder=4)
            ax.scatter([goal_x], [goal_y], c=["#ffd040"], s=80, zorder=5)

        # 目標圓圈
        self._draw_goals(ax, yaw)

        # 資訊面板（中文）
        near_idx = int(real_dist.argmin())
        near_d = float(real_dist[near_idx])
        near_angle = -180.0 + near_idx * 5.0
        action_text = actions[0].detach().cpu().tolist() if actions is not None else ["?", "?"]
        text_lines = [
            f"步驟={step}  動作={action_text}",
            f"座標系={frame_label}  航向角={np.degrees(yaw):+.1f}°",
            f"最近障礙: {near_d:.2f}m @ bin{near_idx} ({near_angle:+.0f}°)",
            f"平均距離={real_dist.mean():.2f}m  <2m={int((real_dist < 2.0).sum())}/72",
            "白=機器人  黃=目標  紅=近距危險",
        ]
        ax.text(
            0.02, 0.98, "\n".join(text_lines),
            transform=ax.transAxes, va="top", ha="left", color="white", fontsize=9,
            bbox={"facecolor": "#202020", "edgecolor": "#606060", "alpha": 0.85},
        )

    def _draw_aux_panel(self, step: int,
                        aux_pred: torch.Tensor | None,
                        aux_gt: torch.Tensor | None):
        """繪製 RNN aux 7D 預測 vs 真實對比面板"""
        ax = self.ax_aux
        ax.cla()
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="white")
        ax.set_title("RNN 輔助預測 vs 真實（7D）", color="white", fontsize=11)

        if aux_pred is None:
            ax.text(0.5, 0.5, "尚無 aux 預測\n(需啟用 --aux_debug)",
                    transform=ax.transAxes, ha="center", va="center",
                    color="#888888", fontsize=12)
            return

        pred = aux_pred[0].detach().cpu().numpy()
        labels = [
            "近1_X (m)", "近1_Y (m)", "近1_距離",
            "近2_X (m)", "近2_Y (m)", "近2_距離",
            "時間步",
        ]

        # 記錄歷史
        self._aux_history_pred.append(pred.copy())
        self._aux_history_steps.append(step)
        if aux_gt is not None:
            gt = aux_gt[0].detach().cpu().numpy()
            self._aux_history_gt.append(gt.copy())
        else:
            gt = None
            self._aux_history_gt.append(np.full(7, np.nan))

        # 限制歷史長度
        max_hist = 100
        if len(self._aux_history_pred) > max_hist:
            self._aux_history_pred = self._aux_history_pred[-max_hist:]
            self._aux_history_gt = self._aux_history_gt[-max_hist:]
            self._aux_history_steps = self._aux_history_steps[-max_hist:]

        # 繪製柱狀圖：pred vs gt
        x_pos = np.arange(7)
        bar_width = 0.35
        bars_pred = ax.bar(x_pos - bar_width / 2, pred, bar_width,
                           color="#4fc3f7", alpha=0.8, label="RNN 預測")
        if gt is not None:
            bars_gt = ax.bar(x_pos + bar_width / 2, gt, bar_width,
                             color="#ff8a65", alpha=0.8, label="真實值")
            # 誤差標示
            errors = np.abs(pred - gt)
            for i, (p, g, e) in enumerate(zip(pred, gt, errors)):
                if e > 0.5:  # 只標示誤差大的
                    ax.text(i, max(p, g) + 0.1, f"Δ{e:.2f}",
                            ha="center", color="#ff5252", fontsize=8)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, color="white", fontsize=8, rotation=30, ha="right")
        ax.set_ylabel("數值", color="white", fontsize=9)
        ax.legend(loc="upper right", fontsize=9, framealpha=0.7)
        ax.axhline(y=0, color="#555555", linewidth=0.5)
        ax.set_ylim(-12, 12)

        # 底部摘要
        if gt is not None:
            mse = float(np.mean((pred - gt) ** 2))
            near1_err = float(np.sqrt((pred[0] - gt[0])**2 + (pred[1] - gt[1])**2))
            near2_err = float(np.sqrt((pred[3] - gt[3])**2 + (pred[4] - gt[4])**2))
            summary = (
                f"MSE={mse:.3f}  "
                f"近1位置誤差={near1_err:.2f}m  "
                f"近2位置誤差={near2_err:.2f}m"
            )
        else:
            summary = "真實值不可用（需 simulator ground truth）"
        ax.text(
            0.5, -0.12, summary,
            transform=ax.transAxes, ha="center", color="#aaaaaa", fontsize=9,
        )

    def _draw_goals(self, ax, yaw: float):
        """繪製場景中所有目標點"""
        try:
            cmd = self.raw_env.command_manager.get_term("goal_command")
            if not hasattr(cmd, "all_goals_pos_w"):
                return
            ng = int(cmd.cfg.num_goals)
            goals_w = cmd.all_goals_pos_w[0, :ng, :2].detach().cpu().numpy()
            robot_w = self.raw_env.scene["robot"].data.root_pos_w[0, :2].detach().cpu().numpy()
        except Exception:
            return

        rel_w = goals_w - robot_w[None, :]
        gx, gy = self._world_rel_to_plot(rel_w, yaw)
        ax.scatter(gx, gy, c="#40d8ff", s=38, marker="x", linewidths=1.2, zorder=4)
        for i, (x, y) in enumerate(zip(gx, gy)):
            ax.text(x + 0.12, y + 0.12, f"G{i}", color="#40d8ff", fontsize=7, zorder=4)

    def close(self):
        if self.plt.fignum_exists(self.fig.number):
            self.plt.close(self.fig)
