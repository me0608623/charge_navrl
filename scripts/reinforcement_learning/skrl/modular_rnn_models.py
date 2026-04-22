"""
Modular RNN Models — 移植自 Warp Drive CustomModuleConnected

核心設計 (來源: new_warp_drive/custom_envs/module_connected.py):
  1. PreprocessRNN: FC → RNN → concat → middle FC → predict head
  2. Gradient detach: preprocess output 傳給 RL head 前切斷梯度
     WD 有 3 個 .detach():
       - line 504: rl_ip = ip.detach()  (obs 副本給 RL concat)
       - line 537: memory_reg.detach()  (hidden state 存儲)
       - line 573: concat_input = rl_in_.detach()  (RL 最終輸入)
     效果: RL loss.backward() 不會流到 preprocess/RNN
  3. Auxiliary loss: RNN 只被 module prediction loss 訓練
     WD target: 7D preprocess_real_data (2 nearest neighbors × (x,y,dist) + timestep)
     WD loss:   log(clamp(L1, 0.01)) × per-dim weight
  4. RL head: 只看 detached features，PPO 只訓練 head

架構:
  Obs (139D env output, 只用 79D)
    → LidarStateExtractor (LiDAR Conv1d 64D + StateMLP 32D = 96D)
      [IsaacLab adaptation] WD 用 Linear(113→64)+ReLU 單層; IL 用 Conv1d+MLP 2-branch
    → PreprocessRNN (FC→RNN→concat→FC = 12D) + .detach()
    → cat(79D_obs, 12D_preprocess) = 91D
    → PolicyHead (91→256→256→256→512→38)
    → ValueHead (91→256→256→256→512→512→1)

WD 對齊參數:
  - RNN type: vanilla RNN (not GRU)
  - module_connect_dim (preprocess_dim): 12
  - memory_dim (hidden_dim): 30
  - PolicyHead: [256, 256, 256, 512]
  - ValueHead: [256, 256, 256, 512, 512]

WD 與 IL 的有意差異 (保留 WD modular principle，搭配 IL 環境適配):
  - Extractor: WD=Linear(113→64); IL=Conv1d(72D LiDAR)+MLP(7D state)=96D
    原因: IL obs layout 不同（79D vs 113D），Conv1d 對 LiDAR 更適合

不依賴 obs60 (MOT features) — 部署可行。
注意: WD train_rnn_car.py 中 spot_state_obs_agent_rate=0，obs 113D 中 60D 障礙物區塊
多為 default/placeholder 值（type-2 state obstacles 未生成）。有效障礙資訊來自 36D LiDAR
+ 7D preprocess_real_data (2 nearest neighbors)。
"""

import torch
import torch.nn as nn
import numpy as np

# ============================================================================
# Observation layout (from 139D env output — 只用 79D)
# ============================================================================

EGO_START, EGO_END = 0, 4        # accel + vel + omega + radius
GOAL_START, GOAL_END = 4, 6      # waypoint (x, y)
LIDAR_START, LIDAR_END = 6, 78   # 72 bins
TIME_START, TIME_END = 78, 79    # remaining ratio (was 138:139 when obs=139D)

STATE_DIM = 7     # ego(4) + goal(2) + time(1)
LIDAR_DIM = 72
USED_OBS_DIM = 79  # 4 + 2 + 72 + 1
RAW_OBS_DIM = 79   # env output dim (no TopK)
NUM_BINS = 19
TOTAL_LOGITS = NUM_BINS * 2  # 38


# ============================================================================
# 2-Branch Feature Extractor (no obs60)
# ============================================================================

class LidarStateExtractor(nn.Module):
    """2-branch extractor: LiDAR Conv1d (64D) + StateMLP (32D) = 96D.

    從 79D env obs 中取 LiDAR(72D) 和 state(7D) 部分。保留 WD modular principle，無 TopK obs。
    """

    def __init__(self):
        super().__init__()
        # Branch 1: LiDAR Conv1d
        self.lidar_conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.lidar_pool = nn.AdaptiveMaxPool1d(1)
        self.lidar_proj = nn.Linear(64, 64)
        self.lidar_ln = nn.LayerNorm(64)

        # Branch 2: State MLP (ego + goal + time = 7D)
        self.state_mlp = nn.Sequential(
            nn.Linear(STATE_DIM, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        self.state_ln = nn.LayerNorm(32)

    @property
    def output_dim(self):
        return 96  # 64 + 32

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: [B, 79] env observation
        Returns:
            [B, 96] feature embedding
        """
        lidar = obs[:, LIDAR_START:LIDAR_END].unsqueeze(1)   # [B, 1, 72]
        lidar = self.lidar_conv(lidar)                        # [B, 64, 18]
        lidar = self.lidar_pool(lidar).squeeze(-1)            # [B, 64]
        lidar = self.lidar_ln(self.lidar_proj(lidar))         # [B, 64]

        ego = obs[:, EGO_START:EGO_END]                       # [B, 4]
        goal = obs[:, GOAL_START:GOAL_END]                     # [B, 2]
        time_feat = obs[:, TIME_START:TIME_END]                # [B, 1]
        state_7d = torch.cat([ego, goal, time_feat], dim=-1)  # [B, 7]
        state = self.state_ln(self.state_mlp(state_7d))       # [B, 32]

        return torch.cat([lidar, state], dim=-1)              # [B, 96]


# ============================================================================
# Preprocess RNN Module (WD: vanilla RNN, not GRU)
# ============================================================================

class PreprocessRNN(nn.Module):
    """模組化 RNN — 移植自 Warp Drive module_connected.py。

    WD 原始設計 (module_connected.py lines 214-573):
      preprocess_info_front: Linear(obs→64)+ReLU
      → RNN(64→30)
      → [if concat_rnn] cat(rnn_out, fc_out)
      → preprocess_info_middle: Linear(→module_connect_dim)+ReLU
      → [training] preprocess_info_back: Linear(→network_feture_dim)+ReLU (for aux loss)
      → .detach() → concat(detached_obs, preprocess_feat) → RL head

    WD detach 位置 (line 573): concat_input = rl_in_.detach()
    → RL loss 永遠不訓練此模組。RNN 只被 module loss (aux) 訓練。

    Predict head (WD-style):
      - 單層 Linear(12→7) + ReLU — 與 WD preprocess_info_back 完全一致
        (WD config: module_network=[64,'rnn',32,'rl'], network_feture_dim=7 → back=[12→7])
      - Target: 7D privileged geometry (2 nearest obstacles body-frame x,y,dist + timestep)
      - Loss: WD module loss — log(clamp(L1, 0.01)) × per-dim weight
      - See wd_aux_targets.py for target construction and loss computation

    WD 對齊:
      - rnn_type='RNN' (vanilla, not GRU) — WD: rnn_layer_type='RNN'
      - hidden_dim=30 — WD: memory_dim=30
      - preprocess_dim=12 — WD: module_connect_dim=12
      - concat_rnn=True — WD train_rnn_car.py 同樣用 True
      - predict_head: Linear(12→7)+ReLU — WD preprocess_info_back 原樣
    """

    def __init__(
        self,
        input_dim: int = 96,
        fc_dim: int = 48,
        hidden_dim: int = 30,
        preprocess_dim: int = 12,
        predict_dim: int = 7,
        concat_rnn: bool = True,
        rnn_type: str = "RNN",  # "RNN" (WD default) or "GRU"
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.concat_rnn = concat_rnn
        self.rnn_type = rnn_type

        # FC front
        self.fc_front = nn.Sequential(
            nn.Linear(input_dim, fc_dim),
            nn.ReLU(),
        )

        # RNN (WD: vanilla RNN by default)
        if rnn_type == "GRU":
            self.rnn = nn.GRU(fc_dim, hidden_dim, num_layers=1, batch_first=False)
        else:
            self.rnn = nn.RNN(fc_dim, hidden_dim, num_layers=1, batch_first=False,
                              nonlinearity='relu')

        # FC middle (after concat)
        middle_input = hidden_dim + fc_dim if concat_rnn else hidden_dim
        self.fc_middle = nn.Sequential(
            nn.Linear(middle_input, preprocess_dim),
            nn.ReLU(),
        )

        # Predict head — WD-style: Linear(12→7)+ReLU (preprocess_info_back)
        # Training only — predicts 7D privileged geometry target for module loss
        self.predict_head = nn.Sequential(
            nn.Linear(preprocess_dim, predict_dim),
            nn.ReLU(),
        )

        self.preprocess_dim = preprocess_dim

    def forward(
        self,
        features: torch.Tensor,
        hidden: torch.Tensor,
        training: bool = False,
        detach_output: bool = False,
    ):
        """Supports both single-step [B, D] and sequence [L, B, D] inputs.

        Single-step mode (features.dim() == 2):
            features: [B, 96]
            Returns: feat [B, 12], prediction [B, 7] or None, new_hidden [1, B, H]

        Sequence mode (features.dim() == 3, for TBPTT aux training):
            features: [L, B, 96]
            Returns: feat [L, B, 12], prediction [L, B, 7] or None, new_hidden [1, B, H]

        Args:
            features: [B, 96] or [L, B, 96] from extractor
            hidden:   [1, B, hidden_dim] RNN hidden state
            training: if True, compute 7D prediction for WD module loss
            detach_output: if True, explicitly detach preprocess feat before return.
        """
        seq_mode = features.dim() == 3  # [L, B, D]

        if seq_mode:
            L, B, D = features.shape
            # FC front: reshape to [L*B, D], apply, reshape back
            fc_out = self.fc_front(features.reshape(L * B, D))  # [L*B, 48]
            fc_out = fc_out.reshape(L, B, -1)                   # [L, B, 48]

            # RNN unroll: input [L, B, 48], hidden [1, B, H]
            rnn_out, new_hidden = self.rnn(fc_out, hidden)      # [L, B, 30], [1, B, 30]

            if self.concat_rnn:
                combined = torch.cat([rnn_out, fc_out], dim=-1) # [L, B, 78]
            else:
                combined = rnn_out                              # [L, B, 30]

            preprocess_feat = self.fc_middle(
                combined.reshape(L * B, -1)).reshape(L, B, -1)  # [L, B, 12]

            prediction = None
            if training:
                prediction = self.predict_head(
                    preprocess_feat.reshape(L * B, -1)).reshape(L, B, -1)  # [L, B, 7]

            if detach_output:
                return preprocess_feat.detach(), prediction, new_hidden
            return preprocess_feat, prediction, new_hidden

        # --- Single-step mode (original path) ---
        fc_out = self.fc_front(features)                    # [B, 48]
        rnn_in = fc_out.unsqueeze(0)                        # [1, B, 48] (seq_len=1)
        rnn_out, new_hidden = self.rnn(rnn_in, hidden)      # [1, B, 30], [1, B, 30]
        rnn_out = rnn_out.squeeze(0)                        # [B, 30]

        if self.concat_rnn:
            combined = torch.cat([rnn_out, fc_out], dim=-1) # [B, 78]
        else:
            combined = rnn_out                              # [B, 30]

        preprocess_feat = self.fc_middle(combined)          # [B, 12]

        prediction = None
        if training:
            prediction = self.predict_head(preprocess_feat) # [B, 7] WD-style privileged target

        # WD 原版 (line 573): concat_input = rl_in_.detach()
        # WD 設計: RNN 永遠不吃 RL gradient。detach 在 RL concat 階段完成。
        # IL: rollout 用 torch.no_grad() 達到同樣效果; PPO 只用 cached rl_in。
        if detach_output:
            return preprocess_feat.detach(), prediction, new_hidden
        return preprocess_feat, prediction, new_hidden


# Backward compatibility alias
PreprocessGRU = PreprocessRNN


# ============================================================================
# RNN State Manager
# ============================================================================

class RNNStateManager:
    """Per-env GRU hidden state 管理器。"""

    def __init__(self, num_envs: int, hidden_dim: int, device: torch.device):
        self.hidden = torch.zeros(1, num_envs, hidden_dim, device=device)

    def get(self) -> torch.Tensor:
        return self.hidden

    def update(self, new_hidden: torch.Tensor):
        self.hidden = new_hidden.detach()

    def reset(self, env_ids: torch.Tensor):
        """env reset 時歸零對應 env 的 hidden state。"""
        if len(env_ids) > 0:
            self.hidden[:, env_ids, :] = 0.0


# ============================================================================
# RL Heads (PPO 只訓練這些)
# ============================================================================

class PolicyHead(nn.Module):
    """Policy head: input_dim → 38 logits (19 accel + 19 omega).

    WD 對齊: spot_policy = [256, 256, 256, 512] (4 hidden layers)
    input = cat(obs_79D, preprocess_12D) = 91D
    """

    def __init__(self, input_dim: int = 91):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, TOTAL_LOGITS),
        )

    def forward(self, rl_input: torch.Tensor) -> torch.Tensor:
        return self.net(rl_input)  # [B, 38]


class ValueHead(nn.Module):
    """Value head: input_dim → scalar.

    WD 對齊: spot_critic = [256, 256, 256, 512, 512] (5 hidden layers)
    input = cat(obs_79D, preprocess_12D) = 91D
    """

    def __init__(self, input_dim: int = 91):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )
        # 初始化: 初期 value 偏負（與 charge_skrl 一致）
        nn.init.orthogonal_(self.net[-1].weight, gain=0.01)
        nn.init.constant_(self.net[-1].bias, -8.0)

    def forward(self, rl_input: torch.Tensor) -> torch.Tensor:
        return self.net(rl_input)  # [B, 1]


# ============================================================================
# Obstacle Policy (Warp Drive obstacle agent 移植)
# ============================================================================

# Obstacle observation layout (per-obstacle, body-frame):
#   own_local_xy(2) + own_vel(2) + robot_rel_xy(2) + robot_vel(2) + d_wall(1) = 9D
OBS_POLICY_OBS_DIM = 9
OBS_POLICY_ACT_DIM = 2  # continuous (vx, vy) velocity command


class ObstaclePolicyFC(nn.Module):
    """Obstacle agent FC policy — parameter shared across all N obstacles.

    Warp Drive 對照: obstacle 用 FullyConnected (256×256)
    Output: 2D continuous velocity (vx, vy)，由 tanh 限制到 [-1, 1] 再乘 speed_limit
    """

    def __init__(self, obs_dim: int = OBS_POLICY_OBS_DIM, act_dim: int = OBS_POLICY_ACT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(128, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.constant_(self.mean_head.bias, 0.0)

    def forward(self, obs: torch.Tensor):
        """
        Args:
            obs: [B, 9] per-obstacle observation (B = num_envs * N_active_obs)
        Returns:
            mean: [B, 2] action mean
            std:  [B, 2] action std
        """
        h = self.net(obs)
        mean = torch.tanh(self.mean_head(h))  # [-1, 1]
        std = self.log_std.exp().expand_as(mean)
        return mean, std

    def sample(self, obs: torch.Tensor):
        """Sample action + compute log_prob for PPO."""
        mean, std = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        action = torch.clamp(action, -1.0, 1.0)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
        """Recompute log_prob + entropy for stored actions."""
        mean, std = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy


class ObstacleValueFC(nn.Module):
    """Obstacle agent value head."""

    def __init__(self, obs_dim: int = OBS_POLICY_OBS_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        nn.init.orthogonal_(self.net[-1].weight, gain=0.01)
        nn.init.constant_(self.net[-1].bias, 0.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)  # [B, 1]
