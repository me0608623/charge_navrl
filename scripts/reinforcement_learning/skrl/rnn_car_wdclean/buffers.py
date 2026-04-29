"""Rollout buffers for charge and obstacle agents -- extracted from train_rnn_car_wdclip.py."""

import torch


class ChargeRolloutBuffer:
    def __init__(self, num_steps, num_envs, rl_input_dim, obs_dim, hidden_dim, device):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = device
        self.rl_inputs = torch.zeros(num_steps, num_envs, rl_input_dim, device=device)
        self.actions = torch.zeros(num_steps, num_envs, 2, dtype=torch.long, device=device)
        self.log_probs = torch.zeros(num_steps, num_envs, device=device)
        self.rewards = torch.zeros(num_steps, num_envs, device=device)
        self.values = torch.zeros(num_steps, num_envs, device=device)
        self.dones = torch.zeros(num_steps, num_envs, device=device)
        self.raw_obs = torch.zeros(num_steps, num_envs, obs_dim, device=device)
        self.hiddens = torch.zeros(num_steps, num_envs, hidden_dim, device=device)
        # WD-style 7D privileged geometry target for module loss
        self.aux_targets = torch.zeros(num_steps, num_envs, 7, device=device)
        self.ptr = 0

    def add(self, rl_input, action, log_prob, reward, value, done, raw_ob, hidden,
            aux_target=None):
        i = self.ptr
        self.rl_inputs[i] = rl_input
        self.actions[i] = action
        self.log_probs[i] = log_prob
        self.rewards[i] = reward
        self.values[i] = value
        self.dones[i] = done
        self.raw_obs[i] = raw_ob
        self.hiddens[i] = hidden.squeeze(0)
        if aux_target is not None:
            self.aux_targets[i] = aux_target
        self.ptr += 1

    def reset(self):
        self.ptr = 0

    def sample_aux_sequences(
        self, seq_len: int, batch_size: int, burn_in: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int] | None:
        """Sample contiguous, non-episode-crossing sequences for TBPTT aux training.

        Algorithm:
          1. Build valid start indices: (t0, env) where t0+seq_len <= T_filled
             AND no done in [t0, t0+seq_len-1) (done at t means episode ends after t,
             so next step starts a new episode -- sequence must not span that boundary).
          2. Randomly sample `batch_size` starts (or fewer if not enough).
          3. Gather obs, target, h0 for each sequence.

        Args:
            seq_len: total sequence length (including burn_in)
            batch_size: desired number of sequences
            burn_in: first `burn_in` steps only warm up hidden, not counted in loss

        Returns:
            (obs_seq, target_seq, h0, valid_count) or None if no valid sequences.
            obs_seq:    [B, L, obs_dim]
            target_seq: [B, L, 7]
            h0:         [1, B, hidden_dim]
            valid_count: total number of valid start positions (for monitoring)
        """
        T_filled = self.ptr  # actual steps stored this rollout
        if T_filled < seq_len:
            return None

        # --- 1. Build valid start mask [T_filled - seq_len + 1, E] ---
        max_t0 = T_filled - seq_len  # inclusive
        E = self.num_envs

        dones_slice = self.dones[:T_filled]  # [T_filled, E]

        if seq_len <= 1:
            valid_mask = torch.ones(max_t0 + 1, E, dtype=torch.bool, device=self.device)
        elif seq_len == 2:
            valid_mask = dones_slice[:max_t0 + 1] < 0.5  # [max_t0+1, E]
        else:
            cum = torch.cumsum(dones_slice, dim=0)  # [T_filled, E]
            end_idx = torch.arange(seq_len - 2, seq_len - 2 + max_t0 + 1, device=self.device)
            cum_end = cum[end_idx]  # [max_t0+1, E]
            cum_start = torch.zeros(1, E, device=self.device)
            if max_t0 > 0:
                start_idx = torch.arange(0, max_t0, device=self.device)
                cum_start = torch.cat([cum_start, cum[start_idx]], dim=0)  # [max_t0+1, E]
            dones_in_window = cum_end - cum_start  # [max_t0+1, E]
            valid_mask = dones_in_window < 0.5

        # --- 2. Get valid (t0, env) pairs ---
        valid_positions = valid_mask.nonzero(as_tuple=False)  # [N_valid, 2]
        n_valid = valid_positions.shape[0]

        if n_valid == 0:
            return None

        actual_batch = min(batch_size, n_valid)
        chosen_idx = torch.randperm(n_valid, device=self.device)[:actual_batch]
        chosen = valid_positions[chosen_idx]  # [B, 2]
        t0s = chosen[:, 0]  # [B]
        envs = chosen[:, 1]  # [B]

        # --- 3. Gather sequences ---
        B = actual_batch
        L = seq_len

        time_offsets = torch.arange(L, device=self.device).unsqueeze(0)  # [1, L]
        t_indices = t0s.unsqueeze(1) + time_offsets  # [B, L]
        e_indices = envs.unsqueeze(1).expand(B, L)  # [B, L]

        obs_seq = self.raw_obs[t_indices, e_indices]        # [B, L, obs_dim]
        target_seq = self.aux_targets[t_indices, e_indices]  # [B, L, 7]
        h0 = self.hiddens[t0s, envs].unsqueeze(0)           # [1, B, hidden_dim]

        return obs_seq, target_seq, h0, n_valid


class ObstacleRolloutBuffer:
    """Flat buffer: each obstacle is an independent sample."""

    def __init__(self, num_steps, num_envs, max_obstacles, obs_dim, act_dim, device):
        self.num_steps = num_steps
        self.B = num_envs * max_obstacles  # flat batch dimension
        self.device = device
        self.obs = torch.zeros(num_steps, self.B, obs_dim, device=device)
        self.actions = torch.zeros(num_steps, self.B, act_dim, device=device)
        self.log_probs = torch.zeros(num_steps, self.B, device=device)
        self.rewards = torch.zeros(num_steps, self.B, device=device)
        self.values = torch.zeros(num_steps, self.B, device=device)
        self.dones = torch.zeros(num_steps, self.B, device=device)
        self.ptr = 0

    def add(self, obs, action, log_prob, reward, value, done):
        i = self.ptr
        self.obs[i] = obs
        self.actions[i] = action
        self.log_probs[i] = log_prob
        self.rewards[i] = reward
        self.values[i] = value
        self.dones[i] = done
        self.ptr += 1

    def reset(self):
        self.ptr = 0
