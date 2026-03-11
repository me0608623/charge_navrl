# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
NavRL-style neural network models for SKRL.

Reference: NavRL (https://github.com/Zhefan-Xu/NavRL)
NavRL: Learning Safe Flight in Dynamic Environments (IEEE RA-L 2025)

Architecture:
1. CNN Feature Extractor for LiDAR (static obstacles)
2. MLP for Dynamic Obstacles
3. Shared Encoder with feature fusion
4. Actor/Critic heads
"""

from typing import Any, Tuple, Optional

import torch
import torch.nn as nn
from einops import rearrange

from skrl.models.torch import Model, GaussianMixin, DeterministicMixin


def make_mlp(num_units: list, activation: str = "leaky_relu", use_layer_norm: bool = True) -> nn.Sequential:
    """Create an MLP with the specified architecture.

    Args:
        num_units: List of layer sizes (e.g., [256, 128] creates Linear(256) -> Linear(128))
        activation: Activation function to use ("leaky_relu", "elu", "relu")
        use_layer_norm: Whether to add LayerNorm after each layer

    Returns:
        nn.Sequential: The MLP module
    """
    layers = []

    activation_map = {
        "leaky_relu": nn.LeakyReLU(),
        "elu": nn.ELU(),
        "relu": nn.ReLU(),
    }

    act = activation_map.get(activation, nn.LeakyReLU())

    for i, n in enumerate(num_units):
        layers.append(nn.LazyLinear(n))
        layers.append(act)
        if use_layer_norm:
            layers.append(nn.LayerNorm(n))

    return nn.Sequential(*layers)


class CNNFeatureExtractor(nn.Module):
    """CNN feature extractor for LiDAR data (static obstacles).

    Based on NavRL's CNN architecture:
    - Conv2d(4, kernel=[5,3], padding=[2,1]) -> ELU
    - Conv2d(16, kernel=[5,3], stride=[2,1], padding=[2,1]) -> ELU
    - Conv2d(16, kernel=[5,3], stride=[2,2], padding=[2,1]) -> ELU
    - Flatten -> Linear(128) -> LayerNorm(128)

    Input shape: (batch, 1, H, W) or (batch, H, W) for LiDAR
    Output shape: (batch, 128)
    """

    def __init__(
        self,
        input_channels: int = 1,
        output_dim: int = 128,
        lidar_horizontal: int = 72,
        lidar_vertical: int = 1,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.lidar_horizontal = lidar_horizontal
        self.lidar_vertical = lidar_vertical

        # CNN layers (from NavRL)
        self.conv1 = nn.LazyConv2d(
            out_channels=4,
            kernel_size=[5, 3],
            padding=[2, 1],
        )
        self.act1 = nn.ELU()

        self.conv2 = nn.LazyConv2d(
            out_channels=16,
            kernel_size=[5, 3],
            stride=[2, 1],
            padding=[2, 1],
        )
        self.act2 = nn.ELU()

        self.conv3 = nn.LazyConv2d(
            out_channels=16,
            kernel_size=[5, 3],
            stride=[2, 2],
            padding=[2, 1],
        )
        self.act3 = nn.ELU()

        # Flatten and linear projection
        self.fc = nn.LazyLinear(output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, lidar_horizontal) or (batch, 1, H, W)

        Returns:
            Feature tensor of shape (batch, 128)
        """
        batch_size = x.shape[0]

        # Reshape if needed: (batch, H) -> (batch, 1, H, 1)
        if x.dim() == 2:
            # (batch, H) -> (batch, 1, H, 1)
            x = x.unsqueeze(1).unsqueeze(-1)
        elif x.dim() == 3:
            # (batch, 1, H) -> (batch, 1, H, 1)
            x = x.unsqueeze(-1)

        # Ensure 4D input for Conv2d
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input (batch, C, H, W), got shape {x.shape}")

        # CNN forward pass
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = self.act3(self.conv3(x))

        # Flatten
        x = rearrange(x, "n c w h -> n (c w h)")

        # Linear projection
        x = self.fc(x)
        x = self.layer_norm(x)

        return x


class DynamicObstacleEncoder(nn.Module):
    """MLP encoder for dynamic obstacle information.

    Based on NavRL's dynamic obstacle encoder:
    - Rearrange/Flatten -> Linear(128) -> LeakyReLU -> LayerNorm(128)
    - Linear(64) -> LeakyReLU -> LayerNorm(64)

    Input shape: (batch, num_obstacles * features_per_obstacle)
    Output shape: (batch, 64)
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 64):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.LeakyReLU(),
            nn.LayerNorm(hidden_dim),
            nn.LazyLinear(output_dim),
            nn.LeakyReLU(),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, num_obstacles * features)

        Returns:
            Feature tensor of shape (batch, 64)
        """
        return self.mlp(x)


class SharedEncoder(nn.Module):
    """Shared feature encoder that fuses all modalities.

    Architecture:
    1. CNN features from LiDAR (128 dim)
    2. State features (8 dim - goal, velocity, etc.)
    3. Dynamic obstacle features (64 dim)
    4. Concatenate -> MLP [256, 256] -> output (256 dim)

    Input:
    - lidar_features: (batch, 128)
    - state_features: (batch, state_dim)
    - dynamic_features: (batch, 64)

    Output:
    - Fused features: (batch, 256)
    """

    def __init__(self, state_dim: int = 8, output_dim: int = 256):
        super().__init__()

        # Calculate input dimension after concatenation
        # lidar (128) + state + dynamic (64)
        fusion_input_dim = 128 + state_dim + 64

        self.shared_mlp = make_mlp([output_dim, output_dim], activation="leaky_relu")

    def forward(
        self,
        lidar_features: torch.Tensor,
        state_features: torch.Tensor,
        dynamic_features: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            lidar_features: CNN features from LiDAR, shape (batch, 128)
            state_features: State observations, shape (batch, state_dim)
            dynamic_features: Dynamic obstacle features, shape (batch, 64)

        Returns:
            Fused features, shape (batch, 256)
        """
        # Concatenate all features
        fused = torch.cat([lidar_features, state_features, dynamic_features], dim=-1)

        # Shared MLP
        output = self.shared_mlp(fused)

        return output


class NavRLPolicy(GaussianMixin, Model):
    """NavRL-style Policy network with CNN for LiDAR.

    Architecture:
    1. CNN Feature Extractor (LiDAR -> 128)
    2. Dynamic Obstacle Encoder (if enabled)
    3. State pass-through
    4. Shared Encoder (fusion -> 256)
    5. Actor head (256 -> action_dim)

    The network processes multi-modal observations and outputs action parameters.
    """

    def __init__(
        self,
        observation_space: Any,
        action_space: Any,
        device: Optional[torch.device] = None,
        # Network configuration
        lidar_dim: int = 72,
        state_dim: int = 8,
        dynamic_dim: int = 0,  # 0 if no dynamic obstacles
        shared_dim: int = 256,
        **kwargs,
    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            **kwargs,
        )

        # Network components
        self.lidar_dim = lidar_dim
        self.state_dim = state_dim
        self.dynamic_dim = dynamic_dim

        # CNN for LiDAR
        self.cnn_encoder = CNNFeatureExtractor(output_dim=128, lidar_horizontal=lidar_dim)

        # MLP for dynamic obstacles (if enabled)
        if dynamic_dim > 0:
            self.dynamic_encoder = DynamicObstacleEncoder(input_dim=dynamic_dim)
        else:
            # Create a dummy encoder that outputs zeros
            self.dynamic_encoder = nn.Sequential(
                nn.LazyLinear(64),
                nn.LeakyReLU(),
                nn.LayerNorm(64),
            )

        # Shared encoder
        self.shared_encoder = SharedEncoder(state_dim=state_dim, output_dim=shared_dim)

        # Actor head (output mean and log_std)
        self.action_dim = action_space.shape[0] if hasattr(action_space, 'shape') else action_space
        self.actor_head = nn.LazyLinear(self.action_dim * 2)  # mean and log_std

    def compute(self, inputs: torch.Tensor, role: str = "") -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute action parameters (mean and log_std).

        Args:
            inputs: Observation tensor, shape (batch, obs_dim)
            role: Role (not used, for compatibility)

        Returns:
            Tuple of (mean, log_std) for action distribution
        """
        batch_size = inputs.shape[0]

        # Parse observations (assuming flattened format)
        # Expected order: [lidar (72), state (8), dynamic (0 or N)]
        lidar_obs = inputs[:, :self.lidar_dim]
        state_obs = inputs[:, self.lidar_dim:self.lidar_dim + self.state_dim]

        if self.dynamic_dim > 0:
            dynamic_start = self.lidar_dim + self.state_dim
            dynamic_obs = inputs[:, dynamic_start:dynamic_start + self.dynamic_dim]
            dynamic_features = self.dynamic_encoder(dynamic_obs)
        else:
            # Use dummy input for dynamic encoder
            dummy_dynamic = torch.zeros(batch_size, 1, device=inputs.device)
            dynamic_features = self.dynamic_encoder(dummy_obs := dummy_dynamic)

        # Encode LiDAR with CNN
        lidar_features = self.cnn_encoder(lidar_obs)

        # Fuse features
        fused_features = self.shared_encoder(lidar_features, state_obs, dynamic_features)

        # Actor head
        output = self.actor_head(fused_features)
        mean, log_std = torch.chunk(output, 2, dim=-1)

        return mean, log_std


class NavRLValue(DeterministicMixin, Model):
    """NavRL-style Value network with CNN for LiDAR.

    Uses the same architecture as the policy but outputs a scalar value.
    Can use privileged information (larger observation space).
    """

    def __init__(
        self,
        observation_space: Any,
        action_space: Any,
        device: Optional[torch.device] = None,
        # Network configuration
        lidar_dim: int = 72,
        state_dim: int = 8,
        dynamic_dim: int = 0,
        shared_dim: int = 256,
        **kwargs,
    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            **kwargs,
        )

        # Network components (same as policy)
        self.lidar_dim = lidar_dim
        self.state_dim = state_dim
        self.dynamic_dim = dynamic_dim

        # CNN for LiDAR
        self.cnn_encoder = CNNFeatureExtractor(output_dim=128, lidar_horizontal=lidar_dim)

        # MLP for dynamic obstacles
        if dynamic_dim > 0:
            self.dynamic_encoder = DynamicObstacleEncoder(input_dim=dynamic_dim)
        else:
            self.dynamic_encoder = nn.Sequential(
                nn.LazyLinear(64),
                nn.LeakyReLU(),
                nn.LayerNorm(64),
            )

        # Shared encoder
        self.shared_encoder = SharedEncoder(state_dim=state_dim, output_dim=shared_dim)

        # Critic head (output scalar value)
        self.critic_head = nn.LazyLinear(1)

    def compute(self, inputs: torch.Tensor, role: str = "") -> torch.Tensor:
        """Compute state value.

        Args:
            inputs: Observation tensor, shape (batch, obs_dim)
            role: Role (not used, for compatibility)

        Returns:
            State value, shape (batch, 1)
        """
        batch_size = inputs.shape[0]

        # Parse observations
        lidar_obs = inputs[:, :self.lidar_dim]
        state_obs = inputs[:, self.lidar_dim:self.lidar_dim + self.state_dim]

        if self.dynamic_dim > 0:
            dynamic_start = self.lidar_dim + self.state_dim
            dynamic_obs = inputs[:, dynamic_start:dynamic_start + self.dynamic_dim]
            dynamic_features = self.dynamic_encoder(dynamic_obs)
        else:
            dummy_dynamic = torch.zeros(batch_size, 1, device=inputs.device)
            dynamic_features = self.dynamic_encoder(dummy_dynamic)

        # Encode LiDAR with CNN
        lidar_features = self.cnn_encoder(lidar_obs)

        # Fuse features
        fused_features = self.shared_encoder(lidar_features, state_obs, dynamic_features)

        # Critic head
        value = self.critic_head(fused_features)

        return value


__all__ = [
    "make_mlp",
    "CNNFeatureExtractor",
    "DynamicObstacleEncoder",
    "SharedEncoder",
    "NavRLPolicy",
    "NavRLValue",
]
