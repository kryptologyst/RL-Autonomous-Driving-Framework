"""Neural network models for RL agents."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List, Union
import gymnasium as gym


class ActorCritic(nn.Module):
    """Actor-Critic network for continuous action spaces."""
    
    def __init__(self, 
                 obs_shape: Union[List[int], Tuple[int, ...]], 
                 action_dim: int,
                 hidden_size: int = 256,
                 conv_channels: List[int] = None,
                 conv_kernels: List[int] = None,
                 conv_strides: List[int] = None):
        """Initialize Actor-Critic network.
        
        Args:
            obs_shape: Observation space shape
            action_dim: Action space dimension
            hidden_size: Hidden layer size
            conv_channels: List of convolution channel sizes
            conv_kernels: List of convolution kernel sizes
            conv_strides: List of convolution stride sizes
        """
        super().__init__()
        
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        
        # Default convolution parameters
        if conv_channels is None:
            conv_channels = [16, 32]
        if conv_kernels is None:
            conv_kernels = [8, 4]
        if conv_strides is None:
            conv_strides = [4, 2]
        
        # Convolutional layers for image observations
        if len(obs_shape) == 3:  # Image observation
            self.conv = self._build_conv_layers(obs_shape, conv_channels, conv_kernels, conv_strides)
            conv_out_size = self._get_conv_out_size(obs_shape)
        else:  # Vector observation
            self.conv = None
            conv_out_size = np.prod(obs_shape)
        
        # Actor (policy) network
        self.actor = nn.Sequential(
            nn.Linear(conv_out_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            nn.Tanh()  # Continuous actions in [-1, 1]
        )
        
        # Critic (value) network
        self.critic = nn.Sequential(
            nn.Linear(conv_out_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _build_conv_layers(self, obs_shape: Tuple[int, ...], 
                          channels: List[int], kernels: List[int], 
                          strides: List[int]) -> nn.Module:
        """Build convolutional layers."""
        layers = []
        in_channels = obs_shape[0]
        
        for out_channels, kernel_size, stride in zip(channels, kernels, strides):
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride),
                nn.ReLU()
            ])
            in_channels = out_channels
        
        return nn.Sequential(*layers)
    
    def _get_conv_out_size(self, shape: Tuple[int, ...]) -> int:
        """Calculate convolutional output size."""
        dummy_input = torch.zeros(1, *shape)
        conv_out = self.conv(dummy_input)
        return int(np.prod(conv_out.size()))
    
    def _init_weights(self, module: nn.Module) -> None:
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            torch.nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.Conv2d):
            torch.nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            torch.nn.init.constant_(module.bias, 0.0)
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            obs: Observation tensor
            
        Returns:
            Tuple of (action_mean, value)
        """
        if self.conv is not None:
            features = self.conv(obs).view(obs.size(0), -1)
        else:
            features = obs.view(obs.size(0), -1)
        
        action_mean = self.actor(features)
        value = self.critic(features)
        
        return action_mean, value
    
    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action from observation.
        
        Args:
            obs: Observation tensor
            deterministic: Whether to use deterministic action
            
        Returns:
            Tuple of (action, log_prob, value)
        """
        action_mean, value = self.forward(obs)
        
        if deterministic:
            action = action_mean
            log_prob = torch.zeros_like(action_mean)
        else:
            # Add noise for exploration
            std = torch.ones_like(action_mean) * 0.1
            dist = torch.distributions.Normal(action_mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        
        return action, log_prob, value


class DQNNetwork(nn.Module):
    """Deep Q-Network for discrete action spaces."""
    
    def __init__(self, 
                 obs_shape: Union[List[int], Tuple[int, ...]], 
                 action_dim: int,
                 hidden_size: int = 512,
                 dueling: bool = True):
        """Initialize DQN network.
        
        Args:
            obs_shape: Observation space shape
            action_dim: Action space dimension
            hidden_size: Hidden layer size
            dueling: Whether to use dueling architecture
        """
        super().__init__()
        
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.dueling = dueling
        
        # Convolutional layers for image observations
        if len(obs_shape) == 3:  # Image observation
            self.conv = nn.Sequential(
                nn.Conv2d(obs_shape[0], 32, 8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=1),
                nn.ReLU()
            )
            conv_out_size = self._get_conv_out_size(obs_shape)
        else:  # Vector observation
            self.conv = None
            conv_out_size = np.prod(obs_shape)
        
        # Feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(conv_out_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        if dueling:
            # Dueling architecture
            self.value_stream = nn.Linear(hidden_size, 1)
            self.advantage_stream = nn.Linear(hidden_size, action_dim)
        else:
            # Standard DQN
            self.q_network = nn.Linear(hidden_size, action_dim)
        
        self.apply(self._init_weights)
    
    def _get_conv_out_size(self, shape: Tuple[int, ...]) -> int:
        """Calculate convolutional output size."""
        dummy_input = torch.zeros(1, *shape)
        conv_out = self.conv(dummy_input)
        return int(np.prod(conv_out.size()))
    
    def _init_weights(self, module: nn.Module) -> None:
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            torch.nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.Conv2d):
            torch.nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            torch.nn.init.constant_(module.bias, 0.0)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            obs: Observation tensor
            
        Returns:
            Q-values tensor
        """
        if self.conv is not None:
            features = self.conv(obs).view(obs.size(0), -1)
        else:
            features = obs.view(obs.size(0), -1)
        
        features = self.feature_extractor(features)
        
        if self.dueling:
            value = self.value_stream(features)
            advantage = self.advantage_stream(features)
            q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
        else:
            q_values = self.q_network(features)
        
        return q_values


class RainbowNetwork(DQNNetwork):
    """Rainbow DQN network with distributional RL."""
    
    def __init__(self, 
                 obs_shape: Union[List[int], Tuple[int, ...]], 
                 action_dim: int,
                 hidden_size: int = 512,
                 n_atoms: int = 51,
                 v_min: float = -10.0,
                 v_max: float = 10.0):
        """Initialize Rainbow network.
        
        Args:
            obs_shape: Observation space shape
            action_dim: Action space dimension
            hidden_size: Hidden layer size
            n_atoms: Number of atoms for distributional RL
            v_min: Minimum value for value distribution
            v_max: Maximum value for value distribution
        """
        super().__init__(obs_shape, action_dim, hidden_size, dueling=True)
        
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        
        # Distributional value and advantage streams
        self.value_stream = nn.Linear(hidden_size, n_atoms)
        self.advantage_stream = nn.Linear(hidden_size, action_dim * n_atoms)
        
        # Value distribution support
        self.register_buffer('support', torch.linspace(v_min, v_max, n_atoms))
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass for distributional RL.
        
        Args:
            obs: Observation tensor
            
        Returns:
            Q-value distributions tensor
        """
        if self.conv is not None:
            features = self.conv(obs).view(obs.size(0), -1)
        else:
            features = obs.view(obs.size(0), -1)
        
        features = self.feature_extractor(features)
        
        # Distributional streams
        value_dist = self.value_stream(features)  # (batch_size, n_atoms)
        advantage_dist = self.advantage_stream(features)  # (batch_size, action_dim * n_atoms)
        advantage_dist = advantage_dist.view(-1, self.action_dim, self.n_atoms)
        
        # Combine value and advantage distributions
        q_dist = value_dist.unsqueeze(1) + advantage_dist - advantage_dist.mean(dim=1, keepdim=True)
        
        return F.softmax(q_dist, dim=-1)  # (batch_size, action_dim, n_atoms)
    
    def get_q_values(self, obs: torch.Tensor) -> torch.Tensor:
        """Get Q-values from distribution.
        
        Args:
            obs: Observation tensor
            
        Returns:
            Q-values tensor
        """
        q_dist = self.forward(obs)
        q_values = torch.sum(q_dist * self.support.unsqueeze(0).unsqueeze(0), dim=-1)
        return q_values
