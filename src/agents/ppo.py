"""Proximal Policy Optimization (PPO) agent implementation."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import logging

from .models import ActorCritic
from ..utils.config import Config


class PPOBuffer:
    """Buffer for storing PPO experience data."""
    
    def __init__(self, buffer_size: int, obs_shape: Tuple[int, ...], 
                 action_dim: int, device: torch.device):
        """Initialize PPO buffer.
        
        Args:
            buffer_size: Maximum buffer size
            obs_shape: Observation space shape
            action_dim: Action space dimension
            device: PyTorch device
        """
        self.buffer_size = buffer_size
        self.device = device
        
        # Storage tensors
        self.observations = torch.zeros((buffer_size, *obs_shape), device=device)
        self.actions = torch.zeros((buffer_size, action_dim), device=device)
        self.rewards = torch.zeros(buffer_size, device=device)
        self.values = torch.zeros(buffer_size, device=device)
        self.log_probs = torch.zeros(buffer_size, device=device)
        self.advantages = torch.zeros(buffer_size, device=device)
        self.returns = torch.zeros(buffer_size, device=device)
        
        self.ptr = 0
        self.size = 0
    
    def add(self, obs: torch.Tensor, action: torch.Tensor, reward: float,
            value: torch.Tensor, log_prob: torch.Tensor) -> None:
        """Add experience to buffer.
        
        Args:
            obs: Observation
            action: Action taken
            reward: Reward received
            value: Value estimate
            log_prob: Log probability of action
        """
        assert self.size < self.buffer_size, "Buffer overflow"
        
        self.observations[self.ptr] = obs.squeeze()
        self.actions[self.ptr] = action.squeeze()
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value.squeeze()
        self.log_probs[self.ptr] = log_prob.squeeze()
        
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)
    
    def compute_advantages(self, gamma: float, lam: float, 
                         next_value: torch.Tensor) -> None:
        """Compute advantages using GAE.
        
        Args:
            gamma: Discount factor
            lam: GAE lambda parameter
            next_value: Value estimate for next state
        """
        advantages = torch.zeros_like(self.rewards)
        last_advantage = 0
        
        for t in reversed(range(self.size)):
            if t == self.size - 1:
                next_non_terminal = 0.0
                next_value_t = next_value
            else:
                next_non_terminal = 1.0
                next_value_t = self.values[t + 1]
            
            delta = self.rewards[t] + gamma * next_value_t * next_non_terminal - self.values[t]
            advantages[t] = last_advantage = delta + gamma * lam * next_non_terminal * last_advantage
        
        self.advantages = advantages
        self.returns = advantages + self.values[:self.size]
    
    def get_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Get random batch from buffer.
        
        Args:
            batch_size: Size of batch to return
            
        Returns:
            Dictionary of batched tensors
        """
        indices = torch.randperm(self.size, device=self.device)[:batch_size]
        
        return {
            'observations': self.observations[indices],
            'actions': self.actions[indices],
            'advantages': self.advantages[indices],
            'returns': self.returns[indices],
            'old_log_probs': self.log_probs[indices]
        }
    
    def clear(self) -> None:
        """Clear buffer."""
        self.ptr = 0
        self.size = 0


class PPOAgent:
    """Proximal Policy Optimization agent."""
    
    def __init__(self, 
                 obs_shape: Tuple[int, ...],
                 action_dim: int,
                 config: Config,
                 device: Optional[torch.device] = None):
        """Initialize PPO agent.
        
        Args:
            obs_shape: Observation space shape
            action_dim: Action space dimension
            config: Configuration object
            device: PyTorch device
        """
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Hyperparameters
        self.lr = config.get('agent.learning_rate', 3e-4)
        self.gamma = config.get('agent.gamma', 0.99)
        self.lam = config.get('ppo.lam', 0.95)
        self.clip_range = config.get('ppo.clip_range', 0.2)
        self.value_loss_coef = config.get('ppo.value_loss_coef', 0.5)
        self.entropy_coef = config.get('ppo.entropy_coef', 0.01)
        self.max_grad_norm = config.get('ppo.max_grad_norm', 0.5)
        self.n_epochs = config.get('ppo.n_epochs', 10)
        self.batch_size = config.get('agent.batch_size', 64)
        self.buffer_size = config.get('agent.buffer_size', 2048)
        
        # Networks
        self.actor_critic = ActorCritic(
            obs_shape=obs_shape,
            action_dim=action_dim,
            hidden_size=config.get('model.hidden_size', 256),
            conv_channels=config.get('model.conv_channels', [16, 32]),
            conv_kernels=config.get('model.conv_kernels', [8, 4]),
            conv_strides=config.get('model.conv_strides', [4, 2])
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.lr)
        
        # Buffer
        self.buffer = PPOBuffer(self.buffer_size, obs_shape, action_dim, self.device)
        
        # Training statistics
        self.training_stats = {
            'actor_loss': deque(maxlen=100),
            'critic_loss': deque(maxlen=100),
            'entropy': deque(maxlen=100),
            'kl_divergence': deque(maxlen=100),
            'clip_fraction': deque(maxlen=100)
        }
    
    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action from observation.
        
        Args:
            obs: Observation tensor
            deterministic: Whether to use deterministic action
            
        Returns:
            Tuple of (action, log_prob, value)
        """
        obs = obs.to(self.device)
        return self.actor_critic.get_action(obs, deterministic)
    
    def store_transition(self, obs: torch.Tensor, action: torch.Tensor, 
                        reward: float, value: torch.Tensor, 
                        log_prob: torch.Tensor) -> None:
        """Store transition in buffer.
        
        Args:
            obs: Observation
            action: Action taken
            reward: Reward received
            value: Value estimate
            log_prob: Log probability of action
        """
        self.buffer.add(obs, action, reward, value, log_prob)
    
    def update(self, next_obs: torch.Tensor) -> Dict[str, float]:
        """Update agent using PPO.
        
        Args:
            next_obs: Next observation for value estimation
            
        Returns:
            Dictionary of training statistics
        """
        if self.buffer.size < self.batch_size:
            return {}
        
        # Compute advantages
        with torch.no_grad():
            _, _, next_value = self.actor_critic.get_action(next_obs, deterministic=True)
        
        self.buffer.compute_advantages(self.gamma, self.lam, next_value)
        
        # Training statistics
        actor_losses = []
        critic_losses = []
        entropies = []
        kl_divergences = []
        clip_fractions = []
        
        # PPO updates
        for _ in range(self.n_epochs):
            batch = self.buffer.get_batch(self.batch_size)
            
            # Forward pass
            action_means, values = self.actor_critic(batch['observations'])
            
            # Compute log probabilities
            std = torch.ones_like(action_means) * 0.1
            dist = torch.distributions.Normal(action_means, std)
            new_log_probs = dist.log_prob(batch['actions']).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1).mean()
            
            # Compute ratios
            ratio = torch.exp(new_log_probs - batch['old_log_probs'])
            
            # Compute losses
            surr1 = ratio * batch['advantages']
            surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * batch['advantages']
            actor_loss = -torch.min(surr1, surr2).mean()
            
            critic_loss = nn.MSELoss()(values.squeeze(), batch['returns'])
            
            # Total loss
            total_loss = actor_loss + self.value_loss_coef * critic_loss - self.entropy_coef * entropy
            
            # Optimize
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            # Statistics
            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
            entropies.append(entropy.item())
            
            # KL divergence
            kl_div = (batch['old_log_probs'] - new_log_probs).mean().item()
            kl_divergences.append(kl_div)
            
            # Clip fraction
            clip_frac = ((ratio - 1.0).abs() > self.clip_range).float().mean().item()
            clip_fractions.append(clip_frac)
        
        # Update statistics
        stats = {
            'actor_loss': np.mean(actor_losses),
            'critic_loss': np.mean(critic_losses),
            'entropy': np.mean(entropies),
            'kl_divergence': np.mean(kl_divergences),
            'clip_fraction': np.mean(clip_fractions)
        }
        
        for key, value in stats.items():
            self.training_stats[key].append(value)
        
        # Clear buffer
        self.buffer.clear()
        
        return stats
    
    def save(self, filepath: str) -> None:
        """Save agent state.
        
        Args:
            filepath: Path to save file
        """
        torch.save({
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config._config,
            'training_stats': dict(self.training_stats)
        }, filepath)
    
    def load(self, filepath: str) -> None:
        """Load agent state.
        
        Args:
            filepath: Path to load file
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'training_stats' in checkpoint:
            for key, values in checkpoint['training_stats'].items():
                self.training_stats[key] = deque(values, maxlen=100)
    
    def get_training_stats(self) -> Dict[str, float]:
        """Get current training statistics.
        
        Returns:
            Dictionary of training statistics
        """
        return {key: np.mean(values) if values else 0.0 
                for key, values in self.training_stats.items()}
