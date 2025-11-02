"""Environment wrapper and utilities for RL Autonomous Driving project."""

import gymnasium as gym
import numpy as np
import torch
from torchvision import transforms
from typing import Tuple, Optional, Union, Dict, Any
import logging


class PreprocessingWrapper(gym.ObservationWrapper):
    """Wrapper for preprocessing observations (resize, grayscale, normalize)."""
    
    def __init__(self, env: gym.Env, target_size: Tuple[int, int] = (64, 64), 
                 grayscale: bool = True, normalize: bool = True):
        """Initialize preprocessing wrapper.
        
        Args:
            env: Gymnasium environment
            target_size: Target size for resizing (height, width)
            grayscale: Whether to convert to grayscale
            normalize: Whether to normalize pixel values
        """
        super().__init__(env)
        self.target_size = target_size
        self.grayscale = grayscale
        self.normalize = normalize
        
        # Define transform pipeline
        transform_list = []
        
        if grayscale and len(self.observation_space.shape) == 3:
            transform_list.append(transforms.Grayscale())
        
        transform_list.extend([
            transforms.Resize(target_size),
            transforms.ToTensor()
        ])
        
        if normalize:
            transform_list.append(transforms.Normalize(mean=[0.5], std=[0.5]))
        
        self.transform = transforms.Compose(transform_list)
        
        # Update observation space
        channels = 1 if grayscale else self.observation_space.shape[2]
        self.observation_space = gym.spaces.Box(
            low=0.0 if not normalize else -1.0,
            high=1.0 if not normalize else 1.0,
            shape=(channels, target_size[0], target_size[1]),
            dtype=np.float32
        )
    
    def observation(self, obs: np.ndarray) -> np.ndarray:
        """Preprocess observation."""
        try:
            # Convert numpy array to PIL Image
            if len(obs.shape) == 3:
                obs = obs.transpose(2, 0, 1)  # HWC to CHW
            obs_tensor = torch.from_numpy(obs).float()
            
            # Apply transforms
            processed = self.transform(obs_tensor)
            return processed.numpy()
        except Exception as e:
            logging.error(f"Error preprocessing observation: {e}")
            return obs


class RewardWrapper(gym.RewardWrapper):
    """Wrapper for reward shaping and normalization."""
    
    def __init__(self, env: gym.Env, reward_scale: float = 1.0, 
                 reward_clip: Optional[Tuple[float, float]] = None):
        """Initialize reward wrapper.
        
        Args:
            env: Gymnasium environment
            reward_scale: Scaling factor for rewards
            reward_clip: Tuple of (min, max) for clipping rewards
        """
        super().__init__(env)
        self.reward_scale = reward_scale
        self.reward_clip = reward_clip
    
    def reward(self, reward: float) -> float:
        """Process reward."""
        reward *= self.reward_scale
        
        if self.reward_clip is not None:
            reward = np.clip(reward, self.reward_clip[0], self.reward_clip[1])
        
        return reward


class ActionWrapper(gym.ActionWrapper):
    """Wrapper for action space modifications."""
    
    def __init__(self, env: gym.Env, action_noise: float = 0.0):
        """Initialize action wrapper.
        
        Args:
            env: Gymnasium environment
            action_noise: Standard deviation of action noise
        """
        super().__init__(env)
        self.action_noise = action_noise
    
    def action(self, action: np.ndarray) -> np.ndarray:
        """Add noise to actions."""
        if self.action_noise > 0:
            noise = np.random.normal(0, self.action_noise, size=action.shape)
            action = action + noise
            
            # Clip to action space bounds
            if isinstance(self.action_space, gym.spaces.Box):
                action = np.clip(action, self.action_space.low, self.action_space.high)
        
        return action


def make_env(env_name: str, **kwargs) -> gym.Env:
    """Create and configure environment.
    
    Args:
        env_name: Name of the environment
        **kwargs: Additional environment arguments
        
    Returns:
        Configured environment
    """
    try:
        env = gym.make(env_name, **kwargs)
        
        # Apply wrappers based on environment type
        if "CarRacing" in env_name:
            env = PreprocessingWrapper(env, target_size=(64, 64), grayscale=True)
            env = RewardWrapper(env, reward_scale=0.01)
            env = ActionWrapper(env, action_noise=0.1)
        elif "Atari" in env_name:
            env = PreprocessingWrapper(env, target_size=(84, 84), grayscale=True)
            env = RewardWrapper(env, reward_clip=(-1.0, 1.0))
        
        return env
    except Exception as e:
        logging.error(f"Error creating environment {env_name}: {e}")
        # Fallback to CartPole
        logging.info("Falling back to CartPole-v1")
        return gym.make("CartPole-v1")


def get_env_info(env: gym.Env) -> Dict[str, Any]:
    """Get environment information.
    
    Args:
        env: Gymnasium environment
        
    Returns:
        Dictionary with environment information
    """
    return {
        "observation_space": env.observation_space,
        "action_space": env.action_space,
        "reward_range": env.reward_range,
        "spec": env.spec,
        "metadata": env.metadata
    }


class EnvironmentManager:
    """Manager for multiple environments."""
    
    def __init__(self, env_configs: Dict[str, Dict[str, Any]]):
        """Initialize environment manager.
        
        Args:
            env_configs: Dictionary mapping env names to their configs
        """
        self.env_configs = env_configs
        self.envs = {}
    
    def create_env(self, env_name: str) -> gym.Env:
        """Create environment by name.
        
        Args:
            env_name: Name of the environment
            
        Returns:
            Created environment
        """
        if env_name not in self.env_configs:
            raise ValueError(f"Environment {env_name} not found in configs")
        
        config = self.env_configs[env_name]
        return make_env(env_name, **config)
    
    def get_env(self, env_name: str) -> gym.Env:
        """Get or create environment.
        
        Args:
            env_name: Name of the environment
            
        Returns:
            Environment instance
        """
        if env_name not in self.envs:
            self.envs[env_name] = self.create_env(env_name)
        return self.envs[env_name]
    
    def close_all(self) -> None:
        """Close all environments."""
        for env in self.envs.values():
            env.close()
        self.envs.clear()
