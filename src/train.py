"""Training script for RL Autonomous Driving project."""

import argparse
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import torch
import gymnasium as gym
from collections import deque

from src.utils.config import Config
from src.envs.environment import make_env, get_env_info
from src.agents.ppo import PPOAgent
from src.utils.logger import setup_logging, TensorBoardLogger, WandBLogger


class Trainer:
    """Main trainer class for RL agents."""
    
    def __init__(self, config: Config):
        """Initialize trainer.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.device = self._get_device()
        
        # Setup logging
        self.logger = setup_logging(config.get('logging.log_dir', './logs'))
        
        # Initialize environment
        self.env = self._create_environment()
        self.eval_env = self._create_environment()
        
        # Get environment info
        env_info = get_env_info(self.env)
        obs_shape = env_info['observation_space'].shape
        action_dim = env_info['action_space'].shape[0] if hasattr(env_info['action_space'], 'shape') else env_info['action_space'].n
        
        # Initialize agent
        self.agent = self._create_agent(obs_shape, action_dim)
        
        # Initialize loggers
        self.tb_logger = None
        self.wandb_logger = None
        
        if config.get('logging.use_tensorboard', True):
            self.tb_logger = TensorBoardLogger(config.get('logging.log_dir', './logs'))
        
        if config.get('logging.use_wandb', False):
            self.wandb_logger = WandBLogger(
                project=config.get('logging.wandb_project', 'rl-autonomous-driving'),
                config=config._config
            )
        
        # Training statistics
        self.training_stats = {
            'episode_rewards': deque(maxlen=100),
            'episode_lengths': deque(maxlen=100),
            'eval_rewards': deque(maxlen=100),
            'timesteps': 0,
            'episodes': 0
        }
        
        # Create directories
        self._create_directories()
    
    def _get_device(self) -> torch.device:
        """Get PyTorch device."""
        device_config = self.config.get('device', 'auto')
        
        if device_config == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                return torch.device('cpu')
        else:
            return torch.device(device_config)
    
    def _create_environment(self) -> gym.Env:
        """Create environment."""
        env_name = self.config.get('environment.name', 'CartPole-v1')
        env_kwargs = {
            'continuous': self.config.get('environment.continuous', False),
            'domain_randomize': self.config.get('environment.domain_randomize', False),
            'render_mode': self.config.get('environment.render_mode', None)
        }
        
        # Remove None values
        env_kwargs = {k: v for k, v in env_kwargs.items() if v is not None}
        
        return make_env(env_name, **env_kwargs)
    
    def _create_agent(self, obs_shape: tuple, action_dim: int) -> PPOAgent:
        """Create agent."""
        return PPOAgent(
            obs_shape=obs_shape,
            action_dim=action_dim,
            config=self.config,
            device=self.device
        )
    
    def _create_directories(self) -> None:
        """Create necessary directories."""
        directories = [
            self.config.get('logging.log_dir', './logs'),
            self.config.get('visualization.plot_dir', './plots'),
            './checkpoints',
            './models'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def train(self) -> None:
        """Main training loop."""
        total_timesteps = self.config.get('training.total_timesteps', 100000)
        eval_freq = self.config.get('training.eval_freq', 10000)
        save_freq = self.config.get('training.save_freq', 50000)
        log_freq = self.config.get('training.log_freq', 1000)
        
        self.logger.info(f"Starting training for {total_timesteps} timesteps")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Environment: {self.env.spec.id if self.env.spec else 'Unknown'}")
        
        start_time = time.time()
        
        while self.training_stats['timesteps'] < total_timesteps:
            # Training episode
            episode_stats = self._train_episode()
            
            # Update statistics
            self.training_stats['episode_rewards'].append(episode_stats['total_reward'])
            self.training_stats['episode_lengths'].append(episode_stats['length'])
            self.training_stats['timesteps'] += episode_stats['length']
            self.training_stats['episodes'] += 1
            
            # Logging
            if self.training_stats['timesteps'] % log_freq == 0:
                self._log_training_stats()
            
            # Evaluation
            if self.training_stats['timesteps'] % eval_freq == 0:
                eval_stats = self._evaluate()
                self.training_stats['eval_rewards'].append(eval_stats['mean_reward'])
                self._log_eval_stats(eval_stats)
            
            # Save model
            if self.training_stats['timesteps'] % save_freq == 0:
                self._save_checkpoint()
        
        # Final evaluation and save
        final_eval = self._evaluate()
        self._save_checkpoint(is_final=True)
        
        training_time = time.time() - start_time
        self.logger.info(f"Training completed in {training_time:.2f} seconds")
        self.logger.info(f"Final evaluation reward: {final_eval['mean_reward']:.2f}")
    
    def _train_episode(self) -> Dict[str, Any]:
        """Train for one episode."""
        obs, _ = self.env.reset()
        obs = torch.FloatTensor(obs).unsqueeze(0)
        
        total_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            # Get action
            action, log_prob, value = self.agent.get_action(obs)
            
            # Take step
            next_obs, reward, terminated, truncated, _ = self.env.step(action.squeeze().cpu().numpy())
            done = terminated or truncated
            
            # Store transition
            self.agent.store_transition(obs, action, reward, value, log_prob)
            
            # Update agent if buffer is full
            if self.agent.buffer.size >= self.agent.buffer_size:
                next_obs_tensor = torch.FloatTensor(next_obs).unsqueeze(0)
                update_stats = self.agent.update(next_obs_tensor)
                
                # Log update stats
                if update_stats and self.tb_logger:
                    for key, value in update_stats.items():
                        self.tb_logger.log_scalar(f'training/{key}', value, self.training_stats['timesteps'])
            
            obs = torch.FloatTensor(next_obs).unsqueeze(0)
            total_reward += reward
            episode_length += 1
        
        return {
            'total_reward': total_reward,
            'length': episode_length
        }
    
    def _evaluate(self) -> Dict[str, Any]:
        """Evaluate agent."""
        n_eval_episodes = self.config.get('training.n_eval_episodes', 5)
        eval_rewards = []
        eval_lengths = []
        
        for _ in range(n_eval_episodes):
            obs, _ = self.eval_env.reset()
            obs = torch.FloatTensor(obs).unsqueeze(0)
            
            total_reward = 0
            episode_length = 0
            done = False
            
            while not done:
                action, _, _ = self.agent.get_action(obs, deterministic=True)
                next_obs, reward, terminated, truncated, _ = self.eval_env.step(action.squeeze().cpu().numpy())
                done = terminated or truncated
                
                obs = torch.FloatTensor(next_obs).unsqueeze(0)
                total_reward += reward
                episode_length += 1
            
            eval_rewards.append(total_reward)
            eval_lengths.append(episode_length)
        
        return {
            'mean_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'mean_length': np.mean(eval_lengths),
            'std_length': np.std(eval_lengths)
        }
    
    def _log_training_stats(self) -> None:
        """Log training statistics."""
        if not self.training_stats['episode_rewards']:
            return
        
        mean_reward = np.mean(self.training_stats['episode_rewards'])
        mean_length = np.mean(self.training_stats['episode_lengths'])
        
        self.logger.info(
            f"Timesteps: {self.training_stats['timesteps']}, "
            f"Episodes: {self.training_stats['episodes']}, "
            f"Mean Reward: {mean_reward:.2f}, "
            f"Mean Length: {mean_length:.2f}"
        )
        
        # TensorBoard logging
        if self.tb_logger:
            self.tb_logger.log_scalar('training/mean_reward', mean_reward, self.training_stats['timesteps'])
            self.tb_logger.log_scalar('training/mean_length', mean_length, self.training_stats['timesteps'])
            self.tb_logger.log_scalar('training/episodes', self.training_stats['episodes'], self.training_stats['timesteps'])
        
        # WandB logging
        if self.wandb_logger:
            self.wandb_logger.log({
                'training/mean_reward': mean_reward,
                'training/mean_length': mean_length,
                'training/episodes': self.training_stats['episodes'],
                'training/timesteps': self.training_stats['timesteps']
            })
    
    def _log_eval_stats(self, eval_stats: Dict[str, Any]) -> None:
        """Log evaluation statistics."""
        self.logger.info(
            f"Evaluation - Mean Reward: {eval_stats['mean_reward']:.2f} ± {eval_stats['std_reward']:.2f}, "
            f"Mean Length: {eval_stats['mean_length']:.2f} ± {eval_stats['std_length']:.2f}"
        )
        
        # TensorBoard logging
        if self.tb_logger:
            self.tb_logger.log_scalar('eval/mean_reward', eval_stats['mean_reward'], self.training_stats['timesteps'])
            self.tb_logger.log_scalar('eval/std_reward', eval_stats['std_reward'], self.training_stats['timesteps'])
            self.tb_logger.log_scalar('eval/mean_length', eval_stats['mean_length'], self.training_stats['timesteps'])
        
        # WandB logging
        if self.wandb_logger:
            self.wandb_logger.log({
                'eval/mean_reward': eval_stats['mean_reward'],
                'eval/std_reward': eval_stats['std_reward'],
                'eval/mean_length': eval_stats['mean_length'],
                'eval/std_length': eval_stats['std_length']
            })
    
    def _save_checkpoint(self, is_final: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint_dir = Path('./checkpoints')
        checkpoint_dir.mkdir(exist_ok=True)
        
        suffix = '_final' if is_final else f'_step_{self.training_stats["timesteps"]}'
        checkpoint_path = checkpoint_dir / f'agent{suffix}.pt'
        
        self.agent.save(str(checkpoint_path))
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    def close(self) -> None:
        """Close environments and loggers."""
        self.env.close()
        self.eval_env.close()
        
        if self.tb_logger:
            self.tb_logger.close()
        
        if self.wandb_logger:
            self.wandb_logger.close()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Train RL agent for autonomous driving')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda, mps)')
    parser.add_argument('--env', type=str, default=None,
                       help='Environment name override')
    parser.add_argument('--timesteps', type=int, default=None,
                       help='Total timesteps override')
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config(args.config)
    
    # Override config with command line arguments
    if args.device != 'auto':
        config.set('device', args.device)
    if args.env:
        config.set('environment.name', args.env)
    if args.timesteps:
        config.set('training.total_timesteps', args.timesteps)
    
    # Create and run trainer
    trainer = Trainer(config)
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        trainer.logger.info("Training interrupted by user")
    finally:
        trainer.close()


if __name__ == '__main__':
    main()
