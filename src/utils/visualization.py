"""Visualization utilities for RL Autonomous Driving project."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from collections import deque
import logging

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class TrainingVisualizer:
    """Visualizer for training progress and results."""
    
    def __init__(self, save_dir: str = './plots'):
        """Initialize visualizer.
        
        Args:
            save_dir: Directory to save plots
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
    
    def plot_learning_curves(self, 
                            episode_rewards: List[float],
                            eval_rewards: Optional[List[float]] = None,
                            window_size: int = 100,
                            save_name: str = 'learning_curves.png') -> None:
        """Plot learning curves.
        
        Args:
            episode_rewards: List of episode rewards
            eval_rewards: List of evaluation rewards (optional)
            window_size: Window size for moving average
            save_name: Name to save the plot
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        episodes = range(len(episode_rewards))
        
        # Plot raw rewards
        ax.plot(episodes, episode_rewards, alpha=0.3, color='blue', label='Episode Rewards')
        
        # Plot moving average
        if len(episode_rewards) >= window_size:
            moving_avg = self._moving_average(episode_rewards, window_size)
            ax.plot(episodes[window_size-1:], moving_avg, color='blue', linewidth=2, 
                   label=f'Moving Average ({window_size})')
        
        # Plot evaluation rewards if provided
        if eval_rewards:
            eval_episodes = np.linspace(0, len(episode_rewards)-1, len(eval_rewards))
            ax.plot(eval_episodes, eval_rewards, color='red', linewidth=2, 
                   marker='o', markersize=4, label='Evaluation Rewards')
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Training Progress')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved learning curves plot: {save_name}")
    
    def plot_training_stats(self, 
                           stats: Dict[str, List[float]],
                           save_name: str = 'training_stats.png') -> None:
        """Plot training statistics.
        
        Args:
            stats: Dictionary of training statistics
            save_name: Name to save the plot
        """
        n_stats = len(stats)
        if n_stats == 0:
            return
        
        fig, axes = plt.subplots(n_stats, 1, figsize=(12, 4 * n_stats))
        if n_stats == 1:
            axes = [axes]
        
        for i, (name, values) in enumerate(stats.items()):
            if not values:
                continue
            
            episodes = range(len(values))
            axes[i].plot(episodes, values, linewidth=2)
            axes[i].set_xlabel('Episode')
            axes[i].set_ylabel(name.replace('_', ' ').title())
            axes[i].set_title(f'{name.replace("_", " ").title()} Over Time')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved training stats plot: {save_name}")
    
    def plot_reward_distribution(self, 
                                rewards: List[float],
                                bins: int = 50,
                                save_name: str = 'reward_distribution.png') -> None:
        """Plot reward distribution.
        
        Args:
            rewards: List of rewards
            bins: Number of bins for histogram
            save_name: Name to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histogram
        ax1.hist(rewards, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Reward')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Reward Distribution')
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        ax2.boxplot(rewards, patch_artist=True, 
                   boxprops=dict(facecolor='lightblue', alpha=0.7))
        ax2.set_ylabel('Reward')
        ax2.set_title('Reward Box Plot')
        ax2.grid(True, alpha=0.3)
        
        # Add statistics
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        ax1.axvline(mean_reward, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_reward:.2f}')
        ax1.legend()
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved reward distribution plot: {save_name}")
    
    def plot_action_distribution(self, 
                                actions: np.ndarray,
                                action_names: Optional[List[str]] = None,
                                save_name: str = 'action_distribution.png') -> None:
        """Plot action distribution.
        
        Args:
            actions: Array of actions
            action_names: Names for action dimensions
            save_name: Name to save the plot
        """
        if len(actions.shape) == 1:
            actions = actions.reshape(-1, 1)
        
        n_actions = actions.shape[1]
        if action_names is None:
            action_names = [f'Action {i}' for i in range(n_actions)]
        
        fig, axes = plt.subplots(1, n_actions, figsize=(5 * n_actions, 5))
        if n_actions == 1:
            axes = [axes]
        
        for i in range(n_actions):
            axes[i].hist(actions[:, i], bins=30, alpha=0.7, color=f'C{i}', edgecolor='black')
            axes[i].set_xlabel(action_names[i])
            axes[i].set_ylabel('Frequency')
            axes[i].set_title(f'{action_names[i]} Distribution')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved action distribution plot: {save_name}")
    
    def plot_episode_lengths(self, 
                            lengths: List[int],
                            window_size: int = 100,
                            save_name: str = 'episode_lengths.png') -> None:
        """Plot episode lengths.
        
        Args:
            lengths: List of episode lengths
            window_size: Window size for moving average
            save_name: Name to save the plot
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        episodes = range(len(lengths))
        
        # Plot raw lengths
        ax.plot(episodes, lengths, alpha=0.3, color='green', label='Episode Lengths')
        
        # Plot moving average
        if len(lengths) >= window_size:
            moving_avg = self._moving_average(lengths, window_size)
            ax.plot(episodes[window_size-1:], moving_avg, color='green', linewidth=2,
                   label=f'Moving Average ({window_size})')
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Length')
        ax.set_title('Episode Lengths Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved episode lengths plot: {save_name}")
    
    def plot_comparison(self, 
                       results: Dict[str, List[float]],
                       title: str = 'Algorithm Comparison',
                       save_name: str = 'comparison.png') -> None:
        """Plot comparison of different algorithms or runs.
        
        Args:
            results: Dictionary mapping algorithm names to reward lists
            title: Plot title
            save_name: Name to save the plot
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for name, rewards in results.items():
            episodes = range(len(rewards))
            
            # Plot raw rewards with transparency
            ax.plot(episodes, rewards, alpha=0.3, label=f'{name} (Raw)')
            
            # Plot moving average
            if len(rewards) >= 50:
                moving_avg = self._moving_average(rewards, 50)
                ax.plot(episodes[49:], moving_avg, linewidth=2, label=f'{name} (Avg)')
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved comparison plot: {save_name}")
    
    def create_summary_plot(self, 
                           episode_rewards: List[float],
                           eval_rewards: Optional[List[float]] = None,
                           episode_lengths: Optional[List[int]] = None,
                           training_stats: Optional[Dict[str, List[float]]] = None,
                           save_name: str = 'training_summary.png') -> None:
        """Create a comprehensive summary plot.
        
        Args:
            episode_rewards: List of episode rewards
            eval_rewards: List of evaluation rewards (optional)
            episode_lengths: List of episode lengths (optional)
            training_stats: Dictionary of training statistics (optional)
            save_name: Name to save the plot
        """
        # Determine subplot layout
        n_plots = 1
        if eval_rewards:
            n_plots += 1
        if episode_lengths:
            n_plots += 1
        if training_stats:
            n_plots += len(training_stats)
        
        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4 * n_plots))
        if n_plots == 1:
            axes = [axes]
        
        plot_idx = 0
        
        # Episode rewards
        episodes = range(len(episode_rewards))
        axes[plot_idx].plot(episodes, episode_rewards, alpha=0.3, color='blue', label='Episode Rewards')
        
        if len(episode_rewards) >= 100:
            moving_avg = self._moving_average(episode_rewards, 100)
            axes[plot_idx].plot(episodes[99:], moving_avg, color='blue', linewidth=2, 
                              label='Moving Average (100)')
        
        axes[plot_idx].set_xlabel('Episode')
        axes[plot_idx].set_ylabel('Reward')
        axes[plot_idx].set_title('Training Rewards')
        axes[plot_idx].legend()
        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1
        
        # Evaluation rewards
        if eval_rewards:
            eval_episodes = np.linspace(0, len(episode_rewards)-1, len(eval_rewards))
            axes[plot_idx].plot(eval_episodes, eval_rewards, color='red', linewidth=2, 
                              marker='o', markersize=4, label='Evaluation Rewards')
            axes[plot_idx].set_xlabel('Episode')
            axes[plot_idx].set_ylabel('Reward')
            axes[plot_idx].set_title('Evaluation Rewards')
            axes[plot_idx].legend()
            axes[plot_idx].grid(True, alpha=0.3)
            plot_idx += 1
        
        # Episode lengths
        if episode_lengths:
            axes[plot_idx].plot(episode_lengths, alpha=0.3, color='green', label='Episode Lengths')
            if len(episode_lengths) >= 100:
                moving_avg = self._moving_average(episode_lengths, 100)
                axes[plot_idx].plot(range(99, len(episode_lengths)), moving_avg, 
                                  color='green', linewidth=2, label='Moving Average (100)')
            axes[plot_idx].set_xlabel('Episode')
            axes[plot_idx].set_ylabel('Length')
            axes[plot_idx].set_title('Episode Lengths')
            axes[plot_idx].legend()
            axes[plot_idx].grid(True, alpha=0.3)
            plot_idx += 1
        
        # Training statistics
        if training_stats:
            for name, values in training_stats.items():
                if values:
                    axes[plot_idx].plot(values, linewidth=2)
                    axes[plot_idx].set_xlabel('Update')
                    axes[plot_idx].set_ylabel(name.replace('_', ' ').title())
                    axes[plot_idx].set_title(f'{name.replace("_", " ").title()} Over Time')
                    axes[plot_idx].grid(True, alpha=0.3)
                    plot_idx += 1
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved summary plot: {save_name}")
    
    def _moving_average(self, data: List[float], window_size: int) -> List[float]:
        """Calculate moving average.
        
        Args:
            data: Input data
            window_size: Window size
            
        Returns:
            Moving average values
        """
        if len(data) < window_size:
            return []
        
        moving_avg = []
        for i in range(window_size - 1, len(data)):
            window_data = data[i - window_size + 1:i + 1]
            moving_avg.append(np.mean(window_data))
        
        return moving_avg


def create_training_report(episode_rewards: List[float],
                          eval_rewards: Optional[List[float]] = None,
                          episode_lengths: Optional[List[int]] = None,
                          training_stats: Optional[Dict[str, List[float]]] = None,
                          save_dir: str = './plots') -> None:
    """Create a comprehensive training report.
    
    Args:
        episode_rewards: List of episode rewards
        eval_rewards: List of evaluation rewards (optional)
        episode_lengths: List of episode lengths (optional)
        training_stats: Dictionary of training statistics (optional)
        save_dir: Directory to save plots
    """
    visualizer = TrainingVisualizer(save_dir)
    
    # Create individual plots
    visualizer.plot_learning_curves(episode_rewards, eval_rewards)
    visualizer.plot_reward_distribution(episode_rewards)
    
    if episode_lengths:
        visualizer.plot_episode_lengths(episode_lengths)
    
    if training_stats:
        visualizer.plot_training_stats(training_stats)
    
    # Create summary plot
    visualizer.create_summary_plot(
        episode_rewards, eval_rewards, episode_lengths, training_stats
    )
    
    print(f"Training report saved to {save_dir}")
