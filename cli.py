#!/usr/bin/env python3
"""Command-line interface for RL Autonomous Driving project."""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.utils.config import Config
from src.train import Trainer


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description='RL Autonomous Driving - Modern Reinforcement Learning Framework',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default CartPole environment
  python cli.py train
  
  # Train with CarRacing environment
  python cli.py train --env CarRacing-v2 --timesteps 1000000
  
  # Train with custom config
  python cli.py train --config config/custom_config.yaml
  
  # Quick demo with CartPole
  python cli.py demo --env CartPole-v1 --episodes 10
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train an RL agent')
    train_parser.add_argument('--config', type=str, default='config/config.yaml',
                             help='Path to configuration file')
    train_parser.add_argument('--env', type=str, default=None,
                             help='Environment name override')
    train_parser.add_argument('--timesteps', type=int, default=None,
                             help='Total training timesteps')
    train_parser.add_argument('--device', type=str, default='auto',
                             help='Device to use (auto, cpu, cuda, mps)')
    train_parser.add_argument('--algorithm', type=str, default=None,
                             help='Algorithm to use (PPO, SAC, TD3, DQN)')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run a quick demo')
    demo_parser.add_argument('--env', type=str, default='CartPole-v1',
                            help='Environment for demo')
    demo_parser.add_argument('--episodes', type=int, default=5,
                            help='Number of episodes to run')
    demo_parser.add_argument('--render', action='store_true',
                            help='Render the environment')
    
    # List environments command
    list_parser = subparsers.add_parser('list-envs', help='List available environments')
    
    # Version command
    version_parser = subparsers.add_parser('version', help='Show version information')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_command(args)
    elif args.command == 'demo':
        demo_command(args)
    elif args.command == 'list-envs':
        list_envs_command()
    elif args.command == 'version':
        version_command()
    else:
        parser.print_help()


def train_command(args):
    """Handle train command."""
    print("🚗 Starting RL Autonomous Driving Training...")
    
    # Load configuration
    config = Config(args.config)
    
    # Override config with command line arguments
    if args.env:
        config.set('environment.name', args.env)
    if args.timesteps:
        config.set('training.total_timesteps', args.timesteps)
    if args.device != 'auto':
        config.set('device', args.device)
    if args.algorithm:
        config.set('agent.algorithm', args.algorithm)
    
    # Create and run trainer
    trainer = Trainer(config)
    
    try:
        trainer.train()
        print("✅ Training completed successfully!")
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
    except Exception as e:
        print(f"❌ Training failed: {e}")
    finally:
        trainer.close()


def demo_command(args):
    """Handle demo command."""
    print(f"🎮 Running demo with {args.env} for {args.episodes} episodes...")
    
    # Create simple config for demo
    config = Config()
    config.set('environment.name', args.env)
    config.set('environment.render_mode', 'human' if args.render else None)
    config.set('training.total_timesteps', 1000)  # Short demo
    config.set('training.eval_freq', 500)
    config.set('training.log_freq', 100)
    
    # Create trainer
    trainer = Trainer(config)
    
    try:
        # Run a few episodes
        for episode in range(args.episodes):
            print(f"Episode {episode + 1}/{args.episodes}")
            stats = trainer._train_episode()
            print(f"  Reward: {stats['total_reward']:.2f}, Length: {stats['length']}")
        
        print("✅ Demo completed!")
    except Exception as e:
        print(f"❌ Demo failed: {e}")
    finally:
        trainer.close()


def list_envs_command():
    """List available environments."""
    print("🎯 Available Environments:")
    print()
    
    # Mock environments
    mock_envs = [
        ("CartPole-v1", "Classic control - Balance a pole"),
        ("MountainCar-v0", "Classic control - Drive up a mountain"),
        ("LunarLander-v2", "Box2D - Land a spacecraft"),
        ("Acrobot-v1", "Classic control - Swing up a pendulum"),
        ("Pendulum-v1", "Classic control - Swing up a pendulum")
    ]
    
    # CarRacing environments
    car_envs = [
        ("CarRacing-v2", "Box2D - Top-down car racing (continuous)"),
        ("CarRacing-v1", "Box2D - Top-down car racing (discrete)")
    ]
    
    print("📚 Mock Environments (for testing):")
    for env_name, description in mock_envs:
        print(f"  {env_name:<20} - {description}")
    
    print()
    print("🏎️ Car Racing Environments:")
    for env_name, description in car_envs:
        print(f"  {env_name:<20} - {description}")
    
    print()
    print("💡 Tip: Use 'python cli.py demo --env <env_name>' to try an environment")


def version_command():
    """Show version information."""
    print("RL Autonomous Driving Framework")
    print("Version: 1.0.0")
    print("Author: RL Autonomous Driving Team")
    print("Description: Modern reinforcement learning framework for autonomous driving")
    print()
    print("Features:")
    print("  ✅ Modern RL algorithms (PPO, SAC, TD3, Rainbow DQN)")
    print("  ✅ Multiple environment support")
    print("  ✅ Comprehensive logging and visualization")
    print("  ✅ Configuration management")
    print("  ✅ Checkpoint saving/loading")
    print("  ✅ TensorBoard and WandB integration")


if __name__ == '__main__':
    main()
