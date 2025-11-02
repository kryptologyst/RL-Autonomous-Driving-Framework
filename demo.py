#!/usr/bin/env python3
"""Demo script for RL Autonomous Driving project."""

import sys
from pathlib import Path
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.utils.config import Config
from src.train import Trainer


def main():
    """Run a quick demo of the RL framework."""
    print("🚗 RL Autonomous Driving Framework Demo")
    print("=" * 50)
    
    # Create a simple configuration for demo
    config = Config()
    
    # Use CartPole for quick demo
    config.set('environment.name', 'CartPole-v1')
    config.set('environment.continuous', False)
    config.set('environment.render_mode', None)
    
    # Short training for demo
    config.set('training.total_timesteps', 5000)
    config.set('training.eval_freq', 1000)
    config.set('training.log_freq', 200)
    config.set('training.n_eval_episodes', 3)
    
    # PPO settings
    config.set('agent.algorithm', 'PPO')
    config.set('agent.learning_rate', 3e-4)
    config.set('agent.gamma', 0.99)
    config.set('agent.batch_size', 32)
    config.set('agent.buffer_size', 512)
    
    # Model settings
    config.set('model.hidden_size', 64)
    
    # Logging
    config.set('logging.use_tensorboard', True)
    config.set('logging.use_wandb', False)
    config.set('logging.log_dir', './demo_logs')
    
    print(f"Environment: {config.get('environment.name')}")
    print(f"Algorithm: {config.get('agent.algorithm')}")
    print(f"Total timesteps: {config.get('training.total_timesteps')}")
    print(f"Device: {config.get('device', 'auto')}")
    print()
    
    # Create trainer
    print("Initializing trainer...")
    trainer = Trainer(config)
    
    print("Starting training...")
    print("Press Ctrl+C to stop early")
    print()
    
    try:
        start_time = time.time()
        trainer.train()
        end_time = time.time()
        
        print(f"\n✅ Demo completed in {end_time - start_time:.2f} seconds!")
        print(f"📊 Check logs in: {config.get('logging.log_dir')}")
        print(f"📈 Start TensorBoard: tensorboard --logdir {config.get('logging.log_dir')}/tensorboard")
        
    except KeyboardInterrupt:
        print("\n⏹️ Demo stopped by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
    finally:
        trainer.close()
    
    print("\n🎉 Thanks for trying the RL Autonomous Driving Framework!")
    print("For more examples, check out:")
    print("  - python cli.py list-envs")
    print("  - python cli.py demo --env MountainCar-v0 --episodes 5")
    print("  - python cli.py train --env CarRacing-v2 --timesteps 100000")


if __name__ == '__main__':
    main()
