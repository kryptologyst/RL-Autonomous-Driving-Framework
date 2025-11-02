#!/usr/bin/env python3
"""Simple working demo of the RL framework."""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def main():
    """Run a simple demo."""
    print("🚗 RL Autonomous Driving Framework - Simple Demo")
    print("=" * 60)
    
    try:
        # Test basic imports one by one
        print("Testing imports...")
        
        from src.utils.config import Config
        print("✅ Config imported successfully")
        
        config = Config()
        print(f"✅ Config created: {type(config)}")
        
        # Test environment
        from src.envs.environment import make_env
        print("✅ Environment module imported successfully")
        
        env = make_env('CartPole-v1')
        print("✅ CartPole environment created successfully")
        
        # Test a simple episode
        obs, info = env.reset()
        print(f"✅ Environment reset successful, obs shape: {obs.shape}")
        
        total_reward = 0
        for step in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        print(f"✅ Episode completed, total reward: {total_reward}")
        env.close()
        
        # Test model
        import torch
        from src.agents.models import ActorCritic
        print("✅ Model module imported successfully")
        
        model = ActorCritic(obs_shape=(4,), action_dim=2)
        print("✅ ActorCritic model created successfully")
        
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        action_mean, value = model(obs_tensor)
        print(f"✅ Model forward pass successful, action shape: {action_mean.shape}, value shape: {value.shape}")
        
        print("\n🎉 All components working correctly!")
        print("The RL Autonomous Driving framework is ready to use!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = main()
    if success:
        print("\n✅ Demo completed successfully!")
    else:
        print("\n❌ Demo failed!")
    sys.exit(0 if success else 1)
