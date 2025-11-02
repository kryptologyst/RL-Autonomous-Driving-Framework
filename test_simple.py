#!/usr/bin/env python3
"""Simple test script for RL Autonomous Driving project."""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_imports():
    """Test basic imports."""
    print("Testing imports...")
    
    try:
        from src.utils.config import Config
        print("✅ Config import successful")
    except Exception as e:
        print(f"❌ Config import failed: {e}")
        return False
    
    try:
        from src.agents.models import ActorCritic
        print("✅ ActorCritic import successful")
    except Exception as e:
        print(f"❌ ActorCritic import failed: {e}")
        return False
    
    try:
        from src.envs.environment import make_env
        print("✅ Environment import successful")
    except Exception as e:
        print(f"❌ Environment import failed: {e}")
        return False
    
    return True

def test_config():
    """Test configuration."""
    print("\nTesting configuration...")
    
    try:
        from src.utils.config import Config
        config = Config()
        
        # Test basic operations
        config.set('test.value', 42)
        assert config.get('test.value') == 42
        print("✅ Config operations successful")
        
        return True
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False

def test_environment():
    """Test environment creation."""
    print("\nTesting environment...")
    
    try:
        from src.envs.environment import make_env
        
        # Test CartPole environment
        env = make_env('CartPole-v1')
        obs, info = env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.close()
        
        print("✅ Environment test successful")
        return True
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        return False

def test_model():
    """Test model creation."""
    print("\nTesting model...")
    
    try:
        import torch
        from src.agents.models import ActorCritic
        
        # Create model
        model = ActorCritic(obs_shape=(4,), action_dim=2)
        
        # Test forward pass
        obs = torch.randn(1, 4)
        action_mean, value = model(obs)
        
        assert action_mean.shape == (1, 2)
        assert value.shape == (1, 1)
        
        print("✅ Model test successful")
        return True
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚗 RL Autonomous Driving Framework - Basic Tests")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_config,
        test_environment,
        test_model
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The framework is working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Check the errors above.")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
