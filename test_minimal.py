#!/usr/bin/env python3
"""Simple test with minimal dependencies."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_simple_config():
    """Test simple config."""
    print("Testing simple config...")
    
    from src.utils.simple_config import SimpleConfig
    config = SimpleConfig()
    
    # Test getting values
    assert config.get('environment.name') == 'CartPole-v1'
    assert config.get('agent.learning_rate') == 3e-4
    
    # Test setting values
    config.set('test.value', 42)
    assert config.get('test.value') == 42
    
    print("✅ Simple config test passed")
    return True

def test_environment():
    """Test environment."""
    print("Testing environment...")
    
    import gymnasium as gym
    env = gym.make('CartPole-v1')
    
    obs, info = env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    env.close()
    
    print("✅ Environment test passed")
    return True

def test_model():
    """Test model."""
    print("Testing model...")
    
    import torch
    import torch.nn as nn
    
    # Simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 2)
        
        def forward(self, x):
            return self.fc(x)
    
    model = SimpleModel()
    obs = torch.randn(1, 4)
    output = model(obs)
    
    assert output.shape == (1, 2)
    print("✅ Model test passed")
    return True

def main():
    """Run tests."""
    print("🚗 RL Autonomous Driving - Simple Tests")
    print("=" * 50)
    
    tests = [test_simple_config, test_environment, test_model]
    
    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")
    
    print(f"\n📊 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed!")
        return True
    else:
        print("⚠️ Some tests failed")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
