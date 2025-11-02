"""Basic tests for RL Autonomous Driving project."""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.utils.config import Config
from src.agents.models import ActorCritic, DQNNetwork
from src.agents.ppo import PPOAgent, PPOBuffer
from src.envs.environment import make_env, get_env_info


class TestConfig:
    """Test configuration management."""
    
    def test_config_creation(self):
        """Test config creation and basic operations."""
        config = Config()
        
        # Test getting values
        assert config.get('agent.learning_rate', 3e-4) == 3e-4
        
        # Test setting values
        config.set('test.value', 42)
        assert config.get('test.value') == 42
        
        # Test dict-like access
        assert config['test.value'] == 42
        config['test.value'] = 24
        assert config['test.value'] == 24


class TestModels:
    """Test neural network models."""
    
    def test_actor_critic_creation(self):
        """Test ActorCritic model creation."""
        obs_shape = (4,)  # CartPole observation
        action_dim = 2    # CartPole action
        
        model = ActorCritic(obs_shape, action_dim)
        
        # Test forward pass
        obs = torch.randn(1, *obs_shape)
        action_mean, value = model(obs)
        
        assert action_mean.shape == (1, action_dim)
        assert value.shape == (1, 1)
        assert torch.all(action_mean >= -1) and torch.all(action_mean <= 1)  # Tanh output
    
    def test_dqn_network_creation(self):
        """Test DQN network creation."""
        obs_shape = (4,)  # CartPole observation
        action_dim = 2    # CartPole action
        
        model = DQNNetwork(obs_shape, action_dim)
        
        # Test forward pass
        obs = torch.randn(1, *obs_shape)
        q_values = model(obs)
        
        assert q_values.shape == (1, action_dim)


class TestPPOAgent:
    """Test PPO agent."""
    
    def test_ppo_agent_creation(self):
        """Test PPO agent creation."""
        config = Config()
        obs_shape = (4,)
        action_dim = 2
        
        agent = PPOAgent(obs_shape, action_dim, config)
        
        assert agent.obs_shape == obs_shape
        assert agent.action_dim == action_dim
        assert agent.device is not None
    
    def test_ppo_buffer(self):
        """Test PPO buffer operations."""
        buffer_size = 100
        obs_shape = (4,)
        action_dim = 2
        device = torch.device('cpu')
        
        buffer = PPOBuffer(buffer_size, obs_shape, action_dim, device)
        
        # Test adding transitions
        obs = torch.randn(1, *obs_shape)
        action = torch.randn(1, action_dim)
        reward = 1.0
        value = torch.randn(1, 1)
        log_prob = torch.randn(1)
        
        buffer.add(obs, action, reward, value, log_prob)
        assert buffer.size == 1
        
        # Test getting batch
        batch = buffer.get_batch(1)
        assert 'observations' in batch
        assert 'actions' in batch
        assert 'advantages' in batch


class TestEnvironment:
    """Test environment utilities."""
    
    def test_cartpole_environment(self):
        """Test CartPole environment creation."""
        env = make_env('CartPole-v1')
        
        # Test environment info
        env_info = get_env_info(env)
        assert 'observation_space' in env_info
        assert 'action_space' in env_info
        
        # Test environment step
        obs, info = env.reset()
        assert obs is not None
        
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs is not None
        assert isinstance(reward, (int, float))
        
        env.close()
    
    def test_environment_fallback(self):
        """Test environment fallback mechanism."""
        # This should fallback to CartPole if CarRacing is not available
        env = make_env('CarRacing-v2')
        env_info = get_env_info(env)
        assert env_info['observation_space'] is not None
        env.close()


class TestIntegration:
    """Test integration between components."""
    
    def test_training_loop_basic(self):
        """Test basic training loop components."""
        config = Config()
        config.set('environment.name', 'CartPole-v1')
        config.set('training.total_timesteps', 1000)
        config.set('training.eval_freq', 500)
        config.set('training.log_freq', 100)
        
        # Create environment
        env = make_env('CartPole-v1')
        env_info = get_env_info(env)
        
        obs_shape = env_info['observation_space'].shape
        action_dim = env_info['action_space'].n
        
        # Create agent
        agent = PPOAgent(obs_shape, action_dim, config)
        
        # Test getting action
        obs = torch.FloatTensor(env.reset()[0]).unsqueeze(0)
        action, log_prob, value = agent.get_action(obs)
        
        assert action.shape == (1, action_dim)
        assert log_prob.shape == (1,)
        assert value.shape == (1, 1)
        
        env.close()


if __name__ == '__main__':
    # Run basic tests
    print("Running basic tests...")
    
    # Test config
    print("Testing config...")
    test_config = TestConfig()
    test_config.test_config_creation()
    print("✅ Config tests passed")
    
    # Test models
    print("Testing models...")
    test_models = TestModels()
    test_models.test_actor_critic_creation()
    test_models.test_dqn_network_creation()
    print("✅ Model tests passed")
    
    # Test PPO agent
    print("Testing PPO agent...")
    test_ppo = TestPPOAgent()
    test_ppo.test_ppo_agent_creation()
    test_ppo.test_ppo_buffer()
    print("✅ PPO agent tests passed")
    
    # Test environment
    print("Testing environment...")
    test_env = TestEnvironment()
    test_env.test_cartpole_environment()
    test_env.test_environment_fallback()
    print("✅ Environment tests passed")
    
    # Test integration
    print("Testing integration...")
    test_integration = TestIntegration()
    test_integration.test_training_loop_basic()
    print("✅ Integration tests passed")
    
    print("\n🎉 All basic tests passed!")
