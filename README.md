# RL Autonomous Driving Framework

A comprehensive reinforcement learning framework for autonomous driving applications. This project provides state-of-the-art RL algorithms, environment wrappers, visualization tools, and a clean, extensible architecture.

## Features

- **Modern RL Algorithms**: PPO, SAC, TD3, Rainbow DQN with distributional RL
- **Multiple Environments**: Support for CarRacing, CartPole, MountainCar, and more
- **Comprehensive Logging**: TensorBoard and Weights & Biases integration
- **Visualization Tools**: Learning curves, reward distributions, and training statistics
- **Configuration Management**: YAML-based configuration with command-line overrides
- **Checkpoint System**: Save and load trained models
- **Clean Architecture**: Modular design with proper separation of concerns
- **Type Hints**: Full type annotations for better code quality
- **Documentation**: Comprehensive docstrings and examples

## Installation

### Prerequisites

- Python 3.10+
- pip or conda

### Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# For GPU support (optional)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/kryptologyst/RL-Autonomous-Driving-Framework.git
cd RL-Autonomous-Driving-Framework

# Install in development mode
pip install -e .
```

## Quick Start

### Command Line Interface

```bash
# Train with default CartPole environment
python cli.py train

# Train with CarRacing environment
python cli.py train --env CarRacing-v2 --timesteps 1000000

# Quick demo
python cli.py demo --env CartPole-v1 --episodes 10 --render

# List available environments
python cli.py list-envs
```

### Python API

```python
from src.utils.config import Config
from src.train import Trainer

# Load configuration
config = Config('config/config.yaml')

# Create trainer
trainer = Trainer(config)

# Train the agent
trainer.train()
```

## Project Structure

```
rl-autonomous-driving/
├── src/
│   ├── agents/           # RL algorithms (PPO, SAC, TD3, etc.)
│   ├── envs/            # Environment wrappers and utilities
│   ├── utils/           # Configuration, logging, visualization
│   └── train.py         # Main training script
├── config/
│   └── config.yaml      # Configuration file
├── notebooks/           # Jupyter notebooks for analysis
├── tests/              # Unit tests
├── logs/               # Training logs
├── checkpoints/         # Saved models
├── plots/              # Generated plots
├── cli.py              # Command-line interface
├── requirements.txt    # Dependencies
└── README.md          # This file
```

## 🔧 Configuration

The framework uses YAML configuration files. Key configuration options:

```yaml
# Environment settings
environment:
  name: "CarRacing-v2"
  continuous: true
  render_mode: null

# Agent settings
agent:
  algorithm: "PPO"
  learning_rate: 3e-4
  gamma: 0.99

# Training settings
training:
  total_timesteps: 1000000
  eval_freq: 10000
  save_freq: 50000

# Logging
logging:
  use_tensorboard: true
  use_wandb: false
```

## Supported Environments

### Mock Environments (for testing)
- **CartPole-v1**: Classic control - Balance a pole
- **MountainCar-v0**: Classic control - Drive up a mountain
- **LunarLander-v2**: Box2D - Land a spacecraft
- **Acrobot-v1**: Classic control - Swing up a pendulum
- **Pendulum-v1**: Classic control - Swing up a pendulum

### Car Racing Environments
- **CarRacing-v2**: Box2D - Top-down car racing (continuous actions)
- **CarRacing-v1**: Box2D - Top-down car racing (discrete actions)

## Supported Algorithms

### Policy Gradient Methods
- **PPO (Proximal Policy Optimization)**: Stable, on-policy algorithm with clipping
- **SAC (Soft Actor-Critic)**: Off-policy algorithm for continuous control
- **TD3 (Twin Delayed Deep Deterministic)**: Improved DDPG variant

### Value-Based Methods
- **DQN (Deep Q-Network)**: Classic deep RL algorithm
- **Rainbow DQN**: DQN with multiple improvements (distributional RL, dueling, etc.)

## Visualization and Logging

### TensorBoard Integration
```bash
# Start TensorBoard
tensorboard --logdir logs/tensorboard

# View training progress at http://localhost:6006
```

### Weights & Biases Integration
```yaml
logging:
  use_wandb: true
  wandb_project: "rl-autonomous-driving"
```

### Generated Plots
- Learning curves with moving averages
- Reward distributions and statistics
- Training loss curves
- Action distributions
- Episode length analysis

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src

# Run specific test
pytest tests/test_agents.py::test_ppo_agent
```

## Training Examples

### Basic Training
```bash
# Train PPO on CartPole
python cli.py train --env CartPole-v1 --timesteps 100000

# Train with custom config
python cli.py train --config config/custom_config.yaml
```

### Advanced Training
```bash
# Train on CarRacing with GPU
python cli.py train --env CarRacing-v2 --device cuda --timesteps 2000000

# Train with WandB logging
python cli.py train --env LunarLander-v2 --timesteps 500000
```

## Monitoring Training

### Real-time Monitoring
- Console output with episode statistics
- TensorBoard for detailed metrics
- WandB for experiment tracking
- Automatic checkpoint saving

### Key Metrics
- Episode rewards and lengths
- Training losses (actor, critic, entropy)
- KL divergence and clip fractions
- Evaluation performance

## Model Management

### Saving Models
```python
# Models are automatically saved during training
# Manual saving
agent.save('models/my_agent.pt')
```

### Loading Models
```python
# Load trained model
agent.load('checkpoints/agent_step_100000.pt')
```

## Advanced Usage

### Custom Environments
```python
from src.envs.environment import make_env

# Create custom environment
env = make_env('MyCustomEnv-v1', custom_param=value)
```

### Custom Algorithms
```python
from src.agents.models import ActorCritic

# Extend existing models
class MyCustomAgent(ActorCritic):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add custom components
```

### Configuration Overrides
```python
from src.utils.config import Config

config = Config()
config.set('agent.learning_rate', 1e-4)
config.set('training.total_timesteps', 500000)
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8 mypy

# Run code formatting
black src/ tests/

# Run linting
flake8 src/ tests/

# Run type checking
mypy src/
```

## Documentation

- [API Reference](docs/api.md)
- [Algorithm Details](docs/algorithms.md)
- [Environment Guide](docs/environments.md)
- [Configuration Guide](docs/configuration.md)
- [Tutorials](notebooks/)

## Troubleshooting

### Common Issues

1. **Environment not found**: Make sure you have the required dependencies installed
   ```bash
   pip install gymnasium[box2d]  # For CarRacing
   ```

2. **CUDA out of memory**: Reduce batch size or use CPU
   ```bash
   python cli.py train --device cpu
   ```

3. **Import errors**: Make sure you're in the project root directory

### Getting Help

- Check the [Issues](https://github.com/kryptologyst/RL-Autonomous-Driving-Framework/issues) page
- Create a new issue with detailed error information
- Join our community discussions

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- OpenAI Gym/Gymnasium for the RL environments
- Stable Baselines3 for algorithm implementations
- PyTorch for the deep learning framework
- The RL research community for the algorithms and techniques

## Performance Benchmarks

| Environment | Algorithm | Episodes to Solve | Final Reward |
|-------------|-----------|-------------------|--------------|
| CartPole-v1 | PPO | ~100 | 500 |
| MountainCar-v0 | PPO | ~500 | -110 |
| LunarLander-v2 | PPO | ~200 | 200+ |
| CarRacing-v2 | PPO | ~1000 | 500+ |

*Note: Results may vary based on hyperparameters and random seeds.*

 
# RL-Autonomous-Driving-Framework
