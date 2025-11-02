"""Setup script for RL Autonomous Driving project."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path, 'r') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
else:
    requirements = [
        "gymnasium>=0.29.0",
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "pyyaml>=6.0",
    ]

setup(
    name="rl-autonomous-driving",
    version="1.0.0",
    author="RL Autonomous Driving Team",
    author_email="team@example.com",
    description="Modern reinforcement learning framework for autonomous driving",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/rl-autonomous-driving",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
        "full": [
            "gymnasium[box2d]>=0.29.0",
            "stable-baselines3>=2.0.0",
            "ray[rllib]>=2.8.0",
            "tensorboard>=2.13.0",
            "wandb>=0.15.0",
            "seaborn>=0.12.0",
            "plotly>=5.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "rl-driving=cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
