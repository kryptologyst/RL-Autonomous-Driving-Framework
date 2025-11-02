#!/usr/bin/env python3
"""Minimal test to check Python environment."""

import sys
import os
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print(f"Current directory: {os.getcwd()}")

# Test basic imports
try:
    import torch
    print(f"✅ PyTorch version: {torch.__version__}")
except ImportError as e:
    print(f"❌ PyTorch import failed: {e}")

try:
    import gymnasium
    print(f"✅ Gymnasium version: {gymnasium.__version__}")
except ImportError as e:
    print(f"❌ Gymnasium import failed: {e}")

try:
    import numpy
    print(f"✅ NumPy version: {numpy.__version__}")
except ImportError as e:
    print(f"❌ NumPy import failed: {e}")

try:
    import matplotlib
    print(f"✅ Matplotlib version: {matplotlib.__version__}")
except ImportError as e:
    print(f"❌ Matplotlib import failed: {e}")

try:
    import yaml
    print("✅ PyYAML import successful")
except ImportError as e:
    print(f"❌ PyYAML import failed: {e}")

print("\n🎉 Basic environment test completed!")
