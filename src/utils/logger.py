"""Logging utilities for RL Autonomous Driving project."""

import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
from datetime import datetime

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def setup_logging(log_dir: str = './logs', level: int = logging.INFO) -> logging.Logger:
    """Setup logging configuration.
    
    Args:
        log_dir: Directory for log files
        level: Logging level
        
    Returns:
        Configured logger
    """
    # Create log directory
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('rl_autonomous_driving')
    logger.setLevel(level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    log_file = Path(log_dir) / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


class TensorBoardLogger:
    """TensorBoard logger wrapper."""
    
    def __init__(self, log_dir: str):
        """Initialize TensorBoard logger.
        
        Args:
            log_dir: Directory for TensorBoard logs
        """
        if not TENSORBOARD_AVAILABLE:
            raise ImportError("TensorBoard not available. Install with: pip install tensorboard")
        
        self.log_dir = Path(log_dir) / 'tensorboard'
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.writer = SummaryWriter(str(self.log_dir))
    
    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log scalar value.
        
        Args:
            tag: Tag for the scalar
            value: Scalar value
            step: Step number
        """
        self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, tag: str, values: Dict[str, float], step: int) -> None:
        """Log multiple scalars.
        
        Args:
            tag: Tag for the scalars
            values: Dictionary of scalar values
            step: Step number
        """
        self.writer.add_scalars(tag, values, step)
    
    def log_histogram(self, tag: str, values: np.ndarray, step: int) -> None:
        """Log histogram.
        
        Args:
            tag: Tag for the histogram
            values: Values for histogram
            step: Step number
        """
        self.writer.add_histogram(tag, values, step)
    
    def log_image(self, tag: str, image: np.ndarray, step: int) -> None:
        """Log image.
        
        Args:
            tag: Tag for the image
            image: Image array
            step: Step number
        """
        self.writer.add_image(tag, image, step)
    
    def close(self) -> None:
        """Close TensorBoard writer."""
        self.writer.close()


class WandBLogger:
    """Weights & Biases logger wrapper."""
    
    def __init__(self, project: str, config: Optional[Dict[str, Any]] = None):
        """Initialize WandB logger.
        
        Args:
            project: WandB project name
            config: Configuration dictionary
        """
        if not WANDB_AVAILABLE:
            raise ImportError("WandB not available. Install with: pip install wandb")
        
        self.project = project
        self.config = config or {}
        
        # Initialize WandB
        wandb.init(project=project, config=config)
    
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Step number (optional)
        """
        if step is not None:
            metrics['step'] = step
        
        wandb.log(metrics)
    
    def log_scalar(self, name: str, value: float, step: Optional[int] = None) -> None:
        """Log scalar value.
        
        Args:
            name: Name of the scalar
            value: Scalar value
            step: Step number (optional)
        """
        self.log({name: value}, step)
    
    def log_scalars(self, prefix: str, values: Dict[str, float], step: Optional[int] = None) -> None:
        """Log multiple scalars with prefix.
        
        Args:
            prefix: Prefix for scalar names
            values: Dictionary of scalar values
            step: Step number (optional)
        """
        prefixed_values = {f"{prefix}/{k}": v for k, v in values.items()}
        self.log(prefixed_values, step)
    
    def log_image(self, name: str, image: np.ndarray, step: Optional[int] = None) -> None:
        """Log image.
        
        Args:
            name: Name of the image
            image: Image array
            step: Step number (optional)
        """
        wandb.log({name: wandb.Image(image)}, step=step)
    
    def log_histogram(self, name: str, values: np.ndarray, step: Optional[int] = None) -> None:
        """Log histogram.
        
        Args:
            name: Name of the histogram
            values: Values for histogram
            step: Step number (optional)
        """
        wandb.log({name: wandb.Histogram(values)}, step=step)
    
    def close(self) -> None:
        """Close WandB run."""
        wandb.finish()


class Logger:
    """Unified logger that can use multiple backends."""
    
    def __init__(self, 
                 use_tensorboard: bool = True,
                 use_wandb: bool = False,
                 log_dir: str = './logs',
                 project: Optional[str] = None,
                 config: Optional[Dict[str, Any]] = None):
        """Initialize unified logger.
        
        Args:
            use_tensorboard: Whether to use TensorBoard
            use_wandb: Whether to use WandB
            log_dir: Directory for logs
            project: WandB project name
            config: Configuration dictionary
        """
        self.tb_logger = None
        self.wandb_logger = None
        
        if use_tensorboard and TENSORBOARD_AVAILABLE:
            try:
                self.tb_logger = TensorBoardLogger(log_dir)
            except Exception as e:
                print(f"Failed to initialize TensorBoard logger: {e}")
        
        if use_wandb and WANDB_AVAILABLE and project:
            try:
                self.wandb_logger = WandBLogger(project, config)
            except Exception as e:
                print(f"Failed to initialize WandB logger: {e}")
    
    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log scalar value to all available loggers.
        
        Args:
            tag: Tag for the scalar
            value: Scalar value
            step: Step number
        """
        if self.tb_logger:
            self.tb_logger.log_scalar(tag, value, step)
        
        if self.wandb_logger:
            self.wandb_logger.log_scalar(tag, value, step)
    
    def log_scalars(self, tag: str, values: Dict[str, float], step: int) -> None:
        """Log multiple scalars to all available loggers.
        
        Args:
            tag: Tag for the scalars
            values: Dictionary of scalar values
            step: Step number
        """
        if self.tb_logger:
            self.tb_logger.log_scalars(tag, values, step)
        
        if self.wandb_logger:
            self.wandb_logger.log_scalars(tag, values, step)
    
    def log_image(self, tag: str, image: np.ndarray, step: int) -> None:
        """Log image to all available loggers.
        
        Args:
            tag: Tag for the image
            image: Image array
            step: Step number
        """
        if self.tb_logger:
            self.tb_logger.log_image(tag, image, step)
        
        if self.wandb_logger:
            self.wandb_logger.log_image(tag, image, step)
    
    def log_histogram(self, tag: str, values: np.ndarray, step: int) -> None:
        """Log histogram to all available loggers.
        
        Args:
            tag: Tag for the histogram
            values: Values for histogram
            step: Step number
        """
        if self.tb_logger:
            self.tb_logger.log_histogram(tag, values, step)
        
        if self.wandb_logger:
            self.wandb_logger.log_histogram(tag, values, step)
    
    def close(self) -> None:
        """Close all loggers."""
        if self.tb_logger:
            self.tb_logger.close()
        
        if self.wandb_logger:
            self.wandb_logger.close()
