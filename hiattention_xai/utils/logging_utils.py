"""
Logging Utilities

Setup and configuration for logging across the project.
"""

import logging
import os
import sys
from datetime import datetime
from typing import Optional


def setup_logging(
    log_dir: str = './logs',
    log_level: str = 'INFO',
    log_to_file: bool = True,
    log_to_console: bool = True,
    experiment_name: Optional[str] = None
) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_dir: Directory for log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_to_file: Whether to log to file
        log_to_console: Whether to log to console
        experiment_name: Name for log file
    
    Returns:
        Configured logger
    """
    # Create log directory
    if log_to_file:
        os.makedirs(log_dir, exist_ok=True)
    
    # Get logger
    logger = logging.getLogger('hiattention_xai')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Format
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_to_file:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        name = experiment_name or 'experiment'
        log_file = os.path.join(log_dir, f'{name}_{timestamp}.log')
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = 'hiattention_xai') -> logging.Logger:
    """Get logger by name."""
    return logging.getLogger(name)


class TrainingLogger:
    """
    Structured logging for training progress.
    """
    
    def __init__(self, logger: logging.Logger, use_wandb: bool = False):
        self.logger = logger
        self.use_wandb = use_wandb
        
        if use_wandb:
            try:
                import wandb
                self.wandb = wandb
            except ImportError:
                self.use_wandb = False
                logger.warning("WandB not installed, disabling WandB logging")
    
    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_metrics: dict,
        learning_rate: float,
        epoch_time: float
    ):
        """Log epoch summary."""
        msg = f"Epoch {epoch} | Loss: {train_loss:.4f} | "
        msg += f"Val F1: {val_metrics.get('f1', 0):.4f} | "
        msg += f"Val AUC: {val_metrics.get('auc_roc', 0):.4f} | "
        msg += f"LR: {learning_rate:.2e} | Time: {epoch_time:.1f}s"
        
        self.logger.info(msg)
        
        if self.use_wandb:
            self.wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'learning_rate': learning_rate,
                'epoch_time': epoch_time,
                **{f'val/{k}': v for k, v in val_metrics.items()}
            })
    
    def log_step(self, step: int, loss: float, lr: float):
        """Log training step."""
        self.logger.debug(f"Step {step} | Loss: {loss:.4f} | LR: {lr:.2e}")
        
        if self.use_wandb:
            self.wandb.log({
                'step': step,
                'train_loss_step': loss,
                'learning_rate': lr
            })
    
    def log_evaluation(self, metrics: dict, dataset_name: str = 'test'):
        """Log evaluation results."""
        self.logger.info(f"Evaluation on {dataset_name}:")
        for key, value in metrics.items():
            if isinstance(value, float):
                self.logger.info(f"  {key}: {value:.4f}")
            else:
                self.logger.info(f"  {key}: {value}")
        
        if self.use_wandb:
            self.wandb.log({f'{dataset_name}/{k}': v for k, v in metrics.items()})
