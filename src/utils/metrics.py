# src/utils/metrics.py
"""Metrics tracking and management utilities."""

import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, deque
import numpy as np
import pandas as pd
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class MetricsTracker:
    """Track and manage training/validation metrics."""
    
    def __init__(self, 
                 save_dir: Optional[Path] = None,
                 window_size: int = 100,
                 save_frequency: int = 10):
        """
        Initialize metrics tracker.
        
        Args:
            save_dir: Directory to save metrics
            window_size: Size of moving average window
            save_frequency: Save metrics every N updates
        """
        self.save_dir = save_dir
        self.window_size = window_size
        self.save_frequency = save_frequency
        
        # Metrics storage
        self.metrics = defaultdict(list)
        self.epoch_metrics = defaultdict(list)
        self.moving_averages = defaultdict(lambda: deque(maxlen=window_size))
        
        # Timing
        self.timers = {}
        self.update_count = 0
        
        # Best metrics tracking
        self.best_metrics = {}
        self.best_epoch = {}
        
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
            self.metrics_file = save_dir / "metrics.json"
            self.summary_file = save_dir / "metrics_summary.json"
    
    def update(self, metrics: Dict[str, float], prefix: str = '') -> None:
        """
        Update metrics.
        
        Args:
            metrics: Dictionary of metric values
            prefix: Prefix for metric names (e.g., 'train/', 'val/')
        """
        for name, value in metrics.items():
            full_name = f"{prefix}{name}" if prefix else name
            
            # Store value
            self.metrics[full_name].append(value)
            self.moving_averages[full_name].append(value)
            
            # Update best metrics
            if 'loss' in name:
                # Lower is better
                if full_name not in self.best_metrics or value < self.best_metrics[full_name]:
                    self.best_metrics[full_name] = value
                    self.best_epoch[full_name] = len(self.metrics[full_name])
            else:
                # Higher is better for most metrics
                if full_name not in self.best_metrics or value > self.best_metrics[full_name]:
                    self.best_metrics[full_name] = value
                    self.best_epoch[full_name] = len(self.metrics[full_name])
        
        self.update_count += 1
        
        # Periodic save
        if self.save_dir and self.update_count % self.save_frequency == 0:
            self.save()
    
    def update_epoch_metrics(self, metrics: Dict[str, float], 
                           epoch: int, prefix: str = '') -> None:
        """Update epoch-level metrics."""
        for name, value in metrics.items():
            full_name = f"{prefix}{name}" if prefix else name
            self.epoch_metrics[full_name].append({
                'epoch': epoch,
                'value': value,
                'timestamp': datetime.now().isoformat()
            })
    
    def get_metric(self, name: str, last_n: Optional[int] = None) -> List[float]:
        """
        Get metric values.
        
        Args:
            name: Metric name
            last_n: Return only last N values
            
        Returns:
            List of metric values
        """
        values = self.metrics.get(name, [])
        
        if last_n:
            return values[-last_n:]
        
        return values
    
    def get_moving_average(self, name: str) -> float:
        """Get moving average for metric."""
        values = self.moving_averages.get(name, [])
        
        if not values:
            return 0.0
        
        return float(np.mean(values))
    
    def get_best(self, name: str) -> Tuple[float, int]:
        """
        Get best value and epoch for metric.
        
        Returns:
            Tuple of (best_value, best_epoch)
        """
        return self.best_metrics.get(name, None), self.best_epoch.get(name, None)
    
    def start_timer(self, name: str) -> None:
        """Start a timer."""
        self.timers[name] = time.time()
    
    def stop_timer(self, name: str) -> float:
        """
        Stop a timer and return elapsed time.
        
        Returns:
            Elapsed time in seconds
        """
        if name not in self.timers:
            return 0.0
        
        elapsed = time.time() - self.timers[name]
        del self.timers[name]
        
        # Store as metric
        self.update({f"{name}_time": elapsed})
        
        return elapsed
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for all metrics."""
        summary = {}
        
        for name, values in self.metrics.items():
            if values:
                summary[name] = {
                    'current': values[-1],
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'best': self.best_metrics.get(name),
                    'best_epoch': self.best_epoch.get(name),
                    'count': len(values)
                }
        
        return summary
    
    def save(self) -> None:
        """Save metrics to file."""
        if not self.save_dir:
            return
        
        # Save raw metrics
        metrics_data = {
            'metrics': dict(self.metrics),
            'epoch_metrics': dict(self.epoch_metrics),
            'best_metrics': self.best_metrics,
            'best_epoch': self.best_epoch,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        # Save summary
        with open(self.summary_file, 'w') as f:
            json.dump(self.get_summary(), f, indent=2)
        
        logger.debug(f"Saved metrics to {self.save_dir}")
    
    def load(self) -> None:
        """Load metrics from file."""
        if not self.save_dir or not self.metrics_file.exists():
            return
        
        with open(self.metrics_file, 'r') as f:
            data = json.load(f)
        
        self.metrics = defaultdict(list, data['metrics'])
        self.epoch_metrics = defaultdict(list, data['epoch_metrics'])
        self.best_metrics = data['best_metrics']
        self.best_epoch = data['best_epoch']
        
        logger.info(f"Loaded metrics from {self.metrics_file}")
    
    def plot_metrics(self, metric_names: Optional[List[str]] = None,
                    save_path: Optional[Path] = None) -> None:
        """
        Plot metrics.
        
        Args:
            metric_names: List of metrics to plot (None for all)
            save_path: Path to save plot
        """
        import matplotlib.pyplot as plt
        
        if metric_names is None:
            metric_names = list(self.metrics.keys())
        
        # Create subplots
        n_metrics = len(metric_names)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        
        if n_metrics == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        for idx, name in enumerate(metric_names):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            values = self.metrics.get(name, [])
            
            if values:
                ax.plot(values, label=name)
                
                # Add moving average
                if len(values) > self.window_size:
                    ma = pd.Series(values).rolling(self.window_size).mean()
                    ax.plot(ma, label=f'MA({self.window_size})', alpha=0.7)
                
                # Mark best
                if name in self.best_metrics:
                    best_idx = self.best_epoch[name] - 1
                    ax.scatter(best_idx, self.best_metrics[name], 
                             color='red', s=100, zorder=5,
                             label=f'Best: {self.best_metrics[name]:.4f}')
                
                ax.set_xlabel('Step')
                ax.set_ylabel(name)
                ax.set_title(name)
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(metric_names), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved metrics plot to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert metrics to pandas DataFrame."""
        # Find maximum length
        max_len = max(len(values) for values in self.metrics.values())
        
        # Pad shorter sequences
        data = {}
        for name, values in self.metrics.items():
            padded_values = values + [np.nan] * (max_len - len(values))
            data[name] = padded_values
        
        return pd.DataFrame(data)
    
    def log_metrics(self, epoch: int, prefix: str = '') -> None:
        """Log current metrics."""
        current_metrics = {}
        
        for name, values in self.metrics.items():
            if values and (not prefix or name.startswith(prefix)):
                current_metrics[name] = values[-1]
        
        if current_metrics:
            metrics_str = ' | '.join([f"{k}: {v:.4f}" for k, v in current_metrics.items()])
            logger.info(f"Epoch {epoch} | {metrics_str}")


class TensorBoardLogger:
    """TensorBoard logging wrapper."""
    
    def __init__(self, log_dir: Path):
        """Initialize TensorBoard logger."""
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir)
            self.enabled = True
            logger.info(f"TensorBoard logging enabled: {log_dir}")
        except ImportError:
            logger.warning("TensorBoard not available. Install with: pip install tensorboard")
            self.enabled = False
    
    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log scalar value."""
        if self.enabled:
            self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int) -> None:
        """Log multiple scalars."""
        if self.enabled:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)
    
    def log_histogram(self, tag: str, values: np.ndarray, step: int) -> None:
        """Log histogram."""
        if self.enabled:
            self.writer.add_histogram(tag, values, step)
    
    def log_image(self, tag: str, img: np.ndarray, step: int) -> None:
        """Log image."""
        if self.enabled:
            self.writer.add_image(tag, img, step)
    
    def log_model_graph(self, model: Any, input_data: Any) -> None:
        """Log model graph."""
        if self.enabled:
            try:
                self.writer.add_graph(model, input_data)
            except Exception as e:
                logger.warning(f"Failed to log model graph: {e}")
    
    def close(self) -> None:
        """Close TensorBoard writer."""
        if self.enabled:
            self.writer.close()


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                     task: str = 'regression') -> Dict[str, float]:
    """
    Calculate standard metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        task: Task type ('regression' or 'classification')
        
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import (
        mean_squared_error, mean_absolute_error, r2_score,
        accuracy_score, precision_recall_fscore_support
    )
    
    metrics = {}
    
    if task == 'regression':
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['r2'] = r2_score(y_true, y_pred)
        
        # Additional metrics
        errors = y_true - y_pred
        metrics['mean_error'] = np.mean(errors)
        metrics['std_error'] = np.std(errors)
        metrics['max_error'] = np.max(np.abs(errors))
        
    elif task == 'classification':
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # Calculate per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1'] = f1
    
    return metrics