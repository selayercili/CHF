# src/utils/early_stopping.py
"""Early stopping implementation for training optimization."""

import numpy as np
import logging
from typing import Optional, Callable, Any

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping to prevent overfitting during training."""
    
    def __init__(self, 
                 patience: int = 10,
                 min_delta: float = 0.0,
                 mode: str = 'min',
                 baseline: Optional[float] = None,
                 restore_best_weights: bool = True):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs with no improvement to wait
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max' - whether lower or higher is better
            baseline: Baseline value to compare against
            restore_best_weights: Whether to restore best weights
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.baseline = baseline
        self.restore_best_weights = restore_best_weights
        
        # Tracking variables
        self.best_score = None
        self.best_epoch = 0
        self.best_weights = None
        self.counter = 0
        self.stopped_epoch = 0
        self.early_stop = False
        
        # Set comparison function
        if mode == 'min':
            self.monitor_op = np.less
            self.min_delta *= -1
        elif mode == 'max':
            self.monitor_op = np.greater
            self.min_delta *= 1
        else:
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")
        
        logger.info(f"EarlyStopping initialized: patience={patience}, mode={mode}")
    
    def __call__(self, current_score: float, model: Optional[Any] = None) -> bool:
        """
        Check if training should stop.
        
        Args:
            current_score: Current metric value
            model: Model instance (for saving best weights)
            
        Returns:
            True if training should stop
        """
        # Initialize best score
        if self.best_score is None:
            self.best_score = current_score
            self.best_epoch = 0
            if model and self.restore_best_weights:
                self._save_model_weights(model)
            return False
        
        # Check for improvement
        if self.monitor_op(current_score - self.min_delta, self.best_score):
            # Improvement found
            self.best_score = current_score
            self.best_epoch = self.counter
            self.counter = 0
            
            if model and self.restore_best_weights:
                self._save_model_weights(model)
            
            logger.debug(f"Improvement found: {current_score:.6f} (best: {self.best_score:.6f})")
        else:
            # No improvement
            self.counter += 1
            logger.debug(f"No improvement for {self.counter} epochs (best: {self.best_score:.6f})")
            
            if self.counter >= self.patience:
                self.early_stop = True
                self.stopped_epoch = self.best_epoch
                logger.info(f"Early stopping triggered after {self.counter} epochs without improvement")
                
                if model and self.restore_best_weights and self.best_weights:
                    self._restore_model_weights(model)
                
                return True
        
        return False
    
    def reset(self) -> None:
        """Reset early stopping state."""
        self.best_score = None
        self.best_epoch = 0
        self.best_weights = None
        self.counter = 0
        self.stopped_epoch = 0
        self.early_stop = False
        logger.debug("EarlyStopping state reset")
    
    def _save_model_weights(self, model: Any) -> None:
        """Save model weights."""
        try:
            if hasattr(model, 'state_dict'):
                # PyTorch model
                import copy
                self.best_weights = copy.deepcopy(model.state_dict())
            elif hasattr(model, 'get_weights'):
                # Keras/TensorFlow model
                self.best_weights = model.get_weights()
            else:
                # Try to copy the entire model
                import copy
                self.best_weights = copy.deepcopy(model)
            logger.debug("Saved best model weights")
        except Exception as e:
            logger.warning(f"Failed to save model weights: {e}")
    
    def _restore_model_weights(self, model: Any) -> None:
        """Restore best model weights."""
        try:
            if hasattr(model, 'load_state_dict') and isinstance(self.best_weights, dict):
                # PyTorch model
                model.load_state_dict(self.best_weights)
            elif hasattr(model, 'set_weights') and isinstance(self.best_weights, list):
                # Keras/TensorFlow model
                model.set_weights(self.best_weights)
            else:
                logger.warning("Cannot restore model weights - incompatible model type")
            logger.info("Restored best model weights")
        except Exception as e:
            logger.error(f"Failed to restore model weights: {e}")
    
    def state_dict(self) -> dict:
        """Get state dictionary for checkpointing."""
        return {
            'best_score': self.best_score,
            'best_epoch': self.best_epoch,
            'counter': self.counter,
            'stopped_epoch': self.stopped_epoch,
            'early_stop': self.early_stop
        }
    
    def load_state_dict(self, state_dict: dict) -> None:
        """Load state from dictionary."""
        self.best_score = state_dict.get('best_score')
        self.best_epoch = state_dict.get('best_epoch', 0)
        self.counter = state_dict.get('counter', 0)
        self.stopped_epoch = state_dict.get('stopped_epoch', 0)
        self.early_stop = state_dict.get('early_stop', False)


class ReduceLROnPlateau:
    """Reduce learning rate when metric stops improving."""
    
    def __init__(self,
                 optimizer: Any,
                 mode: str = 'min',
                 factor: float = 0.5,
                 patience: int = 10,
                 min_lr: float = 1e-7,
                 verbose: bool = True):
        """
        Initialize learning rate scheduler.
        
        Args:
            optimizer: Optimizer instance
            mode: 'min' or 'max'
            factor: Factor to reduce LR by
            patience: Epochs to wait before reducing
            min_lr: Minimum learning rate
            verbose: Print messages
        """
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.verbose = verbose
        
        # Tracking
        self.best_score = None
        self.counter = 0
        self.current_lr = self._get_lr()
        
        # Set comparison function
        if mode == 'min':
            self.monitor_op = np.less
        else:
            self.monitor_op = np.greater
    
    def step(self, current_score: float) -> bool:
        """
        Update learning rate based on metric.
        
        Args:
            current_score: Current metric value
            
        Returns:
            True if LR was reduced
        """
        if self.best_score is None:
            self.best_score = current_score
            return False
        
        if self.monitor_op(current_score, self.best_score):
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            
            if self.counter >= self.patience:
                return self._reduce_lr()
        
        return False
    
    def _reduce_lr(self) -> bool:
        """Reduce learning rate."""
        old_lr = self._get_lr()
        new_lr = max(old_lr * self.factor, self.min_lr)
        
        if new_lr < old_lr:
            self._set_lr(new_lr)
            self.counter = 0
            
            if self.verbose:
                logger.info(f"Reduced learning rate: {old_lr:.2e} -> {new_lr:.2e}")
            
            return True
        
        return False
    
    def _get_lr(self) -> float:
        """Get current learning rate."""
        if hasattr(self.optimizer, 'param_groups'):
            # PyTorch optimizer
            return self.optimizer.param_groups[0]['lr']
        elif hasattr(self.optimizer, 'learning_rate'):
            # TensorFlow optimizer
            return float(self.optimizer.learning_rate)
        else:
            return 0.0
    
    def _set_lr(self, lr: float) -> None:
        """Set learning rate."""
        if hasattr(self.optimizer, 'param_groups'):
            # PyTorch optimizer
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        elif hasattr(self.optimizer, 'learning_rate'):
            # TensorFlow optimizer
            self.optimizer.learning_rate = lr


class ModelCheckpoint:
    """Save model checkpoints based on metric improvements."""
    
    def __init__(self,
                 filepath: str,
                 monitor: str = 'val_loss',
                 mode: str = 'min',
                 save_best_only: bool = True,
                 save_weights_only: bool = False,
                 verbose: bool = True):
        """
        Initialize model checkpoint callback.
        
        Args:
            filepath: Path to save model
            monitor: Metric to monitor
            mode: 'min' or 'max'
            save_best_only: Only save improving models
            save_weights_only: Only save weights (not full model)
            verbose: Print messages
        """
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.verbose = verbose
        
        self.best_score = None
        
        # Set comparison function
        if mode == 'min':
            self.monitor_op = np.less
        else:
            self.monitor_op = np.greater
    
    def __call__(self, model: Any, current_score: float, epoch: int) -> bool:
        """
        Check and save model if improved.
        
        Args:
            model: Model instance
            current_score: Current metric value
            epoch: Current epoch
            
        Returns:
            True if model was saved
        """
        if self.save_best_only:
            if self.best_score is None or self.monitor_op(current_score, self.best_score):
                self.best_score = current_score
                self._save_model(model, epoch)
                
                if self.verbose:
                    logger.info(f"Model checkpoint saved: {self.monitor}={current_score:.6f}")
                
                return True
        else:
            self._save_model(model, epoch)
            return True
        
        return False
    
    def _save_model(self, model: Any, epoch: int) -> None:
        """Save model to file."""
        # Format filepath with epoch
        filepath = self.filepath.format(epoch=epoch)
        
        try:
            if hasattr(model, 'save'):
                # Model has save method
                model.save(filepath)
            elif hasattr(model, 'state_dict'):
                # PyTorch model
                import torch
                if self.save_weights_only:
                    torch.save(model.state_dict(), filepath)
                else:
                    torch.save(model, filepath)
            else:
                # Pickle model
                import pickle
                with open(filepath, 'wb') as f:
                    pickle.dump(model, f)
            
            logger.debug(f"Model saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
