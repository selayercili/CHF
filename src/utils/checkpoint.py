# src/utils/checkpoint.py
"""Checkpoint management utilities for model saving and loading."""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import torch
import pickle
import logging
import pandas as pd

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages model checkpoints with automatic versioning and cleanup."""
    
    def __init__(self, 
                 checkpoint_dir: Path,
                 model_name: str,
                 max_checkpoints: int = 5,
                 save_best_only: bool = False):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Base directory for checkpoints
            model_name: Name of the model
            max_checkpoints: Maximum number of checkpoints to keep
            save_best_only: Only save improving checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir) / model_name
        self.model_name = model_name
        self.max_checkpoints = max_checkpoints
        self.save_best_only = save_best_only
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Track checkpoints
        self.checkpoints = []
        self.best_metric = None
        self.best_checkpoint = None
        
        # Load existing checkpoints
        self._load_checkpoint_history()
        
        logger.info(f"CheckpointManager initialized for {model_name}")
        logger.info(f"Checkpoint directory: {self.checkpoint_dir}")
    
    def save_checkpoint(self,
                       model: Any,
                       epoch: int,
                       metrics: Dict[str, float],
                       optimizer: Optional[Any] = None,
                       metadata: Optional[Dict[str, Any]] = None,
                       is_best: bool = False) -> Path:
        """
        Save model checkpoint.
        
        Args:
            model: Model to save
            epoch: Current epoch
            metrics: Validation metrics
            optimizer: Optimizer state
            metadata: Additional metadata
            is_best: Whether this is the best model
            
        Returns:
            Path to saved checkpoint
        """
        # Determine checkpoint path
        checkpoint_name = f"epoch_{epoch:04d}.pth"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # Prepare checkpoint data
        checkpoint_data = {
            'epoch': epoch,
            'model_name': self.model_name,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'metadata': metadata or {}
        }
        
        # Handle different model types
        if hasattr(model, 'state_dict'):
            # PyTorch model
            checkpoint_data['model_state_dict'] = model.state_dict()
            if optimizer and hasattr(optimizer, 'state_dict'):
                checkpoint_data['optimizer_state_dict'] = optimizer.state_dict()
        elif hasattr(model, 'save'):
            # Custom save method
            temp_path = checkpoint_path.with_suffix('.tmp')
            model.save(temp_path, metadata=checkpoint_data)
            shutil.move(temp_path, checkpoint_path)
            logger.info(f"Saved checkpoint using custom save method: {checkpoint_path}")
            self._update_checkpoint_history(checkpoint_path, metrics, epoch)
            return checkpoint_path
        else:
            # Pickle entire model
            checkpoint_data['model'] = model
            if optimizer:
                checkpoint_data['optimizer'] = optimizer
        
        # Save checkpoint
        if checkpoint_path.suffix == '.pth':
            torch.save(checkpoint_data, checkpoint_path)
        else:
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint_data, f)
        
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Save as best if specified
        if is_best:
            self._save_best_checkpoint(checkpoint_path, metrics)
        
        # Update checkpoint history
        self._update_checkpoint_history(checkpoint_path, metrics, epoch)
        
        # Cleanup old checkpoints
        self._cleanup_checkpoints()
        
        return checkpoint_path
    
    def load_checkpoint(self, 
                       checkpoint_path: Optional[Path] = None,
                       load_best: bool = True) -> Dict[str, Any]:
        """
        Load checkpoint.
        
        Args:
            checkpoint_path: Specific checkpoint to load
            load_best: Load best checkpoint if path not specified
            
        Returns:
            Checkpoint data dictionary
        """
        if checkpoint_path is None:
            if load_best and self.best_checkpoint:
                checkpoint_path = self.best_checkpoint
            else:
                # Load latest checkpoint
                checkpoint_path = self.get_latest_checkpoint()
        
        if checkpoint_path is None or not checkpoint_path.exists():
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
        
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        
        # Load based on file type
        if checkpoint_path.suffix == '.pth':
            checkpoint_data = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        else:
            with open(checkpoint_path, 'rb') as f:
                checkpoint_data = pickle.load(f)
        
        return checkpoint_data
    
    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get the most recent checkpoint."""
        checkpoints = list(self.checkpoint_dir.glob("epoch_*.pth"))
        
        if not checkpoints:
            return None
        
        # Sort by epoch number
        checkpoints.sort(key=lambda x: int(x.stem.split('_')[1]))
        return checkpoints[-1]
    
    def get_best_checkpoint(self) -> Optional[Path]:
        """Get the best checkpoint."""
        best_path = self.checkpoint_dir / "best_model.pth"
        
        if best_path.exists():
            return best_path
        
        return self.best_checkpoint
    
    def _save_best_checkpoint(self, checkpoint_path: Path, metrics: Dict[str, float]) -> None:
        """Save checkpoint as best model."""
        best_path = self.checkpoint_dir / "best_model.pth"
        
        # Copy checkpoint to best_model.pth
        shutil.copy2(checkpoint_path, best_path)
        
        # Save best metrics
        best_info = {
            'source_checkpoint': str(checkpoint_path),
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.checkpoint_dir / "best_model_info.json", 'w') as f:
            json.dump(best_info, f, indent=2)
        
        self.best_checkpoint = best_path
        logger.info(f"Saved best model: {best_path}")
    
    def _update_checkpoint_history(self, 
                                 checkpoint_path: Path, 
                                 metrics: Dict[str, float],
                                 epoch: int) -> None:
        """Update checkpoint history."""
        checkpoint_info = {
            'path': str(checkpoint_path),
            'epoch': epoch,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        self.checkpoints.append(checkpoint_info)
        
        # Save history
        history_path = self.checkpoint_dir / "checkpoint_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.checkpoints, f, indent=2)
    
    def _load_checkpoint_history(self) -> None:
        """Load existing checkpoint history."""
        history_path = self.checkpoint_dir / "checkpoint_history.json"
        
        if history_path.exists():
            with open(history_path, 'r') as f:
                self.checkpoints = json.load(f)
            logger.debug(f"Loaded {len(self.checkpoints)} checkpoint records")
        
        # Check for best model
        best_info_path = self.checkpoint_dir / "best_model_info.json"
        if best_info_path.exists():
            with open(best_info_path, 'r') as f:
                best_info = json.load(f)
                self.best_checkpoint = self.checkpoint_dir / "best_model.pth"
                self.best_metric = best_info.get('metrics', {})
    
    def _cleanup_checkpoints(self) -> None:
        """Remove old checkpoints keeping only the most recent ones."""
        if self.max_checkpoints <= 0:
            return
        
        # Get all epoch checkpoints
        epoch_checkpoints = list(self.checkpoint_dir.glob("epoch_*.pth"))
        
        # Sort by epoch number
        epoch_checkpoints.sort(key=lambda x: int(x.stem.split('_')[1]))
        
        # Keep only the most recent checkpoints
        if len(epoch_checkpoints) > self.max_checkpoints:
            checkpoints_to_remove = epoch_checkpoints[:-self.max_checkpoints]
            
            for checkpoint in checkpoints_to_remove:
                checkpoint.unlink()
                logger.debug(f"Removed old checkpoint: {checkpoint}")
            
            # Update checkpoint history
            kept_paths = [str(cp) for cp in epoch_checkpoints[-self.max_checkpoints:]]
            self.checkpoints = [cp for cp in self.checkpoints if cp['path'] in kept_paths]
    
    def get_checkpoint_info(self) -> List[Dict[str, Any]]:
        """Get information about all checkpoints."""
        return self.checkpoints
    
    def compare_checkpoints(self, metric: str = 'loss') -> pd.DataFrame:
        """
        Compare checkpoints by a specific metric.
        
        Args:
            metric: Metric to compare
            
        Returns:
            DataFrame with checkpoint comparisons
        """
        import pandas as pd
        
        data = []
        for cp in self.checkpoints:
            row = {
                'epoch': cp['epoch'],
                'timestamp': cp['timestamp'],
                metric: cp['metrics'].get(metric, None)
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        df = df.sort_values('epoch')
        
        return df


def save_training_state(checkpoint_dir: Path,
                       epoch: int,
                       model: Any,
                       optimizer: Any,
                       scheduler: Optional[Any] = None,
                       scaler: Optional[Any] = None,
                       metrics: Optional[Dict[str, float]] = None,
                       config: Optional[Dict[str, Any]] = None) -> None:
    """
    Save complete training state.
    
    Args:
        checkpoint_dir: Directory to save checkpoint
        epoch: Current epoch
        model: Model instance
        optimizer: Optimizer instance
        scheduler: Learning rate scheduler
        scaler: Gradient scaler for mixed precision
        metrics: Current metrics
        config: Training configuration
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    state = {
        'epoch': epoch,
        'timestamp': datetime.now().isoformat(),
        'metrics': metrics or {}
    }
    
    # Save model state
    if hasattr(model, 'state_dict'):
        state['model_state_dict'] = model.state_dict()
    else:
        state['model'] = model
    
    # Save optimizer state
    if hasattr(optimizer, 'state_dict'):
        state['optimizer_state_dict'] = optimizer.state_dict()
    else:
        state['optimizer'] = optimizer
    
    # Save scheduler state
    if scheduler and hasattr(scheduler, 'state_dict'):
        state['scheduler_state_dict'] = scheduler.state_dict()
    
    # Save scaler state
    if scaler and hasattr(scaler, 'state_dict'):
        state['scaler_state_dict'] = scaler.state_dict()
    
    # Save configuration
    if config:
        state['config'] = config
    
    # Save checkpoint
    checkpoint_path = checkpoint_dir / f"training_state_epoch_{epoch}.pth"
    torch.save(state, checkpoint_path)
    
    logger.info(f"Saved training state: {checkpoint_path}")
    
    # Also save a 'latest' checkpoint
    latest_path = checkpoint_dir / "latest_training_state.pth"
    shutil.copy2(checkpoint_path, latest_path)


def load_training_state(checkpoint_path: Path,
                       model: Any,
                       optimizer: Any,
                       scheduler: Optional[Any] = None,
                       scaler: Optional[Any] = None) -> Tuple[int, Dict[str, float]]:
    """
    Load training state from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint
        model: Model instance
        optimizer: Optimizer instance
        scheduler: Learning rate scheduler
        scaler: Gradient scaler
        
    Returns:
        Tuple of (epoch, metrics)
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    logger.info(f"Loading training state from: {checkpoint_path}")
    
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Load model state
    if 'model_state_dict' in state:
        model.load_state_dict(state['model_state_dict'])
    elif 'model' in state:
        # Handle full model pickle
        return state['model'], state.get('epoch', 0), state.get('metrics', {})
    
    # Load optimizer state
    if 'optimizer_state_dict' in state and hasattr(optimizer, 'load_state_dict'):
        optimizer.load_state_dict(state['optimizer_state_dict'])
    
    # Load scheduler state
    if scheduler and 'scheduler_state_dict' in state:
        scheduler.load_state_dict(state['scheduler_state_dict'])
    
    # Load scaler state
    if scaler and 'scaler_state_dict' in state:
        scaler.load_state_dict(state['scaler_state_dict'])
    
    epoch = state.get('epoch', 0)
    metrics = state.get('metrics', {})
    
    logger.info(f"Resumed from epoch {epoch}")
    
    return epoch, metrics
