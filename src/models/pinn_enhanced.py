# src/models/pinn.py
"""
Physics-Informed Neural Network (PINN) Model Implementation

This module implements a PINN with configurable physics equations
loaded from a configuration file for flexibility.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


class PhysicsConstraints:
    """Manages physics constraints for PINN."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize physics constraints from configuration.
        
        Args:
            config_path: Path to physics configuration file
        """
        self.config = self._load_config(config_path)
        self.equations = self.config.get('physics_equations', {})
        self.variable_mapping = self.config.get('variable_mapping', {})
        self.physical_constants = self.config.get('physical_constants', {})
        self.active_constraints = []
        
        # Load active constraints
        for name, eq_config in self.equations.items():
            if eq_config.get('active', True):
                self.active_constraints.append(name)
        
        logger.info(f"Loaded {len(self.active_constraints)} active physics constraints")
    
    def _load_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
        """Load physics configuration from file."""
        if config_path is None:
            config_path = Path("configs/physics_configs.yaml")
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            logger.warning(f"Physics config not found at {config_path}, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default physics configuration."""
        return {
            'physics_equations': {
                'positive_chf': {
                    'type': 'bound_constraint',
                    'weight': 0.5,
                    'active': True
                },
                'mass_flux_monotonicity': {
                    'type': 'gradient_constraint',
                    'weight': 0.3,
                    'active': True
                }
            },
            'variable_mapping': {
                'pressure': 0,
                'mass_flux': 1,
                'x_exit': 2,
                'diameter_heated': 3,
                'diameter_equivalent': 4,
                'length': 5
            }
        }
    
    def compute_physics_loss(self, inputs: torch.Tensor, outputs: torch.Tensor,
                           model: nn.Module) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total physics loss and individual components.
        
        Args:
            inputs: Input tensor
            outputs: Model predictions
            model: Neural network model
            
        Returns:
            Tuple of (total_loss, loss_components)
        """
        total_loss = torch.tensor(0.0, device=inputs.device, requires_grad=True)
        loss_components = {}
        
        for constraint_name in self.active_constraints:
            constraint_config = self.equations[constraint_name]
            constraint_type = constraint_config['type']
            weight = constraint_config.get('weight', 1.0)
            
            # Compute constraint-specific loss
            if constraint_type == 'gradient_constraint':
                loss = self._compute_gradient_constraint(
                    inputs, outputs, model, constraint_config
                )
            elif constraint_type == 'bound_constraint':
                loss = self._compute_bound_constraint(
                    outputs, constraint_config
                )
            elif constraint_type == 'relation_constraint':
                loss = self._compute_relation_constraint(
                    inputs, outputs, constraint_config
                )
            else:
                logger.warning(f"Unknown constraint type: {constraint_type}")
                continue
            
            # Add to total loss
            weighted_loss = weight * loss
            total_loss = total_loss + weighted_loss
            loss_components[constraint_name] = loss.item()
        
        return total_loss, loss_components
    
    def _compute_gradient_constraint(self, inputs: torch.Tensor, outputs: torch.Tensor,
                                   model: nn.Module, config: Dict[str, Any]) -> torch.Tensor:
        """Compute gradient-based constraint loss."""
        # Get variable indices
        variables = config.get('variables', [])
        if len(variables) < 2:
            return torch.tensor(0.0, device=inputs.device)
        
        input_var = variables[0]
        var_idx = self.variable_mapping.get(input_var, 0)
        
        # Enable gradients
        inputs = inputs.detach().requires_grad_(True)
        outputs = model(inputs)
        
        # Compute gradient
        grad = torch.autograd.grad(
            outputs.sum(), inputs, 
            create_graph=True, retain_graph=True
        )[0]
        
        # Get specific gradient
        var_grad = grad[:, var_idx]
        
        # Apply constraint based on equation
        equation = config.get('equation', '')
        if '> 0' in equation:
            # Penalize negative gradients
            loss = torch.relu(-var_grad).mean()
        elif '< 0' in equation:
            # Penalize positive gradients
            loss = torch.relu(var_grad).mean()
        else:
            loss = torch.abs(var_grad).mean()
        
        return loss
    
    def _compute_bound_constraint(self, outputs: torch.Tensor, 
                                config: Dict[str, Any]) -> torch.Tensor:
        """Compute bound constraint loss."""
        min_value = config.get('min_value', -np.inf)
        max_value = config.get('max_value', np.inf)
        
        loss = torch.tensor(0.0, device=outputs.device)
        
        if min_value > -np.inf:
            # Penalize values below minimum
            loss = loss + torch.relu(min_value - outputs).mean()
        
        if max_value < np.inf:
            # Penalize values above maximum
            loss = loss + torch.relu(outputs - max_value).mean()
        
        return loss
    
    def _compute_relation_constraint(self, inputs: torch.Tensor, outputs: torch.Tensor,
                                   config: Dict[str, Any]) -> torch.Tensor:
        """Compute relation-based constraint loss."""
        # This is a placeholder for more complex physics relations
        # Can be extended based on specific equations
        return torch.tensor(0.0, device=inputs.device)


class PINNModel(nn.Module):
    """Neural network architecture for PINN."""
    
    def __init__(self, input_size: int, hidden_layers: List[int], 
                 activation: str = 'tanh', dropout_rate: float = 0.1):
        """
        Initialize PINN model architecture.
        
        Args:
            input_size: Number of input features
            hidden_layers: List of hidden layer sizes
            activation: Activation function
            dropout_rate: Dropout rate
        """
        super().__init__()
        
        # Build layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            
            # Activation
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'elu':
                layers.append(nn.ELU())
            else:
                raise ValueError(f"Unknown activation: {activation}")
            
            # Dropout
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier initialization
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)


class Pinn:
    """Physics-Informed Neural Network wrapper."""
    
    def __init__(self, learning_rate: float = 0.001, 
                 lambda_physics: float = 0.01,
                 optimizer: str = 'adam',
                 **kwargs):
        """
        Initialize PINN.
        
        Args:
            learning_rate: Learning rate
            lambda_physics: Weight for physics loss
            optimizer: Optimizer type
            **kwargs: Additional configuration
        """
        self.learning_rate = learning_rate
        self.lambda_physics = lambda_physics
        self.optimizer_type = optimizer
        
        # Extract configuration
        self.architecture_config = kwargs.get('architecture', {})
        self.physics_config = kwargs.get('physics', {})
        self.debug = kwargs.get('debug', False)
        
        # Model components (initialized later)
        self.model = None
        self.optimizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        
        # Physics constraints
        physics_config_path = self.physics_config.get('equations_config')
        if physics_config_path:
            physics_config_path = Path(physics_config_path)
        self.physics_constraints = PhysicsConstraints(physics_config_path)
        
        # Training state
        self.is_fitted = False
        self.epoch = 0
        self.best_loss = float('inf')
        
        # Adaptive weighting
        self.use_adaptive_weighting = self.physics_config.get('adaptive_weighting', False)
        self.physics_warmup_epochs = self.physics_config.get('warmup_epochs', 10)
        
        logger.info(f"PINN initialized with physics weight: {lambda_physics}")
    
    def _build_model(self, input_size: int) -> PINNModel:
        """Build the neural network model."""
        hidden_layers = self.architecture_config.get('hidden_layers', [128, 128, 64])
        activation = self.architecture_config.get('activation', 'tanh')
        dropout_rate = self.architecture_config.get('dropout_rate', 0.1)
        
        model = PINNModel(
            input_size=input_size,
            hidden_layers=hidden_layers,
            activation=activation,
            dropout_rate=dropout_rate
        ).to(self.device)
        
        logger.info(f"Built PINN model: {input_size} -> {hidden_layers} -> 1")
        
        return model
    
    def _prepare_data(self, data: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare data for training/validation."""
        # Find target column
        target_col = [col for col in data.columns if 'chf' in col.lower()][-1]
        
        X = data.drop(target_col, axis=1).values
        y = data[target_col].values
        
        # Scale data
        if not self.is_fitted:
            X_scaled = self.input_scaler.fit_transform(X)
            y_scaled = self.target_scaler.fit_transform(y.reshape(-1, 1)).flatten()
            
            # Initialize model
            input_size = X.shape[1]
            self.model = self._build_model(input_size)
            
            # Initialize optimizer
            if self.optimizer_type == 'adam':
                self.optimizer = torch.optim.Adam(
                    self.model.parameters(), 
                    lr=self.learning_rate
                )
            elif self.optimizer_type == 'lbfgs':
                self.optimizer = torch.optim.LBFGS(
                    self.model.parameters(),
                    lr=self.learning_rate
                )
            else:
                raise ValueError(f"Unknown optimizer: {self.optimizer_type}")
        else:
            X_scaled = self.input_scaler.transform(X)
            y_scaled = self.target_scaler.transform(y.reshape(-1, 1)).flatten()
        
        # Convert to tensors
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_scaled, dtype=torch.float32).to(self.device)
        
        return X_tensor, y_tensor
    
    def _compute_adaptive_weight(self) -> float:
        """Compute adaptive physics weight based on training progress."""
        if self.epoch < self.physics_warmup_epochs:
            # Gradually increase physics weight
            return self.lambda_physics * (self.epoch / self.physics_warmup_epochs)
        else:
            return self.lambda_physics
    
    def train_epoch(self, train_data: pd.DataFrame, 
                   batch_size: int = 32,
                   optimizer: Any = None) -> Dict[str, float]:
        """Train for one epoch."""
        X_train, y_train = self._prepare_data(train_data)
        
        # Create data loader
        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        
        self.model.train()
        
        # Track losses
        total_loss_sum = 0.0
        data_loss_sum = 0.0
        physics_loss_sum = 0.0
        physics_components = {}
        
        # Get adaptive weight
        current_physics_weight = self._compute_adaptive_weight() if self.use_adaptive_weighting else self.lambda_physics
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            
            # Data loss
            data_loss = nn.functional.mse_loss(outputs.squeeze(), targets)
            
            # Physics loss
            physics_loss, components = self.physics_constraints.compute_physics_loss(
                inputs, outputs, self.model
            )
            
            # Combined loss
            total_loss = data_loss + current_physics_weight * physics_loss
            
            # Backward pass
            if torch.isfinite(total_loss):
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
            else:
                logger.warning(f"Non-finite loss encountered at batch {batch_idx}")
            
            # Accumulate losses
            total_loss_sum += total_loss.item()
            data_loss_sum += data_loss.item()
            physics_loss_sum += physics_loss.item()
            
            # Accumulate physics components
            for name, value in components.items():
                if name not in physics_components:
                    physics_components[name] = 0.0
                physics_components[name] += value
        
        # Average losses
        n_batches = len(dataloader)
        avg_total_loss = total_loss_sum / n_batches
        avg_data_loss = data_loss_sum / n_batches
        avg_physics_loss = physics_loss_sum / n_batches
        
        # Average physics components
        for name in physics_components:
            physics_components[name] /= n_batches
        
        # Update training state
        self.epoch += 1
        self.is_fitted = True
        
        # Prepare metrics
        metrics = {
            'loss': avg_total_loss,
            'data_loss': avg_data_loss,
            'physics_loss': avg_physics_loss,
            'rmse': np.sqrt(avg_data_loss),
            'physics_weight': current_physics_weight
        }
        
        # Add physics components to metrics
        for name, value in physics_components.items():
            metrics[f'physics_{name}'] = value
        
        # Debug logging
        if self.debug and self.epoch % 10 == 0:
            logger.debug(f"Epoch {self.epoch}: {metrics}")
        
        return metrics
    
    def validate(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """Validate model (physics loss not computed during validation)."""
        X_test, y_test = self._prepare_data(test_data)
        
        self.model.eval()
        
        with torch.no_grad():
            predictions = self.model(X_test).squeeze()
            mse = nn.functional.mse_loss(predictions, y_test)
            mae = nn.functional.l1_loss(predictions, y_test)
        
        # Calculate R² score
        ss_res = torch.sum((y_test - predictions) ** 2)
        ss_tot = torch.sum((y_test - y_test.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        
        return {
            'loss': mse.item(),
            'rmse': np.sqrt(mse.item()),
            'mae': mae.item(),
            'r2': r2.item()
        }
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")
        
        # Prepare data
        target_cols = [col for col in data.columns if 'chf' in col.lower()]
        if target_cols:
            X = data.drop(target_cols[-1], axis=1).values
        else:
            X = data.values
        
        # Scale and convert
        X_scaled = self.input_scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            predictions_scaled = self.model(X_tensor).squeeze().cpu().numpy()
        
        # Inverse transform
        predictions = self.target_scaler.inverse_transform(
            predictions_scaled.reshape(-1, 1)
        ).flatten()
        
        return predictions
    
    def save(self, path: Path, metadata: Dict[str, Any] = None):
        """Save model state."""
        save_dict = {
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'epoch': self.epoch,
            'architecture_config': self.architecture_config,
            'physics_config': self.physics_config,
            'input_scaler': self.input_scaler,
            'target_scaler': self.target_scaler,
            'is_fitted': self.is_fitted,
            'best_loss': self.best_loss
        }
        
        if metadata:
            save_dict.update(metadata)
        
        torch.save(save_dict, path)
        logger.info(f"PINN model saved to {path}")
    
    def load(self, path: Path):
        """Load model state."""
        # Add safe globals
        import sklearn.preprocessing
        torch.serialization.add_safe_globals([sklearn.preprocessing.StandardScaler])
        
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        # Restore configuration
        self.architecture_config = checkpoint.get('architecture_config', {})
        self.physics_config = checkpoint.get('physics_config', {})
        
        # Rebuild model
        input_size = checkpoint['model_state']['network.0.weight'].shape[1]
        self.model = self._build_model(input_size)
        self.model.load_state_dict(checkpoint['model_state'])
        
        # Restore optimizer
        if self.optimizer_type == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), 
                lr=self.learning_rate
            )
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        # Restore scalers
        self.input_scaler = checkpoint['input_scaler']
        self.target_scaler = checkpoint['target_scaler']
        
        # Restore state
        self.epoch = checkpoint['epoch']
        self.is_fitted = checkpoint['is_fitted']
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        
        logger.info(f"PINN model loaded from {path}")
        
        return checkpoint
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get gradient-based feature importance."""
        if not self.is_fitted:
            raise ValueError("Model must be trained first")
        
        # Create dummy input
        dummy_input = torch.ones(1, self.model.network[0].in_features, 
                                device=self.device, requires_grad=True)
        
        # Forward pass
        output = self.model(dummy_input)
        
        # Backward pass
        output.backward()
        
        # Get gradients
        gradients = torch.abs(dummy_input.grad).cpu().numpy().flatten()
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': [f'feature_{i}' for i in range(len(gradients))],
            'importance': gradients
        }).sort_values('importance', ascending=False)
        
        return importance_df