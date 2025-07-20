"""
Physics-Informed Neural Network (PINN) Model Implementation

Combines data-driven loss with physics-based constraints for heat flux prediction.
"""
import torch # type: ignore
import torch.nn as nn # type: ignore
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import pickle
from sklearn.preprocessing import StandardScaler
import CoolProp.CoolProp as CP 
import logging
import traceback

# Setup logging
logger = logging.getLogger(__name__)

class Pinn:
    """PINN model wrapper with physics constraints for heat flux prediction."""
    
    def __init__(self, hidden_size: int = 64, learning_rate: float = 0.001, 
                 lambda_physics: float = 0.01, **kwargs):  # Reduced physics weight
        """
        Args:
            hidden_size: Neurons per hidden layer.
            learning_rate: Optimizer learning rate.
            lambda_physics: Weight for physics loss term (reduced default).
        """
        logger.info(f"Initializing PINN model with hidden_size={hidden_size}, lr={learning_rate}, lambda_physics={lambda_physics}")
        
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.lambda_physics = lambda_physics
        self.model = None
        self.optimizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_fitted = False
        self.epoch = 0
        self.input_size = None
        self.input_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        
        # Add debugging flags
        self.debug = kwargs.get('debug', False)
        self.physics_warmup_epochs = kwargs.get('physics_warmup_epochs', 10)  # Gradually introduce physics
        
        # CHF Equation Parameters (will be registered to model later)
        # Store initial values, actual nn.Parameters created in _build_model
        self.chf_param_init = {
            'A': 0.5,   # Base coefficient
            'B': 1e-4,  # Quality coefficient
            'C': 0.1    # Length offset
        }
        self.bowring_params = {}  # Will hold actual nn.Parameters after model creation
        
        logger.info(f"PINN initialized on device: {self.device}")

    def _convert_units(self, inputs: torch.Tensor, chf_exp: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Convert inputs to SI units and scale CHF from MW/m² to W/m²."""
        eps = 1e-8  # Small epsilon to avoid log(0)
        
        # Convert pressure (MPa → Pa) and ensure positive
        pressure_Pa = (torch.abs(inputs[:, 1]) + eps) * 1e6  
        
        # Convert diameters/length (mm → m)
        D_h_m = (torch.abs(inputs[:, 5]) + eps) * 1e-3  
        length_m = (torch.abs(inputs[:, 6]) + eps) * 1e-3  
        
        # Convert CHF (MW/m² → W/m²) if provided
        chf_Wm2 = chf_exp * 1e6 if chf_exp is not None else None
        
        return {
            'pressure_Pa': pressure_Pa,
            'mass_flux': torch.abs(inputs[:, 2]) + eps,
            'x_e_out': inputs[:, 3],  # Can be negative
            'D_h_m': D_h_m,
            'length_m': length_m,
            'geom_tube': inputs[:, 7] if inputs.shape[1] > 6 else torch.zeros_like(pressure_Pa),
            'geom_annulus': inputs[:, 8] if inputs.shape[1] > 7 else torch.zeros_like(pressure_Pa),
            'chf_Wm2': chf_Wm2  # Only included if chf_exp was passed
        }
        
    def _build_model(self, input_size: int) -> nn.Module:
        """Enhanced 4-layer MLP with better initialization and dropout."""
        logger.info(f"Building PINN model architecture with input_size={input_size}")
        
        model = nn.Sequential(
            nn.Linear(input_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, 1)
        ).to(self.device)
        
        # Better weight initialization
        for layer in model:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
        
        # Create CHF parameters as nn.Parameters and register them to the model
        logger.info("Registering CHF parameters to model")
        for name, init_value in self.chf_param_init.items():
            param = nn.Parameter(torch.tensor(init_value, device=self.device))
            model.register_parameter(f'chf_{name}', param)
            self.bowring_params[name] = param  # Store reference
            logger.debug(f"Registered CHF parameter {name} with initial value {init_value}")
        
        # Log model structure
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model built with {total_params} trainable parameters")
        
        return model

    def _physics_loss(self, inputs: torch.Tensor, predictions: torch.Tensor) -> torch.Tensor:
        """Physics loss using the new CHF equation."""
        # Convert inputs to SI units
        converted = self._convert_units(inputs)
        pressure_Pa = converted['pressure_Pa']
        mass_flux = converted['mass_flux']
        x_e_out = converted['x_e_out']
        D_h_m = converted['D_h_m']
        length_m = converted['length_m']
        
        # Get latent heat using CoolProp
        h_fg = torch.stack([
            torch.tensor(CP.PropsSI('Hvap', 'P', p.item(), 'Q', 1, 'Water'), 
            device=self.device
        ) for p in pressure_Pa])
        
        # Extract parameters
        A = self.bowring_params['A']
        B = self.bowring_params['B']
        C = self.bowring_params['C']
        
        # Calculate new CHF equation
        numerator = A - B * x_e_out * h_fg
        denominator = C + length_m - (4 * B * length_m) / (mass_flux * D_h_m)
        q_chf_new = numerator / denominator
        
        # Convert model predictions to W/m²
        q_chf_pred = predictions.squeeze()
        if hasattr(self, 'target_scaler'):
            q_chf_pred = self.target_scaler.inverse_transform(
                q_chf_pred.cpu().numpy().reshape(-1, 1)
            ).flatten()
            q_chf_pred = torch.tensor(q_chf_pred, device=self.device) * 1e6
        else:
            q_chf_pred = q_chf_pred * 1e6

        # Physics loss (MSE between predictions and new equation)
        physics_loss = torch.mean((q_chf_pred - q_chf_new) ** 2)
        
        # Add penalty for negative denominators
        penalty = torch.mean(torch.relu(-denominator) ** 2)
        physics_loss += 0.1 * penalty
        
        return physics_loss

    def _prepare_data(self, data: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """Converts DataFrame to tensors and initializes model if needed."""
        target_col = [col for col in data.columns if 'chf_exp' in col][0]
        X = data.drop(target_col, axis=1).values
        y = data[target_col].values
        
        logger.debug(f"Preparing data: X shape={X.shape}, y shape={y.shape}")
        
        # Add scaling with better handling
        if not self.is_fitted:
            X = self.input_scaler.fit_transform(X)
            y = self.target_scaler.fit_transform(y.reshape(-1, 1)).flatten()
            logger.info("Fitted scalers to training data")
        else:
            X = self.input_scaler.transform(X)
            y = self.target_scaler.transform(y.reshape(-1, 1)).flatten()
        
        if self.model is None:
            self.input_size = X.shape[1]
            self.model = self._build_model(self.input_size)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
            logger.info(f"Model and optimizer initialized for {self.input_size} input features")
        
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)
        return X_tensor, y_tensor

    def train_epoch(self, train_data: pd.DataFrame, batch_size: int = 32, 
                   optimizer: Any = None) -> Dict[str, float]:
        """Trains for one epoch with adaptive physics loss weighting."""
        X_train, y_train = self._prepare_data(train_data)
        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.model.train()
        total_loss = 0.0
        total_data_loss = 0.0
        total_physics_loss = 0.0
        
        # Adaptive physics weight (gradual introduction)
        current_physics_weight = self.lambda_physics
        if self.epoch < self.physics_warmup_epochs:
            # Gradually increase physics weight
            current_physics_weight = self.lambda_physics * (self.epoch / self.physics_warmup_epochs)
        
        logger.debug(f"Starting epoch {self.epoch + 1} with physics weight {current_physics_weight:.6f}")
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            # Data loss (MSE)
            data_loss = torch.nn.functional.mse_loss(outputs.squeeze(), targets)
            
            # Physics loss (with adaptive weight)
            physics_loss = self._physics_loss(inputs, outputs)
            
            # Combined loss
            loss = data_loss + current_physics_weight * physics_loss
            
            # Check for NaN/Inf
            if not torch.isfinite(loss):
                logger.warning(f"Non-finite loss at epoch {self.epoch}, batch {batch_idx}: loss={loss}, data_loss={data_loss}, physics_loss={physics_loss}")
                continue
            
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            total_data_loss += data_loss.item()
            total_physics_loss += physics_loss.item()
        
        avg_loss = total_loss / len(dataloader)
        avg_data_loss = total_data_loss / len(dataloader)
        avg_physics_loss = total_physics_loss / len(dataloader)
        
        self.epoch += 1
        self.is_fitted = True
        
        # Enhanced metrics
        metrics = {
            'loss': avg_loss,
            'rmse': np.sqrt(avg_data_loss),
            'data_loss': avg_data_loss,
            'physics_loss': avg_physics_loss,
            'physics_weight': current_physics_weight
        }
        
        # Log training progress
        logger.info(f"Epoch {self.epoch} completed: loss={avg_loss:.6f}, data_loss={avg_data_loss:.6f}, physics_loss={avg_physics_loss:.6f}")
        
        # Log CHF parameters
        if self.epoch % 10 == 0:  # Log every 10 epochs
            chf_params = self.get_chf_parameters()
            logger.info(f"CHF parameters at epoch {self.epoch}: {chf_params}")
        
        return metrics

    def validate(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """Validates model (physics loss not computed during validation)."""
        X_test, y_test = self._prepare_data(test_data)
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_test).squeeze()
            mse = torch.nn.functional.mse_loss(predictions, y_test)
            mae = torch.nn.functional.l1_loss(predictions, y_test)
        
        metrics = {'loss': mse.item(), 'rmse': np.sqrt(mse.item()), 'mae': mae.item()}
        logger.debug(f"Validation metrics: {metrics}")
        return metrics

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Makes predictions (same as other models)."""
        target_col = [col for col in data.columns if 'chf_exp' in col][0] if any('chf_exp' in col for col in data.columns) else None
        if target_col and target_col in data.columns:
            X = data.drop(target_col, axis=1).values
        else:
            X = data.values
            
        X = self.input_scaler.transform(X)  # Scale input
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            predictions_scaled = self.model(X_tensor).squeeze().cpu().numpy()
        
        # Inverse scale predictions
        predictions_scaled = predictions_scaled.reshape(-1, 1)
        return self.target_scaler.inverse_transform(predictions_scaled).flatten()

    def save(self, path: Path, metadata: Dict[str, Any] = None):
        """Saves model state."""
        logger.info(f"PINN save() called with path: {path}")
        
        if self.model is None:
            logger.error("Model not initialized - nothing to save")
            raise ValueError("Model not initialized - nothing to save")
        
        try:
            # Log current state
            logger.info(f"Model state: is_fitted={self.is_fitted}, epoch={self.epoch}, input_size={self.input_size}")
            
            # Extract CHF parameters as regular values (not nn.Parameter objects)
            chf_params_dict = {name: param.item() for name, param in self.bowring_params.items()}
            logger.info(f"CHF parameters to save: {chf_params_dict}")
            
            save_dict = {
                'model_state': self.model.state_dict(),
                'optimizer_state': self.optimizer.state_dict() if self.optimizer else None,
                'epoch': self.epoch,
                'input_size': self.input_size,
                'hidden_size': self.hidden_size,
                'lambda_physics': self.lambda_physics,
                'is_fitted': self.is_fitted,
                # Serialize scalers like in NN model
                'input_scaler': pickle.dumps(self.input_scaler),
                'target_scaler': pickle.dumps(self.target_scaler),
                'chf_params': chf_params_dict  # Save as regular dict, not nn.Parameters
            }
            
            if metadata:
                save_dict['metadata'] = metadata
                logger.debug(f"Metadata included: {metadata}")
            
            # Ensure path has .pth extension (CRITICAL FIX)
            path = Path(path)
            original_path = path
            if not path.suffix:
                path = path.with_suffix('.pth')
                logger.warning(f"Path had no suffix, added .pth: {original_path} -> {path}")
            elif path.suffix != '.pth':
                logger.warning(f"Path has non-.pth suffix: {path}")
            
            # Check if directory exists
            if not path.parent.exists():
                logger.warning(f"Directory {path.parent} does not exist, creating it")
                path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save the checkpoint
            logger.info(f"Saving PINN checkpoint to: {path}")
            torch.save(save_dict, path)
            
            # Verify the file was created
            if path.exists():
                file_size = path.stat().st_size
                logger.info(f"✓ Successfully saved PINN model to {path} (size: {file_size} bytes)")
            else:
                logger.error(f"ERROR: File not found after save: {path}")
                raise IOError(f"Failed to save model to {path}")
                
        except Exception as e:
            logger.error(f"ERROR saving PINN model: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def load(self, path: Path):
        """Loads model state with proper handling of sklearn objects."""
        logger.info(f"Loading PINN model from: {path}")
        
        from sklearn.preprocessing import StandardScaler
        
        # Add safe globals for sklearn objects
        torch.serialization.add_safe_globals([StandardScaler])
        
        try:
            # Check if file exists
            if not path.exists():
                logger.error(f"Model file not found: {path}")
                raise FileNotFoundError(f"Model file not found: {path}")
            
            file_size = path.stat().st_size
            logger.info(f"Loading file of size: {file_size} bytes")
            
            # Load with weights_only=False since we have sklearn objects
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            
            self.input_size = checkpoint['input_size']
            self.hidden_size = checkpoint['hidden_size']
            self.lambda_physics = checkpoint['lambda_physics']
            self.epoch = checkpoint['epoch']
            self.is_fitted = checkpoint['is_fitted']
            
            logger.info(f"Loaded checkpoint info: epoch={self.epoch}, input_size={self.input_size}")
            
            # Load scalers (they were pickled)
            if 'input_scaler' in checkpoint:
                self.input_scaler = pickle.loads(checkpoint['input_scaler'])
                logger.debug("Loaded input scaler")
            if 'target_scaler' in checkpoint:
                self.target_scaler = pickle.loads(checkpoint['target_scaler'])
                logger.debug("Loaded target scaler")
            
            # Load CHF parameter initial values if available
            if 'chf_params' in checkpoint:
                self.chf_param_init = checkpoint['chf_params']
                logger.info(f"Loaded CHF parameters: {self.chf_param_init}")
            
            # Build model and load state
            self.model = self._build_model(self.input_size)
            self.model.load_state_dict(checkpoint['model_state'])
            logger.info("Model architecture rebuilt and state loaded")
            
            # Rebuild optimizer
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
            if checkpoint.get('optimizer_state'):
                self.optimizer.load_state_dict(checkpoint['optimizer_state'])
                logger.debug("Optimizer state loaded")
            
            # Update bowring_params references to the loaded parameters
            for name in self.chf_param_init.keys():
                param_name = f'chf_{name}'
                if hasattr(self.model, param_name):
                    self.bowring_params[name] = getattr(self.model, param_name)
            
            logger.info(f"✓ Successfully loaded PINN model from {path} (epoch {self.epoch})")
            
            return checkpoint.get('metadata', {})
            
        except Exception as e:
            logger.error(f"ERROR loading PINN model: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def get_feature_importance(self) -> pd.DataFrame:
        """Gradient-based feature importance (similar to NN)."""
        dummy_input = torch.ones(1, self.input_size, device=self.device, requires_grad=True)
        output = self.model(dummy_input)
        output.backward()
        gradients = torch.abs(dummy_input.grad).cpu().numpy().flatten()
        return pd.DataFrame({
            'feature': [f'feature_{i}' for i in range(len(gradients))],
            'importance': gradients
        }).sort_values('importance', ascending=False)
    
    def get_chf_parameters(self) -> Dict[str, float]:
        """Returns the learned CHF equation parameters."""
        return {name: param.item() for name, param in self.bowring_params.items()}
