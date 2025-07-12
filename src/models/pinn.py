"""
Physics-Informed Neural Network (PINN) Model Implementation

Combines data-driven loss with physics-based constraints for heat flux prediction.
"""
import torch # type: ignore
import torch.nn as nn # type: ignore
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple  # <-- Added Tuple import
from pathlib import Path
import pickle
from sklearn.preprocessing import StandardScaler

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
        
        # CHF Equation Parameters (learnable)
        self.chf_params = {
            'ln_A': nn.Parameter(torch.tensor(0.0)),  # ln(A) coefficient
            'B': nn.Parameter(torch.tensor(1.0)),     # Quality interaction coefficient
            'n1': nn.Parameter(torch.tensor(0.8)),    # mass_flux exponent
            'n2': nn.Parameter(torch.tensor(0.3)),    # pressure exponent  
            'n3': nn.Parameter(torch.tensor(0.5)),    # quality term exponent
            'n4': nn.Parameter(torch.tensor(-0.2)),   # length exponent (negative expected)
            'n5': nn.Parameter(torch.tensor(0.0)),    # tube geometry coefficient
            'n6': nn.Parameter(torch.tensor(0.0))     # annulus geometry coefficient
        }

    def _build_model(self, input_size: int) -> nn.Module:
        """Enhanced 4-layer MLP with better initialization and dropout."""
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
        
        # Move CHF parameters to device and register them
        for name, param in self.chf_params.items():
            param.data = param.data.to(self.device)
            model.register_parameter(f'chf_{name}', param)
        
        return model

    def _physics_loss(self, inputs: torch.Tensor, predictions: torch.Tensor) -> torch.Tensor:
        """
        CHF Physics-Informed Loss using the derived equation:
        ln(chf_exp) = ln(A) + n1*ln(mass_flux) + n2*ln(pressure) + n3*ln(1 + B*x_e_out) + n4*ln(length) + n5*geom_tube + n6*geom_annulus
        """
        # Extract physical parameters (adjust indices based on your column order)
        # Assuming column order: pressure, mass_flux, x_e_out, D_e, D_h, length, geom_tube, geom_annulus
        eps = 1e-8  # Small epsilon to prevent log(0)
        
        # Extract variables (assuming scaled inputs)
        pressure = torch.abs(inputs[:, 0]) + eps      # Ensure positive
        mass_flux = torch.abs(inputs[:, 1]) + eps     # Ensure positive  
        x_e_out = inputs[:, 2]                        # Can be negative
        length = torch.abs(inputs[:, 5]) + eps        # Ensure positive
        
        # Geometry variables (assuming one-hot encoded)
        geom_tube = inputs[:, 6] if inputs.shape[1] > 6 else torch.zeros_like(pressure)
        geom_annulus = inputs[:, 7] if inputs.shape[1] > 7 else torch.zeros_like(pressure)
        
        # Get CHF parameters
        ln_A = self.chf_params['ln_A']
        B = self.chf_params['B'] 
        n1 = self.chf_params['n1']
        n2 = self.chf_params['n2']
        n3 = self.chf_params['n3']
        n4 = self.chf_params['n4']
        n5 = self.chf_params['n5']
        n6 = self.chf_params['n6']
        
        # CHF equation: ln(chf_exp) = ln(A) + n1*ln(mass_flux) + n2*ln(pressure) + n3*ln(1 + B*x_e_out) + n4*ln(length) + n5*geom_tube + n6*geom_annulus
        ln_chf_physics = (ln_A + 
                         n1 * torch.log(mass_flux) + 
                         n2 * torch.log(pressure) + 
                         n3 * torch.log(torch.abs(1 + B * x_e_out) + eps) + 
                         n4 * torch.log(length) + 
                         n5 * geom_tube + 
                         n6 * geom_annulus)
        
        # Convert to actual CHF values
        chf_physics = torch.exp(ln_chf_physics)
        
        # Neural network prediction (convert from scaled space if needed)
        chf_pred = predictions.squeeze()
        
        # Convert predictions back to original scale for physics comparison
        if hasattr(self, 'target_scaler') and self.target_scaler is not None:
            # Convert to numpy for inverse transform, then back to torch
            chf_pred_np = chf_pred.detach().cpu().numpy().reshape(-1, 1)
            chf_pred_unscaled = self.target_scaler.inverse_transform(chf_pred_np).flatten()
            chf_pred_original = torch.tensor(chf_pred_unscaled, device=self.device)
        else:
            chf_pred_original = chf_pred
            
        # Physics constraint: Neural network prediction should follow CHF equation
        physics_loss = torch.mean((chf_pred_original - chf_physics) ** 2)
        
        # Additional physics constraints
        physics_losses = [physics_loss]
        
        # 1. Ensure CHF is positive
        negative_chf = torch.relu(-chf_pred_original)
        physics_losses.append(torch.mean(negative_chf ** 2))
        
        # 2. Parameter bounds (soft constraints)
        # Mass flux exponent should be positive (n1 > 0)
        n1_constraint = torch.relu(-n1) ** 2
        physics_losses.append(n1_constraint)
        
        # Pressure exponent should be positive (n2 > 0)  
        n2_constraint = torch.relu(-n2) ** 2
        physics_losses.append(n2_constraint)
        
        # Length exponent should be negative (n4 < 0)
        n4_constraint = torch.relu(n4) ** 2
        physics_losses.append(n4_constraint)
        
        # Combine all physics losses
        weights = [1.0, 0.1, 0.1, 0.1, 0.1]  # Main equation loss has highest weight
        total_physics_loss = sum(w * loss for w, loss in zip(weights, physics_losses))
        
        # Debug logging
        if self.debug and torch.rand(1).item() < 0.01:  # Log 1% of batches
            print(f"CHF Physics Loss: {physics_loss.item():.6f}")
            print(f"CHF Parameters: ln_A={ln_A.item():.3f}, B={B.item():.3f}, n1={n1.item():.3f}, n2={n2.item():.3f}, n3={n3.item():.3f}, n4={n4.item():.3f}")
            print(f"Total Physics Loss: {total_physics_loss.item():.6f}")
        
        # Safety check
        if not torch.isfinite(total_physics_loss).all():
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        return total_physics_loss

    def _prepare_data(self, data: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """Converts DataFrame to tensors and initializes model if needed."""
        target_col = [col for col in data.columns if 'chf_exp' in col][0]
        X = data.drop(target_col, axis=1).values
        y = data[target_col].values
        
        # Add scaling with better handling
        if not self.is_fitted:
            X = self.input_scaler.fit_transform(X)
            y = self.target_scaler.fit_transform(y.reshape(-1, 1)).flatten()
        else:
            X = self.input_scaler.transform(X)
            y = self.target_scaler.transform(y.reshape(-1, 1)).flatten()
        
        if self.model is None:
            self.input_size = X.shape[1]
            self.model = self._build_model(self.input_size)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
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
                print(f"Warning: Non-finite loss at epoch {self.epoch}, batch {batch_idx}")
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
        
        # Debug output
        if self.debug:
            print(f"Epoch {self.epoch}: Data Loss: {avg_data_loss:.6f}, "
                  f"Physics Loss: {avg_physics_loss:.6f}, "
                  f"Physics Weight: {current_physics_weight:.6f}")
        
        return metrics

    def validate(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """Validates model (physics loss not computed during validation)."""
        X_test, y_test = self._prepare_data(test_data)
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_test).squeeze()
            mse = torch.nn.functional.mse_loss(predictions, y_test)
            mae = torch.nn.functional.l1_loss(predictions, y_test)
        return {'loss': mse.item(), 'rmse': np.sqrt(mse.item()), 'mae': mae.item()}

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
        save_dict = {
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'epoch': self.epoch,
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'lambda_physics': self.lambda_physics,
            'is_fitted': self.is_fitted,
            'input_scaler': self.input_scaler,
            'target_scaler': self.target_scaler,
            'chf_params': self.chf_params  # Save CHF parameters
        }
        if metadata:
            save_dict.update(metadata)
        torch.save(save_dict, path)

    def load(self, path: Path):
        """Loads model state with proper handling of sklearn objects."""
        from sklearn.preprocessing import StandardScaler
        
        # Add safe globals for sklearn objects
        torch.serialization.add_safe_globals([StandardScaler])
        
        # Load with weights_only=False since we have sklearn objects
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.input_size = checkpoint['input_size']
        self.hidden_size = checkpoint['hidden_size']
        self.lambda_physics = checkpoint['lambda_physics']
        self.model = self._build_model(self.input_size)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.epoch = checkpoint['epoch']
        self.is_fitted = checkpoint['is_fitted']
        
        # Load scalers if they exist
        if 'input_scaler' in checkpoint:
            self.input_scaler = checkpoint['input_scaler']
        if 'target_scaler' in checkpoint:
            self.target_scaler = checkpoint['target_scaler']
        
        # Load CHF parameters if they exist
        if 'chf_params' in checkpoint:
            self.chf_params = checkpoint['chf_params']
            
        return checkpoint

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
        return {name: param.item() for name, param in self.chf_params.items()}