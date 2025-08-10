# src/models/__init__.py
"""Model implementations package with centralized registry."""

from .xgboost import Xgboost
from .lightgbm import Lightgbm
from .neural_network import NeuralNetwork
from .svm import Svm
from .pinn import Pinn
from .gpr import GaussianProcess

# Model registry for easy access
model_registry = {
    'xgboost': Xgboost,
    'lightgbm': Lightgbm,
    'neural_network': NeuralNetwork,
    'svm': Svm,
    'pinn': Pinn,
    'gpr' : GaussianProcess
}

# Export all models and registry
__all__ = [
    'Xgboost',
    'Lightgbm', 
    'NeuralNetwork',
    'Svm',
    'Pinn',
    'gpr',
    'model_registry'
]