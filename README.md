# Critical Heat Flux (CHF) Prediction Project

A comprehensive machine learning framework for predicting Critical Heat Flux in aerospace materials using multiple models including XGBoost, LightGBM, Neural Networks, Support Vector Machines, and Physics-Informed Neural Networks (PINNs).

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Models](#models)
- [Usage](#usage)
- [Results](#results)
- [Development](#development)

## Overview

This project implements a complete ML pipeline for CHF prediction:
- **Data Pipeline**: Automated download, preprocessing, and splitting
- **Model Training**: Multiple ML models with hyperparameter tuning
- **Evaluation**: Comprehensive metrics and visualizations
- **Physics-Informed**: PINN implementation with configurable physics constraints

## Project Structure

```
CHF/
├── configs/
│   ├── model_configs.yaml      # Model hyperparameters
│   ├── physics_configs.yaml     # PINN physics equations
│   └── logging_config.yaml      # Logging configuration
├── data/
│   ├── raw/                     # Original dataset
│   └── processed/               # Train/test splits
├── logs/                        # Training and evaluation logs
├── reports/
│   ├── figures/                 # Evaluation plots
│   └── metrics/                 # Performance reports
├── scripts/
│   ├── download_and_organize.py # Data pipeline
│   ├── train.py                 # Model training
│   ├── test.py                  # Model testing
│   └── evaluate.py              # Generate reports
├── src/
│   ├── data.py                  # Data utilities
│   ├── plotting.py              # EDA plots
│   ├── models/                  # Model implementations
│   ├── evaluation/              # Evaluation utilities
│   └── utils/                   # Shared utilities
├── weights/                     # Saved model checkpoints
└── results/                     # Test predictions
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA (optional, for GPU acceleration)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd CHF
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure Kaggle API (for data download):
```bash
# Create ~/.kaggle/kaggle.json with your API credentials
mkdir -p ~/.kaggle
echo '{"username":"YOUR_USERNAME","key":"YOUR_API_KEY"}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

## Quick Start

Run the complete pipeline:

```bash
# 1. Download and prepare data
python scripts/download_and_organize.py

# 2. Train all models
python scripts/train.py

# 3. Test models
python scripts/test.py

# 4. Generate evaluation reports
python scripts/evaluate.py
```

Or use the convenience script:
```bash
bash run_pipeline.sh
```

## Configuration

### Model Configuration (`configs/model_configs.yaml`)

Configure model hyperparameters and training settings:

```yaml
xgboost:
  init_params:
    objective: "reg:squarederror"
    learning_rate: [0.01, 0.05, 0.1]  # List for hyperparameter tuning
    max_depth: [3, 5, 7]
  tuning:
    method: "random"  # or "grid"
    n_iter: 20
  epochs: 30
```

### Physics Configuration (`configs/physics_configs.yaml`)

Configure PINN physics constraints:

```yaml
pinn:
  equations:
    - name: "mass_flux_monotonicity"
      weight: 0.3
      description: "CHF increases with mass flux"
    - name: "pressure_monotonicity" 
      weight: 0.2
      description: "CHF increases with pressure"
```

### Logging Configuration (`configs/logging_config.yaml`)

Control logging behavior:

```yaml
logging:
  level: INFO  # DEBUG, INFO, WARNING, ERROR
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file_logging: true
  console_logging: true
```

## Models

### Available Models

1. **XGBoost** - Gradient boosting with automatic hyperparameter tuning
2. **LightGBM** - Fast gradient boosting
3. **Neural Network** - Deep learning with configurable architecture
4. **SVM** - Support Vector Machine with RBF kernel
5. **PINN** - Physics-Informed Neural Network with CHF physics

### Adding New Models

1. Create a new model class in `src/models/`
2. Implement required methods: `train_epoch()`, `validate()`, `predict()`
3. Add configuration to `configs/model_configs.yaml`

## Usage

### Training Individual Models

```bash
# Train specific model
python scripts/train.py --models xgboost lightgbm

# Train with custom config
python scripts/train.py --config configs/custom_config.yaml

# Debug mode
python scripts/train.py --debug
```

### Testing Models

```bash
# Test all models
python scripts/test.py

# Test specific models
python scripts/test.py --models xgboost pinn

# Use latest weights instead of best
python scripts/test.py --weights-type latest
```

### Evaluation and Visualization

```bash
# Generate all reports
python scripts/evaluate.py

# Only generate plots (no PDF)
python scripts/evaluate.py --format png

# Custom results directory
python scripts/evaluate.py --results-dir custom_results/
```

### Hyperparameter Tuning

Models support automatic hyperparameter tuning. Configure in `model_configs.yaml`:

```yaml
model_name:
  init_params:
    param1: [value1, value2, value3]  # List of values to try
  tuning:
    method: "random"  # or "grid"
    n_iter: 50        # For random search
    cv: 5             # Cross-validation folds
```

## Results

After running the pipeline, find:

- **Model Weights**: `weights/{model_name}/best_model.pth`
- **Predictions**: `results/{model_name}/predictions.csv`
- **Metrics**: `results/{model_name}/metrics.json`
- **Plots**: `reports/figures/`
- **Reports**: `reports/evaluation_report_*.pdf`

### Interpreting Results

Key metrics to consider:
- **RMSE**: Root Mean Square Error (lower is better)
- **MAE**: Mean Absolute Error (lower is better)
- **R²**: Coefficient of determination (higher is better)
- **MAPE**: Mean Absolute Percentage Error (lower is better)

## Development

### Code Style

- Follow PEP 8
- Use type hints
- Add docstrings to all functions
- Include file paths at the top of each file

### Testing

```bash
# Run unit tests
pytest tests/

# Run with coverage
pytest --cov=src tests/
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

## Advanced Features

### Physics-Informed Neural Networks

Configure custom physics equations in `configs/physics_configs.yaml`:

```yaml
custom_equation:
  expression: "dCHF_dP + alpha * CHF"
  weight: 0.5
  variables: ["pressure", "chf_pred"]
  parameters:
    alpha: 0.1
```

### Ensemble Methods

Combine predictions from multiple models:

```python
from src.utils.ensemble import create_ensemble

ensemble = create_ensemble(['xgboost', 'lightgbm', 'neural_network'])
predictions = ensemble.predict(test_data)
```

### Custom Metrics

Add custom evaluation metrics in `src/evaluation/metrics.py`:

```python
def custom_metric(y_true, y_pred):
    # Your implementation
    return metric_value
```

## Troubleshooting

### Common Issues

1. **Memory Error**: Reduce batch size or use data sampling
2. **Convergence Issues**: Adjust learning rate or use early stopping
3. **GPU Not Found**: Check CUDA installation and PyTorch version

### Debug Mode

Enable detailed logging:
```bash
python scripts/train.py --debug
```

Check logs in `logs/` directory for detailed information.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{chf_prediction,
  title = {Critical Heat Flux Prediction Framework},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/chf}
}
```

## License

This project is licensed under the MIT License - see LICENSE file for details.