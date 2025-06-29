"""
Plotting utilities for model evaluation.

This module contains functions for creating various evaluation plots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, Any, Tuple, Optional


def plot_predictions_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, 
                              model_name: str) -> plt.Figure:
    """
    Create a scatter plot of predictions vs actual values.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        model_name: Name of the model
        
    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create scatter plot
    scatter = ax.scatter(y_true, y_pred, alpha=0.6, s=50, c=y_true, cmap='viridis')
    
    # Add perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    # Add trend line
    z = np.polyfit(y_true, y_pred, 1)
    p = np.poly1d(z)
    ax.plot(y_true, p(y_true), 'g-', alpha=0.8, lw=2, label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
    
    # Calculate R²
    r2 = np.corrcoef(y_true, y_pred)[0, 1]**2
    
    # Styling
    ax.set_xlabel('Actual Values', fontsize=12)
    ax.set_ylabel('Predicted Values', fontsize=12)
    ax.set_title(f'{model_name} - Predictions vs Actual\nR² = {r2:.4f}', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    plt.colorbar(scatter, ax=ax, label='Actual Value')
    
    plt.tight_layout()
    return fig


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, 
                   model_name: str) -> plt.Figure:
    """
    Create residual plots for model diagnostics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        model_name: Name of the model
        
    Returns:
        matplotlib figure
    """
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Residuals vs Predicted
    ax = axes[0, 0]
    ax.scatter(y_pred, residuals, alpha=0.6, s=40)
    ax.axhline(y=0, color='r', linestyle='--', lw=2)
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Residuals')
    ax.set_title('Residuals vs Predicted')
    ax.grid(True, alpha=0.3)
    
    # Add moving average
    sorted_idx = np.argsort(y_pred)
    window = max(len(y_pred) // 20, 10)
    moving_avg = pd.Series(residuals[sorted_idx]).rolling(window=window, center=True).mean()
    ax.plot(y_pred[sorted_idx], moving_avg, 'g-', lw=2, label='Moving Average')
    ax.legend()
    
    # 2. Residuals vs Actual
    ax = axes[0, 1]
    ax.scatter(y_true, residuals, alpha=0.6, s=40, c='orange')
    ax.axhline(y=0, color='r', linestyle='--', lw=2)
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Residuals')
    ax.set_title('Residuals vs Actual')
    ax.grid(True, alpha=0.3)
    
    # 3. Histogram of Residuals
    ax = axes[1, 0]
    n, bins, patches = ax.hist(residuals, bins=30, density=True, alpha=0.7, color='blue', edgecolor='black')
    
    # Add normal distribution overlay
    mu, sigma = residuals.mean(), residuals.std()
    x = np.linspace(residuals.min(), residuals.max(), 100)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', lw=2, label=f'Normal(μ={mu:.2f}, σ={sigma:.2f})')
    ax.set_xlabel('Residuals')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of Residuals')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Scale-Location Plot
    ax = axes[1, 1]
    standardized_residuals = residuals / residuals.std()
    ax.scatter(y_pred, np.sqrt(np.abs(standardized_residuals)), alpha=0.6, s=40, c='green')
    
    # Add smoothed line
    sorted_idx = np.argsort(y_pred)
    smoothed = pd.Series(np.sqrt(np.abs(standardized_residuals[sorted_idx]))).rolling(window=window, center=True).mean()
    ax.plot(y_pred[sorted_idx], smoothed, 'r-', lw=2)
    
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('√|Standardized Residuals|')
    ax.set_title('Scale-Location Plot')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'{model_name} - Residual Analysis', fontsize=16)
    plt.tight_layout()
    return fig


def plot_qq_plot(y_true: np.ndarray, y_pred: np.ndarray, 
                 model_name: str) -> plt.Figure:
    """
    Create Q-Q plot to check normality of residuals.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        model_name: Name of the model
        
    Returns:
        matplotlib figure
    """
    residuals = y_true - y_pred
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Calculate theoretical quantiles
    stats.probplot(residuals, dist="norm", plot=ax)
    
    # Customize
    ax.set_title(f'{model_name} - Q-Q Plot of Residuals', fontsize=14)
    ax.set_xlabel('Theoretical Quantiles')
    ax.set_ylabel('Sample Quantiles')
    ax.grid(True, alpha=0.3)
    
    # Add R² value for the Q-Q plot
    (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm", plot=None)
    ax.text(0.05, 0.95, f'R² = {r**2:.4f}', transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return fig


def plot_feature_importance(importance_df: pd.DataFrame, model_name: str, 
                           top_n: int = 20) -> plt.Figure:
    """
    Plot feature importance.
    
    Args:
        importance_df: DataFrame with 'feature' and 'importance' columns
        model_name: Name of the model
        top_n: Number of top features to show
        
    Returns:
        matplotlib figure
    """
    # Select top features
    top_features = importance_df.nlargest(top_n, 'importance')
    
    fig, ax = plt.subplots(figsize=(10, max(8, top_n * 0.4)))
    
    # Create horizontal bar plot
    bars = ax.barh(top_features['feature'], top_features['importance'])
    
    # Color bars based on importance
    colors = plt.cm.viridis(top_features['importance'] / top_features['importance'].max())
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_ylabel('Features', fontsize=12)
    ax.set_title(f'{model_name} - Top {top_n} Feature Importance', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (feature, importance) in enumerate(zip(top_features['feature'], top_features['importance'])):
        ax.text(importance, i, f' {importance:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    return fig


def plot_model_comparison(comparison_df: pd.DataFrame) -> plt.Figure:
    """
    Create comparison plots for multiple models.
    
    Args:
        comparison_df: DataFrame with model names and metrics
        
    Returns:
        matplotlib figure
    """
    # Select numeric columns (metrics)
    metric_columns = [col for col in comparison_df.columns if col != 'model' and comparison_df[col].dtype in ['float64', 'int64']]
    
    n_metrics = len(metric_columns)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    
    if n_metrics == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes
    else:
        axes = axes.ravel()
    
    for idx, metric in enumerate(metric_columns):
        ax = axes[idx]
        
        # Create bar plot
        data = comparison_df[['model', metric]].dropna()
        bars = ax.bar(data['model'], data[metric])
        
        # Color bars
        colors = plt.cm.Set3(np.linspace(0, 1, len(bars)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Model', fontsize=10)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=10)
        ax.set_title(f'{metric.replace("_", " ").title()}', fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
    
    # Hide empty subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Model Performance Comparison', fontsize=16)
    plt.tight_layout()
    return fig


def plot_error_distribution(y_true: np.ndarray, y_pred: np.ndarray, 
                           model_name: str) -> plt.Figure:
    """
    Plot error distribution with statistics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        model_name: Name of the model
        
    Returns:
        matplotlib figure
    """
    errors = y_true - y_pred
    abs_errors = np.abs(errors)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Error Distribution
    ax = axes[0, 0]
    ax.hist(errors, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
    
    # Add KDE
    kde = stats.gaussian_kde(errors)
    x_range = np.linspace(errors.min(), errors.max(), 100)
    ax.plot(x_range, kde(x_range), 'r-', lw=2, label='KDE')
    
    # Add statistics
    ax.axvline(errors.mean(), color='green', linestyle='--', lw=2, label=f'Mean: {errors.mean():.3f}')
    ax.axvline(np.median(errors), color='orange', linestyle='--', lw=2, label=f'Median: {np.median(errors):.3f}')
    
    ax.set_xlabel('Prediction Error')
    ax.set_ylabel('Density')
    ax.set_title('Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Absolute Error Distribution
    ax = axes[0, 1]
    ax.hist(abs_errors, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax.axvline(abs_errors.mean(), color='red', linestyle='--', lw=2, label=f'MAE: {abs_errors.mean():.3f}')
    ax.set_xlabel('Absolute Error')
    ax.set_ylabel('Frequency')
    ax.set_title('Absolute Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Error Percentiles
    ax = axes[1, 0]
    percentiles = np.arange(0, 101, 5)
    error_percentiles = np.percentile(abs_errors, percentiles)
    ax.plot(percentiles, error_percentiles, 'b-', lw=2, marker='o')
    ax.fill_between(percentiles, 0, error_percentiles, alpha=0.3)
    ax.set_xlabel('Percentile')
    ax.set_ylabel('Absolute Error')
    ax.set_title('Error Percentiles')
    ax.grid(True, alpha=0.3)
    
    # Add key percentiles
    for p in [50, 90, 95]:
        val = np.percentile(abs_errors, p)
        ax.annotate(f'{p}%: {val:.3f}', xy=(p, val), xytext=(p+5, val+0.1),
                   arrowprops=dict(arrowstyle='->', color='red'))
    
    # 4. Box Plot by Error Magnitude
    ax = axes[1, 1]
    
    # Create bins for actual values
    n_bins = 5
    bins = pd.qcut(y_true, q=n_bins, labels=[f'Q{i+1}' for i in range(n_bins)])
    
    # Create box plot
    error_by_bin = pd.DataFrame({'bin': bins, 'error': errors})
    error_by_bin.boxplot(column='error', by='bin', ax=ax)
    ax.set_xlabel('Actual Value Quintile')
    ax.set_ylabel('Prediction Error')
    ax.set_title('Error Distribution by Actual Value')
    plt.sca(ax)
    plt.xticks(rotation=0)
    
    plt.suptitle(f'{model_name} - Error Analysis', fontsize=16)
    plt.tight_layout()
    return fig


def plot_prediction_intervals(y_true: np.ndarray, y_pred: np.ndarray,
                            intervals: Dict[str, np.ndarray], 
                            model_name: str) -> plt.Figure:
    """
    Plot predictions with confidence/prediction intervals.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        intervals: Dictionary with 'lower' and 'upper' bounds
        model_name: Name of the model
        
    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Sort by predicted values for better visualization
    sorted_idx = np.argsort(y_pred)
    
    # Plot intervals
    ax.fill_between(range(len(y_pred)), 
                   intervals['lower'][sorted_idx], 
                   intervals['upper'][sorted_idx],
                   alpha=0.3, color='blue', label='95% Prediction Interval')
    
    # Plot predictions and actual
    ax.scatter(range(len(y_pred)), y_true[sorted_idx], alpha=0.6, s=30, 
              color='green', label='Actual', zorder=3)
    ax.plot(range(len(y_pred)), y_pred[sorted_idx], 'r-', lw=2, 
           label='Predicted', zorder=2)
    
    # Calculate coverage
    coverage = np.mean((y_true >= intervals['lower']) & (y_true <= intervals['upper']))
    
    ax.set_xlabel('Sample Index (sorted by prediction)')
    ax.set_ylabel('Value')
    ax.set_title(f'{model_name} - Prediction Intervals\nCoverage: {coverage:.1%}', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_learning_curves(train_sizes: np.ndarray, train_scores: np.ndarray,
                        val_scores: np.ndarray, model_name: str) -> plt.Figure:
    """
    Plot learning curves showing model performance vs training size.
    
    Args:
        train_sizes: Array of training set sizes
        train_scores: Training scores for each size
        val_scores: Validation scores for each size
        model_name: Name of the model
        
    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Calculate mean and std
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    # Plot with confidence intervals
    ax.plot(train_sizes, train_mean, 'b-', lw=2, label='Training Score')
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                   alpha=0.3, color='blue')
    
    ax.plot(train_sizes, val_mean, 'r-', lw=2, label='Validation Score')
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                   alpha=0.3, color='red')
    
    ax.set_xlabel('Training Set Size')
    ax.set_ylabel('Score')
    ax.set_title(f'{model_name} - Learning Curves', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig