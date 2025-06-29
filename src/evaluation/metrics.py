"""
Metrics utilities for model evaluation.

This module contains functions for calculating various evaluation metrics
and creating comprehensive reports.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error, explained_variance_score,
    max_error, median_absolute_error
)
from scipy import stats


def calculate_additional_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive set of regression metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary of metrics
    """
    # Basic metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Additional metrics
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mean_absolute_percentage_error(y_true, y_pred) * 100,  # as percentage
        'explained_variance': explained_variance_score(y_true, y_pred),
        'max_error': max_error(y_true, y_pred),
        'median_absolute_error': median_absolute_error(y_true, y_pred),
    }
    
    # Custom metrics
    errors = y_true - y_pred
    
    # Normalized metrics
    metrics['normalized_rmse'] = rmse / (y_true.max() - y_true.min())
    metrics['coefficient_of_variation'] = rmse / y_true.mean()
    
    # Error percentiles
    metrics['error_5th_percentile'] = np.percentile(np.abs(errors), 5)
    metrics['error_95th_percentile'] = np.percentile(np.abs(errors), 95)
    metrics['error_iqr'] = np.percentile(np.abs(errors), 75) - np.percentile(np.abs(errors), 25)
    
    # Bias metrics
    metrics['mean_error'] = np.mean(errors)
    metrics['bias'] = metrics['mean_error']
    metrics['error_std'] = np.std(errors)
    
    # Directional metrics
    metrics['underestimation_rate'] = np.mean(errors > 0)  # Rate of predictions being too low
    metrics['overestimation_rate'] = np.mean(errors < 0)   # Rate of predictions being too high
    
    # Outlier metrics
    z_scores = np.abs(stats.zscore(errors))
    metrics['outlier_rate_3sigma'] = np.mean(z_scores > 3)
    
    # Correlation metrics
    metrics['pearson_correlation'] = np.corrcoef(y_true, y_pred)[0, 1]
    metrics['spearman_correlation'] = stats.spearmanr(y_true, y_pred)[0]
    
    # Convert all to float for JSON serialization
    metrics = {k: float(v) for k, v in metrics.items()}
    
    return metrics


def calculate_prediction_intervals(y_true: np.ndarray, y_pred: np.ndarray, 
                                 confidence: float = 0.95) -> Dict[str, np.ndarray]:
    """
    Calculate prediction intervals based on residuals.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        confidence: Confidence level (default: 0.95)
        
    Returns:
        Dictionary with 'lower' and 'upper' bounds
    """
    residuals = y_true - y_pred
    
    # Calculate standard error of residuals
    n = len(residuals)
    dof = n - 2  # degrees of freedom (assuming simple linear model)
    residual_std = np.sqrt(np.sum(residuals**2) / dof)
    
    # Calculate t-value
    alpha = 1 - confidence
    t_value = stats.t.ppf(1 - alpha/2, dof)
    
    # Calculate intervals
    margin = t_value * residual_std * np.sqrt(1 + 1/n)
    
    intervals = {
        'lower': y_pred - margin,
        'upper': y_pred + margin,
        'margin': margin,
        'confidence': confidence
    }
    
    return intervals


def calculate_cross_validation_metrics(cv_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate cross-validation results.
    
    Args:
        cv_results: List of dictionaries with metrics from each fold
        
    Returns:
        Dictionary with aggregated metrics
    """
    # Convert to DataFrame for easier aggregation
    cv_df = pd.DataFrame(cv_results)
    
    # Calculate statistics for each metric
    aggregated = {}
    
    for metric in cv_df.columns:
        if cv_df[metric].dtype in ['float64', 'int64']:
            aggregated[f'{metric}_mean'] = cv_df[metric].mean()
            aggregated[f'{metric}_std'] = cv_df[metric].std()
            aggregated[f'{metric}_min'] = cv_df[metric].min()
            aggregated[f'{metric}_max'] = cv_df[metric].max()
            aggregated[f'{metric}_cv'] = cv_df[metric].std() / cv_df[metric].mean()  # Coefficient of variation
    
    aggregated['n_folds'] = len(cv_results)
    
    return aggregated


def create_metrics_report(all_results: Dict[str, Dict[str, Any]]) -> str:
    """
    Create a comprehensive metrics report in Markdown format.
    
    Args:
        all_results: Dictionary of model results
        
    Returns:
        Markdown formatted report string
    """
    report = []
    report.append("# Model Evaluation Report")
    report.append(f"\nGenerated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Executive Summary
    report.append("## Executive Summary\n")
    
    # Find best model for each metric
    best_models = {}
    all_metrics = set()
    
    for model_name, results in all_results.items():
        for metric in results['metrics']:
            all_metrics.add(metric)
            
    for metric in all_metrics:
        best_value = None
        best_model = None
        
        for model_name, results in all_results.items():
            if metric in results['metrics']:
                value = results['metrics'][metric]
                
                # Determine if higher or lower is better
                if metric in ['mse', 'rmse', 'mae', 'mape', 'max_error', 'median_absolute_error']:
                    if best_value is None or value < best_value:
                        best_value = value
                        best_model = model_name
                else:
                    if best_value is None or value > best_value:
                        best_value = value
                        best_model = model_name
        
        best_models[metric] = (best_model, best_value)
    
    # Summary table
    report.append("### Best Performing Models by Metric\n")
    report.append("| Metric | Best Model | Value |")
    report.append("|--------|------------|-------|")
    
    for metric, (model, value) in best_models.items():
        report.append(f"| {metric.replace('_', ' ').title()} | {model} | {value:.4f} |")
    
    # Detailed Results
    report.append("\n## Detailed Model Results\n")
    
    for model_name, results in all_results.items():
        report.append(f"### {model_name}\n")
        
        # Model info
        report.append(f"- **Weights Path**: {results.get('weights_path', 'N/A')}")
        report.append(f"- **Test Size**: {results.get('test_size', 'N/A')}")
        report.append(f"- **Evaluation Time**: {results.get('timestamp', 'N/A')}\n")
        
        # Metrics table
        report.append("#### Performance Metrics\n")
        report.append("| Metric | Value |")
        report.append("|--------|-------|")
        
        for metric, value in sorted(results['metrics'].items()):
            if isinstance(value, float):
                report.append(f"| {metric.replace('_', ' ').title()} | {value:.4f} |")
            else:
                report.append(f"| {metric.replace('_', ' ').title()} | {value} |")
        
        report.append("")
    
    # Model Comparison
    report.append("## Model Comparison\n")
    
    # Create comparison DataFrame
    comparison_data = []
    for model_name, results in all_results.items():
        row = {'Model': model_name}
        row.update(results['metrics'])
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Key metrics comparison
    key_metrics = ['rmse', 'mae', 'r2', 'mape']
    available_metrics = [m for m in key_metrics if m in comparison_df.columns]
    
    if available_metrics:
        report.append("### Key Metrics Comparison\n")
        subset_df = comparison_df[['Model'] + available_metrics]
        report.append(subset_df.to_markdown(index=False))
    
    # Statistical Analysis
    report.append("\n## Statistical Analysis\n")
    
    # Metrics correlation
    if len(all_results) > 1:
        numeric_cols = comparison_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = comparison_df[numeric_cols].corr()
            
            report.append("### Metrics Correlation\n")
            report.append("Strong correlations (|r| > 0.8) between metrics:\n")
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_value = corr_matrix.iloc[i, j]
                    if abs(corr_value) > 0.8:
                        metric1 = corr_matrix.columns[i]
                        metric2 = corr_matrix.columns[j]
                        report.append(f"- {metric1} vs {metric2}: r = {corr_value:.3f}")
    
    # Recommendations
    report.append("\n## Recommendations\n")
    
    # Find overall best model
    if 'rmse' in all_metrics:
        best_overall = best_models['rmse'][0]
        report.append(f"1. **Best Overall Model**: {best_overall} (based on RMSE)")
    elif 'mae' in all_metrics:
        best_overall = best_models['mae'][0]
        report.append(f"1. **Best Overall Model**: {best_overall} (based on MAE)")
    
    # Model-specific recommendations
    report.append("\n2. **Model-Specific Insights**:\n")
    
    for model_name, results in all_results.items():
        metrics = results['metrics']
        
        # Check for bias
        if 'bias' in metrics and abs(metrics['bias']) > 0.1:
            direction = "overestimates" if metrics['bias'] < 0 else "underestimates"
            report.append(f"   - {model_name} systematically {direction} (bias: {metrics['bias']:.3f})")
        
        # Check for high variance
        if 'error_std' in metrics and 'rmse' in metrics:
            if metrics['error_std'] > metrics['rmse'] * 0.9:
                report.append(f"   - {model_name} shows high prediction variance")
    
    return "\n".join(report)


def calculate_model_complexity(model: Any) -> Dict[str, Any]:
    """
    Calculate model complexity metrics.
    
    Args:
        model: Trained model object
        
    Returns:
        Dictionary with complexity metrics
    """
    complexity = {}
    
    # Try to get number of parameters
    if hasattr(model, 'get_params'):
        params = model.get_params()
        complexity['n_parameters'] = len(params)
        complexity['parameters'] = params
    
    # For tree-based models
    if hasattr(model, 'tree_'):
        complexity['n_nodes'] = model.tree_.node_count
        complexity['max_depth'] = model.tree_.max_depth
    
    # For ensemble models
    if hasattr(model, 'n_estimators'):
        complexity['n_estimators'] = model.n_estimators
    
    # For neural networks (PyTorch)
    if hasattr(model, 'parameters'):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        complexity['total_parameters'] = total_params
        complexity['trainable_parameters'] = trainable_params
    
    return complexity


def compare_models_statistical_significance(results1: Dict[str, Any], 
                                          results2: Dict[str, Any],
                                          metric: str = 'rmse') -> Dict[str, Any]:
    """
    Perform statistical significance test between two models.
    
    Args:
        results1: Results from first model
        results2: Results from second model
        metric: Metric to compare
        
    Returns:
        Dictionary with test results
    """
    # Get predictions and calculate errors
    y_true = results1['actual']
    errors1 = np.abs(y_true - results1['predictions'])
    errors2 = np.abs(y_true - results2['predictions'])
    
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(errors1, errors2)
    
    # Wilcoxon signed-rank test (non-parametric alternative)
    w_stat, w_p_value = stats.wilcoxon(errors1, errors2)
    
    # Effect size (Cohen's d)
    mean_diff = np.mean(errors1 - errors2)
    pooled_std = np.sqrt((np.std(errors1)**2 + np.std(errors2)**2) / 2)
    cohens_d = mean_diff / pooled_std
    
    return {
        'metric': metric,
        'model1': results1['model_name'],
        'model2': results2['model_name'],
        'model1_mean_error': np.mean(errors1),
        'model2_mean_error': np.mean(errors2),
        'difference': mean_diff,
        't_statistic': t_stat,
        'p_value': p_value,
        'wilcoxon_statistic': w_stat,
        'wilcoxon_p_value': w_p_value,
        'cohens_d': cohens_d,
        'significant_at_0.05': p_value < 0.05,
        'better_model': results1['model_name'] if mean_diff < 0 else results2['model_name']
    }