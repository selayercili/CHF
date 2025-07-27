#!/usr/bin/env python3
# scripts/evaluate.py
"""
Model Evaluation and Visualization Script with SMOTE Comparison Support

This script generates comprehensive evaluation plots and reports
for all tested models, including comparisons between SMOTE and regular data.

Usage:
    python scripts/evaluate.py [--data-type DATA_TYPE] [--comparison-mode] [--format FORMAT]
"""

import os
import sys
import argparse
import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import warnings
from scipy import stats

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Imports after path setup
from src.utils import setup_logging, get_logger
from src.evaluation import (
    plot_predictions_vs_actual,
    plot_residuals,
    plot_qq_plot,
    plot_feature_importance,
    plot_model_comparison,
    plot_error_distribution,
    plot_prediction_intervals,
    calculate_prediction_intervals,
    create_metrics_report
)

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)


class ModelEvaluator:
    """Handles comprehensive model evaluation and visualization."""
    
    def __init__(self, debug: bool = False, data_type: str = 'both', comparison_mode: bool = False):
        """
        Initialize the ModelEvaluator.
        
        Args:
            debug: Enable debug logging
            data_type: Data type to evaluate ('smote', 'regular', or 'both')
            comparison_mode: Enable SMOTE vs Regular comparison mode
        """
        # Setup logging
        log_level = 'DEBUG' if debug else 'INFO'
        setup_logging(log_level=log_level)
        self.logger = get_logger(__name__)
        
        # Store settings
        self.data_type = data_type
        self.comparison_mode = comparison_mode
        
        # Setup directories based on data type
        self.setup_directories()
        
        # Set plotting style
        self.setup_plotting_style()
        
        self.logger.info("="*60)
        self.logger.info("Model Evaluator Initialized")
        self.logger.info(f"Data type: {data_type}")
        self.logger.info(f"Comparison mode: {comparison_mode}")
        self.logger.info("="*60)
    
    def setup_directories(self):
        """Setup directories based on data type and mode."""
        if self.comparison_mode or self.data_type == 'both':
            self.results_dirs = {
                'smote': Path("results_smote"),
                'regular': Path("results_regular")
            }
            self.figures_dir = Path("reports/comparison/figures")
            self.reports_dir = Path("reports/comparison")
        elif self.data_type == 'smote':
            self.results_dirs = {'smote': Path("results_smote")}
            self.figures_dir = Path("reports_smote/figures")
            self.reports_dir = Path("reports_smote")
        else:  # regular
            self.results_dirs = {'regular': Path("results_regular")}
            self.figures_dir = Path("reports_regular/figures")
            self.reports_dir = Path("reports_regular")
        
        # Create directories
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(exist_ok=True)
    
    def setup_plotting_style(self):
        """Setup matplotlib plotting style."""
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # Update matplotlib parameters
        plt.rcParams.update({
            'figure.figsize': (10, 8),
            'font.size': 12,
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'xtick.labelsize': 11,
            'ytick.labelsize': 11,
            'legend.fontsize': 11,
            'figure.dpi': 100,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight'
        })
    
    def load_model_results(self, model_name: str, results_dir: Path) -> Optional[Dict[str, Any]]:
        """Load results for a specific model."""
        model_results_dir = results_dir / model_name
        
        # Check if directory exists
        if not model_results_dir.exists():
            self.logger.warning(f"No results directory found for {model_name} in {results_dir}")
            return None
        
        # Load full results pickle
        results_path = model_results_dir / 'full_results.pkl'
        if not results_path.exists():
            self.logger.warning(f"No results pickle found for {model_name} in {results_dir}")
            return None
        
        try:
            with open(results_path, 'rb') as f:
                results = pickle.load(f)
            
            self.logger.debug(f"Loaded results for {model_name} from {results_dir}")
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to load results for {model_name}: {str(e)}")
            return None
    
    def load_all_results(self) -> Dict[str, Dict[str, Any]]:
        """Load results for all models from all data types."""
        all_results = {}
        
        for data_type, results_dir in self.results_dirs.items():
            if not results_dir.exists():
                self.logger.warning(f"Results directory not found: {results_dir}")
                continue
                
            # Find all model directories
            for model_dir in results_dir.iterdir():
                if model_dir.is_dir():
                    model_name = model_dir.name
                    results = self.load_model_results(model_name, results_dir)
                    if results:
                        # Key includes both model name and data type
                        key = f"{model_name}_{data_type}"
                        all_results[key] = results
        
        self.logger.info(f"Loaded results for {len(all_results)} model-data combinations")
        return all_results
    
    def evaluate_single_model(self, model_key: str, results: Dict[str, Any]) -> None:
        """
        Generate evaluation plots for a single model.
        
        Args:
            model_key: Key identifying model and data type (e.g., "xgboost_smote")
            results: Model results dictionary
        """
        self.logger.info(f"\nEvaluating {model_key}...")
        
        # Parse model name and data type
        parts = model_key.rsplit('_', 1)
        model_name = parts[0]
        data_type = parts[1] if len(parts) > 1 else ''
        
        # Create model-specific directory
        model_figures_dir = self.figures_dir / model_key
        model_figures_dir.mkdir(exist_ok=True)
        
        # Extract data
        y_true = results['actual']
        y_pred = results['predictions']
        
        # Create plots
        plot_functions = [
            ('predictions_vs_actual', plot_predictions_vs_actual),
            ('residuals', plot_residuals),
            ('qq_plot', plot_qq_plot),
            ('error_distribution', plot_error_distribution)
        ]
        
        title_suffix = f" ({data_type.upper()} data)" if data_type else ""
        
        for plot_name, plot_func in plot_functions:
            try:
                fig = plot_func(y_true, y_pred, model_name + title_suffix)
                fig.savefig(model_figures_dir / f'{plot_name}.png')
                plt.close(fig)
                self.logger.debug(f"Created {plot_name} plot for {model_key}")
            except Exception as e:
                self.logger.error(f"Failed to create {plot_name} plot: {str(e)}")
        
        # Feature importance plot
        if 'feature_importance' in results and results['feature_importance'] is not None:
            try:
                fig = plot_feature_importance(results['feature_importance'], model_name + title_suffix)
                fig.savefig(model_figures_dir / 'feature_importance.png')
                plt.close(fig)
            except Exception as e:
                self.logger.error(f"Failed to create feature importance plot: {str(e)}")
        
        # Prediction intervals
        try:
            intervals = calculate_prediction_intervals(y_true, y_pred)
            fig = plot_prediction_intervals(y_true, y_pred, intervals, model_name + title_suffix)
            fig.savefig(model_figures_dir / 'prediction_intervals.png')
            plt.close(fig)
        except Exception as e:
            self.logger.error(f"Failed to create prediction intervals plot: {str(e)}")
        
        self.logger.info(f"✓ Generated plots for {model_key}")
    
    def create_comparison_plots(self, all_results: Dict[str, Dict[str, Any]]) -> None:
        """Create plots comparing all models."""
        self.logger.info("\nCreating model comparison plots...")
        
        if self.comparison_mode:
            # Create SMOTE vs Regular comparison plots
            self.create_smote_comparison_plots(all_results)
        
        # Create within-data-type comparisons
        for data_type in self.results_dirs.keys():
            data_type_results = {k: v for k, v in all_results.items() if k.endswith(f"_{data_type}")}
            
            if len(data_type_results) > 1:
                self.create_data_type_comparison_plots(data_type_results, data_type)
    
    def create_smote_comparison_plots(self, all_results: Dict[str, Dict[str, Any]]) -> None:
        """Create plots specifically comparing SMOTE vs Regular training."""
        self.logger.info("Creating SMOTE vs Regular comparison plots...")
        
        # Extract model names
        model_names = set()
        for key in all_results.keys():
            model_name = key.rsplit('_', 1)[0]
            model_names.add(model_name)
        
        # 1. Performance improvement chart
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        improvement_data = []
        for model_name in model_names:
            smote_key = f"{model_name}_smote"
            regular_key = f"{model_name}_regular"
            
            if smote_key in all_results and regular_key in all_results:
                smote_metrics = all_results[smote_key]['metrics']
                regular_metrics = all_results[regular_key]['metrics']
                
                # Calculate improvements
                rmse_imp = (regular_metrics['rmse'] - smote_metrics['rmse']) / regular_metrics['rmse'] * 100
                r2_imp = (smote_metrics['r2'] - regular_metrics['r2']) / abs(regular_metrics['r2']) * 100
                mae_imp = (regular_metrics['mae'] - smote_metrics['mae']) / regular_metrics['mae'] * 100
                
                improvement_data.append({
                    'model': model_name,
                    'RMSE': rmse_imp,
                    'R²': r2_imp,
                    'MAE': mae_imp
                })
        
        if improvement_data:
            imp_df = pd.DataFrame(improvement_data)
            imp_df.set_index('model').plot(kind='bar', ax=axes[0])
            axes[0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            axes[0].set_title('Performance Improvement with SMOTE (%)', fontsize=14)
            axes[0].set_ylabel('Improvement (%)')
            axes[0].legend(title='Metric')
            axes[0].grid(True, alpha=0.3)
        
        # 2. Side-by-side metric comparison
        comparison_data = []
        for key, results in all_results.items():
            parts = key.rsplit('_', 1)
            comparison_data.append({
                'Model': parts[0],
                'Data Type': parts[1].upper(),
                'RMSE': results['metrics']['rmse'],
                'MAE': results['metrics']['mae'],
                'R²': results['metrics']['r2']
            })
        
        if comparison_data:
            comp_df = pd.DataFrame(comparison_data)
            
            # Plot RMSE comparison
            pivot_rmse = comp_df.pivot(index='Model', columns='Data Type', values='RMSE')
            pivot_rmse.plot(kind='bar', ax=axes[1])
            axes[1].set_title('RMSE Comparison: SMOTE vs Regular', fontsize=14)
            axes[1].set_ylabel('RMSE')
            axes[1].legend(title='Training Data')
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig.savefig(self.figures_dir / 'smote_vs_regular_comparison.png')
        plt.close(fig)
        
        # 3. Create heatmap of all metrics
        self.create_metrics_heatmap(all_results)
    
    def create_metrics_heatmap(self, all_results: Dict[str, Dict[str, Any]]) -> None:
        """Create a heatmap showing all metrics for all model-data combinations."""
        metrics_data = []
        
        for key, results in all_results.items():
            parts = key.rsplit('_', 1)
            model_name = parts[0]
            data_type = parts[1].upper()
            
            row_data = {
                'Model': f"{model_name} ({data_type})",
                'RMSE': results['metrics']['rmse'],
                'MAE': results['metrics']['mae'],
                'R²': results['metrics']['r2'],
                'Max Error': results['metrics'].get('max_error', np.nan)
            }
            metrics_data.append(row_data)
        
        if not metrics_data:
            return
        
        # Create DataFrame and normalize
        metrics_df = pd.DataFrame(metrics_data).set_index('Model')
        
        # Normalize metrics to 0-1 scale for heatmap
        normalized_df = metrics_df.copy()
        for col in metrics_df.columns:
            if col == 'R²':  # Higher is better
                normalized_df[col] = (metrics_df[col] - metrics_df[col].min()) / (metrics_df[col].max() - metrics_df[col].min())
            else:  # Lower is better
                normalized_df[col] = 1 - (metrics_df[col] - metrics_df[col].min()) / (metrics_df[col].max() - metrics_df[col].min())
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(normalized_df, annot=metrics_df, fmt='.4f', cmap='RdYlGn', 
                    cbar_kws={'label': 'Performance (1.0 = Best)'}, ax=ax)
        ax.set_title('Model Performance Heatmap\n(Normalized: 1.0 = Best Performance)', fontsize=14)
        plt.tight_layout()
        fig.savefig(self.figures_dir / 'performance_heatmap.png')
        plt.close(fig)
    
    def create_data_type_comparison_plots(self, results: Dict[str, Dict[str, Any]], data_type: str) -> None:
        """Create comparison plots for models within a data type."""
        # Prepare comparison data
        comparison_data = []
        for model_key, model_results in results.items():
            model_name = model_key.rsplit('_', 1)[0]
            metrics = model_results['metrics'].copy()
            metrics['model'] = model_name
            comparison_data.append(metrics)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Create comparison plot
        try:
            fig = plot_model_comparison(comparison_df)
            fig.suptitle(f'Model Performance Comparison - {data_type.upper()} Data', fontsize=16)
            fig.savefig(self.figures_dir / f'model_comparison_{data_type}.png')
            plt.close(fig)
        except Exception as e:
            self.logger.error(f"Failed to create model comparison plot: {str(e)}")
        
        # Create combined scatter plot
        self.create_multi_model_scatter(results, data_type)
        
        # Create error distribution comparison
        self.create_error_distribution_comparison(results, data_type)
    
    def create_multi_model_scatter(self, results: Dict[str, Dict[str, Any]], data_type: str) -> None:
        """Create scatter plots comparing predictions across models."""
        n_models = len(results)
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        
        if n_models == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        for idx, (model_key, model_results) in enumerate(results.items()):
            if idx >= len(axes):
                break
            
            ax = axes[idx]
            model_name = model_key.rsplit('_', 1)[0]
            y_true = model_results['actual']
            y_pred = model_results['predictions']
            
            # Scatter plot
            ax.scatter(y_true, y_pred, alpha=0.5, s=30)
            
            # Perfect prediction line
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
            
            # Labels and title
            ax.set_xlabel('Actual Values')
            ax.set_ylabel('Predicted Values')
            ax.set_title(f'{model_name}')
            
            # Add metrics
            r2 = model_results['metrics'].get('r2', 'N/A')
            rmse = model_results['metrics'].get('rmse', 'N/A')
            text = f'R² = {r2:.3f}\nRMSE = {rmse:.3f}' if isinstance(r2, float) else 'Metrics N/A'
            ax.text(0.05, 0.95, text, transform=ax.transAxes, 
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Hide unused subplots
        for idx in range(len(results), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(f'Model Predictions Comparison - {data_type.upper()} Data', fontsize=16)
        plt.tight_layout()
        
        fig.savefig(self.figures_dir / f'predictions_comparison_{data_type}.png')
        plt.close(fig)
    
    def create_error_distribution_comparison(self, results: Dict[str, Dict[str, Any]], data_type: str) -> None:
        """Create error distribution comparison plot."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for model_key, model_results in results.items():
            model_name = model_key.rsplit('_', 1)[0]
            y_true = model_results['actual']
            y_pred = model_results['predictions']
            errors = y_true - y_pred
            
            # Plot KDE
            kde = stats.gaussian_kde(errors)
            x_range = np.linspace(errors.min(), errors.max(), 100)
            ax.plot(x_range, kde(x_range), label=model_name, lw=2)
        
        ax.set_xlabel('Prediction Error')
        ax.set_ylabel('Density')
        ax.set_title(f'Error Distribution Comparison - {data_type.upper()} Data')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        fig.savefig(self.figures_dir / f'error_distribution_comparison_{data_type}.png')
        plt.close(fig)
    
    def create_report(self, all_results: Dict[str, Dict[str, Any]], 
                     output_format: str = 'all') -> None:
        """Create comprehensive evaluation report."""
        self.logger.info("\nCreating evaluation report...")
        
        # Create markdown report
        if output_format in ['markdown', 'all']:
            self.create_markdown_report(all_results)
        
        # Create PDF report
        if output_format in ['pdf', 'all']:
            self.create_pdf_report(all_results)
        
        # Create detailed metrics report
        metrics_report = create_metrics_report(all_results)
        report_path = self.reports_dir / 'detailed_metrics.md'
        with open(report_path, 'w') as f:
            f.write(metrics_report)
        
        self.logger.info(f"✓ Created metrics report: {report_path}")
    
    def create_markdown_report(self, all_results: Dict[str, Dict[str, Any]]) -> None:
        """Create markdown evaluation report with SMOTE comparison."""
        report_path = self.reports_dir / 'evaluation_report.md'
        
        with open(report_path, 'w') as f:
            # Header
            f.write("# CHF Model Evaluation Report")
            if self.comparison_mode:
                f.write(" - SMOTE vs Regular Comparison")
            f.write("\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            
            # Find best models overall
            best_rmse = min(all_results.items(), 
                          key=lambda x: x[1]['metrics'].get('rmse', float('inf')))
            best_r2 = max(all_results.items(), 
                        key=lambda x: x[1]['metrics'].get('r2', -float('inf')))
            
            f.write(f"- **Best RMSE Overall**: {best_rmse[0]} ({best_rmse[1]['metrics']['rmse']:.6f})\n")
            f.write(f"- **Best R² Overall**: {best_r2[0]} ({best_r2[1]['metrics']['r2']:.6f})\n")
            f.write(f"- **Total Models Evaluated**: {len(all_results)}\n")
            
            if self.comparison_mode:
                # Find models that benefit most from SMOTE
                smote_improvements = []
                for model_name in set(k.rsplit('_', 1)[0] for k in all_results.keys()):
                    smote_key = f"{model_name}_smote"
                    regular_key = f"{model_name}_regular"
                    
                    if smote_key in all_results and regular_key in all_results:
                        smote_rmse = all_results[smote_key]['metrics']['rmse']
                        regular_rmse = all_results[regular_key]['metrics']['rmse']
                        improvement = (regular_rmse - smote_rmse) / regular_rmse * 100
                        smote_improvements.append((model_name, improvement))
                
                smote_improvements.sort(key=lambda x: x[1], reverse=True)
                
                f.write("\n### SMOTE Impact Summary\n\n")
                f.write("Models ranked by RMSE improvement with SMOTE:\n\n")
                for model, imp in smote_improvements:
                    f.write(f"- **{model}**: {imp:+.1f}% improvement\n")
            
            # Model Performance Details
            f.write("\n## Model Performance Details\n\n")
            
            # Group by data type
            for data_type in ['regular', 'smote']:
                data_type_results = {k: v for k, v in all_results.items() if k.endswith(f"_{data_type}")}
                
                if data_type_results:
                    f.write(f"\n### {data_type.upper()} Data Models\n\n")
                    
                    for model_key, results in sorted(data_type_results.items()):
                        model_name = model_key.rsplit('_', 1)[0]
                        f.write(f"#### {model_name}\n\n")
                        
                        # Metrics table
                        f.write("| Metric | Value |\n")
                        f.write("|--------|-------|\n")
                        
                        metrics = results['metrics']
                        for metric_name, value in sorted(metrics.items()):
                            if isinstance(value, float):
                                f.write(f"| {metric_name} | {value:.6f} |\n")
                        
                        f.write("\n")
            
            # Direct SMOTE vs Regular Comparison
            if self.comparison_mode:
                f.write("\n## Direct SMOTE vs Regular Comparison\n\n")
                
                for model_name in set(k.rsplit('_', 1)[0] for k in all_results.keys()):
                    smote_key = f"{model_name}_smote"
                    regular_key = f"{model_name}_regular"
                    
                    if smote_key in all_results and regular_key in all_results:
                        f.write(f"### {model_name}\n\n")
                        
                        smote_metrics = all_results[smote_key]['metrics']
                        regular_metrics = all_results[regular_key]['metrics']
                        
                        # Comparison table
                        f.write("| Metric | Regular | SMOTE | Improvement |\n")
                        f.write("|--------|---------|-------|-------------|\n")
                        
                        for metric in ['rmse', 'mae', 'r2', 'max_error']:
                            if metric in smote_metrics and metric in regular_metrics:
                                reg_val = regular_metrics[metric]
                                smote_val = smote_metrics[metric]
                                
                                if metric == 'r2':  # Higher is better
                                    imp = (smote_val - reg_val) / abs(reg_val) * 100 if reg_val != 0 else 0
                                else:  # Lower is better
                                    imp = (reg_val - smote_val) / reg_val * 100 if reg_val != 0 else 0
                                
                                f.write(f"| {metric.upper()} | {reg_val:.6f} | {smote_val:.6f} | {imp:+.1f}% |\n")
                        
                        f.write("\n")
            
            # Recommendations
            f.write("\n## Recommendations\n\n")
            
            if self.comparison_mode:
                # Recommend based on SMOTE comparison
                if smote_improvements:
                    best_improvement = smote_improvements[0]
                    if best_improvement[1] > 0:
                        f.write(f"1. **Use SMOTE for {best_improvement[0]}**: Shows {best_improvement[1]:.1f}% improvement\n")
                    
                    # Models that don't benefit from SMOTE
                    no_benefit = [m for m, imp in smote_improvements if imp <= 0]
                    if no_benefit:
                        f.write(f"2. **Use regular data for**: {', '.join(no_benefit)} (no improvement with SMOTE)\n")
            
            # Overall best model recommendation
            f.write(f"\n3. **Best Overall Model**: {best_rmse[0].replace('_', ' ').title()} ")
            f.write(f"(RMSE: {best_rmse[1]['metrics']['rmse']:.6f})")
        
        self.logger.info(f"✓ Created markdown report: {report_path}")
    
    def create_pdf_report(self, all_results: Dict[str, Dict[str, Any]]) -> None:
        """Create PDF evaluation report."""
        pdf_path = self.reports_dir / f'evaluation_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
        
        with PdfPages(pdf_path) as pdf:
            # Title page
            fig = plt.figure(figsize=(8.5, 11))
            fig.text(0.5, 0.7, 'CHF Model Evaluation Report', 
                    ha='center', va='center', fontsize=24, weight='bold')
            
            if self.comparison_mode:
                fig.text(0.5, 0.65, 'SMOTE vs Regular Data Comparison',
                        ha='center', va='center', fontsize=18)
            
            fig.text(0.5, 0.55, f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
                    ha='center', va='center', fontsize=12)
            fig.text(0.5, 0.5, f'Number of models evaluated: {len(all_results)}',
                    ha='center', va='center', fontsize=12)
            
            # Add summary statistics
            best_model = min(all_results.items(), 
                           key=lambda x: x[1]['metrics'].get('rmse', float('inf')))
            fig.text(0.5, 0.4, f'Best Model: {best_model[0]}',
                    ha='center', va='center', fontsize=14, weight='bold')
            fig.text(0.5, 0.35, f'RMSE: {best_model[1]["metrics"]["rmse"]:.6f}',
                    ha='center', va='center', fontsize=12)
            
            plt.axis('off')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Add all plots
            for plot_file in sorted(self.figures_dir.rglob('*.png')):
                try:
                    img = plt.imread(plot_file)
                    fig, ax = plt.subplots(figsize=(8.5, 11))
                    ax.imshow(img)
                    ax.axis('off')
                    
                    # Add title based on file path
                    title = plot_file.stem.replace('_', ' ').title()
                    if plot_file.parent != self.figures_dir:
                        parent_name = plot_file.parent.name
                        if '_' in parent_name:
                            model, data_type = parent_name.rsplit('_', 1)
                            title = f"{model} ({data_type.upper()}) - {title}"
                        else:
                            title = f"{parent_name} - {title}"
                    
                    ax.set_title(title, fontsize=14, pad=20)
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close()
                except Exception as e:
                    self.logger.warning(f"Failed to add {plot_file} to PDF: {str(e)}")
        
        self.logger.info(f"✓ Created PDF report: {pdf_path}")
    
    def run_evaluation(self, output_format: str = 'all') -> None:
        """
        Run complete evaluation pipeline.
        
        Args:
            output_format: Output format ('png', 'pdf', 'markdown', or 'all')
        """
        self.logger.info("Starting model evaluation pipeline...")
        
        # Load all results
        all_results = self.load_all_results()
        
        if not all_results:
            self.logger.error("No model results found. Please run test.py first.")
            return
        
        self.logger.info(f"Found results for {len(all_results)} model-data combinations")
        
        # Evaluate each model
        for model_key, results in all_results.items():
            try:
                self.evaluate_single_model(model_key, results)
            except Exception as e:
                self.logger.error(f"Failed to evaluate {model_key}: {str(e)}")
                self.logger.exception("Detailed traceback:")
        
        # Create comparison plots
        try:
            self.create_comparison_plots(all_results)
        except Exception as e:
            self.logger.error(f"Failed to create comparison plots: {str(e)}")
            self.logger.exception("Detailed traceback:")
        
        # Create report
        try:
            self.create_report(all_results, output_format)
        except Exception as e:
            self.logger.error(f"Failed to create report: {str(e)}")
            self.logger.exception("Detailed traceback:")
        
        self.logger.info(f"\n✅ Evaluation complete! Check the {self.reports_dir}/ directory for outputs.")


def main():
    """Main entry point for the evaluation script."""
    parser = argparse.ArgumentParser(
        description="Generate evaluation plots and reports for tested models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate comparison report for both SMOTE and regular models
  python scripts/evaluate.py --comparison-mode
  
  # Evaluate only SMOTE models
  python scripts/evaluate.py --data-type smote
  
  # Evaluate only regular models
  python scripts/evaluate.py --data-type regular
  
  # Generate PDF report only
  python scripts/evaluate.py --format pdf --comparison-mode
  
  # Debug mode
  python scripts/evaluate.py --debug --comparison-mode
        """
    )
    
    parser.add_argument(
        '--data-type',
        type=str,
        choices=['smote', 'regular', 'both'],
        default='both',
        help='Data type to evaluate (default: both)'
    )
    
    parser.add_argument(
        '--comparison-mode',
        action='store_true',
        help='Enable SMOTE vs Regular comparison mode'
    )
    
    parser.add_argument(
        '--format',
        type=str,
        choices=['png', 'pdf', 'markdown', 'all'],
        default='all',
        help='Output format for reports'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = ModelEvaluator(
        debug=args.debug,
        data_type=args.data_type,
        comparison_mode=args.comparison_mode or args.data_type == 'both'
    )
    
    # Run evaluation
    try:
        evaluator.run_evaluation(output_format=args.format)
    except KeyboardInterrupt:
        evaluator.logger.warning("\nEvaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        evaluator.logger.error(f"Evaluation failed: {str(e)}")
        if args.debug:
            evaluator.logger.exception("Detailed traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()