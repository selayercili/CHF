#!/usr/bin/env python3
# scripts/evaluate.py
"""
Model Evaluation and Visualization Script

This script generates comprehensive evaluation plots and reports
for all tested models.

Usage:
    python scripts/evaluate.py [--results-dir RESULTS_DIR] [--format FORMAT]
"""

import os
import sys
import argparse
import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import warnings

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
    
    def __init__(self, results_dir: str = "./results", debug: bool = False):
        """
        Initialize the ModelEvaluator.
        
        Args:
            results_dir: Directory containing model test results
            debug: Enable debug logging
        """
        # Setup logging
        log_level = 'DEBUG' if debug else 'INFO'
        setup_logging(log_level=log_level)
        self.logger = get_logger(__name__)
        
        # Setup directories
        self.results_dir = Path(results_dir)
        self.figures_dir = Path("reports/figures")
        self.reports_dir = Path("reports")
        
        # Create directories
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(exist_ok=True)
        
        # Set plotting style
        self.setup_plotting_style()
        
        self.logger.info("="*60)
        self.logger.info("Model Evaluator Initialized")
        self.logger.info(f"Results directory: {self.results_dir}")
        self.logger.info(f"Figures directory: {self.figures_dir}")
        self.logger.info("="*60)
    
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
    
    def load_model_results(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Load results for a specific model."""
        model_results_dir = self.results_dir / model_name
        
        # Check if directory exists
        if not model_results_dir.exists():
            self.logger.warning(f"No results directory found for {model_name}")
            return None
        
        # Load full results pickle
        results_path = model_results_dir / 'full_results.pkl'
        if not results_path.exists():
            self.logger.warning(f"No results pickle found for {model_name}")
            return None
        
        try:
            with open(results_path, 'rb') as f:
                results = pickle.load(f)
            
            self.logger.debug(f"Loaded results for {model_name}")
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to load results for {model_name}: {str(e)}")
            return None
    
    def load_all_results(self) -> Dict[str, Dict[str, Any]]:
        """Load results for all models."""
        all_results = {}
        
        # Find all model directories
        for model_dir in self.results_dir.iterdir():
            if model_dir.is_dir():
                model_name = model_dir.name
                results = self.load_model_results(model_name)
                if results:
                    all_results[model_name] = results
        
        self.logger.info(f"Loaded results for {len(all_results)} models")
        return all_results
    
    def evaluate_single_model(self, model_name: str, results: Dict[str, Any]) -> None:
        """
        Generate evaluation plots for a single model.
        
        Args:
            model_name: Name of the model
            results: Model results dictionary
        """
        self.logger.info(f"\nEvaluating {model_name}...")
        
        # Create model-specific directory
        model_figures_dir = self.figures_dir / model_name
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
        
        for plot_name, plot_func in plot_functions:
            try:
                fig = plot_func(y_true, y_pred, model_name)
                fig.savefig(model_figures_dir / f'{plot_name}.png')
                plt.close(fig)
                self.logger.debug(f"Created {plot_name} plot for {model_name}")
            except Exception as e:
                self.logger.error(f"Failed to create {plot_name} plot: {str(e)}")
        
        # Feature importance plot
        if 'feature_importance' in results and results['feature_importance'] is not None:
            try:
                fig = plot_feature_importance(results['feature_importance'], model_name)
                fig.savefig(model_figures_dir / 'feature_importance.png')
                plt.close(fig)
            except Exception as e:
                self.logger.error(f"Failed to create feature importance plot: {str(e)}")
        
        # Prediction intervals
        try:
            intervals = calculate_prediction_intervals(y_true, y_pred)
            fig = plot_prediction_intervals(y_true, y_pred, intervals, model_name)
            fig.savefig(model_figures_dir / 'prediction_intervals.png')
            plt.close(fig)
        except Exception as e:
            self.logger.error(f"Failed to create prediction intervals plot: {str(e)}")
        
        self.logger.info(f"✓ Generated plots for {model_name}")
    
    def create_comparison_plots(self, all_results: Dict[str, Dict[str, Any]]) -> None:
        """Create plots comparing all models."""
        self.logger.info("\nCreating model comparison plots...")
        
        # Prepare comparison data
        comparison_data = []
        for model_name, results in all_results.items():
            metrics = results['metrics'].copy()
            metrics['model'] = model_name
            comparison_data.append(metrics)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # 1. Model Comparison Bar Plot
        try:
            fig = plot_model_comparison(comparison_df)
            fig.savefig(self.figures_dir / 'model_comparison.png')
            plt.close(fig)
        except Exception as e:
            self.logger.error(f"Failed to create model comparison plot: {str(e)}")
        
        # 2. Prediction Scatter for All Models
        self.create_multi_model_scatter(all_results)
        
        # 3. Error Distribution Comparison
        self.create_error_distribution_comparison(all_results)
        
        # 4. Performance Radar Chart
        self.create_performance_radar(comparison_df)
        
        self.logger.info("✓ Created comparison plots")
    
    def create_multi_model_scatter(self, all_results: Dict[str, Dict[str, Any]]) -> None:
        """Create scatter plots comparing predictions across models."""
        n_models = len(all_results)
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        
        if n_models == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        for idx, (model_name, results) in enumerate(all_results.items()):
            if idx >= len(axes):
                break
            
            ax = axes[idx]
            y_true = results['actual']
            y_pred = results['predictions']
            
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
            r2 = results['metrics'].get('r2', 'N/A')
            rmse = results['metrics'].get('rmse', 'N/A')
            text = f'R² = {r2:.3f}\nRMSE = {rmse:.3f}' if isinstance(r2, float) else 'Metrics N/A'
            ax.text(0.05, 0.95, text, transform=ax.transAxes, 
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Hide unused subplots
        for idx in range(len(all_results), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Model Predictions Comparison', fontsize=16)
        plt.tight_layout()
        
        fig.savefig(self.figures_dir / 'predictions_comparison.png')
        plt.close(fig)
    
    def create_error_distribution_comparison(self, all_results: Dict[str, Dict[str, Any]]) -> None:
        """Create error distribution comparison plot."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for model_name, results in all_results.items():
            y_true = results['actual']
            y_pred = results['predictions']
            errors = y_true - y_pred
            
            # Plot KDE
            from scipy import stats
            kde = stats.gaussian_kde(errors)
            x_range = np.linspace(errors.min(), errors.max(), 100)
            ax.plot(x_range, kde(x_range), label=model_name, lw=2)
        
        ax.set_xlabel('Prediction Error')
        ax.set_ylabel('Density')
        ax.set_title('Error Distribution Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        fig.savefig(self.figures_dir / 'error_distribution_comparison.png')
        plt.close(fig)
    
    def create_performance_radar(self, comparison_df: pd.DataFrame) -> None:
        """Create radar chart comparing model performance."""
        from math import pi
        
        # Select metrics for radar chart
        metrics = ['rmse', 'mae', 'r2', 'max_error']
        available_metrics = [m for m in metrics if m in comparison_df.columns]
        
        if len(available_metrics) < 3:
            self.logger.warning("Not enough metrics for radar chart")
            return
        
        # Normalize metrics (0-1 scale, where 1 is best)
        normalized_df = comparison_df.copy()
        for metric in available_metrics:
            if metric == 'r2':
                # Higher is better
                normalized_df[metric] = (comparison_df[metric] - comparison_df[metric].min()) / \
                                       (comparison_df[metric].max() - comparison_df[metric].min())
            else:
                # Lower is better
                normalized_df[metric] = 1 - (comparison_df[metric] - comparison_df[metric].min()) / \
                                       (comparison_df[metric].max() - comparison_df[metric].min())
        
        # Create radar chart
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Angles for each metric
        angles = [n / float(len(available_metrics)) * 2 * pi for n in range(len(available_metrics))]
        angles += angles[:1]
        
        # Plot each model
        for idx, row in normalized_df.iterrows():
            values = [row[metric] for metric in available_metrics]
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, label=row['model'])
            ax.fill(angles, values, alpha=0.25)
        
        # Fix axis to go in the right order and start at 12 o'clock
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)
        
        # Draw axis lines for each angle and label
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(available_metrics)
        
        # Set y-axis limits and labels
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8'])
        
        # Add legend and title
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
        ax.set_title('Model Performance Comparison\n(Normalized metrics, 1.0 = best)', 
                    y=1.08, fontsize=14)
        
        plt.tight_layout()
        fig.savefig(self.figures_dir / 'performance_radar.png')
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
        """Create markdown evaluation report."""
        report_path = self.reports_dir / 'evaluation_report.md'
        
        with open(report_path, 'w') as f:
            # Header
            f.write("# CHF Model Evaluation Report\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            
            # Find best models
            best_rmse = min(all_results.items(), 
                          key=lambda x: x[1]['metrics'].get('rmse', float('inf')))
            best_r2 = max(all_results.items(), 
                        key=lambda x: x[1]['metrics'].get('r2', -float('inf')))
            
            f.write(f"- **Best RMSE**: {best_rmse[0]} ({best_rmse[1]['metrics']['rmse']:.6f})\n")
            f.write(f"- **Best R²**: {best_r2[0]} ({best_r2[1]['metrics']['r2']:.6f})\n")
            f.write(f"- **Total Models Evaluated**: {len(all_results)}\n\n")
            
            # Model Details
            f.write("## Model Performance Details\n\n")
            
            for model_name, results in sorted(all_results.items()):
                f.write(f"### {model_name}\n\n")
                
                # Metrics table
                f.write("| Metric | Value |\n")
                f.write("|--------|-------|\n")
                
                metrics = results['metrics']
                for metric_name, value in sorted(metrics.items()):
                    if isinstance(value, float):
                        f.write(f"| {metric_name} | {value:.6f} |\n")
                
                f.write("\n")
                
                # Plot references
                f.write("**Plots:**\n")
                f.write(f"- [Predictions vs Actual](figures/{model_name}/predictions_vs_actual.png)\n")
                f.write(f"- [Residuals Analysis](figures/{model_name}/residuals.png)\n")
                f.write(f"- [Error Distribution](figures/{model_name}/error_distribution.png)\n")
                
                if 'feature_importance' in results:
                    f.write(f"- [Feature Importance](figures/{model_name}/feature_importance.png)\n")
                
                f.write("\n")
            
            # Comparison Section
            f.write("## Model Comparison\n\n")
            f.write("![Model Comparison](figures/model_comparison.png)\n\n")
            f.write("![Performance Radar](figures/performance_radar.png)\n\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            f.write(f"Based on the evaluation results, **{best_rmse[0]}** shows the best overall ")
            f.write(f"performance with the lowest RMSE of {best_rmse[1]['metrics']['rmse']:.6f}.\n\n")
            
            # Check for overfitting
            f.write("### Potential Issues\n\n")
            for model_name, results in all_results.items():
                metrics = results['metrics']
                if 'mean_error' in metrics and abs(metrics['mean_error']) > 0.1:
                    f.write(f"- **{model_name}**: Shows bias (mean error: {metrics['mean_error']:.6f})\n")
        
        self.logger.info(f"✓ Created markdown report: {report_path}")
    
    def create_pdf_report(self, all_results: Dict[str, Dict[str, Any]]) -> None:
        """Create PDF evaluation report."""
        pdf_path = self.reports_dir / f'evaluation_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
        
        with PdfPages(pdf_path) as pdf:
            # Title page
            fig = plt.figure(figsize=(8.5, 11))
            fig.text(0.5, 0.7, 'CHF Model Evaluation Report', 
                    ha='center', va='center', fontsize=24, weight='bold')
            fig.text(0.5, 0.6, f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
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
                        title = f"{plot_file.parent.name} - {title}"
                    
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
        
        self.logger.info(f"Found results for {len(all_results)} models")
        
        # Evaluate each model
        for model_name, results in all_results.items():
            try:
                self.evaluate_single_model(model_name, results)
            except Exception as e:
                self.logger.error(f"Failed to evaluate {model_name}: {str(e)}")
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
        
        self.logger.info("\n✅ Evaluation complete! Check the reports/ directory for outputs.")


def main():
    """Main entry point for the evaluation script."""
    parser = argparse.ArgumentParser(
        description="Generate evaluation plots and reports for tested models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all reports and plots
  python scripts/evaluate.py
  
  # Only generate PNG plots
  python scripts/evaluate.py --format png
  
  # Generate PDF report only
  python scripts/evaluate.py --format pdf
  
  # Use custom results directory
  python scripts/evaluate.py --results-dir custom_results/
        """
    )
    
    parser.add_argument(
        '--results-dir',
        type=str,
        default='./results',
        help='Directory containing model test results'
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
    evaluator = ModelEvaluator(results_dir=args.results_dir, debug=args.debug)
    
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