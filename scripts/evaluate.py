#!/usr/bin/env python3
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
from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import evaluation utilities
from src.evaluation.plotting import (
    plot_predictions_vs_actual,
    plot_residuals,
    plot_qq_plot,
    plot_feature_importance,
    plot_model_comparison,
    plot_error_distribution,
    plot_prediction_intervals,
    plot_learning_curves
)

from src.evaluation.metrics import (
    calculate_additional_metrics,
    create_metrics_report,
    calculate_prediction_intervals
)


class ModelEvaluator:
    """Handles comprehensive model evaluation and visualization."""
    
    def __init__(self, results_dir: str = "./results"):
        """
        Initialize the ModelEvaluator.
        
        Args:
            results_dir: Directory containing model test results
        """
        self.results_dir = Path(results_dir)
        self.figures_dir = Path("reports/figures")
        self.reports_dir = Path("reports")
        
        # Create directories
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(exist_ok=True)
        
        # Set plotting style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
    def load_model_results(self, model_name: str) -> Dict[str, Any]:
        """Load results for a specific model."""
        model_results_dir = self.results_dir / model_name
        
        # Load full results pickle
        results_path = model_results_dir / 'full_results.pkl'
        if not results_path.exists():
            print(f"Warning: No results found for {model_name}")
            return None
            
        with open(results_path, 'rb') as f:
            results = pickle.load(f)
            
        return results
    
    def load_all_results(self) -> Dict[str, Dict[str, Any]]:
        """Load results for all models."""
        all_results = {}
        
        for model_dir in self.results_dir.iterdir():
            if model_dir.is_dir():
                model_name = model_dir.name
                results = self.load_model_results(model_name)
                if results:
                    all_results[model_name] = results
                    
        return all_results
    
    def evaluate_single_model(self, model_name: str, results: Dict[str, Any]) -> None:
        """
        Generate evaluation plots for a single model.
        
        Args:
            model_name: Name of the model
            results: Model results dictionary
        """
        print(f"\nEvaluating {model_name}...")
        
        # Create model-specific directory
        model_figures_dir = self.figures_dir / model_name
        model_figures_dir.mkdir(exist_ok=True)
        
        # Extract data
        y_true = results['actual']
        y_pred = results['predictions']
        
        # 1. Predictions vs Actual
        fig = plot_predictions_vs_actual(y_true, y_pred, model_name)
        fig.savefig(model_figures_dir / 'predictions_vs_actual.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Residuals Plot
        fig = plot_residuals(y_true, y_pred, model_name)
        fig.savefig(model_figures_dir / 'residuals.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Q-Q Plot
        fig = plot_qq_plot(y_true, y_pred, model_name)
        fig.savefig(model_figures_dir / 'qq_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Error Distribution
        fig = plot_error_distribution(y_true, y_pred, model_name)
        fig.savefig(model_figures_dir / 'error_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Feature Importance (if available)
        if 'feature_importance' in results and results['feature_importance'] is not None:
            fig = plot_feature_importance(results['feature_importance'], model_name)
            fig.savefig(model_figures_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 6. Prediction Intervals
        intervals = calculate_prediction_intervals(y_true, y_pred)
        fig = plot_prediction_intervals(y_true, y_pred, intervals, model_name)
        fig.savefig(model_figures_dir / 'prediction_intervals.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Calculate additional metrics
        additional_metrics = calculate_additional_metrics(y_true, y_pred)
        
        # Update results with additional metrics
        results['metrics'].update(additional_metrics)
        
        print(f"✓ Generated plots for {model_name}")
    
    def create_comparison_plots(self, all_results: Dict[str, Dict[str, Any]]) -> None:
        """Create plots comparing all models."""
        print("\nCreating model comparison plots...")
        
        # Prepare comparison data
        comparison_data = []
        for model_name, results in all_results.items():
            metrics = results['metrics'].copy()
            metrics['model'] = model_name
            comparison_data.append(metrics)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # 1. Model Comparison Bar Plot
        fig = plot_model_comparison(comparison_df)
        fig.savefig(self.figures_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Prediction Scatter for All Models
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for idx, (model_name, results) in enumerate(all_results.items()):
            if idx < 4:  # Show up to 4 models
                ax = axes[idx]
                y_true = results['actual']
                y_pred = results['predictions']
                
                ax.scatter(y_true, y_pred, alpha=0.5, s=30)
                ax.plot([y_true.min(), y_true.max()], 
                       [y_true.min(), y_true.max()], 
                       'r--', lw=2)
                ax.set_xlabel('Actual Values')
                ax.set_ylabel('Predicted Values')
                ax.set_title(f'{model_name}')
                
                # Add R² score
                r2 = results['metrics'].get('r2', 'N/A')
                ax.text(0.05, 0.95, f'R² = {r2:.3f}' if isinstance(r2, float) else 'R² = N/A',
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('Model Predictions Comparison', fontsize=16)
        plt.tight_layout()
        fig.savefig(self.figures_dir / 'predictions_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Error Distribution Comparison
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
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
        
        fig.savefig(self.figures_dir / 'error_distribution_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Created comparison plots")
    
    def create_report(self, all_results: Dict[str, Dict[str, Any]]) -> None:
        """Create a comprehensive evaluation report."""
        print("\nCreating evaluation report...")
        
        # Create PDF report
        pdf_path = self.reports_dir / f'evaluation_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
        
        with PdfPages(pdf_path) as pdf:
            # Title page
            fig = plt.figure(figsize=(8.5, 11))
            fig.text(0.5, 0.7, 'Model Evaluation Report', 
                    ha='center', va='center', fontsize=24, weight='bold')
            fig.text(0.5, 0.6, f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
                    ha='center', va='center', fontsize=12)
            fig.text(0.5, 0.5, f'Number of models evaluated: {len(all_results)}',
                    ha='center', va='center', fontsize=12)
            plt.axis('off')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Add all plots
            for plot_file in sorted(self.figures_dir.rglob('*.png')):
                img = plt.imread(plot_file)
                fig, ax = plt.subplots(figsize=(8.5, 11))
                ax.imshow(img)
                ax.axis('off')
                ax.set_title(plot_file.stem.replace('_', ' ').title(), fontsize=14, pad=20)
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
        
        print(f"✓ Created PDF report: {pdf_path}")
        
        # Create detailed metrics report
        metrics_report = create_metrics_report(all_results)
        report_path = self.reports_dir / 'detailed_metrics.md'
        with open(report_path, 'w') as f:
            f.write(metrics_report)
        
        print(f"✓ Created metrics report: {report_path}")
    
    def run_evaluation(self, output_format: str = 'all') -> None:
        """
        Run complete evaluation pipeline.
        
        Args:
            output_format: Output format ('png', 'pdf', or 'all')
        """
        print("Starting model evaluation pipeline...")
        
        # Load all results
        all_results = self.load_all_results()
        
        if not all_results:
            print("No model results found. Please run test_models.py first.")
            return
        
        print(f"Found results for {len(all_results)} models")
        
        # Evaluate each model
        for model_name, results in all_results.items():
            self.evaluate_single_model(model_name, results)
        
        # Create comparison plots
        self.create_comparison_plots(all_results)
        
        # Create report
        if output_format in ['pdf', 'all']:
            self.create_report(all_results)
        
        print("\n✅ Evaluation complete! Check the reports/figures directory for outputs.")


def main():
    """Main entry point for the evaluation script."""
    parser = argparse.ArgumentParser(
        description="Generate evaluation plots and reports for tested models"
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
        choices=['png', 'pdf', 'all'],
        default='all',
        help='Output format for reports'
    )
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = ModelEvaluator(results_dir=args.results_dir)
    
    # Run evaluation
    try:
        evaluator.run_evaluation(output_format=args.format)
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
    except Exception as e:
        print(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
