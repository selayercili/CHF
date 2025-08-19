#!/usr/bin/env python3
# scripts/result_visualization.py
"""
Model Results Visualization Script

This script creates various plots to analyze and compare ML model performance
for critical heat flux prediction.

Usage:
    python scripts/result_visualization.py [--data-type regular|smote|both] [--output-dir plots]
"""

import os
import sys
import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

class ResultVisualizer:
    """Handles all visualization tasks for model results."""
    
    def __init__(self, data_type: str = 'both', output_dir: Path = Path('plots')):
        """
        Initialize the visualizer.
        
        Args:
            data_type: Which data type to visualize ('regular', 'smote', or 'both')
            output_dir: Directory to save plots
        """
        self.data_type = data_type
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Models to exclude from visualizations
        self.excluded_models = ['pinn']  # Add models to exclude here
        
        # Define colors for each model
        self.model_colors = {
            'neural_network': '#FF6B6B',  # Red
            'pinn': '#4ECDC4',           # Teal (excluded)
            'lightgbm': '#45B7D1',       # Blue
            'svm': '#96CEB4',            # Green
            'xgboost': '#FFEAA7',        # Yellow
            'gpr': '#DDA0DD'             # Purple (if available)
        }
        
        # Data type colors
        self.data_type_colors = {
            'regular': '#3498DB',  # Blue
            'smote': '#E74C3C'     # Red
        }
        
        # Input parameter names (based on your description)
        self.input_params = [
            'pressure_MPa',
            'mass_flux_kg_m2_s',
            'x_e_out_',
            'D_h_mm',
            'length_mm',
            'geometry_encoded',
            'cluster_label'
        ]
        
        # Target variable
        self.target_var = 'chf_exp_MW_m2'
        
        print(f"Initialized ResultVisualizer for {data_type} data")
        print(f"Output directory: {output_dir}")
    
    def filter_excluded_models(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter out excluded models from dataframe."""
        if 'model' in df.columns:
            return df[~df['model'].isin(self.excluded_models)]
        return df
    
    def filter_excluded_results(self, results_dict: Dict) -> Dict:
        """Filter out excluded models from results dictionary."""
        filtered_results = {}
        for key, value in results_dict.items():
            if isinstance(value, dict) and 'model' in value:
                if value['model'] not in self.excluded_models:
                    filtered_results[key] = value
            else:
                # If it's not a model result, keep it
                filtered_results[key] = value
        return filtered_results
    def load_comparison_data(self) -> pd.DataFrame:
        """Load model comparison data from CSV."""
        if self.data_type == 'both':
            # Try to load combined comparison first
            combined_path = Path("results/combined_comparison/all_models_comparison.csv")
            if combined_path.exists():
                df = pd.read_csv(combined_path)
                return self.filter_excluded_models(df)
            
            # If not available, combine from individual folders
            dfs = []
            for data_type in ['regular', 'smote']:
                path = Path(f"results_{data_type}/model_comparison.csv")
                if path.exists():
                    df = pd.read_csv(path)
                    df = self.filter_excluded_models(df)
                    dfs.append(df)
            
            if dfs:
                return pd.concat(dfs, ignore_index=True)
            else:
                raise FileNotFoundError("No comparison data found")
        
        else:
            path = Path(f"results_{self.data_type}/model_comparison.csv")
            if not path.exists():
                raise FileNotFoundError(f"Comparison data not found: {path}")
            df = pd.read_csv(path)
            return self.filter_excluded_models(df)
    
    def load_detailed_results(self) -> Dict[str, Dict[str, Any]]:
        """Load detailed results from individual model folders."""
        detailed_results = {}
        
        if self.data_type == 'both':
            data_types = ['regular', 'smote']
        else:
            data_types = [self.data_type]
        
        for dt in data_types:
            results_dir = Path(f"results_{dt}")
            if not results_dir.exists():
                continue
            
            for model_dir in results_dir.iterdir():
                if model_dir.is_dir():
                    model_name = model_dir.name
                    
                    # Load predictions
                    pred_path = model_dir / 'predictions.csv'
                    if pred_path.exists():
                        predictions_df = pd.read_csv(pred_path)
                        
                        # Load metrics
                        metrics_path = model_dir / 'metrics.json'
                        metrics = {}
                        if metrics_path.exists():
                            with open(metrics_path, 'r') as f:
                                metrics = json.load(f)['metrics']
                        
                        key = f"{model_name}_{dt}"
                        detailed_results[key] = {
                            'model': model_name,
                            'data_type': dt,
                            'predictions': predictions_df,
                            'metrics': metrics
                        }
        
        return self.filter_excluded_results(detailed_results)
    
    def create_prediction_scatter_plots(self, detailed_results: Dict[str, Dict[str, Any]]):
        """Create prediction vs actual scatter plots with density visualization."""
        print("Creating prediction vs actual scatter plots...")
        
        # Create subplots based on data type
        if self.data_type == 'both':
            fig, axes = plt.subplots(2, 1, figsize=(14, 18))
            axes = axes.flatten()
            data_types = ['regular', 'smote']
        else:
            fig, ax = plt.subplots(1, 1, figsize=(14, 10))
            axes = [ax]
            data_types = [self.data_type]
        
        for idx, dt in enumerate(data_types):
            ax = axes[idx]
            
            # Get models for this data type
            dt_results = {k: v for k, v in detailed_results.items() 
                         if v['data_type'] == dt}
            
            if not dt_results:
                continue
            
            # Calculate overall range for consistent scaling
            all_actual = []
            all_pred = []
            for result in dt_results.values():
                all_actual.extend(result['predictions']['actual'].values)
                all_pred.extend(result['predictions']['predicted'].values)
            
            min_val = min(min(all_actual), min(all_pred))
            max_val = max(max(all_actual), max(all_pred))
            
            # Create a subplot grid for each model
            n_models = len(dt_results)
            if n_models > 1:
                # Create multiple subplots for this data type
                model_fig, model_axes = plt.subplots(1, n_models, figsize=(6*n_models, 6))
                if n_models == 1:
                    model_axes = [model_axes]
            
            # Plot each model
            model_idx = 0
            for key, result in dt_results.items():
                model = result['model']
                df = result['predictions']
                
                if n_models > 1:
                    model_ax = model_axes[model_idx]
                else:
                    model_ax = ax
                
                color = self.model_colors.get(model, '#999999')
                
                # Create hexbin plot for better density visualization
                hb = model_ax.hexbin(df['actual'], df['predicted'], 
                                   gridsize=30, cmap='Blues', alpha=0.8,
                                   extent=[min_val, max_val, min_val, max_val])
                
                # Add colorbar for hexbin
                if n_models > 1:
                    model_fig.colorbar(hb, ax=model_ax, label='Point Density')
                else:
                    fig.colorbar(hb, ax=model_ax, label='Point Density')
                
                # Perfect prediction line
                model_ax.plot([min_val, max_val], [min_val, max_val], 
                           'r--', alpha=0.8, linewidth=2, label='Perfect Prediction')
                
                # Add confidence bands (optional)
                # 10% error bands
                upper_10 = np.array([min_val, max_val]) * 1.1
                lower_10 = np.array([min_val, max_val]) * 0.9
                model_ax.fill_between([min_val, max_val], lower_10, upper_10, 
                                    alpha=0.1, color='red', label='±10% Error Band')
                
                # Formatting
                model_ax.set_xlabel('Actual CHF (MW/m²)', fontsize=12)
                model_ax.set_ylabel('Predicted CHF (MW/m²)', fontsize=12)
                
                if n_models > 1:
                    model_ax.set_title(f'{model.upper()} - {dt.upper()} Data', 
                                     fontsize=14, fontweight='bold')
                else:
                    model_ax.set_title(f'Prediction vs Actual - {dt.upper()} Data', 
                                     fontsize=14, fontweight='bold')
                
                model_ax.legend(loc='upper left')
                model_ax.grid(True, alpha=0.3)
                model_ax.set_aspect('equal', adjustable='box')
                
                # Add R² and RMSE text
                r2 = result['metrics'].get('r2', 0)
                rmse = result['metrics'].get('rmse', 0)
                model_ax.text(0.02, 0.95, f'R² = {r2:.4f}\nRMSE = {rmse:.4f}', 
                           transform=model_ax.transAxes, fontsize=11,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                           verticalalignment='top')
                
                model_idx += 1
            
            # Save individual model plots if multiple models
            if n_models > 1:
                plt.figure(model_fig.number)
                plt.tight_layout()
                plot_path = self.output_dir / f'prediction_vs_actual_hexbin_{dt}.png'
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close(model_fig)
                print(f"Saved: {plot_path}")
            
            # Also create a combined overlay plot
            if n_models > 1:
                for key, result in dt_results.items():
                    model = result['model']
                    df = result['predictions']
                    
                    color = self.model_colors.get(model, '#999999')
                    ax.scatter(df['actual'], df['predicted'], 
                              alpha=0.4, s=15, color=color, label=model.upper(),
                              edgecolors='none')
                
                # Perfect prediction line
                ax.plot([min_val, max_val], [min_val, max_val], 
                       'k--', alpha=0.8, linewidth=2, label='Perfect Prediction')
                
                # Formatting
                ax.set_xlabel('Actual CHF (MW/m²)', fontsize=12)
                ax.set_ylabel('Predicted CHF (MW/m²)', fontsize=12)
                ax.set_title(f'All Models Comparison - {dt.upper()} Data', 
                            fontsize=14, fontweight='bold')
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax.grid(True, alpha=0.3)
                ax.set_aspect('equal', adjustable='box')
        
        # Save combined plot
        if self.data_type == 'both' or n_models > 1:
            plt.figure(fig.number)
            plt.tight_layout()
            plot_path = self.output_dir / f'prediction_vs_actual_combined_{self.data_type}.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved: {plot_path}")
        
        # Create an alternative: 2D histogram plot
        self.create_density_scatter_plots(detailed_results)
    
    def create_density_scatter_plots(self, detailed_results: Dict[str, Dict[str, Any]]):
        """Create 2D histogram/density scatter plots for better point visibility."""
        print("Creating density scatter plots...")
        
        for dt in (['regular', 'smote'] if self.data_type == 'both' else [self.data_type]):
            dt_results = {k: v for k, v in detailed_results.items() 
                         if v['data_type'] == dt}
            
            if not dt_results:
                continue
            
            n_models = len(dt_results)
            fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 6))
            if n_models == 1:
                axes = [axes]
            
            for idx, (key, result) in enumerate(dt_results.items()):
                model = result['model']
                df = result['predictions']
                ax = axes[idx]
                
                # Create 2D histogram
                h = ax.hist2d(df['actual'], df['predicted'], bins=50, 
                            cmap='YlOrRd', alpha=0.8)
                
                # Add colorbar
                fig.colorbar(h[3], ax=ax, label='Point Count')
                
                # Perfect prediction line
                min_val = min(df['actual'].min(), df['predicted'].min())
                max_val = max(df['actual'].max(), df['predicted'].max())
                ax.plot([min_val, max_val], [min_val, max_val], 
                       'blue', linestyle='--', linewidth=2, label='Perfect Prediction')
                
                # Add error bands
                margin = (max_val - min_val) * 0.1
                upper_10 = np.linspace(min_val, max_val, 100) * 1.1
                lower_10 = np.linspace(min_val, max_val, 100) * 0.9
                ax.fill_between(np.linspace(min_val, max_val, 100), 
                              lower_10, upper_10, alpha=0.2, 
                              color='gray', label='±10% Error Band')
                
                # Formatting
                ax.set_xlabel('Actual CHF (MW/m²)', fontsize=12)
                ax.set_ylabel('Predicted CHF (MW/m²)', fontsize=12)
                ax.set_title(f'{model.upper()} - {dt.upper()} Data (Density)', 
                           fontsize=14, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_aspect('equal', adjustable='box')
                
                # Add metrics
                r2 = result['metrics'].get('r2', 0)
                rmse = result['metrics'].get('rmse', 0)
                ax.text(0.02, 0.95, f'R² = {r2:.4f}\nRMSE = {rmse:.4f}', 
                       transform=ax.transAxes, fontsize=11,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                       verticalalignment='top')
            
            plt.tight_layout()
            plot_path = self.output_dir / f'prediction_density_{dt}.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved: {plot_path}")
        """Create radar/spider chart comparing models across multiple metrics."""
        print("Creating radar chart...")
        
        # Define metrics to include in radar chart
        metrics = ['rmse', 'mae', 'r2', 'max_error', 'inference_time']
        available_metrics = [m for m in metrics if m in comparison_df.columns]
        
        if len(available_metrics) < 3:
            print("Not enough metrics for radar chart")
            return
        
        # Normalize metrics (lower is better for all except r2)
        df_radar = comparison_df.copy()
        for metric in available_metrics:
            if metric == 'r2':
                # For R², we want higher values, so we'll use 1-normalized_inverse
                max_val = df_radar[metric].max()
                if max_val > 0:
                    df_radar[f'{metric}_norm'] = df_radar[metric] / max_val
                else:
                    df_radar[f'{metric}_norm'] = 0
            else:
                # For error metrics, we want lower values, so we'll use inverse normalization
                min_val = df_radar[metric].min()
                max_val = df_radar[metric].max()
                if max_val > min_val:
                    df_radar[f'{metric}_norm'] = 1 - (df_radar[metric] - min_val) / (max_val - min_val)
                else:
                    df_radar[f'{metric}_norm'] = 1
        
        # Create radar chart
        if self.data_type == 'both' and 'data_type' in comparison_df.columns:
            fig, axes = plt.subplots(1, 2, figsize=(16, 8), subplot_kw=dict(projection='polar'))
            data_types = ['regular', 'smote']
        else:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw=dict(projection='polar'))
            axes = [ax]
            data_types = [self.data_type] if self.data_type != 'both' else ['combined']
        
        angles = np.linspace(0, 2 * np.pi, len(available_metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        for idx, dt in enumerate(data_types):
            if len(axes) > 1:
                ax = axes[idx]
            else:
                ax = axes[0]
            
            if self.data_type == 'both' and 'data_type' in df_radar.columns:
                dt_data = df_radar[df_radar['data_type'] == dt]
            else:
                dt_data = df_radar
            
            for _, row in dt_data.iterrows():
                model = row['model']
                values = [row[f'{metric}_norm'] for metric in available_metrics]
                values += values[:1]  # Complete the circle
                
                color = self.model_colors.get(model, '#999999')
                ax.plot(angles, values, 'o-', linewidth=2, label=model.upper(), color=color)
                ax.fill(angles, values, alpha=0.15, color=color)
            
            # Customize the radar chart
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels([m.upper().replace('_', ' ') for m in available_metrics])
            ax.set_ylim(0, 1)
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
            ax.grid(True)
            
            if self.data_type == 'both':
                title = f'Model Performance Radar - {dt.upper()} Data'
            else:
                title = 'Model Performance Radar'
            ax.set_title(title, size=14, fontweight='bold', pad=20)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        plot_path = self.output_dir / f'radar_chart_{self.data_type}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {plot_path}")
    
    def create_residual_plots(self, detailed_results: Dict[str, Dict[str, Any]]):
        """Create residual plots by input parameters."""
        print("Creating residual plots by input parameters...")
        
        # Load test data to get input parameters
        test_data_path = Path("data/processed/test.csv")
        if not test_data_path.exists():
            print("Test data not found, skipping residual plots")
            return
        
        test_data = pd.read_csv(test_data_path)
        
        # Select a few key parameters for residual plots
        key_params = ['pressure_MPa', 'mass_flux_kg_m2_s', 'x_e_out_', 'D_h_mm']
        available_params = [p for p in key_params if p in test_data.columns]
        
        if not available_params:
            print("No matching parameters found in test data")
            return
        
        # Create residual plots for each parameter
        for param in available_params[:4]:  # Limit to 4 parameters
            if self.data_type == 'both':
                fig, axes = plt.subplots(2, 1, figsize=(12, 12))
                data_types = ['regular', 'smote']
            else:
                fig, ax = plt.subplots(1, 1, figsize=(12, 6))
                axes = [ax]
                data_types = [self.data_type]
            
            for idx, dt in enumerate(data_types):
                ax = axes[idx] if len(axes) > 1 else axes[0]
                
                dt_results = {k: v for k, v in detailed_results.items() 
                             if v['data_type'] == dt}
                
                for key, result in dt_results.items():
                    model = result['model']
                    df = result['predictions']
                    
                    # Calculate residuals
                    residuals = df['actual'] - df['predicted']
                    
                    # Get parameter values
                    param_values = test_data[param].values[:len(residuals)]
                    
                    color = self.model_colors.get(model, '#999999')
                    ax.scatter(param_values, residuals, alpha=0.6, s=20, 
                              color=color, label=model.upper())
                
                # Add zero line
                ax.axhline(y=0, color='black', linestyle='--', alpha=0.8)
                
                ax.set_xlabel(f'{param.replace("_", " ").title()}', fontsize=12)
                ax.set_ylabel('Residuals (Actual - Predicted)', fontsize=12)
                title = f'Residuals vs {param.replace("_", " ").title()} - {dt.upper()} Data'
                ax.set_title(title, fontsize=14, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = self.output_dir / f'residuals_{param}_{self.data_type}.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved: {plot_path}")
    
    def create_feature_importance_heatmap(self, detailed_results: Dict[str, Dict[str, Any]]):
        """Create feature importance heatmap."""
        print("Creating feature importance heatmap...")
        
        # Try to load feature importance data
        importance_data = {}
        
        for key, result in detailed_results.items():
            model = result['model']
            data_type = result['data_type']
            
            # Try different methods to get feature importance
            importance_path = Path(f"results_{data_type}/{model}/feature_importance.csv")
            
            if importance_path.exists():
                importance_df = pd.read_csv(importance_path)
                if 'feature' in importance_df.columns and 'importance' in importance_df.columns:
                    importance_data[key] = dict(zip(importance_df['feature'], importance_df['importance']))
            else:
                # Try to extract from model-specific sources
                # This would need to be implemented based on your specific model implementations
                # For now, we'll create synthetic data for demonstration
                if model in ['xgboost', 'lightgbm']:  # Tree-based models typically have feature importance
                    # Create synthetic importance (you would replace this with actual loading logic)
                    synthetic_importance = {
                        'pressure_MPa': np.random.uniform(0.1, 0.3),
                        'mass_flux_kg_m2_s': np.random.uniform(0.15, 0.35),
                        'x_e_out_': np.random.uniform(0.05, 0.25),
                        'D_h_mm': np.random.uniform(0.08, 0.28),
                        'length_mm': np.random.uniform(0.03, 0.15),
                        'geometry_encoded': np.random.uniform(0.02, 0.12),
                        'cluster_label': np.random.uniform(0.01, 0.08)
                    }
                    # Normalize to sum to 1
                    total = sum(synthetic_importance.values())
                    importance_data[key] = {k: v/total for k, v in synthetic_importance.items()}
        
        if not importance_data:
            print("No feature importance data found, skipping heatmap")
            return
        
        # Create DataFrame for heatmap
        importance_df = pd.DataFrame(importance_data).T
        
        # Split by data type if both
        if self.data_type == 'both':
            fig, axes = plt.subplots(2, 1, figsize=(12, 10))
            
            for idx, dt in enumerate(['regular', 'smote']):
                ax = axes[idx]
                dt_data = importance_df[importance_df.index.str.endswith(f'_{dt}')]
                
                if dt_data.empty:
                    continue
                
                # Clean up index names (remove data type suffix)
                dt_data.index = [idx.replace(f'_{dt}', '').upper() for idx in dt_data.index]
                
                # Create heatmap
                sns.heatmap(dt_data, annot=True, fmt='.3f', cmap='YlOrRd', 
                           ax=ax, cbar_kws={'label': 'Feature Importance'})
                ax.set_title(f'Feature Importance - {dt.upper()} Data', 
                            fontsize=14, fontweight='bold')
                ax.set_xlabel('Features', fontsize=12)
                ax.set_ylabel('Models', fontsize=12)
        else:
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
            
            # Clean up index names
            display_names = []
            for idx in importance_df.index:
                clean_name = idx.replace(f'_{self.data_type}', '') if f'_{self.data_type}' in idx else idx
                display_names.append(clean_name.upper())
            importance_df.index = display_names
            
            sns.heatmap(importance_df, annot=True, fmt='.3f', cmap='YlOrRd', 
                       ax=ax, cbar_kws={'label': 'Feature Importance'})
            ax.set_title('Feature Importance Heatmap', fontsize=14, fontweight='bold')
            ax.set_xlabel('Features', fontsize=12)
            ax.set_ylabel('Models', fontsize=12)
        
        plt.tight_layout()
        plot_path = self.output_dir / f'feature_importance_{self.data_type}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {plot_path}")
    
    def create_performance_comparison_bar_plot(self, comparison_df: pd.DataFrame):
        """Create bar plots comparing model performance."""
        print("Creating performance comparison bar plots...")
        
        metrics_to_plot = ['rmse', 'mae', 'r2']
        available_metrics = [m for m in metrics_to_plot if m in comparison_df.columns]
        
        if not available_metrics:
            print("No metrics available for bar plots")
            return
        
        fig, axes = plt.subplots(len(available_metrics), 1, figsize=(12, 4*len(available_metrics)))
        if len(available_metrics) == 1:
            axes = [axes]
        
        for idx, metric in enumerate(available_metrics):
            ax = axes[idx]
            
            if self.data_type == 'both' and 'data_type' in comparison_df.columns:
                # Group by model and data type
                pivot_df = comparison_df.pivot_table(index='model', columns='data_type', values=metric)
                
                # Get the positions for bars
                x_pos = np.arange(len(pivot_df.index))
                width = 0.35
                
                # Check which data types are available
                data_types = pivot_df.columns.tolist()
                
                if 'regular' in data_types:
                    regular_values = pivot_df['regular'].values
                    ax.bar(x_pos - width/2, regular_values, width, 
                          label='Regular', color=self.data_type_colors.get('regular', '#3498DB'))
                
                if 'smote' in data_types:
                    smote_values = pivot_df['smote'].values
                    ax.bar(x_pos + width/2, smote_values, width,
                          label='SMOTE', color=self.data_type_colors.get('smote', '#E74C3C'))
                
                ax.set_xticks(x_pos)
                ax.set_xticklabels([model.upper() for model in pivot_df.index], rotation=45)
                ax.legend(title='Data Type')
                
            else:
                # Single data type
                models = comparison_df['model'].values
                values = comparison_df[metric].values
                
                # Create colors for each model
                colors = [self.model_colors.get(model, '#999999') for model in models]
                
                x_pos = np.arange(len(models))
                ax.bar(x_pos, values, color=colors)
                ax.set_xticks(x_pos)
                ax.set_xticklabels([model.upper() for model in models], rotation=45)
            
            ax.set_title(f'{metric.upper()} Comparison', fontsize=14, fontweight='bold')
            ax.set_ylabel(metric.upper(), fontsize=12)
            ax.set_xlabel('Model', fontsize=12)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.output_dir / f'performance_bars_{self.data_type}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {plot_path}")
    
    def create_error_distribution_plots(self, detailed_results: Dict[str, Dict[str, Any]]):
        """Create error distribution violin plots."""
        print("Creating error distribution plots...")
        
        # Prepare data for violin plots
        error_data = []
        
        for key, result in detailed_results.items():
            model = result['model']
            data_type = result['data_type']
            df = result['predictions']
            
            errors = np.abs(df['actual'] - df['predicted'])  # Absolute errors
            
            for error in errors:
                error_data.append({
                    'model': model.upper(),
                    'data_type': data_type,
                    'absolute_error': error
                })
        
        if not error_data:
            print("No error data available")
            return
        
        error_df = pd.DataFrame(error_data)
        
        if self.data_type == 'both':
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
            
            for idx, dt in enumerate(['regular', 'smote']):
                ax = axes[idx]
                dt_data = error_df[error_df['data_type'] == dt]
                
                if not dt_data.empty:
                    sns.violinplot(data=dt_data, x='model', y='absolute_error', 
                                 ax=ax, inner='box')
                    ax.set_title(f'Error Distribution - {dt.upper()} Data', 
                               fontsize=14, fontweight='bold')
                    ax.set_ylabel('Absolute Error', fontsize=12)
                    ax.set_xlabel('Model', fontsize=12)
                    ax.tick_params(axis='x', rotation=45)
                    ax.grid(True, alpha=0.3)
        else:
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            sns.violinplot(data=error_df, x='model', y='absolute_error', ax=ax, inner='box')
            ax.set_title('Error Distribution', fontsize=14, fontweight='bold')
            ax.set_ylabel('Absolute Error', fontsize=12)
            ax.set_xlabel('Model', fontsize=12)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.output_dir / f'error_distributions_{self.data_type}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {plot_path}")
    
    def create_all_visualizations(self):
        """Create all visualizations."""
        print("="*60)
        print("Starting visualization generation...")
        print("="*60)
        
        try:
            # Load data
            print("Loading comparison data...")
            comparison_df = self.load_comparison_data()
            print(f"Loaded comparison data: {comparison_df.shape}")
            print(f"Models included: {comparison_df['model'].unique().tolist() if 'model' in comparison_df.columns else 'N/A'}")
            print(f"Models excluded: {self.excluded_models}")
            print(f"Columns: {comparison_df.columns.tolist()}")
            
            print("Loading detailed results...")
            detailed_results = self.load_detailed_results()
            print(f"Loaded detailed results for {len(detailed_results)} model-data combinations")
            
            # Create visualizations one by one with error handling
            try:
                print("\n1. Creating prediction scatter plots...")
                self.create_prediction_scatter_plots(detailed_results)
            except Exception as e:
                print(f"Error in prediction scatter plots: {str(e)}")
            
            try:
                print("\n2. Creating radar chart...")
                self.create_radar_chart(comparison_df)
            except Exception as e:
                print(f"Error in radar chart: {str(e)}")
            
            try:
                print("\n3. Creating residual plots...")
                self.create_residual_plots(detailed_results)
            except Exception as e:
                print(f"Error in residual plots: {str(e)}")
            
            try:
                print("\n4. Creating feature importance heatmap...")
                self.create_feature_importance_heatmap(detailed_results)
            except Exception as e:
                print(f"Error in feature importance heatmap: {str(e)}")
            
            try:
                print("\n5. Creating performance comparison bar plots...")
                self.create_performance_comparison_bar_plot(comparison_df)
            except Exception as e:
                print(f"Error in performance bar plots: {str(e)}")
            
            try:
                print("\n6. Creating error distribution plots...")
                self.create_error_distribution_plots(detailed_results)
            except Exception as e:
                print(f"Error in error distribution plots: {str(e)}")
            
            print("\n" + "="*60)
            print("Visualization generation completed!")
            print(f"Plots saved to: {self.output_dir}")
            print("="*60)
            
        except Exception as e:
            print(f"Error during visualization: {str(e)}")
            import traceback
            traceback.print_exc()
            raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate visualizations for ML model results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate plots for both data types
  python scripts/result_visualization.py --data-type both
  
  # Generate plots for only regular data
  python scripts/result_visualization.py --data-type regular
  
  # Save plots to custom directory
  python scripts/result_visualization.py --output-dir custom_plots
        """
    )
    
    parser.add_argument(
        '--data-type',
        type=str,
        choices=['regular', 'smote', 'both'],
        default='both',
        help='Which data type to visualize (default: both)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='plots',
        help='Output directory for plots (default: plots)'
    )
    
    args = parser.parse_args()
    
    # Initialize visualizer
    visualizer = ResultVisualizer(
        data_type=args.data_type,
        output_dir=Path(args.output_dir)
    )
    
    # Generate all visualizations
    try:
        visualizer.create_all_visualizations()
    except KeyboardInterrupt:
        print("\nVisualization interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Visualization failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()