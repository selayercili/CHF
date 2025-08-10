#!/usr/bin/env python3
# scripts/hyperparameter_search.py
"""
Hyperparameter Search Script for LightGBM

This script performs a grid search over the two most important LightGBM parameters
and produces visualizations of the results.

Usage:
    python scripts/hyperparameter_search.py [--n-points 25] [--cv-folds 3]
"""

import os
import sys
import argparse
import json
import pickle
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import lightgbm as lgb
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class LightGBMHyperparameterSearch:
    """Performs hyperparameter search for LightGBM model."""
    
    def __init__(self, n_points: int = 25, cv_folds: int = 3, random_state: int = 42):
        """
        Initialize the hyperparameter searcher.
        
        Args:
            n_points: Approximate number of parameter combinations to test
            cv_folds: Number of cross-validation folds
            random_state: Random seed for reproducibility
        """
        self.n_points = n_points
        self.cv_folds = cv_folds
        self.random_state = random_state
        
        # Setup paths
        self.data_dir = Path("data/processed")
        self.results_dir = Path("results/hyperparameter_search")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Define parameter grids for the two most important parameters
        self.setup_parameter_grid()
        
        # Storage for results
        self.search_results = []
        self.best_params = None
        self.best_score = float('inf')
        
        print("="*60)
        print("LightGBM Hyperparameter Search")
        print("="*60)
        print(f"Parameter combinations: {len(self.param_combinations)}")
        print(f"CV Folds: {cv_folds}")
        print(f"Results directory: {self.results_dir}")
        print("="*60)
    
    def setup_parameter_grid(self):
        """Setup the parameter grid for search."""
        # Calculate grid size to get approximately n_points
        grid_size = int(np.sqrt(self.n_points))
        
        # Two most important LightGBM parameters
        # 1. num_leaves: controls tree complexity (2^max_depth - 1 typically)
        # 2. learning_rate: controls boosting speed
        
        self.param_ranges = {
            'num_leaves': np.logspace(np.log10(10), np.log10(200), grid_size).astype(int),
            'learning_rate': np.logspace(np.log10(0.01), np.log10(0.3), grid_size)
        }
        
        # Create all combinations
        self.param_combinations = []
        for num_leaves in self.param_ranges['num_leaves']:
            for learning_rate in self.param_ranges['learning_rate']:
                self.param_combinations.append({
                    'num_leaves': int(num_leaves),
                    'learning_rate': float(learning_rate)
                })
        
        print(f"Parameter ranges:")
        print(f"  num_leaves: {self.param_ranges['num_leaves'].min()} to {self.param_ranges['num_leaves'].max()}")
        print(f"  learning_rate: {self.param_ranges['learning_rate'].min():.4f} to {self.param_ranges['learning_rate'].max():.4f}")
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load training and validation data."""
        print("\nLoading data...")
        
        # Try to load SMOTE data first, fallback to regular
        train_path = self.data_dir / "train_resampled.csv"
        if not train_path.exists():
            train_path = self.data_dir / "train.csv"
        
        test_path = self.data_dir / "test.csv"
        
        if not train_path.exists():
            raise FileNotFoundError(f"Training data not found at {train_path}")
        
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path) if test_path.exists() else None
        
        print(f"✓ Loaded training data: {train_df.shape}")
        if test_df is not None:
            print(f"✓ Loaded test data: {test_df.shape}")
        
        return train_df, test_df
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for training."""
        # Assume last column is target
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        return X, y
    
    def evaluate_params(self, params: Dict[str, Any], X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate a single parameter combination using cross-validation.
        
        Args:
            params: Parameter dictionary
            X: Features
            y: Target
            
        Returns:
            Dictionary with evaluation results
        """
        # Base parameters that stay constant
        base_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'n_estimators': 100,
            'max_depth': -1,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': self.random_state,
            'n_jobs': -1,
            'verbosity': -1
        }
        
        # Merge with search parameters
        model_params = {**base_params, **params}
        
        # Perform cross-validation
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        cv_scores = []
        train_scores = []
        val_times = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train model
            start_time = time.time()
            model = lgb.LGBMRegressor(**model_params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.callback.log_evaluation(0)]
            )
            train_time = time.time() - start_time
            
            # Evaluate
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)
            
            train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
            val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            
            train_scores.append(train_rmse)
            cv_scores.append(val_rmse)
            val_times.append(train_time)
        
        # Calculate statistics
        results = {
            'params': params,
            'cv_rmse_mean': np.mean(cv_scores),
            'cv_rmse_std': np.std(cv_scores),
            'train_rmse_mean': np.mean(train_scores),
            'train_rmse_std': np.std(train_scores),
            'cv_scores': cv_scores,
            'train_scores': train_scores,
            'mean_train_time': np.mean(val_times),
            'overfit_score': np.mean(cv_scores) - np.mean(train_scores)  # Higher means more overfit
        }
        
        return results
    
    def run_search(self) -> None:
        """Run the hyperparameter search."""
        # Load data
        train_df, test_df = self.load_data()
        X, y = self.prepare_data(train_df)
        
        print(f"\nSearching {len(self.param_combinations)} parameter combinations...")
        print("This may take a while...\n")
        
        # Progress bar
        pbar = tqdm(self.param_combinations, desc="Grid Search")
        
        for params in pbar:
            # Evaluate parameters
            results = self.evaluate_params(params, X, y)
            self.search_results.append(results)
            
            # Update best parameters
            if results['cv_rmse_mean'] < self.best_score:
                self.best_score = results['cv_rmse_mean']
                self.best_params = params
                pbar.set_postfix({'Best RMSE': f'{self.best_score:.4f}'})
        
        print(f"\n✓ Search complete!")
        print(f"Best parameters: {self.best_params}")
        print(f"Best CV RMSE: {self.best_score:.4f}")
        
        # Train final model with best parameters on full training set
        if test_df is not None:
            self.evaluate_best_on_test(X, y, test_df)
    
    def evaluate_best_on_test(self, X_train: np.ndarray, y_train: np.ndarray, 
                             test_df: pd.DataFrame) -> None:
        """Evaluate best model on test set."""
        print("\nEvaluating best model on test set...")
        
        X_test, y_test = self.prepare_data(test_df)
        
        # Train on full training set with best params
        base_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'n_estimators': 100,
            'random_state': self.random_state,
            'verbosity': -1
        }
        
        best_model = lgb.LGBMRegressor(**{**base_params, **self.best_params})
        best_model.fit(X_train, y_train)
        
        # Predict and evaluate
        test_pred = best_model.predict(X_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        test_mae = mean_absolute_error(y_test, test_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        print(f"Test Set Performance:")
        print(f"  RMSE: {test_rmse:.4f}")
        print(f"  MAE: {test_mae:.4f}")
        print(f"  R²: {test_r2:.4f}")
        
        # Store test results
        self.test_results = {
            'rmse': test_rmse,
            'mae': test_mae,
            'r2': test_r2,
            'predictions': test_pred,
            'actual': y_test
        }
    
    def create_plots(self) -> None:
        """Create three interesting plots from the search results."""
        print("\nGenerating visualizations...")
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # Convert results to DataFrame for easier manipulation
        results_df = pd.DataFrame(self.search_results)
        
        # Extract parameter values
        results_df['num_leaves'] = results_df['params'].apply(lambda x: x['num_leaves'])
        results_df['learning_rate'] = results_df['params'].apply(lambda x: x['learning_rate'])
        
        # ===== PLOT 1: Heatmap of CV RMSE =====
        ax1 = fig.add_subplot(gs[0, :])
        
        # Create pivot table for heatmap
        pivot_rmse = results_df.pivot(
            index='num_leaves',
            columns='learning_rate',
            values='cv_rmse_mean'
        )
        
        # Create heatmap
        im = ax1.imshow(pivot_rmse.values, aspect='auto', cmap='RdYlGn_r', 
                       interpolation='bilinear')
        
        # Set ticks and labels
        ax1.set_xticks(np.arange(len(pivot_rmse.columns)))
        ax1.set_yticks(np.arange(len(pivot_rmse.index)))
        ax1.set_xticklabels([f'{x:.3f}' for x in pivot_rmse.columns], rotation=45)
        ax1.set_yticklabels([f'{int(x)}' for x in pivot_rmse.index])
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax1)
        cbar.set_label('CV RMSE', rotation=270, labelpad=15)
        
        # Add best point marker
        best_result = results_df.loc[results_df['cv_rmse_mean'].idxmin()]
        best_x = list(pivot_rmse.columns).index(best_result['learning_rate'])
        best_y = list(pivot_rmse.index).index(best_result['num_leaves'])
        ax1.scatter(best_x, best_y, marker='*', s=500, c='blue', edgecolor='white', linewidth=2)
        
        # Labels and title
        ax1.set_xlabel('Learning Rate', fontsize=12)
        ax1.set_ylabel('Number of Leaves', fontsize=12)
        ax1.set_title('Hyperparameter Grid Search: Cross-Validation RMSE Heatmap\n(★ = Best Parameters)', fontsize=14)
        
        # Add text annotations for top 5 best combinations
        top_5 = results_df.nsmallest(5, 'cv_rmse_mean')
        for _, row in top_5.iterrows():
            x = list(pivot_rmse.columns).index(row['learning_rate'])
            y = list(pivot_rmse.index).index(row['num_leaves'])
            ax1.text(x, y, f'{row["cv_rmse_mean"]:.3f}', ha='center', va='center',
                    fontsize=8, color='white', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.5))
        
        # ===== PLOT 2: Performance vs Complexity Trade-off =====
        ax2 = fig.add_subplot(gs[1, 0])
        
        # Create scatter plot with color gradient
        scatter = ax2.scatter(results_df['num_leaves'], 
                            results_df['cv_rmse_mean'],
                            c=results_df['learning_rate'],
                            s=100, alpha=0.6, cmap='viridis')
        
        # Highlight best point
        ax2.scatter(best_result['num_leaves'], best_result['cv_rmse_mean'],
                   marker='*', s=500, c='red', edgecolor='black', linewidth=2,
                   label='Best Parameters', zorder=5)
        
        # Add trend line
        z = np.polyfit(results_df['num_leaves'], results_df['cv_rmse_mean'], 2)
        p = np.poly1d(z)
        x_trend = np.linspace(results_df['num_leaves'].min(), results_df['num_leaves'].max(), 100)
        ax2.plot(x_trend, p(x_trend), 'r--', alpha=0.5, linewidth=2, label='Trend')
        
        # Colorbar
        cbar2 = plt.colorbar(scatter, ax=ax2)
        cbar2.set_label('Learning Rate', rotation=270, labelpad=15)
        
        # Labels
        ax2.set_xlabel('Number of Leaves (Model Complexity)', fontsize=12)
        ax2.set_ylabel('CV RMSE', fontsize=12)
        ax2.set_title('Performance vs Model Complexity\n(Color = Learning Rate)', fontsize=14)
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        # ===== PLOT 3: Overfitting Analysis =====
        ax3 = fig.add_subplot(gs[1, 1])
        
        # Calculate overfitting score for each configuration
        results_df['overfit_ratio'] = results_df['cv_rmse_mean'] / results_df['train_rmse_mean']
        
        # Create contour plot
        pivot_overfit = results_df.pivot(
            index='num_leaves',
            columns='learning_rate',
            values='overfit_ratio'
        )
        
        X, Y = np.meshgrid(
            np.arange(len(pivot_overfit.columns)),
            np.arange(len(pivot_overfit.index))
        )
        
        # Contour plot
        contour = ax3.contour(X, Y, pivot_overfit.values, levels=15, colors='black', alpha=0.4)
        contourf = ax3.contourf(X, Y, pivot_overfit.values, levels=15, cmap='RdBu_r')
        ax3.clabel(contour, inline=True, fontsize=8)
        
        # Mark regions
        ax3.axhline(y=best_y, color='green', linestyle='--', alpha=0.5)
        ax3.axvline(x=best_x, color='green', linestyle='--', alpha=0.5)
        ax3.scatter(best_x, best_y, marker='*', s=500, c='lime', edgecolor='black', linewidth=2)
        
        # Colorbar
        cbar3 = plt.colorbar(contourf, ax=ax3)
        cbar3.set_label('Overfitting Ratio (CV/Train)', rotation=270, labelpad=15)
        
        # Set ticks and labels
        ax3.set_xticks(np.arange(len(pivot_overfit.columns)))
        ax3.set_yticks(np.arange(len(pivot_overfit.index)))
        ax3.set_xticklabels([f'{x:.3f}' for x in pivot_overfit.columns], rotation=45)
        ax3.set_yticklabels([f'{int(x)}' for x in pivot_overfit.index])
        
        # Labels
        ax3.set_xlabel('Learning Rate', fontsize=12)
        ax3.set_ylabel('Number of Leaves', fontsize=12)
        ax3.set_title('Overfitting Analysis\n(Blue = Underfitting, Red = Overfitting)', fontsize=14)
        
        # Add text box with best parameters
        textstr = f'Best Parameters:\n' \
                 f'num_leaves: {best_result["num_leaves"]}\n' \
                 f'learning_rate: {best_result["learning_rate"]:.4f}\n' \
                 f'CV RMSE: {best_result["cv_rmse_mean"]:.4f}\n' \
                 f'Train RMSE: {best_result["train_rmse_mean"]:.4f}'
        
        fig.text(0.02, 0.02, textstr, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Overall title
        fig.suptitle('LightGBM Hyperparameter Search Results', fontsize=16, y=0.98)
        
        # Save plot
        plot_path = self.results_dir / f'hyperparameter_search_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plots saved to: {plot_path}")
        
        plt.show()
    
    def save_results(self) -> None:
        """Save search results to file."""
        # Save as JSON
        results_dict = {
            'best_params': self.best_params,
            'best_cv_rmse': self.best_score,
            'n_combinations_tested': len(self.search_results),
            'cv_folds': self.cv_folds,
            'timestamp': datetime.now().isoformat(),
            'search_results': self.search_results
        }
        
        # Add test results if available
        if hasattr(self, 'test_results'):
            results_dict['test_results'] = {
                'rmse': self.test_results['rmse'],
                'mae': self.test_results['mae'],
                'r2': self.test_results['r2']
            }
        
        json_path = self.results_dir / f'search_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(json_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"✓ Results saved to: {json_path}")
        
        # Also save as CSV for easy analysis
        results_df = pd.DataFrame(self.search_results)
        results_df['num_leaves'] = results_df['params'].apply(lambda x: x['num_leaves'])
        results_df['learning_rate'] = results_df['params'].apply(lambda x: x['learning_rate'])
        
        csv_path = self.results_dir / f'search_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        results_df[['num_leaves', 'learning_rate', 'cv_rmse_mean', 'cv_rmse_std', 
                   'train_rmse_mean', 'overfit_score']].to_csv(csv_path, index=False)
        
        print(f"✓ CSV results saved to: {csv_path}")
    
    def print_summary(self) -> None:
        """Print summary of search results."""
        print("\n" + "="*60)
        print("HYPERPARAMETER SEARCH SUMMARY")
        print("="*60)
        
        # Convert to DataFrame for analysis
        results_df = pd.DataFrame(self.search_results)
        results_df['num_leaves'] = results_df['params'].apply(lambda x: x['num_leaves'])
        results_df['learning_rate'] = results_df['params'].apply(lambda x: x['learning_rate'])
        
        # Top 5 configurations
        print("\nTop 5 Parameter Combinations:")
        print("-" * 40)
        top_5 = results_df.nsmallest(5, 'cv_rmse_mean')
        for i, row in enumerate(top_5.itertuples(), 1):
            print(f"{i}. num_leaves={row.num_leaves}, lr={row.learning_rate:.4f}")
            print(f"   CV RMSE: {row.cv_rmse_mean:.4f} ± {row.cv_rmse_std:.4f}")
        
        # Parameter sensitivity
        print("\nParameter Sensitivity Analysis:")
        print("-" * 40)
        
        # Correlation with performance
        corr_leaves = results_df[['num_leaves', 'cv_rmse_mean']].corr().iloc[0, 1]
        corr_lr = results_df[['learning_rate', 'cv_rmse_mean']].corr().iloc[0, 1]
        
        print(f"Correlation with CV RMSE:")
        print(f"  num_leaves: {corr_leaves:.3f} {'(negative = more leaves → better)' if corr_leaves < 0 else '(positive = fewer leaves → better)'}")
        print(f"  learning_rate: {corr_lr:.3f} {'(negative = higher lr → better)' if corr_lr < 0 else '(positive = lower lr → better)'}")
        
        # Overfitting analysis
        print("\nOverfitting Analysis:")
        print("-" * 40)
        
        least_overfit = results_df.nsmallest(1, 'overfit_score').iloc[0]
        most_overfit = results_df.nlargest(1, 'overfit_score').iloc[0]
        
        print(f"Least overfitting: num_leaves={least_overfit['num_leaves']}, lr={least_overfit['learning_rate']:.4f}")
        print(f"  Train-Val Gap: {least_overfit['overfit_score']:.4f}")
        
        print(f"Most overfitting: num_leaves={most_overfit['num_leaves']}, lr={most_overfit['learning_rate']:.4f}")
        print(f"  Train-Val Gap: {most_overfit['overfit_score']:.4f}")
        
        print("="*60)


def main():
    """Main entry point for hyperparameter search."""
    parser = argparse.ArgumentParser(
        description="Hyperparameter search for LightGBM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard search with ~25 points
  python scripts/hyperparameter_search.py
  
  # More thorough search with 49 points
  python scripts/hyperparameter_search.py --n-points 49
  
  # Quick search with 2-fold CV
  python scripts/hyperparameter_search.py --cv-folds 2
        """
    )
    
    parser.add_argument(
        '--n-points',
        type=int,
        default=25,
        help='Approximate number of parameter combinations to test (default: 25)'
    )
    
    parser.add_argument(
        '--cv-folds',
        type=int,
        default=3,
        help='Number of cross-validation folds (default: 3)'
    )
    
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Initialize searcher
    searcher = LightGBMHyperparameterSearch(
        n_points=args.n_points,
        cv_folds=args.cv_folds,
        random_state=args.random_state
    )
    
    try:
        # Run search
        searcher.run_search()
        
        # Create visualizations
        searcher.create_plots()
        
        # Save results
        searcher.save_results()
        
        # Print summary
        searcher.print_summary()
        
    except KeyboardInterrupt:
        print("\n⚠️ Search interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Search failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
