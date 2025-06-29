# src/evaluation/__init__.py
"""Model evaluation utilities package."""

from .plotting import (
    plot_predictions_vs_actual,
    plot_residuals,
    plot_qq_plot,
    plot_feature_importance,
    plot_model_comparison,
    plot_error_distribution,
    plot_prediction_intervals,
    plot_learning_curves
)

from .metrics import (
    calculate_additional_metrics,
    calculate_prediction_intervals,
    calculate_cross_validation_metrics,
    create_metrics_report,
    calculate_model_complexity,
    compare_models_statistical_significance
)

__all__ = [
    # Plotting functions
    'plot_predictions_vs_actual',
    'plot_residuals',
    'plot_qq_plot',
    'plot_feature_importance',
    'plot_model_comparison',
    'plot_error_distribution',
    'plot_prediction_intervals',
    'plot_learning_curves',
    
    # Metrics functions
    'calculate_additional_metrics',
    'calculate_prediction_intervals',
    'calculate_cross_validation_metrics',
    'create_metrics_report',
    'calculate_model_complexity',
    'compare_models_statistical_significance'
]