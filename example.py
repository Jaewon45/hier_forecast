from reconciliation import (
    DataLoader,
    MatrixReconciliation,
    ForecastingModels,
    Visualization,
    calculate_metrics,
    export_reconciliation_results
)
import numpy as np
import pandas as pd

def main():
    # Initialize data loader
    data_loader = DataLoader('SampleHierForecastingBASF_share.xlsx')
    
    # Load and preprocess data
    df = data_loader.load_data()
    
    # Define hierarchy
    hierarchy = data_loader.define_hierarchy()
    
    # Split data
    train, val, test = data_loader.split_data()
    
    # Get bottom level series
    bottom_series = data_loader.get_bottom_level_series()
    
    # Prepare time series data
    train_series = data_loader.prepare_time_series(train, bottom_series)
    val_series = data_loader.prepare_time_series(val, bottom_series)
    
    # Initialize forecasting models
    forecaster = ForecastingModels()
    
    # Generate base forecasts
    base_forecasts = forecaster.forecast_all_series(
        {name: train_series[i] for i, name in enumerate(bottom_series)},
        n_periods=len(val),
        method='linear_regression'
    )
    
    # Initialize matrix reconciliation
    reconciler = MatrixReconciliation(hierarchy, method='bottom_up')
    
    # Create summing matrix
    S = reconciler.create_summing_matrix()
    
    # Create reconciliation matrix
    G = reconciler.create_reconciliation_matrix()
    
    # Reconcile forecasts
    reconciled_forecasts = reconciler.reconcile_forecasts(
        np.array([base_forecasts[name] for name in bottom_series])
    )
    
    # Convert reconciled forecasts to dictionary format
    reconciled_dict = {
        name: reconciled_forecasts[i] 
        for i, name in enumerate(bottom_series)
    }
    
    # Evaluate reconciliation
    metrics = reconciler.evaluate_reconciliation(
        np.array([base_forecasts[name] for name in bottom_series]),
        val_series
    )
    
    # Convert metrics to dictionary format
    metrics_dict = {
        name: {
            'mae': metrics['mae'],
            'rmse': metrics['rmse'],
            'mape': metrics['mape']
        }
        for name in bottom_series
    }
    
    # Export results to CSV files
    export_reconciliation_results(
        base_forecasts=base_forecasts,
        reconciled_forecasts=reconciled_dict,
        actual_values={name: val_series[i] for i, name in enumerate(bottom_series)},
        dates=val['Date'],
        metrics=metrics_dict,
        output_dir='output'
    )
    
    # Print metrics
    print("Reconciliation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric.upper()}: {value:.2f}")
    
    # Plot results
    Visualization.plot_hierarchical_forecasts(
        {name: val_series[i] for i, name in enumerate(bottom_series)},
        {name: reconciled_forecasts[i] for i, name in enumerate(bottom_series)},
        "Hierarchical Forecasts"
    )
    
    # Check hierarchy violations
    violations = check_hierarchy_violations(
        pd.DataFrame(reconciled_forecasts.T, columns=bottom_series),
        hierarchy
    )
    
    Visualization.plot_hierarchy_violations(
        violations,
        "Hierarchy Violations"
    )

if __name__ == "__main__":
    main() 