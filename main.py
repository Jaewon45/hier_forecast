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
from darts.models import (
    LinearRegressionModel,
    ExponentialSmoothing,
    Prophet,
    AutoARIMA
)

def main():
    # Configuration
    DATA_PATH = 'data/SampleHierForecastingBASF_share.xlsx'
    OUTPUT_DIR = 'output'
    FORECAST_HORIZONS = [1, 3, 6, 12]
    
    # Initialize data loader
    print("Loading and preprocessing data...")
    data_loader = DataLoader(DATA_PATH)
    
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
    print("\nInitializing forecasting models...")
    models = [
        LinearRegressionModel(lags=12),
        ExponentialSmoothing(),
        Prophet(),
        AutoARIMA()
    ]
    
    # Initialize forecaster
    forecaster = ForecastingModels()
    
    # Dictionary to store all metrics
    all_metrics = {}
    
    # Train and evaluate each model
    print("\nTraining and evaluating models...")
    for model in models:
        model_name = str(model).split('(')[0]
        print(f"\nTraining {model_name}...")
        
        # Generate base forecasts
        base_forecasts = forecaster.forecast_all_series(
            {name: train_series[i] for i, name in enumerate(bottom_series)},
            n_periods=len(val),
            model=model
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
        
        # Store metrics
        all_metrics[model_name] = {
            'EBIT': {
                'mae': metrics['mae'],
                'rmse': metrics['rmse'],
                'mape': metrics['mape']
            }
        }
        
        # Plot results
        Visualization.plot_hierarchical_forecasts(
            {name: val_series[i] for i, name in enumerate(bottom_series)},
            {name: reconciled_forecasts[i] for i, name in enumerate(bottom_series)},
            f"{model_name} Forecasts"
        )
    
    # Export results
    print("\nExporting results...")
    export_reconciliation_results(
        base_forecasts=base_forecasts,
        reconciled_forecasts=reconciled_dict,
        actual_values={name: val_series[i] for i, name in enumerate(bottom_series)},
        dates=val['Date'],
        metrics=all_metrics,
        output_dir=OUTPUT_DIR
    )
    
    print("\nAnalysis complete! Results saved to the output directory.")

if __name__ == "__main__":
    main() 