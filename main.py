from reconciliation import (
    DataLoader,
    MatrixReconciliation,
    ForecastingModels,
    Visualization,
    ModelEvaluator,
    calculate_metrics,
    export_reconciliation_results
)
from darts.dataprocessing.transformers import (
    MinTReconciliator, BottomUpReconciliator, TopDownReconciliator
)
import numpy as np
import pandas as pd
from darts.models import (
    LinearRegressionModel,
    ExponentialSmoothing,
    Prophet,
    AutoARIMA
)
import os

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
    
    # Define reconciliation methods to test (only Bottom-Up and Top-Down)
    RECONCILIATION_METHODS = [
        ('matrix_bottom_up',MatrixReconciliation(hierarchy, method='bottom_up')), 
        ('matrix_top_down',MatrixReconciliation(hierarchy, method='top_down')),
    #    ('darts_bottom_up',BottomUpReconciliator()), 
      #  ('darts_top_down',TopDownReconciliator()),
       # ('darts_MiNT',MinTReconciliator(method="ols")),
        ]

    # Initialize forecasting models
    print("\nInitializing forecasting models...")
    models = [
        ('LinearRegression', LinearRegressionModel(lags=12)),
        ('ExponentialSmoothing', ExponentialSmoothing()),
        ('Prophet', Prophet()),
        ('AutoARIMA', AutoARIMA())
    ]
    
    # Initialize forecaster
    forecaster = ForecastingModels()
    
    # Dictionary to store all metrics
    all_metrics = {}
    
    # Train and evaluate each model
    print("\nTraining and evaluating models...")
    for model_name, model in models:
        print(f"\nProcessing {model_name}...")
        
        # Generate base forecasts
        base_forecasts = forecaster.forecast_all_series(
            {name: train_series[i] for i, name in enumerate(bottom_series)},
            n_periods=len(val),
            model=model
        )
        base_forecasts_array = np.array([base_forecasts[name] for name in bottom_series])
        actual_values = np.array(val_series)
        
        # Get reconciled forecasts
        reconciliated_forecasts_array = []
        for method in RECONCILIATION_METHODS:
                reconciliated_forecasts_array.append((method[0], method[1].transform(base_forecasts_array)))
        print("Evaluating...")
    metrics_data = {
                'Base': {
                    'Model': model_name,
                    'Method': 'Base',
                    'MAE': np.mean(np.abs(actual_values[0] - base_forecasts_array[0])),
                    'RMSE': np.sqrt(np.mean((actual_values[0] - base_forecasts_array[0]) ** 2)),
                    'MAPE': np.mean(np.abs((actual_values[0] - base_forecasts_array[0]) / actual_values[0])) * 100
                }
            }   
    
    for method in reconciliated_forecasts_array:
         temp= method[1]
         metrics_data[method[0]]={
                    'Model': model_name,
                    'Method': method[0],
                    'MAE': np.mean(np.abs(actual_values[0] - temp[0])),
                    'RMSE': np.sqrt(np.mean((actual_values[0] - temp[0]) ** 2)),
                    'MAPE': np.mean(np.abs((actual_values[0] - temp[0]) / actual_values[0])) * 100
                }
    
    metrics_df = pd.DataFrame.from_dict(metrics_data, orient='index')
    metrics_df.reset_index(inplace=True)
    metrics_df.rename(columns={'index': 'Type'}, inplace=True)
    metrics_df = metrics_df[['Model', 'Method', 'MAE', 'RMSE', 'MAPE']]

    main_metrics_path = os.path.join('output', 'all_models_metrics.xlsx')
    if os.path.exists(main_metrics_path):
        with pd.ExcelWriter(main_metrics_path, mode='a', if_sheet_exists='overlay') as writer:
            metrics_df.to_excel(writer, sheet_name='Metrics', startrow=writer.sheets['Metrics'].max_row, index=False, header=False)
    else:
        with pd.ExcelWriter(main_metrics_path) as writer:
            metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
        
    model_output_dir = os.path.join(OUTPUT_DIR, model_name)
    os.makedirs(model_output_dir, exist_ok=True)
        
    bottom_up_reconciler.evaluate_reconciliation(
        base_forecasts_array,
        actual_values,
        model_name)

        
    print("\nAnalysis complete! Results saved to the output directory.")

if __name__ == "__main__":
    main() 