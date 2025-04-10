import numpy as np
from typing import Dict, List, Union
import pandas as pd
import os

def calculate_metrics(y_true: np.ndarray,
                     y_pred: np.ndarray,
                     metrics: List[str] = ['mae', 'rmse', 'mape']) -> Dict[str, float]:
    """
    Calculate various metrics for evaluating forecasts.
    
    Args:
        y_true (np.ndarray): True values
        y_pred (np.ndarray): Predicted values
        metrics (List[str]): List of metrics to calculate
        
    Returns:
        Dict[str, float]: Dictionary of metric values
    """
    results = {}
    
    for metric in metrics:
        if metric == 'mae':
            results['mae'] = np.mean(np.abs(y_true - y_pred))
        elif metric == 'rmse':
            results['rmse'] = np.sqrt(np.mean((y_true - y_pred)**2))
        elif metric == 'mape':
            results['mape'] = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
        elif metric == 'r2':
            results['r2'] = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)
            
    return results

def print_metrics(metrics: Dict[str, float], prefix: str = '') -> None:
    """
    Print metrics in a formatted way.
    
    Args:
        metrics (Dict[str, float]): Dictionary of metric values
        prefix (str): Prefix to add to each metric name
    """
    for metric, value in metrics.items():
        if metric == 'mape':
            print(f"{prefix}{metric.upper()}: {value:.2f}%")
        else:
            print(f"{prefix}{metric.upper()}: {value:.2f}")

def create_time_features(df: pd.DataFrame,
                        date_col: str = 'Date') -> pd.DataFrame:
    """
    Create time-based features from a date column.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        date_col (str): Name of the date column
        
    Returns:
        pd.DataFrame: DataFrame with added time features
    """
    df = df.copy()
    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month
    df['day'] = df[date_col].dt.day
    df['dayofweek'] = df[date_col].dt.dayofweek
    df['quarter'] = df[date_col].dt.quarter
    df['dayofyear'] = df[date_col].dt.dayofyear
    
    return df

def check_hierarchy_violations(df: pd.DataFrame,
                             hierarchy: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Check for violations in the hierarchical relationships.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        hierarchy (Dict[str, List[str]]): Dictionary defining the hierarchy
        
    Returns:
        pd.DataFrame: DataFrame containing violation checks
    """
    violations = {}
    
    for parent, children in hierarchy.items():
        # Calculate the sum of children
        children_sum = df[children].sum(axis=1)
        # Calculate the difference with parent
        violations[f"{parent}_diff"] = df[parent] - children_sum
        
    return pd.DataFrame(violations)

def create_lagged_features(df: pd.DataFrame,
                         columns: List[str],
                         lags: List[int]) -> pd.DataFrame:
    """
    Create lagged features for specified columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (List[str]): Columns to create lags for
        lags (List[int]): List of lag values
        
    Returns:
        pd.DataFrame: DataFrame with added lagged features
    """
    df = df.copy()
    
    for col in columns:
        for lag in lags:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)
            
    return df

def prepare_forecast_data(df: pd.DataFrame,
                        target_col: str,
                        feature_cols: List[str],
                        n_lags: int = 12) -> tuple:
    """
    Prepare data for forecasting by creating lagged features.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        target_col (str): Target column name
        feature_cols (List[str]): List of feature columns
        n_lags (int): Number of lags to create
        
    Returns:
        tuple: (X, y) arrays for training
    """
    # Create lagged features
    df_lagged = create_lagged_features(df, feature_cols, list(range(1, n_lags + 1)))
    
    # Drop rows with NaN values
    df_lagged = df_lagged.dropna()
    
    # Prepare X and y
    X = df_lagged[[f"{col}_lag_{lag}" for col in feature_cols for lag in range(1, n_lags + 1)]].values
    y = df_lagged[target_col].values
    
    return X, y

def export_predictions_to_excel(
    base_forecasts: Dict[str, np.ndarray],
    reconciled_forecasts: Dict[str, np.ndarray],
    actual_values: Dict[str, np.ndarray],
    dates: pd.DatetimeIndex,
    output_dir: str = 'output',
    filename: str = 'forecasts.xlsx'
) -> None:
    """
    Export predictions and actual values to an Excel file with separate sheets.
    
    Args:
        base_forecasts: Dictionary of base forecasts for each series
        reconciled_forecasts: Dictionary of reconciled forecasts for each series
        actual_values: Dictionary of actual values for each series
        dates: Dates corresponding to the predictions
        output_dir: Directory to save the Excel file
        filename: Name of the Excel file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create DataFrames for each forecast type
    base_df = pd.DataFrame(index=dates)
    reconciled_df = pd.DataFrame(index=dates)
    
    # Add predictions and actual values to each DataFrame
    for series_name in base_forecasts.keys():
        base_df[f'{series_name}_predicted'] = base_forecasts[series_name]
        base_df[f'{series_name}_actual'] = actual_values[series_name]
        reconciled_df[f'{series_name}_predicted'] = reconciled_forecasts[series_name]
        reconciled_df[f'{series_name}_actual'] = actual_values[series_name]
    
    # Save to Excel with multiple sheets
    output_path = os.path.join(output_dir, filename)
    with pd.ExcelWriter(output_path) as writer:
        base_df.to_excel(writer, sheet_name='Base Forecasts')
        reconciled_df.to_excel(writer, sheet_name='Reconciled Forecasts')
    
    print(f"Forecasts saved to {output_path}")

def export_metrics_to_csv(metrics: Dict[str, Dict[str, float]], output_dir: str = 'output'):
    """
    Export metrics to a CSV file, focusing only on EBIT metrics.
    
    Args:
        metrics: Dictionary of metrics for each series and model
        output_dir: Directory to save the CSV file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert metrics to DataFrame, only for EBIT
    rows = []
    for model, metrics_values in metrics.get('EBIT', {}).items():
        row = {
            'model': model,
            'mae': metrics_values.get('mae', np.nan),
            'rmse': metrics_values.get('rmse', np.nan),
            'mape': metrics_values.get('mape', np.nan)
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Save to CSV
    output_path = os.path.join(output_dir, 'metrics.csv')
    df.to_csv(output_path, index=False)
    print(f"Metrics saved to {output_path}")

def export_reconciliation_results(
    base_forecasts: Dict[str, np.ndarray],
    reconciled_forecasts: Dict[str, np.ndarray],
    actual_values: Dict[str, np.ndarray],
    dates: pd.DatetimeIndex,
    metrics: Dict[str, Dict[str, Dict[str, float]]],
    output_dir: str = 'output'
):
    """
    Export all reconciliation results to files.
    
    Args:
        base_forecasts: Dictionary of base forecasts for each series
        reconciled_forecasts: Dictionary of reconciled forecasts for each series
        actual_values: Dictionary of actual values for each series
        dates: DatetimeIndex for the forecasts
        metrics: Dictionary of metrics for each series and model
        output_dir: Directory to save the files
    """
    # Export forecasts to Excel
    export_predictions_to_excel(
        base_forecasts,
        reconciled_forecasts,
        actual_values,
        dates,
        output_dir
    )
    
    # Export metrics to CSV
    export_metrics_to_csv(metrics, output_dir) 