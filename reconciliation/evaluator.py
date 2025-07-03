import pandas as pd
from typing import Dict, List, Union
import numpy as np
import os
import matplotlib as plt

class ModelEvaluator:
    """
    A class for exporting and plotting experiment results.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path

    def export_predictions_to_excel(self,
        base_forecasts: Dict[str, np.ndarray],
        reconciled_forecasts: Dict[str, np.ndarray],
        actual_values: Dict[str, np.ndarray],
        dates: pd.DatetimeIndex,
        output_dir: str = 'output',
        filename: str = 'forecasts.xlsx') -> None:
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
            #change this!!!
            reconciled_df.to_excel(writer, sheet_name='Reconciled Forecasts')
        
        print(f"Forecasts saved to {output_path}")

    def export_metrics_to_csv(self, metrics: Dict[str, Dict[str, float]], output_dir: str = 'output'):
        """
        Export metrics to a CSV file, focusing only on EBIT metrics.
        
        Args:
            metrics: Dictionary of metrics for each series and model
            output_dir: Directory to save the CSV file
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save to CSV
        output_path = os.path.join(output_dir, 'metrics.csv')
        metrics.to_csv(output_path, index=False)
        print(f"Metrics saved to {output_path}")

    def export_reconciliation_results(self,
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

    def get_metrics_from_reconciliation(self,base_forecasts_array, reconciliated_forecasts_array, actual_values,model_name):
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
                metrics_data[method[0]]={
                    'Model': model_name,
                    'Method':method[0],
                    'MAE': np.mean(np.abs(actual_values[0] - method[1])),
                    'RMSE': np.sqrt(np.mean((actual_values[0] - method[1]) ** 2)),
                    'MAPE': np.mean(np.abs((actual_values[0] - method[1]) / actual_values[0])) * 100,
                }
            
        # Convert metrics to DataFrame and save
        metrics_df = pd.DataFrame.from_dict(metrics_data, orient='index')
        metrics_df.reset_index(inplace=True)
        metrics_df.rename(columns={'index': 'Type'}, inplace=True)
        metrics_df = metrics_df[['Model', 'Method', 'MAE', 'RMSE', 'MAPE']]
        self.export_metrics_to_csv(metrics_df)
        return metrics_df

    def plot_reconciliation(self, base_forecasts, reconciliated_forecasts, actual_values, model_name, metrics_df):
            """
            Plot the base forecasts, bottom-up reconciled forecasts, top-down reconciled forecasts, and actual values.
            
            Args:
                base_forecasts: Base forecasts for all series
                bottom_up_forecasts: Bottom-up reconciled forecasts
                top_down_forecasts: Top-down reconciled forecasts
                actual_values: Actual values (can be list or numpy array)
                model_name: Name of the model for output directory
            """
            # Create output directory if it doesn't exist
            output_dir = os.path.join('output', model_name)
            os.makedirs(output_dir, exist_ok=True)
            
            # Convert inputs to numpy arrays if they are lists
            if isinstance(actual_values, list):
                actual_values = np.array(actual_values)
            if isinstance(base_forecasts, list):
                base_forecasts = np.array(base_forecasts)
            if isinstance(bottom_up_forecasts, list):
                bottom_up_forecasts = np.array(bottom_up_forecasts)
            if isinstance(top_down_forecasts, list):
                top_down_forecasts = np.array(top_down_forecasts)
            
            # Get the number of series from the actual values shape
            n_series = actual_values.shape[0]
            
            # Create a figure with subplots for each series
            fig, axes = plt.subplots(n_series, 1, figsize=(12, 4*n_series))
            if n_series == 1:
                axes = [axes]
            
            # Get series names based on the actual number of series
            series_names = ['EBIT', 'ContributionMargin1', 'EBITDA', 'NetSales', '-VariableCosts', '-FixCosts', '-DepreciationAmortization']
            
            # Plot each series
            for i, (ax, series_name) in enumerate(zip(axes, series_names[:n_series])):
                # Plot base forecasts
                ax.plot(base_forecasts[i], label='Base Forecast', color='blue', linestyle='--')
                ax.plot(actual_values[i], label='Actual', color='red')
                
                for method in reconciliated_forecasts:
                    temp = method[1]
                    ax.plot(temp[i], label=method[0], linestyle='-')
                
                ax.set_title(f'{series_name} Forecasts')
                ax.set_xlabel('Time')
                ax.set_ylabel('Value')
                ax.legend()
                ax.grid(True)
            
            plt.tight_layout()
            
            # Save the plot
            plt.savefig(os.path.join(output_dir, 'forecasts.png'), dpi=300)
            plt.close()     

    def evaluate_reconciliation(self, base_forecasts, actual_values, reconciliated_forecasts, metrics, reconciliation_methods, model_name):
        """
        Evaluate reconciliation performance and save results.
        
        Args:
            base_forecasts: Base forecasts array
            actual_values: Actual values array
            model_name: Name of the model being evaluated
            
        Returns:
            Dictionary containing metrics for EBIT
        """
        print(f"Starting evaluation for {model_name}...")
        print(f"Base forecasts shape: {base_forecasts.shape}")
        print(f"Actual values shape: {actual_values.shape}")
        
        for reconciler in reconciliation_methods:
            print(f"Getting {reconciler[0]} reconciled forecasts")
        
        # Calculate metrics for EBIT (first series)
        print("Calculating metrics for EBIT...")
        
        # Create output directory
        output_dir = f"output/{model_name}"
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        
        series_names = ['EBIT', 'ContributionMargin1', 'EBITDA', 'NetSales', '-VariableCosts', '-FixCosts', '-DepreciationAmortization']
        
        # Save all forecasts and metrics to Excel
        print("Saving forecasts and metrics to Excel...")
        with pd.ExcelWriter(f"{output_dir}/forecasts.xlsx") as writer:
            # Save base forecasts (using only the available series)
            pd.DataFrame(base_forecasts.T, columns=series_names[:base_forecasts.shape[0]]).to_excel(
                writer, sheet_name='Base_Forecasts', index=False
            )
            pd.DataFrame(actual_values.T, columns=series_names[:actual_values.shape[0]]).to_excel(
                writer, sheet_name='Actual_Values', index=False
            )
            for fc in reconciliated_forecasts:
                pd.DataFrame(fc[1].T, columns=series_names[:fc[1].shape[0]]).to_excel(
                writer, sheet_name=fc[0], index=False)

            pd.DataFrame(actual_values.T, columns=series_names[:actual_values.shape[0]]).to_excel(
                writer, sheet_name='Actual_Values', index=False
            )
        
        # Plot all forecasts
        print("Generating forecast plots...")
        self.plot_reconciliation(
            base_forecasts,
            reconciliated_forecasts,
            actual_values,
            model_name,
            metrics
        )
        
        print(f"Completed evaluation for {model_name}")
