import numpy as np
from typing import List, Dict, Tuple, Optional
import pandas as pd
from scipy.linalg import block_diag
import os
import matplotlib.pyplot as plt
import logging
import warnings

# Suppress warnings and logging messages
warnings.filterwarnings('ignore')
logging.getLogger('prophet').setLevel(logging.ERROR)
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)

class MatrixReconciliation:
    """
    A class for implementing matrix-based hierarchical reconciliation of forecasts.
    This implementation is independent of Darts and uses pure NumPy/Pandas operations.
    """
    
    def __init__(self, hierarchy: Dict[str, List[str]], method: str = 'bottom_up'):
        """
        Initialize the MatrixReconciliation class.
        
        Args:
            hierarchy (Dict[str, List[str]]): Dictionary defining the hierarchical structure.
                Keys are parent nodes, values are lists of child nodes.
            method (str): Reconciliation method to use. Options: 'bottom_up', 'top_down', 'mint'
        """
        self.hierarchy = hierarchy
        self.method = method
        self.S = None  # Summing matrix
        self.G = None  # Reconciliation matrix
        
    def create_summing_matrix(self) -> np.ndarray:
        """
        Create the summing matrix S that defines the hierarchical relationships.
        Handles two separate paths (ContributionMargin1 and EBITDA) and combines them into a block diagonal matrix.
        
        Returns:
            np.ndarray: The block diagonal summing matrix S
        """
        # Path 1: ContributionMargin1 hierarchy
        m1 = 3  # NetSales, -VariableCosts, -FixCosts
        n1 = 4  # EBIT + ContributionMargin1 + Bottom level (2)

        S1 = np.zeros((n1, m1))
        # EBIT = NetSales - VariableCosts - FixCosts
        S1[0] = np.array([1, 1, 1])
        # ContributionMargin1 = NetSales - VariableCosts
        S1[1] = np.array([1, 1, 0])
        # Bottom level identity matrix
        S1[2] = np.array([1, 0, 0])  # NetSales
        S1[3] = np.array([0, 1, 0])  # -VariableCosts

        # Path 2: EBITDA hierarchy
        m2 = 2  # EBITDA, -DepreciationAmortization
        n2 = 3  # EBIT + Bottom level (2)

        S2 = np.zeros((n2, m2))
        # EBIT = EBITDA - DepreciationAmortization
        S2[0] = np.array([1, 1])
        # Bottom level identity matrix
        S2[1] = np.array([1, 0])   # EBITDA
        S2[2] = np.array([0, 1])   # -DepreciationAmortization

        # Create block diagonal matrix combining both paths
        S_block = block_diag(S1, S2)
        self.S = S_block
        return S_block
    
    def create_reconciliation_matrix(self, method: Optional[str] = None) -> np.ndarray:
        """
        Create the reconciliation matrix G based on the chosen method.
        
        Args:
            method (str, optional): Override the default method for this call
            
        Returns:
            np.ndarray: The reconciliation matrix G
        """
        if method is None:
            method = self.method
            
        if self.S is None:
            self.create_summing_matrix()
            
        n_bottom = self.S.shape[1]  # Number of bottom level nodes
        n_total = self.S.shape[0]   # Total number of nodes
            
        if method == 'bottom_up':
            # For bottom-up, G is a zero matrix with identity in the bottom-right corner
            G = np.zeros((n_bottom, n_total))
            G[:, -n_bottom:] = np.eye(n_bottom)
            
        elif method == 'top_down':
            # For top-down, we need historical proportions
            G = np.zeros((n_bottom, n_total))
            
        elif method == 'mint':
            # For MinT, we need covariance information
            G = np.zeros((n_bottom, n_total))
            
        else:
            raise ValueError(f"Unknown reconciliation method: {method}")
            
        self.G = G
        return G
    
    def reconcile_forecasts(self, base_forecasts: np.ndarray) -> np.ndarray:
        """
        Reconcile base forecasts using the chosen method.
        
        Args:
            base_forecasts (np.ndarray): Base forecasts for bottom level series
            
        Returns:
            np.ndarray: Reconciled forecasts for all levels
        """
        if self.S is None:
            self.create_summing_matrix()
            
        # Ensure base_forecasts has the right shape
        if base_forecasts.ndim == 1:
            base_forecasts = base_forecasts.reshape(-1, 1)
            
        # Get dimensions
        n_bottom = self.S.shape[1]  # Number of bottom level nodes
        
        # Ensure base_forecasts has correct number of rows
        if base_forecasts.shape[0] != n_bottom:
            raise ValueError(f"Base forecasts must have {n_bottom} rows (bottom level nodes), got {base_forecasts.shape[0]}")
            
        # Apply reconciliation: y_hat = S * y_base
        reconciled = self.S @ base_forecasts
        
        return reconciled
    
    def evaluate_reconciliation(self, 
                              base_forecasts: np.ndarray, 
                              actual_values: np.ndarray,
                              model_name: str = 'default',
                              metrics: List[str] = ['mae', 'rmse', 'mape']) -> Dict[str, float]:
        """
        Evaluate the reconciliation performance using specified metrics.
        Saves all forecasts but only evaluates metrics for EBIT.
        
        Args:
            base_forecasts (np.ndarray): Base forecasts
            actual_values (np.ndarray): Actual values
            model_name (str): Name of the model/predictor
            metrics (List[str]): List of metrics to compute
            
        Returns:
            Dict[str, float]: Dictionary of metric values for EBIT only
        """
        reconciled = self.reconcile_forecasts(base_forecasts)
        
        # Extract only EBIT forecasts (first row of reconciled forecasts)
        reconciled_ebit = reconciled[0]  # First row is EBIT
        actual_ebit = actual_values[0]   # First row is EBIT
        
        # Calculate metrics only for EBIT
        results = {}
        for metric in metrics:
            if metric == 'mae':
                results['mae'] = np.mean(np.abs(reconciled_ebit - actual_ebit))
            elif metric == 'rmse':
                results['rmse'] = np.sqrt(np.mean((reconciled_ebit - actual_ebit)**2))
            elif metric == 'mape':
                results['mape'] = np.mean(np.abs((reconciled_ebit - actual_ebit) / actual_ebit)) * 100
                
        # Save all forecasts and metrics to Excel file
        output_dir = os.path.join('output', model_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Create Excel file with multiple sheets
        excel_path = os.path.join(output_dir, 'forecasts.xlsx')
        with pd.ExcelWriter(excel_path) as writer:
            # Save all base forecasts
            pd.DataFrame(base_forecasts).to_excel(writer, sheet_name='Base_Forecasts', index=False)
            
            # Save all reconciled forecasts
            pd.DataFrame(reconciled).to_excel(writer, sheet_name='Reconciled_Forecasts', index=False)
            
            # Save all actual values
            pd.DataFrame(actual_values).to_excel(writer, sheet_name='Actual_Values', index=False)
            
            # Add EBIT metrics to a separate sheet
            metrics_df = pd.DataFrame([results])
            metrics_df.to_excel(writer, sheet_name='EBIT_Metrics', index=False)
        
        return results
    
    def plot_reconciliation(self, base_forecasts, reconciled_forecasts, actual_values, model_name):
        """
        Plot the base forecasts, reconciled forecasts, and actual values.
        
        Args:
            base_forecasts: Base forecasts for all series
            reconciled_forecasts: Reconciled forecasts
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
        if isinstance(reconciled_forecasts, list):
            reconciled_forecasts = np.array(reconciled_forecasts)
        
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
            
            # Plot reconciled forecasts using the specified method
            ax.plot(reconciled_forecasts[i], label=f'Reconciled ({self.method})', color='green', linestyle='-')
            
            # Plot actual values
            ax.plot(actual_values[i], label='Actual', color='red')
            
            ax.set_title(f'{series_name} Forecasts')
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(output_dir, 'forecasts.png'), dpi=300)
        plt.close()
        
        # Save forecasts to Excel
        excel_path = os.path.join(output_dir, 'forecasts.xlsx')
        with pd.ExcelWriter(excel_path) as writer:
            # Save base forecasts
            base_df = pd.DataFrame(base_forecasts.T, columns=series_names[:base_forecasts.shape[0]])
            base_df.to_excel(writer, sheet_name='Base_Forecasts', index=False)
            
            # Save reconciled forecasts
            reconciled_df = pd.DataFrame(reconciled_forecasts.T, columns=series_names[:reconciled_forecasts.shape[0]])
            reconciled_df.to_excel(writer, sheet_name='Reconciled_Forecasts', index=False)
            
            # Save actual values
            actual_df = pd.DataFrame(actual_values.T, columns=series_names[:actual_values.shape[0]])
            actual_df.to_excel(writer, sheet_name='Actual_Values', index=False)
            
            # Calculate metrics only for EBIT
            metrics_data = {
                'Base': {
                    'Model': model_name,
                    'Method': 'Base',
                    'MAE': np.mean(np.abs(actual_values[0] - base_forecasts[0])),
                    'RMSE': np.sqrt(np.mean((actual_values[0] - base_forecasts[0]) ** 2)),
                    'MAPE': np.mean(np.abs((actual_values[0] - base_forecasts[0]) / actual_values[0])) * 100
                },
                'Reconciled': {
                    'Model': model_name,
                    'Method': self.method,
                    'MAE': np.mean(np.abs(actual_values[0] - reconciled_forecasts[0])),
                    'RMSE': np.sqrt(np.mean((actual_values[0] - reconciled_forecasts[0]) ** 2)),
                    'MAPE': np.mean(np.abs((actual_values[0] - reconciled_forecasts[0]) / actual_values[0])) * 100
                }
            }
            
            # Convert metrics to DataFrame
            metrics_df = pd.DataFrame.from_dict(metrics_data, orient='index')
            metrics_df.reset_index(inplace=True)
            metrics_df.rename(columns={'index': 'Type'}, inplace=True)
            metrics_df = metrics_df[['Model', 'Method', 'MAE', 'RMSE', 'MAPE']]
            
            # Save metrics to Excel
            metrics_df.to_excel(writer, sheet_name='EBIT_Metrics', index=False)
            
            # Save metrics to the main output directory
            main_metrics_path = os.path.join('output', 'all_models_metrics.xlsx')
            if os.path.exists(main_metrics_path):
                # If file exists, append to it
                with pd.ExcelWriter(main_metrics_path, mode='a', if_sheet_exists='overlay') as main_writer:
                    metrics_df.to_excel(main_writer, sheet_name='Metrics', startrow=main_writer.sheets['Metrics'].max_row, index=False, header=False)
            else:
                # If file doesn't exist, create it
                with pd.ExcelWriter(main_metrics_path) as main_writer:
                    metrics_df.to_excel(main_writer, sheet_name='Metrics', index=False) 