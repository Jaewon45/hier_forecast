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
    
    def create_reconciliation_matrix(self) -> None:
        """
        Create the reconciliation matrix G based on the chosen method.
        For bottom-up: G is a zero matrix with identity in the bottom-right corner
        For top-down: G is a zero matrix with proportions in the appropriate positions
        """
        # Get dimensions
        n_total = self.S.shape[0]  # Total number of nodes
        n_bottom = self.S.shape[1]  # Number of bottom level nodes
        
        # Initialize G matrix
        self.G = np.zeros((n_total, n_bottom))
        
        if self.method == 'bottom_up':
            # For bottom-up: G is zero matrix with identity in bottom-right corner
            self.G[-n_bottom:, -n_bottom:] = np.eye(n_bottom)
        elif self.method == 'top_down':
            # For top-down: G is zero matrix with proportions
            # The proportions should be based on historical data
            # For now, we'll use equal proportions as a placeholder
            # TODO: Replace with actual historical proportions
            proportions = np.ones(n_bottom) / n_bottom
            self.G[0, :] = proportions  # Top level gets all proportions
        else:
            raise ValueError(f"Unknown reconciliation method: {self.method}")
    
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
            
        if self.G is None:
            self.create_reconciliation_matrix()
            
        # Ensure base_forecasts has the right shape
        if base_forecasts.ndim == 1:
            base_forecasts = base_forecasts.reshape(-1, 1)
            
        # Get dimensions
        n_bottom = self.S.shape[1]  # Number of bottom level nodes
        
        # Ensure base_forecasts has correct number of rows
        if base_forecasts.shape[0] != n_bottom:
            raise ValueError(f"Base forecasts must have {n_bottom} rows (bottom level nodes), got {base_forecasts.shape[0]}")
            
        # Apply reconciliation based on method
        if self.method == 'bottom_up':
            # For bottom-up: y_hat = S * y_base
            reconciled = self.S @ base_forecasts
        elif self.method == 'top_down':
            # For top-down: y_hat = G * y_base
            reconciled = self.G @ base_forecasts
        else:
            raise ValueError(f"Unknown reconciliation method: {self.method}")
        
        return reconciled
    
    def evaluate_reconciliation(self, base_forecasts, actual_values, model_name):
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
        
        # Create separate reconcilers for bottom-up and top-down
        bottom_up_reconciler = MatrixReconciliation(self.hierarchy, method='bottom_up')
        top_down_reconciler = MatrixReconciliation(self.hierarchy, method='top_down')
        
        # Get reconciled forecasts
        print("Getting bottom-up reconciled forecasts...")
        bottom_up_forecasts = bottom_up_reconciler.reconcile_forecasts(base_forecasts)
        print(f"Bottom-up forecasts shape: {bottom_up_forecasts.shape}")
        
        print("Getting top-down reconciled forecasts...")
        top_down_forecasts = top_down_reconciler.reconcile_forecasts(base_forecasts)
        print(f"Top-down forecasts shape: {top_down_forecasts.shape}")
        
        # Calculate metrics for EBIT (first series)
        print("Calculating metrics for EBIT...")
        results = {
            'mae_bu': np.mean(np.abs(actual_values[0] - bottom_up_forecasts[0])),
            'mae_td': np.mean(np.abs(actual_values[0] - top_down_forecasts[0])),
            'rmse_bu': np.sqrt(np.mean((actual_values[0] - bottom_up_forecasts[0]) ** 2)),
            'rmse_td': np.sqrt(np.mean((actual_values[0] - top_down_forecasts[0]) ** 2)),
            'mape_bu': np.mean(np.abs((actual_values[0] - bottom_up_forecasts[0]) / actual_values[0])) * 100,
            'mape_td': np.mean(np.abs((actual_values[0] - top_down_forecasts[0]) / actual_values[0])) * 100
        }
        
        # Create output directory
        output_dir = f"output/{model_name}"
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Define series names based on the actual number of series
        series_names = ['EBIT', 'ContributionMargin1', 'EBITDA', 'NetSales', '-VariableCosts', '-FixCosts', '-DepreciationAmortization']
        
        # Save all forecasts and metrics to Excel
        print("Saving forecasts and metrics to Excel...")
        with pd.ExcelWriter(f"{output_dir}/forecasts.xlsx") as writer:
            # Save base forecasts (using only the available series)
            pd.DataFrame(base_forecasts.T, columns=series_names[:base_forecasts.shape[0]]).to_excel(
                writer, sheet_name='Base_Forecasts', index=False
            )
            
            # Save bottom-up reconciled forecasts
            pd.DataFrame(bottom_up_forecasts.T, columns=series_names[:bottom_up_forecasts.shape[0]]).to_excel(
                writer, sheet_name='BottomUp_Forecasts', index=False
            )
            
            # Save top-down reconciled forecasts
            pd.DataFrame(top_down_forecasts.T, columns=series_names[:top_down_forecasts.shape[0]]).to_excel(
                writer, sheet_name='TopDown_Forecasts', index=False
            )
            
            # Save actual values
            pd.DataFrame(actual_values.T, columns=series_names[:actual_values.shape[0]]).to_excel(
                writer, sheet_name='Actual_Values', index=False
            )
            
            # Save metrics
            metrics_df = pd.DataFrame({
                'Metric': ['MAE', 'RMSE', 'MAPE'],
                'Bottom-Up': [results['mae_bu'], results['rmse_bu'], results['mape_bu']],
                'Top-Down': [results['mae_td'], results['rmse_td'], results['mape_td']]
            })
            metrics_df.to_excel(writer, sheet_name='EBIT_Metrics', index=False)
        
        # Plot all forecasts
        print("Generating forecast plot...")
        self.plot_reconciliation(
            base_forecasts,
            bottom_up_forecasts,
            top_down_forecasts,
            actual_values,
            model_name
        )
        
        print(f"Completed evaluation for {model_name}")
        return results
    
    def plot_reconciliation(self, base_forecasts, bottom_up_forecasts, top_down_forecasts, actual_values, model_name):
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
            
            # Plot bottom-up reconciled forecasts
            ax.plot(bottom_up_forecasts[i], label='Bottom-Up Reconciled', color='green', linestyle='-')
            
            # Plot top-down reconciled forecasts
            ax.plot(top_down_forecasts[i], label='Top-Down Reconciled', color='purple', linestyle='-.')
            
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
            
            # Save bottom-up reconciled forecasts
            bottom_up_df = pd.DataFrame(bottom_up_forecasts.T, columns=series_names[:bottom_up_forecasts.shape[0]])
            bottom_up_df.to_excel(writer, sheet_name='BottomUp_Forecasts', index=False)
            
            # Save top-down reconciled forecasts
            top_down_df = pd.DataFrame(top_down_forecasts.T, columns=series_names[:top_down_forecasts.shape[0]])
            top_down_df.to_excel(writer, sheet_name='TopDown_Forecasts', index=False)
            
            # Save actual values
            actual_df = pd.DataFrame(actual_values.T, columns=series_names[:actual_values.shape[0]])
            actual_df.to_excel(writer, sheet_name='Actual_Values', index=False)
            
            # Calculate metrics for EBIT
            metrics_data = {
                'Base': {
                    'Model': model_name,
                    'Method': 'Base',
                    'MAE': np.mean(np.abs(actual_values[0] - base_forecasts[0])),
                    'RMSE': np.sqrt(np.mean((actual_values[0] - base_forecasts[0]) ** 2)),
                    'MAPE': np.mean(np.abs((actual_values[0] - base_forecasts[0]) / actual_values[0])) * 100
                },
                'Bottom-Up': {
                    'Model': model_name,
                    'Method': 'Bottom-Up',
                    'MAE': np.mean(np.abs(actual_values[0] - bottom_up_forecasts[0])),
                    'RMSE': np.sqrt(np.mean((actual_values[0] - bottom_up_forecasts[0]) ** 2)),
                    'MAPE': np.mean(np.abs((actual_values[0] - bottom_up_forecasts[0]) / actual_values[0])) * 100
                },
                'Top-Down': {
                    'Model': model_name,
                    'Method': 'Top-Down',
                    'MAE': np.mean(np.abs(actual_values[0] - top_down_forecasts[0])),
                    'RMSE': np.sqrt(np.mean((actual_values[0] - top_down_forecasts[0]) ** 2)),
                    'MAPE': np.mean(np.abs((actual_values[0] - top_down_forecasts[0]) / actual_values[0])) * 100
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