import numpy as np
from typing import List, Dict, Tuple, Optional
import pandas as pd

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
        
        Returns:
            np.ndarray: The summing matrix S
        """
        # Get all unique nodes in the hierarchy
        all_nodes = set()
        for parent, children in self.hierarchy.items():
            all_nodes.add(parent)
            all_nodes.update(children)
        all_nodes = sorted(list(all_nodes))
        
        print("\nDebug: Creating Summing Matrix")
        print(f"All nodes: {all_nodes}")
        print(f"Number of total nodes: {len(all_nodes)}")
        
        # Create mapping from node names to indices
        node_to_idx = {node: idx for idx, node in enumerate(all_nodes)}
        print(f"Node to index mapping: {node_to_idx}")
        
        # Identify bottom level nodes - these are nodes that don't appear as parents
        # and are not in any parent's children list
        parent_nodes = set(self.hierarchy.keys())
        all_children = set()
        for children in self.hierarchy.values():
            all_children.update(children)
            
        # Bottom nodes are those that are children but not parents
        bottom_nodes = sorted(list(all_children - parent_nodes))
        
        # If no bottom nodes found, use all nodes that are not parents
        if not bottom_nodes:
            bottom_nodes = sorted(list(set(all_nodes) - parent_nodes))
            
        print(f"Bottom level nodes: {bottom_nodes}")
        print(f"Number of bottom nodes: {len(bottom_nodes)}")
        
        # Initialize summing matrix
        n_total = len(all_nodes)  # Total number of nodes
        n_bottom = len(bottom_nodes)  # Number of bottom level nodes
        S = np.zeros((n_total, n_bottom))
        print(f"Summing matrix S shape: {S.shape}")
        
        # Create mapping from bottom nodes to column indices
        bottom_to_col = {node: idx for idx, node in enumerate(bottom_nodes)}
        print(f"Bottom to column mapping: {bottom_to_col}")
        
        # Fill the summing matrix
        for node in all_nodes:
            if node in bottom_nodes:  # Bottom level node
                # Bottom level nodes map to themselves
                S[node_to_idx[node], bottom_to_col[node]] = 1
            else:  # Aggregated node
                # Find all bottom level descendants
                def get_bottom_descendants(node):
                    descendants = []
                    if node in bottom_nodes:
                        descendants.append(node)
                    else:
                        for child in self.hierarchy[node]:
                            descendants.extend(get_bottom_descendants(child))
                    return descendants
                
                bottom_descendants = get_bottom_descendants(node)
                print(f"Node {node} bottom descendants: {bottom_descendants}")
                for bottom_node in bottom_descendants:
                    S[node_to_idx[node], bottom_to_col[bottom_node]] = 1
        
        print("\nFinal Summing Matrix S:")
        print(S)
        self.S = S
        return S
    
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
        
        print("\nDebug: Creating Reconciliation Matrix")
        print(f"Method: {method}")
        print(f"Number of bottom nodes: {n_bottom}")
        print(f"Number of total nodes: {n_total}")
            
        if method == 'bottom_up':
            # For bottom-up, G is just the identity matrix for bottom level
            G = np.eye(n_bottom)
            print("Using identity matrix for bottom-up reconciliation")
            
        elif method == 'top_down':
            # For top-down, we need historical proportions
            G = np.zeros((n_bottom, n_total))
            print("Using zero matrix for top-down reconciliation (placeholder)")
            
        elif method == 'mint':
            # For MinT, we need covariance information
            G = np.zeros((n_bottom, n_total))
            print("Using zero matrix for MinT reconciliation (placeholder)")
            
        else:
            raise ValueError(f"Unknown reconciliation method: {method}")
            
        print(f"Reconciliation matrix G shape: {G.shape}")
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
        if self.G is None:
            self.create_reconciliation_matrix()
            
        print("\nDebug: Reconciling Forecasts")
        print(f"Input base_forecasts shape: {base_forecasts.shape}")
            
        # Ensure base_forecasts has the right shape
        if base_forecasts.ndim == 1:
            base_forecasts = base_forecasts.reshape(-1, 1)
            print(f"Reshaped base_forecasts to: {base_forecasts.shape}")
            
        # Get dimensions
        n_bottom = self.S.shape[1]  # Number of bottom level nodes
        n_total = self.S.shape[0]   # Total number of nodes
        
        print(f"Summing matrix S shape: {self.S.shape}")
        print(f"Reconciliation matrix G shape: {self.G.shape}")
        
        # Ensure base_forecasts has correct number of rows
        if base_forecasts.shape[0] != n_bottom:
            raise ValueError(f"Base forecasts must have {n_bottom} rows (bottom level nodes), got {base_forecasts.shape[0]}")
            
        # Apply reconciliation: y_hat = S * G * y_base
        # First multiply G with base_forecasts
        intermediate = self.G @ base_forecasts
        print(f"Intermediate result shape (G @ base_forecasts): {intermediate.shape}")
        
        # Then multiply S with the intermediate result
        reconciled = self.S @ intermediate
        print(f"Final reconciled forecasts shape: {reconciled.shape}")
        
        return reconciled
    
    def evaluate_reconciliation(self, 
                              base_forecasts: np.ndarray, 
                              actual_values: np.ndarray,
                              metrics: List[str] = ['mae', 'rmse', 'mape']) -> Dict[str, float]:
        """
        Evaluate the reconciliation performance using specified metrics.
        
        Args:
            base_forecasts (np.ndarray): Base forecasts
            actual_values (np.ndarray): Actual values
            metrics (List[str]): List of metrics to compute
            
        Returns:
            Dict[str, float]: Dictionary of metric values
        """
        reconciled = self.reconcile_forecasts(base_forecasts)
        results = {}
        
        for metric in metrics:
            if metric == 'mae':
                results['mae'] = np.mean(np.abs(reconciled - actual_values))
            elif metric == 'rmse':
                results['rmse'] = np.sqrt(np.mean((reconciled - actual_values)**2))
            elif metric == 'mape':
                results['mape'] = np.mean(np.abs((reconciled - actual_values) / actual_values)) * 100
                
        return results
    
    def plot_reconciliation(self, 
                          base_forecasts: np.ndarray,
                          reconciled_forecasts: np.ndarray,
                          actual_values: np.ndarray,
                          series_names: List[str]) -> None:
        """
        Plot the base forecasts, reconciled forecasts, and actual values.
        
        Args:
            base_forecasts (np.ndarray): Base forecasts
            reconciled_forecasts (np.ndarray): Reconciled forecasts
            actual_values (np.ndarray): Actual values
            series_names (List[str]): Names of the time series
        """
        import matplotlib.pyplot as plt
        
        n_series = len(series_names)
        fig, axes = plt.subplots(n_series, 1, figsize=(10, 3*n_series))
        
        for i, (ax, name) in enumerate(zip(axes, series_names)):
            ax.plot(base_forecasts[i], label='Base Forecast', linestyle='--')
            ax.plot(reconciled_forecasts[i], label='Reconciled', linestyle='-')
            ax.plot(actual_values[i], label='Actual', color='black')
            ax.set_title(name)
            ax.legend()
            
        plt.tight_layout()
        plt.show() 