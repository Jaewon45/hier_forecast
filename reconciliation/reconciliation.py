import numpy as np
from typing import Dict, List, Tuple
from scipy.optimize import minimize

class Reconciliation:
    def __init__(self, hierarchy: Dict[str, List[str]]):
        """
        Initialize reconciliation class.
        
        Args:
            hierarchy: Dictionary representing the hierarchical structure
        """
        self.hierarchy = hierarchy
        self.bottom_level_series = self._get_bottom_level_series()
        self.aggregation_matrix = self._create_aggregation_matrix()
        
    def _get_bottom_level_series(self) -> List[str]:
        """
        Get list of bottom level series from hierarchy.
        
        Returns:
            List of bottom level series names
        """
        bottom_level = []
        for parent, children in self.hierarchy.items():
            if not children:  # No children means it's a bottom level series
                bottom_level.append(parent)
            else:
                for child in children:
                    if child not in self.hierarchy:  # Child not in hierarchy means it's bottom level
                        bottom_level.append(child)
        return bottom_level
        
    def _create_aggregation_matrix(self) -> np.ndarray:
        """
        Create aggregation matrix S based on hierarchy.
        
        Returns:
            Aggregation matrix as numpy array
        """
        all_series = list(self.hierarchy.keys()) + self.bottom_level_series
        n_bottom = len(self.bottom_level_series)
        n_total = len(all_series)
        
        S = np.zeros((n_total, n_bottom))
        
        for i, series in enumerate(all_series):
            if series in self.bottom_level_series:
                j = self.bottom_level_series.index(series)
                S[i, j] = 1
            else:
                for child in self.hierarchy[series]:
                    if child in self.bottom_level_series:
                        j = self.bottom_level_series.index(child)
                        S[i, j] = 1
                        
        return S
        
    def _calculate_weights(self, 
                         base_forecasts: Dict[str, np.ndarray],
                         actual_values: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Calculate weights for reconciliation based on forecast errors.
        
        Args:
            base_forecasts: Dictionary of base forecasts
            actual_values: Dictionary of actual values
            
        Returns:
            Weight matrix as numpy array
        """
        errors = {}
        for series in base_forecasts:
            if series in actual_values:
                errors[series] = np.mean(np.abs(base_forecasts[series] - actual_values[series]))
                
        # Convert errors to weights (inverse of errors)
        weights = np.array([1.0 / (errors.get(series, 1.0)) for series in self.bottom_level_series])
        return np.diag(weights)
        
    def reconcile_forecasts(self,
                          base_forecasts: Dict[str, np.ndarray],
                          actual_values: Dict[str, np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Reconcile forecasts using optimal combination method.
        
        Args:
            base_forecasts: Dictionary of base forecasts
            actual_values: Dictionary of actual values for weight calculation
            
        Returns:
            Dictionary of reconciled forecasts
        """
        # Convert forecasts to matrix form
        n_bottom = len(self.bottom_level_series)
        n_total = self.aggregation_matrix.shape[0]
        h = len(next(iter(base_forecasts.values())))  # Forecast horizon
        
        y_hat = np.zeros((n_total, h))
        for i, series in enumerate(self.bottom_level_series):
            if series in base_forecasts:
                y_hat[i] = base_forecasts[series]
                
        # Calculate weights if actual values are provided
        if actual_values is not None:
            W = self._calculate_weights(base_forecasts, actual_values)
        else:
            W = np.eye(n_bottom)
            
        # Calculate optimal combination matrix
        S = self.aggregation_matrix
        P = np.linalg.inv(S.T @ W @ S) @ S.T @ W
        
        # Reconcile forecasts
        y_tilde = S @ P @ y_hat
        
        # Convert back to dictionary format
        reconciled_forecasts = {}
        for i, series in enumerate(self.bottom_level_series):
            reconciled_forecasts[series] = y_tilde[i]
            
        return reconciled_forecasts
        
    def evaluate_reconciliation(self,
                              base_forecasts: Dict[str, np.ndarray],
                              reconciled_forecasts: Dict[str, np.ndarray],
                              actual_values: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Evaluate reconciliation performance using various metrics.
        
        Args:
            base_forecasts: Dictionary of base forecasts
            reconciled_forecasts: Dictionary of reconciled forecasts
            actual_values: Dictionary of actual values
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {}
        
        for series in actual_values:
            if series in base_forecasts and series in reconciled_forecasts:
                y_true = actual_values[series]
                y_base = base_forecasts[series]
                y_recon = reconciled_forecasts[series]
                
                # Mean Absolute Error
                mae_base = np.mean(np.abs(y_true - y_base))
                mae_recon = np.mean(np.abs(y_true - y_recon))
                
                # Mean Squared Error
                mse_base = np.mean((y_true - y_base) ** 2)
                mse_recon = np.mean((y_true - y_recon) ** 2)
                
                # Mean Absolute Percentage Error
                mape_base = np.mean(np.abs((y_true - y_base) / y_true)) * 100
                mape_recon = np.mean(np.abs((y_true - y_recon) / y_true)) * 100
                
                metrics[series] = {
                    'MAE_base': mae_base,
                    'MAE_recon': mae_recon,
                    'MSE_base': mse_base,
                    'MSE_recon': mse_recon,
                    'MAPE_base': mape_base,
                    'MAPE_recon': mape_recon
                }
                
        return metrics 