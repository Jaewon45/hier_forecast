import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Optional
import pandas as pd

class Visualization:
    """
    A class for visualizing forecasting results and hierarchical relationships.
    """
    
    @staticmethod
    def plot_forecasts(actual: np.ndarray,
                      predicted: np.ndarray,
                      title: str,
                      series_name: str,
                      save_path: Optional[str] = None) -> None:
        """
        Plot actual vs predicted values for a single series.
        
        Args:
            actual (np.ndarray): Actual values
            predicted (np.ndarray): Predicted values
            title (str): Plot title
            series_name (str): Name of the series
            save_path (str, optional): Path to save the plot
        """
        plt.figure(figsize=(12, 6))
        plt.plot(actual, label='Actual', marker='o')
        plt.plot(predicted, label='Predicted', marker='s')
        plt.title(f'{title} - {series_name}')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
    @staticmethod
    def plot_hierarchical_forecasts(actual: Dict[str, np.ndarray],
                                  predicted: Dict[str, np.ndarray],
                                  title: str,
                                  save_path: Optional[str] = None) -> None:
        """
        Plot actual vs predicted values for multiple series in a hierarchy.
        
        Args:
            actual (Dict[str, np.ndarray]): Dictionary of actual values
            predicted (Dict[str, np.ndarray]): Dictionary of predicted values
            title (str): Plot title
            save_path (str, optional): Path to save the plot
        """
        n_series = len(actual)
        fig, axes = plt.subplots(n_series, 1, figsize=(12, 3*n_series))
        
        for (ax, (series_name, actual_values)) in zip(axes, actual.items()):
            ax.plot(actual_values, label='Actual', marker='o')
            ax.plot(predicted[series_name], label='Predicted', marker='s')
            ax.set_title(series_name)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
    @staticmethod
    def plot_hierarchy_violations(violations: pd.DataFrame,
                                title: str,
                                save_path: Optional[str] = None) -> None:
        """
        Plot hierarchy violation checks over time.
        
        Args:
            violations (pd.DataFrame): DataFrame containing violation checks
            title (str): Plot title
            save_path (str, optional): Path to save the plot
        """
        plt.figure(figsize=(12, 6))
        
        for col in violations.columns:
            plt.plot(violations[col], label=col)
            
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Violation Amount')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
    @staticmethod
    def plot_metrics_comparison(metrics: Dict[str, Dict[str, float]],
                              title: str,
                              save_path: Optional[str] = None) -> None:
        """
        Plot comparison of metrics across different models or methods.
        
        Args:
            metrics (Dict[str, Dict[str, float]]): Dictionary of metrics for each model/method
            title (str): Plot title
            save_path (str, optional): Path to save the plot
        """
        metric_names = list(next(iter(metrics.values())).keys())
        model_names = list(metrics.keys())
        
        x = np.arange(len(metric_names))
        width = 0.8 / len(model_names)
        
        plt.figure(figsize=(12, 6))
        
        for i, model_name in enumerate(model_names):
            values = [metrics[model_name][metric] for metric in metric_names]
            plt.bar(x + i*width, values, width, label=model_name)
            
        plt.xlabel('Metrics')
        plt.ylabel('Value')
        plt.title(title)
        plt.xticks(x + width*(len(model_names)-1)/2, metric_names)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
    @staticmethod
    def plot_forecast_components(actual: np.ndarray,
                               trend: np.ndarray,
                               seasonal: np.ndarray,
                               residual: np.ndarray,
                               title: str,
                               save_path: Optional[str] = None) -> None:
        """
        Plot the components of a time series decomposition.
        
        Args:
            actual (np.ndarray): Original time series
            trend (np.ndarray): Trend component
            seasonal (np.ndarray): Seasonal component
            residual (np.ndarray): Residual component
            title (str): Plot title
            save_path (str, optional): Path to save the plot
        """
        fig, axes = plt.subplots(4, 1, figsize=(12, 12))
        
        components = {
            'Original': actual,
            'Trend': trend,
            'Seasonal': seasonal,
            'Residual': residual
        }
        
        for ax, (name, values) in zip(axes, components.items()):
            ax.plot(values)
            ax.set_title(name)
            ax.grid(True, alpha=0.3)
            
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show() 