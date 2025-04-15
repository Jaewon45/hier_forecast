import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
from prophet import Prophet

class ForecastingModels:
    def __init__(self):
        """Initialize forecasting models."""
        self.models = {
            'LinearRegression': LinearRegression(),
            'ExponentialSmoothing': None,  # Will be initialized per series
            'ARIMA': None  # Will be initialized per series
        }
        
    def train_linear_regression(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train linear regression model.
        
        Args:
            X: Feature matrix
            y: Target values
        """
        self.models['LinearRegression'].fit(X, y)
        
    def train_exponential_smoothing(self, series: np.ndarray) -> None:
        """
        Train exponential smoothing model.
        
        Args:
            series: Time series data
        """
        self.models['ExponentialSmoothing'] = ExponentialSmoothing(
            series,
            trend='add',
            seasonal='add',
            seasonal_periods=12
        ).fit()
        
    def train_arima(self, series: np.ndarray) -> None:
        """
        Train ARIMA model.
        
        Args:
            series: Time series data
        """
        self.models['ARIMA'] = ARIMA(series, order=(1,1,1)).fit()
        
    def forecast_linear_regression(self, X: np.ndarray) -> np.ndarray:
        """
        Generate forecasts using linear regression.
        
        Args:
            X: Feature matrix for forecasting
            
        Returns:
            Forecasted values
        """
        return self.models['LinearRegression'].predict(X)
        
    def forecast_exponential_smoothing(self, steps: int) -> np.ndarray:
        """
        Generate forecasts using exponential smoothing.
        
        Args:
            steps: Number of steps to forecast
            
        Returns:
            Forecasted values
        """
        return self.models['ExponentialSmoothing'].forecast(steps)
        
    def forecast_arima(self, steps: int) -> np.ndarray:
        """
        Generate forecasts using ARIMA.
        
        Args:
            steps: Number of steps to forecast
            
        Returns:
            Forecasted values
        """
        return self.models['ARIMA'].forecast(steps)
        
    def generate_forecasts(self, 
                         series_dict: Dict[str, np.ndarray],
                         forecast_horizon: int = 12) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Generate forecasts for all series using all models.
        
        Args:
            series_dict: Dictionary of time series data
            forecast_horizon: Number of steps to forecast
            
        Returns:
            Dictionary of forecasts for each model and series
        """
        forecasts = {}
        
        for series_name, series_data in series_dict.items():
            if series_name.endswith('_test'):
                continue
                
            series_forecasts = {}
            
            # Prepare data for linear regression
            X = np.arange(len(series_data)).reshape(-1, 1)
            y = series_data.reshape(-1, 1)
            
            # Train and forecast with linear regression
            self.train_linear_regression(X, y)
            X_future = np.arange(len(series_data), len(series_data) + forecast_horizon).reshape(-1, 1)
            series_forecasts['LinearRegression'] = self.forecast_linear_regression(X_future).flatten()
            
            # Train and forecast with exponential smoothing
            self.train_exponential_smoothing(series_data)
            series_forecasts['ExponentialSmoothing'] = self.forecast_exponential_smoothing(forecast_horizon)
            
            # Train and forecast with ARIMA
            self.train_arima(series_data)
            series_forecasts['ARIMA'] = self.forecast_arima(forecast_horizon)
            
            forecasts[series_name] = series_forecasts
            
        return forecasts

    def forecast_all_series(self, 
                          series_dict: Dict[str, np.ndarray],
                          n_periods: int = 12,
                          validation_period: int = 12,
                          model: Optional[Union[str, object]] = None) -> Dict[str, np.ndarray]:
        """
        Generate forecasts for all series using either the specified model or the best performing model based on validation.
        
        Args:
            series_dict: Dictionary of time series data
            n_periods: Number of steps to forecast
            validation_period: Number of periods to use for validation
            model: Optional model to use. Can be:
                  - String name ('LinearRegression', 'ExponentialSmoothing', 'ARIMA', or 'Prophet')
                  - Model instance (any class name containing these model names)
                  - None (will select best model based on validation)
            
        Returns:
            Dictionary of forecasts for each series
        """
        best_forecasts = {}
        
        for series_name, series_data in series_dict.items():
            if series_name.endswith('_test'):
                continue
                
            # Split data into training and validation
            train_data = series_data[:-validation_period]
            val_data = series_data[-validation_period:]
            
            # Prepare data for linear regression
            X_train = np.arange(len(train_data)).reshape(-1, 1)
            y_train = train_data.reshape(-1, 1)
            X_val = np.arange(len(train_data), len(train_data) + validation_period).reshape(-1, 1)
            
            if model is not None:
                # Handle both string model names and model instances
                if isinstance(model, str):
                    model_name = model
                else:
                    # Extract class name and check if it contains any of our model names
                    class_name = model.__class__.__name__
                    if 'LinearRegression' in class_name:
                        model_name = 'LinearRegression'
                    elif 'ExponentialSmoothing' in class_name:
                        model_name = 'ExponentialSmoothing'
                    elif 'ARIMA' in class_name:
                        model_name = 'ARIMA'
                    elif 'Prophet' in class_name:
                        model_name = 'Prophet'
                    else:
                        raise ValueError(f"Unknown model: {class_name}. Must contain one of: 'LinearRegression', 'ExponentialSmoothing', 'ARIMA', 'Prophet'")
                
                if model_name == 'LinearRegression':
                    self.train_linear_regression(X_train, y_train)
                    X_future = np.arange(len(series_data), len(series_data) + n_periods).reshape(-1, 1)
                    best_forecasts[series_name] = self.forecast_linear_regression(X_future).flatten()
                elif model_name == 'ExponentialSmoothing':
                    self.train_exponential_smoothing(series_data)
                    best_forecasts[series_name] = self.forecast_exponential_smoothing(n_periods)
                elif model_name == 'ARIMA':
                    self.train_arima(series_data)
                    best_forecasts[series_name] = self.forecast_arima(n_periods)
                elif model_name == 'Prophet':
                    # Convert numpy array to pandas DataFrame for Prophet
                    df = pd.DataFrame({
                        'ds': pd.date_range(start='2020-01-01', periods=len(series_data), freq='M'),
                        'y': series_data
                    })
                    prophet_model = Prophet()
                    prophet_model.fit(df)
                    future = prophet_model.make_future_dataframe(periods=n_periods, freq='M')
                    forecast = prophet_model.predict(future)
                    best_forecasts[series_name] = forecast['yhat'].values[-n_periods:]
                else:
                    raise ValueError(f"Unknown model: {model_name}. Must be one of: 'LinearRegression', 'ExponentialSmoothing', 'ARIMA', 'Prophet'")
            else:
                # Train and validate each model
                model_performance = {}
                
                # Linear Regression
                self.train_linear_regression(X_train, y_train)
                lr_forecast = self.forecast_linear_regression(X_val).flatten()
                model_performance['LinearRegression'] = np.mean(np.abs(lr_forecast - val_data))
                
                # Exponential Smoothing
                self.train_exponential_smoothing(train_data)
                exp_forecast = self.forecast_exponential_smoothing(validation_period)
                model_performance['ExponentialSmoothing'] = np.mean(np.abs(exp_forecast - val_data))
                
                # ARIMA
                self.train_arima(train_data)
                arima_forecast = self.forecast_arima(validation_period)
                model_performance['ARIMA'] = np.mean(np.abs(arima_forecast - val_data))
                
                # Prophet
                df = pd.DataFrame({
                    'ds': pd.date_range(start='2020-01-01', periods=len(train_data), freq='M'),
                    'y': train_data
                })
                prophet_model = Prophet()
                prophet_model.fit(df)
                future = prophet_model.make_future_dataframe(periods=validation_period, freq='M')
                prophet_forecast = prophet_model.predict(future)
                model_performance['Prophet'] = np.mean(np.abs(prophet_forecast['yhat'].values[-validation_period:] - val_data))
                
                # Select best model based on validation MAE
                best_model = min(model_performance.items(), key=lambda x: x[1])[0]
                
                # Generate final forecast using best model
                if best_model == 'LinearRegression':
                    X_future = np.arange(len(series_data), len(series_data) + n_periods).reshape(-1, 1)
                    best_forecasts[series_name] = self.forecast_linear_regression(X_future).flatten()
                elif best_model == 'ExponentialSmoothing':
                    self.train_exponential_smoothing(series_data)  # Retrain on full data
                    best_forecasts[series_name] = self.forecast_exponential_smoothing(n_periods)
                elif best_model == 'ARIMA':
                    self.train_arima(series_data)  # Retrain on full data
                    best_forecasts[series_name] = self.forecast_arima(n_periods)
                elif best_model == 'Prophet':
                    df = pd.DataFrame({
                        'ds': pd.date_range(start='2020-01-01', periods=len(series_data), freq='M'),
                        'y': series_data
                    })
                    prophet_model = Prophet()
                    prophet_model.fit(df)
                    future = prophet_model.make_future_dataframe(periods=n_periods, freq='M')
                    forecast = prophet_model.predict(future)
                    best_forecasts[series_name] = forecast['yhat'].values[-n_periods:]
                
                print(f"Selected {best_model} for {series_name} (MAE: {model_performance[best_model]:.2f})")
            
        return best_forecasts 