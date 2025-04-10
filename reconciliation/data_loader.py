import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
from pathlib import Path

class DataLoader:
    """
    A class for loading and preprocessing hierarchical forecasting data.
    """
    
    def __init__(self, file_path: str):
        """
        Initialize DataLoader with file path.
        
        Args:
            file_path: Path to the Excel file containing the data
        """
        self.file_path = file_path
        self.df = None
        self.hierarchy = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load and preprocess data from Excel file.
        
        Returns:
            DataFrame with processed data
        """
        # Load data
        self.df = pd.read_excel(self.file_path)
        
        # Convert date column
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        
        # Create negative columns for costs
        self.df['-VariableCosts'] = -self.df['VariableCosts']
        self.df['-DepreciationAmortization'] = -self.df['DepreciationAmortization']
        self.df['-FixCosts'] = -self.df['FixCosts']
        
        # Sort by date
        self.df = self.df.sort_values('Date')
        
        return self.df
    
    def define_hierarchy(self) -> Dict[str, List[str]]:
        """
        Define the hierarchical structure of the data.
        
        Returns:
            Dictionary representing the hierarchy
        """
        self.hierarchy = {
            'EBIT': ['EBITDA'],
            'DepreciationAmortization': ['EBITDA'],
            'ContributionMargin1': ['EBIT'],
            '-FixCosts': ['EBIT'],
            'NetSales': ['ContributionMargin1'],
            '-VariableCosts': ['ContributionMargin1']
        }
        return self.hierarchy
    
    def split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/validation/test sets.
        
        Returns:
            Tuple of (train, validation, test) DataFrames
        """
        if self.df is None:
            self.load_data()
            
        # Split by year
        train = self.df[self.df['Date'].dt.year < 2022]
        val = self.df[self.df['Date'].dt.year == 2022]
        test = self.df[self.df['Date'].dt.year > 2022]
        
        return train, val, test
    
    def get_bottom_level_series(self) -> List[str]:
        """
        Get the list of bottom-level series in the hierarchy.
        
        Returns:
            List of bottom-level series names
        """
        return ['NetSales', '-VariableCosts', '-FixCosts', 'EBITDA', '-DepreciationAmortization']
    
    def prepare_time_series(self, df: pd.DataFrame, series_names: List[str]) -> List[np.ndarray]:
        """
        Prepare time series data for forecasting.
        
        Args:
            df: DataFrame containing the data
            series_names: List of series to prepare
            
        Returns:
            List of numpy arrays containing the time series
        """
        series_data = []
        for name in series_names:
            if name in df.columns:
                series_data.append(df[name].values)
            else:
                raise ValueError(f"Series {name} not found in data")
        return series_data
    
    def create_time_series_dict(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Create dictionary of numpy arrays for each component.
        
        Args:
            train_df: Training data DataFrame
            test_df: Test data DataFrame
            
        Returns:
            Dictionary containing training and test series
        """
        series_dict = {}
        columns = ['EBIT', 'ContributionMargin1', 'EBITDA',
                  'NetSales', '-VariableCosts', '-FixCosts', '-DepreciationAmortization']
        
        for col in columns:
            # Training series
            series_dict[col] = train_df[col].values
            # Test series (for evaluation)
            series_dict[f"{col}_test"] = test_df[col].values
            
        return series_dict
    
    def get_series_names(self) -> List[str]:
        """
        Get the names of all series in the hierarchy.
        
        Returns:
            List[str]: List of series names
        """
        if self.hierarchy is None:
            self.define_hierarchy()
            
        all_series = set()
        for parent, children in self.hierarchy.items():
            all_series.add(parent)
            all_series.update(children)
            
        return sorted(list(all_series)) 