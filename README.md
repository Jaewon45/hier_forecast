# Hierarchical Forecasting System

This project implements a hierarchical forecasting system with multiple models, reconciliation methods, and evaluation metrics.

## Project Structure

```
.
├── main.py                  # Main script to run the complete workflow
├── reconciliation/          # Core forecasting and reconciliation modules
│   ├── __init__.py
│   ├── data_loader.py      # Data loading and preprocessing
│   ├── models.py           # Forecasting models
│   ├── reconciliation.py   # Reconciliation methods
│   ├── utils.py            # Utility functions
│   └── visualization.py    # Plotting functions
├── data/                    # Data directory
│   └── SampleHierForecastingBASF_share.xlsx
└── output/                  # Output directory for results
    ├── forecasts.xlsx      # Base and reconciled forecasts
    └── metrics.csv         # Model performance metrics
```

## Workflow

The forecasting workflow consists of the following steps:

1. **Data Loading and Preprocessing**
   - Load data from Excel file
   - Define hierarchy structure
   - Split data into train/validation sets
   - Prepare time series data

2. **Model Training and Forecasting**
   - Train multiple models (Linear Regression, Exponential Smoothing, ARIMA)
   - Generate base forecasts
   - Perform backtesting
   - Evaluate model performance

3. **Reconciliation**
   - Apply reconciliation methods (Bottom-up, Top-down, MinT)
   - Generate reconciled forecasts
   - Evaluate reconciliation performance

4. **Results Export**
   - Export forecasts to Excel (separate sheets for base and reconciled)
   - Export metrics to CSV
   - Generate visualizations

## Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the complete workflow:
```bash
python main.py
```

3. View results:
   - Check `output/forecasts.xlsx` for base and reconciled forecasts
   - Check `output/metrics.csv` for model performance metrics
   - Visualizations will be displayed during execution

## Configuration

The main script can be configured by modifying parameters in `main.py`:
- Data file path
- Forecast horizon
- Models to use
- Reconciliation methods
- Output directory

## Example Output

1. `forecasts.xlsx`:
   - Sheet "Base Forecasts": Original model predictions
   - Sheet "Reconciled Forecasts": Hierarchically reconciled predictions

2. `metrics.csv`:
   - Model performance metrics (MAE, RMSE, MAPE)
   - Comparison across different models and horizons

## Dependencies

- pandas
- numpy
- scipy
- statsmodels
- matplotlib
- openpyxl (for Excel export) 