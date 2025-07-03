import pandas as pd
from darts import TimeSeries
from darts.models import (Prophet, LinearRegressionModel, ARIMA,  ExponentialSmoothing, XGBModel,  NBEATSModel, GlobalNaiveAggregate, NaiveDrift)
from darts.metrics import mape, rmse, r2_score, mae, smape
import matplotlib as plt
import re


def score_function(row):
    """custom score function weighting different priorities, test feature"""
    return (row['yearly_sMAPE'] +
            (row['yearly_R2'] * -0.5) +
            (row['max_quarterly_RMSE'] * 0.5) +
            (row['max_quarterly_RMSE'] * 0.5) +
            row['yearly_RMSE'])

def get_winners(df, val):
    """Evaluate model performance across quarters and years, returning detailed metrics for export.
    
    Args:
        df: DataFrame containing model predictions with 'Name' and 'Predictions' columns
        val: Actual values to compare against (TimeSeries or pandas Series)
        
    Returns:
        tuple: (quarterly_results_df, summary_stats_df)
            - quarterly_results_df: Metrics broken down by model and quarter
            - summary_stats_df: Aggregated yearly metrics with quarterly statistics
    """
    df = df.set_index('Date')
    if isinstance(val, TimeSeries):
        val = val.to_series()
    val = val.reindex(df.index)
    df['Actuals'] = val

    df['Quarter'] = df.index.quarter
    df['Year'] = df.index.year

    quarterly_results = []

    for (model_name, quarter), group in df.groupby(['Name', 'Quarter']):
        pred = TimeSeries.from_series(group['Predictions'])
        true = TimeSeries.from_series(group['Actuals'])

        mae_val = mae(true, pred)
        smape_val = smape(true, pred)
        rmse_val = rmse(true, pred)
        r2_val = r2_score(true, pred)

        quarterly_results.append({
            'Model': model_name,
            'Quarter': f'Q{quarter}',
            'MAE': mae_val,
            'sMAPE': smape_val,
            'RMSE': rmse_val,
            'R2': r2_val
        })

    quarterly_df = pd.DataFrame(quarterly_results)

    yearly_stats = []
    for model_name in df['Name'].unique():
        model_mask = df['Name'] == model_name
        pred = TimeSeries.from_series(df[model_mask]['Predictions'])
        true = TimeSeries.from_series(df[model_mask]['Actuals'])

        yearly_stats.append({
            'Model': model_name,
            'yearly_RMSE': rmse(true, pred),
            'yearly_MAE': mae(true, pred),
            'yearly_R2': r2_score(true, pred),
            'yearly_sMAPE': smape(true, pred)
        })

    yearly_df = pd.DataFrame(yearly_stats)

    quarterly_agg = quarterly_df.groupby('Model').agg({
        'RMSE': ['mean', 'var', 'max'],
        'sMAPE': ['mean', 'var', 'max']
    })

    quarterly_agg.columns = [
        'avg_quarterly_RMSE', 'var_quarterly_RMSE', 'max_quarterly_RMSE',
        'avg_quarterly_sMAPE', 'var_quarterly_sMAPE', 'max_quarterly_sMAPE'
    ]
    quarterly_agg = quarterly_agg.reset_index()

    summary_df = pd.merge(yearly_df, quarterly_agg, on='Model')
    return quarterly_df,  summary_df

def get_inflation(past_cov):
    """function for testing purposes, gets inflation rate as an additional covariate. not used in final pipeline"""
    inflation = pd.read_excel('globalinflation.xlsx')
    inflation = inflation.T
    inflation.index = pd.to_datetime(inflation.index, format='%Y')

    start_date = inflation.index.min()
    end_date = min(inflation.index.max() + pd.DateOffset(months=11),past_cov.time_index.max() )
    date_range = pd.date_range(start=start_date, end=end_date, freq='MS')
    inflation = inflation.reindex(date_range)
    #inflation = inflation.ffill()
    inflation = inflation.interpolate(method="linear")
    inflation = inflation.reset_index()
    inflation.columns = ["Date","inflation_rate"]

    return past_cov.stack(TimeSeries.from_dataframe(inflation, time_col="Date", value_cols="inflation_rate"))

def load_data(file_path='SampleHierForecastingBASF_share.xlsx'):
  """Load and preprocess financial data from Excel file.
    
    Args:
        file_path: Path to Excel file containing financial data
        
    Returns:
        DataFrame: Processed data with datetime index and numeric columns
    """
  df = pd.read_excel(file_path)
  df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%y')
  df = df.sort_values(by='Date')
  numeric_cols = ['EBIT', 'EBITDA', 'DepreciationAmortization', 'VariableCosts', 'NetSales', 'ContributionMargin1', 'FixCosts']
  df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
  return df

def get_metrics(pred, val, name):
  """
    Standardized performance metrics to evaluate model performance. Print performance according to different metrics

    Args:
        pred: Predicted time series from model
        val: Validation period time series, should be same shape as pred
        name: Name of model as String

    """
  print(f"MAE for {name}: {mae(val,pred):.2f}")
  print(f"MAPE for {name}: {mape(val,pred):.2f}")
  print(f"RMSE for {name}: {rmse(val, pred):.2f}")
  print(f"R^2 for {name}: {r2_score(val, pred):.2f}")
  print(f"SMAPE for {name}: {smape(val, pred):.2f}")

  print()

def get_metrics_batched(preds,val, listlength=3, target='EBIT'):
  """
    Standardized performance metrics to evaluate model performance. Prints performance of all models and returns top x performers according to target

    Args:
        preds: Dictionary of form name of model : predicted time series from model
        val: Validation period time series
        listlength: number of top models to return
      
    Returns:
        sorted_models:{model_name:RMSE value}

    """
  values = {}

  for key in preds:
    s=preds[key]
    curr_mae=mae(val,s)
    print(f"MAE for {key}: {curr_mae:.2f}")
    curr_mape=mape(val,s)
    print(f"MAPE for {key}: {curr_mape:.2f}")
    curr_rmse=rmse(val, s)
    print(f"RMSE for {key}: {curr_rmse:.2f}")
    print(f"R^2 for {key}: {r2_score(val, s):.2f}")
    print(f"SMAPE for {key}: {smape(val, s):.2f}")
    values[key]={'MAE': curr_mae,
                           'MAPE': curr_mape,
                           'RMSE':curr_rmse}

    print()
  sorted_models = sorted(values.items(), key=lambda x: x[1]['RMSE'])
  sorted_models = sorted_models[:listlength]
  return dict(sorted_models)

def backtestmodels(fittedmodels, data, fh = 12, past_cov = None, retrain=True, start=pd.Timestamp('2022-01-01'), train_length=24, stride=3, plot=True):
  backtest_results_dict = {}
  for model_name, model in fittedmodels.items():
    kwargs = {
                'series': data,
                'start': start,
                'forecast_horizon': fh,
                'stride': stride,
                'retrain': retrain,
                'last_points_only': True,
                'verbose': False,
                'overlap_end' : True
            }
    if model.supports_past_covariates and past_cov:
      kwargs['past_covariates'] = past_cov

    backtested = model.historical_forecasts(**kwargs)
    backtest_results_dict[model_name+', backtested'] = backtested

  get_metrics_batched(backtest_results_dict, data['EBIT'])

  if plot:
        for model_name, forecast in backtest_results_dict.items():
            print(f"{model_name} - first forecast time: {forecast[0].start_time()}")
            forecast[0].plot(label=model_name)
        data['EBIT'].plot(label='Actual')
        plt.legend()
        plt.show()
  return backtest_results_dict

def compare_models_multivariate(data, val, models):
  """
    Evaluate multiple forecasting models. supports multivariate models data may be hierarchical or nonhierarchical.

    Args:
        data: Training data TimeSeries.
        val: Validation period time series

    Returns:
        Two dictionaries:
        - fitted_models: {model_name: trained_model}
        - predictions: {model_name: forecast_series}

    Example:
        >>> models = [XGBModel(lags=12)]
        >>> fitted, preds = compare_models_multivariate(train, val, models)
    """
  fittedmodels = {}
  predictions = {}
  for m in models:
    model_name = re.match(r"^([A-Za-z0-9_]+)\(", str(m)).group(1)
    m.fit(data)
    pred = m.predict(n=len(val))
    fittedmodels[model_name]=m
    predictions[model_name]=pred
  for_testing = {k:v['EBIT'] for k,v in predictions.items()}
  get_metrics_batched(for_testing, val['EBIT'])
  return fittedmodels, predictions

def compare_models_univariate(data, val, models, past_cov=None):
  """
    Evaluate multiple forecasting models. supports univariate models with and without covariates.
    Note that this doesn't currently work with reconciliation. Do not pass data with hierarchy implemented.

    Args:
        data: Training data TimeSeries. Should consist of only target variable.
        val: Validation period time series, include target and past_covariates
        past_covariates: covariates, optional

    Returns:
        Two dictionaries:
        - fitted_models: {model_name: trained_model}
        - predictions: {model_name: forecast_series}

    Example:
        >>> models = [AutoARIMA(), XGBModel(lags=12)]
        >>> fitted, preds = compare_models_univariate(train, val, covariates, models)
    """
  #check performance of passed models on data, return all fitted models in case of future evaluation needs
  fittedmodels = {}
  predictions = {}
  for m in models:
    model_name = re.match(r"^([A-Za-z0-9_]+)\(", str(m)).group(1)+', Uni'
    if m.supports_past_covariates and past_cov:
      m.fit(data, past_covariates=past_cov)
      pred = m.predict(n=len(val), past_covariates=past_cov)
    else:
      m.fit(data)
      pred = m.predict(n=len(val))
    fittedmodels[model_name]=m
    predictions[model_name]=pred
  get_metrics_batched(predictions, val)
  return fittedmodels, predictions

def compare_models_reconciliated(data, val, models, reconciliators, reconciliator_names):
  """
    Evaluate multiple multivariate forecasting models using different reconciliations methods. compare based on performance with EBIT.

    Args:
        data: Training data TimeSeries
        val: Validation period time series
        models: a dictionary of fitted models
        reconciliators: initialized reconcilors
        reconciliator_names: corresponding names. order matters.

    Returns:
        - reconciled_predictions: {model_name: forecast_series}
    """
  reconciled_predictions={}
  for k,v in models.items():
    print(f"Testing {k}")
    for i in range(len(reconciliators)):
      reconciled_predictions[k+", "+reconciliator_names[i]] = reconciliators[i].transform(v)
  for_testing = {k:v['EBIT'] for k,v in reconciled_predictions.items()}
  get_metrics_batched(for_testing, val['EBIT'])
  return reconciled_predictions

def get_best_per_series(data, val, models, past_cov=None):
  """
    Evaluate multiple forecasting models. supports univariate and multivariate models.
    Note that this doesn't currently work with reconciliation. Do not pass data with hierarchy implemented.

    Args:
        data: Training data TimeSeries. Should consist of only target variable.
        val: Validation period time series, includes target and past_covariates
        past_covariates: Hierarchical covariates, only require

    Returns:
        Two dictionaries:
        - fitted_models: {model_name: trained_model}
        - predictions: {model_name: forecast_series}

    Example:
        >>> models = [AutoARIMA(), XGBModel(lags=12)]
        >>> fitted, preds = compare_models_simple(train, val, covariates, models)
    """
  #check performance of passed models on data, return all fitted models in case of future evaluation needs
  best = {}
  for s in data.components:
    predictions = {}
    rmse_scores = {}
    for m in models:
      model_name = re.match(r"^([A-Za-z0-9_]+)\(", str(m)).group(1)
      if m.supports_past_covariates and past_cov:
          m.fit(data[s], past_covariates=past_cov)
          pred = m.predict(n=len(val[s]), past_covariates=past_cov)
      else:
          m.fit(data[s])
          pred = m.predict(n=len(val[s]))
      predictions[model_name]=pred
      rmse_scores[model_name]=rmse(pred,val[s])    
    best_for_series = min(rmse_scores.items(), key=lambda x: x[1])[0]
    print(f"{s}: {best_for_series}")
    best[s]=predictions[best_for_series]
  if len(best) == 1:
      return best[0]
  else:
      temp = best['EBIT']
      for s,pred in best.items():
        if s != 'EBIT':
          temp = temp.concatenate(pred,axis=1)
      return temp

def apply_hierarchy(df, hvar = 'var0'):
  """
  Apply hierarchy to data.
  Args: 
      df: data to apply hierarchy to. may be DataFrame or TimeSeries
      hvar: controls the path in the hierarchy selected
  Returns:
      series: the given df, with hierarchy applied
      target: target series
      covariates: all series other than target
      hierarchy: the hierarchy selected, as a dictionary
  """
  if hvar == 'var1':
    hierarchy = {
          'EBIT': ['EBITDA'],
          '-DepreciationAmortization': ['EBITDA']
      }
    cols = ["EBITDA", "-DepreciationAmortization"]
  elif hvar == 'var2':
    hierarchy = {
          'ContributionMargin1': ['EBIT'],
          '-FixCosts': ['EBIT'],
          'NetSales': ['ContributionMargin1'],
          '-VariableCosts': ['ContributionMargin1']
      }
    cols = ["ContributionMargin1", "-FixCosts", "NetSales", "-VariableCosts"]
  else:
    hierarchy = {
          'EBIT': ['EBITDA'],
          '-DepreciationAmortization': ['EBITDA'],
          'ContributionMargin1': ['EBIT'],
          '-FixCosts': ['EBIT'],
          'NetSales': ['ContributionMargin1'],
          '-VariableCosts': ['ContributionMargin1']
      }
    cols = ["EBITDA", "-DepreciationAmortization", "ContributionMargin1", "-FixCosts", "NetSales", "-VariableCosts"]

  if isinstance(df, pd.DataFrame):
    df['-VariableCosts'] = -df['VariableCosts']
    df['-FixCosts'] = -df['FixCosts']
    df['-DepreciationAmortization'] = -df['DepreciationAmortization']
    series = TimeSeries.from_dataframe(df, time_col="Date", value_cols=cols+['EBIT'])
    series = series.with_hierarchy(hierarchy)
    target = series["EBIT"]
    covariates = series[cols]

  elif isinstance(df, TimeSeries):
    series = df[cols+['EBIT']].with_hierarchy(hierarchy)
    target = series["EBIT"]
    covariates = series[cols]

  else:
    raise TypeError("df must be a pandas DataFrame or TimeSeries")

  return series, target, covariates, hierarchy

def get_levels(ts):
  """Reverse the hierarchy mapping to show parent-child relationships. 
    
    Args:
        ts: TimeSeries with hierarchy
        
    Returns:
        dict: {parent: [child1, child2,...]} mapping
    """
  hierarchy = ts.hierarchy
  reversed = {}
  for key, value in ts.hierarchy.items():
    if value[0] in reversed:
      reversed[value[0]].append(key)
    else:
      reversed[value[0]] = [key]
  return reversed