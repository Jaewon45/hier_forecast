
import pandas as pd
import numpy as np
import matplotlib
from darts import TimeSeries
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib.dates as mdates



def load_val(file_path='data/SampleHierForecastingBASF_share.xlsx'):
  df = pd.read_excel(file_path)
  df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%y')
  df = df.sort_values(by='Date')
  numeric_cols = ['EBIT', #'EBITDA', 'DepreciationAmortization',
                  'VariableCosts', 'NetSales', 'ContributionMargin1', 'FixCosts']
  df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
  df = df[['Date','EBIT']]
  return df

def load_predictions(varnum=0):
    preds = pd.read_csv('output/predictions_var'+str(varnum)) 
    pivoted = preds.pivot( index='Date', columns='Name', values='Predictions')
    pivoted.index = pd.to_datetime(pivoted.index)
    return pivoted

def load_metrics(varnum=0):
    metrics = pd.read_csv('output/results_var'+str(varnum), index_col=0) 
    metrics.reset_index()
    metrics = metrics.set_index('Model')
    return metrics



def plot_vs_val(series_of_interest,title, actuals,months_back=36):
    actuals[-months_back:-12].plot(label="Actuals", lw=0.9)
    for name, forecast in series_of_interest:
        forecast['EBIT'].plot(label=name, lw=0.9)
    plt.title(title)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Models")
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.ylabel('RMSE')
    out_file = 'output/'+''.join(title.split()).lower()
    plt.savefig(out_file, bbox_inches='tight')
    
    



def make_heat_map(s):
  
    rmse_series = s.copy() 

    rmse_series.index = rmse_series.index.str.replace('Per Series', 'Multi Model', regex=False)

    rmse_series = rmse_series[~rmse_series.index.str.contains('uni', case=False)]

    baselines = {}
    for idx in rmse_series.index:
        if ',' not in idx:
            baselines[idx.strip()] = rmse_series[idx]

    # 4. Build data table
    data = []
    for idx, rmse in rmse_series.items():
        parts = idx.split(',')
        model = parts[0].strip()
        method = parts[1].strip() if len(parts) > 1 else 'Baseline'
        if model in baselines:
            baseline_rmse = baselines[model]
            delta_pct = 100 * (rmse - baseline_rmse) / baseline_rmse
            data.append((method, model, rmse, delta_pct))

    # 5. Create DataFrame and pivot
    df = pd.DataFrame(data, columns=['Reconciliation', 'Model', 'RMSE', 'Percent Change'])

    # Custom sorting: unreconciled first, MiNT last
    method_order = sorted(df['Reconciliation'].unique(), key=lambda x: (x != 'Baseline', x == 'MiNT', x))

    # Pivot for values and annotations
    heatmap_vals = df.pivot(index='Reconciliation', columns='Model', values='Percent Change').loc[method_order]
    rmse_vals = df.pivot(index='Reconciliation', columns='Model', values='RMSE').loc[method_order]
    annot = rmse_vals.round(0).astype(int).astype(str) + "\n(" + heatmap_vals.round(1).astype(str) + "%)"

    # 6. Custom diverging colormap with light center
    colors = ["red", "floralwhite", "royalblue"]  # blue → light cream → red
    custom_cmap = mcolors.LinearSegmentedColormap.from_list("blue_cream_red", colors)
    max_abs = np.ceil(np.abs(heatmap_vals.values).max())
    # 7. Plot
    plt.figure(figsize=(12, 7))
    sns.heatmap(
        heatmap_vals,
        annot=annot,
        fmt='',
        cmap=custom_cmap,
        center=0,
        linewidths=0.5,
        linecolor='white',
        vmin=-max_abs, vmax=max_abs,
        cbar_kws={'label': '% Change from Baseline'},
        annot_kws={'fontsize': 10}
    )

    plt.title('RMSE Compared to Baseline Predictions (Lower is Better)', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.gca().grid(False)
    plt.tight_layout()
    plt.savefig('output/heatmap')
    plt.show()