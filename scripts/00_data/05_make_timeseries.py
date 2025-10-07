"""
Load the processed LoS data and combine with indicator data to create a full timeseries.

Params:
    - config["config"]["scenario"]
    - config["config"]["variable"]

Inputs:
    - os.path.join(config['paths']['tempdir'], 'indicators', {scenario}, 'indicator_series.parquet')
    - os.path.join(config['paths']['tempdir'], 'indicators', SCENARIO, 'standardised_series.parquet')
    - os.path.join(config['paths']['tempdir'], 'los', {scenario}, f"monthly_los_level{level}_melted.csv")

Outputs:
    - os.path.join(config['paths']['resultsdir'], 'ts_with_levels.csv')
"""
#%%
import os

os.environ['USE_PYGEOS'] = '0'

import pandas as pd
import utils

plot_kwargs = {'bbox_inches': "tight", 'dpi': 200}


def create_ts(df):
    """Turn dataframe into a timeseries with a datetime index."""
    df = df.copy()
    df['Date'] = pd.to_datetime(df[['Year', 'Month']].assign(day=1))
    df["Date"] = df["Date"] + pd.offsets.MonthEnd(0)
    df = df.drop(columns=['Year', 'Month'])
    ts = df.set_index('Date', drop=True)
    return ts


def subset_column(df, column, value):
    """Wrapper function for subsetting a dataframe by a column value."""
    df = df[df[column] == value]
    df = df.drop(columns=[column])
    return df


def add_los_level(ts, level:int, dir:str, scenario:str=None):
    level_df = pd.read_csv(os.path.join(dir, f"monthly_los_level{level}_melted.csv"), index_col=[0])
    level_df['ensemble'] = level_df['Ensemble'].apply(lambda x: "{}{}".format(scenario, x))
    level_df = level_df.drop(columns='Ensemble')
    ts_level = create_ts(level_df)
    ts_level["LoS"] = ts_level["LoS"].astype(int)
    ts = pd.merge(ts, ts_level, on=['Date', 'RZ_ID', 'ensemble'], suffixes=('', f'_l{level}'))
    return ts


def main(config):
    
    indir = config['paths']['tempdir']
    outdir = config['paths']['resultsdir']

    SCENARIO = config["config"]["scenarios"][config["config"]["scenario"]]
    VARIABLE = config["config"]["variables"][config["config"]["variable"]]

    outdir = os.path.join(outdir, 'full_timeseries', SCENARIO)
    os.makedirs(outdir, exist_ok=True)

    # load indicator data
    df_ind = pd.read_parquet(os.path.join(indir, 'indicators', SCENARIO, 'indicator_series.parquet'))
    df_spi = pd.read_parquet(os.path.join(indir, 'indicators', SCENARIO, 'standardised_series.parquet'))

    # subset to just the variable, scenario and buffer we want
    df_ind = subset_column(df_ind, 'Variable', VARIABLE)
    df_ind = subset_column(df_ind, 'scenario', SCENARIO.upper())
    df_ind = subset_column(df_ind, 'buffer', 500)

    df_spi = subset_column(df_spi, 'Variable', VARIABLE)
    df_spi = subset_column(df_spi, 'scenario', SCENARIO.upper())
    df_spi = subset_column(df_spi, 'buffer', 500)

    df_ind = df_ind.rename(columns={'Value': f'{VARIABLE}_total'})
    df_spi = df_spi.rename(columns={'Value': f'{VARIABLE}_total'})

    ts_ind = create_ts(df_ind)
    ts_spi = create_ts(df_spi)

    # add LoS severity data
    level = 0
    ts_los = pd.read_csv(os.path.join(indir, 'los', SCENARIO, f"monthly_los_level{level}_melted.csv"), index_col=[0])
    ts_los['ensemble'] = ts_los['Ensemble'].apply(lambda x: "{}{}".format(SCENARIO.upper(), x))
    ts_los = ts_los.drop(columns='Ensemble')
    ts_los = create_ts(ts_los)
    ts_los['LoS'] = ts_los['LoS'].astype(int)
    ts = ts_ind.copy()
    ts = pd.merge(ts, ts_spi, on=['Date', 'RZ_ID', 'ensemble'])
    ts = pd.merge(ts, ts_los, on=['Date', 'RZ_ID', 'ensemble'])
    
    ts = ts[[
        'RZ_ID', 'ensemble', 'LoS', f'{VARIABLE}_total', 'anomaly_mean', 'deficit_mean',
        'anomaly_q50', 'deficit_q50', 'deficit_q75', 'deficit_q90',
        'si6', 'si12', 'si24'
        ]]

    los_dir = os.path.join(indir, 'los', SCENARIO)

    ts = add_los_level(ts, 1, los_dir, SCENARIO.upper())
    ts = add_los_level(ts, 2, los_dir, SCENARIO.upper())
    ts = add_los_level(ts, 3, los_dir, SCENARIO.upper())
    ts = add_los_level(ts, 4, los_dir, SCENARIO.upper())

    # save time series
    ts.to_csv(os.path.join(outdir, 'ts_with_levels.csv'))

    print(f"Saved full timeseries to {os.path.join(outdir, 'ts_with_levels.csv')}")


if __name__ == "__main__":
    # setup
    wd = os.path.join(os.path.dirname(__file__), "../..")
    os.chdir(wd); print(f"Working directory: {os.getcwd()}")
    config = utils.load_config()
    main(config)
# %%