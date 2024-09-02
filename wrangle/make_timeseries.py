"""
Input files:
------------
1. <tempdir>/<SCENARIO>/indicator_series.parquet
2. <tempdir>/<SCENARIO>/standardised_series.parquet
3. <tempdir>/<SCENARIO>/monthly_los_melted.csv


Output files:
-------------
1. <resdir>/<SCENARIO>/full_timeseries/ts.csv
2. <resdir>/<SCENARIO>/full_timeseries/ts_with.levels.csv
"""
#%%
import os
import json
os.environ['USE_PYGEOS'] = '0'
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

plot_kwargs = {'bbox_inches': "tight", 'dpi': 200}

def create_ts(df):
    df = df.copy()
    df['Date'] = pd.to_datetime(df[['Year', 'Month']].assign(day=16))
    df = df.drop(columns=['Year', 'Month'])
    ts = df.set_index('Date', drop=True)
    return ts


def subset_column(df, column, value):
    df = df[df[column] == value]
    df = df.drop(columns=[column])
    return df

def load_config(dirname=None):
    if dirname is None:
        dirname = os.getcwd()
    config_path = os.path.join(dirname, "config.json")
    with open(config_path, "r") as config_fh:
        config = json.load(config_fh)
    return config

config = load_config("..")
tempdir = config['paths']['tempdir']
resdir = config['paths']['resultsdir']
outdir = config['paths']['resultsdir']
figdir = config['paths']['figdir']

SCENARIO = 'ff'
VARIABLE = 'ep' # ['ep', 'prbc'], can't remember what prbc was for

# load indicator data
df_ind = pd.read_parquet(os.path.join(tempdir, 'indicators', SCENARIO, 'indicator_series.parquet'))
df_spi = pd.read_parquet(os.path.join(tempdir, 'indicators', SCENARIO, 'standardised_series.parquet'))

#%%
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

#%%
# severity data
def add_los_level(ts, level:int):
    level_df = pd.read_csv(os.path.join(tempdir, 'los', '240403', SCENARIO, f"monthly_los_level{level}_melted.csv"), index_col=[0])
    level_df['ensemble'] = level_df['Ensemble'].apply(lambda x: "{}{}".format(SCENARIO.upper(), x))
    level_df = level_df.drop(columns='Ensemble')
    ts_level = create_ts(level_df)
    ts_level["LoS"] = ts_level["LoS"].astype(int)
    ts = pd.merge(ts, ts_level, on=['Date', 'RZ_ID', 'ensemble'], suffixes=('', f'_l{level}'))
    return ts

level = 0
ts_los = pd.read_csv(os.path.join(tempdir, 'los', '240403', SCENARIO, f"monthly_los_level{level}_melted.csv"), index_col=[0])
ts_los['ensemble'] = ts_los['Ensemble'].apply(lambda x: "{}{}".format(SCENARIO.upper(), x))
ts_los = ts_los.drop(columns='Ensemble')
ts_los = create_ts(ts_los)
ts_los['LoS'] = ts_los['LoS'].astype(int)
ts = ts_ind.copy()
ts = pd.merge(ts, ts_spi, on=['Date', 'RZ_ID', 'ensemble'])
ts = pd.merge(ts, ts_los, on=['Date', 'RZ_ID', 'ensemble'])
ts = ts[['RZ_ID', 'ensemble', 'LoS', 'ep_total', 'anomaly_mean', 'deficit_mean',
         'anomaly_q50', 'deficit_q50', 'deficit_q75', 'deficit_q90',
         'si6', 'si12', 'si24']]

ts = add_los_level(ts, 1)
ts = add_los_level(ts, 2)
ts = add_los_level(ts, 3)
ts = add_los_level(ts, 4)
ts.head()
#%%
ts.to_csv(os.path.join(resdir, 'full_timeseries', '240403', SCENARIO, 'ts_with_levels.csv'))
# %%