# %%
import os
import pandas as pd
import geopandas as gpd

wd = '/Users/alison/Documents/RAPID/correlation-analysis/data'

full_ts = pd.read_csv('/Users/alison/Documents/RAPID/correlation-analysis/data/results/full_timeseries/240403/ff/ts_with_levels.csv')
wrz = gpd.read_file('/Users/alison/Documents/RAPID/correlation-analysis/data/input/WRZ/WRZ.shp')
monthly_l0 = pd.read_csv('/Users/alison/Documents/RAPID/correlation-analysis/data/temp/los/240403/ff/monthly_los_level0_melted.csv')
monthly_l1 = pd.read_csv('/Users/alison/Documents/RAPID/correlation-analysis/data/temp/los/240403/ff/monthly_los_level1_melted.csv')
monthly_l2 = pd.read_csv('/Users/alison/Documents/RAPID/correlation-analysis/data/temp/los/240403/ff/monthly_los_level2_melted.csv')
monthly_l3 = pd.read_csv('/Users/alison/Documents/RAPID/correlation-analysis/data/temp/los/240403/ff/monthly_los_level3_melted.csv')
monthly_l4 = pd.read_csv('/Users/alison/Documents/RAPID/correlation-analysis/data/temp/los/240403/ff/monthly_los_level4_melted.csv')

print(wrz['RZ_ID'].nunique())
print(full_ts['RZ_ID'].nunique())
print(monthly_l0['RZ_ID'].nunique())
# %%
# see which ones aren't there
wrz[~wrz['RZ_ID'].isin(full_ts['RZ_ID'])]['RZ_ID'].unique()

# %%
# do the same for the monthly data
wrz[~wrz['RZ_ID'].isin(monthly_l0['RZ_ID'])]['RZ_ID'].unique()
# %%
monthly_l1[~monthly_l1['RZ_ID'].isin(full_ts['RZ_ID'])]['RZ_ID'].unique()
# %%
monthly_l2[~monthly_l2['RZ_ID'].isin(full_ts['RZ_ID'])]['RZ_ID'].unique()
# %%
monthly_l3[~monthly_l3['RZ_ID'].isin(full_ts['RZ_ID'])]['RZ_ID'].unique()
# %%
monthly_l4[~monthly_l4['RZ_ID'].isin(full_ts['RZ_ID'])]['RZ_ID'].unique()
# %%
