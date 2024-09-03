"""Process raw LoS data.
Input files:
------------
1. <datadir>/los/<VERSION>/<SCENARIO>/National_Model__jobid_<ensemble>_nodal_globalPlotVars_selected.csv
2. <datadir>/WRZ/wrz_code.xlsx

Output files:
-------------
1. <outdir>/<VERSION>/<SCENARIO>/monthly_los_level0_melted.csv
2. <outdir>/<VERSION>/<SCENARIO>/monthly_los_level1_melted.csv
3. <outdir>/<VERSION>/<SCENARIO>/monthly_los_level2_melted.csv
4. <outdir>/<VERSION>/<SCENARIO>/monthly_los_level3_melted.csv
5. <outdir>/<VERSION>/<SCENARIO>/monthly_los_level4_melted.csv
"""
#%%
import os
os.environ['USE_PYGEOS'] = '0'
import utils
import pandas as pd

config = utils.load_config('..')
tempdir = config['paths']['tempdir']
datadir = config['paths']['datadir']
outdir = config['paths']['tempdir']
figdir = config['paths']['figdir']

SCENARIO = "ff"
VERSION = '240403'
print(outdir)

if True: # 02-09-2024
    wrz_code = pd.read_excel(os.path.join(datadir, 'WRZ', 'wrz_code.xlsx'))
    wrz_code['RZ ID'] = pd.to_numeric(wrz_code['RZ ID'])
    wrz_dict = {wrz_code: f'{rz_id:.0f}' for wrz_code, rz_id in zip(wrz_code['WREW Code'], wrz_code['RZ ID'])}

wrz_dict
# %%
if False: # 02-09-2024
    import geopandas as gpd
    wrz_code = gpd.read_file(os.path.join(datadir, 'WRZ', 'WRZ.shp'))
    wrz_dict = {wrz_code: f'{rz_id:.0f}' for wrz_code, rz_id in zip(wrz_code['WRZ_NAME'], wrz_code['RZ_ID'])}
wrz_dict
#%%
# Load the LoS data and process it
dfs_binary = []
dfs_l1 = []
dfs_l2 = []
dfs_l3 = []
dfs_l4 = []

for ensemble in range(1, 101):
    df = pd.read_csv(os.path.join(
        datadir, 'los', VERSION, SCENARIO.upper(), f'National_Model__jobid_{ensemble}_nodal_globalPlotVars_selected.csv'),
        header=1
        )
    los_cols = [col for col in df.columns if "LoS" in col]
    df = df[['Year', 'Day'] + los_cols]
    df['Date'] = pd.to_datetime(df['Year'] * 1000 + df['Day'], format='%Y%j')
    df = df.drop(columns=["Day", "Year"])
    df = df.set_index("Date")

    # group all columns in same WR'
    df.columns = df.columns.map(wrz_dict)
    valid_columns = df.columns[(df.columns != 'nan') & (~df.columns.isnull())]
    df = df[valid_columns]
    df.columns.name = "RZ_ID"
    df = df.groupby('RZ_ID', axis=1).max() # aggregate all WRz (should be all zero)
    df.columns = df.columns.astype(int)
    df = df.sort_values(by='RZ_ID', axis=1)

    # monthly counts
    df_binary = (df > 0).astype(int).groupby(pd.Grouper(freq="M")).agg(sum)
    df_l1 = (df == 1).astype(int).groupby(pd.Grouper(freq="M")).agg(sum)
    df_l2 = (df == 2).astype(int).groupby(pd.Grouper(freq="M")).agg(sum)
    df_l3 = (df == 3).astype(int).groupby(pd.Grouper(freq="M")).agg(sum)
    df_l4 = (df == 4).astype(int).groupby(pd.Grouper(freq="M")).agg(sum)
    
    df_binary['Ensemble'] = int(ensemble)
    df_l1['Ensemble'] = int(ensemble)
    df_l2['Ensemble'] = int(ensemble)
    df_l3['Ensemble'] = int(ensemble)
    df_l4['Ensemble'] = int(ensemble)
 
    dfs_binary.append(df_binary)
    dfs_l1.append(df_l1)
    dfs_l2.append(df_l2)
    dfs_l3.append(df_l3)
    dfs_l4.append(df_l4)
    
    
df_binary = pd.concat(dfs_binary, axis=0).sort_values(by=["Date", "Ensemble"])
df_l1 = pd.concat(dfs_l1, axis=0).sort_values(by=["Date", "Ensemble"])
df_l2 = pd.concat(dfs_l2, axis=0).sort_values(by=["Date", "Ensemble"])
df_l3 = pd.concat(dfs_l3, axis=0).sort_values(by=["Date", "Ensemble"])
df_l4 = pd.concat(dfs_l4, axis=0).sort_values(by=["Date", "Ensemble"])
df_binary.head()

#%%
# Melt the dfs so that they are in long format
def melt_los_df(df):
    rz_id_cols = df.columns[:-1]
    df_melted = df.reset_index()
    df_melted = df_melted.melt(id_vars=['Date', 'Ensemble'], value_vars=rz_id_cols,
                               var_name='RZ_ID', value_name='LoS', ignore_index=True)

    df_melted = df_melted[['RZ_ID', 'Date', 'Ensemble', 'LoS']]
    df_melted = df_melted.drop_duplicates()
    df_melted = df_melted.sort_values(by=['RZ_ID', "Ensemble", 'Date']).reset_index(drop=True)
    
    df_melted['Year'] = df_melted['Date'].dt.year
    df_melted['Month'] = df_melted['Date'].dt.month
    df_melted = df_melted.drop(columns='Date')
    
    df_melted = df_melted[['RZ_ID', 'Ensemble', 'Year', 'Month', 'LoS']]
    df_melted = df_melted.sort_values(by=['RZ_ID', 'Ensemble', 'Year', 'Month', 'LoS'])
    return df_melted

df_binary_melted = melt_los_df(df_binary)
df_l1_melted = melt_los_df(df_l1)
df_l2_melted = melt_los_df(df_l2)
df_l3_melted = melt_los_df(df_l3)
df_l4_melted = melt_los_df(df_l4)
df_l4_melted.head()

#%%
# Save the melted dfs
df_binary_melted.to_csv(os.path.join(outdir, VERSION, SCENARIO.lower(), 'monthly_los_level0_melted.csv'))
df_l1_melted.to_csv(os.path.join(outdir, VERSION, SCENARIO.lower(), 'monthly_los_level1_melted.csv'))
df_l2_melted.to_csv(os.path.join(outdir, VERSION, SCENARIO.lower(), 'monthly_los_level2_melted.csv'))
df_l3_melted.to_csv(os.path.join(outdir, VERSION, SCENARIO.lower(), 'monthly_los_level3_melted.csv'))
df_l4_melted.to_csv(os.path.join(outdir, VERSION, SCENARIO.lower(), 'monthly_los_level4_melted.csv'))

# %% check we still have 99 WRZs
print(df_binary_melted['RZ_ID'].nunique())
assert df_binary_melted['RZ_ID'].nunique() == 99