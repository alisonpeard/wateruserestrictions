# %%
import os
os.environ['USE_PYGEOS'] = '0'
from glob import glob
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy
import cartopy.crs as ccrs

wd = os.path.join(os.path.expanduser("~"), "Documents", "RAPID", "correlation-analysis")
datadir = os.path.join(wd, "data", "results")
wrz_folders = os.listdir(os.path.join(datadir, "cv"))
wrz_folders = [f for f in wrz_folders if not f.startswith(".")]

# %% ----Average the metrics for each WRZ----
training = 'FF2'
trend = 'raw'
lag = 'ma.s'
toremove = ['.trend', '.raw', '.lag.', '.ma.s', '.ma.t', '.ep_total', '.si6', '.si12', '.si24']

for folder in wrz_folders:
    filedir = os.path.join(datadir, "cv", folder)
    inds = []
    for ind in ['si6', 'si12', 'si24']:
        try:
            si = pd.read_csv(os.path.join(filedir, trend, ind, lag, f"{training}.csv"), index_col=0, skipinitialspace=True)
            si['indicator'] = ind

            columns = si.columns
            for r in toremove:
                columns = [x.replace(r, '') for x in columns]
            si.columns = columns
            inds.append(si)

        except FileNotFoundError:
            pass
    if len(inds) == 0:
        continue

    df = pd.concat(inds)
    df_avg = df.groupby('indicator').mean()
    df_avg = df_avg.loc[['si6', 'si12', 'si24']]
    df_avg.to_csv(os.path.join(filedir, "mafits__avg.csv"))

# %% ----Read in the averaged metrics----
fits = []
for folder in wrz_folders:
    files = glob(os.path.join(datadir, "cv", folder, "mafits__avg.csv"))
    fits += files
fits

# %%
gdf = gpd.read_file(os.path.join(wd, "data", "input", 'WRZ', "WRZ.shp"))
gdf['wrz'] = gdf['WRZ_NAME'].str.lower().replace(' ', '_', regex=True)
gdf['wrz'] = gdf['wrz'].str.replace('.', '', regex=False)
gdf['wrz'] = gdf['wrz'].str.replace('-', '_', regex=False)
gdf['wrz'] = gdf['wrz'].str.replace('/', '', regex=False)
gdf['wrz'] = gdf['wrz'].str.replace('(', '', regex=False)
gdf['wrz'] = gdf['wrz'].str.replace(')', '', regex=False)
key_df = gdf[['RZ_ID', 'wrz']]
key_df.columns = 'rz_id', 'wrz'
key_df.to_csv(os.path.join(datadir, "..", "wrz_key.csv"), index=False)

# %%
dfs = []
for f in fits:
    df = pd.read_csv(f, index_col=0, skipinitialspace=True)
    df['wrz'] = f.split("/")[-2]
    dfs.append(df)
df = pd.concat(dfs).reset_index(drop=False)#.dropna()

# %%
gdf = gpd.read_file(os.path.join(wd, "data", "input", 'WRZ', "WRZ.shp"))
boundaries = gdf.set_geometry(gdf['geometry'].boundary).to_crs(4326)
gdf = gdf.merge(df, left_on='RZ_ID', right_on='rz_id')
gdf = gdf.to_crs(4326) # nicer plotting

# %% load key for missing data
all_wrz = gpd.read_file(os.path.join(wd, "data", "input", 'WRZ', "WRZ.shp")).to_crs(4326)
missing = pd.read_csv(os.path.join(datadir, "full_timeseries", "240403", "missingdata.csv"))
missing = missing[missing['scenario'] == training[:-1].lower()]
missing = missing[missing['model'] == 1].set_index('RZ_ID')
missing['geometry'] = all_wrz.set_index('RZ_ID')['geometry']
missing = gpd.GeoDataFrame(missing, geometry='geometry').set_crs(4326)
present = missing.index.tolist()

#%% check all available data is plotted
if False:
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), subplot_kw={'projection': ccrs.PlateCarree()})

    ax = axs[0]
    gdf.plot(color='red', ax=ax)
    boundaries.plot(color='k', ax=ax, linewidth=0.1)
    ax.add_feature(cartopy.feature.OCEAN, color='lightblue')
    ax.add_feature(cartopy.feature.LAND, color='tan')
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title('Modelled data')

    ax = axs[1]
    missing.plot(color='red', ax=ax)
    boundaries.plot(color='k', ax=ax, linewidth=0.1)
    ax.add_feature(cartopy.feature.OCEAN, color='lightblue')
    ax.add_feature(cartopy.feature.LAND, color='tan')
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title('Data that should be modelled')

    not_modelled = [x for x in present if x not in gdf['RZ_ID'].unique().tolist()]
    falsely_modelled = [x for x in gdf['RZ_ID'].unique().tolist() if x not in present]

    print(f"Number of WRZ not modelled: {len(not_modelled)}: {not_modelled}")
    print(f"Number of WRZ falsely modelled: {len(falsely_modelled)}: {falsely_modelled}")
    # compare to what is present
    modelled = gdf['RZ_ID'].unique().tolist()
    missing = [x for x in present if x not in modelled]
    gdf = gdf[gdf['RZ_ID'].isin(present)] # make sure only count valid wrz

# %% Option to model a specific WRZ
if False:
    rz_id = 127
    gdf_sub = all_wrz.copy()
    gdf_sub['RZ_ID'] = (gdf_sub['RZ_ID'] == rz_id).astype(int)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5), subplot_kw={'projection': ccrs.PlateCarree()})
    gdf_sub.plot('RZ_ID', categorical=True, legend=True, ax=ax, cmap='coolwarm')
    boundaries.plot(color='k', ax=ax, linewidth=0.1)
    ax.add_feature(cartopy.feature.OCEAN, color='lightblue')
    ax.add_feature(cartopy.feature.LAND, color='tan')
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title('Modelled data')


# %% Plotting metrics
cmap = 'YlOrRd'

if True:
    # subset by indicator
    indicator = 'si24'
    gdf_sub = gdf[gdf['indicator'] == indicator]
else:
    # subset by best
    aggfunc = {
        'Brier': 'min',
        'BCE': 'min',
        'RMSE': 'min',
        'Precision': 'max',
        'Recall': 'max',
        'F1': 'max',
        'AUROC': 'max',
        'geometry': 'first'
    }
    gdf_sub = gdf.groupby(['RZ_ID']).agg(aggfunc).reset_index()
    gdf_sub = gpd.GeoDataFrame(gdf_sub, geometry='geometry').set_crs(4326)

fig, axs = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True,
                        subplot_kw={'projection': ccrs.PlateCarree()})
ax = axs[0, 0]
gdf_sub.plot('Brier', ax=ax, legend=True, cmap=f"{cmap}_r")
ax.set_title('Brier Score')

ax = axs[0, 1]
gdf_sub.plot('BCE', ax=ax, legend=True, cmap=f"{cmap}_r")
ax.set_title('Binary Cross-entropy')

ax = axs[0, 2]
gdf_sub.plot('RMSE', ax=ax, legend=True, cmap=f"{cmap}_r")
ax.set_title('Root-mean-squared Error (days)')

ax = axs[1, 2]
gdf_sub.plot('Precision', ax=ax, legend=True, cmap=cmap)
ax.set_title('Precision')

ax = axs[1, 0]
gdf_sub.plot('Recall', ax=ax, legend=True, cmap=cmap)
ax.set_title('Recall')

# ax = axs[1, 1]
# gdf.plot('F1', ax=ax, legend=True, cmap=cmap)
# ax.set_title('F1 score')

ax = axs[1, 1]
gdf_sub.plot('AUROC', ax=ax, legend=True, cmap=cmap)
ax.set_title('AUR-ROC')

fig.suptitle("Predictive power of ZABI GLM across the UK")

for ax in axs.flatten():
    ax.add_feature(cartopy.feature.OCEAN, color='lightblue')
    ax.add_feature(cartopy.feature.LAND, color='tan')
    boundaries.plot(color='k', ax=ax, linewidth=0.1)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.label_outer()

# %% ----Plotting Bernoulli coefficients----
# set max columns to infinity
cmap = 'coolwarm'
titles = ['Intercept', 'Current month', 'Previous 2 months', 'Previous 3 months',
          'Previous 6 months', 'Previous 9 months', 'Previous 12 months',
          'Previous 24 months', 'Previous 36 months', 'Previous 48 months']

fig, axs = plt.subplots(2, 4, figsize=(16, 8), sharex=True, sharey=True,
                        subplot_kw={'projection': ccrs.PlateCarree()})

coeffients = ['ber.intercept', 'ber', 'ber2', 'ber3',
              'ber6', 'ber9', 'ber12', 'ber24']

for i, coef in enumerate(coeffients):
    ax = axs.flatten()[i]
    gdf_sub.plot(coef, ax=ax, legend=True, cmap=cmap, norm=colors.CenteredNorm())
    ax.set_title(coef)
    ax.add_feature(cartopy.feature.OCEAN, color='lightblue')
    ax.add_feature(cartopy.feature.LAND, color='tan')
    boundaries.plot(color='k', ax=ax, linewidth=0.1)
    ax.set_title(titles[i])
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.label_outer()

fig.suptitle("Bernoulli coefficients of ZABI GLM across the UK");

# %% Plotting Binomial
fig, axs = plt.subplots(2, 4, figsize=(16, 8), sharex=True, sharey=True,
                        subplot_kw={'projection': ccrs.PlateCarree()})

coefficients = ['bin.intercept', 'bin', 'bin2', 'bin3',
                'bin6', 'bin9', 'bin12', 'bin24']

for i, coef in enumerate(coefficients):
    ax = axs.flatten()[i]
    gdf_sub.plot(coef, ax=ax, legend=True, cmap=cmap, norm=colors.CenteredNorm())
    ax.set_title(coef)
    ax.add_feature(cartopy.feature.OCEAN, color='lightblue')
    ax.add_feature(cartopy.feature.LAND, color='tan')
    boundaries.plot(color='k', ax=ax, linewidth=0.1)
    ax.set_title(titles[i])
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.label_outer()
fig.suptitle("Binomial coefficients of ZABI GLM across the UK")

# %% check who is missing
codes = pd.read_excel(os.path.join(wd, "data", "input", "WRZ", "WRZ_code.xlsx"))
code_list = codes['RZ ID'].dropna().unique().tolist()
code_list = [int(x) for x in code_list]
result_codes = gdf['RZ_ID'].unique().tolist()
missing_codes = [x for x in code_list if x not in result_codes]
codes[codes['RZ ID'].isin(missing_codes)]
print(f"Missing codes: {missing_codes}")
print(f"Number of missing codes: {len(missing_codes)}")
# %%

