# %%
import os
from glob import glob
import pandas as pd
import geopandas as gpd
os.environ['USE_PYGEOS'] = '0'
import matplotlib.pyplot as plt

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
    # df['wrz'] = f.split("/")[-2]
    dfs.append(df)
df = pd.concat(dfs).reset_index(drop=False).dropna()

# %%
gdf = gpd.read_file(os.path.join(wd, "data", "input", 'WRZ', "WRZ.shp"))
boundaries = gdf.set_geometry(gdf['geometry'].boundary).to_crs(4326)
gdf = gdf.merge(df, left_on='RZ_ID', right_on='rz_id')
gdf = gdf.to_crs(4326) # nicer plotting

# %% Plotting metrics
import cartopy
import cartopy.crs as ccrs

cmap = 'YlOrRd'

# subset by indicator
indicator = 'si6'
gdf_sub = gdf[gdf['indicator'] == indicator]
# # subset by best
# aggfunc = {
#     'Brier': 'min',
#     'BCE': 'min',
#     'RMSE': 'min',
#     'Precision': 'max',
#     'Recall': 'max',
#     'F1': 'max',
#     'AUROC': 'max',
#     'geometry': 'first'
# }
# gdf_sub = gdf.groupby(['RZ_ID']).agg(aggfunc).reset_index()
# gdf_sub = gpd.GeoDataFrame(gdf_sub, geometry='geometry').set_crs(4326)

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

# %% Plotting Bernoulli coefficients
import matplotlib.colors as colors

cmap = 'coolwarm'
titles = ['Intercept', 'Current month', 'Previous 2 months', 'Previous 3 months',
          'Previous 6 months', 'Previous 9 months', 'Previous 12 months',
          'Previous 24 months', 'Previous 36 months', 'Previous 48 months']

fig, axs = plt.subplots(2, 5, figsize=(16, 8), sharex=True, sharey=True,
                        subplot_kw={'projection': ccrs.PlateCarree()})

coeffients = ['ber.intercept', 'ber', 'ber2', 'ber3',
              'ber6', 'ber9', 'ber12', 'ber24',
              'ber36', 'ber48']

for i, coef in enumerate(coeffients):
    ax = axs.flatten()[i]
    gdf.plot(coef, ax=ax, legend=True, cmap=cmap, norm=colors.CenteredNorm())
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
fig, axs = plt.subplots(2, 5, figsize=(16, 8), sharex=True, sharey=True,
                        subplot_kw={'projection': ccrs.PlateCarree()})

coefficients = ['bin.intercept', 'bin', 'bin2', 'bin3',
                'bin6', 'bin9', 'bin12', 'bin24',
                'bin36', 'bin48']

for i, coef in enumerate(coefficients):
    ax = axs.flatten()[i]
    gdf.plot(coef, ax=ax, legend=True, cmap=cmap, norm=colors.CenteredNorm())
    ax.set_title(coef)
    ax.add_feature(cartopy.feature.OCEAN, color='lightblue')
    ax.add_feature(cartopy.feature.LAND, color='tan')
    boundaries.plot(color='k', ax=ax, linewidth=0.1)
    ax.set_title(titles[i])
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.label_outer()

fig.suptitle("Binomial coefficients of ZABI GLM across the UK")

# %%
