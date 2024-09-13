# %%
eps = 0.01 # threshold for excluding zones

# %%
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import dataframe_image as dfi

wd = os.path.join(os.path.expanduser("~"), "Documents", "RAPID", "correlation-analysis")
datadir = os.path.join(wd, "data", "results", "full_timeseries", "240403")
figdir = '/Users/alison/Documents/RAPID/correlation-analysis/figures'

# %%
scenarios = ['bs', 'nf', 'ff']
dfs = []
for scen in scenarios:
    ts = pd.read_csv(os.path.join(datadir, scen, f"ts_with_levels.csv"))[['Date', 'ensemble', 'RZ_ID', 'LoS']]
    ts['Date'] = pd.to_datetime(ts['Date'])
    ts['ensemble'] = ts['ensemble'].apply(lambda x: x.replace(scen.upper(), '')).astype(int)
    ts["LoS_binary"] = (ts["LoS"] > 0).astype(int)

    ts = ts.sort_values(by=['RZ_ID', 'ensemble', 'Date'])
    ts = ts[['RZ_ID', 'ensemble', 'LoS_binary']].groupby(['RZ_ID', 'ensemble']).agg({'LoS_binary': [sum, pd.Series.count]}).sort_index()
    ts.columns = ['sum', 'count']
    ts['ratio'] = ts['sum'] / ts['count']

    ts_summary = ts[['ratio']].groupby('RZ_ID').agg({'ratio': ['min', 'max', 'std', 'mean']})
    ts_summary['scenario'] = scen
    dfs.append(ts_summary)
ts_summary = pd.concat(dfs).groupby(['RZ_ID', 'scenario']).mean()

# reorder index
scenario_index = pd.CategoricalIndex(ts_summary.index.levels[1], categories=scenarios, ordered=True)
ts_summary.index = ts_summary.index.set_levels(scenario_index, level=1)
ts_summary = ts_summary.sort_index(level=1).sort_index(level=0)
ts_summary

# %% rules 
idx = pd.IndexSlice
ts_summary['model'] = [1] * len(ts_summary)
ts_summary['model'] = ts_summary[idx['ratio', 'mean']].apply(lambda x: 0 if x < eps else 2 if x > 1 - eps else 1)


# %% for display
cmap = plt.cm.twilight.copy()
cmap.set_under(color='wheat')
cmap.set_over(color='wheat')

cmap2 = plt.cm.twilight.copy()
cmap2.set_under(color='wheat')
cmap2.set_over(color=cmap2(0.25))

cmap2 = mpl.colors.ListedColormap(['wheat', cmap(0.25), cmap(0.5)])
ts_styled = ts_summary.style.background_gradient(
        cmap=cmap2, subset=['model'], vmin=-1, vmax=3).background_gradient(
            cmap=cmap, vmin=eps, vmax=1-eps, subset=[
                ('ratio', 'mean'),
                ('ratio', 'std'),
                ('ratio', 'min'),
                ('ratio', 'max')]
                ).format(precision=4)

ts_styled.to_excel(os.path.join(figdir, 'missingdata.xlsx'), float_format="%.4f")
ts_styled

# %% Plot these on a map
os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.feature as feature

wrz = gpd.read_file(os.path.join(wd, 'data', 'input', 'WRZ', 'WRZ.shp'))
wrz = wrz.to_crs(4326)

ts_summary = ts_summary.reset_index().set_index('RZ_ID')
ts_summary['geometry'] = wrz.set_index('RZ_ID')['geometry']
ts_summary = gpd.GeoDataFrame(ts_summary, geometry='geometry').set_crs(4326)
# %%
legend = {0: f'<{eps:.1%} average WUR', 1: 'Balanced classes', 2: f'>{1-eps:.1%} average WUR'}
ts_summary['alias'] = ts_summary['model'].apply(lambda x: legend[x])

# %% define colorbar and legend
import matplotlib.ticker as mticker

categories = [*legend.values()]
cmap3 = cmap2.copy()
norm = mpl.colors.BoundaryNorm(range(len(categories) + 1), cmap3.N)

def plot_background(wrz, ax, title):
    wrz.boundary.plot(ax=ax, linewidth=0.1, color='k')
    ax.add_feature(feature.BORDERS, color='k', linewidth=0.1)
    ax.add_feature(feature.LAND, color='tan')
    ax.add_feature(feature.OCEAN, color='lightblue')
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

fig, axs = plt.subplots(1, 3, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(15, 5))
for ax, scen in zip(axs, scenarios):
    plot_background(wrz, ax, scen.upper())
    ts_sub = ts_summary[ts_summary['scenario'] == scen]
    im = ts_sub.plot('model', ax=ax, legend=True, categorical=True,
                     cmap=cmap3, norm=norm,
                     legend_kwds={'labels': categories}
                     )
    
    number_excluded = (ts_sub['model'] != 1).sum()
    ax.set_title(f"{scen.upper()} ({number_excluded} excluded)")

fig.suptitle("Missing data in the UK")

# %% Also plot the mean %WUR for each scenario
fig, axs = plt.subplots(1, 3, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(15, 5))
for ax, scen in zip(axs, scenarios):
    plot_background(wrz, ax, scen.upper())
    ts_sub = ts_summary[ts_summary['scenario'] == scen]
    im = ts_sub.plot(('ratio', 'mean'), ax=ax, cmap='YlOrRd', legend=True)

fig.suptitle("Average %WUR across the UK")

# %% Plot the standard deviation of the %WUR for each scenario
fig, axs = plt.subplots(1, 3, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(15, 5))
for ax, scen in zip(axs, scenarios):
    plot_background(wrz, ax, scen.upper())
    ts_sub = ts_summary[ts_summary['scenario'] == scen]
    im = ts_sub.plot(('ratio', 'std'), ax=ax, cmap='YlOrRd', legend=True)

fig.suptitle("Standard deviation of %WUR across the UK")

# %% Save results
ts_summary.columns = ['scenario', 'min', 'max', 'std', 'mean', 'model', 'geometry', 'comment']
ts_summary = ts_summary.drop(columns='geometry')
ts_summary.to_csv(os.path.join(datadir, 'missingdata.csv'))

# %%