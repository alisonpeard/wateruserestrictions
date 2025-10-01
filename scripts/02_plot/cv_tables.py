"""
Create summary table of metrics for different indicators.
"""
# %%
import os
import pandas as pd
import dataframe_image as dfi
import numpy as np

pd.set_option('display.precision', 4)

scenario = 'ff'
wrz = ['london', 'united_utilities_grid', 'ruthamford_north'][2]
wd = os.path.join(os.path.expanduser("~"), "Documents", "drought-indicators", "analysis")
datadir = os.path.join(wd, "data", "results")
ensembledir = os.path.join(datadir, "cv", wrz)
figdir = '/Users/alison/Documents/drought-indicators/analysis/figures'

# %%
dfs = []
toremove = ['.trend', '.raw', '.lag.', '.ma.s', '.ma.t', '.ep_total', '.si6', '.si12', '.si24']

# for display
ind_rename = {'ep_total': 'EP', 'si6': 'SI6', 'si12': 'SI12', 'si24': 'SI24'}
trend_rename = {'trend': 'Yes', 'raw': 'No'}
lag_rename = {'lag.': 'Pointwise', 'ma.s': 'Simple MA', 'ma.t': 'Triangular MA'}
column_rename = {'CRPS..WUR.days.': 'CRPS'}

for root, dirs, files in os.walk(ensembledir):
    for file in files:
        if file.endswith(f"{scenario}.csv"):
        # if file.endswith("bootstrap.csv"):
            df = pd.read_csv(os.path.join(root, file), index_col=0, skipinitialspace=True)
            df = df.replace('NA', pd.NA)
            columns = df.columns

            for r in toremove:
                columns = [x.replace(r, '') for x in columns]

            df.columns = columns
            df = df.rename(columns=column_rename)

            df['Lag type'] = lag_rename[root.split('/')[-1]]
            df['Indicator'] = ind_rename[root.split('/')[-2]]
            df['Detrended'] = trend_rename[root.split('/')[-3]]
            df['WRZ'] = root.split('/')[-4].title()
            dfs.append(df)

# %%
def order_columns(df):
    df['Indicator'] = pd.Categorical(df['Indicator'], categories=['EP', 'SI6', 'SI12', 'SI24'], ordered=True)
    df["Detrended"] = pd.Categorical(df["Detrended"], categories=["Yes", "No"], ordered=True)
    df = df.sort_values(['WRZ', 'Indicator', 'Detrended', 'Lag type'])
    return df

# %% look at scores
df = pd.concat(dfs).reset_index(drop=True)
df = order_columns(df)

df = df[df["Detrended"] == "No"].copy().drop(columns=['Detrended'])
df = df[df["Lag type"].isin(['Pointwise', 'Simple MA'])].copy()
df = df.groupby(['WRZ', 'Indicator', 'Lag type']).mean()[['BCE', 'Brier', 'AUROC', 'F1', 'RMSE', 'CRPS']]

metrics_to_minimize = ['BCE', 'Brier', 'RMSE', 'CRPS']
metrics_to_maximize = ['AUROC', 'F1']

cmap = "Blues"
df_styled = df.style.background_gradient(
    cmap=f"{cmap}_r", subset=metrics_to_minimize
    ).background_gradient(
        cmap=cmap, subset=metrics_to_maximize
        ).format(
            precision=4
            )

dfi.export(df_styled, os.path.join(figdir, f'indicators_{wrz}_{scenario}.png'),
           table_conversion='matplotlib',
           dpi=300)

df_styled.to_excel(os.path.join(figdir, f'indicators_{wrz}_{scenario}.xlsx'), float_format="%.4f")
df_styled

# %%
def bootstrap_select(data):
    count = np.sum(data != 0)
    proportion = int(count / len(data) * 100)
    return proportion

def bootstrap_mean(data):
    data = data[data != 0]
    if len(data) == 0:
        return 0
    else:
        return data.mean()

i = 0
aggfunc = [bootstrap_select, bootstrap_mean][i]

df = pd.concat(dfs).reset_index(drop=True)
df = order_columns(df)
coefs = ['ber.intercept', 'ber', 'ber3', 'ber6', 'ber12','ber24']

df = df[df["Detrended"] == "No"].copy().drop(columns=['Detrended'])
df = df[df["Lag type"].isin(['Pointwise', 'Simple MA'])].copy()
df = df.groupby(['WRZ', 'Indicator', 'Lag type'])[coefs].agg(aggfunc)

df_abs = df[coefs].abs()
absmax = df[coefs].abs().max().max()
vmin = [50, 0][i]
vmax = absmax
gmap = df_abs.values

kwargs = [{"vmin": vmin, "vmax": vmax}, {"gmap": gmap}][i]

df_styled = df.style.background_gradient(
    cmap='Blues', subset=coefs, axis=None, **kwargs
    ).format(
        precision=2
        )

dfi.export(df_styled, os.path.join(figdir, f'bercoefs_{wrz}_{scenario}.png'),
           table_conversion='matplotlib',
           dpi=300)

df_styled.to_excel(os.path.join(figdir, f'bercoefs_{wrz}_{scenario}.xlsx'), float_format="%.4f")
df_styled
# %%
i = 0
aggfunc = [bootstrap_select, bootstrap_mean][i]

df = pd.concat(dfs).reset_index(drop=True)
df = order_columns(df)
coefs = ['bin.intercept', 'bin', 'bin3', 'bin6','bin12','bin24']

df = df[df["Detrended"] == "No"].copy().drop(columns=['Detrended'])
df = df[df["Lag type"].isin(['Pointwise', 'Simple MA'])].copy()
df["ncoefs"] = df[coefs].astype(bool).sum(axis=1)
# coefs += ['ncoefs']

df = df.groupby(['WRZ', 'Indicator', 'Lag type'])[coefs].agg({
    **{c: aggfunc for c in coefs},
    # 'ncoefs': 'mean'
})

df_abs = df[coefs].abs()
absmax = df[coefs].abs().max().max()
vmin = 0
vmax = absmax
gmap = df_abs

def normalise(x):
    return (x - x.min()) / (x.max() - x.min())

# gmap["ncoefs"] = 0.5 * absmax * normalise(gmap["ncoefs"])

kwargs = [{"vmin": vmin, "vmax": vmax}, {"gmap": gmap.values}][1]

df_styled = df.style.background_gradient(
    cmap='Blues', subset=coefs, axis=None, **kwargs
    ).format(
        precision=2
        )

dfi.export(df_styled, os.path.join(figdir, f'bincoefs_{wrz}_{scenario}.png'),
           table_conversion='matplotlib',
           dpi=300)

df_styled.to_excel(os.path.join(figdir, f'bincoefs_{wrz}_{scenario}.xlsx'), float_format="%.4f")
df_styled
# %%