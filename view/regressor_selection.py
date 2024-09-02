# %%
import os
import pandas as pd
import dataframe_image as dfi

wd = os.path.join(os.path.expanduser("~"), "Documents", "RAPID", "correlation-analysis")
datadir = os.path.join(wd, "data", "results")
ensembledir = os.path.join(datadir, "cv", "london")
figdir = '/Users/alison/Documents/RAPID/correlation-analysis/figures'

# %%
pd.set_option('display.precision', 4)
dfs = []
toremove = ['.trend', '.raw', '.lag.', '.ma.s', '.ma.t', '.ep_total', '.si6', '.si12', '.si24']

# for display
ind_rename = {'ep_total': 'EP', 'si6': 'SI6', 'si12': 'SI12', 'si24': 'SI24'}
trend_rename = {'trend': 'Yes', 'raw': 'No'}
lag_rename = {'lag.': 'Pointwise', 'ma.s': 'Simple MA', 'ma.t': 'Triangular MA'}

for root, dirs, files in os.walk(ensembledir):
    for file in files:
        if file.endswith("FF2.csv"):
            df = pd.read_csv(os.path.join(root, file), index_col=0, skipinitialspace=True)
            df = df.replace('NA', pd.NA)
            columns = df.columns
            for r in toremove:
                columns = [x.replace(r, '') for x in columns]

            df.columns = columns

            df['Lag type'] = lag_rename[root.split('/')[-1]]
            df['Indicator'] = ind_rename[root.split('/')[-2]]
            df['Trend'] = trend_rename[root.split('/')[-3]]
            df['WRZ'] = root.split('/')[-4].title()
            dfs.append(df)

df = pd.concat(dfs).reset_index(drop=True)#.dropna()

df = df.groupby(['WRZ', 'Indicator', 'Trend', 'Lag type']).mean()[['BCE', 'Brier', 'AUROC', 'F1', 'RMSE']]# , 'Recall', 'Precision']]
df = df.loc[:, ['EP', 'SI6', 'SI12', 'SI24'], :, :]
df_styled = df.style.background_gradient(
    cmap='RdYlGn_r', subset=['BCE', 'Brier', 'RMSE']
    ).background_gradient(
        cmap='RdYlGn', subset=['AUROC', 'F1'] # , 'Precision', 'Recall'
        ).format(
            precision=4
            )
dfi.export(df_styled, os.path.join(figdir, 'modelselection.png'),
           table_conversion='matplotlib',
           dpi=300)
df_styled.to_excel(os.path.join(figdir, 'modelselection.xlsx'), float_format="%.4f")
df_styled
# %%
