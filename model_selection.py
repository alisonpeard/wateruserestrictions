#%%
import os
import numpy as np
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def whitespace_remover(dataframe):
    for i in dataframe.columns:
        if dataframe[i].dtype == 'object':
            dataframe[i] = dataframe[i].map(str.strip)
        else:
            pass
 
wrz = 'london'
datadir = f"/Users/alison/Documents/RAPID/correlation-analysis/data/results/cv_glmnet/{wrz}/"
#%%
files = glob.glob(os.path.join(datadir, "fits*.csv"))

N = len(files)
print(f"Found {N} files")
# load all the files and calculate average and std for each entry
dfs = []
for f in files:
    df = pd.read_csv(f, index_col=0, skipinitialspace=True)
    whitespace_remover(df)
    df = df.replace('NA', np.nan)
    # df = df.fillna(0.) # not super because minimising some values
    df = df.astype(float)
    if len(df) > 0:
        dfs.append(df)

df = pd.concat(dfs)
# %%

df = df.reset_index()
df = df.rename(columns={'index': 'indicator'})
# %%
fig, axs = plt.subplots(1, 4, figsize=(20, 5))

violin_kws = {'palette': 'colorblind', 'linewidth': .5}

df_plot = df.melt(id_vars='indicator', value_vars=['BCE', 'Brier'], var_name='Binary metric')
sns.violinplot(data=df_plot, x='indicator', y='value', hue='Binary metric', ax=axs[0], **violin_kws)

df_plot = df.melt(id_vars='indicator', value_vars=['AUROC', 'Precision', 'Recall', 'F1', 'F2'], var_name='Confusion metric')
sns.violinplot(data=df_plot, x='indicator', y='value', hue='Confusion metric', ax=axs[1], **violin_kws)

df_plot = df.melt(id_vars='indicator', value_vars=['zibi.r2'], var_name='R-squared')
sns.violinplot(data=df_plot, x='indicator', y='value', hue='R-squared', legend=False, ax=axs[2], **violin_kws)

df_plot = df.melt(id_vars='indicator', value_vars=['zibi.rmse'], var_name='RMSE')
sns.violinplot(data=df_plot, x='indicator', y='value', hue='RMSE', legend=False, ax=axs[3], **violin_kws)

plt.suptitle(f"ZABI model metrics for {wrz.title().replace('_', ' ')}")
del df_plot

# %%
df_avg = df.groupby("indicator").mean()
df_std = df.groupby("indicator").std()
df_q25 = df.groupby("indicator").quantile(0.25)
df_q75 = df.groupby("indicator").quantile(0.75)

df_avg = df_avg.round(4)
df_std = df_std.round(4)
df_q25 = df_q25.round(4)
df_q75 = df_q75.round(4)

# %% save
df_avg.to_csv(os.path.join(datadir, "glmnet_mean.csv"))
df_std.to_csv(os.path.join(datadir, "glmnet_std.csv"))
df_q25.to_csv(os.path.join(datadir, "glmnet_q25.csv"))
df_q75.to_csv(os.path.join(datadir, "glmnet_q75.csv"))

# %% --- Save metrics dataframes to paste into Appendex ----
metrics = ['BCE', 'Brier', 'AUROC', 'Precision', 'Recall', 'F1', 'F2', 'bin.rmse', 'zibi.rmse']
print(df_avg[metrics])
print(df_std[metrics])
df_avg[metrics].replace(0, '').to_csv(os.path.join(datadir, "glmnet_metrics_mean.csv"))
df_std[metrics].replace(0, '').to_csv(os.path.join(datadir, "glmnet_metrics_std.csv"))
df_q25[metrics].replace(0, '').to_csv(os.path.join(datadir, "glmnet_metrics_q25.csv"))
df_q75[metrics].replace(0, '').to_csv(os.path.join(datadir, "glmnet_metrics_q75.csv"))

# %% ---- Save coefficient dataframes to paste into Appendex ----
bernoulli = ['ber.intercept', 'ber.l0', 'ber.l1', 'ber.l2', 'ber.l3', 'ber.l4', 'ber.l5', 'ber.l6', 'ber.ma']
binomial = ['bin.intercept', 'bin.l0', 'bin.l1', 'bin.l2', 'bin.l3', 'bin.l4', 'bin.l5', 'bin.l6', 'bin.ma']
print(df_avg[bernoulli])
print(df_avg[binomial])
df_avg[bernoulli].replace(0, '').to_csv(os.path.join(datadir, "glmnet_bernoulli_mean.csv"))
df_avg[binomial].replace(0, '').to_csv(os.path.join(datadir, "glmnet_binomial_mean.csv"))

# %%
# find seed corresponding to lowest zibi.rmse
scores = {}
for file, _df in zip(files, dfs):
    seed = int(file.split('_')[-1].split('.')[0])
    rmse = _df.loc['si6', 'zibi.rmse']
    scores[seed] = rmse
scores
scores_df = pd.DataFrame.from_dict(scores, orient='index', columns=['zibi.rmse'])
scores_df.sort_values(by='zibi.rmse', ascending=True)

# %% ---- Heatmaps of coefficients ----
# prep for aesthetics
xlabels = [x.replace('ber.l', 'Lag ') for x in bernoulli]
xlabels = [x.replace('ber.i', 'I') for x in xlabels]
xlabels = [x.replace('ber.ma', 'Moving Avg') for x in xlabels]    

ylabels = [*df_avg.index]
ylabels = [y.replace('_', ' ').title() for y in ylabels]
ylabels

# change plot style so there is no grid in the background
sns.set_style("white")

# set cmap center to 0
cmap = 'bwr'
vmin = min(df_avg[bernoulli].min().min(), df_avg[binomial].min().min())
vmax = max(df_avg[bernoulli].max().max(), df_avg[binomial].max().max())
cmap = sns.diverging_palette(220, 20, as_cmap=True)
norm = plt.Normalize(vmin=vmin, vmax=vmax)

# plot heatmaps
fig, axs = plt.subplots(1, 2, figsize=(20, 6))
sns.heatmap(df_avg[bernoulli].replace(0, np.nan), annot=True, ax=axs[0],
            cmap=cmap, norm=norm, center=0, cbar_kws={'label': 'Coefficient'})
sns.heatmap(df_avg[binomial].replace(0, np.nan), annot=True, ax=axs[1],
            cmap=cmap, norm=norm, center=0, cbar_kws={'label': 'Coefficient'})

# labels
axs[0].set_title("Bernoulli coefficients")
axs[1].set_title("Binomial coefficients")
for ax in axs:
    ax.set_xlabel('Lag')
    ax.set_xticklabels(xlabels, rotation=45, ha='right')
    ax.set_yticklabels(ylabels, rotation=0)
    ax.label_outer()
plt.suptitle(f"Averaged ZABI model coefficients for {wrz.title().replace('_', ' ')}")

# %% ---- Distribution of coefficients for indicators ----
INDICATOR = 'si12'
df_plot = df[df['indicator'] == INDICATOR]
df_plot = df_plot.melt(id_vars=['indicator'], value_vars=bernoulli+binomial, var_name='Coefficient')
df_plot['Model'] = df_plot['Coefficient'].str.split('.').str[0]
df_plot['coeff'] = df_plot['Coefficient'].str.split('.').str[1]

fig, ax = plt.subplots(figsize=(12, 2))

sns.violinplot(data=df_plot, x='coeff', y='value', hue='Model', ax=ax, linewidth=.1)
plt.suptitle(f"ZABI model coefficients for {INDICATOR.upper()} for {wrz.title().replace('_', ' ')}")

del df_plot
# %%
