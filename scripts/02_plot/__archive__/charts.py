#%%
import os
import numpy as np
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
trend = 'raw'
lag = 'ma.s'
scenario = 'ff'
datadir = f"/Users/alison/Documents/drought-indicators/analysis/data/results/cv/{wrz}/{trend}"

binary_metrics = ['BCE', 'Brier']
confusion_metrics = ['Precision', 'Recall', 'F1', 'F2']
regression_metrics = ['RMSE', "CRPS"]

# %%
dfs = []
toremove = ['.trend', '.raw', '.lag.', '.ma.s', '.ma.t', '.ep_total', '.si6', '.si12', '.si24']
ind_rename = {'ep_total': 'EP', 'si6': 'SI6', 'si12': 'SI12', 'si24': 'SI24'}
trend_rename = {'trend': 'Yes', 'raw': 'No'}
lag_rename = {'lag.': 'Pointwise', 'ma.s': 'Simple MA', 'ma.t': 'Triangular MA'}

for root, dirs, files in os.walk(datadir):
    for file in files:
        if root.endswith(lag) and file.endswith(f"{scenario}.csv"):
            df = pd.read_csv(os.path.join(root, file), index_col=0, skipinitialspace=True)

            columns = df.columns
            for r in toremove:
                columns = [x.replace(r, '') for x in columns]
            
            df.columns = columns
            df = df.rename(columns={'CRPS..WUR.days.': 'CRPS'})
            df['indicator'] = root.split('/')[-2]
            dfs.append(df)

df = pd.concat(dfs).reset_index(drop=True).dropna()
df.groupby(['indicator']).mean()
df = df.set_index('indicator').loc[['ep_total', 'si6', 'si12', 'si24']].reset_index(drop=False)

# %% gridspec test
# Create a figure
from matplotlib.gridspec import GridSpec

violin_kws = {'palette': 'colorblind', 'linewidth': .5}

fig = plt.figure(figsize=(12, 6))
gs = GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 2])
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[0, 1])
ax4 = fig.add_subplot(gs[1, 1])
ax5 = fig.add_subplot(gs[:, 2])
axs = [ax1, ax2, ax3, ax4, ax5]

ax = ax1
df_plot = df.melt(id_vars='indicator', value_vars='BCE', var_name='Metric')
sns.violinplot(data=df_plot, x='indicator', y='value', hue='Metric', ax=ax, **violin_kws)
ax.set_title('Binary metrics')
ax = ax2
df_plot = df.melt(id_vars='indicator', value_vars='Brier', var_name='Metric')
sns.violinplot(data=df_plot, x='indicator', y='value', hue='Metric', ax=ax, **violin_kws)

ax = ax3
df_plot = df.melt(id_vars='indicator', value_vars='Precision', var_name='Metric')
sns.violinplot(data=df_plot, x='indicator', y='value', hue='Metric', ax=ax, **violin_kws)
ax.set_title('Confusion metrics')
ax = ax4
df_plot = df.melt(id_vars='indicator', value_vars=['Recall', 'F1', 'F2'], var_name='Metric')
sns.violinplot(data=df_plot, x='indicator', y='value', hue='Metric', ax=ax, **violin_kws)

ax = ax5
df_plot = df.melt(id_vars='indicator', value_vars=['CRPS'], var_name='Metric')
sns.violinplot(data=df_plot, x='indicator', y='value', hue='Metric', ax=ax, **violin_kws)
ax.set_title('Regression metrics')

for ax in axs:
    # ax.label_outer()
    pass

# %% Violin plots of metrics
fig, axs = plt.subplots(1, 3, figsize=(16, 4))

violin_kws = {'palette': 'colorblind', 'linewidth': .5}

ax = axs[0]
df_plot = df.melt(id_vars='indicator', value_vars=binary_metrics, var_name='Binary metric')
sns.violinplot(data=df_plot, x='indicator', y='value', hue='Binary metric', ax=ax, **violin_kws)
ax.set_title('Binary metrics')

ax = axs[1]
# note 'AUROC' removed because inflated by class imbalance
df_plot = df.melt(id_vars='indicator', value_vars=confusion_metrics, var_name='Confusion metric')
sns.violinplot(data=df_plot, x='indicator', y='value', hue='Confusion metric', ax=ax, **violin_kws)
ax.set_title('Confusion metrics')

ax = axs[2] 
df_plot = df.melt(id_vars='indicator', value_vars=regression_metrics, var_name='CRPS')
sns.violinplot(data=df_plot, x='indicator', y='value', hue='CRPS', legend=False, ax=ax, **violin_kws)
ymin, ymax = df_plot['value'].min(), df_plot['value'].max()
ax.set_title('CRPS')

for ax in axs:
    ax.set_ylabel("")
    ax.set_xlabel("Indicator")
    ax.set_xticks(ax.get_xticks(), ['EP', 'SI6', 'SI12', 'SI24'])

plt.suptitle(f"ZABI metric distributions for {wrz.title().replace('_', ' ')}")

# %%
df_avg = df.groupby("indicator").mean()
df_std = df.groupby("indicator").std()
df_q25 = df.groupby("indicator").quantile(0.25)
df_q75 = df.groupby("indicator").quantile(0.75)

df_avg = df_avg.round(4)
df_std = df_std.round(4)
df_q25 = df_q25.round(4)
df_q75 = df_q75.round(4)

df_avg.to_csv(os.path.join(datadir, "mafits__avg.csv"))

# %% ---- Heatmaps of coefficients ----
# prep for aesthetics
xlabels = df.columns
xlabels = [x for x in xlabels if x not in binary_metrics+confusion_metrics+regression_metrics+['AUROC','indicator']]
ylabels = [*df_avg.index]
ylabels = [y.replace('_', ' ').title() for y in ylabels]
bernoulli = [x for x in xlabels if 'ber' in x]
binomial = [x for x in xlabels if 'bin' in x]
xlabels = [x.replace('.', ' ').replace('ber', '').upper() for x in bernoulli]
xlabels = ['intercept', 'current'] + xlabels[2:]
# change plot style so there is no grid in the background
sns.set_style("white")

# set cmap center to 0
cmap = sns.diverging_palette(220, 20, as_cmap=True)

# sort index
df_avg.index = pd.CategoricalIndex(df_avg.index, categories=["ep_total", "si6", "si12", "si24"])
df_avg.sort_index(level=0, inplace=True)

# plot heatmaps
fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True, sharex=True, layout='tight')
im = sns.heatmap(df_avg[bernoulli].replace(0, np.nan), annot=True, ax=axs[0],
            linewidths=.5, linecolor='white', 
            cmap=cmap, center=0, cbar_kws={'label': 'Coefficient'})
sns.heatmap(df_avg[binomial].replace(0, np.nan), annot=True, ax=axs[1],
            linewidths=.5, linecolor='white',
            cmap=cmap, center=0, cbar_kws={'label': 'Coefficient'})

# labels
axs[0].set_title("Bernoulli coefficients")
axs[1].set_title("Binomial coefficients")
for ax in axs:
    ax.set_xlabel('Lag')
    ax.set_xticklabels(xlabels, rotation=45, ha='right')
    ax.set_ylabel('Indicator')
    ax.label_outer()

    for _, spine in ax.spines.items():
        spine.set_visible(True)

fig.suptitle(f"Averaged ZABI model coefficients for {wrz.title().replace('_', ' ')}")

# %%
