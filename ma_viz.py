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
 
wrz = 'lincolnshire'
datadir = f"/Users/alison/Documents/RAPID/correlation-analysis/data/results/cv_glmnet/{wrz}/"
si6 = pd.read_csv(os.path.join(datadir, "mafits__si6.trendFF1.csv"), index_col=0, skipinitialspace=True)
si12 = pd.read_csv(os.path.join(datadir, "mafits__si12.trendFF1.csv"), index_col=0, skipinitialspace=True)
si24 = pd.read_csv(os.path.join(datadir, "mafits__si24.trendFF1.csv"), index_col=0, skipinitialspace=True)
si6['indicator'] = 'si6'
si12['indicator'] = 'si12'
si24['indicator'] = 'si24'

colnames = si6.columns
colnames = [x.replace('.si6.trend', '') for x in colnames]

si6.columns = colnames
si12.columns = colnames
si24.columns = colnames

df = pd.concat([si6, si12, si24])
df.columns = colnames
df.groupby('indicator').mean()

# %% Violin plots of metrics
fig, axs = plt.subplots(1, 3, figsize=(16, 4))

violin_kws = {'palette': 'colorblind', 'linewidth': .5}

ax = axs[0]
binary_metrics = ['BCE', 'Brier']
df_plot = df.melt(id_vars='indicator', value_vars=binary_metrics, var_name='Binary metric')
sns.violinplot(data=df_plot, x='indicator', y='value', hue='Binary metric', ax=ax, **violin_kws)
ax.set_title('Binary metrics')

ax = axs[1]
# note 'AUROC' removed because inflated by class imbalance
confusion_metrics = ['Precision', 'Recall', 'F1', 'F2']
df_plot = df.melt(id_vars='indicator', value_vars=confusion_metrics, var_name='Confusion metric')
sns.violinplot(data=df_plot, x='indicator', y='value', hue='Confusion metric', ax=ax, **violin_kws)
ax.set_title('Confusion metrics')

ax = axs[2] 
regression_metrics = ['RMSE']
df_plot = df.melt(id_vars='indicator', value_vars=regression_metrics, var_name='RMSE')
sns.violinplot(data=df_plot, x='indicator', y='value', hue='RMSE', legend=False, ax=ax, **violin_kws)
ymin, ymax = df_plot['value'].min(), df_plot['value'].max()
# ax.set_ylim(9.8, 10.3)
ax.set_title('RMSE')

for ax in axs:
    ax.set_ylabel("")
    ax.set_xlabel("Indicator")
    ax.set_xticks(ax.get_xticks(), ['SI6', 'SI12', 'SI24'])

plt.suptitle(f"ZABI metric distributions for {wrz.title().replace('_', ' ')}")
# del df_plot

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
cmap = 'bwr'
cmap = sns.diverging_palette(220, 20, as_cmap=True)

# sort index
df_avg.index = pd.CategoricalIndex(df_avg.index, categories=["si6", "si12", "si24"])
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
    # ax.set_yticklabels(ylabels, rotation=0)
    ax.set_ylabel('Indicator')
    ax.label_outer()

    for _, spine in ax.spines.items():
        spine.set_visible(True)
fig.suptitle(f"Averaged ZABI model coefficients for {wrz.title().replace('_', ' ')}")

# %% ---- Distribution of coefficients for indicators ----
INDICATOR = 'si6'
df_plot = df[df['indicator'] == INDICATOR]
df_plot = df_plot.melt(id_vars=['indicator'], value_vars=bernoulli+binomial, var_name='Coefficient')
df_plot['Model'] = df_plot['Coefficient'].str.split('.').str[0].str.title()
df_plot['coeff'] = df_plot['Coefficient'].apply(lambda x: '.'.join(x.split('.')[1:]))

fig, ax = plt.subplots(figsize=(12, 2))

sns.violinplot(data=df_plot, x='coeff', y='value', hue='Model', ax=ax, linewidth=.1)
# ax.set_ylim([-1,1])
ax.set_xticklabels(xlabels, rotation=45, ha='right')
plt.suptitle(f"ZABI model coefficients for {INDICATOR.upper()} for {wrz.title().replace('_', ' ')}")

# del df_plot
# %%
