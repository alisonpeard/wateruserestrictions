#%%
import os
import numpy as np
import glob
import pandas as pd


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
import seaborn as sns
df = df.reset_index()
df = df.rename(columns={'index': 'indicator'})
# %%
fig, axs = plt.subplots(2, 2, figsize=(20, 12))

violin_kws = {'palette': 'colorblind', 'linewidth': 1, 'inner': 'quartile'}

df_plot = df.melt(id_vars='indicator', value_vars=['BCE', 'Brier'], var_name='Binary metric')
sns.violinplot(data=df_plot, x='indicator', y='value', hue='Binary metric', ax=axs[0, 0], **violin_kws)

df_plot = df.melt(id_vars='indicator', value_vars=['AUROC', 'Precision', 'Recall', 'F1', 'F2'], var_name='Confusion metric')
sns.violinplot(data=df_plot, x='indicator', y='value', hue='Confusion metric', ax=axs[0, 1])

sns.violinplot(data=df, x='indicator', y='zibi.r2', ax=axs[1, 0])
axs[1,0].set_ylabel('R-squared')
sns.violinplot(data=df, x='indicator', y='zibi.rmse', ax=axs[1, 1])
axs[1,1].set_ylabel('RMSE')

for ax in axs[0, :]:
    ax.set_ylabel('Score')
    sns.move_legend(ax, "center left")

del df_plot
# %%

df = sns.load_dataset("titanic")
df
# %%
sns.violinplot(data=df, x="class", y="age", hue="alive")
# %%
df_avg = df.groupby(level=0).mean()
df_std = df.groupby(level=0).std()
df_q25 = df.groupby(level=0).quantile(0.25)
df_q75 = df.groupby(level=0).quantile(0.75)

df_avg = df_avg.round(4)
df_std = df_std.round(4)
df_q25 = df_q25.round(4)
df_q75 = df_q75.round(4)

# %% save
df_avg.to_csv(os.path.join(datadir, "glmnet_mean.csv"))
df_std.to_csv(os.path.join(datadir, "glmnet_std.csv"))
df_q25.to_csv(os.path.join(datadir, "glmnet_q25.csv"))
df_q75.to_csv(os.path.join(datadir, "glmnet_q75.csv"))

# %%
# VISUALISATION
# plot zibi.rmse with 95% CI bars
import matplotlib.pyplot as plt
import seaborn as sns

df_avg = pd.read_csv(os.path.join(datadir, "glmnet_mean.csv"), index_col=0)
df_std = pd.read_csv(os.path.join(datadir, "glmnet_std.csv"), index_col=0)
df_q25 = pd.read_csv(os.path.join(datadir, "glmnet_q25.csv"), index_col=0)
df_q75 = pd.read_csv(os.path.join(datadir, "glmnet_q75.csv"), index_col=0)

# %%
# save dataframe to copy-paste into appendix
metrics = ['BCE', 'Brier', 'AUROC', 'Precision', 'Recall', 'F1', 'F2', 'bin.rmse', 'zibi.rmse']
print(df_avg[metrics])
print(df_std[metrics])
df_avg[metrics].replace(0, '').to_csv(os.path.join(datadir, "glmnet_metrics_mean.csv"))
df_std[metrics].replace(0, '').to_csv(os.path.join(datadir, "glmnet_metrics_std.csv"))
df_q25[metrics].replace(0, '').to_csv(os.path.join(datadir, "glmnet_metrics_q25.csv"))
df_q75[metrics].replace(0, '').to_csv(os.path.join(datadir, "glmnet_metrics_q75.csv"))

# %%
bernoulli = ['ber.intercept', 'ber.l0', 'ber.l1', 'ber.l2', 'ber.l3', 'ber.l4', 'ber.l5', 'ber.l6', 'ber.ma']
binomial = ['bin.intercept', 'bin.l0', 'bin.l1', 'bin.l2', 'bin.l3', 'bin.l4', 'bin.l5', 'bin.l6', 'bin.ma']
print(df_avg[bernoulli])
print(df_avg[binomial])
df_avg[bernoulli].replace(0, '').to_csv(os.path.join(datadir, "glmnet_bernoulli_mean.csv"))
df_avg[binomial].replace(0, '').to_csv(os.path.join(datadir, "glmnet_binomial_mean.csv"))

# %%
# find seed corresponding to lowest zibi.rmse
scores = {}
for file, df in zip(files, dfs):
    seed = int(file.split('_')[-1].split('.')[0])
    rmse = df.loc['si6', 'zibi.rmse']
    scores[seed] = rmse
scores
scores_df = pd.DataFrame.from_dict(scores, orient='index', columns=['zibi.rmse'])
scores_df.sort_values(by='zibi.rmse', ascending=True)
# %% ---- Plot metrics ----
from matplotlib.transforms import Affine2D
sns.set_context("talk")
sns.set_style("whitegrid")
sns.set_palette("colorblind")

def plot_errorbar(metric, title, ax, trans):
    yerr = [df_avg[metric] - df_q25[metric], df_q75[metric] - df_avg[metric]]
    yerr = [x.apply(lambda x: max(0, x)) for x in yerr] # do better lately
    ax.errorbar(df_avg.index, df_avg[metric],
                yerr, fmt='o', capsize=5, capthick=2, elinewidth=2,
                label=title, transform=trans)
    ax.set_xticks(range(3), labels=['SPI 6', 'SPI 12', 'SPI 24'], rotation=35)
    ax.set_xlabel("")
    ax.legend(loc='upper left')

fig, axs = plt.subplots(1, 4, figsize=(20, 5))

ax = axs[0]
trans1 = Affine2D().translate(-0.1, 0.0) + ax.transData # shift errorbars left
trans2 = Affine2D().translate(+0.1, 0.0) + ax.transData # shift errorbars right
plot_errorbar('BCE', 'Binary cross entropy', ax, trans1)
plot_errorbar('Brier', 'Brier score', ax, trans2)
ax.legend(loc='upper left')

ax = axs[1]
trans1 = Affine2D().translate(-0.1, 0.0) + ax.transData
trans2 = Affine2D().translate(+0.1, 0.0) + ax.transData
trans3 = Affine2D().translate(+0.2, 0.0) + ax.transData
trans4 = Affine2D().translate(+0.3, 0.0) + ax.transData
plot_errorbar('Precision', 'Precision', ax, trans1)
plot_errorbar('Recall', 'Recall', ax, None)
plot_errorbar('F1', 'F1 score', ax, trans2)
plot_errorbar('F2', 'F2 (CSI)', ax, trans3) # CSI
plot_errorbar('AUROC', 'AUROC', ax, trans4)
ax.legend(loc='upper left', fontsize='small')

ax = axs[2]
plot_errorbar('zibi.r2', 'R-squared', ax, None)
ax.legend(loc='upper left')

ax = axs[3]
plot_errorbar('zibi.rmse', 'RMSE', ax, None)
ax.legend(loc='upper left')

plt.suptitle(f"ZABI model metrics with interquartile range for {wrz.title().replace('_', ' ')}")
fig.tight_layout()

# %% ---- Heatmaps of coefficients ----
import seaborn as sns

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
indicator = 'si6'

ber = df_avg[bernoulli].loc[indicator]
ber_lower = df_q25[bernoulli].loc[indicator]
ber_upper = df_q75[bernoulli].loc[indicator]
bin  = df_avg[binomial].loc[indicator]
bin_lower = df_q25[binomial].loc[indicator]
bin_upper = df_q75[binomial].loc[indicator]

ber_yerr = [ber - ber_lower, ber_upper - ber]
ber_yerr = [x.apply(lambda x: max(0, x)) for x in ber_yerr]
bin_yerr = [bin - bin_lower, bin_upper - bin]
bin_yerr = [x.apply(lambda x: max(0, x)) for x in bin_yerr]

fig, ax = plt.subplots(figsize=(12, 6))

trans1 = Affine2D().translate(-0.1, 0.0) + ax.transData
trans2 = Affine2D().translate(+0.1, 0.0) + ax.transData

ax.errorbar(ber.index, ber.values, yerr=ber_yerr,
            fmt='o', capsize=5, capthick=2, elinewidth=2,
            label='Bernoulli', transform=trans1)

ax.errorbar(ber.index, bin.values, yerr=bin_yerr,
            fmt='o', capsize=5, capthick=2, elinewidth=2,
            label='Binomial', transform=trans2)

ax.set_xticks(ticks=ax.get_xticks(), labels=xlabels, rotation=35)
ax.set_xlabel("")
ax.set_ylabel('Coefficient value')
ax.legend(loc='upper left')

plt.suptitle(f"ZABI model coefficients for {indicator.title()} with IQ range for {wrz.title().replace('_', ' ')}")
# %%
