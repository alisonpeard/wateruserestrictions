"""
Calculate standardised drought indices (SPI, SPEI) from aggregated monthly data. 
Original code by David Pritchard.

Inputs:
    - os.path.join(config['paths']["tempdir"], {scenario}, 'aggregated_series.parquet')

Outputs:
    - os.path.join(config['paths']["tempdir"], {scenario}, 'parameters.csv')
    - os.path.join(config['paths']["tempdir"], {scenario}, 'standardised_series.parquet')
    - os.path.join(config['paths']["tempdir"], {scenario}, "si_plots", "{distribution}_win{window}_{buffer}_{variable}_{ensemble}_{RZ_ID}.png")
"""
# %%
import os
import itertools

import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt

import utils

def calculate_rolling_sums(df, factor_columns, value_column, window, time_columns=['Year', 'Month']):
    # handles dataframe with factor columns
    # assumes dataframe sorted appropriately (i.e. by factors then in time order)
    df0 = df.set_index(time_columns)
    df1 = df0.groupby(factor_columns)[value_column].rolling(window).sum()
    df0 = df0.reset_index()
    df1 = df1.reset_index()
    df1 = df1.rename(columns={value_column: 'rolling_sum'})
    df1 = df0.merge(df1)
    return df1


def fit_distribution(
        df, factor_columns, value_column, tempdir, window, scenarios=['BS', 'NF', 'FF'], month_column='Month', variable=None,
        variable_column='Variable', distributions={'prbc': [scipy.stats.gamma], 'ep': [scipy.stats.genextreme]},
        save_plots=True,
):
    # could also calculate and save goodness-of-fit metrics
    # partly accommodates multiple candidate distributions

    # For checking distribution fits
    plot_dir = os.path.join(tempdir, 'si_plots')
    if (not os.path.exists(plot_dir)) and save_plots:
        os.makedirs(plot_dir)

    bounds = {
        'gamma': {'a': (0.001, 1000.0), 'loc': (0.0, 0.0), 'scale': (0.001, 100.0)},
        'fisk': {'c': (-100.0, 100.0), 'loc': (-100.0, 100.0), 'scale': (0.001, 100.0)},
        'genextreme': {'c': (-100.0, 100.0), 'loc': (-100.0, 100.0), 'scale': (0.001, 100.0)},
    }

    df0 = df.loc[df['scenario'].isin(scenarios)]

    # Store parameters etc in a dictionary so can write out at the end
    fitting_details = {factor_column: [] for factor_column in factor_columns}
    for detail in ['distribution', 'window', 'shape', 'loc', 'scale', 'success', 'message', 'nllf']:
        fitting_details[detail] = []

    factor_combinations = df.groupby(factor_columns).first().index

    for _factor_values in factor_combinations:
        df1 = df0
        for factor_column, factor_value in zip(factor_columns, _factor_values):
            df1 = df1.loc[df1[factor_column] == factor_value]

        if df1[month_column].min() == 1:
            print('fitting:', scenarios, window, _factor_values)

        if variable is None:
            _distributions = distributions[df1[variable_column].tolist()[0]]
        else:
            _distributions = distributions[variable]

        for distribution in _distributions:

            if df1[month_column].min() == 1:
                fig, axs = plt.subplots(3, 4, sharex=True, sharey=True, figsize=(12, 9))
                i = 0
                j = 0

            data = df1.loc[np.isfinite(df1[value_column]), value_column]
            res = scipy.stats.fit(distribution, data, bounds=bounds[distribution.name])
            res.plot(ax=axs[i, j])

            # Increment indexes for plot matrix
            i += 1
            j += 1
            if i == 3:
                i = 0
            if j == 4:
                j = 0

            # Add parameters and details to output dictionary
            for factor_column, factor_value in zip(factor_columns, _factor_values):
                fitting_details[factor_column].append(factor_value)
            fitting_details['distribution'].append(distribution.name)
            fitting_details['window'].append(window)
            if distribution.name == 'gamma':
                fitting_details['shape'].append(getattr(res.params, 'a'))
            elif distribution.name in ['fisk', 'genextreme']:
                fitting_details['shape'].append(getattr(res.params, 'c'))
            fitting_details['loc'].append(getattr(res.params, 'loc'))
            fitting_details['scale'].append(getattr(res.params, 'scale'))
            fitting_details['success'].append(res.success)
            fitting_details['message'].append(res.message)
            fitting_details['nllf'].append(res.nllf(res.params, data))

            # Save plots
            if save_plots and (df1[month_column].min() == 12):
                m = 1
                for ax in axs.flat:
                    ax.set(title=m)
                    ax.get_legend().remove()
                    ax.xaxis.label.set_visible(False)
                    ax.yaxis.label.set_visible(False)
                    m += 1
                title_elements = [
                    factor + '=' + str(value) for factor, value in zip(factor_columns, _factor_values)
                    if factor != month_column
                ]
                plt.suptitle(', '.join(str(item) for item in title_elements))
                plt.tight_layout(pad=1.02)
                output_name = '_'.join(
                    str(item) for factor, item in zip(factor_columns, _factor_values)
                    if factor != month_column
                )
                output_name = distribution.name + '_win' + str(window) + '_' + output_name + '.png'
                output_path = os.path.join(plot_dir, output_name)
                plt.savefig(output_path)
                plt.close()

    df_output = pd.DataFrame(fitting_details)

    return df_output


def prepare_indices(df, tempdir, scenario, q=None, windows=[6, 12, 24]):
    # takes distribution name, parameters

    # Check order - still needed?
    df = df.sort_values(['RZ_ID', 'buffer', 'Variable', 'ensemble', 'Year', 'Month'])

    distribution_dfs = []

    for window in windows:

        df1 = calculate_rolling_sums(
            df, factor_columns=['buffer', 'Variable', 'ensemble', 'RZ_ID'], value_column='Value', window=window,
        )
        df1['window'] = window

        df2 = fit_distribution(
            df1, factor_columns=['buffer', 'Variable', 'RZ_ID', 'Month'], value_column='rolling_sum',
            tempdir=os.path.join(tempdir, scenario), window=window, scenarios=[scenario],
        )
        df2['window'] = window
        distribution_dfs.append(df2)

    df2 = pd.concat(distribution_dfs)
    df2.to_csv(os.path.join(tempdir, scenario, 'parameters.csv'), index=False)


def calculate_indices(
        df, tempdir, scenario, q=None, parameters_file='parameters.csv', windows=[6, 12, 24],
        factor_columns=['buffer', 'RZ_ID', 'Variable'], variable_column='Variable', spi_variable='prbc',
        spei_variable='ep',
):
    # assume only one distribution being passed in for parameters
    # assumes that index is in order

    # Again part of attempt to make flexible re number of factors in dataframe
    factor_combinations = df.groupby(factor_columns).first().index

    # Check order - still needed?
    df = df.sort_values(['RZ_ID', 'buffer', 'Variable', 'ensemble', 'Year', 'Month'])

    # TODO: Check expected number of rows and/or count of dates - to ensure no missing values etc
    # assert df.shape[0] == (df['RZ_ID'].nunique() * df['buffer'].nunique() * df['Variable'].nunique() * df['Y'])

    df_pars = pd.read_csv(os.path.join(tempdir, scenario, parameters_file))

    dfs = []
    for window in windows:

        df1 = calculate_rolling_sums(
            df, factor_columns=['buffer', 'Variable', 'ensemble', 'RZ_ID'], value_column='Value', window=window,
        )
        df1['window'] = window

        df_pars1 = df_pars.loc[df_pars['window'] == window]

        for _factor_values in factor_combinations:
            print('calculating indices:', scenario, window, _factor_values)

            df2 = df1
            df_pars2 = df_pars1
            for factor_column, factor_value in zip(factor_columns, _factor_values):
                df2 = df2.loc[df2[factor_column] == factor_value]
                df_pars2 = df_pars2.loc[df_pars2[factor_column] == factor_value]

            dist_name = df_pars2['distribution'].tolist()[0]

            if dist_name == 'gamma':
                distribution = scipy.stats.gamma
            elif dist_name == 'genextreme':
                distribution = scipy.stats.genextreme
            else:
                raise ValueError('Only gamma and genextreme distributions are currently supported.')

            for month in range(1, 12+1):

                # TODO: Flexibility on column names? - default arguments
                tmp = df_pars2.loc[
                    (df_pars2['Month'] == month) & (df_pars2['distribution'] == distribution.name),
                    ['shape', 'loc', 'scale']
                ]
                shape = tmp['shape']
                loc = tmp['loc']
                scale = tmp['scale']

                #
                df3 = df2.loc[df2['Month'] == month].copy()
                cdf = distribution.cdf(df3['rolling_sum'], shape, loc=loc, scale=scale)
                ppf = scipy.stats.norm.ppf(cdf)

                df3['si'] = ppf
                dfs.append(df3)

    df4 = pd.concat(dfs)
    df4 = df4.sort_index()

    # Format so compatible with indicator series dataframe
    df4['si_variable'] = 'si' + df4['window'].astype(str)
    df4 = df4.drop(columns=['Value', 'rolling_sum', 'window'])
    df4 = df4.pivot(
        index=[_ for _ in df4.columns if _ not in ['si_variable', 'si']], columns='si_variable', values='si'
    ).reset_index()
    df4 = df4.sort_values(['Variable', 'buffer', 'RZ_ID', 'ensemble', 'Year', 'Month'])

    df4.to_parquet(os.path.join(tempdir, scenario, 'standardised_series.parquet'), index=False)


def main(config, scenarios=["bs", "nf", "ff"]):
    tempdir = config['paths']["tempdir"]

    dfs = {}
    for scenario in scenarios:
        input_path = os.path.join(tempdir, scenario, 'aggregated_series.parquet')
        dfs[scenario] = pd.read_parquet(input_path)

    import multiprocessing as mp

    manager = mp.Manager()
    q = manager.Queue()
    n_processes = min(mp.cpu_count(), len(scenarios))
    pool = mp.Pool(n_processes)

    jobs = []
    for scenario in scenarios:
        job = pool.apply_async(prepare_indices, (dfs[scenario], tempdir, scenario, q))
        jobs.append(job)
    for job in jobs:
        job.get()

    jobs = []
    for scenario in scenarios:
        job = pool.apply_async(calculate_indices, (dfs[scenario], tempdir, scenario, q))
        jobs.append(job)
    for job in jobs:
        job.get()

    pool.close()
    pool.join()


if __name__ == '__main__':

    wd = os.path.join(os.path.dirname(__file__), "../..")
    os.chdir(wd); print(f"Working directory: {os.getcwd()}")
    config = utils.load_config()

    main(config)

# %%