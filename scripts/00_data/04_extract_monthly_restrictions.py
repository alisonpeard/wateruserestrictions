"""
Process the LoS data, counting the number of days in each month at each level of severity.

Params:
    - config["config"]["scenario"]
Inputs:
    - os.path.join(config['paths']['datadir'], 'los', {scenario}.upper(),'National_Model__jobid_{ensemble}_nodal_globalPlotVars_selected.csv')
    - os.path.join(config['paths']['datadir'], 'WRZ', 'wrz_code.xlsx')

Outputs:
    - os.path.join(config['paths']['tempdir'], "los", {scenario}, "monthly_los_level0_melted.csv")
    - os.path.join(config['paths']['tempdir'], "los", {scenario}, "monthly_los_level1_melted.csv")
    - os.path.join(config['paths']['tempdir'], "los", {scenario}, "monthly_los_level2_melted.csv")
    - os.path.join(config['paths']['tempdir'], "los", {scenario}, "monthly_los_level3_melted.csv")
    - os.path.join(config['paths']['tempdir'], "los", {scenario}, "monthly_los_level4_melted.csv")
"""
#%%
import os
os.environ['USE_PYGEOS'] = '0'
from tqdm import tqdm
import pprint

import utils
import pandas as pd


def check_wrz_count(x, expected=99, enforce=False, ensemble_member=""):
    """Check we have the expected number of unique WRZ codes"""
    n_unique = pd.unique(x).size
    if n_unique != expected:
        msg = f"Found {n_unique} unique WRZ codes, expected {expected} ({ensemble_member})."
        if enforce:
            raise ValueError(msg)
        else:
            print("WARNING: " + msg)
    return n_unique


def melt_los_df(df):
        """Melt the dfs so that they are in long format"""
        rz_id_cols = df.columns[:-1]
        df_melted = df.reset_index()
        df_melted = df_melted.melt(id_vars=['Date', 'Ensemble'], value_vars=rz_id_cols,
                                var_name='RZ_ID', value_name='LoS', ignore_index=True)

        df_melted = df_melted[['RZ_ID', 'Date', 'Ensemble', 'LoS']]
        df_melted = df_melted.drop_duplicates()
        df_melted = df_melted.sort_values(by=['RZ_ID', "Ensemble", 'Date']).reset_index(drop=True)
        
        df_melted['Year'] = df_melted['Date'].dt.year
        df_melted['Month'] = df_melted['Date'].dt.month
        df_melted = df_melted.drop(columns='Date')
        
        df_melted = df_melted[['RZ_ID', 'Ensemble', 'Year', 'Month', 'LoS']]
        df_melted = df_melted.sort_values(by=['RZ_ID', 'Ensemble', 'Year', 'Month', 'LoS'])
        return df_melted


def main(config):
    datadir = config['paths']['datadir']
    outdir = config['paths']['tempdir']

    # choose which data to load
    SCENARIO = config["config"]["scenarios"][config["config"]["scenario"]]
    outdir = os.path.join(outdir, "los", SCENARIO)
    os.makedirs(os.path.join(outdir), exist_ok=True)

    print(f"Loading data from {outdir}.")

    # create a dict to map WRZ code to RZ ID
    wrz_code = pd.read_excel(os.path.join(datadir, 'WRZ', 'wrz_code.xlsx'))
    wrz_code['RZ ID'] = pd.to_numeric(wrz_code['RZ ID'])
    wrz_dict = {wrz_code: f'{rz_id:.0f}' for wrz_code, rz_id in zip(wrz_code['WREW Code'], wrz_code['RZ ID'])}

    pprint.pprint(wrz_dict)

    # Load the LoS data and process it
    dfs_binary = []
    dfs_l1 = []
    dfs_l2 = []
    dfs_l3 = []
    dfs_l4 = []

    # main processing loop
    for ensemble in (pbar := tqdm(range(1, 101), desc=f"Processing LoS data for {SCENARIO}")):
        # load and format the date columns
        df = pd.read_csv(os.path.join(
            datadir, 'los', SCENARIO.upper(),
            f'National_Model__jobid_{ensemble}_nodal_globalPlotVars_selected.csv'
            ), header=1
        )

        los_cols = [col for col in df.columns if "LoS" in col]

        check_wrz_count(los_cols, ensemble_member=ensemble)

        df = df[['Year', 'Day'] + los_cols].copy()

        check_wrz_count(df.columns, ensemble_member=ensemble)

        df['Date'] = pd.to_datetime(df['Year'] * 1000 + df['Day'], format='%Y%j')
        df = df.drop(columns=["Day", "Year"])
        df = df.set_index("Date")

        # group all columns in same WRZ
        df.columns = df.columns.map(wrz_dict)
        valid_columns = df.columns[(df.columns != 'nan') & (~df.columns.isnull())]
        df = df[valid_columns]
        df.columns.name = "RZ_ID"
        df = df.T.groupby('RZ_ID').max().T
        
        df.columns = df.columns.astype(int)
        df = df.sort_values(by='RZ_ID', axis=1)

        # monthly counts for days at each LoS level
        df_l0 = (df > 0).astype(int).groupby(pd.Grouper(freq="ME")).agg("sum")
        df_l1 = (df == 1).astype(int).groupby(pd.Grouper(freq="ME")).agg("sum")
        df_l2 = (df == 2).astype(int).groupby(pd.Grouper(freq="ME")).agg("sum")
        df_l3 = (df == 3).astype(int).groupby(pd.Grouper(freq="ME")).agg("sum")
        df_l4 = (df == 4).astype(int).groupby(pd.Grouper(freq="ME")).agg("sum")
        
        # add ensemble column
        df_l0['Ensemble'] = int(ensemble)
        df_l1['Ensemble'] = int(ensemble)
        df_l2['Ensemble'] = int(ensemble)
        df_l3['Ensemble'] = int(ensemble)
        df_l4['Ensemble'] = int(ensemble)
    
        dfs_binary.append(df_l0)
        dfs_l1.append(df_l1)
        dfs_l2.append(df_l2)
        dfs_l3.append(df_l3)
        dfs_l4.append(df_l4)
        
    # concatenate all the ensemble members
    df_l0 = pd.concat(dfs_binary, axis=0).sort_values(by=["Date", "Ensemble"])
    df_l1 = pd.concat(dfs_l1, axis=0).sort_values(by=["Date", "Ensemble"])
    df_l2 = pd.concat(dfs_l2, axis=0).sort_values(by=["Date", "Ensemble"])
    df_l3 = pd.concat(dfs_l3, axis=0).sort_values(by=["Date", "Ensemble"])
    df_l4 = pd.concat(dfs_l4, axis=0).sort_values(by=["Date", "Ensemble"])

    df_l0_melted = melt_los_df(df_l0)
    df_l1_melted = melt_los_df(df_l1)
    df_l2_melted = melt_los_df(df_l2)
    df_l3_melted = melt_los_df(df_l3)
    df_l4_melted = melt_los_df(df_l4)

    check_wrz_count(df_l0_melted['RZ_ID'])

    # save the melted dfs
    df_l0_melted.to_csv(os.path.join(outdir, 'monthly_los_level0_melted.csv'))
    df_l1_melted.to_csv(os.path.join(outdir, 'monthly_los_level1_melted.csv'))
    df_l2_melted.to_csv(os.path.join(outdir, 'monthly_los_level2_melted.csv'))
    df_l3_melted.to_csv(os.path.join(outdir, 'monthly_los_level3_melted.csv'))
    df_l4_melted.to_csv(os.path.join(outdir, 'monthly_los_level4_melted.csv'))

    print(f"Finished processing LoS data for {SCENARIO}. Outputs saved to {outdir}.")


if __name__ == "__main__":
    wd = os.path.join(os.path.dirname(__file__), "../..")
    os.chdir(wd); print(f"Working directory: {os.getcwd()}")
    config = utils.load_config()
    main(config)
# %%