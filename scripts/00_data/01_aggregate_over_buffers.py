"""
Aggregate Weather@Home data into timeseries over each water resource zone (WRZ) for
a range of buffers around the WRZ centroid and save results as parquet files in the tempdir.

Inputs:
    - os.path.join(config['paths']["datadir"], 'w@h', f"{scenario}.zip")
    - os.path.join(config['paths']["datadir"], 'river_basins', 'ukcp18-uk-land-river-hires.gpkg')
    - os.path.join(config['paths']["tempdir"], "wrz_buffer.gpkg")
    - os.path.join(config['paths']["tempdir"], 'wrz.gpkg')

Outputs:
    - os.path.join(config['paths']["tempdir"], "indicators", {scenario}, 'aggregated_series.parquet')
"""
# %%
import os
import zipfile

import pandas as pd
import geopandas as gpd
from tqdm import tqdm

import utils


YEARS = {'bs': range(1975, 2004+1), 'nf': range(2020, 2049+1), 'ff': range(2070, 2099+1)}


def condition_scenario(filename, year):
    if (filename.split('.')[-1] == 'csv') and (year in filename):
        return True
    else:
        return False


def read_wah_file(filename, wah_archive=None, wahpath=None):
    """Read one WAH scenario data file stored in CSV format."""
    if wah_archive is None:
        open_archive_passed = False
        wah_archive = zipfile.ZipFile(wahpath, 'r')
    else:
        open_archive_passed = True

    file = wah_archive.open(filename)
    df = pd.read_csv(file, index_col=0)
    df['lat'] = df['lat'].round(2)
    df['lon'] = df['lon'].round(2)
    df['time'] = pd.to_datetime(df['time'])
    df['Year'] = df['time'].dt.year
    df['Month'] = df['time'].dt.month
    df['prbc'] *= 86_400  # mm/s to mm/d
    df['ep'] = df['prbc'] - df['pepm']  # correction to data in input files

    df = pd.melt(
        df, id_vars=['lat', 'lon', 'Year', 'Month', 'ensemble'], value_vars=['prbc', 'ep'], var_name='Variable',
        value_name='Value',
    )

    if not open_archive_passed:
        wah_archive.close()

    return df


def read_wah(scenarios, datadir):
    """Read all WAH data files for one or more scenarios."""

    # List will contain one dataframe per year for each WAH scenario

    dfs = []

    for scenario in (pbar := tqdm(scenarios)):
        pbar.set_description(f"Reading {scenario}")

        wahpath = os.path.join(datadir, 'w@h', f"{scenario}.zip")

        with zipfile.ZipFile(wahpath, 'r') as archive:
            for year in YEARS[scenario]:
                files_year = [file for file in archive.namelist() if condition_scenario(file, str(year))]
                if len(files_year) == 1:
                    df = read_wah_file(files_year[0], archive)
                    dfs.append(df)
                else:
                    raise ValueError(
                        f'There should be one file per year, but {len(files_year)} files were found for year {year}.'
                    )

    df = pd.concat(dfs)

    return df


def prepare_max_extents(datadir, tempdir):
    """Intersect WRZ and river basins polygons to find maximum extent/domain to include for WRZ."""
    # from process_indicator_for_basins.py

    wrz = gpd.read_file(os.path.join(tempdir, 'wrz.gpkg'))
    wrz = wrz[['RZ_ID', 'geometry']].drop_duplicates()  # moved up in order of things

    basins = gpd.read_file(os.path.join(datadir, 'river_basins', 'ukcp18-uk-land-river-hires.gpkg')).to_crs(4326)
    assert basins.crs == wrz.crs

    wrz_basins = wrz[['RZ_ID', 'geometry']].sjoin(basins[['id', 'geometry']], how='left', predicate='intersects')  # 36 -> 15

    # Get full basin extent (rather than just the areas of intersection - i.e. replace partial extents) - i.e. union
    basin_geoms = {idx: geom for idx, geom in zip(basins.id, basins.geometry)}
    wrz_basins['geometry'] = wrz_basins['id'].apply(lambda x: basin_geoms[x])
    wrz_basins = wrz_basins.set_geometry('geometry')

    basins_dissolved = wrz_basins.dissolve(by='RZ_ID').reset_index()

    assert basins_dissolved['RZ_ID'].nunique() == wrz['RZ_ID'].nunique(), 'Number of RZ IDs changed.'

    return basins_dissolved


def create_buffered_extents(tempdir, wrz_basins):
    # Intersect the WRZ/river (union) polygons with the buffer radius polygons
    # - wrz_basins = basins_dissolved

    wrz = gpd.read_file(os.path.join(tempdir, 'wrz.gpkg'))
    wrz_buffer = gpd.read_file(os.path.join(tempdir, "wrz_buffer.gpkg"))

    _extents = []

    for wrz_id in wrz['RZ_ID'].unique():
        extents = wrz_basins.loc[wrz_basins['RZ_ID'] == wrz_id, ['RZ_ID', 'geometry']].overlay(
            wrz_buffer.loc[wrz_buffer['RZ_ID'] == wrz_id], how='intersection',
        )
        extents = extents.drop(columns='RZ_ID_2')
        extents = extents.rename(columns={'RZ_ID_1': 'RZ_ID'})
        _extents.append(extents)

    extents = gpd.GeoDataFrame(pd.concat(_extents, ignore_index=True), crs=_extents[0].crs)

    return extents


def get_wah_coords(datadir, scenario='nf'):
    """ Get coordinates of all grid cell centres in WAH dataset."""
    wahpath = os.path.join(datadir, 'w@h', f"{scenario}.zip")

    with zipfile.ZipFile(wahpath, 'r') as archive:
        files = archive.namelist()
        df = read_wah_file(files[0], archive)

    df_tmp = df.loc[
        (df['ensemble'] == df['ensemble'].unique()[0]) & (df['Year'] == df['Year'].unique()[0])
        & (df['Month'] == df['Month'].unique()[0]),
        ['lat', 'lon']
    ]

    wah_coords = gpd.GeoDataFrame(df_tmp, geometry=gpd.points_from_xy(df_tmp['lon'], df_tmp['lat']), crs="EPSG:4326")

    return wah_coords


def identify_relevant_points(tempdir, wah_coords):  # wrz_buffer, wrz_row,
    """Identify points within each buffer radius for a specified WRZ."""
    # based on process_full_ts.py

    wrz = gpd.read_file(os.path.join(tempdir, "wrz.gpkg"))
    wrz_buffer = gpd.read_file(os.path.join(tempdir, "wrz_buffer.gpkg"))

    gdfs = []
    for wrz_row in wrz.itertuples():
        buffers_wrz = wrz_buffer[wrz_buffer['RZ_ID'] == wrz_row.RZ_ID]
        assert wah_coords.crs == buffers_wrz.crs, "Dataframes have different coordinate reference systems"
        wah_map = gpd.overlay(wah_coords, buffers_wrz, how='intersection')
        gdfs.append(wah_map)

    gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=gdfs[0].crs)

    return gdf


def aggregate_spatially(tempdir, df, buffered_extents, scenarios=['bs', 'nf', 'ff']):
    for scenario in scenarios:

        ensemble_group = [scenario.upper() + str(i) for i in range(1, 100+1)]
        df_sub = df.loc[df['ensemble'].isin(ensemble_group)]

        gdf = gpd.GeoDataFrame(df_sub, geometry=gpd.points_from_xy(df_sub['lon'], df_sub['lat']), crs="EPSG:4326")

        dfs = []
        for _, row in tqdm(buffered_extents.iterrows(), desc=f"Aggregating WRZs in {scenario}", total=len(buffered_extents)):
            wah_df = gpd.clip(gdf, row['geometry'])
            df_agg = wah_df.groupby(['Variable', 'ensemble', 'Year', 'Month'])['Value'].mean().reset_index()
            df_agg['RZ_ID'] = row['RZ_ID']
            df_agg['buffer'] = row['buffer']
            df_agg['scenario'] = scenario
            dfs.append(df_agg)

        df_agg = pd.concat(dfs)
        df_agg = df_agg.sort_values(['Variable', 'ensemble', 'Year', 'Month'])

        output_path = os.path.join(tempdir, scenario, 'aggregated_series.parquet')
        df_agg.to_parquet(output_path, index=False)


def main(config, scenarios=["bs", "nf", "ff"]):
    datadir = config['paths']["datadir"]
    tempdir = config['paths']["tempdir"]

    for scenario in scenarios:
        if not os.path.exists(os.path.join(tempdir, scenario)):
            os.makedirs(os.path.join(tempdir, scenario))

    df = read_wah(scenarios, datadir)
    wrz_basins = prepare_max_extents(datadir, tempdir)
    buffered_extents = create_buffered_extents(tempdir, wrz_basins)
    aggregate_spatially(tempdir, df, buffered_extents)


if __name__ == '__main__':
    wd = os.path.join(os.path.dirname(__file__), "../..")
    os.chdir(wd); print(f"Working directory: {os.getcwd()}")
    config = utils.load_config()
    main(config)

# %%