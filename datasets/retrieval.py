import logging
import os
import cdsapi
import numpy as np
import pandas as pd

from .utils import unzip_csv_or_nc_batch


def retieve_era5(
    dataset: str = "reanalysis-era5-single-levels-timeseries",
    center: tuple[float, float] = (5.5, 56.5),
    radius: float = 4.5,
    step: float = 0.5,
    start_date: str = '2025-02-01',
    end_date: str = '2025-09-30',
    variables: list[str] = [
        "2m_dewpoint_temperature",
        "mean_sea_level_pressure",
        "skin_temperature",
        "surface_pressure",
        "surface_solar_radiation_downwards",
        "sea_surface_temperature",
        "surface_thermal_radiation_downwards",
        "2m_temperature",
        "total_precipitation",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "100m_u_component_of_wind",
        "100m_v_component_of_wind",
        "mean_wave_direction",
        "mean_wave_period",
        "significant_height_of_combined_wind_waves_and_swell"
    ],
    out_dir: str = './data',
    decompress: bool = True,
    data_format: str = 'csv'
) -> None:
    
    center_lon, center_lat = center
    output_dir = os.path.join(out_dir, f"{start_date}--{end_date}")

    lon_vals = np.arange(center_lon - radius, center_lon + radius + step, step)
    lat_vals = np.arange(center_lat - radius, center_lat + radius + step, step)

    # Create list of (lon, lat) pairs
    tasks = [(lon, lat) for lon in lon_vals for lat in lat_vals]

    def data_filename(lon, lat, output_dir=output_dir) -> str:
        return os.path.join(output_dir, f"data_lon{lon:.2f}_lat{lat:.2f}_compr.zip")

    def data_already_exists(lon, lat) -> bool:
        return os.path.isfile(data_filename(lon, lat))

    def download_data(lon, lat, start_date=start_date, end_date=end_date) -> None:
        try:
            request = {
                "variable": variables,
                "location": {"longitude": lon, "latitude": lat},
                "date": [f"{start_date}/{end_date}"],
                "data_format": data_format
            }
            fname = data_filename(lon, lat)
            client = cdsapi.Client()
            client.retrieve(dataset, request).download(fname)
        except Exception as e:
            logging.error(f'Cannot download file {fname}: \n{e}')

    def retrieve_point(args) -> None:
        lon, lat = args
        if not data_already_exists(lon, lat):
            download_data(lon, lat)

    for tsk in tasks:
        retrieve_point(tsk)

    if decompress:
        unzip_csv_or_nc_batch(output_dir)


def grab_df_from_era5_csvs(output_dir, lat_vals, lon_vals) -> pd.DataFrame:
    dfs = []
    for lon in lon_vals:
        for lat in lat_vals:
            for k in range(1,3):
                fname = os.path.join(output_dir, f"data_lon{lon:.2f}_lat{lat:.2f}_{k :2d}.csv")
                if os.path.isfile(fname):
                    if k < 2:
                        df_ = pd.read_csv(fname)
                    else:
                        df_ = df_.merge(pd.read_csv(fname), 'left', ['valid_time', 'latitude', 'longitude'])
            dfs.append(df_)

    raw_df = pd.concat(dfs, axis=0, ignore_index=True)
    raw_df = raw_df.rename(columns={
        'valid_time': 'time',
        'latitude': 'lat',
        'longitude': 'lon'
    })
    return raw_df
