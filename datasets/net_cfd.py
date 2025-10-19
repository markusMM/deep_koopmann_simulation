import xarray as xr
import numpy as np
from typing import Tuple, List
import logging

from common import (
    BBOX_INIT,
    TARGET_FEATURES, 
)

def load_netcdf(filepath: str) -> xr.Dataset:
    """
    Conceptual function to load ERA5/ICOADS NetCDF data using xarray.
    
    Updated to handle typical ERA5 features and time-step conventions.
    """
    logging.info(f"Loading data from NetCDF file: {filepath}...")
    
    # -------------------------------------------------------------------------
    # Actual file loading using xarray
    # -------------------------------------------------------------------------
    try:
        # xarray handles loading from local path OR a remote HTTP/S URL
        ds = xr.open_dataset(filepath)
        
        # Ensure 'time', 'latitude', 'longitude' exist (or rename if ERA5 uses 'lat'/'lon')
        # ERA5 files often use 'time', 'latitude', 'longitude'. We assume 'lat'/'lon' are standard.
        if 'latitude' in ds.coords and 'longitude' in ds.coords:
             ds = ds.rename({'latitude': 'lat', 'longitude': 'lon'})

    except FileNotFoundError:
        logging.warning(f"File not found at {filepath}. Generating mock ERA5-style data instead.")
        # Fallback to mock data with all 15 required ERA5 variables
        times = np.arange(np.datetime64('2025-01-01'), np.datetime64('2025-01-03'), np.timedelta64(1, 'h')) # 48 hours of data
        lats = np.linspace(BBOX_INIT[0], BBOX_INIT[1], 30) # Reduced grid for faster mock generation
        lons = np.linspace(BBOX_INIT[2], BBOX_INIT[3], 30)
        
        n_times, n_lats, n_lons = len(times), len(lats), len(lons)
        
        data = {}
        for var in TARGET_FEATURES:
            if 'temp' in var or 'radiation' in var or 'pressure' in var:
                # Use random data centered around realistic values
                data[var] = (('time', 'lat', 'lon'), np.random.rand(n_times, n_lats, n_lons) * 10 + 280) # Kelvin/Pressure scale
            elif 'wind' in var:
                # Wind components, potentially positive or negative
                data[var] = (('time', 'lat', 'lon'), np.random.randn(n_times, n_lats, n_lons) * 8)
            elif 'wave' in var or 'precipitation' in var:
                # Wave height/precipitation (non-negative)
                data[var] = (('time', 'lat', 'lon'), np.abs(np.random.randn(n_times, n_lats, n_lons)) * 2) 

        ds = xr.Dataset(
            data,
            coords={'time': times, 'lat': lats, 'lon': lons}
        )
    # -------------------------------------------------------------------------
    
    return ds



def filter_data_to_region(ds: xr.Dataset, bbox: Tuple[float, float, float, float]) -> xr.Dataset:
    """Filters the NetCDF dataset to a specific geographic bounding box (e.g., around Hamburg)."""
    lat_min, lat_max, lon_min, lon_max = bbox
    
    # Simple spatial selection (conceptual)
    ds_filtered = ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
    
    logging.info(f"Filtered data to BBOX: {bbox}. Shape: {ds_filtered['air_temperature'].shape}")
    return ds_filtered


def prepare_data_for_koopman(ds: xr.Dataset, variables: List[str]) -> np.ndarray:
    """
    Extracts relevant variables and flattens the spatio-temporal data into
    (Time, Features) for Koopman analysis. Each spatial grid point becomes a feature.
    
    Returns array of shape (Timesteps, State_Dimension).
    """
    feature_list = []
    
    for var in variables:
        if var in ds:
            # Flattens (lat, lon) dimensions into a single feature dimension
            var_data = ds[var].values.reshape(len(ds['time']), -1)
            feature_list.append(var_data)
            
    if not feature_list:
        raise ValueError("No matching variables found in the dataset.")
        
    # Concatenate all variables along the feature dimension (axis=1)
    X = np.concatenate(feature_list, axis=1)
    logging.info(f"Prepared Koopman input matrix X with shape: {X.shape}")
    return X
