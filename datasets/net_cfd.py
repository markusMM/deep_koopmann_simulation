import xarray as xr
import numpy as np
from typing import Tuple, List
import logging

# Define the coordinates for the Hamburg area (mocked)
HAMBURG_BBOX = (53.0, 54.5, 8.0, 11.0)  # (lat_min, lat_max, lon_min, lon_max)
TARGET_VARIABLES = ['air_temperature', 'sea_surface_temp', 'wind_speed_u', 'wind_speed_v', 'wave_height']
FORECAST_HORIZON = 48 # hours

def load_icoads_netcdf(filepath: str) -> xr.Dataset:
    """
    Conceptual function to load ICOADS NetCDF data using xarray.
    
    In a real scenario, this would handle file opening and basic validation.
    """
    print(f"Loading data from NetCDF file: {filepath}...")
    
    # -------------------------------------------------------------------------
    # Actual file loading using xarray
    # -------------------------------------------------------------------------
    try:
        # Assuming 'filepath' points to the .nc file, e.g., './data/icoads_sample.nc'
        ds = xr.open_dataset(filepath)
    except FileNotFoundError:
        print(f"File not found at {filepath}. Generating mock data instead.")
        # Fallback to mock data if file loading fails (useful for local development without the file)
        times = np.arange(np.datetime64('2025-01-01'), np.datetime64('2025-01-05'), np.timedelta64(1, 'h'))
        lats = np.linspace(HAMBURG_BBOX[0], HAMBURG_BBOX[1], 50)
        lons = np.linspace(HAMBURG_BBOX[2], HAMBURG_BBOX[3], 50)
        
        data = {
            'air_temperature': (('time', 'lat', 'lon'), np.random.rand(len(times), len(lats), len(lons)) * 10 + 273.15),
            'wind_speed_u': (('time', 'lat', 'lon'), np.random.randn(len(times), len(lats), len(lons)) * 5),
            'wind_speed_v': (('time', 'lat', 'lon'), np.random.randn(len(times), len(lats), len(lons)) * 5),
            'wave_height': (('time', 'lat', 'lon'), np.random.rand(len(times), len(lats), len(lons)) * 3),
        }
        
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
    
    print(f"Filtered data to BBOX: {bbox}. Shape: {ds_filtered['air_temperature'].shape}")
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
    print(f"Prepared Koopman input matrix X with shape: {X.shape}")
    return X
