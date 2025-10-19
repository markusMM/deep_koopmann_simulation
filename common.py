import os
import torch
import configparser

# %% CFD init

CFD_CFG = configparser.ConfigParser()

config_path = os.environ.get('CFD_CONFIG')
if config_path and os.path.isfile(config_path):
    CFD_CFG.read(config_path)
else:
    # Fallback config as dictionary; supports multiple sections.
    CFD_CFG.read_dict({
        'default':{
            'grid_res_init': '0.1',
            'feature_set': 'ERA5_TARGET_FEATURES',
            'forecast_horizon': "48",
            "bb_lon_min": "5",
            "bb_lon_max": "9.5",
            "bb_lat_min": "51.25",
            "bb_lat_max": "55.75",
        }
    })


GRID_RES = float(CFD_CFG.get('default', 'grid_res_init'))

# Define the coordinates for the Hamburg area (mocked)
BBOX_INIT = (
    float(CFD_CFG.get('default', 'bb_lat_min')), 
    float(CFD_CFG.get('default', 'bb_lat_max')), 
    float(CFD_CFG.get('default', 'bb_lon_min')), 
    float(CFD_CFG.get('default', 'bb_lon_max'))
)  # (lat_min, lat_max, lon_min, lon_max)

# --- ERA5 TARGET VARIABLES ---
# These variables are selected for high-resolution maritime forecasting and
# include components necessary for CFD lifting (10m U/V components).
ERA5_TARGET_FEATURES = [
    "2m_dewpoint_temperature",                  # For humidity and fog/visibility
    "mean_sea_level_pressure",                  # For synoptic wind patterns
    "surface_pressure",                         # For local stability analysis
    "surface_solar_radiation_downwards",        # For diurnal cycle and energy balance
    "sea_surface_temperature",                  # Crucial for marine boundary layer stability
    "surface_thermal_radiation_downwards",      # Affects nocturnal cooling
    "2m_temperature",                           # Standard air temperature
    "total_precipitation",                      # For rainfall and operational impact
    "10m_u_component_of_wind",                  # U (Zonal) - Primary wind driver for CFD
    "10m_v_component_of_wind",                  # V (Meridional) - Primary wind driver for CFD
    "100m_u_component_of_wind",                 # Wind shear for stability
    "100m_v_component_of_wind",                 # Wind shear for stability
    "mean_wave_direction",                      # Direct ship movement input
    "mean_wave_period",                         # Direct ship movement input
    "significant_height_of_combined_wind_waves_and_swell" # Wave height
]
TARGET_FEATURES = locals().get(CFD_CFG.get('default', 'feature_set'), ERA5_TARGET_FEATURES)
FORECAST_HORIZON = int(CFD_CFG.get('default', 'forecast_horizon')) # hours

# %% Backend init

BCK_CFG = configparser.ConfigParser()

config_path = os.environ.get('BCK_CONFIG')
if config_path and os.path.isfile(config_path):
    BCK_CFG.read(config_path)
else:
    # Fallback config as dictionary; supports multiple sections.
    BCK_CFG.read_dict({
        'default': 
        {
            "device": "cuda" if torch.cuda.is_available() else "cpu"  # type: ignore
        }
    })

# Helper function to move data to the appropriate device (CPU or GPU)
DEVICE = torch.device(BCK_CFG.get('default', 'device'))
