import numpy as np
import xarray as xr
from datasets.net_cfd import prepare_data_for_koopman, TARGET_VARIABLES

# spacial resolution
GRID_RES = 0.1

def calculate_vorticity(u_field: np.ndarray, v_field: np.ndarray) -> np.ndarray:
    """
    Conceptual calculation of relative vorticity (∂v/∂x - ∂u/∂y) from wind fields.
    This is a key lifting function for Koopman analysis of fluid dynamics.
    
    Args:
        u_field (np.ndarray): Zonal wind component (time, lat, lon).
        v_field (np.ndarray): Meridional wind component (time, lat, lon).
        
    Returns:
        np.ndarray: Vorticity field (time, lat, lon).
    """
    print("Calculating conceptual relative vorticity...")
    # Use central difference for spatial derivatives (conceptual implementation)
    
    # Mocking the actual calculation, which would involve numpy.gradient
    vorticity = np.zeros_like(u_field)
    
    # Partial derivative dV/dX (longitude gradient of V)
    dv_dx = np.gradient(v_field, axis=2) / (GRID_RES) 
    # Partial derivative dU/dY (latitude gradient of U)
    du_dy = np.gradient(u_field, axis=1) / (GRID_RES) 
    
    vorticity = dv_dx - du_dy
    return vorticity


def calculate_divergence(u_field: np.ndarray, v_field: np.ndarray) -> np.ndarray:
    """
    Conceptual calculation of horizontal divergence (∂u/∂x + ∂v/∂y).
    
    Args:
        u_field (np.ndarray): Zonal wind component.
        v_field (np.ndarray): Meridional wind component.
        
    Returns:
        np.ndarray: Divergence field.
    """
    print("Calculating conceptual divergence...")
    
    # Partial derivative dU/dX
    du_dx = np.gradient(u_field, axis=2) / (GRID_RES)
    # Partial derivative dV/dY
    dv_dy = np.gradient(v_field, axis=1) / (GRID_RES) 

    divergence = du_dx + dv_dy
    return divergence

def create_koopman_features(ds: xr.Dataset) -> np.ndarray:
    """
    Generates a lifted state space by adding non-linear/CFD features.
    
    Args:
        ds (xr.Dataset): Filtered NetCDF data.
        
    Returns:
        np.ndarray: Lifted feature matrix (Timesteps, New_State_Dimension).
    """
    print("Creating CFD-based Koopman lifting features...")
    
    # 1. Base features (already flattened in net_cfd.py)
    X_base = prepare_data_for_koopman(ds, TARGET_VARIABLES)
    
    # 2. CFD features
    u = ds['wind_speed_u'].values
    v = ds['wind_speed_v'].values
    
    # Calculate Vorticity and flatten
    vorticity_field = calculate_vorticity(u, v)
    X_vorticity = vorticity_field.reshape(vorticity_field.shape[0], -1)
    
    # Calculate Divergence and flatten
    divergence_field = calculate_divergence(u, v)
    X_divergence = divergence_field.reshape(divergence_field.shape[0], -1)
    
    # 3. Polynomial features (e.g., X^2, X*Y) - Mocked.
    # X_poly = X_base**2 
    
    # Concatenate all features
    X_lifted = np.concatenate([X_base, X_vorticity, X_divergence], axis=1)
    
    print(f"Lifted Koopman state space created with shape: {X_lifted.shape}")
    return X_lifted
