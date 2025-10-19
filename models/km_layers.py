import torch
import torch.nn as nn

# Note: In a real system, GRID_RES is passed dynamically or calibrated.
from common import GRID_RES


class KoopmanEncoder(nn.Module):
    """
    The Encoder network that maps the input state x(t) to the lifted state psi(x(t)).
    This is the non-linear part of the Deep Koopman architecture.
    
    This replaces the previous KoopmanLinearLayer.
    """
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Input x shape: (Batch, Input_Dim) -> Output shape: (Batch, Latent_Dim)"""
        return self.net(x)


class CFDLiftingLayer(nn.Module):
    """
    Differentiable PyTorch layer to calculate CFD-based features (Vorticity, Divergence)
    from U and V fields, effectively performing the Koopman lifting operation.
    
    The input is expected to be a batch of gridded wind data for a single timestep.
    """
    def __init__(self, grid_res: float = GRID_RES):
        super().__init__()
        self.grid_res = grid_res

    def _calculate_finite_difference(self, field: torch.Tensor, dim: int) -> torch.Tensor:
        """
        Calculates the central finite difference (numerical gradient) along a spatial dimension.
        
        Args:
            field (torch.Tensor): Input field (B, C, Lat, Lon).
            dim (int): Spatial dimension to calculate the gradient (2 for Lat/Y, 3 for Lon/X).
            
        Returns:
            torch.Tensor: The numerical derivative.
        """
        # Roll tensors to approximate (i+1) and (i-1) points for central difference
        field_pos = torch.roll(field, shifts=-1, dims=dim)
        field_neg = torch.roll(field, shifts=1, dims=dim)
        
        # Central difference: (f(i+1) - f(i-1)) / (2 * dx)
        derivative = (field_pos - field_neg) / (2.0 * self.grid_res)
        
        # Note: Boundary conditions (BCs) are typically handled here. 
        # For simplicity, using periodic BCs via torch.roll.
        
        return derivative

    def forward(self, uv_field: torch.Tensor) -> torch.Tensor:
        """
        Input tensor shape must be (Batch, Channels=2, Latitude, Longitude).
        Channels[0] = U (Zonal), Channels[1] = V (Meridional).
        
        Args:
            uv_field (torch.Tensor): The gridded U and V wind fields.
            
        Returns:
            torch.Tensor: Lifted features (Original U/V + Vorticity + Divergence) flattened.
        """
        # Separate fields
        u_field = uv_field[:, 0:1, :, :]  # (B, 1, Lat, Lon)
        v_field = uv_field[:, 1:2, :, :]  # (B, 1, Lat, Lon)
        
        # 1. Calculate Partial Derivatives
        
        # Gradients w.r.t Longitude (X-axis, dim=3)
        du_dx = self._calculate_finite_difference(u_field, dim=3)
        dv_dx = self._calculate_finite_difference(v_field, dim=3)
        
        # Gradients w.r.t Latitude (Y-axis, dim=2)
        du_dy = self._calculate_finite_difference(u_field, dim=2)
        dv_dy = self._calculate_finite_difference(v_field, dim=2)
        
        # 2. Calculate CFD Features
        
        # Vorticity: (∂v/∂x - ∂u/∂y)
        vorticity = dv_dx - du_dy
        
        # Divergence: (∂u/∂x + ∂v/∂y)
        divergence = du_dx + dv_dy
        
        # 3. Concatenate and Flatten the Lifted State
        
        # Concatenate original fields (U, V) and new features (Vorticity, Divergence)
        lifted_features_4D = torch.cat([u_field, v_field, vorticity, divergence], dim=1) 
        
        # Flatten for input to the Koopman linear/encoder layer
        # Output shape: (Batch, 4 * Lat * Lon)
        batch_size = lifted_features_4D.size(0)
        return lifted_features_4D.view(batch_size, -1)


class KoopmanEvolutionLayer(nn.Module):
    """
    The core Koopman matrix K, which is a linear operator acting on the lifted space.
    """
    def __init__(self, latent_dim: int):
        super().__init__()
        # K is the transition matrix (linear evolution operator)
        self.K = nn.Linear(latent_dim, latent_dim, bias=False) 

    def forward(self, psi_t: torch.Tensor) -> torch.Tensor:
        """
        Evolves the state: psi(t+1) = K * psi(t)
        Input psi_t shape: (Batch, Latent_Dim) -> Output shape: (Batch, Latent_Dim)
        """
        return self.K(psi_t)
