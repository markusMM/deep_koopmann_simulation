import numpy as np
import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Dict
from torch.linalg import pinv, svd
import logging
from common import DEVICE

def create_hankel_matrix(data: torch.Tensor, k: int) -> torch.Tensor:
    """
    Creates a Hankel matrix from a time-series data tensor.
    
    The Hankel matrix is crucial for delay embedding, where each row
    of H represents an 'embedded' state [x(t), x(t-1), ..., x(t-k+1)].
    
    Args:
        data (torch.Tensor): Time-series data (T, State_Dim).
        k (int): Number of time-delay embeddings (number of rows in the embedded state).
        
    Returns:
        torch.Tensor: Hankel matrix H (T - k + 1, k * State_Dim).
    """
    T, _ = data.shape
    if T < k:
        raise ValueError(f"Time series length ({T}) must be greater than embedding delay k ({k}).")
    
    # Each block is x(t), x(t-1), ... x(t-k+1)
    H_blocks = [data[i:T - k + i + 1, :] for i in range(k)]
    
    # Concatenate horizontally to form the Hankel matrix
    # Shape: (Time instances, Embedded State Dimension)
    H = torch.cat(H_blocks, dim=1)
    
    # The current time step H[i, :] corresponds to the state at time t=i + k - 1
    return H.to(DEVICE)


class HAVOCAnalysis:
    """
    Implements the Hankel Alternative View of Koopman (HAVOC) algorithm, 
    a variant of DMD/Koopman that handles control inputs (U).
    
    The model is identified as a linear system in the lifted Hankel space:
    Psi(t+1) = A * Psi(t) + B * U(t)
    
    All calculations are hardware accelerated using PyTorch.
    """
    def __init__(
        self, 
        state_dim: int, 
        control_dim: int, 
        k_delay: int, 
        r_rank: int,
        full_reconstruction: bool
    ) -> None:
        """
        Args:
            state_dim (int): Dimension of the raw state x (e.g., U/V fields flattened).
            control_dim (int): Dimension of the external control u (e.g., ship speed, route).
            k_delay (int): Number of time delays for the Hankel embedding.
            r_rank (int): Rank truncation for SVD (dimension of the Koopman observable space).
            full_reconstruction (bool): Wether to use H(t+1) together with Psi as forecasting operator.
        """
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.k_delay = k_delay
        self.r_rank = r_rank
        self.full_reconstruction = full_reconstruction
        self.losses = []
        
        # Embedded dimension of the Koopman state (Psi)
        self.embedded_dim = state_dim * k_delay
        
        self.A: Optional[torch.Tensor] = None # Autonomous dynamics matrix (Koopman Operator)
        self.B: Optional[torch.Tensor] = None # Control matrix
        self.V_r: Optional[torch.Tensor] = None # SVD basis vectors for prediction
        self.V_r_pinv: Optional[torch.Tensor] = None # Pseudo-inverse for state projection

    def fit(
        self, X: (torch.Tensor|np.ndarray), 
        U: (torch.Tensor|np.ndarray), 
    ) -> "HAVOCAnalysis":
        """
        Calculates the Koopman (A) and Control (B) matrices.
        
        Args:
            X (torch.Tensor): State data (T, State_Dim).
            U (torch.Tensor): Control input data (T, Control_Dim).
            
        Returns:
            float: The reconstruction loss (Frobenius norm).
        """
        if isinstance(X, np.ndarray): X = torch.tensor(X, torch.float32)
        if isinstance(U, np.ndarray): U = torch.tensor(U, torch.float32)
        X = X.to(DEVICE)
        U = U.to(DEVICE)
        
        # 1. Create Hankel Matrix
        # H shape: (T - k_delay + 1, k_delay * State_Dim)
        H = create_hankel_matrix(X, self.k_delay)
        
        # Separate into H_past (current state) and H_future (next state)
        # H_past: H[t] (state at time t)
        # H_future: H[t+1] (state at time t+1)
        H_past = H[:-1, :]
        H_future = H[1:, :]
        
        # Align control input U with the past states
        # U_past is control input applied at time t to get to state t+1
        U_past = U[self.k_delay:-1, :] 

        # 2. Perform SVD on the past states H_past
        # U_svd: Left singular vectors, S: Singular values, Vh: Right singular vectors (transposed)
        _, _, Vh = svd(H_past, full_matrices=False)
        
        # Truncate SVD to the rank r_rank
        self.V_r = Vh[:self.r_rank, :].T # V_r maps the embedded state to the Koopman observable space (Psi)
        self.V_r_pinv = pinv(self.V_r)
        
        # 3. Project the Hankel states onto the Koopman observable space (Psi)
        # Psi = H_past @ V_r
        Psi_past = H_past @ self.V_r  # type: ignore
        Psi_future = H_future @ self.V_r  # type: ignore
        
        # 4. Form the regression matrix Lambda
        # Lambda_past: [ Psi(t) ; U(t) ] 
        # Note: Psi is already transposed relative to the original paper's notation
        Lambda_past = torch.cat([Psi_past, U_past], dim=1) # Shape: (Time instances - k_delay - 1, r_rank + Control_Dim)
        
        # 5. Solve for the System Matrix [A; B] using Least Squares (Pseudo-inverse)
        
        # Target: Psi_future
        # Matrix to solve: [A; B] = Psi_future @ Lambda_past_pinv
        Lambda_past_pinv = pinv(Lambda_past)
        
        # K_hat shape: (r_rank, r_rank + Control_Dim)
        K_hat = Lambda_past_pinv @ Psi_future

        # 6. Extract A and B
        self.A = K_hat[:self.r_rank, :self.r_rank] # Koopman Dynamics
        self.B = K_hat[:self.r_rank, self.r_rank:] # Control Matrix

        # 7. Calculate reconstruction loss
        Psi_pred = Lambda_past @ K_hat 
        loss = torch.linalg.norm(Psi_future - Psi_pred, 'fro').item()
        self.losses.append(loss)
        
        logging.info(f"HAVOC fit complete. Koopman (A) shape: {self.A.shape}, Control (B) shape: {self.B.shape}")  # type: ignore
        logging.info(f"Reconstruction Loss (in Koopman space): {loss:.4f}")
        return self

    def predict(self, x_t_history: torch.Tensor, U_sequence: torch.Tensor) -> torch.Tensor:
        """
        Forecasts the state using the learned HAVOC model.

        The calculation is sequenced in 5 steps:

            1.  Create the initial embedded state (Psi_t)
                & Flatten the history to create the first Hankel vector H_t (1, k_delay * State_Dim)

            2.  Linear Evolution: Psi(t+1) = A * Psi(t) + B * U(t)

            3.  Use the first row of V_r (pseudo-inverse) to approximately recover the next state x(t+1)
                The HAVOC paper uses V_r[:, 0] to project the lifted state back to the original state dimension.
                In practice, V_r is often defined differently, but conceptually, we need a map back.
                We will use the V_r_pinv to get the next Hankel vector H(t+1) first

            4.  Extract the primary state x(t+1) from the Hankel vector
                The newest state is the first (or last, depending on construction) block of the Hankel vector.
                Assuming H_t_plus_1 is [x(t+1), x(t), x(t-1), ...]

            5.  Update the current state for the next step (Shift the Hankel vector)
                This involves creating a new Hankel vector for time t+1
                Option A (Simple): Reconstruct the full H(t+1) and use that for Psi(t+1) next loop
                This is complex and can quickly become inaccurate.
                Option B (Standard Koopman): Use the predicted Psi(t+1) directly for the next evolution
        
        Args:
            x_t_history (torch.Tensor): The k_delay history of states (k_delay, State_Dim).
            U_sequence (torch.Tensor): Control inputs for the prediction horizon (steps, Control_Dim).
            
        Returns:
            torch.Tensor: Array of forecasted states (steps, State_Dim).
        """
        if self.A is None or self.B is None:
            raise RuntimeError("Model must be fitted before prediction.")
        
        U_sequence = U_sequence.to(DEVICE)
        
        # step 1
        H_t = x_t_history.flip(dims=[0]).flatten().unsqueeze(0).to(DEVICE) 
        
        # Project H_t into the Koopman observable space Psi_t
        Psi_t = H_t @ self.V_r  # type: ignore
        
        forecasts: List[torch.Tensor] = []
        
        for i in range(U_sequence.shape[0]):
            u_t = U_sequence[i, :].unsqueeze(0) # (1, Control_Dim)
            
            # step 2
            Psi_t_plus_1 = (Psi_t @ self.A) + (u_t @ self.B)
            
            # step 3
            H_t_plus_1 = Psi_t_plus_1 @ self.V_r_pinv  # type: ignore
            
            # step 4
            x_t_plus_1 = H_t_plus_1[:, :self.state_dim]
            
            forecasts.append(x_t_plus_1.squeeze(0))
            
            # ## step 5
            if self.full_reconstruction:
                # Option A: Full Reconstruction and Shifting (More robust)
                
                # H_t_old_part contains the shifted history [x(t), x(t-1), ..., x(t-k+2)]
                H_t_old_part = H_t_plus_1[:, self.state_dim:]
                
                # New Hankel vector H_t_next = [x(t+1) | x(t) | x(t-1) | ...]
                # This explicitly uses the predicted observation x(t+1) and the rest of the reconstructed history.
                H_t_next = torch.cat([x_t_plus_1, H_t_old_part], dim=1)
                
                # Project the new, shifted Hankel vector back to the Koopman space for the next step
                Psi_t = H_t_next @ self.V_r  # type: ignore
                
            else:
                # op B
                Psi_t = Psi_t_plus_1 

        return torch.stack(forecasts)
