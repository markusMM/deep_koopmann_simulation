import numpy as np
import torch
from torch.linalg import pinv
from typing import Optional, Tuple
import logging


class ClassicalKoopmanAnalysis:
    """
    Implements the classical Koopman analysis via Linear Least Squares.
    
    Finds the Koopman matrix K that minimizes || Y - K * X ||_F, 
    where X is the current state data and Y is the next state data.
    """
    def __init__(
        self, 
        state_dim: int,
        return_torch_tensors: bool = True
    ) -> None:
        self.state_dim = state_dim
        self.K: Optional[np.ndarray] = None # The Koopman Operator
        self.torch_tensors = return_torch_tensors
        self.losses = []

    def fit(
        self, 
        X: (np.ndarray|torch.Tensor), 
        Y: (np.ndarray|torch.Tensor)
    ) -> "ClassicalKoopmanAnalysis":
        """
        Calculates the Koopman operator K.

        The Koopman operator K is found by `K = Y * X_pseudo_inverse`

        Least-Squares is often preferred over direct pinv for stability, 
        but pinv is conceptually clearer for :math:`K = Y X^+`
        
        Args:
            X (np.ndarray|torch.Tensor): Current state matrix (T, State_Dim).
            Y (np.ndarray|torch.Tensor): Next state matrix (T, State_Dim).
            
        Returns:
            float: The reconstruction loss (Frobenius norm).
        """
        if X.shape[1] != self.state_dim or Y.shape[1] != self.state_dim:
            raise ValueError("Input dimensions must match the initialized state_dim.")

        if isinstance(X, np.ndarray):
            X = torch.tensor(X)
        if isinstance(Y, np.ndarray):
            Y = torch.tensor(X)
        
        # Calculate the pseudo-inverse of X
        X_pinv = pinv(X.T @ X) @ X.T # Equivalent to linalg.pinv if not using full SVD
        
        # Koopman Operator K
        self.K = Y.T @ X_pinv.T
        
        # Calculate reconstruction loss
        Y_pred = X @ self.K.T  # type: ignore
        loss = float(torch.linalg.norm(Y - Y_pred, 'fro'))
        self.losses.append(loss)
        
        logging.info(f"Classical Koopman fit complete. Operator K shape: {self.K.shape}, Loss: {loss:.4f}")  # type: ignore
        return self

    def predict(
        self, 
        x_t: (np.ndarray|torch.Tensor), 
        steps: int = 1
    ) -> (np.ndarray|torch.Tensor):
        """
        Forecasts the state using the learned Koopman operator.
        
        Args:
            x_t (np.ndarray): The initial state vector (State_Dim,).
            steps (int): Number of steps to forecast.
            
        Returns:
            np.ndarray: Array of forecasted states (steps, State_Dim).
        """
        if self.K is None:
            raise RuntimeError("Model must be fitted before prediction.")
            
        current_x = x_t
        forecasts = []
        
        for _ in range(steps):
            # x_{t+1} = K * x_t (in matrix form: K @ x_t)
            next_x = self.K @ current_x
            forecasts.append(next_x)
            current_x = next_x

        forecasts = torch.tensor(forecasts)
        if not self.torch_tensors:
            forecasts = np.array(forecasts)
        
        return forecasts
