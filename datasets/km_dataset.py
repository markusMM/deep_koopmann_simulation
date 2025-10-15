import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple

class KoopmanDatasetBase(Dataset):
    """
    Base Dataset for Koopman analysis, handles time-lagged data.
    
    The dataset yields pairs (x_t, x_{t+1}) for training the Koopman operator.
    """
    def __init__(self, data: np.ndarray, time_lag: int = 1):
        """
        Args:
            data (np.ndarray): The full time series data (Timesteps, State_Dimension).
            time_lag (int): The prediction step (k in x_{t+k}).
        """
        self.data = data
        self.time_lag = time_lag
        self.T = self.data.shape[0] - time_lag
        
    def __len__(self):
        # We lose 'time_lag' number of steps at the end
        return self.T

class ClassicalKoopmanDataset(KoopmanDatasetBase):
    """Dataset for Classical (non-deep) Koopman Model using raw NumPy arrays."""
    def __init__(self, data: np.ndarray, time_lag: int = 1):
        super().__init__(data, time_lag)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        # x_t is data at time step idx
        x_t = self.data[idx] 
        # x_{t+1} is data at time step idx + time_lag
        x_t_plus_k = self.data[idx + self.time_lag] 
        return x_t, x_t_plus_k

class DeepKoopmanDataset(KoopmanDatasetBase):
    """Dataset for Deep Koopman Model, converts data to PyTorch tensors."""
    def __init__(self, data: np.ndarray, time_lag: int = 1):
        super().__init__(data, time_lag)
        # Convert the full NumPy array to a single PyTorch tensor once
        self.data_tensor = torch.from_numpy(data).float()

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # x_t is data at time step idx
        x_t = self.data_tensor[idx] 
        # x_{t+1} is data at time step idx + time_lag
        x_t_plus_k = self.data_tensor[idx + self.time_lag] 
        return x_t, x_t_plus_k
