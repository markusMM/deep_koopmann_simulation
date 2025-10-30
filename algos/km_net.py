import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Any, Dict, Tuple

# Import Koopman components
from algos.km_classical import ClassicalKoopmanAnalysis
from algos.km_layers import KoopmanEncoder, KoopmanEvolutionLayer
from datasets.km_dataset import ClassicalKoopmanDataset, DeepKoopmanDataset


class KoopmanModelWrapper:
    """Base class for all Koopman models, defining common interfaces."""
    def __init__(self, input_dim: int, **kwargs):
        self.input_dim = input_dim
        self.model: Any = None
        
    def fit(self, dataset: Any) -> float:
        """Fit the model to the training data."""
        raise NotImplementedError
        
    def forecast(self, x_t: Any, steps: int) -> Any:
        """Generate a multi-step forecast."""
        raise NotImplementedError


class ClassicalKoopmanWrapper(KoopmanModelWrapper):
    """Wrapper for the non-deep Koopman analysis (Classical Least-Squares)."""
    def __init__(self, input_dim: int):
        super().__init__(input_dim)
        # The Classical model uses the state_dim directly
        self.model = ClassicalKoopmanAnalysis(state_dim=input_dim)

    def fit(self, dataset: ClassicalKoopmanDataset) -> float:
        """Prepares data from dataset and fits the classical model."""
        X_list, Y_list = zip(*[dataset[i] for i in range(len(dataset))])
        
        # X is (T, State_Dim), Y is (T, State_Dim)
        X = torch.tensor(X_list)
        Y = torch.tensor(Y_list)
        
        return self.model.fit(X, Y)

    def forecast(self, x_t: torch.Tensor, steps: int = 48) -> (np.ndarray|torch.Tensor):
        """Generates NumPy forecast."""
        return self.model.predict(x_t, steps=steps)


class DeepKoopmanWrapper(pl.LightningModule, KoopmanModelWrapper):
    """
    Wrapper for the Deep Koopman analysis using PyTorch Lightning.
    The model consists of an Encoder, the linear Koopman Operator, and a Decoder (Identity/Linear).
    """
    def __init__(self, input_dim: int, koopman_dim: int = 128):
        super().__init__()
        KoopmanModelWrapper.__init__(self, input_dim) # Initialize base wrapper part
        self.save_hyperparameters() # Saves input_dim and koopman_dim
        
        self.encoder = KoopmanEncoder(input_dim, koopman_dim)
        self.koopman_op = KoopmanEvolutionLayer(koopman_dim)
        # Decoder (optional: for reconstruction in original space, here just linear)
        self.decoder = nn.Linear(koopman_dim, input_dim, bias=False) 
        
    def forward(self, x_t: torch.Tensor) -> torch.Tensor:
        """
        Performs one-step prediction: x_t -> g_t -> g_{t+1} -> x_{t+1}_pred
        """
        g_t = self.encoder(x_t)
        g_t_plus_1 = self.koopman_op(g_t)
        x_t_plus_1_pred = self.decoder(g_t_plus_1)
        return x_t_plus_1_pred
        
    def step(self, batch: Tuple[torch.Tensor, torch.Tensor], stage: str):
        """Standard training/validation step."""
        x_t, x_t_plus_1_true = batch
        
        # 1. Prediction (using the full Koopman chain)
        x_t_plus_1_pred = self.forward(x_t) 
        
        # 2. Lifting Loss (ensure g_{t+1} is close to g_t propagated by K)
        g_t = self.encoder(x_t)
        g_t_plus_1_true = self.encoder(x_t_plus_1_true) # Ideal lifted state
        g_t_plus_1_pred = self.koopman_op(g_t)          # Propagated lifted state
        
        # Total Loss: Combine prediction loss (original space) and lifting loss (Koopman space)
        pred_loss = F.mse_loss(x_t_plus_1_pred, x_t_plus_1_true)
        lift_loss = F.mse_loss(g_t_plus_1_pred, g_t_plus_1_true)
        
        # Koopman training often involves multiple loss components
        total_loss = pred_loss + 0.1 * lift_loss # Weighted combination
        
        self.log(f'{stage}_loss', total_loss)
        return total_loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self.step(batch, 'val')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def forecast(self, x_t: torch.Tensor, steps: int = 48) -> torch.Tensor:
        """Generates PyTorch multi-step forecast iteratively in the Koopman space."""
        self.eval()
        with torch.no_grad():
            current_x = x_t.unsqueeze(0) # (1, State_Dim)
            forecasts = []
            
            # Lift the initial state
            g_t = self.encoder(current_x) 
            
            for _ in range(steps):
                # Propagate in the linear Koopman space
                g_t = self.koopman_op(g_t) 
                
                # Decode the latent state back to the original space
                x_t_plus_1 = self.decoder(g_t)
                forecasts.append(x_t_plus_1.squeeze(0))
                
            return torch.stack(forecasts)
