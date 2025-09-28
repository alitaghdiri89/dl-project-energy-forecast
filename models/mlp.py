import torch
import torch.nn as nn
from utils import set_seed

set_seed(42)

class MLPModel(nn.Module):
    
    REQUIRED_KEYS = ["hidden_size", "num_layers", "dropout", "activation"]

    def __init__(self, input_size: int, horizon: int, dropout: float, 
                 hidden_size: int, num_layers: int, activation: str):
        super().__init__()
        
        self.input_size = input_size 
        
        # Map activation string to torch function
        self.activations = {
            "ReLU": nn.ReLU(),
            "GELU": nn.GELU(),
            "LeakyReLU": nn.LeakyReLU(),
            "Leaky_relu_0.1": nn.LeakyReLU(0.1),
            "Tanh": nn.Tanh(),
        }
        self.activation_func = self.activations[activation]
        
        # Build the sequential MLP layers
        layers = []
        current_in_size = self.input_size
        
        for i in range(num_layers):
            current_out_size = hidden_size
            
            # Linear layer
            layers.append(nn.Linear(current_in_size, current_out_size))
            
            # Batch Normalization (applied after linear layer before activation)
            # BN layer is typically nn.BatchNorm1d(output_size)
            layers.append(nn.BatchNorm1d(current_out_size)) 
            
            # Activation function
            layers.append(self.activation_func)
            
            # Dropout
            layers.append(nn.Dropout(dropout))
            
            # Update input size for the next layer
            current_in_size = current_out_size

        # Final output layer
        layers.append(nn.Linear(current_in_size, horizon))
        
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x is assumed to be (B, T, I) similar to the LSTM
        
        # 1. Take the features from the last time step (T)
        # Resulting shape: (B, I) or (B, input_size)
        out = x[:, -1, :]
        
        # 2. Pass through the MLP
        out = self.mlp(out)  # (B, horizon)
        
        return out

    @staticmethod
    def suggest_config(trial) -> dict:
        """
        Suggests hyperparameter configurations for the MLP model using Optuna.
        """
        return {
            "hidden_size": trial.suggest_int("hidden_size", 32, 256, step=32),
            "num_layers": trial.suggest_int("num_layers", 1, 6),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5, step=0.1),
            "activation": trial.suggest_categorical("activation", ["ReLU", "GELU", "LeakyReLU", "Leaky_relu_0.1", "Tanh"]),
        }
