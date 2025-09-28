import torch.nn as nn
from utils import set_seed

set_seed(42)

class GRUModel(nn.Module):

    REQUIRED_KEYS = ["hidden_size", "num_layers", "dropout"]

    def __init__(self, input_size: int, horizon: int, dropout, hidden_size, num_layers):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )
        self.bn = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        out, _ = self.gru(x)    # (B, T, H)
        out = out[:, -1, :]     # last step
        out = self.bn(out)
        out = self.dropout(out)
        out = self.fc(out)      # (B, horizon)
        return out

    @staticmethod
    def suggest_config(trial) -> dict:
        return {
            "hidden_size":  trial.suggest_int("hidden_size", 32, 256, step=32),
            "num_layers":   trial.suggest_int("num_layers", 1, 4),
            "dropout":      trial.suggest_float("dropout", 0.1, 0.5, step=0.1),
        }
