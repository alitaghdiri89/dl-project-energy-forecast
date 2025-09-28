import torch.nn as nn
from utils import set_seed

set_seed(42)


class Chomp1d(nn.Module):
    """Remove extra padding at the end after causal convolution"""
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """A residual block for TCN"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        # Residual connection
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.drop2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNModel(nn.Module):
    REQUIRED_KEYS = ["hidden_size", "num_layers", "dropout", "kernel_size"]

    def __init__(self, input_size: int, horizon: int, dropout, hidden_size, num_layers, kernel_size=3):
        super().__init__()
        layers = []
        in_channels = input_size
        for i in range(num_layers):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation
            out_channels = hidden_size
            layers.append(
                TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                              dilation=dilation, padding=padding, dropout=dropout)
            )
            in_channels = out_channels

        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        # Input x: (B, T, input_size)
        x = x.transpose(1, 2)  # (B, input_size, T) for Conv1d
        out = self.network(x)  # (B, hidden_size, T)
        out = out[:, :, -1]    # last time step
        out = self.fc(out)     # (B, horizon)
        return out

    @staticmethod
    def suggest_config(trial) -> dict:
        return {
            "hidden_size":  trial.suggest_int("hidden_size", 32, 256, step=32),
            "num_layers":   trial.suggest_int("num_layers", 1, 4),
            "dropout":      trial.suggest_float("dropout", 0.1, 0.5, step=0.1),
            "kernel_size":  trial.suggest_categorical("kernel_size", [2, 3, 5, 7]),
        }
