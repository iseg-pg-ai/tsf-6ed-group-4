import torch
from torch import nn


class CNN1D(nn.Module):
    """Rede Convolucional Unidimensional (Slide 81-84, 87, 92-94).

    Aplica filtros temporais unidimensionais com camadas de pooling e densas.
    """

    def __init__(self, n_features: int, window_size: int):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=n_features, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        # Após dois maxpools de 2 em W=28: 28 -> 14 -> 7
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * (window_size // 4), 64)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, window, features) -> (batch, features, window)
        x = x.permute(0, 2, 1)
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.dropout(self.relu3(self.fc1(x)))
        return self.fc2(x)
