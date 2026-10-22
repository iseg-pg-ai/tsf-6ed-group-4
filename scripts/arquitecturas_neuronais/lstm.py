import torch
from torch import nn


class LSTMModel(nn.Module):
    """Long Short-Term Memory (Slide 97-98, 102).

    Utiliza três portas: esquecimento (forget), entrada (input) e saída (output)
    com célula de memória (c_t).
    """

    def __init__(self, n_features: int, hidden_size: int = 50):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])
