import torch
from torch import nn


class GRUModel(nn.Module):
    """Gated Recurrent Unit (Slide 99, 103).

    Variante simplificada da LSTM com apenas duas portas:
    atualização (z_t) e reposição (r_t).
    """

    def __init__(self, n_features: int, hidden_size: int = 50):
        super().__init__()
        self.gru = nn.GRU(input_size=n_features, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])
