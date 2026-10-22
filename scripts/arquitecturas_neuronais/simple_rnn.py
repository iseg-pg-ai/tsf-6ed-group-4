import torch
from torch import nn


class SimpleRNNModel(nn.Module):
    """Rede Recorrente Simples (Slide 95-96, 101).

    h_t = tanh(W_x x_t + U_h h_{t-1} + b)
    """

    def __init__(self, n_features: int, hidden_size: int = 50):
        super().__init__()
        self.rnn = nn.RNN(input_size=n_features, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])
