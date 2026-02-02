import torch
import torch.nn as nn

class DeepAR(nn.Module):
    def __init__(self, input_size, hidden_size=40):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.mu = nn.Linear(hidden_size, 1)
        self.sigma = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        mu = self.mu(out)
        sigma = torch.exp(self.sigma(out))
        return mu, sigma