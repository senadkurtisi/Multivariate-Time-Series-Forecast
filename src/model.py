import torch.nn as nn

class PollutionForecast(nn.Module):
  def __init__(self, input_dim, hidden_dim, num_layers):
    super().__init__()

    self.lstm = nn.LSTM(input_dim, hidden_dim, \
                        num_layers, batch_first=True)
    self.fc = nn.Linear(hidden_dim, 1)

  def forward(self, input):
    _, (hidden, _) = self.lstm(input)
    out = self.fc(hidden[-1, :, :])
    
    return out.view(input.size(0))
