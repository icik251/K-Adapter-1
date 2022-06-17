from torch import nn
import torch

# a = torch.Tensor([1, 2, 3,5,6])
# b = torch.Tensor([1, 2, 3])

# print(a + b)


class RNNModel(nn.Module):
    def __init__(self):
        super(RNNModel, self).__init__()
        self.input_size = 768
        self.hidden_size = 256
        self.num_layers = 2
        self.num_classes = 1

        self.lstm = nn.LSTM(
            self.input_size, self.hidden_size, self.num_layers, batch_first=True
        )

        self.linear_layers = nn.Sequential(
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(128, self.num_classes),
        )

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        # Forward propagate LSTM
        out, _ = self.lstm(
            x, (h0, c0)
        )  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.linear_layers(out[:, -1, :])
        return out


rnn_model = RNNModel()

x = torch.randn(1, 1, 768)
y = rnn_model(x)
from torchviz import make_dot
make_dot(y.mean(), params=dict(rnn_model.named_parameters()))
