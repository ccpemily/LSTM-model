from torch import nn

class LSTMModule(nn.Module):
    def __init__(self, input_size:int, output_size:int, hidden_size:int, num_layers:int) -> None:
        super().__init__()
        self.cell = nn.LSTM(input_size, hidden_size, num_layers)
        self.calc_forward = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        x, _ = self.cell(input)
        s, b, h = x.shape
        x = x.view(s * b, h)
        x = self.calc_forward(x)
        x = x.view(s, b, -1)
        return x

