import torch.nn as nn
import torch


class GRU(nn.Module):
    def __init__(self,
                 factors,  # 输入token的dim
                 batch_size,
                 device,
                 drop_ratio,
                 num_layers
                 ):
        super().__init__()
        self.batch_size = batch_size
        self.device = device
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size=factors, hidden_size=4*factors, num_layers=self.num_layers)
        # utilize the GRU model in torch.nn
        self.head = nn.Sequential(
            nn.Linear(in_features=4 * factors, out_features=64),
            nn.Dropout(drop_ratio),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=2),
            # nn.Dropout(drop_ratio),
            # nn.Softmax(dim=-1)
        )

    def forward(self, _x):
        self.gru.flatten_parameters()
        x, _ = self.gru(_x)  # _x is input, size (seq_len, batch, input_size)
        s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        x = x.view(s * b, h)
        x = self.head(x)
        x = torch.squeeze(x)
        return x


if __name__ == '__main__':
    model = GRU(factors=21, batch_size=400, device='cpu', drop_ratio=0.,num_layers=1)
    input = torch.randn(1, 400, 21)
    output = model(input)
    print(output.shape)
