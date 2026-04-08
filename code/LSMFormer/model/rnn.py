import torch.nn as nn
import torch


class RNN(nn.Module):
    def __init__(self, factors, batch_size, device, drop_ratio):
        super(RNN, self).__init__()
        self.factors = factors
        self.batch_size = batch_size
        self.drop_ratio = drop_ratio
        self.device = device

        self.rnn = nn.RNN(input_size=factors, hidden_size=4 * factors, num_layers=1, bias=False)  # , batch_first=True
        self.head = nn.Sequential(
            nn.Linear(in_features=4 * factors, out_features=2 * factors),
            nn.Dropout(drop_ratio),
            nn.Linear(in_features=2 * factors, out_features=2),
            nn.Dropout(drop_ratio),
            nn.Softmax(dim=-1)
        )

    def forward(self, input):
        h0 = torch.randn(1, input.shape[1], 4 * self.factors).to(self.device)
        output, hn = self.rnn(input, h0)
        output = self.head(output)
        output = torch.squeeze(output)
        return output


if __name__ == '__main__':
    model = RNN(factors=21, batch_size=400, device='cpu', drop_ratio=0.)
    # print(model)
    input = torch.randn(1, 400, 21)
    output = model(input)
    # print(output)
    print(output.shape)
