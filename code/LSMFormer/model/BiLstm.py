import torch.nn as nn
import torch


class BiLstm(nn.Module):
    def __init__(self,
                 factors,  # 输入token的dim
                 # hidden_layers,
                 batch_size,
                 device,
                 drop_ratio,
                 num_layers
                 ):
        super(BiLstm, self).__init__()
        self.factors = factors
        # self.hidden_layers = hidden_layers
        self.batch_size = batch_size
        self.device = device
        self.num_layers = num_layers
        self.backbone1 = nn.LSTM(input_size=factors, hidden_size=4 * factors, num_layers=num_layers, bidirectional=True)

        self.head = nn.Sequential(
            nn.Linear(in_features=4 * factors * 2, out_features=4 * factors),
            nn.Dropout(drop_ratio),
            nn.Linear(in_features=4 * factors, out_features=2),
            nn.Dropout(drop_ratio),
            nn.Softmax(dim=-1)
        )

    def forward(self, input):
        # print('input', input.shape)
        h0 = torch.randn(2*self.num_layers, input.shape[1], 4 * self.factors).to(self.device)
        c0 = torch.randn(2*self.num_layers, input.shape[1], 4 * self.factors).to(self.device)
        output, _ = self.backbone1(input, (h0, c0))
        output = self.head(output)
        # print('head', output.shape)
        output = torch.squeeze(output)
        # print('suqeeze', output.shape)
        return output

if __name__ == '__main__':
    model = BiLstm(factors=21, batch_size=400, device='cpu', drop_ratio=0., num_layers=3)
    input = torch.randn(1, 400, 21)
    output = model(input)
    print(output.shape)
