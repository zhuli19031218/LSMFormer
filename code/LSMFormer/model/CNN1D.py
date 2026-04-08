import torch.nn as nn
import torch
from torchsummary import summary
from thop import clever_format, profile


class CNN1D(nn.Module):
    """
        Parameters：
        - input_size: feature size
        - hidden_size: number of hidden units
        - output_size: number of output
        - num_layers: layers of LSTM to stack
    """

    def __init__(self,
                 factors,  # 输入token的dim
                 batch_size,
                 device,
                 drop_ratio,
                 num_layers,
                 channels=20
                 ):
        super().__init__()
        self.batch_size = batch_size
        self.device = device
        self.num_layers = num_layers
        self.cnn1d = nn.Sequential()
        for num in range(num_layers):
            self.cnn1d.add_module(f"cnn1d{num}", nn.Conv1d(num+1 if num < 1 else channels, channels, 3, padding=1))
            self.cnn1d.add_module(f"ReLU{num}", nn.ReLU())
            # self.cnn1d.add_module(f"MaxPool1d{num}", nn.MaxPool1d(2, padding=factors % 2))
        # self.cnn1d = nn.Conv1d(1, 20, 3)  # 输出通道数为20
        # self.relu = nn.ReLU()
        self.max_pool1d = nn.MaxPool1d(2, padding=factors % 2)
        # (factors+num_layers*2-1)//(2**num_layers)*channels
        self.head = nn.Sequential(
            nn.Linear(in_features=int(channels*(factors+factors % 2)/2), out_features=256),  # 经过卷积和池化层后输入尺寸为（21-2）/2
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(256, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        # x = x.permute(1, 0, 2)
        x = self.cnn1d(x)
        # # print(x.shape)
        # x = self.relu(x)
        x = self.max_pool1d(x)
        # print(x.shape)
        batch, lays, s = x.shape
        x = x.view(batch, lays * s)
        # print(x.shape)
        x = self.head(x)
        x = torch.squeeze(x)
        return x


if __name__ == '__main__':
    # print(torch.cuda.is_available())
    model = CNN1D(factors=21, batch_size=400, device='cpu', drop_ratio=0.1, num_layers=2)
    input_data = torch.randn(1, 400, 21)
    output = model(input_data)
    print(output.shape)
    # summary(model, (1, 21))  # (1,21)指的是没句话用21维度的向量表示
    #
    # dummy_input = torch.randn(1, 1, 21).to('cpu')  # (1, 1, 21)指的是batch_size=1, seq_len=1,factors_dim=21
    # flops, params = profile(model.to('cpu'), (dummy_input,), verbose=False)
    # --------------------------------------------------------#
    #   flops * 2是因为profile没有将卷积作为两个operations
    #   有些论文将卷积算乘法、加法两个operations。此时乘2
    #   有些论文只考虑乘法的运算次数，忽略加法。此时不乘2
    # # --------------------------------------------------------#
    # flops = flops * 2
    # flops, params = clever_format([flops, params], "%.3f")
    # print('Total GFLOPS: %s' % (flops))
    # print('Total params: %s' % (params))
