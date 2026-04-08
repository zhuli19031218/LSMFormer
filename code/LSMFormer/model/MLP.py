# -*- encoding:utf-8 -*-
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self,
                 in_features=21,  # 输入token的dim
                 batch_size=400,
                 device="cpu",
                 drop_ratio=0.3,
                 num_layers=3,
                 channels=20,
                 cell=None,
                 num_class=2
                 ):
        super(MLP, self).__init__()
        if cell is None:
            cell = [128, 256, 128]
        self.fc = nn.Sequential()
        self.fc.add_module(f"linear{0}", nn.Linear(in_features=in_features, out_features=cell[0]))
        self.fc.add_module(f"ReLU{0}", nn.ReLU())
        self.fc.add_module(f"drop{0}", nn.Dropout(drop_ratio))
        # cell = [128, 256, 128]
        for num in range(len(cell)-1):
            self.fc.add_module(f"linear{num+1}", nn.Linear(in_features=cell[num], out_features=cell[num+1]))
            self.fc.add_module(f"ReLU{num+1}", nn.ReLU())
            self.fc.add_module(f"drop{num+1}", nn.Dropout(drop_ratio))
        self.fc2 = nn.Linear(in_features=cell[-1], out_features=num_class)

    def forward(self, x):
        # print(x.shape)
        x = x.squeeze(0)
        x = self.fc(x)
        # print(x.shape)
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    model = MLP(in_features=21, cell=[128, 256, 128], num_class=2)
    print(model)
    input_data = torch.randn(1, 400, 21)
    output = model(input_data)
    print(output.shape)
