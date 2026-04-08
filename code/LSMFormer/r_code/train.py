import os
import argparse
import multiprocessing
from pathlib import Path
from PIL import Image
import pandas as pd
import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
# from informer_byol import BYOL1
from contrast.BYOL.BYOL_Model import BYOL
import time
# main
# dataset
class SongDataset(Dataset):
    def __init__(self, df1, df2, df3, device='cpu'):
        super().__init__()
        self.data1 = df1
        self.data2 = df2
        self.data3 = df3
        self.device = device
    def __len__(self):
        return self.data1.shape[0]

    def __getitem__(self, index):
        tens1 = torch.FloatTensor(self.data1.iloc[index, :].to_numpy()).to(self.device)
        tens2 = torch.FloatTensor(self.data2.iloc[index, :].to_numpy()).to(self.device)
        tens3 = torch.FloatTensor(self.data3.iloc[index, :].to_numpy()).to(self.device)

        return tens1, tens2, tens3
        # return tens1, tens2

class SongDataset1(Dataset):
    def __init__(self, df1, device='cpu'):
        super().__init__()
        self.data1 = df1
        self.device = device
    def __len__(self):
        return self.data1.shape[0]

    def __getitem__(self, index):
        tens1 = torch.FloatTensor(self.data1.iloc[index, :].to_numpy()).to(self.device)

        return tens1

if __name__ == '__main__':
    print(torch.cuda.is_available())
    df1 = pd.read_csv(r"D:\sss\contrast\BYOL\data\new_RC_data\rc_augment1.csv")
    df2 = pd.read_csv(r"D:\sss\contrast\BYOL\data\new_RC_data\rc_augment4.csv")
    df3 = pd.read_csv(r"D:\sss\contrast\BYOL\data\new_RC_data\RC_norm300000.csv")

    ds = SongDataset(df1, df2, df3, device='cuda')
    train_loader = DataLoader(ds, batch_size=2000, num_workers=0, shuffle=False)
    models = BYOL(hidden_layer=-1, image_size=256).to('cuda')
    opt = torch.optim.Adam(models.parameters(), lr=5e-4)
    min_loss = 1000
    for e in range(50):
        total_loss = 0
        star_time = time.time()
        for a, b, c in train_loader:
            loss1 = models(a, b, c)
            loss1.backward()
            opt.step()
            opt.zero_grad()
            models.update_moving_average()
            total_loss += loss1.detach().cpu()
        end_time = time.time()
        running_time = end_time-star_time
        if total_loss < min_loss:
            min_loss = total_loss
            torch.save(models.state_dict(), r'D:\sss\contrast\BYOL\SaveModel\RC_GAF_IR1.pth')
        print(total_loss, 'time cost: %.1f' % running_time)
    torch.save(models.state_dict(), r'D:\sss\contrast\BYOL\SaveModel\RC_GAF_IR1_last.pth')
